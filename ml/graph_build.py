"""Build the message-passing graph over the borings ∪ soil_labels union.

Why a union graph: soil_labels (the only labeled nodes) are spatially DISJOINT from
borings — median nearest-boring distance ≈ 2.6 km, none within 50 ft. The pipeline's
boring-only `edges` therefore cannot connect a single label to a boring. Recomputing
kNN/Delaunay/same_geology over the combined 69,407-node set (in native feet, metric-
correct) lets each label pick up nearby borings and same-geology neighbours so the
covariate-rich, dense boring cloud can inform the sparse labels.

Node ids are namespaced to keep the two id-spaces disjoint:
  borings      -> 'b:' + boring_id          (e.g. 'b:B0000007')
  soil_labels  -> 'l:' + str(objectid)      (e.g. 'l:12345')

Reuses `pipeline.graph.compute_edges` verbatim — only the node loader differs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pipeline import db
from pipeline.config import Config
from pipeline.graph import _canon, compute_edges


def _load_union_nodes(con):
    """Return (ids, xy, types) for borings ∪ soil_labels in native feet (EPSG:102711)."""
    rows = con.execute(
        """
        SELECT 'b:' || boring_id AS id, ST_X(geom_native), ST_Y(geom_native), 'boring' AS t
        FROM borings WHERE geom_native IS NOT NULL
        UNION ALL
        SELECT 'l:' || CAST(objectid AS VARCHAR), ST_X(geom_native), ST_Y(geom_native), 'label'
        FROM soil_labels WHERE geom_native IS NOT NULL
        ORDER BY id
        """
    ).fetchall()
    ids = [r[0] for r in rows]
    xy = np.asarray([[r[1], r[2]] for r in rows], dtype=float) if rows else np.empty((0, 2))
    types = {r[0]: r[3] for r in rows}
    return ids, xy, types


def _load_union_units(con) -> dict:
    """Map namespaced node id -> surficial unit, from both covariate tables."""
    units: dict[str, str] = {}
    for bid, unit in con.execute(
            "SELECT boring_id, surficial_unit FROM boring_covariates").fetchall():
        if unit is not None:
            units["b:" + str(bid)] = unit
    # soil_label_covariates keys its id column `id` = CAST(objectid AS VARCHAR)
    for lid, unit in con.execute(
            "SELECT id, surficial_unit FROM soil_label_covariates").fetchall():
        if unit is not None:
            units["l:" + str(lid)] = unit
    return units


def _add_label_boring_edges(edges: dict, ids, xy, types, k: int, use_rust: bool = False,
                            log=None) -> int:
    """Add `label_boring` edges from each label to its k nearest borings (in feet)."""
    if use_rust:
        try:
            import soilbot_rs
            new = soilbot_rs.label_boring_edges(
                list(ids), np.ascontiguousarray(xy, dtype=float), dict(types), int(k))
            added = 0
            for key, w in new.items():
                if key not in edges:
                    edges[key] = w
                    added += 1
            return added
        except ImportError:
            if log:
                log.warning("rust_unavailable_fallback_scipy")
    b_idx = [i for i, nid in enumerate(ids) if types[nid] == "boring"]
    l_idx = [i for i, nid in enumerate(ids) if types[nid] == "label"]
    if not b_idx or not l_idx:
        return 0
    b_xy = xy[b_idx]
    l_xy = xy[l_idx]
    tree = cKDTree(b_xy)
    kk = min(k, len(b_idx))
    dist, nn = tree.query(l_xy, k=kk)
    if kk == 1:
        dist = dist[:, None]; nn = nn[:, None]
    added = 0
    for li, (drow, nrow) in enumerate(zip(dist, nn)):
        lid = ids[l_idx[li]]
        for d, bj in zip(drow, nrow):
            bid = ids[b_idx[int(bj)]]
            a, b = _canon(lid, bid)
            key = (a, b, "label_boring")
            if key not in edges:
                edges[key] = round(1.0 / (1.0 + float(d)), 6)
                added += 1
    return added


def build_union_edges(con, config: Config, log) -> dict:
    """Compute union edges, persist to `ml_edges` + the union-edges parquet, return stats."""
    g = config["graph"]
    ids, xy, types = _load_union_nodes(con)
    n = len(ids)
    if n == 0:
        log.warning("ml_no_nodes")
        return {"nodes": 0, "edges": 0}
    units = _load_union_units(con)
    edges = compute_edges(ids, xy, units, g, log)

    # Cross-type bridge: connect every LABEL to its k nearest BORINGS. Labels are spatially
    # disjoint from borings (~2.6 km), so plain union-kNN leaves a quarter of labels unable to
    # reach a boring within the GNN's receptive field. These edges guarantee each label a
    # boring neighbourhood at hop 1 (a kriging/IDW-style neighbourhood), so the covariate-rich
    # boring cloud always informs the labels. Distinct edge_type -> independently ablatable.
    n_bridge = _add_label_boring_edges(edges, ids, xy, types,
                                       k=int(config.get("ml", "cross_k", default=6)),
                                       use_rust=bool(g.get("use_rust", False)), log=log)
    log.info("label_boring_bridge", edges=n_bridge,
             k=int(config.get("ml", "cross_k", default=6)))

    con.execute("DELETE FROM ml_edges")
    if edges:
        # Vectorized bulk insert via a registered dict-of-numpy (executemany is ~200x slower
        # for ~1.3M rows: 4.7s vs 18+ min). Build columnar arrays from the edge dict.
        keys = list(edges.keys())
        src = np.array([k[0] for k in keys], dtype=object)
        dst = np.array([k[1] for k in keys], dtype=object)
        etype = np.array([k[2] for k in keys], dtype=object)
        weight = np.array([edges[k] for k in keys], dtype=np.float64)
        src_t = np.array([types[k[0]] for k in keys], dtype=object)
        dst_t = np.array([types[k[1]] for k in keys], dtype=object)
        con.register("ml_edges_stage",
                     {"src": src, "dst": dst, "edge_type": etype, "weight": weight,
                      "src_type": src_t, "dst_type": dst_t})
        con.execute(
            "INSERT INTO ml_edges (src, dst, edge_type, weight, src_type, dst_type) "
            "SELECT src, dst, edge_type, weight, src_type, dst_type FROM ml_edges_stage")
        con.unregister("ml_edges_stage")
    pq = config.get("ml", "union_edges_parquet", default="data/ml/union_edges.parquet")
    pq_abs = config.abspath(pq)
    pq_abs.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT src, dst, edge_type, weight, src_type, dst_type FROM ml_edges "
        f"ORDER BY edge_type, src, dst) TO '{pq_abs}' (FORMAT PARQUET)")

    by_type = dict(con.execute(
        "SELECT edge_type, COUNT(*) FROM ml_edges GROUP BY 1 ORDER BY 1").fetchall())
    n_label = sum(1 for t in types.values() if t == "label")
    n_boring = n - n_label
    log.info("ml_edges_built", nodes=n, borings=n_boring, labels=n_label,
             total=len(edges), **{f"n_{t}": c for t, c in by_type.items()},
             parquet=str(pq_abs))
    return {"nodes": n, "borings": n_boring, "labels": n_label,
            "edges": len(edges), "by_type": by_type, "parquet": str(pq_abs)}
