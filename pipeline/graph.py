"""Phase 5 — graph inputs for the downstream GNN.

Builds three undirected edge types over the boring nodes and exports edges.parquet:
  * knn          — k nearest neighbours (cKDTree)
  * delaunay     — Delaunay triangulation edges (planar neighbour graph)
  * same_geology — nearby borings (within a radius) sharing a surficial geology unit

Neighbour math runs in the NATIVE projected CRS (EPSG:102711, US feet) so Euclidean
distance is metric-correct; doing kNN in raw lon/lat would distort at NJ's latitude.
No model is trained — this only produces graph inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from . import db
from .config import Config

FT_PER_M = 3.280839895


def _load_nodes(con):
    rows = con.execute(
        "SELECT boring_id, ST_X(geom_native), ST_Y(geom_native) "
        "FROM borings WHERE geom_native IS NOT NULL ORDER BY boring_id"
    ).fetchall()
    ids = [r[0] for r in rows]
    xy = np.asarray([[r[1], r[2]] for r in rows], dtype=float) if rows else np.empty((0, 2))
    return ids, xy


def _canon(a: str, b: str):
    return (a, b) if a < b else (b, a)


def compute_edges(ids: list, xy: np.ndarray, units: dict, g: dict, log) -> dict:
    """Pure edge construction over arbitrary nodes (reused by the boring graph and the
    ML borings∪soil_labels union graph). `ids` are node ids, `xy` their projected
    coordinates in feet (metric-correct kNN), `units` maps id -> surficial unit (for
    same_geology edges), `g` is the graph config block. Returns {(a,b,type): weight}.
    """
    n = len(ids)
    k = int(g["knn_k"])
    edges: dict[tuple, float] = {}
    if g.get("use_scipy", True) and n > k + 1:
        tree = cKDTree(xy)

        # --- kNN ---
        dist, idx = tree.query(xy, k=k + 1)
        for i in range(n):
            ai = ids[i]
            for jj, d in zip(idx[i, 1:], dist[i, 1:]):  # skip self at column 0
                a, b = _canon(ai, ids[int(jj)])
                edges[(a, b, "knn")] = round(1.0 / (1.0 + float(d)), 6)

        # --- Delaunay (QJ joggles to survive duplicate/co-linear points) ---
        if g.get("delaunay", True):
            try:
                tri = Delaunay(xy, qhull_options="QJ")
                for t in tri.simplices:
                    for u in range(3):
                        for v in range(u + 1, 3):
                            a, b = _canon(ids[int(t[u])], ids[int(t[v])])
                            key = (a, b, "delaunay")
                            if key not in edges:
                                d = float(np.hypot(*(xy[int(t[u])] - xy[int(t[v])])))
                                edges[key] = round(1.0 / (1.0 + d), 6)
            except Exception as exc:  # noqa: BLE001
                log.warning("delaunay_failed", error=str(exc)[:140])

        # --- same geology unit, bounded by radius + neighbour cap ---
        if g.get("same_geology_edges", True) and units:
            radius_ft = float(g.get("same_geology_radius_m", 2000)) * FT_PER_M
            kg = min(n, int(g.get("same_geology_k", 26)))
            dist2, idx2 = tree.query(xy, k=kg)
            for i in range(n):
                ui = units.get(ids[i])
                if not ui:
                    continue
                for jj, d in zip(idx2[i, 1:], dist2[i, 1:]):
                    if d > radius_ft:
                        break  # neighbours are distance-sorted
                    if units.get(ids[int(jj)]) == ui:
                        a, b = _canon(ids[i], ids[int(jj)])
                        edges.setdefault((a, b, "same_geology"), 1.0)
    else:
        log.warning("scipy_path_skipped", nodes=n)

    # Drop self-loops: co-located nodes (identical coords) can make cKDTree return the
    # query point itself at a non-zero column, yielding an a==a pair after canonicalization.
    return {k: v for k, v in edges.items() if k[0] != k[1]}


def build_edges(con, config: Config, log) -> dict:
    g = config["graph"]
    ids, xy = _load_nodes(con)
    n = len(ids)
    if n == 0:
        log.warning("no_nodes")
        return {"nodes": 0, "edges": 0, "by_type": {}}
    units = dict(con.execute(
        "SELECT boring_id, surficial_unit FROM boring_covariates").fetchall())
    edges = compute_edges(ids, xy, units, g, log)
    con.execute("DELETE FROM edges")
    if edges:
        con.executemany(
            "INSERT INTO edges (src, dst, edge_type, weight) VALUES (?, ?, ?, ?)",
            [[a, b, t, w] for (a, b, t), w in edges.items()],
        )
    pq = config.path("edges_parquet")
    con.execute(
        f"COPY (SELECT src, dst, edge_type, weight FROM edges ORDER BY edge_type, src, dst) "
        f"TO '{pq}' (FORMAT PARQUET)"
    )
    by_type = dict(con.execute(
        "SELECT edge_type, COUNT(*) FROM edges GROUP BY 1 ORDER BY 1").fetchall())
    log.info("edges_built", nodes=n, total=len(edges),
             **{f"n_{t}": c for t, c in by_type.items()})
    return {"nodes": n, "edges": len(edges), "by_type": by_type, "parquet": str(pq)}


def run(config: Config, log) -> dict:
    con = db.connect(config)
    db.bootstrap(con)
    res = build_edges(con, config, log)
    con.close()
    return res
