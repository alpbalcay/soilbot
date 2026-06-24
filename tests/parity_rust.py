"""Parity harness: Rust (soilbot_rs) vs the Python/scipy implementations.

Validates that the Rust ports produce the SAME edges and node features as the existing
scipy/numpy code, on the real DuckDB store. Run:  .venv/bin/python tests/parity_rust.py

Exits non-zero on any parity failure. Skips (exit 0 with a notice) if soilbot_rs is not
built or the DB has no nodes yet.
"""
from __future__ import annotations

import sys

import numpy as np

try:
    import soilbot_rs
except ImportError:
    print("SKIP: soilbot_rs not built (run `maturin develop --release` in soilbot-rs/)")
    sys.exit(0)

from pipeline import db, graph
from pipeline.config import Config
from ml import graph_build


class _NullLog:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _all_equidistant_ties(only_a, only_b, distfn, eps=1e-6):
    """True if every edge in `only_a` has, in `only_b`, an edge sharing a node at an EQUAL
    distance — i.e. the two edge sets differ only by which of several equidistant neighbours
    was selected. scipy.cKDTree and kiddo break exact distance ties differently, so a handful
    of such swaps is expected on gridded/co-located NJDOT borings and is modeling-irrelevant."""
    for (a, b, _t) in only_a:
        da = distfn(a, b)
        if not any(
            (x in (a, b) or y in (a, b)) and abs(distfn(x, y) - da) < eps
            for (x, y, _u) in only_b
        ):
            return False
    return True


def _compare_edges(name, py_edges, rs_edges, distfn, weight_atol=1e-6,
                   tie_frac=0.001, dela_tol_frac=0.01):
    py_keys, py_w = set(py_edges), py_edges
    rs_keys, rs_w = set(rs_edges), rs_edges
    ok = True
    # knn/same_geology/label_boring: must be identical UP TO equidistant tie-breaking.
    for t in ("knn", "same_geology", "label_boring"):
        pk = {k for k in py_keys if k[2] == t}
        rk = {k for k in rs_keys if k[2] == t}
        if not pk and not rk:
            continue
        only_py, only_rs = pk - rk, rk - pk
        if not only_py and not only_rs:
            print(f"  ok   [{name}/{t}] {len(pk)} edges match exactly")
            continue
        sym = len(only_py) + len(only_rs)
        frac = sym / max(len(pk), 1)
        ties = (_all_equidistant_ties(only_py, only_rs, distfn)
                and _all_equidistant_ties(only_rs, only_py, distfn))
        good = ties and frac <= tie_frac
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} [{name}/{t}] py={len(pk)} rs={len(rk)} "
              f"symdiff={sym} ({frac:.4%}) equidistant_ties={ties}")
    # delaunay: QJ-joggle vs spade differ on degenerate (co-linear corridor) points.
    pk = {k for k in py_keys if k[2] == "delaunay"}
    rk = {k for k in rs_keys if k[2] == "delaunay"}
    if pk or rk:
        sym = len(pk ^ rk)
        frac = sym / max(len(pk), 1)
        good = frac <= dela_tol_frac
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} [{name}/delaunay] py={len(pk)} rs={len(rk)} "
              f"symdiff={sym} ({frac:.4%} <= {dela_tol_frac:.2%}; degenerate-point joggle)")
    # weights on shared keys must be bit-comparable.
    shared = py_keys & rs_keys
    if shared:
        maxdiff = max(abs(py_w[k] - rs_w[k]) for k in shared)
        good = maxdiff <= weight_atol
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} [{name}/weights] max |Δweight| over "
              f"{len(shared)} shared = {maxdiff:.2e}")
    return ok


def _distfn(ids, xy):
    pos = {nid: i for i, nid in enumerate(ids)}
    return lambda a, b: float(np.hypot(*(xy[pos[a]] - xy[pos[b]])))


def check_boring_graph(con, config):
    print("== boring graph (pipeline.graph) ==")
    g = config["graph"]
    ids, xy = graph._load_nodes(con)
    if not ids:
        print("  SKIP: no boring nodes")
        return True
    units = dict(con.execute(
        "SELECT boring_id, surficial_unit FROM boring_covariates").fetchall())
    py = graph.compute_edges(ids, xy, units, g, _NullLog())
    rs = soilbot_rs.compute_edges(
        ids, xy, {k: v for k, v in units.items() if v is not None},
        int(g["knn_k"]), bool(g.get("delaunay", True)),
        bool(g.get("same_geology_edges", True)),
        float(g.get("same_geology_radius_m", 2000)), int(g.get("same_geology_k", 26)))
    return _compare_edges(f"boring n={len(ids)}", py, rs, _distfn(ids, xy))


def check_union_graph(con, config):
    print("== union graph (ml.graph_build) ==")
    g = config["graph"]
    ids, xy, types = graph_build._load_union_nodes(con)
    if not ids:
        print("  SKIP: no union nodes")
        return True
    units = graph_build._load_union_units(con)
    # base edges
    py = graph.compute_edges(ids, xy, units, g, _NullLog())
    rs = soilbot_rs.compute_edges(
        ids, xy, units, int(g["knn_k"]), bool(g.get("delaunay", True)),
        bool(g.get("same_geology_edges", True)),
        float(g.get("same_geology_radius_m", 2000)), int(g.get("same_geology_k", 26)))
    dfn = _distfn(ids, xy)
    ok = _compare_edges(f"union n={len(ids)}", py, rs, dfn)
    # label_boring bridge
    k = int(config.get("ml", "cross_k", default=6))
    py_lb = dict(py)
    graph_build._add_label_boring_edges(py_lb, ids, xy, types, k)
    py_lb_only = {kk: vv for kk, vv in py_lb.items() if kk[2] == "label_boring"}
    rs_lb = soilbot_rs.label_boring_edges(ids, xy, types, k)
    ok = _compare_edges(f"union-bridge n={len(ids)}", py_lb_only, rs_lb, dfn) and ok
    return ok


def check_features(con):
    print("== node features (ml.data numeric block) ==")
    from ml.data import _NODE_SQL
    rows = con.execute(_NODE_SQL).fetchall()
    if not rows:
        print("  SKIP: no nodes")
        return True
    n = len(rows)
    xy = np.asarray([[r[2], r[3]] for r in rows], dtype=np.float64)
    elev_raw = [r[14] for r in rows]

    # --- reference: exact copy of the ml/data.py numeric block ---
    x_mean = xy.mean(0); x_scale = xy.std(0).max() or 1.0
    coords = (xy - x_mean) / x_scale
    elev = np.asarray([np.nan if v is None else float(v) for v in elev_raw])
    elev_present = ~np.isnan(elev)
    elev_mean = elev[elev_present].mean() if elev_present.any() else 0.0
    elev_std = elev[elev_present].std() if elev_present.any() else 1.0
    elev_norm = np.where(elev_present, (np.nan_to_num(elev) - elev_mean) / (elev_std or 1.0), 0.0)
    fourier = []
    for f in (0.5, 1.0, 2.0, 4.0, 8.0):
        fourier.append(np.sin(f * coords[:, 0])); fourier.append(np.cos(f * coords[:, 0]))
        fourier.append(np.sin(f * coords[:, 1])); fourier.append(np.cos(f * coords[:, 1]))
    feats = [coords[:, 0], coords[:, 1], elev_norm] + fourier
    masks = [np.ones(n), np.ones(n), elev_present.astype(float)] + [np.ones(n)] * len(fourier)
    x_num_ref = np.stack(feats, axis=1).astype(np.float32)
    x_mask_ref = np.stack(masks, axis=1).astype(np.float32)

    # --- rust ---
    x_num_rs, x_mask_rs, stats = soilbot_rs.assemble_features(
        xy, elev, [0.5, 1.0, 2.0, 4.0, 8.0])

    ok = True
    dmax = float(np.max(np.abs(x_num_ref - x_num_rs)))
    s = "ok  " if dmax <= 1e-5 else "FAIL"
    if dmax > 1e-5:
        ok = False
    print(f"  {s} [x_num] shape {x_num_rs.shape} max |Δ| = {dmax:.2e}")
    mask_eq = bool(np.array_equal(x_mask_ref, x_mask_rs))
    print(f"  {'ok  ' if mask_eq else 'FAIL'} [x_mask] exact equal = {mask_eq}")
    ok = ok and mask_eq
    sd = (abs(stats["x_scale"] - float(x_scale)) +
          abs(stats["elev_mean"] - float(elev_mean)) +
          abs(stats["elev_std"] - float(elev_std)))
    print(f"  {'ok  ' if sd < 1e-3 else 'FAIL'} [stats] Σ|Δ| = {sd:.2e}")
    return ok and sd < 1e-3


def main():
    import os
    config = Config.load()
    db_override = os.environ.get("SOILBOT_DB")
    if db_override:
        # Connect read-only to a snapshot copy (the live DB may be write-locked by a running job).
        import duckdb
        ext_dir = config.extension_dir
        con = duckdb.connect(db_override, read_only=True)
        con.execute(f"SET extension_directory='{ext_dir}'")
        con.execute("LOAD spatial")
    else:
        con = db.connect(config, read_only=True)
    results = []
    results.append(check_boring_graph(con, config))
    results.append(check_union_graph(con, config))
    results.append(check_features(con))
    con.close()
    if all(results):
        print("\nPARITY OK")
        sys.exit(0)
    print("\nPARITY FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
