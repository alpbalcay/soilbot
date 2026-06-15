"""A0.5 — assemble the labeled-graph data layer.

Steps (idempotent):
  1. Attach geology/SSURGO covariates to soil_labels -> soil_label_covariates
     (reuses pipeline.covariates.assign_point_covariates).
  2. Build the borings ∪ soil_labels union graph -> ml_edges + union parquet.
  3. Connectivity gate: every labeled node must reach >=1 boring within `hops`
     (= GNN depth) or it can never receive boring-derived messages.

Run: python -m ml.assemble
"""
from __future__ import annotations

import numpy as np

from pipeline import covariates, db
from pipeline.config import Config
from pipeline.logging_setup import new_run_id, setup

from .graph_build import build_union_edges


def connectivity_report(con, hops: int = 3) -> dict:
    """Fraction of label nodes with a boring within `hops` hops on the union graph."""
    rows = con.execute("SELECT src, dst, src_type, dst_type FROM ml_edges").fetchall()
    index: dict[str, int] = {}
    for s, d, st, dt in rows:
        if s not in index:
            index[s] = len(index)
        if d not in index:
            index[d] = len(index)
    n = len(index)
    if n == 0:
        return {"labels": 0, "connected": 0, "frac": 0.0}
    from scipy.sparse import csr_matrix
    si = np.fromiter((index[s] for s, *_ in rows), dtype=np.int64, count=len(rows))
    di = np.fromiter((index[d] for _, d, *_ in rows), dtype=np.int64, count=len(rows))
    data = np.ones(len(rows) * 2, dtype=np.float32)
    A = csr_matrix((data, (np.concatenate([si, di]), np.concatenate([di, si]))),
                   shape=(n, n))
    is_boring = np.zeros(n, dtype=np.float32)
    is_label = np.zeros(n, dtype=bool)
    for node, i in index.items():
        if node.startswith("b:"):
            is_boring[i] = 1.0
        else:
            is_label[i] = True
    reach = is_boring.copy()
    for _ in range(hops):
        reach = np.maximum(reach, (A @ reach > 0).astype(np.float32))
    label_idx = np.where(is_label)[0]
    connected = int((reach[label_idx] > 0).sum())
    total = int(is_label.sum())
    return {"labels": total, "connected": connected,
            "frac": connected / total if total else 0.0, "hops": hops}


def run(config: Config, log) -> dict:
    con = db.connect(config)
    db.bootstrap(con)

    # 1. covariates for soil_labels (geology attaches; SSURGO stays null until footprints)
    cov = covariates.assign_point_covariates(
        con, config, log, point_table="soil_labels", point_id="objectid",
        dest_table="soil_label_covariates")

    # 2. union graph
    edges = build_union_edges(con, config, log)

    # 3. connectivity gate
    conn = connectivity_report(con, hops=int(config.get("ml", "model", "layers", default=3)))
    log.info("ml_connectivity", **conn)
    gate_ok = conn["frac"] >= 0.95
    log.info("ml_assemble_done", gate_ok=gate_ok,
             soil_label_cov=cov, union=edges["by_type"], connectivity=conn)
    con.close()
    return {"covariates": cov, "edges": edges, "connectivity": conn, "gate_ok": gate_ok}


if __name__ == "__main__":
    cfg = Config.load(None)
    rid = new_run_id()
    logger = setup(cfg.path("log_dir"), "ml.log", rid, "assemble", console=True)
    res = run(cfg, logger)
    print("\n=== A0.5 assemble ===")
    print("soil_label_covariates:", res["covariates"])
    print("union edges:", {k: res["edges"][k] for k in ("nodes", "borings", "labels", "edges")})
    print("by_type:", res["edges"]["by_type"])
    print("connectivity:", res["connectivity"])
    print("GATE (>=95% labels reach a boring within receptive field):",
          "PASS" if res["gate_ok"] else "FAIL")
