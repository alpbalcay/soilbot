"""Smoke test for B2 — physics-grounded inputs to the depth-resolved SPT model.

Validates (1) the leakage guard, (2) that Dataset3D loads the non-leaky physics features with
sane coverage, and (3) that the SoilGNN3D decoder accepts the physics block and produces the
right shapes. Does NOT run a full training (that's the end-to-end A/B). Run:

    SOILBOT_DB=<writable copy> .venv/bin/python tests/smoke_b2.py

Skips (exit 0) if soilbot_rs is unbuilt or there are no parsed strata yet.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ml.data3d import LEAKY_COLS, SAFE_PHYS_COLS


class _Log:
    def info(self, ev, **k):
        print("  info", ev, {x: k[x] for x in list(k)[:6]})

    def warning(self, ev, **k):
        print("  warn", ev, k)


def main():
    ok = True

    # (1) leakage guard — SAFE inputs must never overlap the spt_n-derived (leaky) columns.
    overlap = set(SAFE_PHYS_COLS) & set(LEAKY_COLS)
    print(f"  {'ok  ' if not overlap else 'FAIL'} leakage-guard SAFE∩LEAKY = {overlap or '∅'}")
    ok = ok and not overlap

    try:
        import soilbot_rs  # noqa: F401
    except ImportError:
        print("SKIP: soilbot_rs not built (leakage guard passed)")
        sys.exit(0 if ok else 1)

    import duckdb

    from pipeline import derive
    from pipeline.config import Config
    from ml.data3d import build_3d_dataset

    config = Config.load()
    dbp = os.environ.get("SOILBOT_DB")
    if dbp:
        config.d["duckdb"]["path"] = os.path.abspath(dbp)

    # need strata + a Phase-A dataset.pt; populate strata_derived if absent/empty
    con = duckdb.connect(str(config.duckdb_path), read_only=True)
    con.execute(f"SET extension_directory='{config.extension_dir}'"); con.execute("LOAD spatial")
    n_strata = con.execute("SELECT COUNT(*) FROM strata").fetchone()[0]
    con.close()
    if n_strata == 0:
        print("SKIP: strata empty (OCR not run)")
        sys.exit(0 if ok else 1)
    out = config.abspath(config.get("ml", "out_dir", default="data/ml"))
    if not (out / "dataset.pt").exists():
        print("SKIP: data/ml/dataset.pt missing (run ml.assemble first)")
        sys.exit(0 if ok else 1)

    print("  populating strata_derived ...")
    config.d.setdefault("soil_engine", {})["enabled"] = True
    derive.run(config, _Log())

    # (2) dataset carries physics features with sane coverage
    d3 = build_3d_dataset(config, _Log())
    if d3.sample_phys is None:
        print("  FAIL: Dataset3D.sample_phys is None despite strata_derived present"); ok = False
    else:
        P = d3.sample_phys.shape[1]
        subset = set(d3.phys_cols or []) <= set(SAFE_PHYS_COLS)
        finite = bool(np.isfinite(d3.sample_phys.numpy()).all())
        # σ'v0 needs only depth+γ (no SPT), so once strata_derived is built nearly every sample
        # should carry it; a low number signals a broken join/alignment, not just sparse SPT.
        cov = float(d3.sample_phys_mask[:, 0].float().mean())
        print(f"  {'ok  ' if subset else 'FAIL'} phys_cols⊆SAFE = {d3.phys_cols}")
        print(f"  {'ok  ' if finite else 'FAIL'} sample_phys finite (no NaN/inf), shape "
              f"{tuple(d3.sample_phys.shape)}")
        print(f"  {'ok  ' if cov > 0.5 else 'FAIL'} σ'v0 coverage over samples = {cov:.2%} (>50%)")
        ok = ok and subset and finite and cov > 0.5

        # (3) model decoder accepts the physics block and yields the right shapes
        import torch

        from ml.data import EDGE_TYPES
        from ml.model import SoilGNN3D, build_rel_index
        model = SoilGNN3D(cat_cardinalities=d3.cat_cardinalities,
                          num_dim=d3.x_num.shape[1] + d3.x_mask.shape[1],
                          edge_types=EDGE_TYPES, n_uscs=max(1, len(d3.uscs_classes)),
                          hidden=32, layers=2, phys_dim=P)
        rel = build_rel_index(d3.edge_index, d3.edge_type, len(EDGE_TYPES), "cpu")
        with torch.no_grad():
            h = model.encode(d3.x_num, d3.x_mask, d3.cat_idx, rel, sample=False)
            sel = d3.sample_node[:8]
            dep = d3.sample_depth_std[:8].unsqueeze(1)
            spt, uscs = model.decode(h[sel], dep, phys=d3.sample_phys[:8],
                                     phys_mask=d3.sample_phys_mask[:8], sample=False)
        shape_ok = spt.shape == (8, 2)
        print(f"  {'ok  ' if shape_ok else 'FAIL'} decoder forward with phys -> spt {tuple(spt.shape)}")
        ok = ok and shape_ok

    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
