"""Smoke test for the Phase-B --geotech ablation — literature USCS-keyed geotech inputs.

Validates (1) the non-leakage stance: the geo columns are a distinct block, disjoint from both the
spt_n-derived LEAKY columns and the B2 SAFE physics columns; (2) Dataset3D carries the per-depth
geo features with sane coverage and no NaN/inf; (3) the SoilGNN3D decoder accepts a COMBINED
physics+geotech block (the way train3d routes them) and yields the right shapes. Does NOT run a full
training. Run:

    PYTHONPATH=. .venv/bin/python tests/smoke_geotech.py

Skips (exit 0) if strata are empty or the Phase-A dataset.pt is missing. Unlike smoke_b2 this needs
neither soilbot_rs nor strata_derived — the geo block is keyed only on uscs_class.

CAVEAT this test encodes: geo features are non-leaky for spt_n (the headline target) but ARE a
function of uscs_class, a secondary Phase-B target — so the USCS@depth metric is leaky under
--geotech. That is a documented property, not something this test can guard at the data layer.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ml.data3d import LEAKY_COLS, SAFE_PHYS_COLS
from ml.geotech_features import GEO_SAMPLE_COLS, interval_geotech


def main():
    ok = True

    # (1) the geo block is distinct: disjoint from spt_n-leaky cols AND the B2 physics cols.
    geo = set(GEO_SAMPLE_COLS)
    leaky_overlap = geo & set(LEAKY_COLS)
    safe_overlap = geo & set(SAFE_PHYS_COLS)
    print(f"  {'ok  ' if not leaky_overlap else 'FAIL'} geo∩LEAKY = {leaky_overlap or '∅'}")
    print(f"  {'ok  ' if not safe_overlap else 'FAIL'} geo∩SAFE_PHYS = {safe_overlap or '∅'}")
    # the per-interval lookup maps known USCS classes and rejects junk
    map_ok = interval_geotech("CL") is not None and interval_geotech("???") is None
    print(f"  {'ok  ' if map_ok else 'FAIL'} interval_geotech maps USCS, rejects junk")
    ok = ok and not leaky_overlap and not safe_overlap and map_ok

    from pipeline.config import Config
    from ml.data3d import build_3d_dataset

    config = Config.load()
    dbp = os.environ.get("SOILBOT_DB")
    if dbp:
        config.d["duckdb"]["path"] = os.path.abspath(dbp)

    out = config.abspath(config.get("ml", "out_dir", default="data/ml"))
    if not (out / "dataset.pt").exists():
        print("SKIP: data/ml/dataset.pt missing (run ml.assemble first)")
        sys.exit(0 if ok else 1)

    class _Log:
        def info(self, ev, **k):
            print("  info", ev, {x: k[x] for x in list(k)[:8]})

        def warning(self, ev, **k):
            print("  warn", ev, k)

    d3 = build_3d_dataset(config, _Log())
    if d3.sample_geo is None:
        print("SKIP: no mappable USCS strata (sample_geo is None)")
        sys.exit(0 if ok else 1)

    # (2) the geo features load with sane coverage and are finite
    G = d3.sample_geo.shape[1]
    cols_ok = d3.geo_cols == list(GEO_SAMPLE_COLS) and G == len(GEO_SAMPLE_COLS)
    finite = bool(np.isfinite(d3.sample_geo.numpy()).all())
    cov = float(d3.sample_geo_mask[:, 0].float().mean())  # uniform across cols (all-7 or none)
    print(f"  {'ok  ' if cols_ok else 'FAIL'} geo_cols = {d3.geo_cols}")
    print(f"  {'ok  ' if finite else 'FAIL'} sample_geo finite, shape {tuple(d3.sample_geo.shape)}")
    print(f"  {'ok  ' if cov > 0.5 else 'FAIL'} USCS-mappable coverage = {cov:.2%} (>50%)")
    ok = ok and cols_ok and finite and cov > 0.5

    # (3) decoder accepts the COMBINED physics+geotech block (train3d's routing), right shapes
    import torch

    from ml.data import EDGE_TYPES
    from ml.model import SoilGNN3D, build_rel_index

    blocks, masks = [], []
    if d3.sample_phys is not None:
        blocks.append(d3.sample_phys); masks.append(d3.sample_phys_mask)
    blocks.append(d3.sample_geo); masks.append(d3.sample_geo_mask)
    s_extra = torch.cat(blocks, dim=1)
    s_extra_mask = torch.cat(masks, dim=1)
    P = s_extra.shape[1]
    model = SoilGNN3D(cat_cardinalities=d3.cat_cardinalities,
                      num_dim=d3.x_num.shape[1] + d3.x_mask.shape[1],
                      edge_types=EDGE_TYPES, n_uscs=max(1, len(d3.uscs_classes)),
                      hidden=32, layers=2, phys_dim=P)
    rel = build_rel_index(d3.edge_index, d3.edge_type, len(EDGE_TYPES), "cpu")
    with torch.no_grad():
        h = model.encode(d3.x_num, d3.x_mask, d3.cat_idx, rel, sample=False)
        sel = d3.sample_node[:8]
        dep = d3.sample_depth_std[:8].unsqueeze(1)
        spt, uscs = model.decode(h[sel], dep, phys=s_extra[:8], phys_mask=s_extra_mask[:8],
                                 sample=False)
    shape_ok = spt.shape == (8, 2)
    print(f"  {'ok  ' if shape_ok else 'FAIL'} decoder forward with phys+geo (dim={P}) -> spt "
          f"{tuple(spt.shape)}")
    ok = ok and shape_ok

    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
