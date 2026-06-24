"""Depth-resolved (3D) supervision built on top of the Phase-A union graph.

The spatial encoder is unchanged — borings are already nodes in the union graph (ml/data.py).
This adds one training tuple per OCR'd spoon-format sample: (boring node, depth) -> (SPT-N,
USCS class), plus a per-boring groundwater target. SPT-N is modeled in log1p space (right-skewed
counts). Honesty sanity-gates drop OCR'd values that are physically implausible (negative/huge
depths, SPT-N > 100 = likely a digit error or drilling refusal) rather than trusting them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pipeline import db
from pipeline.config import Config

from .data import Dataset

# physical sanity bounds (feet / blows) — OCR rows outside these are dropped
DEPTH_MIN_FT, DEPTH_MAX_FT = 0.0, 200.0
SPT_MIN, SPT_MAX = 0, 100

# B2 — non-leaky per-sample physics features from `strata_derived`. These depend only on
# depth + USCS + groundwater (NOT on spt_n, the prediction target), so they are safe as model
# inputs. The LEAKY columns below ARE functions of spt_n and must NEVER be fed as inputs.
SAFE_PHYS_COLS = ["sigma_eff_v0_tsf", "sigma_v0_tsf", "gamma_pcf", "cn"]
LEAKY_COLS = ["n60", "n1_60", "phi_peck_deg", "phi_hatanaka_deg", "dr_pct", "su_tsf",
              "e_modulus_tsf", "m_constrained_tsf", "allow_bearing_tsf", "n1_60cs", "crr"]
assert not (set(SAFE_PHYS_COLS) & set(LEAKY_COLS)), "leaky column wired into SAFE_PHYS_COLS"


@dataclass
class Dataset3D:
    # graph (copied from the Phase-A Dataset so train3d loads one file)
    node_ids: list
    node_type: np.ndarray
    xy: np.ndarray
    x_num: torch.Tensor
    x_mask: torch.Tensor
    cat_idx: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    cat_cardinalities: list
    # depth-resolved supervision (one row per spoon sample that passes the sanity gates)
    sample_node: torch.Tensor       # [M] long, index into the graph nodes (a boring)
    sample_depth_ft: torch.Tensor   # [M] raw depth (ft) — for reporting
    sample_depth_std: torch.Tensor  # [M] standardized depth (model input)
    y_spt_log: torch.Tensor         # [M] log1p(SPT-N), -1.0 where missing/gated
    y_uscs: torch.Tensor            # [M] long USCS-at-depth class, -1 where missing
    # per-boring groundwater
    gw_node: torch.Tensor           # [G] long node index
    gw_ft: torch.Tensor             # [G] groundwater depth (ft)
    uscs_classes: list
    depth_mean: float
    depth_std: float
    # B2 — non-leaky physics inputs from strata_derived (defaulted so old caches still load)
    sample_phys: torch.Tensor = None        # [M, P] standardized σ'v0/σv0/γ/CN (0 where missing)
    sample_phys_mask: torch.Tensor = None   # [M, P] 1.0 where strata_derived present
    phys_cols: list = None                  # the P SAFE_PHYS_COLS, in column order
    phys_mean: list = None                  # per-column standardization mean (present-only)
    phys_std: list = None                   # per-column standardization std

    def save(self, path):
        torch.save(self.__dict__, path)

    @staticmethod
    def load(path) -> "Dataset3D":
        return Dataset3D(**torch.load(path, weights_only=False))


def build_3d_dataset(config: Config, log=None) -> Dataset3D:
    out = config.abspath(config.get("ml", "out_dir", default="data/ml"))
    ds = Dataset.load(out / "dataset.pt")
    node_index = {nid: i for i, nid in enumerate(ds.node_ids)}
    con = db.connect(config, read_only=True)

    # B2: pull the non-leaky physics features per interval from strata_derived if present.
    has_derived = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='strata_derived'"
    ).fetchone()[0] > 0
    # spoon samples: need a valid depth and at least one of SPT-N / USCS
    if has_derived:
        phys_sel = ", ".join(f"d.{c}" for c in SAFE_PHYS_COLS)
        rows = con.execute(f"""
            SELECT s.boring_id, s.top_depth, s.spt_n, s.uscs_class, {phys_sel}
            FROM strata s
            LEFT JOIN strata_derived d
              ON d.boring_id = s.boring_id AND d.interval_index = s.interval_index
            WHERE s.top_depth IS NOT NULL AND (s.spt_n IS NOT NULL OR s.uscs_class IS NOT NULL)
            ORDER BY s.boring_id, s.interval_index
        """).fetchall()
    else:
        rows = con.execute("""
            SELECT boring_id, top_depth, spt_n, uscs_class
            FROM strata
            WHERE top_depth IS NOT NULL AND (spt_n IS NOT NULL OR uscs_class IS NOT NULL)
            ORDER BY boring_id, interval_index
        """).fetchall()
    gw_rows = con.execute("""
        SELECT boring_id, MIN(gw_depth) FROM strata
        WHERE gw_depth IS NOT NULL GROUP BY boring_id
    """).fetchall()
    con.close()

    uscs_vocab: dict[str, int] = {}
    s_node, s_depth, s_spt, s_uscs, s_phys = [], [], [], [], []
    dropped = 0
    for row in rows:
        bid, top, spt, uscs = row[0], row[1], row[2], row[3]
        phys = list(row[4:]) if has_derived else []
        ni = node_index.get("b:" + str(bid))
        if ni is None or top is None or not (DEPTH_MIN_FT <= top <= DEPTH_MAX_FT):
            dropped += 1
            continue
        spt_ok = spt is not None and SPT_MIN <= spt <= SPT_MAX
        if uscs is not None:
            uscs_vocab.setdefault(uscs, len(uscs_vocab))
        s_node.append(ni)
        s_depth.append(float(top))
        s_spt.append(float(spt) if spt_ok else np.nan)
        s_uscs.append(uscs_vocab[uscs] if uscs is not None else -1)
        s_phys.append(phys)

    uscs_classes = [None] * len(uscs_vocab)
    for k, v in uscs_vocab.items():
        uscs_classes[v] = k

    # B2: standardize physics features per column (present-only), with a presence mask. Missing
    # strata_derived -> all-zero value + mask 0, so the model degrades to depth-only (== B1).
    sample_phys = sample_phys_mask = phys_cols = phys_mean = phys_std = None
    n_phys_samples = 0
    if has_derived and s_phys:
        P = len(SAFE_PHYS_COLS)
        raw = np.array([[np.nan if v is None else float(v) for v in r] for r in s_phys],
                       dtype=np.float64).reshape(-1, P)
        present = ~np.isnan(raw)
        std_arr = np.zeros_like(raw, dtype=np.float32)
        phys_mean, phys_std = [], []
        for j in range(P):
            col, pj = raw[:, j], present[:, j]
            mu = float(col[pj].mean()) if pj.any() else 0.0
            sd = float(col[pj].std()) if pj.any() else 1.0
            sd = sd or 1.0
            phys_mean.append(mu); phys_std.append(sd)
            std_arr[:, j] = np.where(pj, (np.nan_to_num(col) - mu) / sd, 0.0)
        sample_phys = torch.tensor(std_arr, dtype=torch.float32)
        sample_phys_mask = torch.tensor(present.astype(np.float32), dtype=torch.float32)
        phys_cols = list(SAFE_PHYS_COLS)
        n_phys_samples = int(present[:, 0].sum())  # samples with σ'v0 present

    depth = np.asarray(s_depth, dtype=np.float64)
    dmean, dstd = (depth.mean(), depth.std() or 1.0) if len(depth) else (0.0, 1.0)
    spt = np.asarray(s_spt, dtype=np.float64)
    y_spt_log = np.where(np.isnan(spt), -1.0, np.log1p(spt))

    gw_node, gw_ft = [], []
    for bid, g in gw_rows:
        ni = node_index.get("b:" + str(bid))
        if ni is not None and g is not None and 0 <= g <= DEPTH_MAX_FT:
            gw_node.append(ni); gw_ft.append(float(g))

    d3 = Dataset3D(
        node_ids=ds.node_ids, node_type=ds.node_type, xy=ds.xy,
        x_num=ds.x_num, x_mask=ds.x_mask, cat_idx=ds.cat_idx,
        edge_index=ds.edge_index, edge_type=ds.edge_type,
        cat_cardinalities=ds.cat_cardinalities,
        sample_node=torch.tensor(s_node, dtype=torch.long),
        sample_depth_ft=torch.tensor(depth, dtype=torch.float32),
        sample_depth_std=torch.tensor((depth - dmean) / dstd, dtype=torch.float32),
        y_spt_log=torch.tensor(y_spt_log, dtype=torch.float32),
        y_uscs=torch.tensor(s_uscs, dtype=torch.long),
        gw_node=torch.tensor(gw_node, dtype=torch.long),
        gw_ft=torch.tensor(gw_ft, dtype=torch.float32),
        uscs_classes=uscs_classes, depth_mean=float(dmean), depth_std=float(dstd),
        sample_phys=sample_phys, sample_phys_mask=sample_phys_mask,
        phys_cols=phys_cols, phys_mean=phys_mean, phys_std=phys_std,
    )
    if log:
        log.info("dataset3d_built", samples=len(s_node), dropped_sanity=dropped,
                 with_spt=int((~np.isnan(spt)).sum()), with_uscs=int((np.asarray(s_uscs) >= 0).sum()),
                 borings=len(set(s_node)), gw_borings=len(gw_node), uscs_classes=len(uscs_classes),
                 phys_cols=phys_cols, phys_samples=n_phys_samples,
                 spt_mean=round(float(np.nanmean(spt)), 1) if len(spt) else None)
    return d3


def build_and_cache_3d(config: Config, log=None) -> Dataset3D:
    d3 = build_3d_dataset(config, log)
    out = config.abspath(config.get("ml", "out_dir", default="data/ml"))
    d3.save(out / "dataset3d.pt")
    return d3
