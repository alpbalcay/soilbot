"""Empirical-Bayes geology prior over soil-type logits.

P(code | surficial_unit) estimated on TRAIN labels only (per fold, no leakage), Dirichlet-
smoothed toward the global class distribution. Returned as per-node prior logits added to the
class head: where a region is undrilled the KL pulls the residual head to ~0 so predictions
relax to the geology-implied distribution; where labels are dense the data overrides it.
This is the firm "geology as informative prior" requirement, and the exploration showed
P(code|unit) concentrates to 0.46-0.73 for many units, so the prior carries real signal.
"""
from __future__ import annotations

import numpy as np


def geology_prior_logits(ds, train_mask: np.ndarray, *, alpha: float = 2.0,
                         weight: float = 1.0) -> np.ndarray:
    """Return [N, n_codes] prior logits from P(code | surficial_unit) on train labels."""
    n_codes = len(ds.code_classes)
    surf_col = _cat_column(ds, "surficial_unit")          # [N] int vocab idx
    y = ds.y_code.numpy()

    # global smoothed distribution
    tr = train_mask & (y >= 0)
    global_counts = np.bincount(y[tr], minlength=n_codes).astype(np.float64)
    global_p = (global_counts + alpha) / (global_counts.sum() + alpha * n_codes)
    global_logp = np.log(global_p)

    # per surficial-unit distribution, shrunk toward global
    units = np.unique(surf_col[tr])
    unit_logp = {}
    for u in units:
        m = tr & (surf_col == u)
        c = np.bincount(y[m], minlength=n_codes).astype(np.float64)
        p = (c + alpha * global_p * n_codes) / (c.sum() + alpha * n_codes)  # EB shrinkage to global
        unit_logp[int(u)] = np.log(p)

    prior = np.tile(global_logp, (len(y), 1))
    for i in range(len(y)):
        lp = unit_logp.get(int(surf_col[i]))
        if lp is not None:
            prior[i] = lp
    # center per-row (logits are shift-invariant) and scale
    prior = prior - prior.mean(1, keepdims=True)
    return (weight * prior).astype(np.float32)


def _cat_column(ds, name) -> np.ndarray:
    from .data import CAT_COLS
    j = CAT_COLS.index(name)
    return ds.cat_idx[:, j].numpy()
