"""Smoke test: ML data assembly + model forward/backward on a synthetic graph (CPU).

Runs without the DB so it works in CI / on a box without the DuckDB store assembled. Asserts
the SoilGNN forward/backward/KL, the spatial-block split disjointness, and the geology-prior
shape. Mirrors the standalone-script + hard-assert style of the other tests/smoke_*.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ml.data import EDGE_TYPES
from ml.model import SoilGNN, build_rel_index
from ml.splits import kfold_block_split, train_val_test_masks


def test_model_forward_backward():
    torch.manual_seed(0)
    N, E = 500, 3000
    card = [10, 8, 6, 12, 9, 4, 4, 4, 4]
    cat = torch.stack([torch.randint(0, c, (N,)) for c in card], dim=1)
    x_num = torch.randn(N, 23); x_mask = torch.ones(N, 23)
    ei = torch.randint(0, N, (2, E)); et = torch.randint(0, len(EDGE_TYPES), (E,))
    rel = build_rel_index(ei, et, len(EDGE_TYPES), "cpu")
    m = SoilGNN(cat_cardinalities=card, num_dim=46, n_codes=82, n_families=15, n_drains=11,
                edge_types=EDGE_TYPES, hidden=32, layers=3, n_uscs=12)
    m.train()
    pl = torch.randn(N, 82)
    fam, code, dr, uscs = m(x_num, x_mask, cat, rel, sample=True, prior_logits=pl)
    assert code.shape == (N, 82) and fam.shape == (N, 15) and uscs.shape == (N, 12)
    loss = (torch.nn.functional.cross_entropy(code, torch.randint(0, 82, (N,)))
            + torch.nn.functional.cross_entropy(uscs, torch.randint(0, 12, (N,))) + m.kl() / N)
    loss.backward()
    assert not any(p.grad is not None and bool(p.grad.isnan().any()) for p in m.parameters())
    # ablation: deterministic + edge subset
    fam, code, dr, uscs = m(x_num, x_mask, cat, rel, sample=False, active_rel={0, 1})
    assert code.shape == (N, 82)
    print("model forward/backward/KL/ablation/aux-head OK")


def test_spatial_splits_disjoint():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 200000, size=(2000, 2))
    labels = rng.integers(0, 20, size=2000)
    fold = kfold_block_split(xy, labels, block_size_ft=20000, folds=5, seed=7)
    assert set(np.unique(fold)).issubset(set(range(5)))
    tr, va, te = train_val_test_masks(fold, test_fold=0, val_fold=1)
    assert not (tr & te).any() and not (tr & va).any() and not (va & te).any()
    # blocks are whole: every node in a 20k-ft cell shares one fold
    bx = (xy[:, 0] // 20000).astype(int); by = (xy[:, 1] // 20000).astype(int)
    cell = bx * 100000 + by
    for c in np.unique(cell):
        assert len(np.unique(fold[cell == c])) == 1
    print("spatial-block splits disjoint + whole-block OK")


def test_3d_model():
    from ml.model import SoilGNN3D, build_rel_index
    torch.manual_seed(0); N, E = 300, 2000
    card = [10, 8, 6, 12, 9, 4, 4, 4, 4]
    cat = torch.stack([torch.randint(0, c, (N,)) for c in card], dim=1)
    xn = torch.randn(N, 23); xm = torch.ones(N, 23)
    ei = torch.randint(0, N, (2, E)); et = torch.randint(0, len(EDGE_TYPES), (E,))
    rel = build_rel_index(ei, et, len(EDGE_TYPES), "cpu")
    m = SoilGNN3D(cat_cardinalities=card, num_dim=46, edge_types=EDGE_TYPES, n_uscs=9,
                  hidden=32, layers=3)
    m.train()
    h = m.encode(xn, xm, cat, rel, sample=True)
    M = 60; sn = torch.randint(0, N, (M,)); sd = torch.randn(M, 1)
    spt, uscs = m.decode(h[sn], sd, sample=True)
    assert spt.shape == (M, 2) and uscs.shape == (M, 9) and m.gw(h[sn]).shape == (M, 2)
    y = torch.randn(M)
    nll = 0.5 * (spt[:, 1] + (y - spt[:, 0]) ** 2 / spt[:, 1].exp()).mean()
    loss = nll + torch.nn.functional.cross_entropy(uscs, torch.randint(0, 9, (M,))) + m.kl() / M
    loss.backward()
    assert not any(p.grad is not None and bool(p.grad.isnan().any()) for p in m.parameters())
    print("3D depth-conditioned model (encode/decode/gw/NLL/KL) OK")


if __name__ == "__main__":
    test_model_forward_backward()
    test_spatial_splits_disjoint()
    test_3d_model()
    print("ALL ML SMOKE PASSED")
