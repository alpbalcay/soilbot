"""Spatial block cross-validation splits.

Borings (and labels) cluster on transport corridors (REPORT §5), so a random split
leaks near-duplicate neighbours across train/test and massively over-states accuracy.
We instead bucket nodes into square blocks in native feet (metric-equal-area, unlike
degree rounding) and assign WHOLE blocks to folds, so train and test are spatially
separated. Blocks are larger than the GNN's receptive field (`block_size_ft`), and an
optional inductive evaluation additionally cuts test->train edges to measure true
extrapolation into undrilled regions.
"""
from __future__ import annotations

import numpy as np


def assign_blocks(xy: np.ndarray, block_size_ft: float) -> np.ndarray:
    """Integer block id per node from its (x,y) in feet."""
    bx = np.floor(xy[:, 0] / block_size_ft).astype(np.int64)
    by = np.floor(xy[:, 1] / block_size_ft).astype(np.int64)
    # pack to a single id
    return bx * 1_000_003 + by


def kfold_block_split(xy: np.ndarray, labels: np.ndarray, *, block_size_ft: float,
                      folds: int, seed: int) -> np.ndarray:
    """Return a fold index in [0, folds) per node, assigning whole blocks to folds.

    Greedy class-stratified assignment: process blocks from largest to smallest and place
    each into the fold that currently has the fewest of that block's dominant class — this
    balances the long-tailed 82-class distribution across spatially-disjoint folds.
    """
    rng = np.random.default_rng(seed)
    block = assign_blocks(xy, block_size_ft)
    uniq, inv = np.unique(block, return_inverse=True)
    n_blocks = len(uniq)

    # dominant class + size per block
    order = rng.permutation(n_blocks)  # tie-break randomness, seeded
    sizes = np.zeros(n_blocks, dtype=np.int64)
    dom = np.full(n_blocks, -1, dtype=np.int64)
    for b in range(n_blocks):
        members = np.where(inv == b)[0]
        sizes[b] = len(members)
        lab = labels[members]
        lab = lab[lab >= 0]
        dom[b] = np.bincount(lab).argmax() if len(lab) else -1

    # process largest blocks first; within size, the seeded permutation breaks ties
    block_order = sorted(range(n_blocks), key=lambda b: (-sizes[b], order[b]))
    fold_of_block = np.full(n_blocks, -1, dtype=np.int64)
    # per-fold per-class running counts
    n_classes = int(labels.max()) + 1 if (labels >= 0).any() else 1
    fold_class = np.zeros((folds, n_classes), dtype=np.int64)
    fold_size = np.zeros(folds, dtype=np.int64)
    for b in block_order:
        c = dom[b]
        if c >= 0:
            # fold with fewest of this class, ties -> smallest fold
            cand = np.lexsort((fold_size, fold_class[:, c]))
        else:
            cand = np.argsort(fold_size)
        f = int(cand[0])
        fold_of_block[b] = f
        fold_size[f] += sizes[b]
        members = np.where(inv == b)[0]
        lab = labels[members]
        lab = lab[lab >= 0]
        if len(lab):
            fold_class[f] += np.bincount(lab, minlength=n_classes)
    return fold_of_block[inv]


def train_val_test_masks(fold: np.ndarray, test_fold: int, val_fold: int):
    """Boolean (train, val, test) masks for a given held-out test/val fold."""
    test = fold == test_fold
    val = fold == val_fold
    train = ~(test | val)
    return train, val, test
