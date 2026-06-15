"""Non-GNN baselines on the identical spatial folds — the bar the GNN must clear.

  nearest_label : predict the class of the nearest TRAIN label (IDW/k-NN spatial baseline).
  rf_covariates : RandomForest on geology + coords only (does the graph add anything?).

Both consume the cached Dataset and the same `kfold_block_split`, so numbers are directly
comparable to ml.train.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

from pipeline.config import Config
from pipeline.logging_setup import new_run_id, setup

from . import eval as ev
from .data import Dataset
from .splits import kfold_block_split


def _probs_from_neighbors(neigh_codes, n_classes):
    """Soft vote over neighbour codes -> probability row."""
    p = np.zeros(n_classes)
    for c in neigh_codes:
        p[c] += 1
    s = p.sum()
    return p / s if s else np.full(n_classes, 1.0 / n_classes)


def nearest_label(ds, fold, test_fold, val_fold, k=5):
    y = ds.y_code.numpy()
    label = y >= 0
    tr = label & (fold != test_fold) & (fold != val_fold)
    te = label & (fold == test_fold)
    n_classes = len(ds.code_classes)
    tree = cKDTree(ds.xy[tr])
    _, nn = tree.query(ds.xy[te], k=min(k, int(tr.sum())))
    nn = np.atleast_2d(nn)
    tr_codes = y[tr]
    probs = np.stack([_probs_from_neighbors(tr_codes[row], n_classes) for row in nn])
    return ev.classification_metrics(probs, y[te])


def rf_covariates(ds, fold, test_fold, val_fold):
    from sklearn.ensemble import RandomForestClassifier
    y = ds.y_code.numpy()
    label = y >= 0
    tr = label & (fold != test_fold) & (fold != val_fold)
    te = label & (fold == test_fold)
    # features: coords + categorical geology indices + numerics
    X = np.concatenate([ds.xy, ds.cat_idx.numpy().astype(float), ds.x_num.numpy()], axis=1)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced_subsample",
                                 min_samples_leaf=2, random_state=0)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])
    # map classifier's class order back to full code space
    full = np.zeros((te.sum(), len(ds.code_classes)))
    full[:, clf.classes_] = proba
    return ev.classification_metrics(full, y[te])


def run(cfg, log, folds=5):
    out = cfg.abspath(cfg.get("ml", "out_dir", default="data/ml"))
    ds = Dataset.load(out / "dataset.pt")
    seed = int(cfg.get("ml", "seed", default=1337))
    fold = kfold_block_split(ds.xy, ds.y_code.numpy(), seed=seed,
                             block_size_ft=float(cfg.get("ml", "splits", "block_size_ft", default=20000)),
                             folds=folds)
    results = {"nearest_label": [], "rf_covariates": []}
    for tf in range(folds):
        vf = (tf + 1) % folds
        nl = nearest_label(ds, fold, tf, vf)
        rf = rf_covariates(ds, fold, tf, vf)
        results["nearest_label"].append(nl)
        results["rf_covariates"].append(rf)
        log.info("baseline_fold", fold=tf,
                 nl_f1=round(nl["macro_f1"], 4), nl_acc=round(nl["accuracy"], 4),
                 rf_f1=round(rf["macro_f1"], 4), rf_acc=round(rf["accuracy"], 4))
    agg = {m: {k: float(np.nanmean([r[k] for r in results[m]]))
               for k in ("macro_f1", "balanced_acc", "accuracy", "nll", "ece")}
           for m in results}
    (out / "baselines.json").write_text(json.dumps({"folds": results, "mean": agg}, indent=2))
    log.info("baselines_done", **{f"{m}_{k}": round(v, 4)
                                  for m, d in agg.items() for k, v in d.items()})
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    cfg = Config.load(None)
    rid = new_run_id()
    logger = setup(cfg.path("log_dir"), "ml.log", rid, "baselines", console=True)
    agg = run(cfg, logger, folds=args.folds)
    print("\n=== Baselines (spatial 5-fold mean) ===")
    for m, d in agg.items():
        print(f"  {m:16} macroF1={d['macro_f1']:.3f} balAcc={d['balanced_acc']:.3f} "
              f"acc={d['accuracy']:.3f} ECE={d['ece']:.3f} NLL={d['nll']:.3f}")
