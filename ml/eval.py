"""Metrics for classification + calibration (the Bayesian payoff).

Accuracy alone is meaningless under the 82-class long tail (the top class AR is ~14% of
labels), so we lead with macro-F1 / balanced accuracy, and report calibration (ECE, NLL)
and an uncertainty->error AUROC: does high predictive entropy flag misclassifications?
"""
from __future__ import annotations

import numpy as np


def classification_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """probs [n, C] posterior-mean class probabilities, y [n] true class indices."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, top_k_accuracy_score
    pred = probs.argmax(1)
    out = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
    }
    C = probs.shape[1]
    if C > 3:
        try:
            out["top3_acc"] = float(top_k_accuracy_score(y, probs, k=3, labels=np.arange(C)))
        except Exception:  # noqa: BLE001
            out["top3_acc"] = float("nan")
    out["nll"] = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0)).mean())
    out["ece"] = expected_calibration_error(probs, y)
    out["unc_auroc"] = uncertainty_error_auroc(probs, y)
    return out


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n
    return float(ece)


def uncertainty_error_auroc(probs: np.ndarray, y: np.ndarray) -> float:
    """AUROC of predictive entropy as a detector of misclassification (higher = better)."""
    from sklearn.metrics import roc_auc_score
    pred = probs.argmax(1)
    err = (pred != y).astype(int)
    if err.sum() == 0 or err.sum() == len(err):
        return float("nan")
    ent = -(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(1)
    return float(roc_auc_score(err, ent))


def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution (lower = better)."""
    from scipy.stats import norm
    sigma = np.clip(sigma, 1e-6, None)
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(crps.mean())


def regression_metrics(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> dict:
    """mu/sigma/y in the model's space (e.g. log1p SPT-N). Reports error + calibration."""
    err = mu - y
    out = {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "crps": gaussian_crps(mu, sigma, y),
        # 90% central credible-interval coverage (well-calibrated -> ~0.90)
        "cov90": float((np.abs(err) <= 1.6449 * np.clip(sigma, 1e-6, None)).mean()),
        "n": int(len(y)),
    }
    return out


def reliability_curve(probs: np.ndarray, y: np.ndarray, bins: int = 15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)
