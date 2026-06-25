"""Diagnose data-noise vs model-ceiling for the B1/B2 depth-resolved SPT models, using the gold set.

Step 5 of the OCR gold-set validation. Intersects the models' out-of-fold predictions
(data/ml/preds_b{1,2}.json, from `ml.train3d --dump-preds`) with the hand-transcribed gold N
values, then scores each model THREE ways on the same matched samples:
  (a) model vs the OCR target it was trained/evaluated on,
  (b) model vs the gold-corrected true N,
  (c) (b) split into "clean" (OCR≈gold) vs "noisy" (OCR wrong) samples,
and reports the OCR label-noise floor (RMSE of OCR-vs-gold). A depth-mean baseline is scored
against gold for reference.

Verdict logic: if the label-noise floor is comparable to the model's apparent error, and the model
scores well on clean gold samples, the B1/B2 gap is data-driven (OCR noise), not a model ceiling.

Run (after `ml.train3d --folds 5 --dump-preds` and `--physics --dump-preds`):
    .venv/bin/python scripts/gold_diag.py
Writes gold/diag.json and prints a summary.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import eval as ev  # noqa: E402

M_TO_FT = 3.280839895
DEPTH_TOL_FT = 2.5
CLEAN_BLOWS = 3       # |OCR - gold| <= this (in blows) counts as a "clean" label
USABLE = {"ok", "med"}


def gold_truth():
    """Per-boring list of (depth_ft_top, depth_ft_bot, true_N) for usable gold rows."""
    out = {}
    for rec in (json.loads(l) for l in open("gold/labels.jsonl")):
        unit = rec["unit"]; k = M_TO_FT if unit == "m" else 1.0
        rows = []
        for r in rec["rows"]:
            if r.get("n_flag") in USABLE and r.get("spt_n") is not None:
                t = r["top"] * k if r["top"] is not None else None
                b = r["bottom"] * k if r["bottom"] is not None else None
                rows.append((t, b, float(r["spt_n"])))
        if rows:
            out[rec["boring_id"]] = rows
    return out


def match_preds(preds, truth):
    """For each prediction in a gold boring, attach the nearest gold true-N within tolerance.
    Returns arrays: pred_mu, pred_sigma (log space), y_ocr (blows), y_gold (blows)."""
    by_boring = {}
    for i, bid in enumerate(preds["boring_id"]):
        by_boring.setdefault(bid, []).append(i)
    mu, sig, yo, yg = [], [], [], []
    for bid, rows in truth.items():
        idxs = by_boring.get(bid, [])
        for i in idxs:
            d = preds["depth_ft"][i]
            yocr_log = preds["y_ocr_log"][i]
            if yocr_log is None or yocr_log < 0:     # no SPT target for this sample
                continue
            best, bestd = None, DEPTH_TOL_FT
            for (t, b, n) in rows:
                for e in (t, b):
                    if e is not None and abs(d - e) <= bestd:
                        best, bestd = n, abs(d - e)
            if best is not None:
                mu.append(preds["pred_mu"][i]); sig.append(preds["pred_sigma"][i])
                yo.append(float(np.expm1(yocr_log))); yg.append(best)
    return (np.array(mu), np.array(sig), np.array(yo), np.array(yg))


def score(mu, sig, y_blows):
    return ev.regression_metrics(mu, sig, np.log1p(y_blows))


def diag_one(tag, preds, truth):
    mu, sig, yo, yg = match_preds(preds, truth)
    if len(mu) == 0:
        return {"tag": tag, "matched": 0}
    noise_rmse = float(np.sqrt(((np.log1p(yo) - np.log1p(yg)) ** 2).mean()))
    clean = np.abs(yo - yg) <= CLEAN_BLOWS
    res = {
        "tag": tag, "matched": int(len(mu)),
        "label_noise_rmse_log": round(noise_rmse, 3),
        "clean_frac": round(float(clean.mean()), 3),
        "vs_ocr": {k: round(v, 3) for k, v in score(mu, sig, yo).items()},
        "vs_gold": {k: round(v, 3) for k, v in score(mu, sig, yg).items()},
        "vs_gold_clean": {k: round(v, 3) for k, v in score(mu[clean], sig[clean], yg[clean]).items()}
        if clean.any() else None,
        "vs_gold_noisy": {k: round(v, 3) for k, v in score(mu[~clean], sig[~clean], yg[~clean]).items()}
        if (~clean).any() else None,
    }
    # depth-mean baseline scored against gold (constant = mean of gold-clean log N)
    base_mu = np.log1p(yg).mean(); base_sd = np.log1p(yg).std() or 1.0
    res["baseline_depth_mean_vs_gold"] = {
        k: round(v, 3) for k, v in
        ev.regression_metrics(np.full(len(yg), base_mu), np.full(len(yg), base_sd), np.log1p(yg)).items()}
    return res


def main():
    out = Path("data/ml")
    truth = gold_truth()
    diags = {}
    for tag, fn in (("B1", "preds_b1.json"), ("B2", "preds_b2.json")):
        p = out / fn
        if not p.exists():
            print(f"SKIP {tag}: {p} missing (run ml.train3d --dump-preds)")
            continue
        diags[tag] = diag_one(tag, json.loads(p.read_text()), truth)

    Path("gold/diag.json").write_text(json.dumps(diags, indent=2))
    print("=== DATA-vs-MODEL DIAGNOSIS (gold subset) ===")
    for tag, d in diags.items():
        if d.get("matched", 0) == 0:
            print(f"{tag}: no matched gold samples"); continue
        print(f"\n{tag}: matched={d['matched']} clean_frac={d['clean_frac']} "
              f"label_noise_RMSE(log)={d['label_noise_rmse_log']}")
        print(f"  model vs OCR target : CRPS={d['vs_ocr']['crps']} RMSE(log)={d['vs_ocr']['rmse']}")
        print(f"  model vs GOLD truth : CRPS={d['vs_gold']['crps']} RMSE(log)={d['vs_gold']['rmse']}")
        if d["vs_gold_clean"]:
            print(f"    on CLEAN gold     : CRPS={d['vs_gold_clean']['crps']} "
                  f"RMSE(log)={d['vs_gold_clean']['rmse']} (n={d['vs_gold_clean']['n']})")
        if d["vs_gold_noisy"]:
            print(f"    on NOISY gold     : CRPS={d['vs_gold_noisy']['crps']} "
                  f"RMSE(log)={d['vs_gold_noisy']['rmse']} (n={d['vs_gold_noisy']['n']})")
        print(f"  depth-mean vs gold  : CRPS={d['baseline_depth_mean_vs_gold']['crps']} "
              f"RMSE(log)={d['baseline_depth_mean_vs_gold']['rmse']}")
    print("\nwrote gold/diag.json")


if __name__ == "__main__":
    main()
