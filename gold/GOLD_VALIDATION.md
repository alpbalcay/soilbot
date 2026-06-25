# OCR Gold-Set Validation

Independent ground truth from **80 SPT-bearing boring logs** (hand-transcribed by vision from the rendered PDFs across 22 vendor log formats), scored against the OCR'd `strata` table. Reproduce: `python scripts/gold_sample.py` → transcribe `gold/labels.jsonl` → `python scripts/gold_score.py`; model diagnosis via `ml.train3d --dump-preds` + `scripts/gold_diag.py`.

## How accurate is the OCR'd SPT-N?

- **Interval detection** (did OCR find the samples that exist?): precision **0.874**, recall **0.481**, F1 **0.621** (matched 208, missed 224, spurious 30). OCR misses roughly half the SPT samples present on the logs.
- **SPT-N value accuracy** on captured samples (n=182): **65%** correct within ±2 blows; MAE **6.16 blows**, median abs err **0.0**, signed bias **0.09** (essentially unbiased).
- **Error taxonomy**: exact=109, gross=35, single_increment=26, within_2=10, off_by_10x=1, transposition=1. 'single_increment' = OCR recorded one 6in/150mm increment instead of N (N = sum of the 2nd+3rd); 'gross' = unrelated misread (often a refusal blow count).
- **USCS class** (n=114): exact **61%**, coarse family **68%**.
- **Combined yield**: with ~50% recall × ~64% value accuracy, only about a third of the true SPT samples are both captured *and* correct.

## Is the heuristic `confidence` meaningful?

Barely usable as a filter — it almost never fires:

| OCR confidence | n | N-accuracy |
|---|---|---|
| =1.0 | 174 | 0.678 |
| 0.7-0.99 | 6 | 0.167 |
| <0.7 | 2 | 0.0 |

Lower confidence does track lower accuracy, but ~96% of values carry `confidence=1.0`, so the score cannot flag the errors it should.

## Data noise vs model ceiling (B1/B2 on the gold subset)

Matched **176** model out-of-fold predictions to gold N. The OCR **label-noise floor is RMSE(log)=0.702** — comparable to the models' total apparent error, i.e. much of the measured error is irreducible label noise.

| model | vs OCR target (CRPS) | vs GOLD truth (CRPS) | on CLEAN gold | depth-mean vs gold |
|---|---|---|---|---|
| B1 | 0.478 | 0.438 | 0.438 | 0.469 |
| B2 | 0.448 | 0.421 | 0.397 | 0.469 |

**Both models predict gold truth *better* than the OCR labels they were scored against** (vs-GOLD CRPS < vs-OCR CRPS) — the noisy eval labels inflate the headline CRPS. Against ground truth both edge the depth-mean baseline, and on clean gold samples B2 improves markedly. **Verdict: the B1/B2 gap is substantially data-driven (OCR noise), not a model ceiling.**

_Caveat: 176 matched gold samples — directional, wide error bars._

## Parser fix — before → after

`pipeline/parse_logs.py` was fixed for the failure modes above (sample-id recall, refusal/WOR handling, increment capture). 'Before' is the parser deployed on the live `strata`; 'after' is the fixed parser re-scored on the same gold borings (`scripts/gold_reocr.py`), window-fair on both.

| metric | before | after |
|---|---|---|
| interval recall | 0.481 | **0.785** |
| interval F1 | 0.621 | **0.759** |
| SPT-N accuracy (±2) | 0.654 | **0.679** |
| correct N values | 119 | **169** |
| MAE (blows) | 6.16 | **4.47** |

Recall and the count of correct SPT values rise sharply (the leading-'S'→digit misread that dropped ~half the samples is fixed); refusals no longer leak a bogus N≈50 and WOR/WOH now read N=0, lowering MAE. Precision dips as expected when recall rises. Apply to the corpus by re-running phase-3 OCR (resettable parse manifest).

## What this implies

- The highest-leverage data fix was the **OCR extractor** — now done (see the before/after above); applying it corpus-wide should shrink the OCR label-noise floor and, per the diagnosis, widen B2's margin over the baseline.

- Model-side, the depth-resolved GNN already carries real signal that the noisy labels mask; cleaner targets should widen the B2-over-baseline margin.

