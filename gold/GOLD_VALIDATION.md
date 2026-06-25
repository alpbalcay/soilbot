# OCR Gold-Set Validation

Independent ground truth from **80 SPT-bearing boring logs** (hand-transcribed by vision from the rendered PDFs across 22 vendor log formats), scored against the OCR'd `strata` table. Reproduce: `python scripts/gold_sample.py` → transcribe `gold/labels.jsonl` → `python scripts/gold_score.py`; model diagnosis via `ml.train3d --dump-preds` + `scripts/gold_diag.py`.

## How accurate is the OCR'd SPT-N?

- **Interval detection** (did OCR find the samples that exist?): precision **0.741**, recall **0.502**, F1 **0.599** (matched 217, missed 215, spurious 76). OCR misses roughly half the SPT samples present on the logs.
- **SPT-N value accuracy** on captured samples (n=188): **64%** correct within ±2 blows; MAE **6.3 blows**, median abs err **0.0**, signed bias **-0.11** (essentially unbiased).
- **Error taxonomy**: exact=109, gross=39, single_increment=27, within_2=11, off_by_10x=1, transposition=1. 'single_increment' = OCR recorded one 6in/150mm increment instead of N (N = sum of the 2nd+3rd); 'gross' = unrelated misread (often a refusal blow count).
- **USCS class** (n=119): exact **60%**, coarse family **66%**.
- **Combined yield**: with ~50% recall × ~64% value accuracy, only about a third of the true SPT samples are both captured *and* correct.

## Is the heuristic `confidence` meaningful?

Barely usable as a filter — it almost never fires:

| OCR confidence | n | N-accuracy |
|---|---|---|
| =1.0 | 180 | 0.661 |
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

## What this implies

- The single highest-leverage data fix is the **OCR extractor**: (1) raise recall (it drops every other sample on many logs), and (2) fix the SPT-N rule so N is the sum of the 2nd+3rd drive increments rather than a single increment, and handle refusals (`50/x`, `100/y`, `WOR`/`WOH`) explicitly.

- Model-side, the depth-resolved GNN already carries real signal that the noisy labels mask; cleaner targets should widen the B2-over-baseline margin.

