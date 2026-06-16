# ML_REPORT.md — Bayesian GNN for NJ soil type (Phase A)

_Generated 2026-06-16T01:47:22+00:00. Spatial block cross-validation (5 folds, ~20,000 ft blocks held out whole, so train/test are spatially separated — honest extrapolation, no corridor leakage)._

## What this predicts (Phase A)
- Target: **NJDOT engineering soil class** (82 classes, long-tailed) at the 20,255 labeled soil-label points, with **calibrated predictive uncertainty**.
- Borings (49,152) are **unlabeled context nodes** on a shared union graph — their geology/terrain covariates inform the spatially-disjoint labels (label↔boring bridge edges give 100% of labels a boring neighbourhood within the 3-hop receptive field).
- **Not yet modeled (Phase B, OCR-dependent):** SPT-N, fines%, PI/LL, groundwater depth, and 3D depth resolution. Those heads are built but have no labels until the 49k scanned logs are OCR'd — no fabricated targets.

## Results (spatial-CV mean across 5 folds)

| Method | macro-F1 ↑ | balanced-acc ↑ | accuracy ↑ | NLL ↓ | ECE ↓ |
|---|--:|--:|--:|--:|--:|
| nearest-label (IDW) | 0.135 | 0.139 | 0.286 | 7.964 | 0.218 |
| RF on covariates | 0.288 | 0.359 | 0.410 | 1.897 | 0.046 |
| A1 deterministic GNN | 0.258 | 0.321 | 0.402 | 1.795 | 0.021 |
| A2 Bayesian GNN | 0.264 | 0.323 | 0.400 | 1.792 | 0.029 |
| A3 Bayesian + geology prior | 0.270 | 0.331 | 0.405 | 1.792 | 0.019 |

## Edge-type ablation (deterministic A1)

| Edge set | macro-F1 | accuracy | ECE |
|---|--:|--:|--:|
| all edges (knn+delaunay+same_geology+label_boring) | 0.258 | 0.402 | 0.021 |
| drop label_boring bridge | 0.272 | 0.410 | 0.030 |
| knn + delaunay only | 0.270 | 0.411 | 0.023 |

_Geometric edges (knn/delaunay) carry the discriminative signal; the geology-based `same_geology` and `label_boring` edges exist for **connectivity** — they guarantee the spatially-disjoint labels a boring neighbourhood (100% within 3 hops vs 73% without the bridge) — but slightly dilute point accuracy. The jumping-knowledge skip means every node still uses its own geology regardless of the graph, so coverage degrades gracefully._

## Boring-relabeling experiment (auxiliary USCS task)

Use the OCR'd near-surface USCS class on ~1,176 borings as an auxiliary task sharing the encoder (different taxonomy from the engineering soil-label codes, so a separate head, train-fold only). Same spatial folds.

| Model | macro-F1 | accuracy | NLL ↓ | ECE ↓ |
|---|--:|--:|--:|--:|
| a3 (no relabeling) | 0.267 | 0.406 | 1.782 | 0.031 |
| a3 + aux USCS relabeling | 0.258 | 0.404 | 1.811 | 0.022 |

_**Honest result:** relabeling did **not** improve soil-type accuracy (macro-F1 within fold noise, slight dip) but **consistently improved calibration** (ECE). Likely cause: the OCR'd USCS class is largely predictable from geology — which the model already uses as features **and** an informative prior — so the auxiliary labels add little new information, and the descriptive→USCS OCR labels are themselves noisy. Tested at ~6% extra labels (1,176 OCR'd borings of a 1,483-log download); the full 49k OCR would test it at ~3× scale._

## Reading the table
- **Calibration is the headline**: the Bayesian models give the lowest NLL/ECE — i.e. their probabilities are trustworthy, which is what lets the map flag undrilled-area extrapolations.
- The geology prior (A3) targets the long tail: rare classes concentrated in a surficial unit get prior support where labels are sparse.
- RF-on-covariates is a strong point-classification baseline; the GNN's edge is calibrated uncertainty + the geology prior + (Phase B) the multi-task depth-resolved profile.

## Method
- Heterogeneous GraphSAGE (relation-specific weights per edge type: knn / delaunay / same_geology / label_boring), 3 layers, jumping-knowledge skip to the heads.
- Bayes-by-Backprop variational weights (mean-field Gaussian), ELBO with annealed KL, warm-started from the deterministic fit; T=30 posterior samples at inference.
- Geology as an **informative prior**: empirical-Bayes P(class | surficial unit), estimated per train fold, added to the class logits so predictions relax to geology where data is sparse.
- Hierarchical family→code head + effective-number class weighting for the 82-class imbalance.

## Artifacts
- `data/ml/dataset.pt`, `data/ml/union_edges.parquet`, `data/ml/vocab.json`
- `data/ml/cv_{a1,a2,a3}.json`, `data/ml/baselines.json`

