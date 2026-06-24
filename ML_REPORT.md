# ML_REPORT.md — Bayesian GNN for NJ soil type (Phase A)

_Generated 2026-06-24T01:59:18+00:00. Spatial block cross-validation (5 folds, ~20,000 ft blocks held out whole, so train/test are spatially separated — honest extrapolation, no corridor leakage)._

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

## B1 — 3D depth-resolved (SPT-N / USCS-at-depth)

Depth-conditioned decoder on the spatial GNN latent, trained on OCR'd 'Blows on Spoon' split-spoon profiles (depth + SPT-N + USCS). Spatial-block CV over borings; SPT-N in log1p space; T=30 posterior samples. **SPT-N is the non-redundant signal geology can't provide** (the relabeling experiment showed coarse soil-class is geology-derivable).

| Predictor | SPT-N CRPS ↓ | 90% coverage | RMSE (log) |
|---|--:|--:|--:|
| B1 3D GNN | 0.509 | 0.889 | 0.913 |
| baseline: depth_mean | 0.509 | 0.934 | 0.902 |
| baseline: geology_depth_gbm | 0.526 | 0.737 | 0.910 |

USCS-at-depth: macro-F1 0.102. SPT-N back-transformed RMSE ≈ 25 blows.

_**Honest status:** at the current OCR scale (~15,679 SPT samples across the spatial folds) the 3D GNN **underperforms a depth-mean baseline** — a heteroscedastic Bayesian model over-fits with this few spatially-CV'd samples. The pipeline (depth conditioning, calibrated SPT intervals, baselines) is validated end-to-end; broader OCR coverage is still needed for the GNN to pay its way. **OCR'd SPT-N values carry digit-error noise** (sanity-gated to 0–100 blows, 0–200 ft); a hand-labeled gold set is still owed before trusting individual N._

## B2 — physics-grounded inputs (effective stress)

Same B1 architecture and raw-SPT-N target, but the decoder additionally consumes the **non-leaky** geotechnical context from `strata_derived`: effective vertical stress σ'v0, total stress σv0, unit weight γ, and the overburden factor CN (each a function of depth + USCS + groundwater only — **never** of SPT-N). The corrected/derived strength properties ((N1)60, φ′, Su) are pure functions of the SPT-N target and are therefore **excluded as inputs** (a code-level allowlist/denylist enforces this).

| Predictor | SPT-N CRPS ↓ | 90% coverage | RMSE (log) |
|---|--:|--:|--:|
| B2 3D GNN (+σ'v0) | 0.500 | 0.895 | 0.902 |
| B1 3D GNN (depth only) | 0.509 | 0.889 | 0.913 |
| baseline: depth_mean | 0.509 | 0.934 | 0.902 |
| baseline: geology_depth_phys_gbm | 0.523 | 0.705 | 0.900 |

SPT-N back-transformed RMSE ≈ 25 blows.

_**Honest status:** adding σ'v0 lowers mean CRPS 0.509→0.500 and RMSE(log), with the gain concentrated in the high-error spatial folds (B2 beats B1 in 3 of 5 folds). At ~15,679 SPT samples, B2 now narrowly edges the depth-mean baseline on CRPS (0.500 vs 0.509) and ties it on RMSE(log), though the baseline stays better-calibrated (90% coverage 0.895 vs 0.934); B1 (depth-only) still does not beat it — a margin within fold noise. σ'v0 is genuinely independent of the SPT-N target, but is computed from **USCS-defaulted unit weights** (an estimate, not lab γ) and standardized over the full sample (a small normalization optimism shared by B1/B2); re-judge as more data accrues._

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

