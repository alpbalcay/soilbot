# CLAUDE.md

Guidance for AI agents working in this repo. Read this first; it captures the build/test/run
commands and the non-obvious invariants that are easy to break.

## What this is

`soilbot` is a geotechnical ML pipeline over the NJDOT GDMS soil-boring database. Three layers:

- **`pipeline/`** — Python ETL: extract borings/labels from ArcGIS, OCR scanned boring logs into
  depth-resolved `strata`, attach covariates (geology/SSURGO/DEM), build the spatial graph, and
  run the soil-equation engine. State lives in a single **DuckDB** store (`data/soilbot.duckdb`).
- **`ml/`** — Bayesian GNN models. Phase A predicts soil type at labeled points; Phase B
  (`train3d.py`) predicts depth-resolved SPT-N + USCS from OCR'd borings.
- **`soilbot-rs/`** — Rust PyO3 extension (`soilbot_rs`): the geotechnical soil-equation engine
  plus parity-verified graph/feature construction.

A **Phase 7 literature-review layer is emerging** (`pipeline/litreview.py`): an OpenAlex harvest of
foundational geotech papers → `lit_*` tables + a committed Obsidian vault. It is gated off and not
yet wired into `run.py` — see the Phase 7 note below.

For the science and results, see `README.md`, `REPORT.md`, and `ML_REPORT.md`. This file is only
about *how to work in the repo*.

## Environment & build

All Python runs through the project venv: **`.venv/bin/python`**.

After editing any `soilbot-rs/src/*.rs`, **rebuild the extension** before trusting Python results
(a stale `.so` silently returns old answers):

```bash
cd soilbot-rs && maturin develop --release
```

Rust unit tests need the libpython link path on `RUSTFLAGS` (PyO3 links against libpython, which
isn't on the default linker path here):

```bash
cd soilbot-rs && \
RUSTFLAGS="-L /var/home/linuxbrew/.linuxbrew/lib" PYO3_PYTHON=../.venv/bin/python \
cargo test --lib            # 10 tests
```

## Run commands

Pipeline (one DuckDB store, phases gate the next):

| Phase | Command | What it does |
|-------|---------|--------------|
| 1 | `python -m pipeline.run --phase 1` | Schema discovery/audit → `schema_audit.md` |
| 2 | `python -m pipeline.run --phase 2 [--no-gpkg]` | Bulk-extract borings/plans/labels → DuckDB + GeoPackage |
| 3 | `python -m pipeline.run --phase 3 --download-logs [--ocr] [--limit N]` | **GATED** scanned-log crawl + OCR → `strata` |
| 4 | `python -m pipeline.run --phase 4 [--dem] [--limit N]` | Covariates: geology, SSURGO, (opt) DEM elevation |
| 5 | `python -m pipeline.run --phase 5` | Node features + `edges.parquet` → `REPORT.md` |
| 6 | `python -m pipeline.run --phase 6` | **GATED** soil-equation engine → `strata_derived` |
| all | `python -m pipeline.run --phase all` | Default scope: phases 1, 2, 4, 5 (heavy 3 & 6 stay gated) |
| 7 | *(emerging — not in `run.py`)* | **GATED** OpenAlex lit-review harvest → `lit_*` tables + Obsidian vault |

`--use-rust` forces the `soilbot_rs` graph/feature path regardless of config flags. The `--phase`
argument only accepts `1`–`6` and `all`.

**Phase 7 (litreview) — emerging, gated.** Not yet wired into `run.py` (no `--phase 7`); invoke the
module directly via `pipeline.litreview.run(config, log)` and the `litreview: false` gate
(`config.yaml`) guards it. It pulls ~40 canonical seed queries from OpenAlex (polite pool via
`mailto`, **no API key**), expands one citation hop, ranks by `cited_by_count`, and persists to
`lit_papers` / `lit_citations` (+ JSON metadata cache). Key config (`config.yaml` `litreview:`
block): `max_papers: 300`, `per_seed: 25`, `hops: 1`, `min_year: 1936`. Outputs split by
trackability: `litreview/vault` (Obsidian graph) is **committed**, while `litreview/pdfs`,
`litreview/fulltext`, and `litreview/metadata` are **gitignored**.

ML (GPU `device: cuda` in config, CPU fallback). Phase A predicts soil type at labeled points;
Phase B is depth-resolved SPT-N + USCS.

```bash
# Phase A (soil type @ labeled points)
.venv/bin/python -m ml.assemble                       # A0.5 prerequisite: labeled-graph union + covariates -> ml_edges
.venv/bin/python -m ml.train --mode a1 --folds 5      # a1 deterministic baseline (+ warm-start)
.venv/bin/python -m ml.train --mode a2 --folds 5      # a2 Bayesian (Bayes-by-Backprop, ELBO), warm-started from a1
.venv/bin/python -m ml.train --mode a3 --folds 5      # a3 = a2 + empirical-Bayes geology prior

# Phase B (depth-resolved)
.venv/bin/python -m ml.train3d --folds 5              # B1 (stress baseline)        -> cv_b1.json
.venv/bin/python -m ml.train3d --physics --folds 5    # B2 (+ physics inputs)       -> cv_b1_physics.json
.venv/bin/python -m ml.report                         # consolidate -> ML_REPORT.md
```

## Reporting & finalization

Optional helpers, run from the repo root:

- `.venv/bin/python scripts/data_report.py` → `data_report.html` — a self-contained Plotly
  dashboard built from the **live DB (read-only)** + the ML CV JSONs. Use it for current corpus
  stats; the markdown `REPORT.md` is a stale Phase-5 snapshot. Safe against a locked DB.
- `bash run_finish.sh` — one-shot, **resumable** finisher: loops OCR mop-up over reset-failed logs
  until the parse count stabilizes, rebuilds the 3D dataset on the full corpus, then runs the final
  5-fold B1; logs to `logs/finish_progress.out` / `logs/finish_ocr.out`. (Contrast
  `scripts/auto_phase6_b2.sh`, the detached watcher below.)

## Tests

Run with the venv from the repo root. Each smoke test **skips cleanly (exit 0)** when `soilbot_rs`
isn't built or the relevant table is empty — a skip is not a pass, read the output.

- `tests/parity_rust.py` — Rust ↔ scipy/numpy parity for edges + node features (read-only DB).
  Prints `PARITY OK`.
- `tests/smoke_soil.py` — runs the soil engine and asserts physical plausibility (σ'v0 > 0 and
  monotone with depth, φ′/Dr in range, `su_tsf > 0` where non-NULL). Prints `SMOKE OK`.
- `tests/smoke_b2.py` — B2 leakage guard + Dataset3D coverage + decoder shapes.
- `tests/smoke_ml.py` — Phase-A model forward/backward + spatial-CV disjointness (synthetic, no DB).

**DB-writing tests + the single-writer lock:** `smoke_soil.py` and `smoke_b2.py` rewrite
`strata_derived`, so they need the writer lock. If the live DB may be in use, point them at a
snapshot copy:

```bash
cp data/soilbot.duckdb /tmp/snap.duckdb
SOILBOT_DB=/tmp/snap.duckdb .venv/bin/python tests/smoke_soil.py
```

## Critical invariants — do not break

- **DuckDB is single-writer.** A running pipeline/OCR job holds an exclusive lock. Never open the
  live DB read-write concurrently — use `read_only=True`, or operate on a snapshot copy via the
  `SOILBOT_DB` env var (above). Opening read-write against a locked DB throws. `scripts/data_report.py`
  respects this (opens `read_only=True`); the Phase 7 harvest reuses the shared
  `RateLimiter`/`backoff_delay`/`manifest_*` politeness + idempotency primitives, so it is re-run-safe.

- **Leakage boundary.** `spt_n` is the prediction target, so anything derived *from* it must never
  be a model input. `ml/data3d.py` encodes this:
  - `SAFE_PHYS_COLS = ["sigma_eff_v0_tsf", "sigma_v0_tsf", "gamma_pcf", "cn"]` — stress-based, safe.
  - `LEAKY_COLS = ["n60", "n1_60", "phi_peck_deg", "phi_hatanaka_deg", "dr_pct", "su_tsf",
    "e_modulus_tsf", "m_constrained_tsf", "allow_bearing_tsf", "n1_60cs", "crr"]` — functions of
    `spt_n`, forbidden as inputs.
  - The guard `assert not (set(SAFE_PHYS_COLS) & set(LEAKY_COLS))` must stay. Only add a column to
    `SAFE_PHYS_COLS` if it is provably independent of `spt_n`.

- **NULL-at-N=0.** Strength/stiffness are linear in N and collapse to a meaningless 0 at
  `spt_n = 0` (a clay does not have zero Su). The engine emits **NULL** there for Su/E/M
  (`soilbot-rs/src/profile.rs`, `let strength_ok = n60 > 0.0;`); φ′/Dr/bearing stay populated
  (defensible at the loosest state). `tests/smoke_soil.py` asserts `su_tsf > 0` wherever non-NULL
  — don't reintroduce 0s.

- **Feature gates default OFF** in `config.yaml`: `graph.use_rust`, `ml.use_rust` (forces the Rust
  feature-assembly path), `soil_engine.enabled`, `ml.b1.physics_features`, and `litreview` (Phase 7).
  The Rust graph/feature paths are parity-verified bit-for-bit against scipy/numpy; if you touch
  them, keep `parity_rust.py` green before flipping any gate on.

## Conventions

- `data/`, `logs/`, and `.claude/` are gitignored — build/run artifacts (DuckDB, parquet, `.pt`
  caches, ML JSONs) stay local and are never committed. Phase 7 follows the same rule: only
  `litreview/vault` (the Obsidian graph) is committed; `litreview/{pdfs,fulltext,metadata}` are not.
- **Don't `git push`** unless explicitly asked. Commit to `main` with the repo's terse one-line
  message style.
- `scripts/auto_phase6_b2.sh` is a detached, idempotent watcher that re-runs phase 6 + B1/B2 +
  report once OCR completes; it polls so it never holds the DB lock or GPU while waiting.
