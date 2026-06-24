# soilbot-rs

Rust data-engineering + geotechnical soil-equation engine for `soilbot`, exposed to the Python
pipeline as a PyO3/maturin extension module (`import soilbot_rs`). Numeric heavy-lifting only —
DuckDB I/O and orchestration stay in Python. Every entry point has a Python fallback at its call
site, so an un-built extension never hard-breaks the pipeline.

## What it provides

| Function | Replaces / adds | Python call site |
|---|---|---|
| `compute_edges(ids, xy, units, knn_k, …)` | kNN/Delaunay/same_geology edges (was scipy) | `pipeline/graph.py` |
| `label_boring_edges(ids, xy, types, k)` | label→boring bridge edges (was scipy) | `ml/graph_build.py` |
| `assemble_features(xy, elevation, freqs)` | coord/elev norm + Fourier encoding (was numpy) | `ml/data.py` |
| `soil_profile(strata cols…, config_json)` | **new** geotechnical engine → `strata_derived` | `pipeline/derive.py` |
| `geodesic_distance(lon1,lat1,lon2,lat2, method)` | **new** haversine/Vincenty ground distance | — |
| `edge_lengths_geodesic(lon, lat, src, dst, method)` | **new** per-edge geodesic length | — |

The graph/feature ports are bit-comparable to scipy/numpy (weights and features identical;
edges identical up to equidistant-neighbour tie-breaking). The soil engine computes, per strata
interval: effective vertical stress σ'v0, N60 → CN → (N1)60, friction angle (Peck,
Hatanaka-Uchida), relative density (Skempton), undrained shear strength (Stroud), Young's /
constrained modulus (AASHTO), Meyerhof allowable bearing, and gated liquefaction (N1)60cs + CRR
(Idriss-Boulanger). Properties gate to the soil behaviour they are valid for and are NULL
otherwise — nothing is fabricated.

## Build

```bash
# one-time toolchain (if cargo is absent)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
pip install -r ../requirements-rs.txt          # maturin, into the active venv

# build + install into the venv (release is important — debug kd-tree/triangulation is slow)
maturin develop --release
```

The wheel builds against the stable abi3 ABI (`abi3-py311`), so one build serves Python ≥ 3.11.

## Test

```bash
# Rust unit tests (textbook worked examples + physical-sanity invariants).
# `extension-module` is OFF for tests so the harness links libpython; point the linker at a
# libpython3.x.so and pass the interpreter:
RUSTFLAGS="-L /path/to/libpython/dir" PYO3_PYTHON=../.venv/bin/python cargo test --lib

# Python parity vs scipy/numpy on the real DuckDB store (graph + features):
SOILBOT_DB=<writable copy> ../.venv/bin/python ../tests/parity_rust.py

# Python smoke + sanity for the soil engine (builds strata_derived from real strata):
SOILBOT_DB=<writable copy> ../.venv/bin/python ../tests/smoke_soil.py
```

## Enable in the pipeline

Flags in `config.yaml` (all default off — opt in per stage):

```yaml
graph: { use_rust: true }        # graph construction via soilbot_rs
ml:    { use_rust: true }        # feature assembly via soilbot_rs
soil_engine: { enabled: true }   # run the derive phase -> strata_derived
```

or `python -m pipeline.run --phase 6` for the soil engine, and `--use-rust` to force the
graph/feature path for a run.
