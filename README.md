# soilbot — NJDOT GDMS geotechnical boring database

Extracts all publicly available **NJDOT Geotechnical Data Management System (GDMS)** soil-boring
data from NJDOT's ArcGIS REST services into a clean, queryable local store (DuckDB + GeoPackage),
structured to feed a downstream **Bayesian Graph Neural Network** that predicts soil type by
location. All endpoints are public/anonymous — no auth tokens.

> **The central honesty constraint:** NJDOT exposes **no structured stratigraphy**. SPT N-values,
> USCS classes, depth intervals, groundwater and sample types live **only inside scanned PDF boring
> logs** and require OCR. This pipeline keeps STRUCTURED data (boring locations, soil-class labels,
> covariates) strictly separate from OCR-only data, and **never fabricates stratigraphy** — the
> `strata` table stays empty until real OCR runs. See `schema_audit.md` and `REPORT.md`.

## Data sources (verified live)

| What | Service | Records |
|---|---|--:|
| Boring locations + scanned-log attachments | `Soil_Borings_Map/FeatureServer/0` (AGOL `HggmsDF7UJsNN1FK`) | 49,152 points |
| Boring plan sheets | `…/FeatureServer/1` | 3,829 polygons |
| Engineering soil-class labels (only structured soil field) | `…/FeatureServer/2` | 20,255 points |
| NJ surficial / bedrock geology | NJDEP `mapsdep.nj.gov/.../Geology/MapServer` layers 25 / 14 | 18,004 / 3,563 |
| SSURGO engineering properties | NRCS Soil Data Access (SDA) POST SQL | NJ survey areas |
| DEM elevation/slope (gated) | USGS 3DEP ImageServer | per boring |

Native CRS is **EPSG:102711** (NJ State Plane, US ft), preserved as `geom_native`; everything is
also reprojected to **EPSG:4326** (`geom_4326`, lon/lat).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # duckdb, requests, pyyaml, pdfplumber, numpy, scipy
# optional fallback geo stack (only if DuckDB-spatial GeoPackage/ST_Transform is unavailable):
#   .venv/bin/pip install -r requirements-geo.txt
```
DuckDB's `spatial` extension is auto-installed on first run and cached to `data/.duckdb_extensions`.

## Usage

```bash
.venv/bin/python -m pipeline.run --phase 1     # discovery & schema audit -> schema_audit.md
.venv/bin/python -m pipeline.run --phase 2     # bulk extract layers 0/1/2 -> DuckDB + GeoPackages
.venv/bin/python -m pipeline.run --phase 4     # covariates: geology + SSURGO (DEM gated)
.venv/bin/python -m pipeline.run --phase 5     # node_features + edges.parquet + REPORT.md
.venv/bin/python -m pipeline.run --phase all   # default scope: 1, 2, 4, 5

# gated heavy steps (off by default):
.venv/bin/python -m pipeline.run --phase 3 --download-logs   # crawl ~49k scanned-log PDFs (~3-4 GB)
.venv/bin/python -m pipeline.run --phase 3 --ocr             # OCR-parse downloaded logs into `strata`
.venv/bin/python -m pipeline.run --phase 4 --dem             # per-boring 3DEP elevation (~49k calls)
.venv/bin/python -m pipeline.run --phase 3 --download-logs --limit 200   # bounded batch
```

Everything is **idempotent and resumable**: each query page / PDF / covariate page is checkpointed
in the DuckDB `manifest` table (+ `.done` sentinels), so re-runs skip completed work and partial
runs recover. Structured JSON logs land in `logs/{extract,download,parse}.log`.

## Outputs

- `data/soilbot.duckdb` — normalized store: `borings`, `boring_plans`, `soil_labels`, `strata`
  (OCR-fed), covariate tables, `boring_covariates`, `edges`, `manifest`, and the `node_features` view.
- `data/borings.gpkg`, `data/boring_plans.gpkg`, `data/soil_labels.gpkg` — geometry (EPSG:4326).
- `data/edges.parquet` — GNN edge list (`src, dst, edge_type ∈ {knn, delaunay, same_geology}, weight`).
- `data/raw/` — verbatim GeoJSON pages. `schema_audit.md`, `REPORT.md` — audit + final accounting.

## GNN-ready schema

- **Nodes** = borings. Features via `node_features`: lon/lat, elevation (DEM/log), slope, surficial &
  bedrock geology unit, SSURGO component/drainage/hydrologic group, coordinate-quality flag.
- **Edges** in `edges.parquet`: k-NN + Delaunay (computed in feet for metric correctness) + a
  same-surficial-geology edge type (radius-bounded).
- **`strata`** (OCR-fed): `boring_id, top/bottom_depth, uscs_class, spt_n, sample_type, gw_depth` —
  populated only by `parse_logs.py` once logs are downloaded and OCR'd.

## Known limitations

- **Stratigraphy needs OCR** (no engine installed): `pipeline/parse_logs.py` is a working scaffold
  (pdfplumber vector-text branch + poppler-rasterize OCR seam + regex extractor with worked
  examples); extraction **accuracy is a TODO**.
- **NJDEP geology sits behind an Imperva Incapsula WAF** that trips after ~4–5 rapid requests. The
  fetch uses browser headers + cookie warm-up + burst-and-rest pacing + resumable offset
  checkpoints. The 116k-polygon SSURGO **spatial** footprint layer is **off by default**
  (`covariates.geology.fetch_ssurgo_spatial`) because fetching it under the WAF is slow; SSURGO
  tabular (via SDA, no WAF) still loads as reference.
- **Sampling bias:** borings cluster on transportation corridors (NJDOT drills for projects), so the
  node set is **not** a spatially uniform sample — see the bias note in `REPORT.md`.
