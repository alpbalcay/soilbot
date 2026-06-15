# REPORT.md — NJDOT GDMS boring database

_Generated 2026-06-15T04:47:14+00:00 (run `r-ad503f71`). See `schema_audit.md` for the full field-level audit._

## 1. Layers found & record counts (structured, REST-extracted)

| Layer | Geometry | Records | Notes |
|---|---|--:|---|
| GDMS Boring Log (0) | point | 49,152 | boring locations + metadata; one scanned PDF each |
| GDMS Boring Plan (1) | polygon | 3,829 | project plan sheets (transport corridors) |
| Geol Soil Egr Label (2) | point | 20,255 | only structured soil class (PRIMARY/SECONDARY_LABEL, DRAINAGE) |

- Coordinate quality (borings): `ok`=49,152.
- Borings linked to a scanned-log attachment URL: **49,151** of 49,152.
- Native CRS EPSG:102711 (NJ State Plane US ft) preserved as `geom_native`; `geom_4326` = WGS84.

## 2. Structured stratigraphy vs OCR-only  ⚠️ the key distinction

- Borings with **STRUCTURED** stratigraphy (SPT-N / USCS / depth intervals): **0 (0.0%)**.
- Borings whose stratigraphy exists **ONLY inside scanned PDF logs (OCR required)**: **49,151 (~100%)**.
- `strata` table rows currently populated: **0**  (parse attempts pending OCR: 0; PDFs downloaded: 0).

> The NJDOT GDMS exposes **no** structured SPT N-values, USCS classes, depth intervals, groundwater or sample types in any REST field. The only path to them is OCR of the scanned Layer-0 PDF attachments (gated behind `--download-logs` + `--ocr`; the OCR engine is not yet installed). No stratigraphy is fabricated — `strata` stays empty until real parsing runs. The lone structured soil signal is the coarse `PRIMARY_LABEL` class on the 20,255 soil-label points.

## 3. Supplementary covariate layers (Phase 4)

| Source | Table | Rows | Per-boring coverage |
|---|---|--:|---|
| NJDEP surficial geology | geology_surficial | 18,004 | 98.3% (48,302) |
| NJDEP bedrock geology | geology_bedrock | 3,563 | 99.7% (49,008) |
| SSURGO footprints (NJDEP L11) | ssurgo_spatial | 0 | 0.0% (0) |
| SSURGO mapunit (SDA) | ssurgo_mapunit | 2,135 | join via mukey |
| SSURGO component (SDA) | ssurgo_component | 7,579 | dominant component per mukey |
| SSURGO horizons (SDA) | ssurgo_chorizon | 37,588 | engineering props (sieve, LL, PI, ksat) |
| SSURGO muaggatt (SDA) | ssurgo_muaggatt | 2,135 | hydrologic group, depth-to-bedrock, WT depth |
| USGS 3DEP DEM | dem_samples | 0 | gated (`--dem`); elevation/slope per boring |

Top surficial units under borings: RAHWAY TILL (6,525), SALT-MARSH AND ESTUARINE  DEPOSITS (6,443), WEATHERED COASTAL PLAIN FORMATIONS (4,227), ALLUVIUM (4,170), WEATHERED SHALE, MUDSTONE, AND SANDSTONE (4,070), LATE WISCONSINAN GLACIAL DELTA DEPOSITS (2,964), UPPER STREAM TERRACE DEPOSITS (1,974), CAPE MAY FORMATION, UNIT 2 (1,936).

_Surficial geology is loaded in full (18,004 polygons) from the NJDEP ArcGIS Hub CDN export, which bypasses the Imperva WAF that hard-blocks paginated reads of the live MapServer; bedrock + SSURGO-tabular come from the live REST/SDA endpoints. The ~1.7% of borings without a surficial unit fall outside mapped polygons (open water / coastal edge)._

## 4. GNN graph inputs (Phase 5)

- Nodes: **49,152** borings (features via the `node_features` view: lon/lat, elevation, slope, surficial/bedrock unit, SSURGO component/drainage/hydrologic group, coord-quality).
- Edges: **941,167** total → `delaunay`=147,431, `knn`=239,776, `same_geology`=553,960  (exported to `edges.parquet`).
- Distances computed in EPSG:102711 (feet) for metric correctness.

## 5. Spatial coverage & sampling bias  ⚠️ for downstream modeling

- Bounding box (lon/lat): (-75.492, 38.959, -73.914, 41.347).
- Occupied ~1 km cells (0.01°): **2,274**; ~10 km cells (0.1°): 187; densest 1 km cell holds **511** borings.
- Borings are **not** a spatially uniform sample: they cluster on transportation corridors (roads/bridges) because NJDOT drills for projects, mapped by the boring-plan polygons. Treat the node distribution as **preferentially sampled** — geology/SSURGO/DEM covariates provide the spatial prior that lets a GNN extrapolate into undrilled areas, and any evaluation split should account for this clustering (e.g. spatial CV).

## 6. Known gaps

- Stratigraphy is OCR-pending (no engine installed); `parse_logs.py` is a scaffold with a working regex extractor + worked examples, accuracy flagged TODO.
- Scanned-log PDF crawl and per-boring DEM are gated off by default (`--download-logs`, `--dem`).
- SSURGO spatial footprints NOT loaded (NJDEP WAF) — per-boring SSURGO class may be null.
- One boring lacks a scanned-log attachment.

## 7. Artifacts

- `data/soilbot.duckdb` — normalized store (borings, strata, covariates, edges, manifest).
- `data/borings.gpkg`, `data/boring_plans.gpkg`, `data/soil_labels.gpkg` — geometry (EPSG:4326).
- `data/edges.parquet` — graph edge list. `data/raw/` — verbatim GeoJSON pages.
- `schema_audit.md` — Phase-1 field audit. `logs/*.log` — structured run logs.

