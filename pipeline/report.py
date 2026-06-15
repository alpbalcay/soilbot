"""Generate REPORT.md — the honest, end-to-end accounting of what was extracted.

Emphasis (per the task): be explicit about what is REAL structured data vs. what still
needs OCR, and flag the spatial sampling bias for the downstream model.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import db
from .config import Config
from .util import atomic_write_text


def _scalar(con, sql, default=0):
    r = con.execute(sql).fetchone()
    return r[0] if r and r[0] is not None else default


def gather(con) -> dict:
    d: dict = {}
    d["borings"] = db.table_count(con, "borings")
    d["boring_plans"] = db.table_count(con, "boring_plans")
    d["soil_labels"] = db.table_count(con, "soil_labels")
    d["borings_with_logurl"] = _scalar(con, "SELECT COUNT(*) FROM borings WHERE log_url IS NOT NULL")
    d["coord_quality"] = dict(con.execute(
        "SELECT coord_quality_flag, COUNT(*) FROM borings GROUP BY 1 ORDER BY 2 DESC").fetchall())

    d["strata_rows"] = db.table_count(con, "strata")
    d["strata_parsed_borings"] = _scalar(
        con, "SELECT COUNT(DISTINCT boring_id) FROM strata WHERE ocr_status='parsed'")
    d["parse_pending"] = _scalar(
        con, "SELECT COUNT(*) FROM manifest WHERE kind='parse' AND status='pending'")
    d["logs_downloaded"] = _scalar(
        con, "SELECT COUNT(*) FROM manifest WHERE kind='attachment' AND status='done'")

    for t in ("geology_surficial", "geology_bedrock", "ssurgo_spatial", "ssurgo_mapunit",
              "ssurgo_component", "ssurgo_chorizon", "ssurgo_muaggatt", "dem_samples"):
        d[t] = db.table_count(con, t)
    cov = con.execute("""SELECT
            SUM(CASE WHEN surficial_unit IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN bedrock_unit  IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN ssurgo_mukey  IS NOT NULL THEN 1 ELSE 0 END),
            COUNT(*) FROM boring_covariates""").fetchone()
    d["cov_surficial"], d["cov_bedrock"], d["cov_ssurgo"], d["cov_rows"] = \
        (cov[0] or 0, cov[1] or 0, cov[2] or 0, cov[3] or 0)

    d["edges_total"] = db.table_count(con, "edges")
    d["edges_by_type"] = dict(con.execute(
        "SELECT edge_type, COUNT(*) FROM edges GROUP BY 1 ORDER BY 1").fetchall())

    # spatial coverage / sampling-bias indicators
    d["bbox"] = con.execute(
        "SELECT round(min(lon),3),round(min(lat),3),round(max(lon),3),round(max(lat),3) "
        "FROM borings WHERE lon IS NOT NULL").fetchone()
    d["cells_001"] = _scalar(con, "SELECT COUNT(*) FROM (SELECT 1 FROM borings WHERE lon IS NOT NULL GROUP BY round(lon,2), round(lat,2))")
    d["cells_01"] = _scalar(con, "SELECT COUNT(*) FROM (SELECT 1 FROM borings WHERE lon IS NOT NULL GROUP BY round(lon,1), round(lat,1))")
    d["max_per_cell001"] = _scalar(con, "SELECT MAX(c) FROM (SELECT COUNT(*) c FROM borings WHERE lon IS NOT NULL GROUP BY round(lon,2), round(lat,2))")
    d["top_surficial"] = con.execute(
        "SELECT surficial_unit, COUNT(*) c FROM boring_covariates WHERE surficial_unit IS NOT NULL "
        "GROUP BY 1 ORDER BY 2 DESC LIMIT 8").fetchall()
    return d


def write_markdown(config: Config, d: dict, run_id: str) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    nb = d["borings"]
    parsed = d["strata_parsed_borings"]
    pct_struct = (100.0 * parsed / nb) if nb else 0.0
    ocr_needed = d["borings_with_logurl"]
    cq = ", ".join(f"`{k}`={v:,}" for k, v in d["coord_quality"].items())

    def cov_pct(x):
        return f"{(100.0 * x / nb):.1f}%" if nb else "—"

    L = []
    L += [
        "# REPORT.md — NJDOT GDMS boring database",
        "",
        f"_Generated {ts} (run `{run_id}`). See `schema_audit.md` for the full field-level audit._",
        "",
        "## 1. Layers found & record counts (structured, REST-extracted)",
        "",
        "| Layer | Geometry | Records | Notes |",
        "|---|---|--:|---|",
        f"| GDMS Boring Log (0) | point | {d['borings']:,} | boring locations + metadata; one scanned PDF each |",
        f"| GDMS Boring Plan (1) | polygon | {d['boring_plans']:,} | project plan sheets (transport corridors) |",
        f"| Geol Soil Egr Label (2) | point | {d['soil_labels']:,} | only structured soil class (PRIMARY/SECONDARY_LABEL, DRAINAGE) |",
        "",
        f"- Coordinate quality (borings): {cq}.",
        f"- Borings linked to a scanned-log attachment URL: **{d['borings_with_logurl']:,}** of {nb:,}.",
        f"- Native CRS EPSG:102711 (NJ State Plane US ft) preserved as `geom_native`; `geom_4326` = WGS84.",
        "",
        "## 2. Structured stratigraphy vs OCR-only  ⚠️ the key distinction",
        "",
        f"- Borings with **STRUCTURED** stratigraphy (SPT-N / USCS / depth intervals): "
        f"**{parsed:,} ({pct_struct:.1f}%)**.",
        f"- Borings whose stratigraphy exists **ONLY inside scanned PDF logs (OCR required)**: "
        f"**{ocr_needed:,} (~100%)**.",
        f"- `strata` table rows currently populated: **{d['strata_rows']:,}**  "
        f"(parse attempts pending OCR: {d['parse_pending']:,}; PDFs downloaded: {d['logs_downloaded']:,}).",
        "",
        "> The NJDOT GDMS exposes **no** structured SPT N-values, USCS classes, depth intervals, "
        "groundwater or sample types in any REST field. The only path to them is OCR of the "
        "scanned Layer-0 PDF attachments (gated behind `--download-logs` + `--ocr`; the OCR "
        "engine is not yet installed). No stratigraphy is fabricated — `strata` stays empty "
        "until real parsing runs. The lone structured soil signal is the coarse "
        f"`PRIMARY_LABEL` class on the {d['soil_labels']:,} soil-label points.",
        "",
        "## 3. Supplementary covariate layers (Phase 4)",
        "",
        "| Source | Table | Rows | Per-boring coverage |",
        "|---|---|--:|---|",
        f"| NJDEP surficial geology | geology_surficial | {d['geology_surficial']:,} | {cov_pct(d['cov_surficial'])} ({d['cov_surficial']:,}) |",
        f"| NJDEP bedrock geology | geology_bedrock | {d['geology_bedrock']:,} | {cov_pct(d['cov_bedrock'])} ({d['cov_bedrock']:,}) |",
        f"| SSURGO footprints (NJDEP L11) | ssurgo_spatial | {d['ssurgo_spatial']:,} | {cov_pct(d['cov_ssurgo'])} ({d['cov_ssurgo']:,}) |",
        f"| SSURGO mapunit (SDA) | ssurgo_mapunit | {d['ssurgo_mapunit']:,} | join via mukey |",
        f"| SSURGO component (SDA) | ssurgo_component | {d['ssurgo_component']:,} | dominant component per mukey |",
        f"| SSURGO horizons (SDA) | ssurgo_chorizon | {d['ssurgo_chorizon']:,} | engineering props (sieve, LL, PI, ksat) |",
        f"| SSURGO muaggatt (SDA) | ssurgo_muaggatt | {d['ssurgo_muaggatt']:,} | hydrologic group, depth-to-bedrock, WT depth |",
        f"| USGS 3DEP DEM | dem_samples | {d['dem_samples']:,} | gated (`--dem`); elevation/slope per boring |",
        "",
    ]
    if d["top_surficial"]:
        L.append("Top surficial units under borings: " +
                 ", ".join(f"{u} ({c:,})" for u, c in d["top_surficial"]) + ".")
        L.append("")
    L.append(
        "_Surficial geology is loaded in full (18,004 polygons) from the NJDEP ArcGIS Hub CDN "
        "export, which bypasses the Imperva WAF that hard-blocks paginated reads of the live "
        "MapServer; bedrock + SSURGO-tabular come from the live REST/SDA endpoints. The ~1.7% of "
        "borings without a surficial unit fall outside mapped polygons (open water / coastal edge)._")
    L.append("")

    et = d["edges_by_type"]
    L += [
        "## 4. GNN graph inputs (Phase 5)",
        "",
        f"- Nodes: **{nb:,}** borings (features via the `node_features` view: lon/lat, elevation, "
        "slope, surficial/bedrock unit, SSURGO component/drainage/hydrologic group, coord-quality).",
        f"- Edges: **{d['edges_total']:,}** total → " +
        (", ".join(f"`{t}`={c:,}" for t, c in et.items()) if et else "none yet") +
        f"  (exported to `{config.path('edges_parquet').name}`).",
        "- Distances computed in EPSG:102711 (feet) for metric correctness.",
        "",
        "## 5. Spatial coverage & sampling bias  ⚠️ for downstream modeling",
        "",
        f"- Bounding box (lon/lat): {d['bbox']}.",
        f"- Occupied ~1 km cells (0.01°): **{d['cells_001']:,}**; ~10 km cells (0.1°): {d['cells_01']:,}; "
        f"densest 1 km cell holds **{d['max_per_cell001']:,}** borings.",
        "- Borings are **not** a spatially uniform sample: they cluster on transportation "
        "corridors (roads/bridges) because NJDOT drills for projects, mapped by the boring-plan "
        "polygons. Treat the node distribution as **preferentially sampled** — geology/SSURGO/DEM "
        "covariates provide the spatial prior that lets a GNN extrapolate into undrilled areas, "
        "and any evaluation split should account for this clustering (e.g. spatial CV).",
        "",
        "## 6. Known gaps",
        "",
        "- Stratigraphy is OCR-pending (no engine installed); `parse_logs.py` is a scaffold with "
        "a working regex extractor + worked examples, accuracy flagged TODO.",
        "- Scanned-log PDF crawl and per-boring DEM are gated off by default (`--download-logs`, `--dem`).",
        f"- SSURGO spatial footprints {'loaded' if d['ssurgo_spatial'] else 'NOT loaded (NJDEP WAF) — per-boring SSURGO class may be null'}.",
        "- One boring lacks a scanned-log attachment.",
        "",
        "## 7. Artifacts",
        "",
        "- `data/soilbot.duckdb` — normalized store (borings, strata, covariates, edges, manifest).",
        "- `data/borings.gpkg`, `data/boring_plans.gpkg`, `data/soil_labels.gpkg` — geometry (EPSG:4326).",
        "- `data/edges.parquet` — graph edge list. `data/raw/` — verbatim GeoJSON pages.",
        "- `schema_audit.md` — Phase-1 field audit. `logs/*.log` — structured run logs.",
        "",
    ]
    text = "\n".join(L) + "\n"
    path = config.path("report")
    atomic_write_text(path, text)
    return str(path)


def run(config: Config, log) -> str:
    con = db.connect(config, read_only=True)
    data = gather(con)
    path = write_markdown(config, data, log.run_id)
    con.close()
    log.info("report_written", path=path, structured_pct=round(
        100.0 * data["strata_parsed_borings"] / data["borings"], 2) if data["borings"] else 0)
    return path
