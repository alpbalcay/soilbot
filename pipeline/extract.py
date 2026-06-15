"""Phase 2 — bulk extraction of the three FeatureServer layers.

Idempotent + resumable: each page is checkpointed in the manifest and a `.done` sentinel;
re-runs skip completed offsets. After all pages load, normalized tables are (re)built, the
attachment catalog populates borings.log_url, and a multi-layer GeoPackage is exported.
"""
from __future__ import annotations

from pathlib import Path

from . import attachments, db, loaders
from .arcgis import ArcGISClient
from .config import Config
from .util import atomic_write_bytes, ensure_dir, page_filename

# boring_plan (small) first to validate quickly, then soil_label, then borings (largest).
LAYER_ORDER = ("boring_plan", "soil_label", "borings")

# GeoPackage layers: (layer_name, output_filename, SELECT with one geom column aliased `geom`).
# NOTE: each layer goes to its OWN .gpkg file. DuckDB-spatial's GDAL GPKG driver segfaults
# when a 2nd COPY appends a layer to an existing file, so we never append. The borings file
# (config paths.gpkg) is the primary deliverable; all geometry also lives in the DuckDB store.
_GPKG_LAYERS = [
    ("borings", "borings.gpkg",
     """SELECT boring_id, objectid, pid, label, filename, lon, lat, elevation,
               source_layer, log_url, coord_quality_flag, geom_4326 AS geom
        FROM borings WHERE geom_4326 IS NOT NULL"""),
    ("boring_plans", "boring_plans.gpkg",
     """SELECT objectid, pid, route, sect, pcontr, pdate, upc, filename,
               geom_4326 AS geom FROM boring_plans WHERE geom_4326 IS NOT NULL"""),
    ("soil_labels", "soil_labels.gpkg",
     """SELECT objectid, label_type, primary_label, secondary_label, drainage,
               pdf_pg_num, url, weburl, lon, lat, geom_4326 AS geom
        FROM soil_labels WHERE geom_4326 IS NOT NULL"""),
]


def _extract_layer(con, client: ArcGISClient, config: Config, log, layer_key: str) -> dict:
    idx = config.layer(layer_key)["index"]
    url = config.layer_url(layer_key)
    page_size = int(config["paging"]["page_size"])
    where = config["paging"]["where"]
    out_fields = config["paging"]["out_fields"]
    fmt = config["paging"]["format"]

    count = client.feature_count(url, where)
    raw_dir = ensure_dir(config.path("raw_dir") / f"layer{idx}")
    log.info("layer_start", key=layer_key, layer=idx, count=count, page_size=page_size)

    pages = rows = skipped = 0
    offset = 0
    while offset < count:
        key = f"layer{idx}:offset={offset}"
        raw_path = raw_dir / page_filename(offset)
        sentinel = Path(str(raw_path) + ".done")
        if db.manifest_is_done(con, "page", key) or sentinel.exists():
            skipped += 1
            offset += page_size
            continue
        page = client.query_page(url, offset, page_size, where, out_fields, fmt)
        atomic_write_bytes(raw_path, page.raw_bytes)
        n = loaders.insert_page(con, layer_key, offset, page.parsed)
        db.manifest_mark(con, "page", key, "done", run_id=log.run_id,
                         rows_out=n, bytes=len(page.raw_bytes))
        sentinel.touch()
        pages += 1
        rows += n
        if pages % 5 == 0 or n < page_size:
            log.info("page_done", key=layer_key, offset=offset, rows=n)
        if n == 0:
            break
        offset += page_size

    log.info("layer_done", key=layer_key, layer=idx, count=count,
             pages_fetched=pages, rows_loaded=rows, pages_skipped=skipped)
    return {"count": count, "pages": pages, "rows": rows, "skipped": skipped}


def export_geopackage(con, config: Config, log) -> list[str]:
    out_dir = config.path("gpkg").parent
    ensure_dir(out_dir)
    written = []
    for name, fname, select in _GPKG_LAYERS:
        path = out_dir / fname
        if path.exists():
            path.unlink()  # rebuild clean (one layer per file; never append)
        con.execute(
            f"COPY ({select}) TO '{path}' "
            f"WITH (FORMAT GDAL, DRIVER 'GPKG', LAYER_NAME '{name}', SRS 'EPSG:4326')"
        )
        log.info("gpkg_layer_written", layer=name, path=str(path))
        written.append(str(path))
    log.info("gpkg_done", files=len(written))
    return written


def run(config: Config, log, export_gpkg: bool = True) -> dict:
    con = db.connect(config)
    db.bootstrap(con)
    client = ArcGISClient(config, log)

    stats = {}
    for layer_key in LAYER_ORDER:
        stats[layer_key] = _extract_layer(con, client, config, log, layer_key)

    norm = loaders.build_normalized(con, config, log)

    try:
        att = attachments.enumerate_bulk(con, client, config, log)
    except Exception:  # noqa: BLE001 - non-fatal; log_url backfill can happen later
        log.exception("attachment_enum_failed")
        att = {"error": True}

    if export_gpkg:
        try:
            export_geopackage(con, config, log)
        except Exception:  # noqa: BLE001
            log.exception("gpkg_export_failed")

    con.close()
    return {"layers": stats, "normalized": norm, "attachments": att}
