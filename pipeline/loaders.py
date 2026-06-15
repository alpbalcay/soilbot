"""GeoJSON page -> DuckDB staging, and staging -> normalized tables.

Pages arrive as ArcGIS `f=geojson`, so their geometry is already WGS84 (EPSG:4326);
that becomes `geom_4326`. The normalized step recovers `geom_native` (EPSG:102711) via
the inverse ST_Transform and derives lon/lat + a coordinate-quality flag.
"""
from __future__ import annotations

import json

import duckdb

from .config import Config

# ---- per-layer staging insert specs --------------------------------------
# Each spec: staging table, ordered (column, geojson-property, converter).
_S = "str"
_I = "int"
_F = "float"

_LAYER_SPECS: dict[str, dict] = {
    "borings": {
        "table": "stg_borings",
        "cols": [
            ("objectid", "OBJECTID", _I), ("lid", "LID", _S), ("label", "LABEL", _S),
            ("filename", "FILENAME", _S), ("pid", "PID", _S), ("bcontr", "BCONTR", _S),
            ("srce", "SRCE", _I), ("localn", "LOCALN", _S), ("oversize", "OVERSIZE", _S),
            ("point_x", "POINT_X", _F), ("point_y", "POINT_Y", _F),
            ("createdate", "CREATEDATE", _S), ("changedate", "CHANGEDATE", _S),
            ("matchid", "MATCHID", _S),
        ],
    },
    "boring_plan": {
        "table": "stg_boring_plan",
        "cols": [
            ("objectid", "OBJECTID", _I), ("pid", "PID", _S), ("filename", "FILENAME", _S),
            ("route", "ROUTE", _S), ("sect", "SECT", _S), ("pcontr", "PCONTR", _S),
            ("pdate", "PDATE", _S), ("upc", "UPC", _S), ("matchid", "MATCHID", _S),
        ],
    },
    "soil_label": {
        "table": "stg_soil_label",
        "cols": [
            ("objectid", "OBJECTID", _I), ("label_type", "LABEL_TYPE", _S),
            ("primary_label", "PRIMARY_LABEL", _S), ("secondary_label", "SECONDARY_LABEL", _S),
            ("drainage", "DRAINAGE", _S), ("pdf_pg_num", "PDF_PG_NUM", _S),
            ("url", "URL", _S), ("weburl", "WEBURL", _S),
        ],
    },
}


def _conv(value, kind: str):
    if value is None or value == "":
        return None
    try:
        if kind == _I:
            return int(value)
        if kind == _F:
            return float(value)
        return str(value)
    except (ValueError, TypeError):
        return None


def insert_page(con: duckdb.DuckDBPyConnection, layer_key: str, offset: int,
                geojson_obj: dict) -> int:
    """Idempotently load one geojson page into the layer's staging table."""
    spec = _LAYER_SPECS[layer_key]
    table = spec["table"]
    cols = spec["cols"]
    features = geojson_obj.get("features", []) or []

    col_names = ["src_offset"] + [c[0] for c in cols] + ["geom_4326"]
    placeholders = ["?"] * (1 + len(cols)) + ["ST_GeomFromGeoJSON(?)"]
    sql = (f"INSERT INTO {table} ({', '.join(col_names)}) "
           f"VALUES ({', '.join(placeholders)})")

    rows = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry")
        row = [offset]
        for _col, prop, kind in cols:
            row.append(_conv(props.get(prop), kind))
        row.append(json.dumps(geom) if geom else None)
        rows.append(row)

    con.execute("BEGIN")
    try:
        con.execute(f"DELETE FROM {table} WHERE src_offset = ?", [offset])
        if rows:
            con.executemany(sql, rows)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(features)


# ---- staging -> normalized ------------------------------------------------
def build_normalized(con: duckdb.DuckDBPyConnection, config: Config, log) -> dict:
    """Rebuild normalized tables from staging. Upserts so DEM elevation / log_url survive."""
    srs = config["crs"]["native_srs"]
    bb = config["crs"]["nj_bbox"]
    tol = 50.0  # feet; round-trip reprojection tolerance for the xy-vs-geom audit

    # borings (points) -------------------------------------------------------
    con.execute(f"""
        INSERT INTO borings
            (boring_id, objectid, pid, label, filename, lon, lat, source_layer,
             coord_quality_flag, geom_native, geom_4326)
        WITH base AS (
            SELECT *, COALESCE(lid, 'OID-' || objectid) AS bid,
                   ST_Transform(geom_4326, 'EPSG:4326', '{srs}', always_xy := true) AS gnat
            FROM stg_borings
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY COALESCE(lid, 'OID-' || objectid) ORDER BY objectid DESC) = 1
        )
        SELECT bid, objectid, pid, label, filename,
               ST_X(geom_4326), ST_Y(geom_4326), 0,
               CASE
                 WHEN geom_4326 IS NULL THEN 'no_geom'
                 WHEN ST_X(geom_4326) NOT BETWEEN {bb['min_lon']} AND {bb['max_lon']}
                   OR ST_Y(geom_4326) NOT BETWEEN {bb['min_lat']} AND {bb['max_lat']} THEN 'out_of_nj_bbox'
                 WHEN point_x IS NOT NULL
                   AND (ABS(ST_X(gnat) - point_x) > {tol} OR ABS(ST_Y(gnat) - point_y) > {tol})
                   THEN 'xy_geom_mismatch'
                 ELSE 'ok'
               END,
               gnat, geom_4326
        FROM base
        ON CONFLICT (boring_id) DO UPDATE SET
            objectid=EXCLUDED.objectid, pid=EXCLUDED.pid, label=EXCLUDED.label,
            filename=EXCLUDED.filename, lon=EXCLUDED.lon, lat=EXCLUDED.lat,
            coord_quality_flag=EXCLUDED.coord_quality_flag,
            geom_native=EXCLUDED.geom_native, geom_4326=EXCLUDED.geom_4326
    """)

    # boring_plans (polygons) -----------------------------------------------
    con.execute(f"""
        INSERT INTO boring_plans
            (objectid, pid, route, sect, pcontr, pdate, upc, filename, geom_native, geom_4326)
        SELECT objectid, pid, route, sect, pcontr, pdate, upc, filename,
               ST_Transform(geom_4326, 'EPSG:4326', '{srs}', always_xy := true), geom_4326
        FROM stg_boring_plan
        QUALIFY ROW_NUMBER() OVER (PARTITION BY objectid ORDER BY src_offset DESC) = 1
        ON CONFLICT (objectid) DO UPDATE SET
            pid=EXCLUDED.pid, route=EXCLUDED.route, sect=EXCLUDED.sect, pcontr=EXCLUDED.pcontr,
            pdate=EXCLUDED.pdate, upc=EXCLUDED.upc, filename=EXCLUDED.filename,
            geom_native=EXCLUDED.geom_native, geom_4326=EXCLUDED.geom_4326
    """)

    # soil_labels (points; the only layer with a structured soil class) ------
    con.execute(f"""
        INSERT INTO soil_labels
            (objectid, label_type, primary_label, secondary_label, drainage, pdf_pg_num,
             url, weburl, lon, lat, geom_native, geom_4326)
        SELECT objectid, label_type, primary_label, secondary_label, drainage,
               TRY_CAST(pdf_pg_num AS INTEGER), url, weburl,
               ST_X(geom_4326), ST_Y(geom_4326),
               ST_Transform(geom_4326, 'EPSG:4326', '{srs}', always_xy := true), geom_4326
        FROM stg_soil_label
        QUALIFY ROW_NUMBER() OVER (PARTITION BY objectid ORDER BY src_offset DESC) = 1
        ON CONFLICT (objectid) DO UPDATE SET
            label_type=EXCLUDED.label_type, primary_label=EXCLUDED.primary_label,
            secondary_label=EXCLUDED.secondary_label, drainage=EXCLUDED.drainage,
            pdf_pg_num=EXCLUDED.pdf_pg_num, url=EXCLUDED.url, weburl=EXCLUDED.weburl,
            lon=EXCLUDED.lon, lat=EXCLUDED.lat,
            geom_native=EXCLUDED.geom_native, geom_4326=EXCLUDED.geom_4326
    """)

    counts = {
        "borings": con.execute("SELECT COUNT(*) FROM borings").fetchone()[0],
        "boring_plans": con.execute("SELECT COUNT(*) FROM boring_plans").fetchone()[0],
        "soil_labels": con.execute("SELECT COUNT(*) FROM soil_labels").fetchone()[0],
    }
    quality = dict(con.execute(
        "SELECT coord_quality_flag, COUNT(*) FROM borings GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall())
    log.info("normalized_built", **{f"n_{k}": v for k, v in counts.items()})
    log.info("coord_quality", **{str(k): v for k, v in quality.items()})
    return {"counts": counts, "coord_quality": quality}
