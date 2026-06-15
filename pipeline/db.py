"""DuckDB connection, spatial extension bootstrap, schema DDL, and the idempotency manifest.

Geometry convention: every spatial table stores BOTH `geom_4326` (WGS84 lon/lat, as
delivered by the ArcGIS `f=geojson` endpoint) and `geom_native` (EPSG:102711, NJ State
Plane US ft, recovered via ST_Transform). Point layers additionally keep the raw
POINT_X/POINT_Y attributes so the server's reprojection can be audited independently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from .config import Config
from .util import ensure_dir

# ---------------------------------------------------------------------------
# DDL — created once, idempotently (CREATE ... IF NOT EXISTS).
# ---------------------------------------------------------------------------
DDL: list[str] = [
    # ---- idempotency manifest ------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS manifest (
        kind        VARCHAR,            -- 'page' | 'attachment' | 'covariate' | 'enum'
        key         VARCHAR,            -- e.g. 'layer0:offset=4000', 'oid=123:aid=4'
        status      VARCHAR,            -- 'pending'|'in_progress'|'done'|'failed'|'skipped'
        rows_out    BIGINT,
        bytes       BIGINT,
        sha256      VARCHAR,
        http_status INTEGER,
        attempts    INTEGER DEFAULT 0,
        run_id      VARCHAR,
        updated_at  TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (kind, key)
    )""",

    # ---- staging (faithful to source attributes; geom is WGS84 from geojson) --
    """
    CREATE TABLE IF NOT EXISTS stg_borings (
        src_offset BIGINT, objectid BIGINT, lid VARCHAR, label VARCHAR, filename VARCHAR,
        pid VARCHAR, bcontr VARCHAR, srce INTEGER, localn VARCHAR, oversize VARCHAR,
        point_x DOUBLE, point_y DOUBLE, createdate VARCHAR, changedate VARCHAR, matchid VARCHAR,
        geom_4326 GEOMETRY, ingested_at TIMESTAMP DEFAULT current_timestamp
    )""",
    """
    CREATE TABLE IF NOT EXISTS stg_boring_plan (
        src_offset BIGINT, objectid BIGINT, pid VARCHAR, filename VARCHAR, route VARCHAR,
        sect VARCHAR, pcontr VARCHAR, pdate VARCHAR, upc VARCHAR, matchid VARCHAR,
        geom_4326 GEOMETRY, ingested_at TIMESTAMP DEFAULT current_timestamp
    )""",
    """
    CREATE TABLE IF NOT EXISTS stg_soil_label (
        src_offset BIGINT, objectid BIGINT, label_type VARCHAR, primary_label VARCHAR,
        secondary_label VARCHAR, drainage VARCHAR, pdf_pg_num VARCHAR, url VARCHAR, weburl VARCHAR,
        geom_4326 GEOMETRY, ingested_at TIMESTAMP DEFAULT current_timestamp
    )""",

    # ---- normalized / GNN-facing --------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS borings (
        boring_id VARCHAR PRIMARY KEY,    -- LID
        objectid BIGINT, pid VARCHAR, label VARCHAR, filename VARCHAR,
        lon DOUBLE, lat DOUBLE, elevation DOUBLE,
        source_layer INTEGER DEFAULT 0,
        log_url VARCHAR, attachment_id BIGINT,
        coord_quality_flag VARCHAR,       -- 'ok'|'out_of_nj_bbox'|'xy_geom_mismatch'|'no_geom'
        geom_native GEOMETRY, geom_4326 GEOMETRY
    )""",
    """
    CREATE TABLE IF NOT EXISTS boring_plans (
        objectid BIGINT PRIMARY KEY, pid VARCHAR, route VARCHAR, sect VARCHAR,
        pcontr VARCHAR, pdate VARCHAR, upc VARCHAR, filename VARCHAR,
        geom_native GEOMETRY, geom_4326 GEOMETRY
    )""",
    """
    CREATE TABLE IF NOT EXISTS soil_labels (
        objectid BIGINT PRIMARY KEY, label_type VARCHAR, primary_label VARCHAR,
        secondary_label VARCHAR, drainage VARCHAR, pdf_pg_num INTEGER, url VARCHAR, weburl VARCHAR,
        lon DOUBLE, lat DOUBLE, geom_native GEOMETRY, geom_4326 GEOMETRY
    )""",

    # ---- attachment catalog (built bulk via queryAttachments; drives downloader)
    """
    CREATE TABLE IF NOT EXISTS boring_attachments (
        objectid BIGINT PRIMARY KEY, parent_lid VARCHAR, attachment_id BIGINT,
        name VARCHAR, content_type VARCHAR, size BIGINT, download_url VARCHAR
    )""",

    # ---- strata: OCR-FED, STARTS EMPTY (no structured source exists) ---------
    "CREATE SEQUENCE IF NOT EXISTS seq_strata START 1",
    """
    CREATE TABLE IF NOT EXISTS strata (
        strata_id BIGINT DEFAULT nextval('seq_strata'),
        boring_id VARCHAR, interval_index INTEGER,
        top_depth DOUBLE, bottom_depth DOUBLE,
        uscs_class VARCHAR, spt_n INTEGER, sample_type VARCHAR,
        gw_depth DOUBLE, elevation DOUBLE,
        source VARCHAR,        -- 'pdfplumber' | 'ocr'
        ocr_status VARCHAR,    -- 'parsed' | 'pending' | 'low_confidence' | 'failed'
        confidence DOUBLE,
        PRIMARY KEY (boring_id, interval_index)
    )""",

    # ---- covariates: SSURGO tabular (SDA) -----------------------------------
    """
    CREATE TABLE IF NOT EXISTS ssurgo_mapunit (
        mukey VARCHAR PRIMARY KEY, musym VARCHAR, muname VARCHAR, mukind VARCHAR, areasymbol VARCHAR
    )""",
    """
    CREATE TABLE IF NOT EXISTS ssurgo_component (
        cokey VARCHAR PRIMARY KEY, mukey VARCHAR, compname VARCHAR, comppct_r INTEGER,
        majcompflag VARCHAR, drainagecl VARCHAR, taxorder VARCHAR, taxclname VARCHAR
    )""",
    """
    CREATE TABLE IF NOT EXISTS ssurgo_chorizon (
        chkey VARCHAR PRIMARY KEY, cokey VARCHAR, hzname VARCHAR,
        hzdept_r DOUBLE, hzdepb_r DOUBLE,
        sandtotal_r DOUBLE, silttotal_r DOUBLE, claytotal_r DOUBLE,
        ll_r DOUBLE, pi_r DOUBLE, ksat_r DOUBLE, awc_r DOUBLE
    )""",
    """
    CREATE TABLE IF NOT EXISTS ssurgo_muaggatt (
        mukey VARCHAR PRIMARY KEY, drclassdcd VARCHAR, hydgrpdcd VARCHAR,
        brockdepmin DOUBLE, wtdepannmin DOUBLE, aws025wta DOUBLE
    )""",
    # SSURGO spatial footprints (NJDEP Geology MapServer layer 11) for point-in-polygon mukey.
    """
    CREATE TABLE IF NOT EXISTS ssurgo_spatial (
        objectid BIGINT, mukey VARCHAR, musym VARCHAR, muname VARCHAR, geom_4326 GEOMETRY
    )""",

    # ---- covariates: NJ geology ---------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS geology_surficial (
        objectid BIGINT, geoname VARCHAR, geoabb VARCHAR, lithology VARCHAR,
        geoage VARCHAR, geom_4326 GEOMETRY
    )""",
    """
    CREATE TABLE IF NOT EXISTS geology_bedrock (
        objectid BIGINT, geoname VARCHAR, geoabb VARCHAR, lithology VARCHAR,
        geoage VARCHAR, geom_4326 GEOMETRY
    )""",

    # ---- covariates: DEM-derived terrain (filled only when --dem) ------------
    """
    CREATE TABLE IF NOT EXISTS dem_samples (
        boring_id VARCHAR PRIMARY KEY, elevation_m DOUBLE, slope_deg DOUBLE, source VARCHAR
    )""",

    # ---- per-boring covariate assignment (point-in-polygon results) ----------
    """
    CREATE TABLE IF NOT EXISTS boring_covariates (
        boring_id VARCHAR PRIMARY KEY,
        surficial_unit VARCHAR, surficial_lithology VARCHAR, surficial_age VARCHAR,
        bedrock_unit VARCHAR, bedrock_lithology VARCHAR,
        ssurgo_mukey VARCHAR, ssurgo_muname VARCHAR,
        ssurgo_component VARCHAR, ssurgo_drainagecl VARCHAR, ssurgo_hydgrp VARCHAR
    )""",

    # ---- graph edges ---------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS edges (
        src VARCHAR, dst VARCHAR, edge_type VARCHAR, weight DOUBLE,
        PRIMARY KEY (src, dst, edge_type)
    )""",
]

# node_features view: created last (depends on the tables above all existing).
NODE_FEATURES_VIEW = """
CREATE OR REPLACE VIEW node_features AS
SELECT
    b.boring_id,
    b.lon, b.lat,
    COALESCE(d.elevation_m, b.elevation) AS elevation,
    d.slope_deg,
    b.coord_quality_flag,
    bc.surficial_unit, bc.surficial_lithology, bc.surficial_age,
    bc.bedrock_unit,
    bc.ssurgo_mukey, bc.ssurgo_muname, bc.ssurgo_component,
    bc.ssurgo_drainagecl, bc.ssurgo_hydgrp,
    s.n_intervals
FROM borings b
LEFT JOIN boring_covariates bc ON bc.boring_id = b.boring_id
LEFT JOIN dem_samples d ON d.boring_id = b.boring_id
LEFT JOIN (
    SELECT boring_id, COUNT(*) AS n_intervals FROM strata GROUP BY boring_id
) s ON s.boring_id = b.boring_id
"""


def connect(config: Config, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB store with the spatial extension loaded (cached to extension_dir)."""
    ensure_dir(config.duckdb_path.parent)
    ext_dir = ensure_dir(config.extension_dir)
    con = duckdb.connect(str(config.duckdb_path), read_only=read_only)
    con.execute(f"SET extension_directory='{ext_dir}'")
    con.execute("SET enable_progress_bar=false")
    try:
        con.execute("INSTALL spatial")
    except duckdb.Error:
        # Already cached / offline — LOAD will succeed if the extension is present.
        pass
    try:
        con.execute("LOAD spatial")
    except duckdb.Error as exc:  # pragma: no cover - surfaced loudly to the operator
        raise RuntimeError(
            "DuckDB spatial extension failed to load. It is required for reprojection and "
            "GeoPackage export. If offline, pre-cache it once with network access, or install "
            "the fallback stack (requirements-geo.txt). Underlying error: " + str(exc)
        ) from exc
    return con


def bootstrap(con: duckdb.DuckDBPyConnection) -> None:
    """Create all tables/sequences/views if absent. Safe to call on every run."""
    for stmt in DDL:
        con.execute(stmt)
    con.execute(NODE_FEATURES_VIEW)


# ---------------------------------------------------------------------------
# Manifest helpers (the single idempotency mechanism, shared by all phases).
# ---------------------------------------------------------------------------
def manifest_status(con: duckdb.DuckDBPyConnection, kind: str, key: str) -> str | None:
    row = con.execute(
        "SELECT status FROM manifest WHERE kind = ? AND key = ?", [kind, key]
    ).fetchone()
    return row[0] if row else None


def manifest_is_done(con: duckdb.DuckDBPyConnection, kind: str, key: str) -> bool:
    return manifest_status(con, kind, key) == "done"


def manifest_mark(con: duckdb.DuckDBPyConnection, kind: str, key: str, status: str,
                  run_id: str | None = None, **fields: Any) -> None:
    """Upsert a manifest row. Extra columns: rows_out, bytes, sha256, http_status, attempts."""
    cols = ["kind", "key", "status", "run_id", "updated_at"]
    vals: list[Any] = [kind, key, status, run_id]
    placeholders = ["?", "?", "?", "?", "current_timestamp"]
    for col in ("rows_out", "bytes", "sha256", "http_status", "attempts"):
        if col in fields:
            cols.append(col)
            vals.append(fields[col])
            placeholders.append("?")
    update_cols = [c for c in cols if c not in ("kind", "key")]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    con.execute(
        f"INSERT INTO manifest ({', '.join(cols)}) VALUES ({', '.join(placeholders)}) "
        f"ON CONFLICT (kind, key) DO UPDATE SET {set_clause}",
        vals,
    )


def manifest_keys_with_status(con: duckdb.DuckDBPyConnection, kind: str,
                              status: str = "done") -> set[str]:
    rows = con.execute(
        "SELECT key FROM manifest WHERE kind = ? AND status = ?", [kind, status]
    ).fetchall()
    return {r[0] for r in rows}


def table_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
