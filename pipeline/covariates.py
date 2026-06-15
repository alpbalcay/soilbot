"""Phase 4 — supplementary covariate layers, as separate joinable tables.

  * NJ geology (NJDEP MapServer): surficial (layer 25) + bedrock (layer 14) polygons.
  * SSURGO spatial footprints (NJDEP layer 11): MUKEY polygons for point-in-polygon.
  * SSURGO tabular (NRCS Soil Data Access, POST SQL): mapunit / component / chorizon / muaggatt.
  * DEM (USGS 3DEP, GATED --dem): per-boring elevation + slope via /identify.

Finally assigns one covariate row per boring via spatial join (geology unit + dominant SSURGO
component), feeding the node_features view.
"""
from __future__ import annotations

import json
import time

import requests

from . import db
from .arcgis import ArcGISClient
from .config import Config
from .util import RateLimiter, atomic_write_bytes, backoff_delay, ensure_dir

# --- geology / ssurgo-spatial field maps: (column, geojson-property, kind) ---
_GEOL_COLS = [("objectid", "OBJECTID", "int"), ("geoname", "GEONAME", "str"),
              ("geoabb", "GEOABB", "str"), ("lithology", "LITHOLOGY", "str"),
              ("geoage", "GEOAGE", "str")]
_SSURGO_SPATIAL_COLS = [("objectid", "OBJECTID", "int"), ("mukey", "MUKEY", "str"),
                        ("musym", "MUSYM", "str"), ("muname", "MUNAME", "str")]


def _conv(v, kind):
    if v is None or v == "":
        return None
    try:
        return int(v) if kind == "int" else (float(v) if kind == "float" else str(v))
    except (ValueError, TypeError):
        return None


def _insert_features(con, table, colmap, features) -> int:
    cols = [c[0] for c in colmap] + ["geom_4326"]
    ph = ["?"] * len(colmap) + ["ST_GeomFromGeoJSON(?)"]
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(ph)})"
    rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        # NJDEP layers are inconsistent: surficial returns UPPERCASE property names,
        # bedrock returns lowercase. Look up case-insensitively.
        lower = {str(k).lower(): v for k, v in props.items()}
        geom = f.get("geometry")
        row = [_conv(lower.get(prop.lower()), kind) for (_c, prop, kind) in colmap]
        row.append(json.dumps(geom) if geom else None)
        rows.append(row)
    if rows:
        con.executemany(sql, rows)
    return len(rows)


def _fetch_polygon_layer(con, client, log, layer_url, table, colmap, label,
                         page_size=2000, burst_size=3, rest_s=90.0,
                         max_requests=None) -> int:
    """Resumable, WAF-tolerant polygon fetch.

    Pages are checkpointed per offset in the manifest (kind='covariate'), so progress
    accumulates across runs / WAF trips. To stay under the Imperva Incapsula threshold
    (~4-5 large requests), we fetch in bursts of `burst_size` then go quiet for `rest_s`
    seconds (the counter resets during the quiet period).
    """
    count = client.feature_count(layer_url)
    maxrc = min(int(page_size), 2000)
    reqs = 0
    offset = 0
    while offset < count:
        key = f"{table}:offset={offset}"
        if db.manifest_is_done(con, "covariate", key):
            offset += maxrc
            continue
        if reqs and reqs % burst_size == 0:
            log.info("waf_rest", label=label, loaded=db.table_count(con, table), rest_s=rest_s)
            time.sleep(rest_s)
            client._warm()  # refresh Incapsula cookies after the quiet period
        page = client.query_page(layer_url, offset, maxrc, fmt="geojson")
        feats = page.parsed.get("features", []) or []
        oids = [f.get("properties", {}).get("OBJECTID") for f in feats]
        oids = [int(o) for o in oids if o is not None]
        con.execute("BEGIN")
        try:
            if oids:  # idempotent: clear this page's oids before re-inserting
                con.execute(f"DELETE FROM {table} WHERE objectid IN ({','.join(map(str, oids))})")
            n = _insert_features(con, table, colmap, feats)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        db.manifest_mark(con, "covariate", key, "done", run_id=log.run_id, rows_out=n)
        reqs += 1
        offset += maxrc
        if max_requests and reqs >= max_requests:
            log.info("covariate_fetch_capped", label=label, requests=reqs,
                     loaded=db.table_count(con, table))
            break
    total = db.table_count(con, table)
    log.info("covariate_layer_loaded", label=label, table=table,
             expected=count, loaded=total, new_requests=reqs)
    return total


def fetch_surficial_bulk(con, config: Config, log) -> int | None:
    """Load the full surficial layer from the WAF-free ArcGIS Hub CDN export.

    The live MapServer is behind an Imperva WAF that hard-blocks paginated bulk reads of the
    18,004-polygon surficial layer (returns HTML challenges on every page). ArcGIS Hub serves a
    pre-generated, CDN-cached GeoJSON of the SAME dataset that is NOT proxied through the WAF.
    The file is downloaded atomically (cached on disk; skipped on resume if already complete) and
    streamed into geology_surficial via DuckDB's ST_Read — the export is already EPSG:4326.

    Returns the loaded row count, or None if no bulk URL is configured / the fetch fails (caller
    then falls back to paginated REST).
    """
    g = config["covariates"]["geology"]
    url = g.get("surficial_bulk_url")
    if not url:
        return None
    expected = int(g.get("surficial_bulk_expected", 0))
    dest = ensure_dir(config.path("covariates_dir")) / "surficial_hub.geojson"

    # Reuse a previously-downloaded file if it parses and has the expected feature count.
    cached_ok = False
    if dest.exists():
        try:
            n = con.execute(f"SELECT count(*) FROM ST_Read('{dest}')").fetchone()[0]
            cached_ok = (not expected) or (n == expected)
            log.info("surficial_bulk_cached", path=str(dest), features=n, ok=cached_ok)
        except Exception:  # noqa: BLE001 - corrupt/partial cache -> re-download
            cached_ok = False

    if not cached_ok:
        r = config.rate("arcgis")
        try:
            resp = requests.get(url, timeout=float(r.get("timeout_s", 600)) if r else 600,
                                headers={"User-Agent": config.user_agent})
            if resp.status_code != 200 or not resp.content:
                log.warning("surficial_bulk_http", status=resp.status_code, bytes=len(resp.content))
                return None
            atomic_write_bytes(dest, resp.content)
        except requests.RequestException as exc:
            log.warning("surficial_bulk_download_failed", error=str(exc)[:140])
            return None
        try:
            n = con.execute(f"SELECT count(*) FROM ST_Read('{dest}')").fetchone()[0]
        except Exception as exc:  # noqa: BLE001 - downloaded bytes weren't valid GeoJSON
            log.warning("surficial_bulk_unreadable", error=str(exc)[:140])
            return None
        log.info("surficial_bulk_downloaded", path=str(dest), bytes=len(resp.content), features=n)
        if expected and n != expected:
            log.warning("surficial_bulk_count_mismatch", expected=expected, got=n)

    # Full refresh from the authoritative export. ST_Read field names match _GEOL_COLS sources;
    # the geometry column is already EPSG:4326 (CRS84 lon/lat), so it drops straight into geom_4326.
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM geology_surficial")
        con.execute(f"""
            INSERT INTO geology_surficial (objectid, geoname, geoabb, lithology, geoage, geom_4326)
            SELECT CAST(OBJECTID AS INTEGER), GEONAME, GEOABB, LITHOLOGY, GEOAGE, geom
            FROM ST_Read('{dest}')
        """)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    total = db.table_count(con, "geology_surficial")
    # Reconcile the manifest: the bulk load supersedes any partial paginated offsets.
    con.execute("DELETE FROM manifest WHERE kind='covariate' AND key LIKE 'geology_surficial:offset=%'")
    db.manifest_mark(con, "covariate", "geology_surficial:bulk", "done",
                     run_id=log.run_id, rows_out=total)
    log.info("surficial_bulk_loaded", table="geology_surficial", loaded=total, source="arcgis_hub_cdn")
    return total


def fetch_geology(con, client, config: Config, log) -> dict:
    g = config["covariates"]["geology"]
    base = g["mapserver"].rstrip("/")
    kw = dict(page_size=int(g.get("page_size", 2000)),
              burst_size=int(g.get("burst_size", 3)), rest_s=float(g.get("rest_s", 90)))
    out = {}

    # Surficial: prefer the WAF-free Hub bulk export; fall back to paginated REST only if it fails.
    bulk = None
    try:
        bulk = fetch_surficial_bulk(con, config, log)
    except Exception as exc:  # noqa: BLE001 - never let the bulk path abort the rest
        log.warning("surficial_bulk_failed", error=str(exc)[:140])
    if bulk is not None:
        out["surficial"] = bulk
    else:
        log.info("surficial_bulk_unavailable", note="falling back to paginated REST (WAF-throttled)")
        try:
            out["surficial"] = _fetch_polygon_layer(
                con, client, log, f"{base}/{g['surficial_layer']}",
                "geology_surficial", _GEOL_COLS, "surficial", **kw)
        except Exception as exc:  # noqa: BLE001
            log.warning("geology_fetch_failed", label="surficial", error=str(exc)[:140])
            out["surficial"] = db.table_count(con, "geology_surficial")

    # Bedrock stays on paginated REST (only 3,563 polys; already loads fine under the WAF pacing).
    try:
        out["bedrock"] = _fetch_polygon_layer(
            con, client, log, f"{base}/{g['bedrock_layer']}",
            "geology_bedrock", _GEOL_COLS, "bedrock", **kw)
    except Exception as exc:  # noqa: BLE001
        log.warning("geology_fetch_failed", label="bedrock", error=str(exc)[:140])
        out["bedrock"] = db.table_count(con, "geology_bedrock")
    return out


def fetch_ssurgo_spatial(con, client, config: Config, log) -> int:
    g = config["covariates"]["geology"]
    if not g.get("fetch_ssurgo_spatial", False):
        log.info("ssurgo_spatial_gated",
                 note="fetch_ssurgo_spatial=false; per-boring SSURGO class stays null (SDA tabular still loaded)")
        return 0
    base = g["mapserver"].rstrip("/")
    layer = g.get("ssurgo_spatial_layer", 11)
    kw = dict(page_size=int(g.get("page_size", 2000)),
              burst_size=int(g.get("burst_size", 3)), rest_s=float(g.get("rest_s", 90)))
    try:
        return _fetch_polygon_layer(con, client, log, f"{base}/{layer}",
                                    "ssurgo_spatial", _SSURGO_SPATIAL_COLS, "ssurgo_spatial", **kw)
    except Exception as exc:  # noqa: BLE001 - non-fatal; ssurgo class join just stays null
        log.warning("ssurgo_spatial_failed", error=str(exc)[:140])
        return db.table_count(con, "ssurgo_spatial")


# --- SSURGO tabular via Soil Data Access (POST SQL) -------------------------
_NJ = "l.areasymbol LIKE 'NJ%'"
_SDA_QUERIES = {
    "ssurgo_mapunit": (
        ["mukey", "musym", "muname", "mukind", "areasymbol"],
        f"""SELECT m.mukey, m.musym, m.muname, m.mukind, l.areasymbol
            FROM mapunit m INNER JOIN legend l ON m.lkey = l.lkey WHERE {_NJ}"""),
    "ssurgo_component": (
        ["cokey", "mukey", "compname", "comppct_r", "majcompflag", "drainagecl", "taxorder", "taxclname"],
        f"""SELECT c.cokey, c.mukey, c.compname, c.comppct_r, c.majcompflag, c.drainagecl,
                   c.taxorder, c.taxclname
            FROM component c INNER JOIN mapunit m ON c.mukey = m.mukey
            INNER JOIN legend l ON m.lkey = l.lkey WHERE {_NJ}"""),
    "ssurgo_chorizon": (
        ["chkey", "cokey", "hzname", "hzdept_r", "hzdepb_r", "sandtotal_r", "silttotal_r",
         "claytotal_r", "ll_r", "pi_r", "ksat_r", "awc_r"],
        f"""SELECT ch.chkey, ch.cokey, ch.hzname, ch.hzdept_r, ch.hzdepb_r, ch.sandtotal_r,
                   ch.silttotal_r, ch.claytotal_r, ch.ll_r, ch.pi_r, ch.ksat_r, ch.awc_r
            FROM chorizon ch INNER JOIN component c ON ch.cokey = c.cokey
            INNER JOIN mapunit m ON c.mukey = m.mukey
            INNER JOIN legend l ON m.lkey = l.lkey WHERE {_NJ}"""),
    "ssurgo_muaggatt": (
        ["mukey", "drclassdcd", "hydgrpdcd", "brockdepmin", "wtdepannmin", "aws025wta"],
        f"""SELECT mu.mukey, mu.drclassdcd, mu.hydgrpdcd, mu.brockdepmin, mu.wtdepannmin, mu.aws025wta
            FROM muaggatt mu INNER JOIN mapunit m ON mu.mukey = m.mukey
            INNER JOIN legend l ON m.lkey = l.lkey WHERE {_NJ}"""),
}


def _sda_query(config: Config, log, sql: str) -> list[list]:
    url = config["covariates"]["ssurgo"]["sda_url"]
    r = config.rate("sda")
    limiter = RateLimiter(float(r.get("rps", 2)))
    retries = int(r.get("max_retries", 5))
    timeout = float(r.get("timeout_s", 180))
    for attempt in range(retries + 1):
        limiter.acquire()
        try:
            resp = requests.post(url, data={"query": sql, "format": "JSON+COLUMNNAME"},
                                 timeout=timeout, headers={"User-Agent": config.user_agent})
        except requests.RequestException:
            if attempt >= retries:
                return []
            time.sleep(backoff_delay(attempt, float(r.get("backoff_base_s", 2)), float(r.get("backoff_max_s", 60))))
            continue
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt >= retries:
                return []
            time.sleep(backoff_delay(attempt, float(r.get("backoff_base_s", 2)), float(r.get("backoff_max_s", 60))))
            continue
        if resp.status_code >= 400 or not resp.text.strip():
            return []
        table = resp.json().get("Table")
        if not table:
            return []
        return table[1:]  # row 0 is the column-name header (format JSON+COLUMNNAME)
    return []


def fetch_ssurgo_tabular(con, config: Config, log) -> dict:
    counts = {}
    for table, (cols, sql) in _SDA_QUERIES.items():
        rows = _sda_query(config, log, sql)
        con.execute(f"DELETE FROM {table}")
        if rows:
            ph = ", ".join(["?"] * len(cols))
            con.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph})", rows)
        counts[table] = len(rows)
        log.info("ssurgo_tabular_loaded", table=table, rows=len(rows))
    return counts


# --- DEM (gated) ------------------------------------------------------------
def fetch_dem(con, config: Config, log, limit: int | None = None) -> dict:
    """GATED: per-boring elevation via 3DEP /identify (~49k calls). Resumable via manifest."""
    img = config["covariates"]["dem"]["imageserver"].rstrip("/")
    r = config.rate("arcgis")
    limiter = RateLimiter(float(r.get("rps", 5)))
    ua = config.user_agent
    work = con.execute(
        "SELECT boring_id, lon, lat FROM borings WHERE lon IS NOT NULL ORDER BY boring_id"
    ).fetchall()
    done = db.manifest_keys_with_status(con, "dem", "done")
    todo = [w for w in work if f"dem:{w[0]}" not in done]
    if limit:
        todo = todo[: int(limit)]
    log.info("dem_start", pending=len(todo))
    ok = 0
    for bid, lon, lat in todo:
        limiter.acquire()
        try:
            resp = requests.get(img + "/identify", params={
                "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint",
                "returnGeometry": "false", "f": "json"},
                timeout=float(r.get("timeout_s", 60)), headers={"User-Agent": ua})
            val = resp.json().get("value")
            elev = float(val) if val not in (None, "NoData") else None
        except Exception:  # noqa: BLE001
            elev = None
        con.execute(
            """INSERT INTO dem_samples (boring_id, elevation_m, slope_deg, source)
               VALUES (?, ?, NULL, '3DEP') ON CONFLICT (boring_id) DO UPDATE SET
               elevation_m = EXCLUDED.elevation_m""", [bid, elev])
        db.manifest_mark(con, "dem", f"dem:{bid}", "done", run_id=log.run_id)
        ok += 1
        if ok % 1000 == 0:
            log.info("dem_progress", done=ok)
    log.info("dem_done", sampled=ok)
    return {"sampled": ok}


# --- per-boring covariate assignment (spatial join) -------------------------
def _try_rtree(con, table):
    try:
        con.execute(f"DROP INDEX IF EXISTS rt_{table}")
        con.execute(f"CREATE INDEX rt_{table} ON {table} USING RTREE (geom_4326)")
    except Exception:  # noqa: BLE001 - index is an optimization, not required for correctness
        pass


def assign_covariates(con, config: Config, log) -> dict:
    for t in ("geology_surficial", "geology_bedrock", "ssurgo_spatial"):
        if db.table_count(con, t) > 0:
            _try_rtree(con, t)
    con.execute("DELETE FROM boring_covariates")
    con.execute("""
        INSERT INTO boring_covariates
            (boring_id, surficial_unit, surficial_lithology, surficial_age,
             bedrock_unit, bedrock_lithology, ssurgo_mukey, ssurgo_muname,
             ssurgo_component, ssurgo_drainagecl, ssurgo_hydgrp)
        WITH dom AS (   -- dominant SSURGO component per mapunit (highest comppct_r)
            SELECT mukey, compname, drainagecl FROM ssurgo_component
            QUALIFY ROW_NUMBER() OVER (PARTITION BY mukey
                    ORDER BY comppct_r DESC NULLS LAST, cokey) = 1
        )
        SELECT b.boring_id,
               gs.geoname, gs.lithology, gs.geoage,
               gb.geoname, gb.lithology,
               sp.mukey, sp.muname, dom.compname, dom.drainagecl, agg.hydgrpdcd
        FROM borings b
        LEFT JOIN geology_surficial gs ON ST_Within(b.geom_4326, gs.geom_4326)
        LEFT JOIN geology_bedrock  gb ON ST_Within(b.geom_4326, gb.geom_4326)
        LEFT JOIN ssurgo_spatial   sp ON ST_Within(b.geom_4326, sp.geom_4326)
        LEFT JOIN dom ON dom.mukey = sp.mukey
        LEFT JOIN ssurgo_muaggatt agg ON agg.mukey = sp.mukey
        QUALIFY ROW_NUMBER() OVER (PARTITION BY b.boring_id ORDER BY sp.mukey NULLS LAST) = 1
    """)
    n = db.table_count(con, "boring_covariates")
    cov = con.execute("""SELECT
            SUM(CASE WHEN surficial_unit IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN bedrock_unit  IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN ssurgo_mukey  IS NOT NULL THEN 1 ELSE 0 END)
        FROM boring_covariates""").fetchone()
    stats = {"rows": n, "with_surficial": cov[0], "with_bedrock": cov[1], "with_ssurgo": cov[2]}
    log.info("covariates_assigned", **stats)
    return stats


def run(config: Config, log, dem: bool = False, dem_limit: int | None = None) -> dict:
    con = db.connect(config)
    db.bootstrap(con)
    # NJDEP MapServer sits behind an Imperva Incapsula WAF -> browser headers, cookie
    # warm-up, gentle rate, and cooldown-retry (njdep rate group).
    client = ArcGISClient(config, log, rate_group="njdep", browser_headers=True)
    client.set_warmup(config["covariates"]["geology"]["mapserver"].rstrip("/"))
    result = {}
    result["geology"] = fetch_geology(con, client, config, log)
    result["ssurgo_spatial"] = fetch_ssurgo_spatial(con, client, config, log)
    result["ssurgo_tabular"] = fetch_ssurgo_tabular(con, config, log)
    if dem:
        result["dem"] = fetch_dem(con, config, log, limit=dem_limit)
    else:
        log.info("dem_gated", note="per-boring DEM skipped; enable with --dem")
    result["assignment"] = assign_covariates(con, config, log)
    con.close()
    return result
