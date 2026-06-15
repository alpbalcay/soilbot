"""Isolate the GeoPackage COPY crash: single-layer, no-SRS, and GeoJSON-driver variants."""
import os
import sys

import duckdb

con = duckdb.connect()
con.execute("SET extension_directory='/var/home/alp/soilbot/data/.duckdb_extensions'")
con.execute("LOAD spatial")
con.execute("CREATE TABLE pts(id INT, geom GEOMETRY)")
con.execute("INSERT INTO pts VALUES (1, ST_Point(-74.5,40.1)),(2, ST_Point(-74.6,40.2))")


def attempt(label, sql, path):
    if path and os.path.exists(path):
        os.remove(path)
    print(f"trying: {label} ...", flush=True)
    try:
        con.execute(sql)
        print(f"  OK: {label}" + (f"  size={os.path.getsize(path)}" if path and os.path.exists(path) else ""), flush=True)
    except Exception as e:
        print(f"  EXC: {label}: {e!r}", flush=True)


mode = sys.argv[1] if len(sys.argv) > 1 else "all"
if mode in ("1", "all"):
    attempt("gpkg single + SRS",
            "COPY (SELECT * FROM pts) TO '/tmp/_g1.gpkg' WITH (FORMAT GDAL, DRIVER 'GPKG', LAYER_NAME 'pts', SRS 'EPSG:4326')",
            "/tmp/_g1.gpkg")
if mode in ("2", "all"):
    attempt("gpkg single no-SRS",
            "COPY (SELECT * FROM pts) TO '/tmp/_g2.gpkg' WITH (FORMAT GDAL, DRIVER 'GPKG')",
            "/tmp/_g2.gpkg")
if mode in ("3", "all"):
    attempt("geojson driver",
            "COPY (SELECT * FROM pts) TO '/tmp/_g3.geojson' WITH (FORMAT GDAL, DRIVER 'GeoJSON')",
            "/tmp/_g3.geojson")
print("done", flush=True)
