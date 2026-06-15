"""Smoke: NULL geometry handling, inverse transform, multi-layer GeoPackage append."""
import json
import os

import duckdb

con = duckdb.connect()
con.execute("SET extension_directory='/var/home/alp/soilbot/data/.duckdb_extensions'")
con.execute("LOAD spatial")

# 1. NULL geometry through ST_GeomFromGeoJSON in executemany
pt = json.dumps({"type": "Point", "coordinates": [-74.5, 40.1]})
con.execute("CREATE TABLE t(a INT, g GEOMETRY)")
con.executemany("INSERT INTO t VALUES (?, ST_GeomFromGeoJSON(?))", [[1, pt], [2, None]])
print("null-geom rows:", con.execute("SELECT a, ST_AsText(g) FROM t ORDER BY a").fetchall())

# 2. inverse transform 4326 -> ESRI:102711
rt = con.execute(
    "SELECT ST_X(ST_Transform(ST_Point(-74.5,40.1),'EPSG:4326','ESRI:102711',always_xy:=true))"
).fetchone()
print("native easting for (-74.5,40.1):", round(rt[0], 1), "ft")

# 3. multi-layer GeoPackage append
gp = "/tmp/_test_multi.gpkg"
if os.path.exists(gp):
    os.remove(gp)
con.execute("CREATE TABLE pts(id INT, geom GEOMETRY)")
con.execute("INSERT INTO pts VALUES (1, ST_Point(-74.5,40.1)),(2, ST_Point(-74.6,40.2))")
con.execute("CREATE TABLE polys(id INT, geom GEOMETRY)")
con.execute("INSERT INTO polys VALUES (1, ST_GeomFromText('POLYGON((-75 40,-74 40,-74 41,-75 41,-75 40))'))")
con.execute(f"COPY (SELECT * FROM pts) TO '{gp}' WITH (FORMAT GDAL, DRIVER 'GPKG', LAYER_NAME 'pts', SRS 'EPSG:4326')")
con.execute(f"COPY (SELECT * FROM polys) TO '{gp}' WITH (FORMAT GDAL, DRIVER 'GPKG', LAYER_NAME 'polys', SRS 'EPSG:4326')")
try:
    npts = con.execute(f"SELECT count(*) FROM ST_Read('{gp}', layer='pts')").fetchone()[0]
    npoly = con.execute(f"SELECT count(*) FROM ST_Read('{gp}', layer='polys')").fetchone()[0]
    print(f"gpkg multi-layer append: pts={npts} polys={npoly}")
except Exception as e:
    print("gpkg layer read ERROR:", repr(e))
