"""Smoke test: DuckDB spatial extension + ST_Transform axis-order behaviour.

Validates the single biggest architectural risk before building on it:
  1. `INSTALL spatial` / `LOAD spatial` succeed on this box (cached to extension_dir).
  2. ST_Transform from ESRI:102711 (NAD83 / NJ State Plane, US ft) -> EPSG:4326 is correct.
  3. The axis-order flag we must use to get (lon, lat) output.

The point (x=492125 ft, y=0) is the projection's grid origin: false easting 150000 m =
492125.984 ft on central meridian -74.5 deg, latitude of origin 38.8333 deg. So a correct
transform must yield approximately lon=-74.5, lat=38.8333.
"""
import duckdb

EXT_DIR = "/var/home/alp/soilbot/data/.duckdb_extensions"

con = duckdb.connect()
con.execute(f"SET extension_directory='{EXT_DIR}'")
con.execute("INSTALL spatial")
con.execute("LOAD spatial")
print("duckdb", duckdb.__version__, "+ spatial loaded OK")

# Default axis order (authority) -> EPSG:4326 is lat,lon
default = con.execute(
    "SELECT ST_AsText(ST_Transform(ST_Point(492125, 0), 'ESRI:102711', 'EPSG:4326'))"
).fetchone()[0]
# always_xy := true -> output is lon,lat (what we want to store as lon/lat)
xy = con.execute(
    "SELECT ST_AsText(ST_Transform(ST_Point(492125, 0), 'ESRI:102711', 'EPSG:4326', always_xy := true))"
).fetchone()[0]
print("default (authority axis):", default)
print("always_xy=true        :", xy)

# Extract the always_xy point and assert it lands at the projection origin.
lon, lat = con.execute(
    "SELECT ST_X(p), ST_Y(p) FROM (SELECT ST_Transform(ST_Point(492125,0),'ESRI:102711','EPSG:4326', always_xy := true) AS p)"
).fetchone()
print(f"parsed lon={lon:.5f} lat={lat:.5f}")
assert -74.6 < lon < -74.4, f"lon off: {lon}"
assert 38.7 < lat < 39.0, f"lat off: {lat}"
print("PASS: ST_Transform 102711->4326 (always_xy) is correct.")
