"""Diagnose the mapsdep.nj.gov WAF: cooldown, cookies, header/size sensitivity."""
import time

import requests

BASE = "https://mapsdep.nj.gov/arcgis/rest/services/Features/Geology/MapServer"
Q = BASE + "/25/query"
BROWSER = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://mapsdep.nj.gov/",
}


def is_json(r):
    return r.text.lstrip()[:1] == "{"


print("waiting 15s for any rate cooldown...", flush=True)
time.sleep(15)

s = requests.Session()
s.headers.update(BROWSER)

# 1. warm-up: hit service root first (browser-like), inspect cookies
r0 = s.get(BASE, params={"f": "json"}, timeout=30)
print(f"root: json={is_json(r0)} status={r0.status_code} set-cookie={'Set-Cookie' in r0.headers} cookies={list(s.cookies.keys())}", flush=True)

# 2. small query on warmed session
r1 = s.get(Q, params={"where": "1=1", "outFields": "GEONAME", "f": "geojson",
                      "resultRecordCount": 5, "resultOffset": 0}, timeout=30)
print(f"small(5): json={is_json(r1)} status={r1.status_code}", flush=True)

# 3. large query (2000, outFields=*) — the real fetch shape
r2 = s.get(Q, params={"where": "1=1", "outFields": "*", "f": "geojson",
                      "resultRecordCount": 2000, "resultOffset": 0}, timeout=60)
print(f"large(2000,*): json={is_json(r2)} status={r2.status_code} bytes={len(r2.content)}", flush=True)

# 4. burst: 8 rapid large requests, paginating, to find the trigger threshold
ok = 0
for i in range(8):
    r = s.get(Q, params={"where": "1=1", "outFields": "*", "f": "geojson",
                         "resultRecordCount": 2000, "resultOffset": i * 2000}, timeout=60)
    j = is_json(r)
    ok += j
    if not j:
        print(f"  burst req {i} (offset {i*2000}): HTML challenge. set-cookie={'Set-Cookie' in r.headers}", flush=True)
        break
    time.sleep(0.2)
print(f"burst: {ok}/8 json before first challenge", flush=True)

# 5. if challenged, does a 30s cooldown + retry clear it?
if ok < 8:
    print("cooldown 30s then retry...", flush=True)
    time.sleep(30)
    r = s.get(Q, params={"where": "1=1", "outFields": "*", "f": "geojson",
                         "resultRecordCount": 2000, "resultOffset": ok * 2000}, timeout=60)
    print(f"after-cooldown: json={is_json(r)} status={r.status_code}", flush=True)
