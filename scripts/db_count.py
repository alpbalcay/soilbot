"""Tiny resilient DB counter for the OCR/B1 loop driver (retries while the writer holds the lock)."""
import sys
import time

import duckdb

_SQL = {
    "spt": "SELECT count(DISTINCT boring_id) FROM strata WHERE spt_n IS NOT NULL",
    "parse": "SELECT count(*) FROM manifest WHERE kind='parse' AND status IN ('done','pending','failed')",
}
which = sys.argv[1] if len(sys.argv) > 1 else "spt"
for _ in range(6):
    try:
        c = duckdb.connect("data/soilbot.duckdb", read_only=True)
        print(c.execute(_SQL[which]).fetchone()[0])
        c.close()
        break
    except Exception:
        time.sleep(3)
else:
    print(0)
