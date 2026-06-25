"""Re-parse strata from cached OCR boxes — applies parser fixes (e.g. the USCS heuristic) WITHOUT
re-running the GPU OCR. Reads data/ocr_cache/<bid>.json (written during OCR), re-runs the box->strata
parse, and rewrites strata + the parse manifest. Single process; parsing is cheap.

Run: `.venv/bin/python scripts/reparse_cache.py [--limit N]`
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import db  # noqa: E402
from pipeline.config import Config  # noqa: E402
from pipeline.logging_setup import new_run_id, setup  # noqa: E402
from pipeline.parse_logs import _insert_rows, extract_from_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load(None)
    log = setup(cfg.path("log_dir"), "parse.log", new_run_id(), "reparse", console=True)
    cache_dir = cfg.abspath("data/ocr_cache")
    files = sorted(glob.glob(str(cache_dir / "*.json")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        log.warning("no_cache", note=f"{cache_dir} empty; run OCR first to build the box cache")
        return

    con = db.connect(cfg)
    db.bootstrap(con)
    parsed = pending = failed = 0
    for i, f in enumerate(files):
        bid = os.path.basename(f)[:-5]  # strip .json
        res = extract_from_cache(bid, cache_dir)
        if res is None:
            continue
        rid = f"parse:{bid}"
        if res.status == "parsed":
            _insert_rows(con, bid, res.rows)
            db.manifest_mark(con, "parse", rid, "done", run_id=log.run_id, rows_out=len(res.rows))
            parsed += 1
        elif res.status == "pending":
            db.manifest_mark(con, "parse", rid, "pending", run_id=log.run_id)
            pending += 1
        else:
            db.manifest_mark(con, "parse", rid, "failed", run_id=log.run_id)
            failed += 1
        if (i + 1) % 2000 == 0:
            log.info("reparse_progress", processed=i + 1, of=len(files), parsed=parsed)
    log.info("reparse_done", cached=len(files), parsed=parsed, pending=pending, failed=failed)
    con.close()
    print(f"reparsed {len(files)} cached borings: parsed={parsed} pending={pending} failed={failed}")


if __name__ == "__main__":
    main()
