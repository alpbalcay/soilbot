"""Parallel OCR: N worker processes do the CPU/GPU-heavy rasterize+easyocr; ONE process writes
to DuckDB (single-writer). This unblocks the throughput ceiling — single-core OCR pegged 1 of 24
cores. Workers never touch the DB (no lock contention); they return parsed ParseResult objects
over a queue and the main process commits strata + manifest.

Resumable via the parse manifest (skips done borings). Processes up to --limit unparsed PDFs per
invocation then exits cleanly, so the driver can checkpoint + re-run B1 between batches.

Run: python scripts/ocr_parallel.py --workers 4 --limit 3000
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# spawned workers start a fresh interpreter without cwd on the path -> ensure the project root
# (parent of scripts/) is importable in every process.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(task_q, result_q):
    # spawned fresh -> imports torch/easyocr in this process, loads its own GPU reader lazily
    from pipeline.parse_logs import extract_with_easyocr, _boring_id_from_filename
    while True:
        pdf = task_q.get()
        if pdf is None:
            break
        bid = _boring_id_from_filename(os.path.basename(pdf))
        try:
            res = extract_with_easyocr(Path(pdf), bid)
            result_q.put((bid, res.status, res.rows, res.note))
        except Exception as exc:  # noqa: BLE001
            result_q.put((bid, "failed", [], f"{type(exc).__name__}: {exc}"[:160]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=3000, help="unparsed PDFs to process this run")
    args = ap.parse_args()

    from pipeline import db
    from pipeline.config import Config
    from pipeline.logging_setup import new_run_id, setup
    from pipeline.parse_logs import _boring_id_from_filename, _insert_rows

    cfg = Config.load(None)
    log = setup(cfg.path("log_dir"), "ml.log", new_run_id(), "ocr_par", console=True)
    con = db.connect(cfg)
    db.bootstrap(con)
    done = db.manifest_keys_with_status(con, "parse", "done") \
        | db.manifest_keys_with_status(con, "parse", "pending") \
        | db.manifest_keys_with_status(con, "parse", "failed")
    logs_dir = cfg.path("logs_pdf_dir")
    pdfs = []
    for f in sorted(glob.glob(str(logs_dir / "*.pdf"))):
        bid = _boring_id_from_filename(os.path.basename(f))
        if f"parse:{bid}" not in done:
            pdfs.append(f)
        if len(pdfs) >= args.limit:
            break
    if not pdfs:
        log.info("ocr_par_nothing_to_do")
        con.close()
        return
    log.info("ocr_par_start", workers=args.workers, batch=len(pdfs))

    task_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    for p in pdfs:
        task_q.put(p)
    for _ in range(args.workers):
        task_q.put(None)
    workers = [mp.Process(target=_worker, args=(task_q, result_q), daemon=True)
               for _ in range(args.workers)]
    for w in workers:
        w.start()

    t0 = time.time()
    parsed = pending = failed = 0
    first_errors = []
    for i in range(len(pdfs)):
        bid, status, rows, note = result_q.get()
        rid = f"parse:{bid}"
        if status == "parsed":
            _insert_rows(con, bid, rows)
            db.manifest_mark(con, "parse", rid, "done", run_id=log.run_id, rows_out=len(rows))
            parsed += 1
        elif status == "pending":
            db.manifest_mark(con, "parse", rid, "pending", run_id=log.run_id)
            pending += 1
        else:
            db.manifest_mark(con, "parse", rid, "failed", run_id=log.run_id)
            failed += 1
            if len(first_errors) < 5:
                first_errors.append(note)
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            log.info("ocr_par_progress", processed=i + 1, of=len(pdfs),
                     parsed=parsed, pending=pending, rate_per_s=round(rate, 2))
    for w in workers:
        w.join(timeout=30)
    dt = time.time() - t0
    log.info("ocr_par_done", processed=len(pdfs), parsed=parsed, pending=pending, failed=failed,
             secs=round(dt, 1), rate_per_s=round(len(pdfs) / dt, 2),
             sample_errors=first_errors)
    con.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")  # CUDA-safe (fork after torch import breaks CUDA)
    main()
