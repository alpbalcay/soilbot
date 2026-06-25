"""Sample a stratified gold set of SPT-bearing boring logs and render their pages for hand/vision
transcription. Step 1 of the OCR gold-set validation (see gold/GOLD_VALIDATION.md).

Picks ~N distinct borings that have OCR'd SPT-N, stratified by the boring's representative SPT-N
magnitude band (the dimension that actually varies — `source` is uniformly 'ocr' and the heuristic
`confidence` is ~always >=0.7), and force-includes every rare mid/low-confidence boring so the
confidence heuristic itself gets validated. Renders each boring's PDF pages to PNG (poppler
pdftoppm) and writes `gold/manifest.json` with the OCR'd strata rows alongside, so the
transcription step has the model's output to compare against.

Deterministic: same --seed reproduces the identical sample. Read-only against the DB.

Run:  .venv/bin/python scripts/gold_sample.py [--n 80] [--seed 1337] [--no-render]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import Config  # noqa: E402

BANDS = [("0-9", 0, 10), ("10-29", 10, 30), ("30-49", 30, 50), ("50+", 50, 10_000)]
CONF_HI = 0.7          # borings with any spt row below this are "rare" -> force-included
RENDER_PX = 1900       # longest-side px; <2000 so multi-image reads work, digits stay legible


def _band(n: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= n < hi:
            return name
    return "50+"


def _rank(seed: int, boring_id: str) -> int:
    """Stable per-boring sort key (deterministic shuffle within a band)."""
    return int(hashlib.md5(f"{seed}:{boring_id}".encode()).hexdigest(), 16)


def _render(pdf: Path, out_dir: Path) -> list[str]:
    """Rasterize all PDF pages to PNG via poppler pdftoppm (same engine as parse_logs._rasterize).
    Returns repo-relative page paths, sorted by page number."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / pdf.stem
    subprocess.run(["pdftoppm", "-png", "-scale-to", str(RENDER_PX), str(pdf), str(prefix)],
                   check=True, capture_output=True, timeout=240)
    return [str(p) for p in sorted(out_dir.glob(f"{pdf.stem}*.png"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    cfg = Config.load(None)
    ap.add_argument("--n", type=int, default=80, help="target boring count")
    ap.add_argument("--seed", type=int, default=int(cfg.d.get("ml", {}).get("seed", 1337)))
    ap.add_argument("--no-render", action="store_true", help="rebuild manifest without rasterizing")
    args = ap.parse_args()

    if not shutil.which("pdftoppm") and not args.no_render:
        print("FAIL: pdftoppm (poppler) not installed; rerun with --no-render or install poppler")
        return 1

    out = Path("gold")
    render_root = out / "render"
    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)

    # Per-boring summary over its SPT rows: median N -> band, min confidence (catch rare flags),
    # row count and depth span (context for transcription).
    rows = con.execute("""
        SELECT boring_id,
               median(spt_n)        AS med_n,
               min(confidence)      AS min_conf,
               count(*)             AS n_rows,
               min(top_depth)       AS d_lo,
               max(bottom_depth)    AS d_hi
        FROM strata WHERE spt_n IS NOT NULL
        GROUP BY boring_id
    """).fetchall()
    borings = [{"boring_id": r[0], "med_n": float(r[1]), "min_conf": float(r[2]),
                "n_rows": int(r[3]), "band": _band(float(r[1]))} for r in rows]

    forced = [b for b in borings if b["min_conf"] < CONF_HI]
    forced_ids = {b["boring_id"] for b in forced}

    # fill the rest evenly across bands, deterministic order, skipping already-forced borings
    by_band: dict[str, list] = {name: [] for name, _, _ in BANDS}
    for b in borings:
        if b["boring_id"] not in forced_ids:
            by_band[b["band"]].append(b)
    for name in by_band:
        by_band[name].sort(key=lambda b: _rank(args.seed, b["boring_id"]))

    chosen = list(forced)
    remaining = max(0, args.n - len(chosen))
    per_band = remaining // len(BANDS)
    for name, _, _ in BANDS:
        chosen += by_band[name][:per_band]
    # top up to exactly n from the global deterministic order if rounding left a shortfall
    if len(chosen) < args.n:
        pool = sorted((b for name in by_band for b in by_band[name][per_band:]),
                      key=lambda b: _rank(args.seed, b["boring_id"]))
        chosen += pool[: args.n - len(chosen)]
    chosen.sort(key=lambda b: b["boring_id"])

    if not args.no_render and render_root.exists():
        shutil.rmtree(render_root)

    manifest = []
    missing = 0
    for b in chosen:
        bid = b["boring_id"]
        pdfs = sorted(glob.glob(f"data/logs/{bid}__*.pdf"))
        if not pdfs:
            print(f"  WARN no PDF on disk for {bid}; skipping")
            missing += 1
            continue
        pdf = Path(pdfs[0])
        pages = [] if args.no_render else _render(pdf, render_root / bid)
        ocr_rows = con.execute("""
            SELECT interval_index, top_depth, bottom_depth, uscs_class, spt_n, sample_type, confidence
            FROM strata WHERE boring_id = ? AND spt_n IS NOT NULL ORDER BY interval_index
        """, [bid]).fetchall()
        manifest.append({
            "boring_id": bid,
            "pdf": str(pdf),
            "band": b["band"],
            "med_n": b["med_n"],
            "min_conf": b["min_conf"],
            "n_pages": len(pages),
            "pages": pages,
            "ocr_depth_span": [
                min((r[1] for r in ocr_rows if r[1] is not None), default=None),
                max((r[2] for r in ocr_rows if r[2] is not None), default=None)],
            "ocr_rows": [{"interval_index": r[0], "top_depth": r[1], "bottom_depth": r[2],
                          "uscs_class": r[3], "spt_n": r[4], "sample_type": r[5],
                          "confidence": r[6]} for r in ocr_rows],
        })
    con.close()

    out.mkdir(exist_ok=True)
    meta = {"seed": args.seed, "n_requested": args.n, "n_borings": len(manifest),
            "n_forced_lowconf": len(forced), "missing_pdf": missing,
            "bands": {name: sum(1 for m in manifest if m["band"] == name) for name, _, _ in BANDS},
            "borings": manifest}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote gold/manifest.json: {len(manifest)} borings "
          f"({meta['bands']}), {len(forced)} forced low-conf, {missing} missing PDFs")
    if not args.no_render:
        npng = sum(m["n_pages"] for m in manifest)
        print(f"rendered {npng} page PNGs under gold/render/ (gitignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
