"""Re-run the OCR PARSER on the gold borings and score against gold — the fast iteration loop for
fixing pipeline/parse_logs.py.

To avoid paying easyocr (GPU) on every parser tweak, OCR is done ONCE and the positioned boxes are
cached to gold/ocr_cache/<boring_id>.json. The score step then re-runs the pure-Python parser on
the cached boxes, so editing parse_logs.py and re-scoring is instant.

  build : OCR the 80 gold PDFs -> gold/ocr_cache/*.json   (run once; ~minutes on GPU)
  score : run the current parser on the cache, score vs gold, print before/after metrics

Run:  .venv/bin/python scripts/gold_reocr.py build      # once
      .venv/bin/python scripts/gold_reocr.py score      # after each parse_logs.py edit
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import gold_score as gscore  # noqa: E402

CACHE = Path("gold/ocr_cache")
MAX_PAGES = 5


def build():
    """Rasterize + easyocr each gold PDF's pages, cache positioned boxes (+ page w/h)."""
    from pipeline import parse_logs as P
    CACHE.mkdir(parents=True, exist_ok=True)
    man = json.load(open("gold/manifest.json"))["borings"]
    for k, b in enumerate(man):
        bid = b["boring_id"]
        outp = CACHE / f"{bid}.json"
        if outp.exists():
            continue
        pdf = Path(b["pdf"])
        try:
            images = P._rasterize(pdf, max_px=3400)
        except Exception as exc:  # noqa: BLE001
            print(f"  {bid}: rasterize failed: {exc}")
            continue
        pages = []
        try:
            for img in images[:MAX_PAGES]:
                p, w, h = P._prep_image(img)
                boxes = P.easyocr_boxes(p)
                pages.append({"w": w, "h": h, "boxes": [list(bx) for bx in boxes]})
        finally:
            P._cleanup_rasters(pdf)
        outp.write_text(json.dumps(pages))
        print(f"  [{k+1}/{len(man)}] {bid}: {len(pages)} pages cached")
    print("cache build done")


def parse_cached(bid):
    """Replicate pipeline.parse_logs.extract_with_easyocr's page logic on cached boxes."""
    from pipeline import parse_logs as P
    fp = CACHE / f"{bid}.json"
    if not fp.exists():
        return []
    pages = json.loads(fp.read_text())
    rows = []
    spoon = False
    for i, pg in enumerate(pages):
        boxes = [tuple(bx) for bx in pg["boxes"]]
        w, h = pg["w"], pg["h"]
        if i == 0:
            spoon = P.is_spoon_format(boxes)
        if spoon:
            r, _ = P.parse_spoon_format(boxes, w, h)
            rows.extend(r)
        else:
            rows.extend(P.parse_boxes_to_strata(boxes, w, h))
            break
    out = []
    for r in rows:
        out.append({"sample_type": r.sample_type, "top_depth": r.top_depth,
                    "bottom_depth": r.bottom_depth, "spt_n": r.spt_n,
                    "uscs_class": r.uscs_class, "confidence": r.confidence})
    return out


def score():
    import importlib
    import pipeline.parse_logs as P
    importlib.reload(P)        # pick up edits without a fresh interpreter
    gold = [json.loads(l) for l in open("gold/labels.jsonl")]
    ocr_by_boring = {rec["boring_id"]: parse_cached(rec["boring_id"]) for rec in gold}
    n_rows = sum(len(v) for v in ocr_by_boring.values())
    scores = gscore.compute_scores(gold, ocr_by_boring)
    print(f"(re-parsed {n_rows} OCR rows from cache over {len(gold)} borings)")
    gscore.print_summary(scores)
    Path("gold/scores_reocr.json").write_text(json.dumps(scores, indent=2))
    return scores


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    if mode == "build":
        build()
    else:
        score()
