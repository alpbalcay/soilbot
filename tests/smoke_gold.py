"""Smoke test for the OCR gold-set validation tooling (scripts/gold_score.py + gold_diag.py).

Unit-checks the matching/error-taxonomy helpers on a tiny synthetic fixture (no DB/network), and,
if gold/labels.jsonl exists, validates the label schema and that the scorer runs and emits
scores.json. Skips cleanly (exit 0) when the gold set has not been built yet.

Run:  .venv/bin/python tests/smoke_gold.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import gold_score as gs  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    ok = ok and cond


def main():
    # --- pure-function unit checks (always run) ---
    check("norm_id S-8A -> (S,8)", gs.norm_id("S-8A") == ("S", 8))
    check("norm_id J10 -> (J,10)", gs.norm_id("J10") == ("J", 10))
    check("coarse_uscs SM -> SAND", gs.coarse_uscs("SM") == "SAND")
    check("coarse_uscs PT -> ORG", gs.coarse_uscs("PT") == "ORG")
    check("to_ft metric", abs(gs.to_ft(1.0, "m") - 3.2808) < 1e-3)
    check("digit exact", gs.digit_class(20, 20) == "exact")
    check("digit within_2", gs.digit_class(20, 22) == "within_2")
    check("digit off_by_10x", gs.digit_class(40, 4) == "off_by_10x")
    check("digit single_increment", gs.digit_class(22, 14) == "single_increment")
    check("digit gross", gs.digit_class(8, 56) == "gross")

    # matching on a synthetic boring: one id-match, one depth-match, one spurious OCR row
    gold_rows = [{"sid": "S-1", "top": 0.0, "bottom": 1.5, "spt_n": 20, "n_flag": "ok"},
                 {"sid": "S-2", "top": 5.0, "bottom": 6.5, "spt_n": 8, "n_flag": "ok"}]
    ocr_rows = [{"sample_type": "S1", "top_depth": 0.0, "bottom_depth": 1.5, "spt_n": 20, "confidence": 1.0},
                {"sample_type": "X9", "top_depth": 6.4, "bottom_depth": None, "spt_n": 9, "confidence": 1.0},
                {"sample_type": "C1", "top_depth": 40.0, "bottom_depth": None, "spt_n": 99, "confidence": 0.5}]
    matched, um_g, um_o = gs.match_rows(gold_rows, ocr_rows, "ft")
    check("match: 2 matched (id + depth)", len(matched) == 2)
    check("match: 1 spurious ocr", len(um_o) == 1 and um_o[0]["sample_type"] == "C1")

    # --- schema + end-to-end (only if the gold set exists) ---
    labels = Path("gold/labels.jsonl")
    if not labels.exists():
        print("SKIP end-to-end: gold/labels.jsonl not built yet")
        print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
        sys.exit(0 if ok else 1)

    recs = [json.loads(l) for l in open(labels)]
    check("labels non-empty", len(recs) > 0)
    flags = {"ok", "med", "lo", "unclear", "refusal"}
    schema_ok = True
    for r in recs:
        if not all(k in r for k in ("boring_id", "unit", "window", "rows", "fmt")):
            schema_ok = False
        if r.get("unit") not in ("m", "ft"):
            schema_ok = False
        for row in r["rows"]:
            if not all(k in row for k in ("sid", "top", "bottom", "spt_n", "n_flag")):
                schema_ok = False
            if row.get("n_flag") not in flags:
                schema_ok = False
            if row.get("spt_n") is not None and row["spt_n"] < 0:
                schema_ok = False
    check("label schema valid", schema_ok)

    if Path("gold/manifest.json").exists():
        os.system(f"{sys.executable} scripts/gold_score.py >/dev/null 2>&1")
        check("scorer emitted scores.json", Path("gold/scores.json").exists())
        sc = json.loads(Path("gold/scores.json").read_text())
        check("scores has spt_n accuracy", sc["spt_n"]["accuracy"] is not None)
        check("interval recall in [0,1]", 0.0 <= sc["interval_detection"]["recall"] <= 1.0)

    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
