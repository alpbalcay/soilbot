"""Apply the literature-derived classification checks to our strata data (non-destructive).

Codifies the machine-applicable actions the classification-knowledge swarm synthesized from the
harvested papers, and writes a `strata_quality` flag table (one row per strata interval) plus a
printed summary. It FLAGS, never rewrites — the degenerate USCS distribution is a parser artifact
that can only be truly fixed by re-OCR with an improved description->USCS heuristic (see
CLASSIFICATION_KNOWLEDGE.md), so silently rewriting classes would fabricate data.

Run: `.venv/bin/python scripts/classify_audit.py`
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import db  # noqa: E402
from pipeline.config import Config  # noqa: E402

# ASTM D2487 legal group symbols (+ the dual symbols the parser emits, + the project 'FILL' token).
LEGAL_USCS = {"GW", "GP", "GM", "GC", "SW", "SP", "SM", "SC", "ML", "CL", "OL", "MH", "CH", "OH",
              "PT", "GW-GM", "GP-GM", "SW-SM", "SP-SM", "SC-SM", "FILL"}
FINE = ("ML", "CL", "OL", "CH", "MH", "OH")          # fine-grained classes (Casagrande chart)
GRANULAR = ("SP", "SW", "SM", "SC", "GP", "GW", "GM", "GC")
# the low/poorly-graded variants the description->USCS heuristic over-emits (W/H/O suffix dropped)
SUFFIX_DEGENERATE = ("ML", "SP", "GP", "CL", "OL")


def run() -> dict:
    cfg = Config.load(None)
    con = db.connect(cfg)
    con.execute("DROP TABLE IF EXISTS strata_quality")
    con.execute("""
        CREATE TABLE strata_quality AS
        SELECT s.boring_id, s.interval_index, s.uscs_class, s.spt_n, s.top_depth, s.bottom_depth,
               s.gw_depth,
               (s.uscs_class IS NOT NULL AND s.uscs_class NOT IN ({legal})) AS flag_illegal_uscs,
               (s.uscs_class IN ({fine}) AND s.spt_n > 50) AS flag_fine_high_spt,
               (s.uscs_class IN ({gran}) AND s.spt_n IS NOT NULL AND s.spt_n < 15
                  AND s.gw_depth IS NOT NULL AND s.bottom_depth > s.gw_depth) AS flag_liquefiable,
               (s.uscs_class IN ({degen})) AS flag_suffix_ambiguous
        FROM strata s
    """.format(
        legal=",".join(f"'{c}'" for c in LEGAL_USCS),
        fine=",".join(f"'{c}'" for c in FINE),
        gran=",".join(f"'{c}'" for c in GRANULAR),
        degen=",".join(f"'{c}'" for c in SUFFIX_DEGENERATE),
    ))

    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    out = {
        "intervals": q("SELECT COUNT(*) FROM strata_quality"),
        "with_uscs": q("SELECT COUNT(*) FROM strata_quality WHERE uscs_class IS NOT NULL"),
        "illegal_uscs": q("SELECT COUNT(*) FROM strata_quality WHERE flag_illegal_uscs"),
        "fine_high_spt": q("SELECT COUNT(*) FROM strata_quality WHERE flag_fine_high_spt"),
        "liquefiable": q("SELECT COUNT(*) FROM strata_quality WHERE flag_liquefiable"),
        "suffix_ambiguous": q("SELECT COUNT(*) FROM strata_quality WHERE flag_suffix_ambiguous"),
        "h_w_o_suffix_present": q(
            "SELECT COUNT(*) FROM strata WHERE uscs_class IN ('CH','MH','SW','GW','OH')"),
    }
    # groundwater fill-down potential (one water table per boring)
    out["gw_intervals"] = q("SELECT COUNT(*) FROM strata WHERE gw_depth IS NOT NULL")
    out["gw_filldown_gain"] = q("""
        SELECT COUNT(*) FROM strata s WHERE s.gw_depth IS NULL
          AND s.boring_id IN (SELECT boring_id FROM strata WHERE gw_depth IS NOT NULL)""")
    con.close()

    print("strata_quality written. summary:")
    for k, v in out.items():
        print(f"  {k:22} {v:,}")
    return out


if __name__ == "__main__":
    run()
