"""Persist the lit_swarm workflow output into lit_properties / lit_property_links.

The swarm returns JSON {ranking:[...], gap:{classified, shortlist, summary}}. This folds the
citation-weighted ranking together with the gap-analysis classification and writes the two property
tables (full rebuild). Run: `.venv/bin/python scripts/persist_props.py <workflow_output.json>`.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import db
from pipeline.config import Config


def persist(path: str) -> dict:
    data = json.loads(open(path, encoding="utf-8").read())
    ranking = data.get("ranking", [])
    gap = data.get("gap", {}) or {}
    gap_by = {c["property"]: c for c in gap.get("classified", [])}
    shortlist = set(gap.get("shortlist", []))

    con = db.connect(Config.load(None))
    db.bootstrap(con)
    con.execute("BEGIN")
    con.execute("DELETE FROM lit_properties")
    con.execute("DELETE FROM lit_property_links")
    n_links = 0
    for r in ranking:
        name = r["property"]
        g = gap_by.get(name, {})
        formula = g.get("formula", "") or ""
        src = g.get("source_hint", "") or ""
        formula_refs = (f"{formula}  [{src}]" if formula and src else formula or src)
        notes = g.get("notes", "") or ""
        if name in shortlist:
            notes = ("⭐ Phase-A shortlist. " + notes).strip()
        con.execute(
            """INSERT INTO lit_properties (name, aliases, category, influence_score, n_papers,
               already_derived, leaky_for_spt, derivable_from_inputs, formula_refs, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [name, "; ".join(r.get("top_names", [])), r.get("category"),
             r.get("influence_score"), r.get("n_papers"),
             bool(g.get("already_derived", False)), bool(g.get("leaky_for_spt", False)),
             bool(g.get("derivable_from_inputs", False)), formula_refs, notes])
        for ev in r.get("evidence", []):
            con.execute(
                "INSERT INTO lit_property_links VALUES (?,?,?) ON CONFLICT DO NOTHING",
                [name, ev.get("id"), (ev.get("role") or "")[:400]])
            n_links += 1
    con.execute("COMMIT")
    con.close()
    return {"properties": len(ranking), "links": n_links, "shortlist": sorted(shortlist)}


if __name__ == "__main__":
    out = persist(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lit_swarm_out.json")
    print(json.dumps(out, indent=2))
