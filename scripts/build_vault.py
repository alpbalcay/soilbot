"""Build a committed Obsidian vault from the harvested literature + ranked soil properties.

Reads lit_papers / lit_citations / lit_properties / lit_property_links (DuckDB, read-only) and
writes `litreview/vault/`:
  - papers/<openalex_id>.md   — frontmatter (title, authors, year, doi, citations) + abstract +
                                [[wikilinks]] to in-set references and to the properties it motivates
  - properties/<name>.md      — influence score, category, derive-status (already in strata_derived?
                                leaky-for-SPT? derivable?), formula, and [[links]] to evidence papers
  - index.md                  — map-of-content: top papers + ranked properties + the Phase-A shortlist

Wikilinks turn the citation graph + property links into Obsidian's interactive graph view. Notes are
named by stable openalex_id / canonical property key; human titles live in frontmatter `title`/aliases.
Run after the lit_swarm workflow has populated lit_properties: `.venv/bin/python scripts/build_vault.py`
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import db
from pipeline.config import Config


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", (s or "").strip()).strip("-")[:80] or "untitled"


def _yaml_escape(s) -> str:
    if s is None:
        return ""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _prop_title(name: str) -> str:
    return name.replace("_", " ")


def build(config: Config) -> dict:
    lit = config.get("litreview", default={}) or {}
    vault = config.abspath(lit.get("vault_dir", "litreview/vault"))
    (vault / "papers").mkdir(parents=True, exist_ok=True)
    (vault / "properties").mkdir(parents=True, exist_ok=True)

    con = db.connect(config, read_only=True)
    papers = con.execute("""
        SELECT openalex_id, doi, title, year, authors, venue, cited_by_count, concepts,
               abstract, oa_url, has_fulltext, seed_topic, hop
        FROM lit_papers ORDER BY cited_by_count DESC
    """).fetchall()
    pcols = ["id", "doi", "title", "year", "authors", "venue", "cites", "concepts",
             "abstract", "oa_url", "has_fulltext", "seed_topic", "hop"]
    P = [dict(zip(pcols, r)) for r in papers]
    by_id = {p["id"]: p for p in P}

    # citation edges (src cites dst), both in-set
    refs_out: dict[str, list] = {}
    for s, d in con.execute("SELECT src_id, dst_id FROM lit_citations").fetchall():
        refs_out.setdefault(s, []).append(d)

    # property tables (may be empty if the swarm hasn't run yet)
    props = []
    plinks: dict[str, list] = {}
    paper_props: dict[str, list] = {}
    if con.execute("SELECT COUNT(*) FROM lit_properties").fetchone()[0]:
        props = con.execute("""
            SELECT name, aliases, category, influence_score, n_papers, already_derived,
                   leaky_for_spt, derivable_from_inputs, formula_refs, notes
            FROM lit_properties ORDER BY influence_score DESC
        """).fetchall()
        for prop, oid, ev in con.execute(
                "SELECT property, openalex_id, evidence FROM lit_property_links").fetchall():
            plinks.setdefault(prop, []).append((oid, ev))
            paper_props.setdefault(oid, []).append(prop)
    con.close()

    # ---- paper notes ------------------------------------------------------------------------
    for p in P:
        cites_links = [f"[[{d}]]" for d in refs_out.get(p["id"], []) if d in by_id]
        prop_links = [f"[[{pr}|{_prop_title(pr)}]]" for pr in sorted(set(paper_props.get(p["id"], [])))]
        fm = [
            "---",
            f"title: {_yaml_escape(p['title'])}",
            f"aliases: [{_yaml_escape(p['id'])}]",
            f"year: {p['year'] or ''}",
            f"authors: {_yaml_escape(p['authors'])}",
            f"venue: {_yaml_escape(p['venue'])}",
            f"doi: {_yaml_escape(p['doi'])}",
            f"citations: {p['cites']}",
            f"seed_topic: {_yaml_escape(p['seed_topic'])}",
            f"hop: {p['hop']}",
            "tags: [paper, geotech]",
            "---",
        ]
        body = [
            f"# {p['title']}",
            "",
            f"**{p['authors'] or 'Unknown'}** — *{p['venue'] or 'n/a'}* ({p['year'] or 'n.d.'}) · "
            f"**{p['cites']:,} citations**"
            + (f" · [DOI](https://doi.org/{p['doi']})" if p["doi"] else "")
            + (f" · [open access]({p['oa_url']})" if p["oa_url"] else ""),
            "",
        ]
        if prop_links:
            body += ["## Soil properties", " · ".join(prop_links), ""]
        if p["abstract"]:
            body += ["## Abstract", p["abstract"], ""]
        if p["concepts"]:
            body += ["## Concepts", ", ".join(p["concepts"].split("|")), ""]
        if cites_links:
            body += [f"## References in this collection ({len(cites_links)})", " ".join(cites_links), ""]
        (vault / "papers" / f"{p['id']}.md").write_text("\n".join(fm + body), encoding="utf-8")

    # ---- property notes ---------------------------------------------------------------------
    prop_cols = ["name", "aliases", "category", "influence_score", "n_papers", "already_derived",
                 "leaky_for_spt", "derivable_from_inputs", "formula_refs", "notes"]
    PR = [dict(zip(prop_cols, r)) for r in props]
    for pr in PR:
        status = []
        status.append("✅ already derived" if pr["already_derived"] else "🆕 not yet derived")
        status.append("⚠️ leaky for SPT-N" if pr["leaky_for_spt"] else "🟢 non-leaky")
        status.append("🔧 derivable from our inputs" if pr["derivable_from_inputs"]
                      else "🚫 not derivable from our inputs")
        ev = plinks.get(pr["name"], [])
        ev_sorted = sorted(ev, key=lambda x: by_id.get(x[0], {}).get("cites", 0), reverse=True)
        fm = [
            "---", f"title: {_yaml_escape(_prop_title(pr['name']))}",
            f"category: {pr['category']}", f"influence_score: {pr['influence_score']}",
            f"n_papers: {pr['n_papers']}", f"already_derived: {bool(pr['already_derived'])}",
            f"leaky_for_spt: {bool(pr['leaky_for_spt'])}",
            f"derivable_from_inputs: {bool(pr['derivable_from_inputs'])}",
            "tags: [property, geotech]", "---",
        ]
        body = [
            f"# {_prop_title(pr['name'])}", "",
            f"**Category:** {pr['category']} · **Influence (citation-weighted):** "
            f"{pr['influence_score']} across {pr['n_papers']} papers", "",
            "**Status:** " + " · ".join(status), "",
        ]
        if pr.get("aliases"):
            body += [f"*Also called:* {pr['aliases']}", ""]
        if pr.get("formula_refs"):
            body += ["## Derivation", pr["formula_refs"], ""]
        if pr.get("notes"):
            body += ["## Notes", pr["notes"], ""]
        if ev_sorted:
            body += [f"## Evidence papers ({len(ev_sorted)})"]
            for oid, _e in ev_sorted[:25]:
                t = by_id.get(oid, {}).get("title", oid)
                c = by_id.get(oid, {}).get("cites", 0)
                body.append(f"- [[{oid}|{t[:80]}]] ({c:,} cites)")
            body.append("")
        (vault / "properties" / f"{pr['name']}.md").write_text("\n".join(fm + body), encoding="utf-8")

    # ---- index / MOC ------------------------------------------------------------------------
    idx = [
        "---", "title: Geotechnical Literature Vault", "tags: [index]", "---",
        "# Geotechnical Literature → Soil Properties",
        "",
        f"Harvested **{len(P)} foundational geotechnical papers** (OpenAlex, citation-ranked) and the "
        f"**{len(PR)} soil properties** they establish. Links below open the Obsidian graph view.",
        "",
        "## Most-cited papers",
    ]
    for p in P[:25]:
        idx.append(f"- [[{p['id']}|{p['title'][:80]}]] — {p['cites']:,} cites ({p['year'] or 'n.d.'})")
    if PR:
        idx += ["", "## Soil properties by literature influence"]
        for pr in PR:
            flags = []
            if not pr["already_derived"] and not pr["leaky_for_spt"] and pr["derivable_from_inputs"]:
                flags.append("**⭐ shortlist**")
            if pr["already_derived"]:
                flags.append("already derived")
            idx.append(f"- [[{pr['name']}|{_prop_title(pr['name'])}]] — influence "
                        f"{pr['influence_score']} / {pr['n_papers']} papers"
                        + (" · " + ", ".join(flags) if flags else ""))
        shortlist = [pr for pr in PR if not pr["already_derived"]
                     and not pr["leaky_for_spt"] and pr["derivable_from_inputs"]]
        if shortlist:
            idx += ["", "## ⭐ Phase-A information-gain shortlist",
                    "Non-leaky, not-yet-derived, derivable-from-our-inputs — candidates to add to the "
                    "soil-type GNN:"]
            for pr in shortlist:
                idx.append(f"- [[{pr['name']}|{_prop_title(pr['name'])}]] — {pr.get('formula_refs') or ''}")
    (vault / "index.md").write_text("\n".join(idx), encoding="utf-8")

    return {"papers": len(P), "properties": len(PR), "vault": str(vault)}


if __name__ == "__main__":
    print(build(Config.load(None)))
