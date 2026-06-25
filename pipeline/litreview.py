"""Phase 7 — deterministic harvest of foundational geotechnical-engineering papers (OpenAlex).

Pulls ~40 canonical seed queries, expands one citation hop via co-citation (references shared by
>=2 seed papers), ranks by `cited_by_count` (the influence signal), downloads open-access full text
where available, and persists to `lit_papers` / `lit_citations` (+ JSON metadata cache). The agent
swarm (scripts/lit_swarm) then reads these tables to extract and rank soil properties; the vault
writer turns them into a committed Obsidian graph.

Reuses the pipeline's HTTP-politeness + idempotency primitives: `RateLimiter`, `backoff_delay`,
`manifest_*`, `atomic_write_*`. All endpoints are public; OpenAlex needs only a `mailto` (polite
pool), no key.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import requests

from . import db
from .config import Config
from .util import RateLimiter, atomic_write_bytes, atomic_write_text, backoff_delay

_OPENALEX_SELECT = (
    "id,doi,title,display_name,publication_year,cited_by_count,authorships,"
    "primary_location,concepts,abstract_inverted_index,referenced_works,open_access"
)

# Topical relevance gate. Co-citation expansion otherwise floats generic high-citation works
# (Vapnik's statistical-learning theory, SAR interferometry, clathrate hydrates) to the top because
# liquefaction/ML geotech papers happen to cite them. Keep only works whose title or OpenAlex
# concepts hit a geotechnical/soil term, so citation ranking surfaces real soil-property papers.
_GEOTECH_TERMS = (
    "soil", "geotechn", "geomechan", "consolidation", "penetration test", " spt", " cpt",
    "liquefaction", "foundation", "clay", "sand", "silt", "shear strength", "undrained",
    "drained", "plasticity", "atterberg", "permeability", "hydraulic conductivity",
    "bearing capacity", "settlement", "overconsolidat", "friction angle", "relative density",
    "effective stress", "pore pressure", "pore-water", "earth pressure", "slope stability",
    "retaining", "embankment", "rock mechanics", "void ratio",
    "compaction", "triaxial", "oedometer", "consolidat", "compressibility", "shear modulus",
    "shear wave", "cone penetration", "standard penetration", "subgrade", "geomaterial",
    "granular soil", "stress history", "critical state soil", "geotechnical",
)

# Off-domain veto. The liquefaction seeds chain into earthquake-source / InSAR / chemistry papers
# (Okada fault models, SAR interferometry, clathrate hydrates) that aren't soil-property sources.
# A block term vetoes a paper even if an allow term also matches.
_BLOCK_TERMS = (
    "interferometry", "clathrate", "hydrate", "scatterers", "tensile fault", "tensile faults",
    "surface deformation", "statistical learning", "machine learning", "neural network",
    "earthquakes and faulting", "faulting", "natural gas", "remote sensing", "sar ",
)


def _is_geotech(work: dict) -> bool:
    hay = (work.get("title", "") + " " + work.get("concepts", "")).lower()
    if any(b in hay for b in _BLOCK_TERMS):
        return False
    return any(t in hay for t in _GEOTECH_TERMS)


def _short_id(openalex_id: str | None) -> str | None:
    """'https://openalex.org/W2041893233' -> 'W2041893233'."""
    if not openalex_id:
        return None
    return openalex_id.rsplit("/", 1)[-1]


def _reconstruct_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)[:6000]


def _get_json(url: str, params: dict, limiter: RateLimiter, log, rate: dict,
              ua: str) -> dict | None:
    """GET JSON with rate-limit + exponential backoff (reuses pipeline politeness primitives)."""
    retries = int(rate.get("max_retries", 5))
    base = float(rate.get("backoff_base_s", 1.0))
    cap = float(rate.get("backoff_max_s", 60))
    timeout = float(rate.get("timeout_s", 30))
    for attempt in range(retries + 1):
        limiter.acquire()
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": ua})
        except requests.RequestException as exc:
            if attempt >= retries:
                log.warning("openalex_conn_giveup", url=url, err=str(exc)[:120])
                return None
            time.sleep(backoff_delay(attempt, base, cap))
            continue
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt >= retries:
                return None
            time.sleep(backoff_delay(attempt, base, cap))
            continue
        if resp.status_code >= 400:
            log.warning("openalex_http_error", status=resp.status_code, url=url)
            return None
        try:
            return resp.json()
        except ValueError:
            if attempt >= retries:
                return None
            time.sleep(backoff_delay(attempt, base, cap))
    return None


def _normalize_work(w: dict, seed_topic: str | None, hop: int) -> dict | None:
    sid = _short_id(w.get("id"))
    if not sid:
        return None
    authors = [a.get("author", {}).get("display_name")
               for a in (w.get("authorships") or [])[:6]]
    authors = [a for a in authors if a]
    loc = w.get("primary_location") or {}
    venue = (loc.get("source") or {}).get("display_name")
    concepts = [c.get("display_name") for c in (w.get("concepts") or [])
                if c.get("level", 9) <= 2 and c.get("score", 0) >= 0.2][:8]
    oa = w.get("open_access") or {}
    return {
        "openalex_id": sid,
        "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
        "title": (w.get("title") or w.get("display_name") or "")[:500],
        "year": w.get("publication_year"),
        "authors": "; ".join(authors),
        "venue": venue,
        "cited_by_count": int(w.get("cited_by_count") or 0),
        "concepts": "|".join(c for c in concepts if c),
        "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
        "oa_url": oa.get("oa_url"),
        "referenced_works": [_short_id(r) for r in (w.get("referenced_works") or [])],
        "seed_topic": seed_topic,
        "hop": hop,
        "source": "openalex",
    }


def harvest(config: Config, log, *, seeds=None, hops=None, max_papers=None,
            per_seed=None, do_fulltext=True) -> dict:
    lit = config.get("litreview", default={}) or {}
    seeds = seeds if seeds is not None else lit.get("seeds", [])
    hops = hops if hops is not None else int(lit.get("hops", 1))
    max_papers = max_papers if max_papers is not None else int(lit.get("max_papers", 300))
    per_seed = per_seed if per_seed is not None else int(lit.get("per_seed", 25))
    min_year = int(lit.get("min_year", 1936))
    email = lit.get("email", "research@example.com")
    url = lit.get("openalex_url", "https://api.openalex.org/works")
    ua = config.user_agent
    rate = config.rate("openalex")
    limiter = RateLimiter(float(rate.get("rps", 8)))

    meta_dir = config.abspath(lit.get("metadata_dir", "litreview/metadata"))
    meta_dir.mkdir(parents=True, exist_ok=True)

    con = db.connect(config)
    db.bootstrap(con)

    pool: dict[str, dict] = {}   # short_id -> normalized work

    # ---- seed queries (hop 0) -----------------------------------------------------------------
    for topic in seeds:
        data = _get_json(url, {
            "search": topic, "per_page": per_seed, "select": _OPENALEX_SELECT,
            "filter": f"from_publication_date:{min_year}-01-01",
            "sort": "cited_by_count:desc", "mailto": email,
        }, limiter, log, rate, ua)
        if not data:
            log.warning("seed_failed", topic=topic)
            continue
        n = 0
        for w in data.get("results", []):
            nw = _normalize_work(w, topic, 0)
            if nw and nw["openalex_id"] not in pool:
                pool[nw["openalex_id"]] = nw
                n += 1
        log.info("seed_done", topic=topic[:48], got=n, pool=len(pool))

    # ---- co-citation expansion (hop 1): references shared by >=2 seed papers -------------------
    if hops >= 1:
        ref_freq: Counter = Counter()
        for w in pool.values():
            for r in set(w["referenced_works"]):
                if r and r not in pool:
                    ref_freq[r] += 1
        candidates = [rid for rid, c in ref_freq.most_common() if c >= 2][:200]
        log.info("expansion_candidates", n=len(candidates))
        for rid in candidates:
            data = _get_json(f"{url}/{rid}", {"select": _OPENALEX_SELECT, "mailto": email},
                             limiter, log, rate, ua)
            if not data:
                continue
            nw = _normalize_work(data, None, 1)
            if (nw and nw.get("year") and nw["year"] >= min_year
                    and nw["openalex_id"] not in pool and _is_geotech(nw)):
                pool[nw["openalex_id"]] = nw

    # ---- topical gate + rank by influence, cap ----------------------------------------------
    # Apply the geotech gate to EVERY work (incl. hop-0 seed hits): OpenAlex full-text search on
    # loose seed phrases ("...Taylor square root time") otherwise returns high-citation off-domain
    # blockbusters (microbiome, mortality, p-value papers) that have no geotech concept. Then rank
    # by citation count.
    on_topic = [w for w in pool.values() if _is_geotech(w)]
    ranked = sorted(on_topic, key=lambda w: w["cited_by_count"], reverse=True)[:max_papers]
    keep_ids = {w["openalex_id"] for w in ranked}
    log.info("ranked", kept=len(ranked), pool=len(pool),
             top=[(w["openalex_id"], w["cited_by_count"]) for w in ranked[:5]])

    # ---- open-access full text --------------------------------------------------------------
    ft_ok = 0
    if do_fulltext:
        ft_ok = _fetch_fulltext(config, log, ranked, keep_ids)

    # ---- persist papers + citation edges + metadata cache (full rebuild) --------------------
    con.execute("BEGIN")
    con.execute("DELETE FROM lit_papers")
    con.execute("DELETE FROM lit_citations")
    for w in ranked:
        con.execute(
            """INSERT INTO lit_papers (openalex_id, doi, title, year, authors, venue,
               cited_by_count, concepts, abstract, oa_url, has_fulltext, seed_topic, hop, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [w["openalex_id"], w["doi"], w["title"], w["year"], w["authors"], w["venue"],
             w["cited_by_count"], w["concepts"], w["abstract"], w["oa_url"],
             bool(w.get("_fulltext")), w["seed_topic"], w["hop"], w["source"]])
        atomic_write_text(meta_dir / f"{w['openalex_id']}.json", json.dumps(w, default=str))
        db.manifest_mark(con, "litpaper", w["openalex_id"], "done", run_id=getattr(log, "run_id", None),
                         rows_out=1)
    n_edges = 0
    for w in ranked:
        for r in set(w["referenced_works"]):
            if r in keep_ids:
                con.execute(
                    "INSERT INTO lit_citations VALUES (?,?) ON CONFLICT DO NOTHING",
                    [w["openalex_id"], r])
                n_edges += 1
    con.execute("COMMIT")
    con.close()

    summary = {"papers": len(ranked), "citations": n_edges, "fulltext": ft_ok,
               "pool": len(pool), "seeds": len(seeds)}
    log.info("harvest_done", **summary)
    return summary


def _fetch_fulltext(config: Config, log, ranked: list[dict], keep_ids: set[str]) -> int:
    """Download OA PDFs and extract text via pdfplumber; tag works with _fulltext True."""
    lit = config.get("litreview", default={}) or {}
    pdf_dir = config.abspath(lit.get("pdf_dir", "litreview/pdfs"))
    ft_dir = config.abspath(lit.get("fulltext_dir", "litreview/fulltext"))
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ft_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pdfplumber
    except ImportError:
        log.warning("pdfplumber_missing_fulltext_skipped")
        return 0
    rate = config.rate("oa_pdf")
    limiter = RateLimiter(float(rate.get("rps", 2)))
    ua = config.user_agent
    timeout = float(rate.get("timeout_s", 60))
    ok = 0
    for w in ranked:
        oa = w.get("oa_url")
        if not oa:
            continue
        ft_path = ft_dir / f"{w['openalex_id']}.txt"
        if ft_path.exists():
            w["_fulltext"] = True
            ok += 1
            continue
        limiter.acquire()
        try:
            resp = requests.get(oa, timeout=timeout, headers={"User-Agent": ua}, stream=True)
            if resp.status_code != 200 or "pdf" not in resp.headers.get("Content-Type", "").lower():
                continue
            pdf_bytes = resp.content
            if len(pdf_bytes) < 2000:
                continue
            pdf_path = pdf_dir / f"{w['openalex_id']}.pdf"
            atomic_write_bytes(pdf_path, pdf_bytes)
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:12]:        # first ~12 pages cover intro/methods/properties
                    text_parts.append(page.extract_text() or "")
            text = "\n".join(text_parts).strip()
            if len(text) > 500:
                atomic_write_text(ft_path, text[:60000])
                w["_fulltext"] = True
                ok += 1
        except Exception as exc:  # noqa: BLE001 — best-effort; paywalls/odd PDFs are expected
            log.warning("oa_fetch_failed", id=w["openalex_id"], err=str(exc)[:100])
    log.info("fulltext_done", ok=ok)
    return ok


def run(config: Config, log, limit: int | None = None) -> dict:
    """Phase-7 entrypoint (called from pipeline.run --litreview)."""
    return harvest(config, log, max_papers=limit)
