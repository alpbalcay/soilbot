"""Second-pass open-access full-text recovery for the harvested papers.

The first harvest only parsed papers whose OpenAlex `oa_url` was a direct PDF with a pdf
content-type. Many OA copies sit behind landing pages (hal.science, hdl.handle.net, ICE library)
or are served with a non-pdf content-type. This pass:
  1. resolves each unparsed paper's DOI through Unpaywall -> best_oa_location.url_for_pdf (direct),
  2. also retries the original OpenAlex oa_url,
  3. downloads following redirects and accepts anything whose bytes start with %PDF (magic sniff),
     not just pdf content-types,
  4. parses with pdfplumber and sets lit_papers.has_fulltext.
Idempotent: skips papers that already have full text. Run: `.venv/bin/python scripts/recover_fulltext.py`
"""
from __future__ import annotations

import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import db  # noqa: E402
from pipeline.config import Config  # noqa: E402
from pipeline.util import RateLimiter, atomic_write_bytes, atomic_write_text, backoff_delay  # noqa: E402

_BROWSER = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36")


def _unpaywall_pdf(doi: str, email: str, limiter: RateLimiter) -> str | None:
    limiter.acquire()
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                         params={"email": email}, timeout=30,
                         headers={"User-Agent": "soilbot/0.1"})
        if r.status_code != 200:
            return None
        j = r.json()
        locs = []
        if j.get("best_oa_location"):
            locs.append(j["best_oa_location"])
        locs += j.get("oa_locations", []) or []
        for loc in locs:
            url = loc.get("url_for_pdf") or loc.get("url")
            if url:
                return url
    except Exception:
        return None
    return None


def _download_pdf(url: str, dest, limiter: RateLimiter, timeout: float) -> bytes | None:
    """Download following redirects; accept by %PDF magic bytes (not content-type)."""
    for attempt in range(3):
        limiter.acquire()
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True,
                             headers={"User-Agent": _BROWSER, "Accept": "application/pdf,*/*"})
        except requests.RequestException:
            time.sleep(backoff_delay(attempt, 1.0, 20)); continue
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            time.sleep(backoff_delay(attempt, 1.0, 20)); continue
        content = r.content
        if content[:5].startswith(b"%PDF") and len(content) > 2000:
            return content
        # some hosts wrap the pdf link in an HTML landing page -> give up here (Unpaywall already
        # gives url_for_pdf for most); avoid scraping arbitrary HTML.
        return None
    return None


def recover() -> dict:
    cfg = Config.load(None)
    lit = cfg.get("litreview", default={}) or {}
    email = lit.get("email", "research@example.com")
    pdf_dir = cfg.abspath(lit.get("pdf_dir", "litreview/pdfs"))
    ft_dir = cfg.abspath(lit.get("fulltext_dir", "litreview/fulltext"))
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ft_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pdfplumber
    except ImportError:
        return {"error": "pdfplumber missing"}

    rate = cfg.rate("oa_pdf")
    dl_lim = RateLimiter(float(rate.get("rps", 2)))
    up_lim = RateLimiter(5.0)
    timeout = float(rate.get("timeout_s", 60))

    con = db.connect(cfg)
    todo = con.execute(
        "SELECT openalex_id, doi, oa_url FROM lit_papers WHERE NOT has_fulltext").fetchall()
    print(f"attempting recovery on {len(todo)} papers without full text")

    recovered = 0
    for i, (oid, doi, oa_url) in enumerate(todo):
        ft_path = ft_dir / f"{oid}.txt"
        if ft_path.exists():
            con.execute("UPDATE lit_papers SET has_fulltext=TRUE WHERE openalex_id=?", [oid])
            recovered += 1
            continue
        # candidate PDF urls: Unpaywall (direct) first, then original oa_url
        urls = []
        if doi:
            u = _unpaywall_pdf(doi, email, up_lim)
            if u:
                urls.append(u)
        if oa_url:
            urls.append(oa_url)
        pdf_bytes = None
        for u in urls:
            pdf_bytes = _download_pdf(u, None, dl_lim, timeout)
            if pdf_bytes:
                break
        if not pdf_bytes:
            continue
        try:
            pdf_path = pdf_dir / f"{oid}.pdf"
            atomic_write_bytes(pdf_path, pdf_bytes)
            parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:12]:
                    parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
            if len(text) > 500:
                atomic_write_text(ft_path, text[:60000])
                con.execute("UPDATE lit_papers SET has_fulltext=TRUE WHERE openalex_id=?", [oid])
                recovered += 1
                if recovered % 5 == 0:
                    print(f"  recovered {recovered} (at paper {i+1}/{len(todo)})")
        except Exception:
            continue

    total_ft = con.execute("SELECT COUNT(*) FROM lit_papers WHERE has_fulltext").fetchone()[0]
    con.close()
    return {"attempted": len(todo), "recovered_this_pass": recovered, "total_fulltext": total_ft}


if __name__ == "__main__":
    import json
    print(json.dumps(recover(), indent=2))
