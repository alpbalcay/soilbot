"""Layer-0 scanned-log attachments.

Two responsibilities:
  * enumerate_bulk()  — fast catalog of (objectid -> attachment) via /queryAttachments,
    used in Phase 2 to populate borings.log_url. ~tens of calls, not 49k.
  * download_logs()   — the GATED, resumable, rate-limited PDF crawl (multi-hour, ~3-4 GB).
    Off by default; only runs behind --download-logs.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from . import db
from .arcgis import ArcGISClient
from .config import Config
from .util import RateLimiter, atomic_write_bytes, backoff_delay, ensure_dir, sha256_bytes

_thread_local = threading.local()


def _session(user_agent: str) -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": user_agent})
        _thread_local.session = s
    return s


# ---------------------------------------------------------------------------
# Phase 2: bulk attachment catalog -> borings.log_url
# ---------------------------------------------------------------------------
def enumerate_bulk(con, client: ArcGISClient, config: Config, log, force: bool = False) -> dict:
    layer_url = config.layer_url("borings")
    enum_key = "layer0:attachments"
    if not force and db.manifest_is_done(con, "enum", enum_key):
        n = db.table_count(con, "boring_attachments")
        log.info("attachment_enum_skip", reason="already_done", catalog=n)
        return {"skipped": True, "catalog": n}

    meta = client.layer_metadata(layer_url)
    if not ArcGISClient.supports_query_attachments(meta):
        log.warning("attachment_enum_unsupported",
                    note="queryAttachments unavailable; log_url left null until --download-logs")
        return {"supported": False, "catalog": 0}

    # Batch object ids (server rejects where=1=1; objectIds list works).
    oids = [r[0] for r in con.execute(
        "SELECT objectid FROM borings WHERE objectid IS NOT NULL ORDER BY objectid").fetchall()]
    batch = 250
    rows = []
    for i in range(0, len(oids), batch):
        chunk = oids[i:i + batch]
        for g in client.attachments_for_oids(layer_url, chunk):
            oid = g.get("parentObjectId")
            infos = g.get("attachmentInfos") or []
            if oid is None or not infos:
                continue
            info = next((x for x in infos if x.get("contentType") == "application/pdf"), infos[0])
            aid = info.get("id")
            url = f"{layer_url}/{oid}/attachments/{aid}"
            rows.append([oid, aid, info.get("name"), info.get("contentType"),
                         info.get("size"), url])
        if (i // batch) % 20 == 0:
            log.info("attachment_enum_progress", oids_done=min(i + batch, len(oids)),
                     total=len(oids), catalog=len(rows))

    con.execute("BEGIN")
    try:
        con.executemany(
            """INSERT INTO boring_attachments
                   (objectid, attachment_id, name, content_type, size, download_url)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT (objectid) DO UPDATE SET
                   attachment_id=EXCLUDED.attachment_id, name=EXCLUDED.name,
                   content_type=EXCLUDED.content_type, size=EXCLUDED.size,
                   download_url=EXCLUDED.download_url""",
            rows,
        )
        # Backfill parent_lid from borings, and push log_url/attachment_id onto borings.
        con.execute("""UPDATE boring_attachments ba SET parent_lid = b.boring_id
                       FROM borings b WHERE b.objectid = ba.objectid""")
        con.execute("""UPDATE borings SET log_url = ba.download_url,
                                          attachment_id = ba.attachment_id
                       FROM boring_attachments ba WHERE ba.objectid = borings.objectid""")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    db.manifest_mark(con, "enum", enum_key, "done", run_id=log.run_id, rows_out=len(rows))
    linked = con.execute("SELECT COUNT(*) FROM borings WHERE log_url IS NOT NULL").fetchone()[0]
    log.info("attachment_enum_done", catalog=len(rows), borings_linked=linked)
    return {"catalog": len(rows), "borings_linked": linked}


# ---------------------------------------------------------------------------
# Phase 3 (gated): resumable PDF crawl
# ---------------------------------------------------------------------------
def _download_one(url: str, target, ua: str, limiter: RateLimiter,
                  max_retries: int, base: float, cap: float, timeout: float) -> dict:
    import time
    sess = _session(ua)
    for attempt in range(max_retries + 1):
        limiter.acquire()
        try:
            resp = sess.get(url, timeout=timeout)
        except requests.RequestException:
            if attempt >= max_retries:
                return {"ok": False, "http_status": None}
            time.sleep(backoff_delay(attempt, base, cap))
            continue
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt >= max_retries:
                return {"ok": False, "http_status": resp.status_code}
            time.sleep(backoff_delay(attempt, base, cap))
            continue
        if resp.status_code >= 400:
            return {"ok": False, "http_status": resp.status_code}
        content = resp.content
        ctype = resp.headers.get("Content-Type", "")
        ok = content[:5] == b"%PDF-" or "pdf" in ctype.lower()
        if ok:
            atomic_write_bytes(target, content)
        return {"ok": ok, "http_status": resp.status_code, "bytes": len(content),
                "sha256": sha256_bytes(content), "content_type": ctype}
    return {"ok": False, "http_status": None}


def download_logs(con, client: ArcGISClient, config: Config, log,
                  max_downloads: int | None = None) -> dict:
    """GATED crawl of Layer-0 PDF attachments into data/logs/ (idempotent, resumable)."""
    if db.table_count(con, "boring_attachments") == 0:
        log.info("attachment_enum_needed")
        enumerate_bulk(con, client, config, log)

    rl = config.rate("attachments")
    limiter = RateLimiter(float(rl.get("rps", 3)))
    concurrency = int(rl.get("concurrency", 4))
    if max_downloads is None:
        max_downloads = rl.get("max_downloads_per_run")
    out_dir = ensure_dir(config.path("logs_pdf_dir"))
    ua = config.user_agent

    work = con.execute(
        """SELECT ba.objectid, ba.attachment_id, ba.download_url,
                  COALESCE(b.boring_id, 'OID-' || ba.objectid)
           FROM boring_attachments ba LEFT JOIN borings b ON b.objectid = ba.objectid
           ORDER BY ba.objectid"""
    ).fetchall()

    pending = []
    for oid, aid, url, bid in work:
        key = f"oid={oid}:aid={aid}"
        if db.manifest_is_done(con, "attachment", key):
            continue
        target = out_dir / f"{bid}__{oid}__{aid}.pdf"
        if target.exists() and target.stat().st_size > 0:
            db.manifest_mark(con, "attachment", key, "done", run_id=log.run_id,
                             bytes=target.stat().st_size)
            continue
        pending.append((key, oid, aid, url, target))
    if max_downloads:
        pending = pending[: int(max_downloads)]

    log.info("download_start", pending=len(pending), concurrency=concurrency,
             rps=rl.get("rps", 3), cap=max_downloads)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(_download_one, url, target, ua, limiter,
                      int(rl.get("max_retries", 5)), float(rl.get("backoff_base_s", 1.0)),
                      float(rl.get("backoff_max_s", 60)), float(rl.get("timeout_s", 120))):
            (key, oid, aid)
            for (key, oid, aid, url, target) in pending
        }
        for fut in as_completed(futs):
            key, oid, aid = futs[fut]
            res = fut.result()
            if res.get("ok"):
                ok += 1
                db.manifest_mark(con, "attachment", key, "done", run_id=log.run_id,
                                 bytes=res.get("bytes"), sha256=res.get("sha256"),
                                 http_status=res.get("http_status"))
            else:
                fail += 1
                db.manifest_mark(con, "attachment", key, "failed", run_id=log.run_id,
                                 http_status=res.get("http_status"))
                log.warning("download_fail", oid=oid, http=res.get("http_status"))
            if (ok + fail) % 500 == 0:
                log.info("download_progress", done=ok, failed=fail)
    log.info("download_done", downloaded=ok, failed=fail)
    return {"downloaded": ok, "failed": fail, "attempted": len(pending)}
