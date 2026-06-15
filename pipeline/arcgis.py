"""Anonymous ArcGIS REST client: layer metadata, counts, paginated queries, attachments.

Handles transport retries (429 / 5xx / connection errors) with exponential backoff + jitter,
and the ArcGIS quirk where an HTTP 200 can still carry an `{"error": {...}}` envelope.
Used by both Phase-2 extraction (FeatureServer) and Phase-4 geology (MapServer).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

from .util import RateLimiter, backoff_delay


class ArcGISError(RuntimeError):
    pass


@dataclass
class Page:
    offset: int
    raw_bytes: bytes
    parsed: dict
    n_features: int
    exceeded: bool


class ArcGISClient:
    def __init__(self, config, logger, rate_group: str = "arcgis",
                 browser_headers: bool = False):
        self.cfg = config
        self.log = logger
        r = config.rate(rate_group)
        self.max_retries = int(r.get("max_retries", 5))
        self.backoff_base = float(r.get("backoff_base_s", 1.0))
        self.backoff_max = float(r.get("backoff_max_s", 60))
        self.timeout = float(r.get("timeout_s", 60))
        self.limiter = RateLimiter(float(r.get("rps", 5)))
        self.session = requests.Session()
        self.warmup_url: str | None = None
        if browser_headers:
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            })
        else:
            self.session.headers.update({"User-Agent": config.user_agent})

    def set_warmup(self, url: str) -> None:
        """Seed WAF cookies (e.g. Imperva Incapsula) by hitting a service root before fetching."""
        self.warmup_url = url
        if url and "Referer" not in self.session.headers:
            self.session.headers["Referer"] = url.split("/arcgis/")[0] + "/"
        self._warm()

    def _warm(self) -> None:
        if not self.warmup_url:
            return
        try:
            self.session.get(self.warmup_url, params={"f": "json"}, timeout=self.timeout)
        except requests.RequestException:
            pass

    # ---- low-level GET with retry -------------------------------------------
    def get_json(self, url: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        params.setdefault("f", "json")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt, reason="conn_error", url=url)
                continue
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_exc = ArcGISError(f"HTTP {resp.status_code}")
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt, reason=f"http_{resp.status_code}", url=url)
                continue
            if resp.status_code >= 400:
                raise ArcGISError(f"HTTP {resp.status_code} for {url}: {resp.text[:300]}")
            try:
                data = resp.json()
            except (json.JSONDecodeError, ValueError):
                # HTML WAF/bot-challenge or truncated body — intermittent, so retry.
                last_exc = ArcGISError("non-JSON response (likely WAF challenge)")
                if attempt >= self.max_retries:
                    raise ArcGISError(
                        f"non-JSON after {self.max_retries} retries from {url}: {resp.text[:200]}"
                    ) from last_exc
                self._sleep(attempt, reason="non_json", url=url)
                self._warm()  # refresh WAF cookies before retrying
                continue
            if isinstance(data, dict) and "error" in data:
                # ArcGIS error envelope on a 200. Retry a few times, then raise.
                last_exc = ArcGISError(f"ArcGIS error {data['error']}")
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt, reason="arcgis_error", url=url)
                continue
            return data
        raise ArcGISError(f"GET failed after {self.max_retries} retries: {url}") from last_exc

    def get_raw(self, url: str, params: dict) -> tuple[bytes, dict]:
        """GET returning (raw_bytes, parsed_json) — used for geojson pages we persist verbatim."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt, reason="conn_error", url=url)
                continue
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_exc = ArcGISError(f"HTTP {resp.status_code}")
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt, reason=f"http_{resp.status_code}", url=url)
                continue
            if resp.status_code >= 400:
                raise ArcGISError(f"HTTP {resp.status_code} for {url}: {resp.text[:300]}")
            try:
                parsed = resp.json()
            except (json.JSONDecodeError, ValueError):
                last_exc = ArcGISError("non-JSON response (likely WAF challenge)")
                if attempt >= self.max_retries:
                    raise ArcGISError(
                        f"non-JSON after {self.max_retries} retries from {url}: {resp.text[:200]}"
                    ) from last_exc
                self._sleep(attempt, reason="non_json", url=url)
                self._warm()  # refresh WAF cookies before retrying
                continue
            if isinstance(parsed, dict) and "error" in parsed:
                last_exc = ArcGISError(f"ArcGIS error {parsed['error']}")
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt, reason="arcgis_error", url=url)
                continue
            return resp.content, parsed
        raise ArcGISError(f"GET failed after {self.max_retries} retries: {url}") from last_exc

    def _sleep(self, attempt: int, reason: str, url: str) -> None:
        delay = backoff_delay(attempt, self.backoff_base, self.backoff_max)
        self.log.warning("retry", reason=reason, attempt=attempt + 1, sleep_s=round(delay, 2),
                         url=url.rsplit("/", 2)[-2:] and "/".join(url.rsplit("/", 2)[-2:]))
        time.sleep(delay)

    # ---- metadata / counts ---------------------------------------------------
    def layer_metadata(self, layer_url: str) -> dict:
        return self.get_json(layer_url, {"f": "json"})

    def feature_count(self, layer_url: str, where: str = "1=1") -> int:
        data = self.get_json(layer_url + "/query",
                             {"where": where, "returnCountOnly": "true", "f": "json"})
        return int(data.get("count", 0))

    def list_services(self) -> dict:
        return self.get_json(self.cfg.feature_server, {"f": "json"})

    # ---- feature pages (geojson, persisted verbatim) -------------------------
    def query_page(self, layer_url: str, offset: int, page_size: int,
                   where: str = "1=1", out_fields: str = "*", fmt: str = "geojson",
                   out_sr: int | None = None) -> Page:
        params = {
            "where": where,
            "outFields": out_fields,
            "f": fmt,
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "returnGeometry": "true",
        }
        if out_sr is not None:
            params["outSR"] = out_sr
        raw, parsed = self.get_raw(layer_url + "/query", params)
        feats = parsed.get("features", []) if isinstance(parsed, dict) else []
        exceeded = bool(
            parsed.get("exceededTransferLimit")
            or parsed.get("properties", {}).get("exceededTransferLimit")
        )
        return Page(offset=offset, raw_bytes=raw, parsed=parsed,
                    n_features=len(feats), exceeded=exceeded)

    def iter_pages(self, layer_url: str, page_size: int, total: int | None = None,
                   where: str = "1=1", out_fields: str = "*", fmt: str = "geojson",
                   skip_offsets: set[int] | None = None,
                   out_sr: int | None = None) -> Iterator[Page]:
        """Yield pages by resultOffset until exhausted. Bounded by `total` when known.

        Pages whose offset is in `skip_offsets` are not fetched (resumability); the
        generator still advances past them.
        """
        skip = skip_offsets or set()
        offset = 0
        while True:
            if total is not None and offset >= total:
                break
            if offset in skip:
                offset += page_size
                continue
            page = self.query_page(layer_url, offset, page_size, where, out_fields, fmt, out_sr)
            yield page
            if page.n_features == 0:
                break
            if total is None and not page.exceeded and page.n_features < page_size:
                break
            offset += page_size

    # ---- attachments ---------------------------------------------------------
    @staticmethod
    def supports_query_attachments(meta: dict) -> bool:
        if meta.get("supportsQueryAttachments"):
            return True
        adv = meta.get("advancedQueryCapabilities", {}) or {}
        return bool(adv.get("supportsQueryAttachments"))

    def attachments_for_oids(self, layer_url: str, oids: list[int]) -> list[dict]:
        """Return attachmentGroups for a batch of object ids via /queryAttachments.

        This server rejects the `where=1=1` bulk form ("Unable to get feature attachments"),
        but accepts an explicit `objectIds` list, so the caller batches ids.
        """
        ids = ",".join(str(o) for o in oids)
        data = self.get_json(layer_url + "/queryAttachments",
                             {"objectIds": ids, "f": "json"})
        return data.get("attachmentGroups", []) or []

    def list_attachments_for(self, layer_url: str, objectid: int) -> list[dict]:
        """Per-feature attachment listing (fallback when bulk query is unsupported)."""
        data = self.get_json(f"{layer_url}/{objectid}/attachments", {"f": "json"})
        return data.get("attachmentInfos", []) or []
