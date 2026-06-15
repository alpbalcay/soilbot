"""Small, dependency-light helpers shared across the pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Write to a temp sibling then os.replace, so readers never see a partial file."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def load_json_file(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter (attempt is 0-based)."""
    raw = min(cap, base * (2 ** attempt))
    return raw * (0.5 + random.random() * 0.5)


class RateLimiter:
    """Thread-safe minimum-interval limiter: >= 1/rps seconds between acquire() calls."""

    def __init__(self, rps: float):
        self.min_interval = (1.0 / rps) if rps and rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.min_interval


def page_filename(offset: int) -> str:
    """Stable, zero-padded raw-page filename so listings sort by offset."""
    return f"page_{offset:08d}.geojson"
