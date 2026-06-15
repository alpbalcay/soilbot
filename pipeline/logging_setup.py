"""Structured JSON-line logging.

Each event is one JSON object per line in the per-phase log file (logs/extract.log,
download.log, parse.log), mirrored at INFO+ to stderr in a terse human form. A run_id
ties all events of one invocation together (and is stamped into the DuckDB manifest).
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .util import ensure_dir

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
}


def new_run_id() -> str:
    return "r-" + uuid.uuid4().hex[:8]


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        soil = record.__dict__.get("soilbot")
        if isinstance(soil, dict):
            payload.update(soil)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        soil = record.__dict__.get("soilbot", {})
        extras = {k: v for k, v in soil.items() if k not in ("run_id", "phase")}
        tail = " ".join(f"{k}={v}" for k, v in extras.items())
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return f"{ts} {record.levelname[0]} {record.getMessage()}" + (f"  {tail}" if tail else "")


class EventLogger:
    """Ergonomic wrapper: log.info('page_done', layer=0, offset=4000, rows=2000)."""

    def __init__(self, logger: logging.Logger, run_id: str, phase: str):
        self._log = logger
        self.run_id = run_id
        self.phase = phase

    def _emit(self, level: int, event: str, fields: dict) -> None:
        soil = {"run_id": self.run_id, "phase": self.phase}
        soil.update(fields)
        self._log.log(level, event, extra={"soilbot": soil})

    def info(self, event: str, **fields) -> None:
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, **fields) -> None:
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, **fields) -> None:
        self._emit(logging.ERROR, event, fields)

    def exception(self, event: str, **fields) -> None:
        self._log.log(logging.ERROR, event, exc_info=True, extra={
            "soilbot": {"run_id": self.run_id, "phase": self.phase, **fields}})


def setup(log_dir: Path, log_filename: str, run_id: str, phase: str,
          console: bool = True) -> EventLogger:
    ensure_dir(log_dir)
    logger = logging.getLogger(f"soilbot.{phase}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fh = logging.FileHandler(Path(log_dir) / log_filename, encoding="utf-8")
    fh.setFormatter(_JsonFormatter())
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(_ConsoleFormatter())
        logger.addHandler(ch)

    return EventLogger(logger, run_id, phase)
