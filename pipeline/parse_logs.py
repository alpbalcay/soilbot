"""Phase 3 (scaffold) — extract stratigraphy from scanned boring-log PDFs.

STATUS: SCAFFOLD. The NJDOT logs are scanned raster images, so the only path to real
stratigraphy is OCR — and no OCR engine is installed here. This module ships:
  * a working layout structure (interface + DB plumbing + idempotency),
  * a pdfplumber vector-text branch (works on the rare PDF that carries selectable text),
  * a poppler-rasterize + pluggable OCR seam (default backend records ocr_status='pending'),
  * a regex strata extractor with worked examples on synthetic text.

ACCURACY IS A TODO. Real intervals are written to `strata` ONLY when genuinely parsed; we
never fabricate. Per-PDF status (parsed / pending / failed) is tracked in the manifest so
REPORT.md can quote OCR coverage honestly.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pdfplumber

from . import db
from .config import Config
from .util import ensure_dir

# USCS group symbols (ASTM D2487). Used to recognize a classification token on a line.
USCS_CODES = {
    "GW", "GP", "GM", "GC", "SW", "SP", "SM", "SC", "ML", "CL", "OL",
    "MH", "CH", "OH", "PT", "GW-GM", "GP-GM", "SW-SM", "SP-SM", "SC-SM",
}

# Regexes — preliminary; tuned per real OCR output during the (TODO) accuracy pass.
_RE_DEPTH = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*[-–to]{1,3}\s*(\d{1,3}(?:\.\d+)?)")
_RE_USCS = re.compile(r"\b(GW-GM|GP-GM|SW-SM|SP-SM|SC-SM|GW|GP|GM|GC|SW|SP|SM|SC|ML|CL|OL|MH|CH|OH|PT)\b")
_RE_SPT = re.compile(r"\b(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})\b")  # blows per 6": a-b-c
_RE_NVAL = re.compile(r"\bN\s*[=:]\s*(\d{1,3})\b", re.IGNORECASE)
_RE_GW = re.compile(r"(?:groundwater|water\s+(?:table|level|encountered|at)|GWT)\D{0,15}(\d{1,3}(?:\.\d+)?)",
                    re.IGNORECASE)
_RE_SAMPLE = re.compile(r"\b(SS|ST|SPT|AU|CO|UD|GP|NQ|HQ|Shelby|Split[\s-]?spoon)\b", re.IGNORECASE)


@dataclass
class StrataRow:
    interval_index: int
    top_depth: Optional[float] = None
    bottom_depth: Optional[float] = None
    uscs_class: Optional[str] = None
    spt_n: Optional[int] = None
    sample_type: Optional[str] = None
    gw_depth: Optional[float] = None
    elevation: Optional[float] = None
    source: str = "pdfplumber"
    ocr_status: str = "parsed"
    confidence: float = 0.0


@dataclass
class ParseResult:
    boring_id: str
    rows: list[StrataRow] = field(default_factory=list)
    status: str = "pending"      # 'parsed' | 'pending' | 'failed'
    source: str = "none"         # 'pdfplumber' | 'ocr' | 'none'
    note: str = ""


# ---- OCR backend seam ------------------------------------------------------
OCRBackend = Callable[[Path], str]  # rasterized page image -> recognized text


def tesseract_backend(image_path: Path) -> str:
    """OCR one page image via the tesseract CLI (if installed). TODO: tune psm/oem + preprocess."""
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract not installed")
    out = subprocess.run(["tesseract", str(image_path), "stdout", "--psm", "6"],
                         capture_output=True, text=True, timeout=120)
    return out.stdout


def _rasterize(pdf_path: Path, dpi: int = 200) -> list[Path]:
    """Rasterize a PDF to PNG page images via poppler's pdftoppm (present on this box)."""
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm (poppler) not installed")
    tmp = ensure_dir(pdf_path.parent / "_raster")
    prefix = tmp / pdf_path.stem
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
                   check=True, capture_output=True, timeout=180)
    return sorted(tmp.glob(f"{pdf_path.stem}*.png"))


# ---- regex extraction ------------------------------------------------------
def parse_text_to_strata(text: str, source: str = "pdfplumber") -> list[StrataRow]:
    """Heuristic line-by-line extraction. Preliminary; accuracy is a TODO.

    Emits a StrataRow only for lines that carry a depth interval (the anchor), attaching any
    USCS / SPT-N / sample / groundwater signal found on the same line.
    """
    rows: list[StrataRow] = []
    last_gw: Optional[float] = None
    idx = 0
    for line in text.splitlines():
        gw_m = _RE_GW.search(line)
        if gw_m:
            try:
                last_gw = float(gw_m.group(1))
            except ValueError:
                pass
        depth_m = _RE_DEPTH.search(line)
        if not depth_m:
            continue
        top, bot = float(depth_m.group(1)), float(depth_m.group(2))
        if bot < top or bot - top > 100:  # implausible interval -> skip (likely a false match)
            continue
        uscs_m = _RE_USCS.search(line)
        spt_m = _RE_SPT.search(line)
        nval_m = _RE_NVAL.search(line)
        sample_m = _RE_SAMPLE.search(line)
        spt_n: Optional[int] = None
        if spt_m:
            spt_n = int(spt_m.group(2)) + int(spt_m.group(3))  # N = sum of last two 6" increments
        elif nval_m:
            spt_n = int(nval_m.group(1))
        signals = sum(x is not None for x in (uscs_m, spt_n, sample_m))
        rows.append(StrataRow(
            interval_index=idx, top_depth=top, bottom_depth=bot,
            uscs_class=(uscs_m.group(1) if uscs_m else None),
            spt_n=spt_n, sample_type=(sample_m.group(1) if sample_m else None),
            gw_depth=last_gw, source=source, ocr_status="parsed",
            confidence=round(min(1.0, 0.3 + 0.2 * signals), 2),
        ))
        idx += 1
    return rows


def extract_from_pdf(pdf_path: Path, boring_id: str,
                     ocr_backend: Optional[OCRBackend] = None) -> ParseResult:
    """Try vector text first; fall back to rasterize+OCR. Records 'pending' if no OCR backend."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as exc:  # noqa: BLE001
        return ParseResult(boring_id, status="failed", note=f"pdfplumber: {exc}")

    if text and len(text.strip()) > 40:  # genuine selectable text
        rows = parse_text_to_strata(text, source="pdfplumber")
        if rows:
            return ParseResult(boring_id, rows=rows, status="parsed", source="pdfplumber")
        return ParseResult(boring_id, status="pending", source="pdfplumber",
                           note="vector text present but no strata pattern matched (TODO: tune regex)")

    # Scanned raster: OCR required.
    if ocr_backend is None:
        return ParseResult(boring_id, status="pending", source="none",
                           note="scanned image; OCR backend not configured")
    try:
        images = _rasterize(pdf_path)
        ocr_text = "\n".join(ocr_backend(img) for img in images)
    except Exception as exc:  # noqa: BLE001
        return ParseResult(boring_id, status="failed", note=f"ocr: {exc}")
    rows = parse_text_to_strata(ocr_text, source="ocr")
    if rows:
        return ParseResult(boring_id, rows=rows, status="parsed", source="ocr")
    return ParseResult(boring_id, status="pending", source="ocr",
                       note="OCR produced no strata pattern (TODO: accuracy)")


# ---- DB plumbing -----------------------------------------------------------
def _insert_rows(con, boring_id: str, rows: list[StrataRow]) -> None:
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM strata WHERE boring_id = ?", [boring_id])
        if rows:
            con.executemany(
                """INSERT INTO strata
                   (boring_id, interval_index, top_depth, bottom_depth, uscs_class, spt_n,
                    sample_type, gw_depth, elevation, source, ocr_status, confidence)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [[boring_id, r.interval_index, r.top_depth, r.bottom_depth, r.uscs_class,
                  r.spt_n, r.sample_type, r.gw_depth, r.elevation, r.source, r.ocr_status,
                  r.confidence] for r in rows],
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def _boring_id_from_filename(name: str) -> str:
    # data/logs/{boring_id}__{oid}__{aid}.pdf
    return name.split("__", 1)[0]


def run(config: Config, log, ocr: bool = False, limit: Optional[int] = None) -> dict:
    """Parse downloaded logs into the strata table. Requires --download-logs to have run."""
    con = db.connect(config)
    db.bootstrap(con)
    logs_dir = config.path("logs_pdf_dir")
    pdfs = sorted(logs_dir.glob("*.pdf")) if logs_dir.exists() else []
    if not pdfs:
        log.warning("no_logs", note="data/logs/ is empty; run `--phase 3 --download-logs` first")
        return {"pdfs": 0, "parsed": 0, "pending": 0, "failed": 0}

    backend: Optional[OCRBackend] = None
    if ocr:
        if shutil.which("tesseract"):
            backend = tesseract_backend
        else:
            log.warning("ocr_unavailable", note="tesseract not installed; logs will stay 'pending'")

    if limit:
        pdfs = pdfs[: int(limit)]
    parsed = pending = failed = 0
    for pdf in pdfs:
        bid = _boring_id_from_filename(pdf.name)
        key = f"parse:{bid}"
        if db.manifest_is_done(con, "parse", key) and not ocr:
            continue
        res = extract_from_pdf(pdf, bid, ocr_backend=backend)
        if res.status == "parsed":
            _insert_rows(con, bid, res.rows)
            parsed += 1
            db.manifest_mark(con, "parse", key, "done", run_id=log.run_id, rows_out=len(res.rows))
        elif res.status == "pending":
            pending += 1
            db.manifest_mark(con, "parse", key, "pending", run_id=log.run_id)
        else:
            failed += 1
            db.manifest_mark(con, "parse", key, "failed", run_id=log.run_id)
            log.warning("parse_failed", boring_id=bid, note=res.note)
    log.info("parse_done", pdfs=len(pdfs), parsed=parsed, pending=pending, failed=failed)
    con.close()
    return {"pdfs": len(pdfs), "parsed": parsed, "pending": pending, "failed": failed}


# Synthetic worked example (runs without any downloaded PDF) — demonstrates the extractor.
EXAMPLE_LOG_TEXT = """\
BORING LOG B-12   Surface Elevation: 42.5 ft
Depth (ft)  Sample  Blows/6in  N   USCS  Description
0.0 - 2.0   SS      3-4-5      9   SM    Brown silty SAND, moist
2.0 - 4.0   SS      5-7-9      16  SM    Brown silty SAND with gravel
4.0 - 8.5   SS      8-11-14    25  SC    Reddish clayey SAND
Groundwater encountered at 6.5 ft during drilling
8.5 - 12.0  ST      N=22           CL    Stiff gray CLAY
"""


def worked_examples() -> list[StrataRow]:
    return parse_text_to_strata(EXAMPLE_LOG_TEXT)
