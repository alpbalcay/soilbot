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


# ---- easyocr backend (GPU; tesseract unavailable without sudo on this box) ----
_EASYOCR_READER = None


def _get_easyocr():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr  # heavy import; deferred
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _EASYOCR_READER


def easyocr_boxes(image_path: Path) -> list[tuple]:
    """Return positioned OCR boxes [(x, y, text, conf)] for layout-aware parsing."""
    reader = _get_easyocr()
    out = []
    for bbox, text, conf in reader.readtext(str(image_path), detail=1, paragraph=False):
        x = float(min(p[0] for p in bbox)); y = float(min(p[1] for p in bbox))
        out.append((x, y, text, float(conf)))
    return out


def easyocr_backend(image_path: Path) -> str:
    """Flattened reading-order text (for the regex path / debugging)."""
    boxes = sorted(easyocr_boxes(image_path), key=lambda b: (round(b[1] / 15), b[0]))
    return "\n".join(b[2] for b in boxes)


def _rasterize(pdf_path: Path, max_px: int = 3400) -> list[Path]:
    """Rasterize a PDF to PNG page images via poppler's pdftoppm, bounding the longest side to
    `max_px`. Using -scale-to (not a fixed -r dpi) means pdftoppm NEVER builds an oversized
    intermediate: NJDOT has a few 36x31-inch plan sheets that at 300 dpi rasterize to 100M+
    pixel PNGs that exhaust 12 GB VRAM and wedge easyocr. A normal letter page -> ~2630x3400,
    effectively ~300 dpi, so OCR quality is unchanged."""
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm (poppler) not installed")
    tmp = ensure_dir(pdf_path.parent / "_raster")
    prefix = tmp / pdf_path.stem
    subprocess.run(["pdftoppm", "-png", "-scale-to", str(max_px), str(pdf_path), str(prefix)],
                   check=True, capture_output=True, timeout=180)
    return sorted(tmp.glob(f"{pdf_path.stem}*.png"))


def _cleanup_rasters(pdf_path: Path) -> None:
    """Delete this PDF's rasterized pages — the _raster dir otherwise grows unbounded (5+ GB)."""
    tmp = pdf_path.parent / "_raster"
    if tmp.exists():
        for f in tmp.glob(f"{pdf_path.stem}*.png"):
            try:
                f.unlink()
            except OSError:
                pass


def _prep_image(img_path: Path, max_dim: int = 3400):
    """Cap oversized scans before OCR -> bounds easyocr GPU memory + runtime. Normal 300-dpi
    letter pages (~2544x3300) are under the cap and pass through untouched; only the rare huge/
    high-res sheets (some reach 100M+ pixels and exhaust 12 GB VRAM) get downscaled. Returns
    (path-to-use, width, height)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # local trusted files; disable the decompression-bomb guard
    with Image.open(img_path) as im:
        w, h = im.size
        if max(w, h) <= max_dim:
            return img_path, w, h
        scale = max_dim / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        ds = im.convert("RGB").resize((nw, nh))
    out = img_path.with_suffix(".ds.png")
    ds.save(out)
    return out, nw, nh


# ---- layout-aware extraction from positioned OCR boxes ---------------------
# NJDOT logs are descriptive (e.g. "moist, m.-f. SAND, some silt, [FILL]"), not clean USCS
# codes, so we map the dominant soil noun + modifier to a coarse USCS group. Approximate by
# construction — confidence reflects that. Honest: this is a description->USCS heuristic.
_RE_PRIMARY_SOIL = re.compile(r"\b(GRAVEL|SAND|SILT|CLAY|PEAT|TOPSOIL|FILL)\b")
_RE_ELEV = re.compile(r"SURFACE\s+ELEVATION\s*[:.]?\s*(-?\d{1,4}(?:\.\d+)?)\s*(m|ft|meters|feet)?",
                      re.IGNORECASE)
_RE_SAMPLE_ID = re.compile(r"\b([SU]-?\d{1,2}|SS-?\d{1,2}|ST-?\d{1,2})\b")
_RE_LEFT_NUM = re.compile(r"^[~\-−]?\s*(\d{1,3})$")


def soil_family_from_text(desc: str) -> tuple[Optional[str], float]:
    """Map a descriptive soil phrase to a USCS group symbol + a confidence in [0,1].

    Reads the gradation/plasticity/organic adjectives the logs do state so the heuristic can emit the
    well-graded (W), high-plasticity (H), and organic (O) classes — not just the poorly-graded/
    low-plasticity defaults. Without this, "well graded"->W, "fat"/"high plasticity"->H and
    "organic"->O are all collapsed to P/L/PT, which is why our strata had CH=1 and zero MH/SW/GW/OH
    (see CLASSIFICATION_KNOWLEDGE.md). Suffix is only assigned when the description states it; absent
    a gradation cue we still default to P/L, honestly reflecting unstated gradation."""
    up = desc.upper()
    if "FILL" in up:
        return "FILL", 0.6
    if "PEAT" in up or "MUCK" in up:
        return "PT", 0.6
    # descriptive cues for the suffix/prefix the base heuristic used to drop
    high_plast = "FAT" in up or "HIGH PLAST" in up or "HIGH-PLAST" in up or "PLASTIC CLAY" in up
    organic = "ORGANIC" in up
    well_graded = "WELL GRADED" in up or "WELL-GRADED" in up
    if "TOPSOIL" in up:
        return "OL", 0.5
    m = _RE_PRIMARY_SOIL.search(up)
    if not m:
        return ("OL", 0.4) if organic else (None, 0.0)
    primary = m.group(1)
    silty = "SILT" in up or "SILTY" in up
    clayey = "CLAY" in up or "CLAYEY" in up
    if primary == "SAND":
        if silty:
            return "SM", 0.45
        if clayey:
            return "SC", 0.45
        return ("SW", 0.5) if well_graded else ("SP", 0.45)
    if primary == "GRAVEL":
        if silty:
            return "GM", 0.45
        if clayey:
            return "GC", 0.45
        return ("GW", 0.5) if well_graded else ("GP", 0.45)
    if primary == "SILT":
        if organic:
            return ("OH" if high_plast else "OL"), 0.5
        return ("MH" if high_plast else "ML"), 0.45
    if primary == "CLAY":
        if organic:
            return ("OH" if high_plast else "OL"), 0.5
        return ("CH" if high_plast else "CL"), 0.45
    return None, 0.0


def _nums_in(text: str) -> list[float]:
    """All numbers in a text box, treating comma as decimal (OCR writes 1.5 as '1,5') and
    stripping stray separators like '4|2' -> [4, 2]."""
    out = []
    for tok in re.split(r"[|/\s]+", text.replace(",", ".")):
        tok = tok.strip(".:-")
        if re.fullmatch(r"\d{1,3}(?:\.\d+)?", tok):
            out.append(float(tok))
    return out


def _spt_n_from_increments(incs: list[float]) -> Optional[int]:
    """SPT N = sum of the 2nd and 3rd 6-inch increments (ASTM D1586)."""
    ints = [int(round(v)) for v in incs if v < 100]   # blow counts; drop recovery-like values
    if len(ints) >= 3:
        return ints[1] + ints[2]
    if len(ints) == 2:
        return ints[0] + ints[1]
    if len(ints) == 1:
        return ints[0]
    return None


# refusal: a high blow count over a partial penetration, e.g. '50/125', '100/3"', '50/25mm'.
_RE_REFUSAL = re.compile(r"\b(\d{2,3})\s*/\s*(\d{1,3})(?:\s*(?:mm|cm|in|\"|'))?\b")
# weight-of-rod / weight-of-hammer (N≈0). easyocr mangles 'WOR'/'WOH' -> 'Woria','Wori','Woai'.
_RE_WOR = re.compile(r"\bW[\.\s]*O[\.\s]*[RHIA]|WEIGHT\s+OF\s+(?:ROD|HAMMER)", re.IGNORECASE)


def _spt_from_blows(blows_text: str, band_text: str) -> tuple[Optional[int], str]:
    """Parse the 'Blows on Spoon' field -> (spt_n, kind). kind ∈ {n, wor, refusal, none}.

    Handles the three things the naive increment-sum got wrong on real logs:
      * WOR/WOH (very soft, N≈0) — detected on the whole sample band (OCR mis-columns it),
      * refusal '50/x' / '100/y' — emit NULL N (don't fabricate the bogus low N the old code did),
      * recovery leakage — only integer blow counts in [0,99] count as increments (drop decimals
        like 0.11 ft and mm values >=100).
    """
    if _RE_WOR.search(band_text):
        return 0, "wor"
    mr = _RE_REFUSAL.search(blows_text)
    if mr and int(mr.group(1)) >= 50:
        return None, "refusal"
    incs = []
    for tok in re.split(r"[|/\s]+", blows_text):
        tok = tok.strip(".:-")
        if re.fullmatch(r"\d{1,3}", tok):          # integer blow count only (no recovery decimals)
            v = int(tok)
            if v < 100:
                incs.append(v)
    n = _spt_n_from_increments(incs)
    return n, ("n" if n is not None else "none")


def _norm_sid(sid: str) -> str:
    """Normalize a sample id to letter+digits, mapping a leading OCR-confused digit to 'S' (the
    dominant split-spoon prefix): '5-1'->'S1', '8-3'->'S3', 'S-4'->'S4', 'J8'->'J8'."""
    s = sid.strip()
    if s and s[0].isdigit():          # leading 'S' misread as a digit -> restore it
        s = "S" + s[1:]
    return s.replace("-", "")


def parse_spoon_format(boxes: list[tuple], w: float, h: float) -> tuple[list[StrataRow], dict]:
    """Parser for the NJDOT 'Blows on Spoon' split-spoon log format (rich: explicit per-sample
    depth intervals + SPT blow counts). Detected by its header anchors. Columns (fractions of
    page width): sample-id ~0.20, sample depth top/bottom ~0.26/0.33, blows-on-spoon ~0.39-0.52,
    recovery ~0.54, soil description ~0.58+.
    """
    flat = "\n".join(b[2] for b in sorted(boxes, key=lambda b: (round(b[1] / 15), b[0])))
    # units: these logs are often metric (e.g. "0.15 m Concrete"); convert depths to feet.
    metric = bool(re.search(r"\d\s*m\b", flat)) or "NAVD 88" in flat
    to_ft = 3.280839895 if metric else 1.0
    header = extract_header_fields(boxes, flat)
    # ground elevation (this format labels it 'GROUND ELEVATION')
    ge = re.search(r"GROUND\s+ELEVATION\s*[:.;]?\s*(-?\d{1,4}(?:[.,]\d+)?)", flat, re.IGNORECASE)
    if ge:
        try:
            header["surface_elevation"] = float(ge.group(1).replace(",", "."))
        except ValueError:
            pass

    # Anchor rows on sample ids in the sample column. easyocr routinely misreads the leading 'S'
    # as a digit ('S-1'->'5-1', 'S-3'->'8-3'), which the old letter-only regex dropped — the main
    # recall hole. Accept either a letter-led id ('S2', 'J8') OR any dash-form id ('5-1', '8-3'),
    # but NOT a bare 1-2 digit casing-blow count ('81'), which has no dash and no letter.
    _re_sid = re.compile(r"^([A-Z]-?\d{1,2}|[A-Z0-9]-\d{1,2})$")
    samples = [(x, y, _norm_sid(t.strip())) for (x, y, t, c) in boxes
               if 0.15 * w < x < 0.25 * w and _re_sid.match(t.strip())]
    samples.sort(key=lambda s: s[1])
    rows: list[StrataRow] = []
    for i, (sx, sy, sid) in enumerate(samples):
        band = [b for b in boxes if abs(b[1] - sy) < 70]
        # WOR/WOH lives in the sample/blows columns; exclude the description column (x>0.56w) so a
        # stray description word can't trigger a false N=0.
        band_text = " ".join(b[2] for b in band if b[0] < 0.56 * w)
        depths = sorted([(b[0], v) for b in band for v in _nums_in(b[2])
                         if 0.24 * w < b[0] < 0.36 * w and v < 100], key=lambda z: z[0])
        top = depths[0][1] * to_ft if depths else None
        bot = depths[1][1] * to_ft if len(depths) > 1 else None
        # blows: the 'Blows on Spoon' column (exclude recovery at x>0.54w); WOR/refusal aware.
        # Left edge 0.35w (not 0.37) — the first 6in increment often sits just right of the depth
        # column and was being dropped, biasing N low (single-increment errors).
        blows_text = " ".join(b[2] for b in sorted(band, key=lambda b: b[0])
                              if 0.35 * w < b[0] < 0.54 * w)
        spt, spt_kind = _spt_from_blows(blows_text, band_text)
        desc = " ".join(b[2] for b in sorted(band, key=lambda b: b[0]) if b[0] > 0.57 * w)
        fam, fam_conf = soil_family_from_text(desc)
        # confidence: a clean 3-increment N is trusted most; a 1-2 increment or WOR less; a row
        # with no depth or no/derived N least. (The old formula was ~always 1.0 — useless.)
        n_inc = len(re.findall(r"\b\d{1,3}\b", blows_text))
        n_conf = 0.9 if spt_kind == "n" and n_inc >= 3 else \
            0.6 if spt_kind in ("n", "wor") else 0.3 if spt_kind == "refusal" else 0.2
        conf = 0.15 + 0.25 * (top is not None) + 0.6 * n_conf
        rows.append(StrataRow(
            interval_index=i,
            top_depth=round(top, 2) if top is not None else None,
            bottom_depth=round(bot, 2) if bot is not None else None,
            uscs_class=fam, spt_n=spt, sample_type=sid.replace("-", ""),
            gw_depth=header["gw_depth"], elevation=header["surface_elevation"],
            source="ocr", ocr_status="parsed", confidence=round(min(1.0, conf), 2),
        ))
    return rows, header


def is_spoon_format(boxes: list[tuple]) -> bool:
    flat = " ".join(b[2] for b in boxes).upper()
    return "BLOWS ON SPOON" in flat or ("ON SPOON" in flat and "STRATIF" in flat)


def extract_header_fields(boxes: list[tuple], flat_text: str) -> dict:
    """Surface elevation + groundwater depth from the header / water-levels area."""
    out = {"surface_elevation": None, "elev_units": None, "gw_depth": None}
    m = _RE_ELEV.search(flat_text)
    if m:
        try:
            out["surface_elevation"] = float(m.group(1))
            out["elev_units"] = (m.group(2) or "").lower() or None
        except ValueError:
            pass
    gw = _RE_GW.search(flat_text)
    if gw:
        try:
            out["gw_depth"] = float(gw.group(1))
        except ValueError:
            pass
    return out


def _depth_axis(boxes: list[tuple], page_w: float):
    """Fit depth = a*y + b from the left-margin numeric tick boxes. Returns (a, b) or None."""
    import numpy as np
    pts = []
    for x, y, text, conf in boxes:
        if x < 0.13 * page_w:
            mm = _RE_LEFT_NUM.match(text.strip())
            if mm:
                v = int(mm.group(1))
                if 0 <= v < 300:
                    pts.append((y, v))
    if len(pts) < 3:
        return None
    pts.sort()
    ys = np.array([p[0] for p in pts], float)
    vs = np.array([p[1] for p in pts], float)
    # robust-ish: require monotone increase of depth with y
    if not (vs[-1] > vs[0]):
        return None
    a, b = np.polyfit(ys, vs, 1)
    if a <= 0:
        return None
    return float(a), float(b)


def parse_boxes_to_strata(boxes: list[tuple], page_w: float, page_h: float) -> list[StrataRow]:
    """Layout-aware extraction: associate description rows with interpolated depths + soil class.

    Honest scope: yields one StrataRow per OCR'd description line that carries a soil keyword,
    with a depth from the calibrated left-axis. USCS is a description heuristic; SPT-N is parsed
    only where explicit blow triplets appear. Many rows will have partial fields — that is real.
    """
    flat = "\n".join(b[2] for b in sorted(boxes, key=lambda b: (round(b[1] / 15), b[0])))
    header = extract_header_fields(boxes, flat)
    axis = _depth_axis(boxes, page_w)
    # description column = right portion of the page (observed x>~0.42*width for the desc text)
    desc_boxes = [(x, y, t, c) for (x, y, t, c) in boxes if x > 0.42 * page_w and len(t) > 8]
    desc_boxes.sort(key=lambda b: b[1])
    rows: list[StrataRow] = []
    idx = 0
    for x, y, text, conf in desc_boxes:
        fam, fam_conf = soil_family_from_text(text)
        if fam is None:
            continue
        depth = (axis[0] * y + axis[1]) if axis else None
        spt = None
        sm = _RE_SPT.search(text)
        if sm:
            spt = int(sm.group(2)) + int(sm.group(3))
        rows.append(StrataRow(
            interval_index=idx,
            top_depth=round(depth, 2) if depth is not None else None,
            bottom_depth=None,
            uscs_class=fam, spt_n=spt, sample_type=None,
            gw_depth=header["gw_depth"], elevation=header["surface_elevation"],
            source="ocr", ocr_status="parsed",
            confidence=round(min(1.0, 0.4 * conf + 0.6 * fam_conf), 2),
        ))
        idx += 1
    # fill bottom_depth from the next row's top (intervals are contiguous on a log)
    for i in range(len(rows) - 1):
        if rows[i].top_depth is not None and rows[i + 1].top_depth is not None:
            rows[i].bottom_depth = rows[i + 1].top_depth
    return rows


# ---- regex extraction (vector-text path / synthetic examples) --------------
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


def extract_with_easyocr(pdf_path: Path, boring_id: str) -> ParseResult:
    """Rasterize + easyocr + layout-aware parse. Coarse soil-class + elevation/GW; depth/SPT
    are sparse (descriptive scanned logs). Status 'parsed' iff ≥1 soil row recovered."""
    try:
        images = _rasterize(pdf_path, max_px=3400)  # bounded at the source (no VRAM blowups)
    except Exception as exc:  # noqa: BLE001
        return ParseResult(boring_id, status="failed", note=f"rasterize: {exc}")
    all_rows: list[StrataRow] = []
    fmt = "generic"
    spoon = False
    try:
        # OCR page 1, detect format. Only the rich 'Blows on Spoon' logs are worth OCR'ing
        # further pages (deep borings continue across pages); non-spoon multi-page documents
        # (plan sheets, notes) are stopped after page 1 — OCR is the cost, so skipping pages
        # 2+ on non-spoon docs is the main throughput win in heavy regions.
        for i, img in enumerate(images[:4]):
            p, w, h = _prep_image(img)  # safety net if a page is still oversized
            boxes = easyocr_boxes(p)
            if i == 0:
                spoon = is_spoon_format(boxes)
                fmt = "spoon" if spoon else "generic"
            if spoon:
                rows, _ = parse_spoon_format(boxes, w, h)
                all_rows.extend(rows)
            else:
                all_rows.extend(parse_boxes_to_strata(boxes, w, h))
                break  # non-spoon: don't OCR remaining pages
    except Exception as exc:  # noqa: BLE001
        return ParseResult(boring_id, status="failed", note=f"easyocr: {exc}")
    finally:
        _cleanup_rasters(pdf_path)
    for i, r in enumerate(all_rows):
        r.interval_index = i
    if all_rows:
        return ParseResult(boring_id, rows=all_rows, status="parsed", source="ocr")
    return ParseResult(boring_id, status="pending", source="ocr",
                       note="OCR produced no soil rows")


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

    use_easyocr = False
    backend: Optional[OCRBackend] = None
    if ocr:
        try:
            import easyocr  # noqa: F401
            use_easyocr = True
        except ImportError:
            if shutil.which("tesseract"):
                backend = tesseract_backend
            else:
                log.warning("ocr_unavailable",
                            note="no easyocr / tesseract; logs will stay 'pending'")

    if limit:
        pdfs = pdfs[: int(limit)]
    parsed = pending = failed = 0
    for pdf in pdfs:
        bid = _boring_id_from_filename(pdf.name)
        key = f"parse:{bid}"
        # Resumable: skip borings already parsed (lets OCR accumulate across runs/restarts).
        if db.manifest_is_done(con, "parse", key):
            continue
        if use_easyocr:
            res = extract_with_easyocr(pdf, bid)
        else:
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
