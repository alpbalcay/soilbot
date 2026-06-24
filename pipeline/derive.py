"""Phase 6 — geotechnical soil-equation engine (soilbot_rs).

Reads the OCR'd `strata` (depth + SPT-N + USCS + groundwater) and computes a depth-resolved
profile of corrected/derived engineering properties into `strata_derived`:
effective vertical stress, N60 / (N1)60, friction angle, relative density, undrained shear
strength, stiffness, allowable bearing, and (gated) liquefaction resistance.

All numerics run in Rust (soilbot_rs.soil_profile); this module only does DuckDB I/O and config
plumbing. Equation parameters come from the `soil_engine:` config block (unit weight / plasticity
default by USCS where lab data is absent). Nothing is fabricated — rows with no usable SPT/USCS
yield NULL derived columns.
"""
from __future__ import annotations

import json

import numpy as np

from . import db
from .config import Config

_OUT_COLS = [
    "depth_ft", "sigma_v0_tsf", "sigma_eff_v0_tsf", "gamma_pcf",
    "n60", "cn", "n1_60", "phi_peck_deg", "phi_hatanaka_deg", "dr_pct", "su_tsf",
    "e_modulus_tsf", "m_constrained_tsf", "allow_bearing_tsf", "n1_60cs", "crr",
]


def _fnum(values) -> np.ndarray:
    """Float column with NaN for None (DuckDB reads NaN back as NULL on insert from object)."""
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def build_derived(con, config: Config, log) -> dict:
    try:
        import soilbot_rs
    except ImportError as exc:  # pragma: no cover - surfaced to the operator
        raise RuntimeError(
            "soil_engine.enabled but soilbot_rs is not built. Build it with "
            "`maturin develop --release` in soilbot-rs/ (see requirements-rs.txt)."
        ) from exc

    rows = con.execute("""
        SELECT boring_id, interval_index, top_depth, bottom_depth, uscs_class,
               spt_n, gw_depth, confidence
        FROM strata
        ORDER BY boring_id, interval_index
    """).fetchall()
    if not rows:
        log.warning("no_strata")
        return {"rows": 0}

    boring_id = [r[0] for r in rows]
    interval_index = [int(r[1]) if r[1] is not None else 0 for r in rows]
    cfg_json = json.dumps(config.get("soil_engine", default={}) or {})
    out = soilbot_rs.soil_profile(
        boring_id, interval_index,
        [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows],
        [None if r[5] is None else float(r[5]) for r in rows],
        [r[6] for r in rows], [r[7] for r in rows], cfg_json)
    out = dict(out)

    n = len(boring_id)
    stage = {
        "boring_id": np.array(boring_id, dtype=object),
        "interval_index": np.array(interval_index, dtype=np.int64),
        "source": np.array(["soil_engine"] * n, dtype=object),
        "confidence": _fnum(out["confidence"]),
    }
    for c in _OUT_COLS:
        stage[c] = _fnum(out[c])

    con.execute("DELETE FROM strata_derived")
    cols = (["boring_id", "interval_index"] + _OUT_COLS + ["source", "confidence"])
    con.register("strata_derived_stage", {c: stage[c] for c in cols})
    con.execute(
        f"INSERT INTO strata_derived ({', '.join(cols)}) "
        f"SELECT {', '.join(cols)} FROM strata_derived_stage")
    con.unregister("strata_derived_stage")

    # coverage accounting (honest: how many intervals actually got each derived property)
    cov = con.execute("""
        SELECT COUNT(*) AS rows,
               COUNT(sigma_eff_v0_tsf) AS with_stress,
               COUNT(n1_60) AS with_spt,
               COUNT(phi_peck_deg) AS with_phi,
               COUNT(su_tsf) AS with_su,
               COUNT(crr) AS with_liq
        FROM strata_derived
    """).fetchone()
    stats = {"rows": cov[0], "with_stress": cov[1], "with_spt": cov[2],
             "with_phi": cov[3], "with_su": cov[4], "with_liq": cov[5]}
    log.info("strata_derived_built", **stats)
    return stats


def run(config: Config, log) -> dict:
    con = db.connect(config)
    db.bootstrap(con)
    res = build_derived(con, config, log)
    con.close()
    return res
