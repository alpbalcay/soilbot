"""Literature-derived per-boring geotechnical node features for the Phase-A information-gain test.

These are the NON-LEAKY, not-yet-derived, derivable properties the lit_swarm gap analysis
shortlisted (plasticity index, fines content, liquid limit, permeability, K0, recompression index,
granular fraction) — each computable from a boring's OCR'd USCS profile + depth + groundwater,
WITHOUT the SPT-N blow count. They are aggregated per boring (whole-profile + near-surface,
thickness-weighted) and attached to boring nodes; the soil-type target lives on separate soil-label
nodes, so these are non-leaky for Phase A. The question they answer: does encoding the boring's
measured USCS profile as physically-meaningful geotech properties add information to the soil-type
GNN beyond geology + SSURGO?

USCS-keyed values follow standard references surfaced by the harvest: Terzaghi, Peck & Mesri (1996)
permeability table; Hazen (1911); USCS plasticity/fines defaults (FHWA-NHI-06-089 / NAVFAC DM-7);
Jaky (1944) / Brooker & Ireland (1965) K0; Terzaghi-Peck-Mesri recompression indices.
"""
from __future__ import annotations

import numpy as np

# Per-USCS-class property defaults. log10k is log10 of hydraulic conductivity in m/s (spans ~10
# orders of magnitude -> use the log). granular=1 for gravels/sands. pi/ll in %, fines in %.
#                  pi    fines  ll   log10k  k0    cr     granular
USCS_PROPS = {
    "GW": (0,    3,   0,   -1.0,  0.45, 0.00, 1),
    "GP": (0,    2,   0,   -1.5,  0.45, 0.00, 1),
    "GM": (4,   25,   0,   -6.0,  0.48, 0.01, 1),
    "GC": (12,  35,  30,   -6.5,  0.55, 0.03, 1),
    "SW": (0,    3,   0,   -3.0,  0.42, 0.00, 1),
    "SP": (0,    2,   0,   -3.5,  0.45, 0.00, 1),
    "SM": (6,   25,   0,   -5.0,  0.50, 0.02, 1),
    "SC": (14,  35,  30,   -7.0,  0.55, 0.03, 1),
    "ML": (8,   85,  30,   -7.0,  0.52, 0.03, 0),
    "CL": (15,  85,  35,   -9.0,  0.58, 0.04, 0),
    "MH": (25,  90,  60,   -7.5,  0.60, 0.06, 0),
    "CH": (40,  90,  65,  -10.0,  0.66, 0.09, 0),
    "OL": (10,  80,  40,   -8.0,  0.55, 0.10, 0),
    "OH": (30,  85,  70,   -8.0,  0.62, 0.12, 0),
    "PT": (0,   50,   0,   -6.0,  0.50, 0.20, 0),
    "FILL": (8, 50,  20,   -5.0,  0.50, 0.03, 0),
}
_PROP_ORDER = ["pi", "fines", "ll", "log10k", "k0", "cr", "granular"]

# the columns emitted per boring (profile-weighted means + a few near-surface means)
GEO_COLS = ["geo_pi", "geo_fines", "geo_ll", "geo_log10k", "geo_k0", "geo_cr",
            "geo_granular_frac", "geo_pi_surf", "geo_log10k_surf", "geo_n_intervals"]
_SURF_FT = 30.0


# Per-interval (depth-resolved) variant for Phase B. A single stratum's literature geotech vector,
# keyed only by its USCS class. Non-leaky w.r.t. spt_n (a function of the USCS label + reference
# tables, never the measured blow count) — same status as the B2 σ'v0/γ inputs. CAVEAT: it IS a
# deterministic function of uscs_class, which is a *secondary* Phase-B target, so it leaks the
# USCS-at-depth head; use it for the SPT-N information-gain test only.
GEO_SAMPLE_COLS = ["geo_" + p for p in _PROP_ORDER]   # geo_pi, geo_fines, ... geo_granular


def interval_geotech(uscs_class) -> "np.ndarray | None":
    """The 7-value literature geotech vector (in _PROP_ORDER) for one stratum's USCS class, or None
    when the class is missing/unmappable (caller masks it out)."""
    cu = _clean_uscs(uscs_class)
    if cu is None:
        return None
    return np.array(USCS_PROPS[cu], dtype=np.float64)


def _clean_uscs(u: str) -> str | None:
    if not u:
        return None
    u = str(u).strip().upper()
    # take the leading USCS token (e.g. "SM-SC" -> "SM", "CL/ML" -> "CL")
    for sep in ("-", "/", " ", ","):
        if sep in u:
            u = u.split(sep)[0]
    return u if u in USCS_PROPS else None


def compute_boring_geotech(con) -> tuple[dict, list]:
    """Return {boring_id: np.array(len(GEO_COLS))} of literature-derived geotech aggregates.

    Thickness-weighted over the USCS profile; near-surface variants use the 0-30 ft intervals.
    Only borings with >=1 mappable USCS interval get a row (others -> missing/mask 0 upstream).
    """
    rows = con.execute("""
        SELECT boring_id, interval_index, top_depth, bottom_depth, uscs_class
        FROM strata WHERE uscs_class IS NOT NULL
        ORDER BY boring_id, interval_index
    """).fetchall()

    by_boring: dict[str, list] = {}
    for bid, _ii, top, bot, uscs in rows:
        cu = _clean_uscs(uscs)
        if cu is None:
            continue
        top = float(top) if top is not None else 0.0
        bot = float(bot) if bot is not None and bot > top else top + 2.0
        thick = max(0.5, min(bot - top, 20.0))   # clamp pathological thicknesses
        by_boring.setdefault(bid, []).append((top, thick, USCS_PROPS[cu]))

    out: dict[str, np.ndarray] = {}
    for bid, intervals in by_boring.items():
        w = np.array([t for _top, t, _p in intervals], dtype=np.float64)
        props = np.array([p for _top, _t, p in intervals], dtype=np.float64)  # [m,7]
        tops = np.array([top for top, _t, _p in intervals])
        wsum = w.sum() or 1.0
        prof = (props * w[:, None]).sum(0) / wsum                              # weighted means
        surf = tops < _SURF_FT
        if surf.any():
            ws = w[surf]; pi_s = (props[surf, 0] * ws).sum() / (ws.sum() or 1.0)
            k_s = (props[surf, 3] * ws).sum() / (ws.sum() or 1.0)
        else:
            pi_s, k_s = prof[0], prof[3]
        out[bid] = np.array([
            prof[0], prof[1], prof[2], prof[3], prof[4], prof[5], prof[6],
            pi_s, k_s, float(len(intervals)),
        ], dtype=np.float32)
    return out, list(GEO_COLS)


def build_geo_tensors(con, node_ids):
    """Map per-boring geotech vectors onto the union-graph node order.

    Returns (x_geo [N,G] standardized float32, x_geo_mask [N,G] 1.0 where present, cols).
    Standardization uses present-boring stats; label nodes + featureless borings get 0 + mask 0.
    """
    geo, cols = compute_boring_geotech(con)
    G = len(cols)
    N = len(node_ids)
    raw = np.full((N, G), np.nan, dtype=np.float64)
    for i, nid in enumerate(node_ids):
        if isinstance(nid, str) and nid.startswith("b:"):
            v = geo.get(nid[2:])
            if v is not None:
                raw[i] = v
    present = ~np.isnan(raw[:, 0])
    x = np.zeros((N, G), dtype=np.float32)
    mask = np.zeros((N, G), dtype=np.float32)
    if present.any():
        mu = np.nanmean(raw[present], axis=0)
        sd = np.nanstd(raw[present], axis=0)
        sd[sd == 0] = 1.0
        x[present] = ((raw[present] - mu) / sd).astype(np.float32)
        mask[present] = 1.0
    return x, mask, cols, int(present.sum())
