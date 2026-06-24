//! Shared constants and small numeric helpers used across the crate.

/// US survey feet per metre — must match `pipeline/graph.FT_PER_M` exactly for edge parity.
pub const FT_PER_M: f64 = 3.280839895;

/// Standard atmospheric reference pressure used to normalise overburden (tsf).
/// 1 atm = 2116.22 psf = 1.0581 tsf.
pub const PA_TSF: f64 = 1.0581;

/// Euclidean distance between two 2-D points held in a flat `[[f64;2]]` slice.
#[inline]
pub fn dist(pts: &[[f64; 2]], i: usize, j: usize) -> f64 {
    let dx = pts[i][0] - pts[j][0];
    let dy = pts[i][1] - pts[j][1];
    (dx * dx + dy * dy).sqrt()
}

/// Edge weight `round(1 / (1 + d), 6)` — mirrors the Python kernel's weighting.
#[inline]
pub fn edge_weight(d: f64) -> f64 {
    let w = 1.0 / (1.0 + d);
    (w * 1.0e6).round() / 1.0e6
}

/// Canonical (lo, hi) index pair ordered by the node-id STRING comparison, matching
/// `pipeline/graph._canon` (which orders the id strings, not the indices). Inputs are
/// loaded `ORDER BY id`, so string order == index order, but we compare strings to be safe.
#[inline]
pub fn canon(ids: &[String], i: usize, j: usize) -> (usize, usize) {
    if ids[i] <= ids[j] {
        (i, j)
    } else {
        (j, i)
    }
}
