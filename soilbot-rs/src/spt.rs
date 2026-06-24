//! SPT-N correction chain: field N → N60 → (N1)60.
//!
//! N60 = N · CE · CB · CS · CR  (energy, borehole-diameter, sampler, rod-length corrections)
//! CN  = √(Pa / σ'v0)  capped at cn_cap  (Liao & Whitman 1986)
//! (N1)60 = N60 · CN
//!
//! References: ASTM D1586-18; Skempton 1986 (CB, CS, CR); Liao & Whitman 1986 (CN).

use crate::common::PA_TSF;
use crate::config::SptParams;

/// Rod-length correction CR by depth band (Skempton 1986 / Youd et al. 2001), where depth ≈ rod
/// length. Returns 1.0 when corrections are disabled.
fn rod_length_cr(depth_ft: f64, enabled: bool) -> f64 {
    if !enabled {
        return 1.0;
    }
    let z_m = depth_ft * 0.3048;
    if z_m < 3.0 {
        0.75
    } else if z_m < 4.0 {
        0.80
    } else if z_m < 6.0 {
        0.85
    } else if z_m < 10.0 {
        0.95
    } else {
        1.0
    }
}

/// Borehole-diameter correction CB (Skempton 1986).
fn borehole_cb(diam_in: f64) -> f64 {
    let mm = diam_in * 25.4;
    if mm <= 120.0 {
        1.0
    } else if mm <= 150.0 {
        1.05
    } else {
        1.15
    }
}

/// Energy-corrected blow count N60 at a given sample depth.
pub fn n60(field_n: f64, depth_ft: f64, p: &SptParams) -> f64 {
    let ce = p.hammer_efficiency_pct / p.energy_ratio_pct;
    let cb = borehole_cb(p.borehole_diam_in);
    let cs = if p.sampler_liner { 1.0 } else { 1.2 }; // liner-room w/o liner → 1.2
    let cr = rod_length_cr(depth_ft, p.rod_length_corrections);
    field_n * ce * cb * cs * cr
}

/// Overburden correction factor CN (Liao & Whitman 1986), capped.
pub fn cn(sigma_eff_v0_tsf: f64, cap: f64) -> f64 {
    (PA_TSF / sigma_eff_v0_tsf).sqrt().min(cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SptParams;

    #[test]
    fn n60_standard_hammer_deep() {
        // ER=60 hammer (CE=1), 4" hole (CB=1), no liner (CS=1.2), depth>10m (CR=1):
        // N60 = 20 * 1 * 1 * 1.2 * 1 = 24.
        let p = SptParams::default();
        assert!((n60(20.0, 40.0, &p) - 24.0).abs() < 1e-9);
    }

    #[test]
    fn cn_liao_whitman_at_one_atm_is_one() {
        // σ'v0 = Pa (1.0581 tsf) → CN = 1.0; capped at 1.7 for shallow/low stress.
        assert!((cn(crate::common::PA_TSF, 1.7) - 1.0).abs() < 1e-6);
        assert!((cn(0.1, 1.7) - 1.7).abs() < 1e-9); // cap engages at low overburden
    }
}
