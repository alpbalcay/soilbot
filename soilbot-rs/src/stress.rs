//! Effective vertical stress profile.
//!
//! Builds a piecewise unit-weight column from a boring's strata intervals, integrates total
//! vertical stress on a fine depth grid, and subtracts hydrostatic pore pressure below the
//! water table to give σ'v0(z). Units: depth ft, γ pcf → stress psf, returned in tsf
//! (÷2000 lb/ton).

use crate::config::SoilEngineParams;

const PSF_PER_TSF: f64 = 2000.0;

/// One strata interval as seen by the stress integrator.
pub struct Layer {
    pub top: f64,
    pub bottom: f64,
    pub gamma_pcf: f64,
}

/// Precomputed cumulative total-stress profile on a uniform grid, with the water table depth.
pub struct StressProfile {
    step: f64,
    /// cum[k] = total vertical stress (psf) at depth k*step (top of cell k).
    cum: Vec<f64>,
    gw_depth: f64,
    gamma_water: f64,
    max_depth: f64,
}

impl StressProfile {
    /// Build from depth-sorted layers. `gw_depth` is the water-table depth (ft); use a large
    /// value (e.g. f64::INFINITY) for "no groundwater observed → assume dry".
    pub fn build(layers: &[Layer], gw_depth: f64, p: &SoilEngineParams) -> StressProfile {
        let step = p.depth_step_ft.max(0.1);
        let max_depth = layers.iter().map(|l| l.bottom).fold(0.0_f64, f64::max).max(step);
        let ncell = (max_depth / step).ceil() as usize + 1;
        let mut cum = vec![0.0_f64; ncell + 1];
        let mut acc = 0.0;
        for k in 0..ncell {
            let z_mid = (k as f64 + 0.5) * step;
            let g = gamma_at(layers, z_mid, p.default_unit_weight_pcf);
            acc += g * step; // psf added across this cell
            cum[k + 1] = acc;
        }
        StressProfile { step, cum, gw_depth, gamma_water: p.water_unit_weight_pcf, max_depth }
    }

    /// Total vertical stress σv0 (tsf) at depth `z` (ft), linearly interpolated on the grid.
    pub fn sigma_v0_tsf(&self, z: f64) -> f64 {
        let z = z.clamp(0.0, self.max_depth);
        let f = z / self.step;
        let k = f.floor() as usize;
        let frac = f - k as f64;
        let lo = self.cum[k.min(self.cum.len() - 1)];
        let hi = self.cum[(k + 1).min(self.cum.len() - 1)];
        (lo + frac * (hi - lo)) / PSF_PER_TSF
    }

    /// Hydrostatic pore pressure u (tsf) at depth `z`.
    pub fn pore_pressure_tsf(&self, z: f64) -> f64 {
        let head = (z - self.gw_depth).max(0.0);
        self.gamma_water * head / PSF_PER_TSF
    }

    /// Effective vertical stress σ'v0 (tsf), floored at a small positive value so √(Pa/σ') and
    /// the modulus/strength correlations stay finite at the surface.
    pub fn sigma_eff_v0_tsf(&self, z: f64) -> f64 {
        (self.sigma_v0_tsf(z) - self.pore_pressure_tsf(z)).max(0.01)
    }
}

/// Unit weight (pcf) at depth `z`: the layer whose [top, bottom) contains z, else `default`.
fn gamma_at(layers: &[Layer], z: f64, default: f64) -> f64 {
    for l in layers {
        if z >= l.top && z < l.bottom {
            return l.gamma_pcf;
        }
    }
    default
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SoilEngineParams;

    #[test]
    fn single_layer_hand_computed() {
        // 30 ft of γ=120 pcf, water table at 10 ft.
        let p = SoilEngineParams::default();
        let layers = vec![Layer { top: 0.0, bottom: 30.0, gamma_pcf: 120.0 }];
        let prof = StressProfile::build(&layers, 10.0, &p);
        // σv0(20) = 120*20/2000 = 1.2 tsf
        assert!((prof.sigma_v0_tsf(20.0) - 1.2).abs() < 1e-3);
        // u(20) = 62.4*(20-10)/2000 = 0.312 tsf ; σ'v0 = 1.2 - 0.312 = 0.888
        assert!((prof.pore_pressure_tsf(20.0) - 0.312).abs() < 1e-3);
        assert!((prof.sigma_eff_v0_tsf(20.0) - 0.888).abs() < 1e-3);
    }

    #[test]
    fn effective_stress_monotonic_and_le_total() {
        let p = SoilEngineParams::default();
        let layers = vec![Layer { top: 0.0, bottom: 50.0, gamma_pcf: 115.0 }];
        let prof = StressProfile::build(&layers, 5.0, &p);
        let mut prev = -1.0;
        for k in 0..=50 {
            let z = k as f64;
            let eff = prof.sigma_eff_v0_tsf(z);
            // eff ≤ total, except in the shallow guard region where eff is floored at 0.01 tsf.
            assert!(eff <= prof.sigma_v0_tsf(z).max(0.01) + 1e-9);
            assert!(eff >= prev - 1e-9); // non-decreasing
            prev = eff;
        }
    }
}
