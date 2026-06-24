//! Derived engineering properties from corrected SPT and index parameters.
//!
//! Each correlation carries its citation and is gated to the soil behaviour it is valid for
//! (granular vs cohesive); callers pass `None` through for inapplicable combinations.

const KPA_PER_TSF: f64 = 95.76; // 1 tsf = 95.76 kPa
const TSF_PER_MPA: f64 = 10.4427; // 1 MPa = 10.4427 tsf

/// Drained friction angle φ′ (deg) of granular soils — Peck, Hanson & Thornburn (1974).
/// φ′ = 27.1 + 0.3·(N1)60 − 0.00054·(N1)60².
pub fn phi_peck(n1_60: f64) -> f64 {
    (27.1 + 0.3 * n1_60 - 0.00054 * n1_60 * n1_60).clamp(20.0, 45.0)
}

/// Friction angle φ′ (deg) — Hatanaka & Uchida (1996): φ = 3.5·√N + 22.3, capped at 40°.
pub fn phi_hatanaka(n60: f64) -> f64 {
    (3.5 * n60.max(0.0).sqrt() + 22.3).min(40.0)
}

/// Relative density Dr (%) of NC recent sands — Skempton (1986): (N1)60 / Dr² ≈ 60.
/// Dr = √((N1)60 / 60) · 100, capped at 100%.
pub fn relative_density(n1_60: f64) -> f64 {
    ((n1_60 / 60.0).max(0.0).sqrt() * 100.0).min(100.0)
}

/// Undrained shear strength Su (tsf) of clays — Stroud (1974): Su = f1·N60 (kPa), f1 keyed to PI.
pub fn su_stroud(n60: f64, pi: f64) -> f64 {
    let f1 = (6.0 - 0.06 * pi).clamp(4.2, 6.5); // kPa per blow
    f1 * n60 / KPA_PER_TSF
}

/// Drained Young's modulus Es (tsf) — AASHTO (1997): Es[MPa] = mult·(N1)60, mult by soil type.
pub fn youngs_modulus(n1_60: f64, uscs: Option<&str>) -> f64 {
    let mult = es_multiplier(uscs);
    mult * n1_60 * TSF_PER_MPA
}

fn es_multiplier(uscs: Option<&str>) -> f64 {
    let u = uscs.map(|s| s.trim().to_uppercase()).unwrap_or_default();
    let first = u.chars().next();
    match first {
        Some('G') => 1.1,
        Some('S') => {
            if u.starts_with("SW") || u.starts_with("SP") {
                1.0
            } else {
                0.7 // SM / SC
            }
        }
        Some('M') => 0.4,
        Some('C') => 0.5,
        Some('O') => 0.3,
        _ => 0.5,
    }
}

/// Constrained (oedometric) modulus M (tsf) from Es, ν≈0.3: M = Es·(1−ν)/((1+ν)(1−2ν)) ≈ 1.35·Es.
pub fn constrained_modulus(es_tsf: f64) -> f64 {
    es_tsf * 1.346
}

/// Meyerhof (1956) settlement-limited allowable bearing pressure qa (tsf) for ~25 mm settlement,
/// depth factor Kd = 1. `footing_width_ft` from config. Uses local N60 as a per-depth indicator.
pub fn meyerhof_bearing(n60: f64, footing_width_ft: f64) -> f64 {
    let b_m = (footing_width_ft * 0.3048).max(0.1);
    let qa_kpa = if b_m <= 1.2 {
        12.0 * n60
    } else {
        let r = (b_m + 0.3) / b_m;
        8.0 * n60 * r * r
    };
    qa_kpa / KPA_PER_TSF
}

/// Idriss & Boulanger (2008/2014) clean-sand-equivalent (N1)60cs from fines content FC (%).
pub fn n1_60_cs(n1_60: f64, fines_pct: f64) -> f64 {
    let fc = fines_pct.max(0.0);
    let d = (1.63 + 9.7 / (fc + 0.01) - (15.7 / (fc + 0.01)).powi(2)).exp();
    n1_60 + d
}

/// Cyclic resistance ratio CRR at M=7.5, σ'v=1 atm — Idriss & Boulanger (2014).
/// x = (N1)60cs clamped to 37.5 (above which soil is treated as non-liquefiable).
pub fn crr_7_5(n1_60_cs: f64) -> f64 {
    let x = n1_60_cs.clamp(0.0, 37.5);
    (x / 14.1 + (x / 126.0).powi(2) - (x / 23.6).powi(3) + (x / 25.4).powi(4) - 2.8).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_peck_textbook() {
        // (N1)60 = 20 → 27.1 + 6.0 - 0.216 = 32.884°
        assert!((phi_peck(20.0) - 32.884).abs() < 1e-3);
        assert!(phi_peck(0.0) >= 20.0 && phi_peck(80.0) <= 45.0); // bounded
    }

    #[test]
    fn phi_hatanaka_capped() {
        // 3.5*sqrt(25)+22.3 = 39.8 ; large N caps at 40.
        assert!((phi_hatanaka(25.0) - 39.8).abs() < 1e-6);
        assert!((phi_hatanaka(400.0) - 40.0).abs() < 1e-9);
    }

    #[test]
    fn relative_density_skempton() {
        // (N1)60=60 → Dr=100% ; 15 → sqrt(0.25)*100 = 50%.
        assert!((relative_density(60.0) - 100.0).abs() < 1e-6);
        assert!((relative_density(15.0) - 50.0).abs() < 1e-6);
    }

    #[test]
    fn su_stroud_high_plasticity() {
        // PI=40 → f1 clamps to 4.2 ; Su = 4.2*N60/95.76. N60=10 → 0.4386 tsf.
        assert!((su_stroud(10.0, 40.0) - 4.2 * 10.0 / 95.76).abs() < 1e-9);
    }

    #[test]
    fn meyerhof_narrow_footing() {
        // B=3ft=0.914m ≤ 1.2 → qa = 12*N60 kPa. N60=10 → 120 kPa = 1.2533 tsf.
        assert!((meyerhof_bearing(10.0, 3.0) - 120.0 / 95.76).abs() < 1e-9);
    }

    #[test]
    fn crr_increases_with_density() {
        assert!(crr_7_5(25.0) > crr_7_5(10.0));
    }
}
