//! Equation parameters for the soil engine, deserialised from the `soil_engine:` block of
//! config.yaml (passed in as JSON by Python). Every field has a documented literature default
//! so the engine runs even with an empty/absent config block.

use std::collections::HashMap;

use serde::Deserialize;

fn d_step() -> f64 { 1.0 }
fn d_60() -> f64 { 60.0 }
fn d_4() -> f64 { 4.0 }
fn d_true() -> bool { true }
fn d_cncap() -> f64 { 1.7 }
fn d_water() -> f64 { 62.4 }
fn d_default_gamma() -> f64 { 120.0 }
fn d_footing() -> f64 { 5.0 }
fn d_mag() -> f64 { 6.0 }
fn d_pga() -> f64 { 0.15 }

#[derive(Deserialize, Clone)]
pub struct SptParams {
    /// Field energy ratio relative to the 60% standard (CE = hammer_efficiency / energy_ratio).
    #[serde(default = "d_60")]
    pub energy_ratio_pct: f64,
    #[serde(default = "d_60")]
    pub hammer_efficiency_pct: f64,
    /// Borehole diameter (in) → CB (Skempton): 1.0 @ 65–115 mm, 1.05 @ 150 mm, 1.15 @ 200 mm.
    #[serde(default = "d_4")]
    pub borehole_diam_in: f64,
    /// Apply rod-length correction CR (depth-banded). Off → CR = 1.0.
    #[serde(default = "d_true")]
    pub rod_length_corrections: bool,
    /// Sampler with a liner room but no liner → CS = 1.2; with liner → 1.0.
    #[serde(default)]
    pub sampler_liner: bool,
    /// Overburden correction cap CN ≤ cn_cap (Liao & Whitman recommend ≤ 1.7).
    #[serde(default = "d_cncap")]
    pub cn_cap: f64,
}
impl Default for SptParams {
    fn default() -> Self {
        SptParams {
            energy_ratio_pct: 60.0,
            hammer_efficiency_pct: 60.0,
            borehole_diam_in: 4.0,
            rod_length_corrections: true,
            sampler_liner: false,
            cn_cap: 1.7,
        }
    }
}

#[derive(Deserialize, Clone, Default)]
pub struct Plasticity {
    #[serde(default)]
    pub pi: f64,
    #[serde(default)]
    pub fines_pct: f64,
}

#[derive(Deserialize, Clone)]
pub struct Bearing {
    #[serde(default = "d_footing")]
    pub footing_width_ft: f64,
}
impl Default for Bearing {
    fn default() -> Self {
        Bearing { footing_width_ft: 5.0 }
    }
}

#[derive(Deserialize, Clone)]
pub struct Liquefaction {
    #[serde(default)]
    pub enabled: bool,
    // Config surface for CSR triggering (FS = CRR·MSF / CSR); the engine currently emits the
    // resistance side ((N1)60cs + CRR). Demand-side CSR using these is the next extension.
    #[allow(dead_code)]
    #[serde(default = "d_mag")]
    pub magnitude: f64,
    #[allow(dead_code)]
    #[serde(default = "d_pga")]
    pub pga_g: f64,
}
impl Default for Liquefaction {
    fn default() -> Self {
        Liquefaction { enabled: false, magnitude: 6.0, pga_g: 0.15 }
    }
}

#[derive(Deserialize, Clone)]
pub struct SoilEngineParams {
    #[serde(default = "d_step")]
    pub depth_step_ft: f64,
    #[serde(default)]
    pub spt: SptParams,
    #[serde(default)]
    pub unit_weight_by_uscs: HashMap<String, f64>,
    #[serde(default)]
    pub plasticity_defaults_by_uscs: HashMap<String, Plasticity>,
    #[serde(default = "d_water")]
    pub water_unit_weight_pcf: f64,
    #[serde(default = "d_default_gamma")]
    pub default_unit_weight_pcf: f64,
    #[serde(default)]
    pub bearing: Bearing,
    #[serde(default)]
    pub liquefaction: Liquefaction,
}

impl Default for SoilEngineParams {
    fn default() -> Self {
        SoilEngineParams {
            depth_step_ft: 1.0,
            spt: SptParams::default(),
            unit_weight_by_uscs: HashMap::new(),
            plasticity_defaults_by_uscs: HashMap::new(),
            water_unit_weight_pcf: 62.4,
            default_unit_weight_pcf: 120.0,
            bearing: Bearing::default(),
            liquefaction: Liquefaction::default(),
        }
    }
}

impl SoilEngineParams {
    /// Parse from a JSON string; falls back to all-defaults on empty/invalid input,
    /// then fills any missing literature tables with built-in defaults.
    pub fn from_json(s: &str) -> Self {
        let mut p: SoilEngineParams = serde_json::from_str(s).unwrap_or_default();
        if p.unit_weight_by_uscs.is_empty() {
            p.unit_weight_by_uscs = builtin_unit_weights();
        }
        if p.plasticity_defaults_by_uscs.is_empty() {
            p.plasticity_defaults_by_uscs = builtin_plasticity();
        }
        p
    }

    /// Moist/total unit weight (pcf) for a USCS class, dual symbols resolving to their first
    /// group, with a configured default for unknown/None.
    pub fn gamma_for(&self, uscs: Option<&str>) -> f64 {
        if let Some(u) = uscs {
            let key = u.trim().to_uppercase();
            if let Some(g) = self.unit_weight_by_uscs.get(&key) {
                return *g;
            }
            // dual symbol "SP-SM" → try the leading group
            if let Some(head) = key.split('-').next() {
                if let Some(g) = self.unit_weight_by_uscs.get(head) {
                    return *g;
                }
            }
        }
        self.default_unit_weight_pcf
    }

    pub fn plasticity_for(&self, uscs: Option<&str>) -> Plasticity {
        if let Some(u) = uscs {
            let key = u.trim().to_uppercase();
            if let Some(p) = self.plasticity_defaults_by_uscs.get(&key) {
                return p.clone();
            }
            if let Some(head) = key.split('-').next() {
                if let Some(p) = self.plasticity_defaults_by_uscs.get(head) {
                    return p.clone();
                }
            }
        }
        Plasticity::default()
    }
}

/// Built-in moist unit weights (pcf) keyed by USCS group — FHWA-NHI-06-089 / NAVFAC DM-7 ranges.
fn builtin_unit_weights() -> HashMap<String, f64> {
    [
        ("GW", 130.0), ("GP", 125.0), ("GM", 125.0), ("GC", 122.0),
        ("SW", 120.0), ("SP", 118.0), ("SM", 120.0), ("SC", 120.0),
        ("ML", 110.0), ("CL", 115.0), ("OL", 100.0),
        ("MH", 105.0), ("CH", 110.0), ("OH", 95.0), ("PT", 70.0),
        ("FILL", 115.0),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), *v))
    .collect()
}

/// Built-in plasticity index (%) and fines content (% passing #200) defaults by USCS group.
fn builtin_plasticity() -> HashMap<String, Plasticity> {
    let mk = |pi: f64, fines: f64| Plasticity { pi, fines_pct: fines };
    [
        ("CL", mk(15.0, 85.0)), ("CH", mk(40.0, 92.0)),
        ("ML", mk(4.0, 75.0)), ("MH", mk(20.0, 80.0)),
        ("OL", mk(8.0, 75.0)), ("OH", mk(25.0, 82.0)),
        ("SC", mk(12.0, 30.0)), ("SM", mk(3.0, 18.0)),
        ("GC", mk(10.0, 22.0)), ("GM", mk(2.0, 14.0)),
        ("SP", mk(0.0, 4.0)), ("SW", mk(0.0, 3.0)),
        ("GP", mk(0.0, 3.0)), ("GW", mk(0.0, 2.0)),
        ("PT", mk(0.0, 0.0)),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.clone()))
    .collect()
}

/// True for coarse-grained (G*/S*) USCS groups — granular correlations apply (φ′, Dr, liquefaction).
pub fn is_granular(uscs: Option<&str>) -> bool {
    matches!(uscs.map(|u| u.trim().chars().next()), Some(Some('G')) | Some(Some('S'))
        | Some(Some('g')) | Some(Some('s')))
}

/// True for fine-grained cohesive groups (C*/M*/O*) — Su correlations apply.
pub fn is_cohesive(uscs: Option<&str>) -> bool {
    matches!(uscs.map(|u| u.trim().chars().next()),
        Some(Some('C')) | Some(Some('M')) | Some(Some('O'))
        | Some(Some('c')) | Some(Some('m')) | Some(Some('o')))
}
