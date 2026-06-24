//! Per-boring orchestration: turn raw strata rows into a depth-resolved geotechnical profile.
//!
//! Input columns arrive sorted by (boring_id, interval_index). For each boring we build a
//! unit-weight column, integrate effective stress, then evaluate the SPT-correction chain and
//! the derived-property correlations at each sampled interval. Output is one row per input
//! interval, returned as a columnar Python dict for the Python side to write to `strata_derived`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::config::{is_cohesive, is_granular, SoilEngineParams};
use crate::geotech;
use crate::spt;
use crate::stress::{Layer, StressProfile};

const SPT_MIN: f64 = 0.0;
const SPT_MAX: f64 = 100.0; // matches ml/data3d sanity gate (>100 = digit error / refusal)

#[derive(Clone)]
struct Row {
    top: Option<f64>,
    bottom: Option<f64>,
    uscs: Option<String>,
    spt: Option<f64>,
    gw: Option<f64>,
    conf: Option<f64>,
}

/// Output column accumulators (one entry per input row).
#[derive(Default)]
struct Out {
    depth: Vec<Option<f64>>,
    sigma_v0: Vec<Option<f64>>,
    sigma_eff: Vec<Option<f64>>,
    gamma: Vec<Option<f64>>,
    n60: Vec<Option<f64>>,
    cn: Vec<Option<f64>>,
    n1_60: Vec<Option<f64>>,
    phi_peck: Vec<Option<f64>>,
    phi_hat: Vec<Option<f64>>,
    dr: Vec<Option<f64>>,
    su: Vec<Option<f64>>,
    e_mod: Vec<Option<f64>>,
    m_con: Vec<Option<f64>>,
    bearing: Vec<Option<f64>>,
    n1_60cs: Vec<Option<f64>>,
    crr: Vec<Option<f64>>,
    conf: Vec<Option<f64>>,
}

impl Out {
    fn push_none(&mut self) {
        self.depth.push(None);
        self.sigma_v0.push(None);
        self.sigma_eff.push(None);
        self.gamma.push(None);
        self.n60.push(None);
        self.cn.push(None);
        self.n1_60.push(None);
        self.phi_peck.push(None);
        self.phi_hat.push(None);
        self.dr.push(None);
        self.su.push(None);
        self.e_mod.push(None);
        self.m_con.push(None);
        self.bearing.push(None);
        self.n1_60cs.push(None);
        self.crr.push(None);
        self.conf.push(None);
    }
}

/// Build the depth-sorted unit-weight column for one boring's rows. Missing bottoms are closed
/// at the next interval's top (or a nominal thickness for the deepest interval).
fn build_layers(rows: &[Row], p: &SoilEngineParams) -> Vec<Layer> {
    let mut idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].top.is_some()).collect();
    idx.sort_by(|&a, &b| rows[a].top.unwrap().partial_cmp(&rows[b].top.unwrap()).unwrap());
    let nominal = (2.0 * p.depth_step_ft).max(2.0);
    let mut layers = Vec::with_capacity(idx.len());
    for (k, &i) in idx.iter().enumerate() {
        let top = rows[i].top.unwrap();
        let next_top = idx.get(k + 1).map(|&j| rows[j].top.unwrap());
        let bottom = match rows[i].bottom {
            Some(b) if b > top => b,
            _ => next_top.filter(|&nt| nt > top).unwrap_or(top + nominal),
        };
        layers.push(Layer { top, bottom, gamma_pcf: p.gamma_for(rows[i].uscs.as_deref()) });
    }
    layers
}

fn process_group(rows: &[Row], p: &SoilEngineParams, out: &mut Out) {
    // per-boring water table: shallowest observed gw, else dry (no pore pressure)
    let gw = rows
        .iter()
        .filter_map(|r| r.gw)
        .filter(|&g| g >= 0.0)
        .fold(f64::INFINITY, f64::min);
    let layers = build_layers(rows, p);
    let prof = StressProfile::build(&layers, gw, p);

    for r in rows {
        let top = match r.top {
            Some(t) if t >= 0.0 => t,
            _ => {
                out.push_none();
                continue;
            }
        };
        let depth = match r.bottom {
            Some(b) if b > top => 0.5 * (top + b),
            _ => top,
        };
        let sigma_v0 = prof.sigma_v0_tsf(depth);
        let sigma_eff = prof.sigma_eff_v0_tsf(depth);
        let gamma = p.gamma_for(r.uscs.as_deref());

        out.depth.push(Some(depth));
        out.sigma_v0.push(Some(sigma_v0));
        out.sigma_eff.push(Some(sigma_eff));
        out.gamma.push(Some(gamma));
        out.conf.push(r.conf);

        let uscs = r.uscs.as_deref();
        let granular = is_granular(uscs);
        let cohesive = is_cohesive(uscs);

        // SPT-derived chain (only for physically plausible blow counts)
        let spt_ok = r.spt.filter(|&n| (SPT_MIN..=SPT_MAX).contains(&n));
        if let Some(n) = spt_ok {
            let n60 = spt::n60(n, depth, &p.spt);
            let cn = spt::cn(sigma_eff, p.spt.cn_cap);
            let n1_60 = n60 * cn;
            out.n60.push(Some(n60));
            out.cn.push(Some(cn));
            out.n1_60.push(Some(n1_60));
            out.e_mod.push(Some(geotech::youngs_modulus(n1_60, uscs)));
            out.m_con
                .push(Some(geotech::constrained_modulus(geotech::youngs_modulus(n1_60, uscs))));
            out.bearing
                .push(Some(geotech::meyerhof_bearing(n60, p.bearing.footing_width_ft)));

            if granular {
                out.phi_peck.push(Some(geotech::phi_peck(n1_60)));
                out.phi_hat.push(Some(geotech::phi_hatanaka(n60)));
                out.dr.push(Some(geotech::relative_density(n1_60)));
            } else {
                out.phi_peck.push(None);
                out.phi_hat.push(None);
                out.dr.push(None);
            }
            if cohesive {
                let pi = p.plasticity_for(uscs).pi;
                out.su.push(Some(geotech::su_stroud(n60, pi)));
            } else {
                out.su.push(None);
            }
            // liquefaction (gated): granular, saturated (below water table)
            if p.liquefaction.enabled && granular && depth >= gw {
                let fc = p.plasticity_for(uscs).fines_pct;
                let n1cs = geotech::n1_60_cs(n1_60, fc);
                out.n1_60cs.push(Some(n1cs));
                out.crr.push(Some(geotech::crr_7_5(n1cs)));
            } else {
                out.n1_60cs.push(None);
                out.crr.push(None);
            }
        } else {
            out.n60.push(None);
            out.cn.push(None);
            out.n1_60.push(None);
            out.phi_peck.push(None);
            out.phi_hat.push(None);
            out.dr.push(None);
            out.su.push(None);
            out.e_mod.push(None);
            out.m_con.push(None);
            out.bearing.push(None);
            out.n1_60cs.push(None);
            out.crr.push(None);
        }
    }
}

/// Compute the depth-resolved geotechnical profile for all borings.
/// Inputs are columnar and MUST be sorted by (boring_id, interval_index). Returns a dict of
/// equal-length output columns (None where a property does not apply / inputs are missing).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn soil_profile(
    py: Python<'_>,
    boring_id: Vec<String>,
    interval_index: Vec<i64>,
    top_depth: Vec<Option<f64>>,
    bottom_depth: Vec<Option<f64>>,
    uscs_class: Vec<Option<String>>,
    spt_n: Vec<Option<f64>>,
    gw_depth: Vec<Option<f64>>,
    confidence: Vec<Option<f64>>,
    config_json: String,
) -> PyResult<Py<PyDict>> {
    let p = SoilEngineParams::from_json(&config_json);
    let n = boring_id.len();
    let rows: Vec<Row> = (0..n)
        .map(|i| Row {
            top: top_depth.get(i).copied().flatten(),
            bottom: bottom_depth.get(i).copied().flatten(),
            uscs: uscs_class.get(i).cloned().flatten(),
            spt: spt_n.get(i).copied().flatten(),
            gw: gw_depth.get(i).copied().flatten(),
            conf: confidence.get(i).copied().flatten(),
        })
        .collect();

    let mut out = Out::default();
    let mut start = 0usize;
    while start < n {
        let mut end = start + 1;
        while end < n && boring_id[end] == boring_id[start] {
            end += 1;
        }
        process_group(&rows[start..end], &p, &mut out);
        start = end;
    }

    let d = PyDict::new(py);
    d.set_item("boring_id", PyList::new(py, &boring_id)?)?;
    d.set_item("interval_index", PyList::new(py, &interval_index)?)?;
    d.set_item("depth_ft", PyList::new(py, &out.depth)?)?;
    d.set_item("sigma_v0_tsf", PyList::new(py, &out.sigma_v0)?)?;
    d.set_item("sigma_eff_v0_tsf", PyList::new(py, &out.sigma_eff)?)?;
    d.set_item("gamma_pcf", PyList::new(py, &out.gamma)?)?;
    d.set_item("n60", PyList::new(py, &out.n60)?)?;
    d.set_item("cn", PyList::new(py, &out.cn)?)?;
    d.set_item("n1_60", PyList::new(py, &out.n1_60)?)?;
    d.set_item("phi_peck_deg", PyList::new(py, &out.phi_peck)?)?;
    d.set_item("phi_hatanaka_deg", PyList::new(py, &out.phi_hat)?)?;
    d.set_item("dr_pct", PyList::new(py, &out.dr)?)?;
    d.set_item("su_tsf", PyList::new(py, &out.su)?)?;
    d.set_item("e_modulus_tsf", PyList::new(py, &out.e_mod)?)?;
    d.set_item("m_constrained_tsf", PyList::new(py, &out.m_con)?)?;
    d.set_item("allow_bearing_tsf", PyList::new(py, &out.bearing)?)?;
    d.set_item("n1_60cs", PyList::new(py, &out.n1_60cs)?)?;
    d.set_item("crr", PyList::new(py, &out.crr)?)?;
    d.set_item("confidence", PyList::new(py, &out.conf)?)?;
    Ok(d.into())
}
