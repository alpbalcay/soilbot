//! soilbot_rs — Rust data-engineering + geotechnical soil-equation engine.
//!
//! A PyO3 extension module imported by the Python pipeline. Numeric heavy-lifting only;
//! DuckDB I/O and orchestration stay in Python. Every function here has a Python fallback
//! at its call site, so an un-built extension never hard-breaks the pipeline.

use pyo3::prelude::*;

mod common;
mod config;
mod features;
mod geo;
mod geotech;
mod graph;
mod profile;
mod spt;
mod stress;

#[pymodule]
fn soilbot_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // graph + features (R1/R2)
    m.add_function(wrap_pyfunction!(graph::compute_edges, m)?)?;
    m.add_function(wrap_pyfunction!(graph::label_boring_edges, m)?)?;
    m.add_function(wrap_pyfunction!(features::assemble_features, m)?)?;
    // soil-equation engine + earth-location (R3)
    m.add_function(wrap_pyfunction!(profile::soil_profile, m)?)?;
    m.add_function(wrap_pyfunction!(geo::geodesic_distance, m)?)?;
    m.add_function(wrap_pyfunction!(geo::edge_lengths_geodesic, m)?)?;
    Ok(())
}
