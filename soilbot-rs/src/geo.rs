//! Earth-location utilities: geodesic distance between lon/lat points (for measuring true
//! ground distance between borings) and a vectorised edge-length helper.
//!
//! `haversine` is fast and accurate to ~0.5% (spherical earth); `vincenty` is the ellipsoidal
//! (WGS-84) inverse solution, accurate to sub-mm but iterative. Both return metres.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const R_EARTH_M: f64 = 6_371_008.8; // mean earth radius (m)
// WGS-84 ellipsoid
const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// Great-circle (haversine) distance in metres between two lon/lat points (degrees).
pub fn haversine_m(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dphi = (lat2 - lat1).to_radians();
    let dlmb = (lon2 - lon1).to_radians();
    let a = (dphi / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dlmb / 2.0).sin().powi(2);
    2.0 * R_EARTH_M * a.sqrt().asin()
}

/// Vincenty inverse (WGS-84 ellipsoid) distance in metres; falls back to haversine on
/// non-convergence (near-antipodal points). Inputs in degrees.
pub fn vincenty_m(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let b = WGS84_A * (1.0 - WGS84_F);
    let u1 = ((1.0 - WGS84_F) * lat1.to_radians().tan()).atan();
    let u2 = ((1.0 - WGS84_F) * lat2.to_radians().tan()).atan();
    let l = (lon2 - lon1).to_radians();
    let (sin_u1, cos_u1) = (u1.sin(), u1.cos());
    let (sin_u2, cos_u2) = (u2.sin(), u2.cos());
    let mut lambda = l;
    for _ in 0..200 {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();
        let sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
        .sqrt();
        if sin_sigma == 0.0 {
            return 0.0; // coincident
        }
        let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        let sigma = sin_sigma.atan2(cos_sigma);
        let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        let cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
        let cos2_sigma_m = if cos_sq_alpha != 0.0 {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
        } else {
            0.0
        };
        let c = WGS84_F / 16.0 * cos_sq_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos_sq_alpha));
        let lambda_prev = lambda;
        lambda = l
            + (1.0 - c)
                * WGS84_F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos2_sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));
        if (lambda - lambda_prev).abs() < 1e-12 {
            let u_sq = cos_sq_alpha * (WGS84_A * WGS84_A - b * b) / (b * b);
            let big_a =
                1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
            let big_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
            let delta_sigma = big_b
                * sin_sigma
                * (cos2_sigma_m
                    + big_b / 4.0
                        * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                            - big_b / 6.0
                                * cos2_sigma_m
                                * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                                * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));
            return b * big_a * (sigma - delta_sigma);
        }
    }
    haversine_m(lon1, lat1, lon2, lat2) // non-convergent → fallback
}

/// Geodesic distance (m) between two single points; `method` is "haversine" (default) or "vincenty".
#[pyfunction]
#[pyo3(signature = (lon1, lat1, lon2, lat2, method="haversine"))]
pub fn geodesic_distance(lon1: f64, lat1: f64, lon2: f64, lat2: f64, method: &str) -> f64 {
    if method.eq_ignore_ascii_case("vincenty") {
        vincenty_m(lon1, lat1, lon2, lat2)
    } else {
        haversine_m(lon1, lat1, lon2, lat2)
    }
}

/// Geodesic length (m) of each edge given node lon/lat arrays and integer endpoint indices.
/// Useful to annotate the graph's `src`/`dst` edges with true ground distance.
#[pyfunction]
#[pyo3(signature = (lon, lat, src_idx, dst_idx, method="haversine"))]
pub fn edge_lengths_geodesic(
    py: Python<'_>,
    lon: PyReadonlyArray1<f64>,
    lat: PyReadonlyArray1<f64>,
    src_idx: PyReadonlyArray1<i64>,
    dst_idx: PyReadonlyArray1<i64>,
    method: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let lon = lon.as_array();
    let lat = lat.as_array();
    let s = src_idx.as_array();
    let d = dst_idx.as_array();
    let use_vin = method.eq_ignore_ascii_case("vincenty");
    let out: Vec<f64> = (0..s.len())
        .map(|e| {
            let (i, j) = (s[e] as usize, d[e] as usize);
            if use_vin {
                vincenty_m(lon[i], lat[i], lon[j], lat[j])
            } else {
                haversine_m(lon[i], lat[i], lon[j], lat[j])
            }
        })
        .collect();
    Ok(PyArray1::from_vec(py, out).into())
}
