//! Node feature assembly — a parity port of the numeric block in `ml/data.py`
//! (coordinate standardisation, elevation normalisation + mask, multi-scale Fourier encoding).
//!
//! Column order MUST match the Python implementation exactly (the model indexes columns):
//!   [ coord_x, coord_y, elev_norm,  (sin fx, cos fx, sin fy, cos fy) for f in freqs ]
//! giving `3 + 4*len(freqs)` columns (23 with the default 5 frequencies).

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Population std-dev of `xs` about `mean` (numpy `ddof=0`).
fn pop_std(xs: impl Iterator<Item = f64>, mean: f64, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for x in xs {
        let d = x - mean;
        acc += d * d;
    }
    (acc / count as f64).sqrt()
}

/// Returns `(x_num [N,K] f32, x_mask [N,K] f32, stats)` where `stats` carries the
/// normalisation constants for logging (`x_mean`, `x_scale`, `elev_mean`, `elev_std`).
/// `elevation` uses NaN to mark missing values (matching the Python `np.nan` convention).
#[pyfunction]
pub fn assemble_features(
    py: Python<'_>,
    xy: PyReadonlyArray2<f64>,
    elevation: PyReadonlyArray1<f64>,
    fourier_freqs: Vec<f64>,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>, Py<PyDict>)> {
    let xy = xy.as_array();
    let elev = elevation.as_array();
    let n = xy.shape()[0];

    // shared-scale coordinate standardisation (preserves aspect ratio)
    let (mut mx, mut my) = (0.0, 0.0);
    for i in 0..n {
        mx += xy[[i, 0]];
        my += xy[[i, 1]];
    }
    mx /= n.max(1) as f64;
    my /= n.max(1) as f64;
    let sx = pop_std((0..n).map(|i| xy[[i, 0]]), mx, n);
    let sy = pop_std((0..n).map(|i| xy[[i, 1]]), my, n);
    let mut x_scale = sx.max(sy);
    if x_scale == 0.0 {
        x_scale = 1.0; // mirrors `xy.std(0).max() or 1.0`
    }

    // elevation stats over present-only values
    let mut esum = 0.0;
    let mut ecnt = 0usize;
    for i in 0..n {
        let e = elev[i];
        if !e.is_nan() {
            esum += e;
            ecnt += 1;
        }
    }
    let elev_mean = if ecnt > 0 { esum / ecnt as f64 } else { 0.0 };
    let elev_std = if ecnt > 0 {
        pop_std((0..n).map(|i| elev[i]).filter(|e| !e.is_nan()), elev_mean, ecnt)
    } else {
        1.0
    };
    let elev_den = if elev_std == 0.0 { 1.0 } else { elev_std };

    let ncol = 3 + 4 * fourier_freqs.len();
    let mut x_num = Array2::<f32>::zeros((n, ncol));
    let mut x_mask = Array2::<f32>::zeros((n, ncol));

    for i in 0..n {
        let cx = (xy[[i, 0]] - mx) / x_scale;
        let cy = (xy[[i, 1]] - my) / x_scale;
        let e = elev[i];
        let present = !e.is_nan();
        let enorm = if present { (e - elev_mean) / elev_den } else { 0.0 };

        x_num[[i, 0]] = cx as f32;
        x_mask[[i, 0]] = 1.0;
        x_num[[i, 1]] = cy as f32;
        x_mask[[i, 1]] = 1.0;
        x_num[[i, 2]] = enorm as f32;
        x_mask[[i, 2]] = if present { 1.0 } else { 0.0 };

        let mut col = 3;
        for &f in &fourier_freqs {
            x_num[[i, col]] = (f * cx).sin() as f32;
            x_mask[[i, col]] = 1.0;
            x_num[[i, col + 1]] = (f * cx).cos() as f32;
            x_mask[[i, col + 1]] = 1.0;
            x_num[[i, col + 2]] = (f * cy).sin() as f32;
            x_mask[[i, col + 2]] = 1.0;
            x_num[[i, col + 3]] = (f * cy).cos() as f32;
            x_mask[[i, col + 3]] = 1.0;
            col += 4;
        }
    }

    let stats = PyDict::new(py);
    stats.set_item("x_mean", vec![mx, my])?;
    stats.set_item("x_scale", x_scale)?;
    stats.set_item("elev_mean", elev_mean)?;
    stats.set_item("elev_std", elev_std)?;

    Ok((
        PyArray2::from_owned_array(py, x_num).into(),
        PyArray2::from_owned_array(py, x_mask).into(),
        stats.into(),
    ))
}
