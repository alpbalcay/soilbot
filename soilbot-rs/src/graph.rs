//! Spatial graph construction — a parity port of `pipeline/graph.compute_edges` and
//! `ml/graph_build._add_label_boring_edges`.
//!
//! Edge semantics (must match the scipy implementation):
//!   * knn          — k nearest neighbours (skip column 0 = self), weight 1/(1+d)
//!   * delaunay     — Delaunay triangulation edges, weight 1/(1+d)
//!   * same_geology — neighbours within `radius` sharing a surficial unit, weight 1.0
//!   * label_boring — each label node to its k nearest borings, weight 1/(1+d)
//!
//! Distances are Euclidean in the native projected CRS (EPSG:102711, US feet), recomputed
//! from coordinates so weights are bit-comparable with scipy regardless of the kd-tree metric.

use std::collections::HashMap;

use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spade::{DelaunayTriangulation, Point2, Triangulation};

use crate::common::{canon, dist, edge_weight, FT_PER_M};

// kiddo kd-tree: f64 coords, u32 indexed buckets, content u32, 2-D.
type Tree = KdTree<f64, u32, 2, 32, u32>;

fn build_tree(pts: &[[f64; 2]]) -> Tree {
    let mut tree: Tree = KdTree::with_capacity(pts.len());
    for (i, p) in pts.iter().enumerate() {
        tree.add(p, i as u32);
    }
    tree
}

/// Delaunay edges as original-index pairs. Mirrors scipy's `Delaunay(xy, qhull_options="QJ")`:
/// duplicate points collapse onto one vertex (first index wins). Returns Err on degenerate
/// input (NaN/inf) so the caller can skip delaunay edges, matching the Python try/except.
fn delaunay_edges(pts: &[[f64; 2]]) -> Result<Vec<(usize, usize)>, spade::InsertionError> {
    let mut tri: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
    let mut handle_to_idx: HashMap<usize, usize> = HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        let h = tri.insert(Point2::new(p[0], p[1]))?;
        handle_to_idx.entry(h.index()).or_insert(i);
    }
    let mut out = Vec::with_capacity(tri.num_undirected_edges());
    for edge in tri.undirected_edges() {
        let [v0, v1] = edge.vertices();
        let i0 = handle_to_idx[&v0.fix().index()];
        let i1 = handle_to_idx[&v1.fix().index()];
        out.push((i0, i1));
    }
    Ok(out)
}

/// Build knn + delaunay + same_geology edges over arbitrary nodes.
/// Returns a Python dict `{(src, dst, edge_type): weight}` — a drop-in for the scipy
/// `compute_edges`, so the existing persistence code in Python is unchanged.
#[pyfunction]
#[pyo3(signature = (ids, xy, units, knn_k, delaunay=true, same_geology=true,
                    same_geology_radius_m=2000.0, same_geology_k=26))]
#[allow(clippy::too_many_arguments)]
pub fn compute_edges(
    py: Python<'_>,
    ids: Vec<String>,
    xy: PyReadonlyArray2<f64>,
    units: HashMap<String, String>,
    knn_k: usize,
    delaunay: bool,
    same_geology: bool,
    same_geology_radius_m: f64,
    same_geology_k: usize,
) -> PyResult<Py<PyDict>> {
    let arr = xy.as_array();
    let n = ids.len();
    let pts: Vec<[f64; 2]> = (0..n).map(|i| [arr[[i, 0]], arr[[i, 1]]]).collect();

    // edge map keyed (lo_idx, hi_idx, type_code): 0=knn, 1=delaunay, 2=same_geology
    let mut edges: HashMap<(usize, usize, u8), f64> = HashMap::new();

    if n > knn_k + 1 {
        let tree = build_tree(&pts);

        // --- kNN: nearest k+1, drop column 0 (self), as in scipy idx[i, 1:] ---
        for i in 0..n {
            let nns = tree.nearest_n::<SquaredEuclidean>(&pts[i], knn_k + 1);
            for nn in nns.iter().skip(1) {
                let j = nn.item as usize;
                let (a, b) = canon(&ids, i, j);
                if a == b {
                    continue;
                }
                edges.insert((a, b, 0), edge_weight(dist(&pts, a, b)));
            }
        }

        // --- same geology unit, bounded by radius + neighbour cap ---
        if same_geology && !units.is_empty() {
            let radius_ft = same_geology_radius_m * FT_PER_M;
            let kg = same_geology_k.min(n);
            for i in 0..n {
                let ui = match units.get(&ids[i]) {
                    Some(u) => u,
                    None => continue,
                };
                let nns = tree.nearest_n::<SquaredEuclidean>(&pts[i], kg);
                for nn in nns.iter().skip(1) {
                    let j = nn.item as usize;
                    let d = dist(&pts, i, j);
                    if d > radius_ft {
                        break; // neighbours are distance-sorted
                    }
                    if units.get(&ids[j]) == Some(ui) {
                        let (a, b) = canon(&ids, i, j);
                        if a == b {
                            continue;
                        }
                        edges.entry((a, b, 2)).or_insert(1.0);
                    }
                }
            }
        }

        // --- Delaunay (separate edge type; coexists with knn on the same pair) ---
        if delaunay {
            if let Ok(de) = delaunay_edges(&pts) {
                for (i, j) in de {
                    let (a, b) = canon(&ids, i, j);
                    if a == b {
                        continue;
                    }
                    edges
                        .entry((a, b, 1))
                        .or_insert_with(|| edge_weight(dist(&pts, a, b)));
                }
            }
        }
    }

    let type_names = ["knn", "delaunay", "same_geology"];
    let out = PyDict::new(py);
    for ((a, b, tc), w) in edges.iter() {
        out.set_item((ids[*a].as_str(), ids[*b].as_str(), type_names[*tc as usize]), *w)?;
    }
    Ok(out.into())
}

/// `label_boring` bridge edges: each label node connects to its k nearest boring nodes.
/// Returns `{(src, dst, "label_boring"): weight}`; the Python caller merges (setdefault).
#[pyfunction]
pub fn label_boring_edges(
    py: Python<'_>,
    ids: Vec<String>,
    xy: PyReadonlyArray2<f64>,
    types: HashMap<String, String>,
    k: usize,
) -> PyResult<Py<PyDict>> {
    let arr = xy.as_array();
    let n = ids.len();
    let pts: Vec<[f64; 2]> = (0..n).map(|i| [arr[[i, 0]], arr[[i, 1]]]).collect();

    let b_idx: Vec<usize> = (0..n)
        .filter(|&i| types.get(&ids[i]).map(|s| s == "boring").unwrap_or(false))
        .collect();
    let l_idx: Vec<usize> = (0..n)
        .filter(|&i| types.get(&ids[i]).map(|s| s == "label").unwrap_or(false))
        .collect();

    let out = PyDict::new(py);
    if b_idx.is_empty() || l_idx.is_empty() {
        return Ok(out.into());
    }

    // kd-tree over borings only; content = position within b_idx.
    let mut tree: Tree = KdTree::with_capacity(b_idx.len());
    for (pos, &bi) in b_idx.iter().enumerate() {
        tree.add(&pts[bi], pos as u32);
    }
    let kk = k.min(b_idx.len());

    for &li in &l_idx {
        let nns = tree.nearest_n::<SquaredEuclidean>(&pts[li], kk);
        for nn in nns.iter() {
            let bi = b_idx[nn.item as usize];
            let (a, b) = canon(&ids, li, bi);
            if a == b {
                continue;
            }
            let key = (ids[a].as_str(), ids[b].as_str(), "label_boring");
            if !out.contains(key)? {
                out.set_item(key, edge_weight(dist(&pts, li, bi)))?;
            }
        }
    }
    Ok(out.into())
}
