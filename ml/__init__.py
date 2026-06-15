"""Bayesian GNN experimentation package for NJ soil-type / property prediction.

Sits beside `pipeline/` and reuses its DuckDB store, config, graph math, and covariate
assignment. Phase A trains on the 20,255 labeled soil-class points (borings are unlabeled
context nodes on a shared graph); Phase B brings OCR'd boring stratigraphy online for the
3D depth-resolved targets. See the project plan for the staged A0..A4 / B0..B1 sequence.
"""
