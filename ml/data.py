"""Assemble the union graph + node features + targets into cached torch tensors.

Produces a `Dataset` (saved to data/ml/dataset.pt + vocab.json) consumed by training:
  - node table over borings ∪ soil_labels (namespaced ids 'b:'/'l:'),
  - categorical geology/SSURGO features -> integer indices (frozen vocab, 0=MISSING/1=OOV),
  - numeric features (normalized coords, elevation) + a per-feature missing mask,
  - targets on label nodes only: soil-type code (82), family (~prefix), drainage,
  - edge_index / edge_type / edge_weight from ml_edges.

Borings carry features but no targets (-1) and act as unlabeled context nodes — the whole
point of the union graph is to let them inform the spatially-disjoint labels.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import torch

from pipeline import db
from pipeline.config import Config

# Categorical feature columns (embedded). Order is fixed and mirrored in the model.
CAT_COLS = ["surficial_unit", "surficial_lithology", "surficial_age",
            "bedrock_unit", "bedrock_lithology",
            "ssurgo_component", "ssurgo_drainagecl", "ssurgo_hydgrp", "node_type"]
EDGE_TYPES = ["knn", "delaunay", "same_geology", "label_boring"]
_FAMILY_RE = re.compile(r"^[A-Za-z]+")


def _family_of(code: str) -> str:
    """NJDOT engineering-soil family = leading alpha prefix (AM-24->AM, M-23->M, AR->AR)."""
    m = _FAMILY_RE.match(code or "")
    return m.group(0) if m else "?"


@dataclass
class Dataset:
    node_ids: list
    node_type: np.ndarray          # 0=boring, 1=label
    xy: np.ndarray                 # [N,2] native feet (raw, for spatial splits)
    x_num: torch.Tensor            # [N, K] normalized numerics
    x_mask: torch.Tensor           # [N, K] 1.0 where the numeric was present
    cat_idx: torch.Tensor          # [N, C] long indices into per-column vocabs
    y_code: torch.Tensor           # [N] long, -1 where unlabeled
    y_family: torch.Tensor         # [N] long, -1 where unlabeled
    y_drain: torch.Tensor          # [N] long, -1 where unlabeled
    y_uscs: torch.Tensor           # [N] long, near-surface USCS from OCR (boring nodes), -1 else
    edge_index: torch.Tensor       # [2, E] long
    edge_type: torch.Tensor        # [E] long (index into EDGE_TYPES)
    edge_weight: torch.Tensor      # [E] float
    vocabs: dict                   # col -> {value: idx}
    code_classes: list             # idx -> code string
    family_classes: list
    drain_classes: list
    uscs_classes: list             # idx -> USCS family string (auxiliary task)
    cat_cardinalities: list        # per CAT_COLS vocab size (incl. MISSING/OOV)

    def save(self, path):
        torch.save({
            "node_ids": self.node_ids, "node_type": self.node_type, "xy": self.xy,
            "x_num": self.x_num, "x_mask": self.x_mask, "cat_idx": self.cat_idx,
            "y_code": self.y_code, "y_family": self.y_family, "y_drain": self.y_drain,
            "y_uscs": self.y_uscs,
            "edge_index": self.edge_index, "edge_type": self.edge_type,
            "edge_weight": self.edge_weight, "vocabs": self.vocabs,
            "code_classes": self.code_classes, "family_classes": self.family_classes,
            "drain_classes": self.drain_classes, "uscs_classes": self.uscs_classes,
            "cat_cardinalities": self.cat_cardinalities,
        }, path)

    @staticmethod
    def load(path) -> "Dataset":
        d = torch.load(path, weights_only=False)
        return Dataset(**d)


_NODE_SQL = """
SELECT 'b:' || b.boring_id AS id, 0 AS node_type,
       ST_X(b.geom_native) AS x, ST_Y(b.geom_native) AS y,
       bc.surficial_unit, bc.surficial_lithology, bc.surficial_age,
       bc.bedrock_unit, bc.bedrock_lithology,
       bc.ssurgo_component, bc.ssurgo_drainagecl, bc.ssurgo_hydgrp,
       CAST(NULL AS VARCHAR) AS primary_label, CAST(NULL AS VARCHAR) AS drainage,
       d.elevation_m AS elevation
FROM borings b
LEFT JOIN boring_covariates bc ON bc.boring_id = b.boring_id
LEFT JOIN dem_samples d ON d.boring_id = b.boring_id
WHERE b.geom_native IS NOT NULL
UNION ALL
SELECT 'l:' || CAST(sl.objectid AS VARCHAR), 1,
       ST_X(sl.geom_native), ST_Y(sl.geom_native),
       sc.surficial_unit, sc.surficial_lithology, sc.surficial_age,
       sc.bedrock_unit, sc.bedrock_lithology,
       sc.ssurgo_component, sc.ssurgo_drainagecl, sc.ssurgo_hydgrp,
       sl.primary_label, sl.drainage,
       CAST(NULL AS DOUBLE)
FROM soil_labels sl
LEFT JOIN soil_label_covariates sc ON sc.id = CAST(sl.objectid AS VARCHAR)
WHERE sl.geom_native IS NOT NULL
"""


def _build_vocab(values) -> dict:
    """value -> idx, reserving 0=MISSING, 1=OOV. None/empty -> MISSING at encode time."""
    vocab = {"__MISSING__": 0, "__OOV__": 1}
    for v in sorted({str(x) for x in values if x is not None and str(x) != ""}):
        vocab[v] = len(vocab)
    return vocab


def _encode(vocab, v) -> int:
    if v is None or str(v) == "":
        return 0
    return vocab.get(str(v), 1)


def _rust_assemble_features(xy, elev, freqs, config, log):
    """Assemble numeric node features with soilbot_rs when `ml.use_rust` is on. Returns
    (x_num, x_mask, x_mean, x_scale, elev_mean, elev_std) or None to use the numpy path.
    Bit-identical to the numpy block (validated by tests/parity_rust.py)."""
    if not (config and config.get("ml", "use_rust", default=False)):
        return None
    try:
        import soilbot_rs
    except ImportError:
        if log:
            log.warning("rust_unavailable_fallback_numpy")
        return None
    x_num, x_mask, stats = soilbot_rs.assemble_features(
        np.ascontiguousarray(xy, dtype=float),
        np.ascontiguousarray(elev, dtype=float), list(freqs))
    return (x_num, x_mask, np.asarray(stats["x_mean"]), float(stats["x_scale"]),
            float(stats["elev_mean"]), float(stats["elev_std"]))


def build_dataset(con, config: Config, log=None) -> Dataset:
    cols = ["id", "node_type", "x", "y", "surficial_unit", "surficial_lithology",
            "surficial_age", "bedrock_unit", "bedrock_lithology", "ssurgo_component",
            "ssurgo_drainagecl", "ssurgo_hydgrp", "primary_label", "drainage", "elevation"]
    rows = con.execute(_NODE_SQL).fetchall()
    R = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    n = len(rows)
    node_ids = R["id"]
    node_type = np.asarray(R["node_type"], dtype=np.int64)
    xy = np.asarray([[R["x"][i], R["y"][i]] for i in range(n)], dtype=np.float64)

    # ---- categorical vocabs (node_type vocab is fixed) ----
    vocabs = {}
    for c in CAT_COLS:
        if c == "node_type":
            vocabs[c] = {"__MISSING__": 0, "__OOV__": 1, "boring": 2, "label": 3}
        else:
            vocabs[c] = _build_vocab(R[c])
    cat_idx = np.zeros((n, len(CAT_COLS)), dtype=np.int64)
    for j, c in enumerate(CAT_COLS):
        if c == "node_type":
            cat_idx[:, j] = [2 if t == 0 else 3 for t in node_type]
        else:
            cat_idx[:, j] = [_encode(vocabs[c], v) for v in R[c]]
    cat_cardinalities = [len(vocabs[c]) for c in CAT_COLS]

    # ---- numerics: normalized coords + multi-scale Fourier coords + elevation ----
    # Fourier positional features let the GNN represent smooth spatial trend (a plain coord
    # pair under-serves it vs RF, which splits freely on x/y). Frequencies span coarse->fine.
    _FREQS = (0.5, 1.0, 2.0, 4.0, 8.0)
    elev = np.asarray([np.nan if v is None else float(v) for v in R["elevation"]])
    _rust_feats = _rust_assemble_features(xy, elev, _FREQS, config, log)
    if _rust_feats is not None:
        x_num, x_mask, x_mean, x_scale, elev_mean, elev_std = _rust_feats
    else:
        x_mean = xy.mean(0); x_scale = xy.std(0).max() or 1.0  # shared scale preserves aspect
        coords = (xy - x_mean) / x_scale
        elev_present = ~np.isnan(elev)
        elev_mean = elev[elev_present].mean() if elev_present.any() else 0.0
        elev_std = elev[elev_present].std() if elev_present.any() else 1.0
        elev_norm = np.where(elev_present,
                             (np.nan_to_num(elev) - elev_mean) / (elev_std or 1.0), 0.0)
        fourier = []
        for f in _FREQS:
            fourier.append(np.sin(f * coords[:, 0])); fourier.append(np.cos(f * coords[:, 0]))
            fourier.append(np.sin(f * coords[:, 1])); fourier.append(np.cos(f * coords[:, 1]))
        feats = [coords[:, 0], coords[:, 1], elev_norm] + fourier
        masks = [np.ones(n), np.ones(n), elev_present.astype(float)] + [np.ones(n)] * len(fourier)
        x_num = np.stack(feats, axis=1).astype(np.float32)
        x_mask = np.stack(masks, axis=1).astype(np.float32)

    # ---- targets (label nodes only) ----
    code_vocab = {}
    fam_vocab = {}
    drain_vocab = {}
    for pl in R["primary_label"]:
        if pl:
            code_vocab.setdefault(pl, len(code_vocab))
            fam = _family_of(pl)
            fam_vocab.setdefault(fam, len(fam_vocab))
    for dr in R["drainage"]:
        if dr:
            drain_vocab.setdefault(str(dr), len(drain_vocab))
    code_classes = [None] * len(code_vocab)
    for k, v in code_vocab.items():
        code_classes[v] = k
    family_classes = [None] * len(fam_vocab)
    for k, v in fam_vocab.items():
        family_classes[v] = k
    drain_classes = [None] * len(drain_vocab)
    for k, v in drain_vocab.items():
        drain_classes[v] = k

    y_code = np.full(n, -1, dtype=np.int64)
    y_family = np.full(n, -1, dtype=np.int64)
    y_drain = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        pl = R["primary_label"][i]
        if pl:
            y_code[i] = code_vocab[pl]
            y_family[i] = fam_vocab[_family_of(pl)]
        dr = R["drainage"][i]
        if dr:
            y_drain[i] = drain_vocab[str(dr)]

    # ---- auxiliary USCS target: near-surface OCR class per boring (relabeling) ----
    # The topmost stratum (smallest interval_index) is the shallowest -> a surface-comparable
    # USCS family. Different taxonomy from the engineering soil_labels, so it feeds a separate
    # head that shares the encoder; it ~3x's the supervised node count.
    top_uscs = con.execute("""
        SELECT boring_id, uscs_class FROM (
            SELECT boring_id, uscs_class,
                   ROW_NUMBER() OVER (PARTITION BY boring_id ORDER BY interval_index) rn
            FROM strata WHERE uscs_class IS NOT NULL
        ) WHERE rn = 1
    """).fetchall()
    uscs_vocab = {}
    for _bid, u in top_uscs:
        uscs_vocab.setdefault(u, len(uscs_vocab))
    uscs_classes = [None] * len(uscs_vocab)
    for k, v in uscs_vocab.items():
        uscs_classes[v] = k
    boring_index = {nid: i for i, nid in enumerate(node_ids)}
    y_uscs = np.full(n, -1, dtype=np.int64)
    for bid, u in top_uscs:
        ni = boring_index.get("b:" + str(bid))
        if ni is not None:
            y_uscs[ni] = uscs_vocab[u]

    # ---- edges from ml_edges -> integer node indices ----
    index = {nid: i for i, nid in enumerate(node_ids)}
    erows = con.execute(
        "SELECT src, dst, edge_type, weight FROM ml_edges").fetchall()
    et_index = {t: i for i, t in enumerate(EDGE_TYPES)}
    src = np.empty(len(erows) * 2, dtype=np.int64)
    dst = np.empty(len(erows) * 2, dtype=np.int64)
    etype = np.empty(len(erows) * 2, dtype=np.int64)
    eweight = np.empty(len(erows) * 2, dtype=np.float32)
    k = 0
    for s, d, t, w in erows:
        si, di, ti = index[s], index[d], et_index.get(t, 0)
        # undirected -> add both directions for message passing
        src[k], dst[k], etype[k], eweight[k] = si, di, ti, w
        src[k + 1], dst[k + 1], etype[k + 1], eweight[k + 1] = di, si, ti, w
        k += 2
    edge_index = torch.from_numpy(np.stack([src, dst]))
    edge_type = torch.from_numpy(etype)
    edge_weight = torch.from_numpy(eweight)

    ds = Dataset(
        node_ids=node_ids, node_type=node_type, xy=xy,
        x_num=torch.from_numpy(x_num), x_mask=torch.from_numpy(x_mask),
        cat_idx=torch.from_numpy(cat_idx),
        y_code=torch.from_numpy(y_code), y_family=torch.from_numpy(y_family),
        y_drain=torch.from_numpy(y_drain), y_uscs=torch.from_numpy(y_uscs),
        edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,
        vocabs=vocabs, code_classes=code_classes, family_classes=family_classes,
        drain_classes=drain_classes, uscs_classes=uscs_classes,
        cat_cardinalities=cat_cardinalities,
    )
    if log:
        log.info("dataset_built", nodes=n, labels=int((node_type == 1).sum()),
                 ocr_uscs_borings=int((y_uscs >= 0).sum()), uscs_classes=len(uscs_classes),
                 edges=int(edge_index.shape[1]), codes=len(code_classes),
                 families=len(family_classes), drains=len(drain_classes),
                 norm={"x_mean": x_mean.tolist(), "x_scale": float(x_scale),
                       "elev_mean": float(elev_mean), "elev_std": float(elev_std)})
    return ds


def build_and_cache(config: Config, log=None) -> Dataset:
    con = db.connect(config, read_only=True)
    ds = build_dataset(con, config, log)
    out = config.abspath(config.get("ml", "out_dir", default="data/ml"))
    out.mkdir(parents=True, exist_ok=True)
    ds.save(out / "dataset.pt")
    with open(out / "vocab.json", "w") as fh:
        json.dump({"cat_vocabs": ds.vocabs, "code_classes": ds.code_classes,
                   "family_classes": ds.family_classes, "drain_classes": ds.drain_classes},
                  fh, indent=2)
    con.close()
    return ds
