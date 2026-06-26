"""B1 — train + spatially-cross-validate the 3D depth-resolved Bayesian model.

The GraphSAGE encoder makes a per-boring spatial latent; a depth-conditioned decoder predicts
SPT-N (heteroscedastic Gaussian, log1p space), USCS-at-depth, and groundwater. Spatial-block CV
over borings (whole regions held out). The headline is calibrated SPT-N intervals (CRPS, 90%
coverage) and beating a geology+depth baseline — proving the depth signal is non-redundant.

Run: python -m ml.train3d --folds 5
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.config import Config
from pipeline.logging_setup import new_run_id, setup

from . import eval as ev
from .data import EDGE_TYPES
from .data3d import Dataset3D
from .model import SoilGNN3D, build_rel_index
from .splits import kfold_block_split


def _device(cfg):
    want = cfg.get("ml", "device", default="cuda")
    return "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"


def _gauss_nll(mu, logvar, y):
    return 0.5 * (logvar + (y - mu) ** 2 / logvar.exp()).mean()


def _boring_folds(d3, folds, seed, block_ft):
    """Fold per sample, assigned by the sample's boring xy (whole borings stay together)."""
    nodes = d3.sample_node.numpy()
    xy = d3.xy[nodes]
    # block on the boring location; labels arg unused for stratification here
    return kfold_block_split(xy, np.zeros(len(nodes), dtype=np.int64),
                             block_size_ft=block_ft, folds=folds, seed=seed)


def train_eval(d3, cfg, device, test_fold, val_fold, fold, log=None, physics=False, geotech=False,
               dump_preds=False):
    mlc = cfg["ml"]; tr = mlc["train"]
    x_num = d3.x_num.to(device); x_mask = d3.x_mask.to(device); cat_idx = d3.cat_idx.to(device)
    rel = build_rel_index(d3.edge_index, d3.edge_type, len(EDGE_TYPES), device)
    s_node = d3.sample_node.to(device)
    s_depth = d3.sample_depth_std.to(device).unsqueeze(1)
    y_spt = d3.y_spt_log.to(device); y_uscs = d3.y_uscs.to(device)

    # Optional per-sample side inputs, both routed through the decoder's phys channel (value+mask):
    #   physics — non-leaky σ'v0/σv0/γ/CN from strata_derived (B2)
    #   geotech — literature USCS-keyed PI/fines/LL/k/K0/Cr/granular (non-leaky for spt_n)
    # They are concatenated into one block; the model only sees the combined phys_dim.
    blocks, masks = [], []
    if bool(physics) and getattr(d3, "sample_phys", None) is not None:
        blocks.append(d3.sample_phys); masks.append(d3.sample_phys_mask)
    if bool(geotech) and getattr(d3, "sample_geo", None) is not None:
        blocks.append(d3.sample_geo); masks.append(d3.sample_geo_mask)
    use_phys = bool(blocks)
    s_phys = torch.cat(blocks, dim=1).to(device) if use_phys else None
    s_phys_mask = torch.cat(masks, dim=1).to(device) if use_phys else None
    phys_dim = s_phys.shape[1] if use_phys else 0

    tr_m = (fold != test_fold) & (fold != val_fold)
    te_m = fold == test_fold
    tr_idx = torch.from_numpy(np.where(tr_m)[0]).to(device)
    te_idx = torch.from_numpy(np.where(te_m)[0]).to(device)

    model = SoilGNN3D(
        cat_cardinalities=d3.cat_cardinalities, num_dim=d3.x_num.shape[1] + d3.x_mask.shape[1],
        edge_types=EDGE_TYPES, n_uscs=max(1, len(d3.uscs_classes)),
        hidden=int(mlc["model"]["hidden"]), layers=int(mlc["model"]["layers"]),
        dropout=float(mlc["model"]["dropout"]),
        prior_sigma=float(mlc["model"]["variational"]["prior_sigma"]),
        phys_dim=phys_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]),
                           weight_decay=float(tr["weight_decay"]))
    epochs = int(tr["epochs"]); anneal = max(1, int(epochs * float(tr["kl_anneal_frac"])))
    n_tr = int(tr_m.sum())

    def _phys(idx):
        return (s_phys[idx], s_phys_mask[idx]) if use_phys else (None, None)

    for ep in range(epochs):
        model.train(); opt.zero_grad()
        h = model.encode(x_num, x_mask, cat_idx, rel, sample=True)
        p, pm = _phys(tr_idx)
        spt, uscs = model.decode(h[s_node[tr_idx]], s_depth[tr_idx], phys=p, phys_mask=pm, sample=True)
        loss = torch.zeros((), device=device)
        ys = y_spt[tr_idx]; m = ys >= 0
        if m.any():
            loss = loss + _gauss_nll(spt[m, 0], spt[m, 1], ys[m])
        yu = y_uscs[tr_idx]; mu_ = yu >= 0
        if mu_.any():
            loss = loss + F.cross_entropy(uscs[mu_], yu[mu_])
        beta = float(mlc["model"]["variational"]["kl_weight"]) * min(1.0, (ep + 1) / anneal)
        loss = loss + beta * model.kl() / max(1, n_tr)
        loss.backward(); opt.step()

    return _evaluate(model, d3, x_num, x_mask, cat_idx, rel, s_node, s_depth, y_spt, y_uscs,
                     te_idx, device, T=int(mlc["model"]["variational"]["mc_samples_eval"]),
                     s_phys=s_phys, s_phys_mask=s_phys_mask, use_phys=use_phys,
                     dump_preds=dump_preds)


@torch.no_grad()
def _evaluate(model, d3, x_num, x_mask, cat_idx, rel, s_node, s_depth, y_spt, y_uscs,
              te_idx, device, T=30, s_phys=None, s_phys_mask=None, use_phys=False, dump_preds=False):
    model.eval()
    if te_idx.numel() == 0:
        return {}
    p_te = s_phys[te_idx] if use_phys else None
    pm_te = s_phys_mask[te_idx] if use_phys else None
    mus, vars, probs = [], [], []
    for _ in range(T):
        h = model.encode(x_num, x_mask, cat_idx, rel, sample=True)
        spt, uscs = model.decode(h[s_node[te_idx]], s_depth[te_idx],
                                 phys=p_te, phys_mask=pm_te, sample=True)
        mus.append(spt[:, 0]); vars.append(spt[:, 1].exp())
        probs.append(F.softmax(uscs, dim=1))
    mu = torch.stack(mus); aleatoric = torch.stack(vars)
    pred_mu = mu.mean(0)
    pred_var = aleatoric.mean(0) + mu.var(0)        # total = aleatoric + epistemic
    prob = torch.stack(probs).mean(0)

    res = {}
    ys = y_spt[te_idx]; m = (ys >= 0).cpu().numpy()
    if m.any():
        # metrics in log1p space AND back-transformed blow counts
        rl = ev.regression_metrics(pred_mu.cpu().numpy()[m], pred_var.sqrt().cpu().numpy()[m],
                                   ys.cpu().numpy()[m])
        n_pred = np.expm1(pred_mu.cpu().numpy()[m]); n_true = np.expm1(ys.cpu().numpy()[m])
        res["spt"] = {**rl, "rmse_blows": float(np.sqrt(((n_pred - n_true) ** 2).mean())),
                      "mae_blows": float(np.abs(n_pred - n_true).mean())}
    yu = y_uscs[te_idx]; mu_ = (yu >= 0).cpu().numpy()
    if mu_.any() and len(d3.uscs_classes) > 1:
        res["uscs"] = ev.classification_metrics(prob.cpu().numpy()[mu_], yu.cpu().numpy()[mu_])
    if dump_preds:
        # per-sample out-of-fold predictions for the gold-set diagnosis (boring_id from node_ids).
        tei = te_idx.cpu().numpy()
        s_node_np = s_node.cpu().numpy()
        res["_preds"] = {
            "boring_id": [str(d3.node_ids[int(s_node_np[i])])[2:] for i in tei],  # strip 'b:'
            "depth_ft": d3.sample_depth_ft.cpu().numpy()[tei],
            "y_ocr_log": y_spt[te_idx].cpu().numpy(),
            "pred_mu": pred_mu.cpu().numpy(),
            "pred_sigma": pred_var.sqrt().cpu().numpy(),
        }
    return res


def baselines(d3, fold, test_fold, val_fold, physics=False, geotech=False):
    """Depth-mean SPT (global) and a geology+depth gradient-boosting regressor. When `physics`/
    `geotech`, the GBM also gets those non-leaky per-sample features — a fair foil for the GNN."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = d3.y_spt_log.numpy(); valid = y >= 0
    tr = (fold != test_fold) & (fold != val_fold) & valid
    te = (fold == test_fold) & valid
    out = {}
    if tr.sum() and te.sum():
        # depth-mean: predict the train mean (constant) -> calibration via train residual std
        mu0 = y[tr].mean(); sd0 = y[tr].std() or 1.0
        out["depth_mean"] = ev.regression_metrics(np.full(te.sum(), mu0),
                                                  np.full(te.sum(), sd0), y[te])
        # geology+depth GBM on node features (no graph) + depth [+ physics/geotech for the foil]
        nodes = d3.sample_node.numpy()
        feats = [d3.cat_idx.numpy()[nodes].astype(float), d3.x_num.numpy()[nodes],
                 d3.sample_depth_std.numpy()[:, None]]
        use_phys = bool(physics) and getattr(d3, "sample_phys", None) is not None
        use_geo = bool(geotech) and getattr(d3, "sample_geo", None) is not None
        parts = ["geology", "depth"]
        if use_phys:
            feats += [d3.sample_phys.numpy(), d3.sample_phys_mask.numpy()]; parts.append("phys")
        if use_geo:
            feats += [d3.sample_geo.numpy(), d3.sample_geo_mask.numpy()]; parts.append("geo")
        X = np.concatenate(feats, axis=1)
        gbm = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        gbm.fit(X[tr], y[tr]); pred = gbm.predict(X[te])
        sd = (y[tr] - gbm.predict(X[tr])).std() or 1.0
        out["_".join(parts) + "_gbm"] = ev.regression_metrics(pred, np.full(te.sum(), sd), y[te])
    return out


def run(cfg, log, folds=5, physics=None, geotech=False, dump_preds=False):
    device = _device(cfg)
    out = cfg.abspath(cfg.get("ml", "out_dir", default="data/ml"))
    d3 = Dataset3D.load(out / "dataset3d.pt")
    if physics is None:
        physics = bool(cfg.get("ml", "b1", "physics_features", default=False))
    physics = physics and getattr(d3, "sample_phys", None) is not None
    geotech = bool(geotech) and getattr(d3, "sample_geo", None) is not None
    tag = ("b2" if physics else "b1") + ("g" if geotech else "")
    seed = int(cfg.get("ml", "seed", default=1337)); torch.manual_seed(seed); np.random.seed(seed)
    block_ft = float(cfg.get("ml", "splits", "block_size_ft", default=20000))
    fold = _boring_folds(d3, folds, seed, block_ft)
    log.info(f"{tag}_start", device=device, samples=int(len(fold)), physics=physics, geotech=geotech,
             phys_cols=getattr(d3, "phys_cols", None), geo_cols=getattr(d3, "geo_cols", None),
             with_spt=int((d3.y_spt_log.numpy() >= 0).sum()), folds=folds)

    model_res, base_res, all_preds = [], [], []
    for tf in range(folds):
        vf = (tf + 1) % folds; t0 = time.time()
        r = train_eval(d3, cfg, device, tf, vf, fold, log, physics=physics, geotech=geotech,
                       dump_preds=dump_preds)
        b = baselines(d3, fold, tf, vf, physics=physics, geotech=geotech)
        if dump_preds and "_preds" in r:
            p = r.pop("_preds"); p["fold"] = [tf] * len(p["boring_id"]); all_preds.append(p)
        model_res.append(r); base_res.append(b)
        spt = r.get("spt", {})
        log.info(f"{tag}_fold", fold=tf, secs=round(time.time() - t0, 1),
                 spt_crps=round(spt.get("crps", float("nan")), 4),
                 spt_cov90=round(spt.get("cov90", float("nan")), 3),
                 spt_rmse_blows=round(spt.get("rmse_blows", float("nan")), 1),
                 uscs_f1=round(r.get("uscs", {}).get("macro_f1", float("nan")), 3))

    agg = _agg(model_res)
    agg_b = {}
    for k in {kk for b in base_res for kk in b}:  # union of baseline names seen across folds
        dicts = [b[k] for b in base_res if k in b]
        if dicts:
            agg_b[k] = {kk: float(np.nanmean([d[kk] for d in dicts])) for kk in dicts[0]}
    base = "cv_b1_physics" if physics else "cv_b1"
    if geotech:
        base += "_geotech"
    (out / f"{base}.json").write_text(json.dumps(
        {"model": {"folds": model_res, "mean": agg}, "baselines": agg_b,
         "physics": physics, "geotech": geotech}, indent=2))
    if dump_preds and all_preds:
        # concatenate fold OOF predictions -> preds_b{1,2}[_geotech].json (gitignored; gold_diag).
        # columnar JSON keeps this dependency-free (no pandas/pyarrow in this env).
        cols = {}
        for p in all_preds:
            for k, v in p.items():
                cols.setdefault(k, []).extend(np.asarray(v).tolist())
        pbase = "preds_b2" if physics else "preds_b1"
        if geotech:
            pbase += "_geotech"
        (out / f"{pbase}.json").write_text(json.dumps(cols))
        log.info(f"{tag}_preds", rows=len(cols["boring_id"]), file=f"{pbase}.json")
    log.info(f"{tag}_done", **{f"spt_{k}": round(v, 4) for k, v in agg.get("spt", {}).items()})
    return {"model": agg, "baselines": agg_b, "physics": physics, "geotech": geotech}


def _agg(res_list):
    out = {}
    for grp in ("spt", "uscs"):
        ds = [r[grp] for r in res_list if grp in r]
        if ds:
            out[grp] = {k: float(np.nanmean([d[k] for d in ds])) for k in ds[0]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--physics", action="store_true",
                    help="B2: add non-leaky σ'v0/σv0/γ/CN per-sample inputs (-> cv_b1_physics.json)")
    ap.add_argument("--geotech", action="store_true",
                    help="add literature USCS-keyed geotech inputs (PI/fines/LL/k/K0/Cr/granular); "
                         "non-leaky for SPT-N, but leaks the USCS@depth head (-> *_geotech.json)")
    ap.add_argument("--dump-preds", action="store_true",
                    help="also write per-sample OOF predictions to preds_b{1,2}.parquet (gold diag)")
    args = ap.parse_args()
    cfg = Config.load(None); rid = new_run_id()
    logger = setup(cfg.path("log_dir"), "ml.log", rid, "train3d", console=True)
    res = run(cfg, logger, folds=args.folds, physics=True if args.physics else None,
              geotech=args.geotech, dump_preds=args.dump_preds)
    s = res["model"].get("spt", {}); u = res["model"].get("uscs", {})
    label = ("B2 (+physics)" if res.get("physics") else "B1") + (" +geotech" if res.get("geotech") else "")
    print(f"\n=== {label} 3D depth-resolved (spatial CV) ===")
    print(f"  SPT-N: CRPS={s.get('crps',float('nan')):.3f} cov90={s.get('cov90',float('nan')):.3f} "
          f"RMSE={s.get('rmse_blows',float('nan')):.1f} MAE={s.get('mae_blows',float('nan')):.1f} blows (n={s.get('n','?')})")
    uscs_note = "  (leaky under --geotech; ignore)" if res.get("geotech") else ""
    print(f"  USCS@depth: macroF1={u.get('macro_f1',float('nan')):.3f} acc={u.get('accuracy',float('nan')):.3f}{uscs_note}")
    for name, sp in res["baselines"].items():
        print(f"  baseline {name:18}: CRPS={sp.get('crps',float('nan')):.3f} "
              f"cov90={sp.get('cov90',float('nan')):.3f} RMSE(log)={sp.get('rmse',float('nan')):.3f}")
