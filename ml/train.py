"""Train + spatially-cross-validate the soil-type GNN.

Modes:
  a1  deterministic GraphSAGE (sample=False, KL off) — baseline + warm-start
  a2  Bayesian (Bayes-by-Backprop, ELBO, KL annealed), warm-started from a1
  a3  a2 + empirical-Bayes geology prior on the class logits

Full-graph training (69k nodes fits on 12 GB); loss masked to the train labels of each
spatial-block fold. Evaluation averages T posterior samples for calibrated probabilities.

Run: python -m ml.train --mode a3 --folds 5
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
from .data import EDGE_TYPES, Dataset
from .model import SoilGNN, build_rel_index
from .priors import geology_prior_logits
from .splits import kfold_block_split


def _device(cfg):
    want = cfg.get("ml", "device", default="cuda")
    return "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"


def _class_weights(y, n, device, beta=0.99):
    # class-balanced by effective number of samples (Cui et al.): gentler than full inverse-
    # frequency, so the long tail is up-weighted without wrecking calibration on the head classes.
    c = np.bincount(y[y >= 0], minlength=n).astype(np.float64)
    eff = (1.0 - np.power(beta, np.clip(c, 1, None))) / (1.0 - beta)
    w = 1.0 / eff
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def focal_ce(logits, target, weight, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    pt = p[torch.arange(len(target)), target]
    loss = -((1 - pt) ** gamma) * logp[torch.arange(len(target)), target]
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def train_one_fold(ds: Dataset, cfg, *, mode, test_fold, val_fold, fold, device,
                   warm_state=None, log=None, active_rel=None, aux=False):
    mlc = cfg["ml"]
    tr_cfg = mlc["train"]
    n = ds.x_num.shape[0]
    bayesian = mode in ("a2", "a3")
    use_prior = mode == "a3"

    label = (ds.y_code.numpy() >= 0)
    train_mask = label & (fold != test_fold) & (fold != val_fold)
    val_mask = label & (fold == val_fold)
    test_mask = label & (fold == test_fold)

    # move tensors to device
    x_num = ds.x_num.to(device); x_mask = ds.x_mask.to(device); cat_idx = ds.cat_idx.to(device)
    y_code = ds.y_code.to(device); y_fam = ds.y_family.to(device); y_dr = ds.y_drain.to(device)
    y_uscs = ds.y_uscs.to(device)
    rel_index = build_rel_index(ds.edge_index, ds.edge_type, len(EDGE_TYPES), device)

    # auxiliary USCS supervision: OCR'd borings in TRAIN folds only (no test/val leakage)
    aux_idx = None
    if aux and len(ds.uscs_classes) > 0:
        aux_mask = (ds.node_type == 0) & (ds.y_uscs.numpy() >= 0) \
            & (fold != test_fold) & (fold != val_fold)
        if aux_mask.any():
            aux_idx = torch.from_numpy(np.where(aux_mask)[0]).to(device)

    prior_logits = None
    if use_prior:
        pl = geology_prior_logits(ds, train_mask,
                                  weight=float(cfg.get("ml", "priors", "geology_prior_weight", default=1.0)))
        prior_logits = torch.from_numpy(pl).to(device)

    model = SoilGNN(
        cat_cardinalities=ds.cat_cardinalities, num_dim=ds.x_num.shape[1] + ds.x_mask.shape[1],
        n_codes=len(ds.code_classes), n_families=len(ds.family_classes),
        n_drains=len(ds.drain_classes), edge_types=EDGE_TYPES,
        hidden=int(mlc["model"]["hidden"]), layers=int(mlc["model"]["layers"]),
        dropout=float(mlc["model"]["dropout"]),
        prior_sigma=float(mlc["model"]["variational"]["prior_sigma"]),
        n_uscs=len(ds.uscs_classes) if aux else 0,
    ).to(device)
    if warm_state is not None:
        model.load_state_dict(warm_state, strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["lr"]),
                           weight_decay=float(tr_cfg["weight_decay"]))
    cw = _class_weights(ds.y_code.numpy(), len(ds.code_classes), device)
    famw = _class_weights(ds.y_family.numpy(), len(ds.family_classes), device)
    epochs = int(tr_cfg["epochs"])
    anneal = max(1, int(epochs * float(tr_cfg["kl_anneal_frac"])))
    n_train = int(train_mask.sum())
    tr_idx = torch.from_numpy(np.where(train_mask)[0]).to(device)

    best_val, best_state, best_metrics = -1.0, None, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        fam_l, code_l, dr_l, uscs_l = model(x_num, x_mask, cat_idx, rel_index,
                                            sample=bayesian, prior_logits=prior_logits,
                                            active_rel=active_rel)
        loss = (focal_ce(code_l[tr_idx], y_code[tr_idx], cw, float(tr_cfg["focal_gamma"]))
                + 0.5 * focal_ce(fam_l[tr_idx], y_fam[tr_idx], famw, 0.0))
        dr_t = y_dr[tr_idx]
        dmask = dr_t >= 0
        if dmask.any():
            loss = loss + 0.3 * F.cross_entropy(dr_l[tr_idx][dmask], dr_t[dmask])
        if aux_idx is not None and uscs_l is not None:
            loss = loss + 0.5 * F.cross_entropy(uscs_l[aux_idx], y_uscs[aux_idx])
        if bayesian:
            beta = float(mlc["model"]["variational"]["kl_weight"]) * min(1.0, (ep + 1) / anneal)
            loss = loss + beta * model.kl() / max(1, n_train)
        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            vm = evaluate(model, ds, x_num, x_mask, cat_idx, rel_index, prior_logits,
                          val_mask, device, bayesian,
                          T=int(mlc["model"]["variational"]["mc_samples_eval"]) if bayesian else 1,
                          active_rel=active_rel)
            if vm["macro_f1"] > best_val:
                best_val = vm["macro_f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = vm
            if log:
                log.info("epoch", mode=mode, fold=test_fold, ep=ep + 1,
                         loss=round(float(loss), 4), val_f1=round(vm["macro_f1"], 4),
                         val_ece=round(vm["ece"], 4))

    # final test with best (by val) weights
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, ds, x_num, x_mask, cat_idx, rel_index, prior_logits,
                            test_mask, device, bayesian,
                            T=int(mlc["model"]["variational"]["mc_samples_eval"]) if bayesian else 1,
                            active_rel=active_rel)
    return {"val": best_metrics, "test": test_metrics, "state": best_state,
            "n_train": n_train, "n_test": int(test_mask.sum())}


@torch.no_grad()
def evaluate(model, ds, x_num, x_mask, cat_idx, rel_index, prior_logits, mask, device,
             bayesian, T=30, active_rel=None):
    model.eval()
    idx = torch.from_numpy(np.where(mask)[0]).to(device)
    if idx.numel() == 0:
        return {"macro_f1": 0.0, "ece": 1.0, "nll": 99.0, "accuracy": 0.0, "balanced_acc": 0.0}
    probs = torch.zeros(idx.shape[0], len(ds.code_classes), device=device)
    reps = T if bayesian else 1
    for _ in range(reps):
        out = model(x_num, x_mask, cat_idx, rel_index, sample=bayesian,
                    prior_logits=prior_logits, active_rel=active_rel)
        code_l = out[1]
        probs += F.softmax(code_l[idx], dim=1)
    probs /= reps
    y = ds.y_code.numpy()[mask]
    return ev.classification_metrics(probs.cpu().numpy(), y)


def run_cv(cfg, log, mode="a3", folds=5, active_rel=None, tag=None, aux=False):
    device = _device(cfg)
    out = cfg.abspath(cfg.get("ml", "out_dir", default="data/ml"))
    ds = Dataset.load(out / "dataset.pt")
    seed = int(cfg.get("ml", "seed", default=1337))
    torch.manual_seed(seed); np.random.seed(seed)

    label = ds.y_code.numpy() >= 0
    fold = kfold_block_split(ds.xy, ds.y_code.numpy(), seed=seed,
                             block_size_ft=float(cfg.get("ml", "splits", "block_size_ft", default=20000)),
                             folds=folds)
    log.info("cv_start", mode=mode, device=device, folds=folds,
             labels=int(label.sum()),
             fold_sizes=[int((label & (fold == f)).sum()) for f in range(folds)])

    # warm-start: a2/a3 reuse a quick a1 deterministic fit per fold
    results = []
    for tf in range(folds):
        vf = (tf + 1) % folds
        t0 = time.time()
        warm = None
        if mode in ("a2", "a3"):
            a1 = train_one_fold(ds, cfg, mode="a1", test_fold=tf, val_fold=vf, fold=fold,
                                device=device, log=None, active_rel=active_rel, aux=aux)
            warm = a1["state"]
        r = train_one_fold(ds, cfg, mode=mode, test_fold=tf, val_fold=vf, fold=fold,
                           device=device, warm_state=warm, log=log, active_rel=active_rel, aux=aux)
        r["fold"] = tf; r["secs"] = round(time.time() - t0, 1)
        del r["state"]
        results.append(r)
        log.info("fold_done", mode=mode, fold=tf, secs=r["secs"],
                 test_f1=round(r["test"]["macro_f1"], 4),
                 test_balacc=round(r["test"]["balanced_acc"], 4),
                 test_ece=round(r["test"]["ece"], 4), test_nll=round(r["test"]["nll"], 4))

    agg = _aggregate(results)
    name = tag or mode
    (out / f"cv_{name}.json").write_text(json.dumps(
        {"mode": mode, "tag": name,
         "active_rel": sorted(active_rel) if active_rel else None,
         "folds": results, "mean": agg}, indent=2))
    log.info("cv_done", mode=mode, tag=name, **{f"mean_{k}": round(v, 4) for k, v in agg.items()})
    return {"mode": mode, "results": results, "mean": agg}


def _aggregate(results):
    keys = ["macro_f1", "balanced_acc", "accuracy", "nll", "ece"]
    agg = {}
    for k in keys:
        vals = [r["test"].get(k, float("nan")) for r in results]
        agg[k] = float(np.nanmean(vals))
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["a1", "a2", "a3"], default="a3")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--edges", default=None,
                    help="comma list of active edge-type indices for ablation "
                         "(0=knn,1=delaunay,2=same_geology,3=label_boring)")
    ap.add_argument("--tag", default=None, help="output name override (cv_<tag>.json)")
    ap.add_argument("--aux", action="store_true",
                    help="add the auxiliary USCS task on OCR'd borings (relabeling)")
    args = ap.parse_args()
    active_rel = set(int(x) for x in args.edges.split(",")) if args.edges else None
    cfg = Config.load(None)
    rid = new_run_id()
    logger = setup(cfg.path("log_dir"), "ml.log", rid, f"train_{args.mode}", console=True)
    res = run_cv(cfg, logger, mode=args.mode, folds=args.folds,
                 active_rel=active_rel, tag=args.tag, aux=args.aux)
    print(f"\n=== CV {args.mode} (spatial {args.folds}-fold) ===")
    for r in res["results"]:
        t = r["test"]
        print(f"  fold {r['fold']}: macroF1={t['macro_f1']:.3f} balAcc={t['balanced_acc']:.3f} "
              f"acc={t['accuracy']:.3f} ECE={t['ece']:.3f} NLL={t['nll']:.3f} ({r['secs']}s)")
    m = res["mean"]
    print(f"  MEAN: macroF1={m['macro_f1']:.3f} balAcc={m['balanced_acc']:.3f} "
          f"acc={m['accuracy']:.3f} ECE={m['ece']:.3f} NLL={m['nll']:.3f}")
