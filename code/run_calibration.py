import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

from conformal import ACI, band_ref_stats, dsrc_mahalanobis, relative_divergence
from data import prepare_dataset
from losses import band_energies, band_saliency_loss, moment_match_loss, rec_loss, relation_consistency_loss
from metrics import auprc, auroc, fpr_on_normal, pearson, preds_range_f1
from model import PatchTSTRecon

SLUG = "scc-calibration-ablation"

TRAIN_CONFIGS = {
    "v2_plain": dict(f1=0, f2=0, f3=0, gating=0),
    "v2_gated": dict(f1=0, f2=0, f3=0, gating=1),
    "v1_full_control": dict(f1=1, f2=1, f3=1, gating=1),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, device):
    m = cfg["model"]
    return PatchTSTRecon(
        c_in=2,
        window=cfg["data"]["window"],
        patch_len=m["patch_len"],
        patch_stride=m["patch_stride"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        e_layers=m["e_layers"],
        d_ff=m["d_ff"],
        dropout=m["dropout"],
        n_bands=cfg["data"]["n_bands"],
    ).to(device)


@torch.no_grad()
def score_model(model, x, cfg, device):
    out = []
    bs = 256
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    model.eval()
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            rec, _, _ = model(xb)
        out.append(((rec - xb) ** 2).mean(dim=(1, 2)).float().cpu())
    return torch.cat(out).numpy()


def train_model(tflags, cfg, src_groups, tgt_pool, device):
    t = cfg["train"]
    m = cfg["model"]
    nb = cfg["data"]["n_bands"]
    d_model = m["d_model"]
    model = build_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["max_steps"])
    amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp)
    gen = np.random.default_rng(cfg["seed"] + 11)
    bs = t["batch_size"]
    sizes = np.array([g["windows"].shape[0] for g in src_groups], dtype=float)
    gw = np.array(cfg["_gating_weights"], dtype=float)
    probs = gw * sizes
    probs = probs / probs.sum()
    losses = []
    t0 = time.time()
    model.train()
    for step in range(1, t["max_steps"] + 1):
        gidx = gen.choice(len(src_groups), size=bs, p=probs)
        src = torch.empty(bs, 2, cfg["data"]["window"], dtype=torch.float32)
        for gi in np.unique(gidx):
            mask = np.where(gidx == gi)[0]
            idx = gen.choice(int(sizes[gi]), size=len(mask), replace=False)
            src[mask] = src_groups[gi]["windows"][torch.from_numpy(idx)]
        ti = gen.choice(tgt_pool.shape[0], size=bs, replace=False)
        tgt = tgt_pool[torch.from_numpy(ti)]
        src = src.to(device)
        tgt = tgt.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            rec_s, rel_s, band_s = model(src)
            loss = rec_loss(rec_s, src)
            if tflags["f1"] or tflags["f2"]:
                _, rel_t, band_t = model(tgt)
            if tflags["f1"]:
                loss = loss + t["lambda_inv"] * moment_match_loss(band_s.reshape(-1, d_model), band_t.reshape(-1, d_model))
            if tflags["f2"]:
                loss = loss + t["lambda_rel"] * relation_consistency_loss(rel_s, rel_t)
            if tflags["f3"]:
                loss = loss + t["lambda_sal"] * band_saliency_loss(src, rec_s, nb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(float(loss.detach()))
        if step % t["log_every"] == 0 or step == t["max_steps"]:
            print(f"  step {step}/{t['max_steps']} loss {np.mean(losses[-200:]):.5f} elapsed {time.time() - t0:.0f}s", flush=True)
    model.eval()
    return model, {"loss_last_mean": float(np.mean(losses[-100:])), "nan": bool(np.isnan(losses).any()),
                   "elapsed_sec": float(time.time() - t0)}


def group_band_mean(g, cfg, rng):
    nb = cfg["data"]["n_bands"]
    x = g["windows"]
    n = x.shape[0]
    idx = np.sort(rng.choice(n, size=min(1024, n), replace=False)) if n > 1024 else np.arange(n)
    e = band_energies(x[torch.from_numpy(idx)], nb).float()
    return e.mean(0)


def compute_gating(cfg, src_groups, tgt_ref):
    rng = np.random.default_rng(cfg["seed"] + 5)
    mu_t = group_band_mean(tgt_ref, cfg, rng)
    ds = [float(torch.norm(group_band_mean(g, cfg, rng) - mu_t)) for g in src_groups]
    ds = np.array(ds)
    w = np.exp(-cfg["gating"]["tau"] * ds / max(ds.mean(), 1e-6))
    return w.tolist(), ds.tolist()


def score_divergence(scores, q_source, window, cap):
    rm = pd.Series(scores).rolling(window, min_periods=1).median().to_numpy()
    d = np.maximum(0.0, rm / max(q_source, 1e-9) - 1.0)
    return np.clip(d, 0.0, cap)


def conformal_eval(scores, labels, q_source, alpha, mode, d_norm, gamma, lam=0.0):
    n = len(scores)
    if mode == "static":
        preds = scores > q_source
        qs = np.full(n, q_source)
    else:
        aci = ACI(q_source, alpha, gamma, lam if mode == "shift" else 0.0)
        preds = np.empty(n, dtype=bool)
        qs = np.empty(n)
        for i in range(n):
            q_t = aci.threshold(d_norm[i] if mode == "shift" else 0.0)
            qs[i] = q_t
            preds[i] = bool(scores[i] > q_t)
            aci.update(scores[i], q_t)
    normal = labels < 0.5
    fpr = fpr_on_normal(preds, normal)
    cov = float(1 - fpr) if not np.isnan(fpr) else float("nan")
    res = preds_range_f1(preds, labels)
    res.update({
        "coverage": cov,
        "abs_coverage_gap": abs(cov - (1 - alpha)) if not np.isnan(cov) else float("nan"),
        "empirical_fpr_at_alpha": fpr,
        "inflation": float(np.mean(qs) / max(q_source, 1e-9)),
        "mean_threshold": float(np.mean(qs)),
        "alarm_rate": float(np.mean(preds)),
        "q_source": float(q_source),
    })
    return res


def eval_grid(scores, labels, src_cal_scores, d_band, cfg):
    out = {}
    gamma = cfg["conformal"]["aci_gamma"]
    for alpha in cfg["alphas"]:
        q_source = float(np.quantile(src_cal_scores, 1 - alpha))
        d_score = score_divergence(scores, q_source, cfg["conformal"]["score_div_window"], cfg["conformal"]["score_div_cap"])
        out[f"static_a{alpha:g}"] = conformal_eval(scores, labels, q_source, alpha, "static", None, gamma)
        out[f"aci_a{alpha:g}"] = conformal_eval(scores, labels, q_source, alpha, "aci", None, gamma)
        for lam in cfg["lambdas"]:
            out[f"shiftband_l{lam:g}_a{alpha:g}"] = conformal_eval(scores, labels, q_source, alpha, "shift", d_band, gamma, lam)
            out[f"shiftscore_l{lam:g}_a{alpha:g}"] = conformal_eval(scores, labels, q_source, alpha, "shift", d_score, gamma, lam)
    return out


def cross_fold_corr(results, cfg):
    rows = []
    for tcfg in cfg["train_configs"]:
        for alpha in cfg["alphas"]:
            for lam in cfg["lambdas"]:
                for div in ("shiftband", "shiftscore"):
                    key = f"{div}_l{lam:g}_a{alpha:g}"
                    xs, ys = [], []
                    for fold in results["folds"]:
                        for seed in results["seeds"]:
                            fdata = results["folds"][fold]
                            if seed not in fdata or tcfg not in fdata[seed]:
                                continue
                            r = fdata[seed][tcfg]
                            if key in r["evals"]:
                                xs.append(r["d_means"][div.split("shift")[1]])
                                ys.append(r["evals"][key]["inflation"])
                    r_, p_ = pearson(xs, ys)
                    rows.append({"train": tcfg, "eval": key, "pearson": r_, "p": p_, "n": len(xs)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default=None, help="comma list, default from config")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--trains", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        cfg["train"]["max_steps"] = 10
        cfg["train"]["batch_size"] = 8
        cfg["train"]["log_every"] = 5
        cfg["data"]["max_windows_per_dataset"] = 300
        cfg["seeds"] = [0]
        cfg["alphas"] = [0.1]
        cfg["lambdas"] = [0.05]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    folds = cfg["folds"]
    if args.folds:
        want = set(args.folds.split(","))
        folds = [f for f in folds if f["name"] in want]
    if args.trains:
        want = set(args.trains.split(","))
        cfg["train_configs"] = [t for t in cfg["train_configs"] if t in want]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    needed = set()
    for fold in folds:
        needed.add(fold["target"])
        needed.update(fold["sources"])
    groups = {}
    for name in sorted(needed):
        offset = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
        rng_d = np.random.default_rng(0 * 7919 + offset)
        groups[name] = prepare_dataset(name, cfg, rng_d)
        g = groups[name]
        print(f"[data] {name}: {tuple(g['windows'].shape)} anomalies={int(g['labels'].sum())}", flush=True)

    results = {"slug": SLUG, "smoke": bool(args.smoke), "device": str(device), "torch": torch.__version__,
               "seeds": seeds, "folds": {}}

    for fold in folds:
        fname = fold["name"]
        tgt = groups[fold["target"]]
        src_groups = [groups[s] for s in fold["sources"]]
        src_train, src_cal_list = [], []
        for g in src_groups:
            lab = g["labels"]
            nm = np.where(lab < 0.5)[0]
            wnorm = g["windows"][torch.from_numpy(nm)]
            nc = max(int(wnorm.shape[0] * cfg["conformal"]["cal_ratio"]), 10)
            src_cal_list.append(wnorm[:nc])
            src_train.append({"name": g["name"], "windows": wnorm[nc:]})
        src_cal = torch.cat(src_cal_list)
        Nt = tgt["windows"].shape[0]
        gen = np.random.default_rng(cfg["seed"] + 17)
        pool_idx = np.sort(gen.choice(Nt, size=min(8000, Nt), replace=False))
        tgt_pool = tgt["windows"][torch.from_numpy(pool_idx)]
        ref_n = max(int(Nt * cfg["conformal"]["cal_ratio"]), 10)
        tgt_ref = {"windows": tgt["windows"][:ref_n]}
        gw_on, ds = compute_gating(cfg, src_groups, tgt_ref)
        e_ref = band_energies(src_cal[: cfg["conformal"]["d_ref_size"]], cfg["data"]["n_bands"])
        ref_stats = band_ref_stats(e_ref)

        normal_idx = np.where(tgt["labels"] < 0.5)[0]
        cal_n = max(int(len(normal_idx) * cfg["conformal"]["cal_ratio"]), 10)
        cal_idx = normal_idx[:cal_n]
        eval_mask = np.ones(Nt, dtype=bool)
        eval_mask[cal_idx] = False
        xe = tgt["windows"][eval_mask]
        le = tgt["labels"][eval_mask].copy()
        if tgt["name"] == "pv":
            rng_inj = np.random.default_rng(cfg["seed"] + 23)
            n = xe.shape[0]
            k = int(n * cfg["data"]["pv"]["injection_fraction"])
            inj = rng_inj.choice(n, size=k, replace=False)
            L = xe.shape[2]
            for i in inj:
                typ = int(rng_inj.integers(0, 3))
                if typ == 0:
                    for _ in range(3):
                        p = int(rng_inj.integers(0, L))
                        ch = int(rng_inj.integers(0, 2))
                        xe[i, ch, p] += 6.0 if rng_inj.random() > 0.5 else -6.0
                elif typ == 1:
                    xe[i] += 4.0 if rng_inj.random() > 0.5 else -4.0
                else:
                    a = int(rng_inj.integers(0, max(L - L // 4, 1)))
                    b = a + int(rng_inj.integers(L // 8, max(L // 8 + 1, L // 3)))
                    xe[i, :, a:b] = 0.0
                le[i] = 1.0
        results["folds"].setdefault(fname, {"target": fold["target"], "gating_dists": ds, "gating_weights": gw_on})

        for seed in seeds:
            cfg["seed"] = seed
            set_seed(seed)
            gen_s = np.random.default_rng(seed + 17)
            pool_idx = np.sort(gen_s.choice(Nt, size=min(8000, Nt), replace=False))
            tgt_pool = tgt["windows"][torch.from_numpy(pool_idx)]
            for tname in cfg["train_configs"]:
                tflags = TRAIN_CONFIGS[tname]
                cfg["_gating_weights"] = gw_on if tflags["gating"] else [1.0] * len(src_train)
                t0 = time.time()
                print(f"[{fname}/seed{seed}/{tname}] training ...", flush=True)
                model, tlog = train_model(tflags, cfg, src_train, tgt_pool, device)
                src_cal_scores = score_model(model, src_cal, cfg, device)
                scores = score_model(model, xe, cfg, device)
                dsrc = dsrc_mahalanobis(ref_stats[0], ref_stats[1], band_energies(xe, cfg["data"]["n_bands"]))
                d_band = relative_divergence(dsrc, cfg["conformal"]["cal_ratio"])
                evals = eval_grid(scores, le, src_cal_scores, d_band, cfg)
                entry = {
                    "auroc": auroc(scores, le),
                    "auprc": auprc(scores, le),
                    "train": tlog,
                    "d_means": {
                        "band": float(np.mean(d_band)),
                        "score": float(np.mean(
                            score_divergence(scores, float(np.quantile(src_cal_scores, 1 - cfg["alphas"][0])),
                                             cfg["conformal"]["score_div_window"], cfg["conformal"]["score_div_cap"]))),
                    },
                    "evals": evals,
                    "elapsed_sec": float(time.time() - t0),
                }
                results["folds"][fname].setdefault(seed, {})[tname] = entry
                print(f"[{fname}/seed{seed}/{tname}] AUROC={entry['auroc']:.3f} "
                      f"cov(shiftband l0.1 a0.1)={evals.get('shiftband_l0.1_a0.1', {}).get('coverage', float('nan')):.3f} "
                      f"cov(static a0.1)={evals.get('static_a0.1', {}).get('coverage', float('nan')):.3f}", flush=True)

    results["cross_fold_corr"] = cross_fold_corr(results, cfg)

    out_dir = os.path.join(cfg["repo_root"], "results", SLUG)
    os.makedirs(out_dir, exist_ok=True)
    for seed in seeds:
        sub = {"slug": SLUG, "seed": seed, "smoke": bool(args.smoke), "folds": {}}
        for fname, fdata in results["folds"].items():
            if seed in fdata:
                sub["folds"][fname] = {"target": fdata["target"], "gating_dists": fdata["gating_dists"],
                                       "gating_weights": fdata["gating_weights"], "runs": fdata[seed]}
        sub["cross_fold_corr"] = results["cross_fold_corr"]
        with open(os.path.join(out_dir, f"seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(sub, f, indent=2, default=float)
    with open(os.path.join(out_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print("saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
