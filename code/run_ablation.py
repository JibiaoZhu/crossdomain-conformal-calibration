import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import yaml

from conformal import ACI, band_ref_stats, dsrc_mahalanobis, relative_divergence
from data import prepare_dataset
from losses import band_energies, band_saliency_loss, moment_match_loss, rec_loss, relation_consistency_loss
from metrics import auprc, auroc, fpr_on_normal, preds_range_f1
from model import PatchTSTRecon
from tta import LowRankAdapter, adapt

SLUG = "ssi-ad-ablation-spectral-relation-band"

VARIANT_FLAGS = {
    "full": dict(f1=1, f2=1, f3=1, f4=1, f5="shift", f6=1),
    "no_f1": dict(f1=0, f2=1, f3=1, f4=1, f5="shift", f6=1),
    "no_f2": dict(f1=1, f2=0, f3=1, f4=1, f5="shift", f6=1),
    "no_f3": dict(f1=1, f2=1, f3=0, f4=1, f5="shift", f6=1),
    "no_f4": dict(f1=1, f2=1, f3=1, f4=0, f5="shift", f6=1),
    "no_f5": dict(f1=1, f2=1, f3=1, f4=1, f5="off", f6=1),
    "no_f6": dict(f1=1, f2=1, f3=1, f4=1, f5="shift", f6=0),
    "fixed_conf": dict(f1=1, f2=1, f3=1, f4=1, f5="aci", f6=1),
    "none": dict(f1=0, f2=0, f3=0, f4=0, f5="off", f6=0),
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
def score_model(model, adapter, x, cfg, device):
    out = []
    bs = 256
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    model.eval()
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            inp = adapter(xb) if adapter is not None else xb
            rec, _, _ = model(inp)
        out.append(((rec - inp) ** 2).mean(dim=(1, 2)).float().cpu())
    return torch.cat(out).numpy()


def train_model(flags, cfg, src_groups, tgt_pool, device):
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
    terms = {"rec": [], "inv": [], "rel": [], "sal": []}
    grad_norms = []
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
            rec_term = rec_loss(rec_s, src)
            loss = rec_term
            if flags["f1"] or flags["f2"]:
                _, rel_t, band_t = model(tgt)
            inv_term = torch.zeros((), device=device)
            rel_term = torch.zeros((), device=device)
            sal_term = torch.zeros((), device=device)
            if flags["f1"]:
                inv_term = moment_match_loss(band_s.reshape(-1, d_model), band_t.reshape(-1, d_model))
                loss = loss + t["lambda_inv"] * inv_term
            if flags["f2"]:
                rel_term = relation_consistency_loss(rel_s, rel_t)
                loss = loss + t["lambda_rel"] * rel_term
            if flags["f3"]:
                sal_term = band_saliency_loss(src, rec_s, nb)
                loss = loss + t["lambda_sal"] * sal_term
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        do_log = (step % t["log_every"] == 0) or (step == t["max_steps"])
        if do_log:
            if amp:
                scaler.unscale_(opt)
            gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn += float(p.grad.detach().float().pow(2).sum())
            grad_norms.append(gn ** 0.5)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(float(loss.detach()))
        terms["rec"].append(float(rec_term.detach()))
        terms["inv"].append(float(inv_term.detach()))
        terms["rel"].append(float(rel_term.detach()))
        terms["sal"].append(float(sal_term.detach()))
        if do_log:
            print(f"  step {step}/{t['max_steps']} loss {np.mean(losses[-t['log_every']:]):.5f} elapsed {time.time() - t0:.0f}s", flush=True)
    log = {
        "loss_last_mean": float(np.mean(losses[-100:])),
        "terms_last_mean": {k: float(np.mean(v[-100:])) for k, v in terms.items()},
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "elapsed_sec": float(time.time() - t0),
        "nan": bool(np.isnan(losses).any() or np.isinf(losses).any()),
    }
    model.eval()
    return model, log


def group_band_mean(g, cfg, rng):
    nb = cfg["data"]["n_bands"]
    x = g["windows"]
    n = x.shape[0]
    if n > 1024:
        idx = np.sort(rng.choice(n, size=1024, replace=False))
    else:
        idx = np.arange(n)
    e = band_energies(x[torch.from_numpy(idx)], nb).float()
    return e.mean(0)


def compute_gating(cfg, src_groups, tgt_ref):
    rng = np.random.default_rng(cfg["seed"] + 5)
    mu_t = group_band_mean(tgt_ref, cfg, rng)
    ds = [float(torch.norm(group_band_mean(g, cfg, rng) - mu_t)) for g in src_groups]
    ds = np.array(ds)
    w = np.exp(-cfg["gating"]["tau"] * ds / max(ds.mean(), 1e-6))
    return w.tolist(), ds.tolist()


def inject_pv(xe, le, cfg, rng):
    xe = xe.clone()
    le = le.copy()
    n = xe.shape[0]
    L = xe.shape[2]
    k = int(n * cfg["data"]["pv"]["injection_fraction"])
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        typ = int(rng.integers(0, 3))
        if typ == 0:
            for _ in range(3):
                p = int(rng.integers(0, L))
                ch = int(rng.integers(0, 2))
                xe[i, ch, p] += 6.0 if rng.random() > 0.5 else -6.0
        elif typ == 1:
            xe[i] += 4.0 if rng.random() > 0.5 else -4.0
        else:
            a = int(rng.integers(0, max(L - L // 4, 1)))
            b = a + int(rng.integers(L // 8, max(L // 8 + 1, L // 3)))
            xe[i, :, a:b] = 0.0
        le[i] = 1.0
    return xe, le


def evaluate(model, adapter, flags, cfg, tgt, ref_stats, device, rng):
    alpha = cfg["conformal"]["alpha"]
    x = tgt["windows"]
    lab = tgt["labels"]
    normal_idx = np.where(lab < 0.5)[0]
    cal_n = max(int(len(normal_idx) * cfg["conformal"]["cal_ratio"]), 10)
    cal_idx = normal_idx[:cal_n]
    eval_mask = np.ones(x.shape[0], dtype=bool)
    eval_mask[cal_idx] = False
    xe = x[eval_mask]
    le = lab[eval_mask].copy()
    if tgt["name"] == "pv":
        xe, le = inject_pv(xe, le, cfg, rng)
    scores = score_model(model, adapter, xe, cfg, device)
    mu_ref, inv_ref = ref_stats
    dsrc = dsrc_mahalanobis(mu_ref, inv_ref, band_energies(xe, cfg["data"]["n_bands"]))
    dnorm = relative_divergence(dsrc, cfg["conformal"]["cal_ratio"])
    q_source = float(cfg["_q_source"])
    n = len(scores)
    f5 = flags["f5"]
    if f5 == "off":
        qs = np.full(n, q_source)
        preds = scores > q_source
    else:
        lam = cfg["conformal"]["shift_lambda"] if f5 == "shift" else 0.0
        aci = ACI(q_source, alpha, cfg["conformal"]["aci_gamma"], lam)
        preds = np.empty(n, dtype=bool)
        qs = np.empty(n)
        for i in range(n):
            q_t = aci.threshold(dnorm[i] if f5 == "shift" else 0.0)
            qs[i] = q_t
            preds[i] = bool(scores[i] > q_t)
            aci.update(scores[i], q_t)
    normal_eval = le < 0.5
    fpr = fpr_on_normal(preds, normal_eval)
    res = {
        **preds_range_f1(preds, le),
        "auroc": auroc(scores, le),
        "auprc": auprc(scores, le),
        "empirical_fpr_at_alpha": fpr,
        "calibration_coverage": float(1 - fpr) if not np.isnan(fpr) else float("nan"),
        "coverage": float(1 - fpr) if not np.isnan(fpr) else float("nan"),
        "mean_threshold": float(np.mean(qs)),
        "inflation": float(np.mean(qs) / max(q_source, 1e-9)),
        "interval_inflation": float(np.mean(qs) / max(q_source, 1e-9)),
        "q_source": q_source,
        "n_eval": int(n),
        "n_eval_normal": int(normal_eval.sum()),
        "n_eval_anomaly": int((~normal_eval).sum()),
        "d_mean": float(np.mean(dnorm)),
        "alarm_rate": float(np.mean(preds)),
    }
    return res


def summarize(results, cfg):
    folds = list(results["folds"].keys())
    variants = [v for v in cfg["variants"] if all(v in results["folds"][f]["variants"] for f in folds)]
    table = {v: {f: results["folds"][f]["variants"][v]["range_f1"] for f in folds} for v in variants}
    cls = {}
    if "full" in table:
        base = table["full"]
        for fac, vname in [("F1", "no_f1"), ("F2", "no_f2"), ("F3", "no_f3"),
                           ("F4", "no_f4"), ("F5", "no_f5"), ("F6", "no_f6")]:
            if vname not in table:
                continue
            diffs = {f: base[f] - table[vname][f] for f in folds}
            mean_d = float(np.mean(list(diffs.values())))
            mean_base = float(np.mean(list(base.values())))
            rel = mean_d / max(mean_base, 1e-9)
            label = "necessary" if rel > 0.10 else ("contributing" if rel > 0.03 else ("harmful" if rel < -0.01 else "marginal"))
            cls[fac] = {"variant": vname, "mean_rf1_drop": mean_d, "relative": rel, "label": label, "per_fold_drop": diffs}
    results["summary"] = {"rf1_table": table, "factor_classification": cls}
    if "full" in table and "fixed_conf" in table:
        results["summary"]["two_level_f5"] = {"shift_full": {f: table["full"][f] for f in folds},
                                              "aci_fixed": {f: table["fixed_conf"][f] for f in folds}}
    lines = [f"# Ablation summary (seed {cfg['seed']})", "", "## R-F1 table", "", "| variant | " + " | ".join(folds) + " |"]
    lines.append("|" + "---|" * (len(folds) + 1))
    for v in variants:
        lines.append(f"| {v} | " + " | ".join(f"{table[v][f]:.4f}" for f in folds) + " |")
    lines += ["", "## Factor classification (R-F1)", ""]
    for fac, d in cls.items():
        lines.append(f"- {fac}: {d['label']} (mean drop {d['mean_rf1_drop']:+.4f}, rel {d['relative']:+.1%})")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("summary:", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--folds", default=None)
    ap.add_argument("--variants", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.smoke:
        cfg["train"]["max_steps"] = 10
        cfg["train"]["batch_size"] = 8
        cfg["train"]["log_every"] = 5
        cfg["data"]["max_windows_per_dataset"] = 300
        cfg["tta"]["steps"] = 20
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    folds = cfg["folds"]
    if args.folds:
        want = set(args.folds.split(","))
        folds = [f for f in folds if f["name"] in want]
    if args.variants:
        want = set(args.variants.split(","))
        cfg["variants"] = [v for v in cfg["variants"] if v in want]

    needed = set()
    for fold in folds:
        needed.add(fold["target"])
        needed.update(fold["sources"])
    groups = {}
    t_all = time.time()
    for name in sorted(needed):
        t0 = time.time()
        offset = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
        rng_d = np.random.default_rng(cfg["seed"] * 7919 + offset)
        groups[name] = prepare_dataset(name, cfg, rng_d)
        g = groups[name]
        print(f"[data] {name}: windows={tuple(g['windows'].shape)} anomalies={int(g['labels'].sum())} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[data] prepared in {time.time() - t_all:.0f}s", flush=True)

    results = {"slug": SLUG, "seed": cfg["seed"], "smoke": bool(args.smoke), "device": str(device),
               "torch": torch.__version__, "folds": {}}

    for fold in folds:
        fname = fold["name"]
        tgt = groups[fold["target"]]
        src_groups = [groups[s] for s in fold["sources"]]
        src_norm = []
        for g in src_groups:
            lab = g["labels"]
            nm = np.where(lab < 0.5)[0]
            src_norm.append({"name": g["name"], "windows": g["windows"][torch.from_numpy(nm)]})
        src_train = []
        src_cal_list = []
        for g in src_norm:
            n = g["windows"].shape[0]
            nc = max(int(n * cfg["conformal"]["cal_ratio"]), 10)
            src_cal_list.append(g["windows"][:nc])
            src_train.append({"name": g["name"], "windows": g["windows"][nc:]})
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
        rec = {"target": fold["target"], "sources": fold["sources"],
               "n_src_train": int(sum(g["windows"].shape[0] for g in src_train)),
               "n_src_cal": int(src_cal.shape[0]), "n_target": int(Nt),
               "gating_dists": ds, "gating_weights": gw_on, "variants": {}}
        results["folds"][fname] = rec

        normal_idx = np.where(tgt["labels"] < 0.5)[0]
        cal_n = max(int(len(normal_idx) * cfg["conformal"]["cal_ratio"]), 10)
        cal_idx = normal_idx[:cal_n]

        for vname in cfg["variants"]:
            flags = VARIANT_FLAGS[vname]
            cfg["_gating_weights"] = gw_on if flags["f6"] else [1.0] * len(src_train)
            t0 = time.time()
            print(f"[{fname}/{vname}] training ...", flush=True)
            model, tlog = train_model(flags, cfg, src_train, tgt_pool, device)
            src_cal_scores = score_model(model, None, src_cal, cfg, device)
            cfg["_q_source"] = float(np.quantile(src_cal_scores, 1 - cfg["conformal"]["alpha"]))
            adapter = None
            tta_sec = 0.0
            if flags["f4"]:
                t0t = time.time()
                adapter = LowRankAdapter(cfg["data"]["window"], rank=cfg["tta"]["rank"]).to(device)
                adapt(model, adapter, tgt["windows"][torch.from_numpy(cal_idx)], cfg, device)
                tta_sec = time.time() - t0t
            res = evaluate(model, adapter, flags, cfg, tgt, ref_stats, device,
                           np.random.default_rng(cfg["seed"] + 23))
            res["train"] = tlog
            res["tta_sec"] = float(tta_sec)
            res["total_sec"] = float(time.time() - t0)
            rec["variants"][vname] = res
            print(f"[{fname}/{vname}] R-F1={res['range_f1']:.4f} AUROC={res['auroc']:.4f} "
                  f"AUPRC={res['auprc']:.4f} FPR={res['empirical_fpr_at_alpha']:.3f} infl={res['inflation']:.2f}", flush=True)

    summarize(results, cfg)
    out_dir = os.path.join(cfg["repo_root"], "results", SLUG)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"seed_{cfg['seed']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print("saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
