import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import yaml

from conformal import ACI, band_ref_stats, dsrc_mahalanobis, relative_divergence
from data import prepare_dataset
from losses import band_energies
from metrics import auprc, auroc, fpr_on_normal, pearson, preds_range_f1
from run_calibration import TRAIN_CONFIGS, compute_gating, score_model, set_seed, train_model

SLUG = "scc-main"


def load_target(spec, groups):
    if isinstance(spec, list):
        ws = torch.cat([groups[s]["windows"] for s in spec], dim=0)
        ls = np.concatenate([groups[s]["labels"] for s in spec])
        return {"name": "+".join(spec), "windows": ws, "labels": ls}
    g = groups[spec]
    return {"name": spec, "windows": g["windows"], "labels": g["labels"]}


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


def base_metrics(preds, scores, labels, qs, q_source, alpha):
    normal = labels < 0.5
    fpr = fpr_on_normal(preds, normal)
    cov = float(1 - fpr) if not np.isnan(fpr) else float("nan")
    res = preds_range_f1(preds, labels)
    res.update({
        "coverage": cov,
        "abs_coverage_gap": abs(cov - (1 - alpha)) if not np.isnan(cov) else float("nan"),
        "empirical_fpr_at_alpha": fpr,
        "inflation": float(np.mean(qs) / max(q_source, 1e-9)),
        "alarm_rate": float(np.mean(preds)),
    })
    return res


def online_eval(scores, labels, q_source, alpha, gamma, d_norm=None, lam=0.0):
    n = len(scores)
    aci = ACI(q_source, alpha, gamma, lam)
    preds = np.empty(n, dtype=bool)
    qs = np.empty(n)
    for i in range(n):
        q_t = aci.threshold(d_norm[i] if d_norm is not None else 0.0)
        qs[i] = q_t
        preds[i] = bool(scores[i] > q_t)
        aci.update(scores[i], q_t)
    return preds, qs


def static_eval(scores, q_source):
    return scores > q_source, np.full(len(scores), q_source)


def mondrian_eval(scores, labels, q_source, alpha, gamma, d_norm, bins):
    n = len(scores)
    edges = np.quantile(d_norm, [1.0 / bins * b for b in range(1, bins)])
    bin_id = np.digitize(d_norm, edges)
    acis = [ACI(q_source, alpha, gamma, 0.0) for _ in range(bins)]
    preds = np.empty(n, dtype=bool)
    qs = np.empty(n)
    for i in range(n):
        b = int(bin_id[i])
        q_t = acis[b].threshold(0.0)
        qs[i] = q_t
        preds[i] = bool(scores[i] > q_t)
        acis[b].update(scores[i], q_t)
    return preds, qs, bin_id


def eval_fold_model(scores, le, src_cal_scores, d_band, cfg):
    gamma = cfg["conformal"]["aci_gamma"]
    out = {}
    d_mean = float(np.mean(d_band))
    for alpha in cfg["alphas"]:
        q_source = float(np.quantile(src_cal_scores, 1 - alpha))
        for mode, (preds, qs) in {
            "static": static_eval(scores, q_source),
            "aci": online_eval(scores, le, q_source, alpha, gamma),
            "shiftband": online_eval(scores, le, q_source, alpha, gamma, d_norm=d_band, lam=0.1),
        }.items():
            r = base_metrics(preds, scores, le, qs, q_source, alpha)
            r["q_source"] = q_source
            r["d_mean"] = d_mean
            out[f"{mode}_a{alpha:g}"] = r
        preds, qs, bin_id = mondrian_eval(scores, le, q_source, alpha, gamma, d_band, cfg["conformal"]["mondrian_bins"])
        r = base_metrics(preds, scores, le, qs, q_source, alpha)
        r["q_source"] = q_source
        r["d_mean"] = d_mean
        normal = le < 0.5
        per_bin = {}
        for b in np.unique(bin_id):
            m = normal & (bin_id == b)
            per_bin[int(b)] = float(preds[m].mean()) if m.any() else float("nan")
        r["mondrian_per_bin_fpr"] = per_bin
        out[f"mondrian_a{alpha:g}"] = r
    return out


def diag_correlation(results, cfg):
    rows = []
    for alpha in cfg["alphas"]:
        xs, ys = [], []
        for fold in results["folds"]:
            for seed in results["seeds"]:
                try:
                    r = results["folds"][fold]["seeds"][seed]["evals"][f"shiftband_a{alpha:g}"]
                except KeyError:
                    fdata = results["folds"][fold].get(str(seed), {}).get("run", {})
                    try:
                        r = fdata["evals"][f"shiftband_a{alpha:g}"]
                    except KeyError:
                        continue
                xs.append(r["d_mean"])
                ys.append(r["inflation"])
        rr, pp = pearson(xs, ys)
        rows.append({"alpha": alpha, "pearson": rr, "p": pp, "n": len(xs)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--folds", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        cfg["train"]["max_steps"] = 20
        cfg["train"]["batch_size"] = 8
        cfg["train"]["log_every"] = 10
        cfg["data"]["max_windows_per_dataset"] = 300
        cfg["seeds"] = [0]
        cfg["alphas"] = [0.1]
    folds = cfg["folds"]
    if args.folds:
        want = set(args.folds.split(","))
        folds = [f for f in folds if f["name"] in want]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    needed = set()
    for fold in folds:
        tgt = fold["target"] if isinstance(fold["target"], list) else [fold["target"]]
        needed.update(tgt)
        needed.update(fold["sources"])
    groups = {}
    for name in sorted(needed):
        offset = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
        rng_d = np.random.default_rng(offset)
        groups[name] = prepare_dataset(name, cfg, rng_d)
        g = groups[name]
        print(f"[data] {name}: {tuple(g['windows'].shape)} anomalies={int(g['labels'].sum())}", flush=True)

    results = {"slug": SLUG, "smoke": bool(args.smoke), "device": str(device), "torch": torch.__version__,
               "seeds": cfg["seeds"], "folds": {}}

    for fold in folds:
        fname = fold["name"]
        tgt = load_target(fold["target"], groups)
        src_train = []
        src_cal_list = []
        for s in fold["sources"]:
            g = groups[s]
            nm = np.where(g["labels"] < 0.5)[0]
            wnorm = g["windows"][torch.from_numpy(nm)]
            nc = max(int(wnorm.shape[0] * cfg["conformal"]["cal_ratio"]), 10)
            src_cal_list.append(wnorm[:nc])
            src_train.append({"name": s, "windows": wnorm[nc:]})
        src_cal = torch.cat(src_cal_list)
        Nt = tgt["windows"].shape[0]

        normal_idx = np.where(tgt["labels"] < 0.5)[0]
        cal_n = max(int(len(normal_idx) * cfg["conformal"]["cal_ratio"]), 10)
        cal_idx = normal_idx[:cal_n]
        eval_mask = np.ones(Nt, dtype=bool)
        eval_mask[cal_idx] = False
        xe = tgt["windows"][eval_mask]
        le = tgt["labels"][eval_mask].copy()
        if "pv" in tgt["name"].split("+"):
            rng_inj = np.random.default_rng(cfg["seed"] + 23)
            xe, le = inject_pv(xe, le, cfg, rng_inj)
        e_ref = band_energies(src_cal[: cfg["conformal"]["d_ref_size"]], cfg["data"]["n_bands"])
        mu_ref, inv_ref = band_ref_stats(e_ref)
        dsrc = dsrc_mahalanobis(mu_ref, inv_ref, band_energies(xe, cfg["data"]["n_bands"]))
        d_band = relative_divergence(dsrc, cfg["conformal"]["cal_ratio"])

        results["folds"][fname] = {"target": tgt["name"], "sources": fold["sources"], "seeds": {}}
        for seed in cfg["seeds"]:
            cfg["seed"] = seed
            set_seed(seed)
            gen = np.random.default_rng(seed + 17)
            pool_idx = np.sort(gen.choice(Nt, size=min(8000, Nt), replace=False))
            tgt_pool = tgt["windows"][torch.from_numpy(pool_idx)]
            cfg["_gating_weights"] = [1.0] * len(src_train)
            t0 = time.time()
            print(f"[{fname}/seed{seed}] training (plain) ...", flush=True)
            model, tlog = train_model(TRAIN_CONFIGS["v2_plain"], cfg, src_train, tgt_pool, device)
            src_cal_scores = score_model(model, src_cal, cfg, device)
            scores = score_model(model, xe, cfg, device)
            evals = eval_fold_model(scores, le, src_cal_scores, d_band, cfg)
            entry = {
                "auroc": auroc(scores, le),
                "auprc": auprc(scores, le),
                "train": tlog,
                "n_eval": int(len(le)),
                "n_eval_normal": int((le < 0.5).sum()),
                "evals": evals,
                "elapsed_sec": float(time.time() - t0),
            }
            results["folds"][fname]["seeds"][seed] = entry
            a01 = evals["aci_a0.1"]
            s01 = evals["static_a0.1"]
            m01 = evals["mondrian_a0.1"]
            print(f"[{fname}/seed{seed}] AUROC={entry['auroc']:.3f} | cov static={s01['coverage']:.3f} "
                  f"aci={a01['coverage']:.3f} shift={evals['shiftband_a0.1']['coverage']:.3f} "
                  f"mondrian={m01['coverage']:.3f} (nominal 0.90)", flush=True)

    results["diag_corr"] = diag_correlation(results, cfg)

    out_dir = os.path.join(cfg["repo_root"], "results", SLUG)
    os.makedirs(out_dir, exist_ok=True)
    for seed in cfg["seeds"]:
        sub = {"slug": SLUG, "seed": seed, "smoke": bool(args.smoke), "folds": {}}
        for fname, fdata in results["folds"].items():
            if seed in fdata["seeds"]:
                sub["folds"][fname] = {"target": fdata["target"], "sources": fdata["sources"], "run": fdata["seeds"][seed]}
        sub["diag_corr"] = results["diag_corr"]
        with open(os.path.join(out_dir, f"seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(sub, f, indent=2, default=float)
    with open(os.path.join(out_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)

    modes = ["static", "aci", "shiftband", "mondrian"]
    lines = [f"# SCC main experiment summary (smoke={bool(args.smoke)})", "", "## Coverage @ alpha=0.1 (mean over seeds; nominal 0.90)", "",
             "| fold | " + " | ".join(modes) + " |", "|" + "---|" * (len(modes) + 1)]
    for fname, fdata in results["folds"].items():
        row = [fname]
        for m in modes:
            vals = [s["evals"][f"{m}_a0.1"]["coverage"] for s in fdata["seeds"].values()]
            row.append(f"{np.mean(vals):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## AUROC (per seed)", ""]
    for fname, fdata in results["folds"].items():
        vals = [round(s["auroc"], 3) for s in fdata["seeds"].values()]
        lines.append(f"- {fname}: {vals}")
    lines += ["", "## D_mean vs inflation correlation (shiftband)", ""]
    for r in results["diag_corr"]:
        lines.append(f"- alpha={r['alpha']}: r={r['pearson']} p={r['p']} n={r['n']}")
    lines += ["", "## Mondrian per-bin FPR @ alpha=0.1", ""]
    for fname, fdata in results["folds"].items():
        rows = [s["evals"]["mondrian_a0.1"].get("mondrian_per_bin_fpr") for s in fdata["seeds"].values()]
        lines.append(f"- {fname}: {rows}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved:", out_dir, "and", out, flush=True)


if __name__ == "__main__":
    main()
