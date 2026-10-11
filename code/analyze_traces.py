import argparse
import glob
import json
import os

import numpy as np
import yaml

from conformal import ACI

NATIVE_RATE_HZ = {"phm16": 1.0, "care": 1.0 / 600.0, "pv": None, "mfpt": 97656.0,
                  "xjtu": 25600.0, "cwru": 12000.0, "paderborn": 128000.0}
FOLD_TARGET = {"F1": "phm16", "F2": "care", "F3": "pv", "F4": "cwru+paderborn", "F5": "mfpt", "F6": "xjtu"}


def load_traces(root, fold, seed):
    p = os.path.join(root, f"{fold}_seed{seed}.npz")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def eval_modes(scores, labels, src_cal, alphas, gamma):
    out = {}
    for alpha in alphas:
        q0 = float(np.quantile(src_cal, 1 - alpha))
        # static
        preds = scores > q0
        qs = np.full(len(scores), q0)
        out[f"static_a{alpha:g}"] = (preds, qs)
        # aci / shiftband in traces; aci baseline recomputed here
        aci = ACI(q0, alpha, gamma, 0.0)
        preds = np.empty(len(scores), dtype=bool)
        qs = np.empty(len(scores))
        for i in range(len(scores)):
            t = aci.threshold(0.0)
            qs[i] = t
            preds[i] = bool(scores[i] > t)
            aci.update(scores[i], t)
        out[f"aci_a{alpha:g}"] = (preds, qs)
        # aci update-rate k=5, k=20
        for k in (5, 20):
            aci = ACI(q0, alpha, gamma, 0.0)
            preds = np.empty(len(scores), dtype=bool)
            qs = np.empty(len(scores))
            for i in range(len(scores)):
                t = aci.threshold(0.0)
                qs[i] = t
                preds[i] = bool(scores[i] > t)
                if (i + 1) % k == 0:
                    aci.update(scores[i], t)
            out[f"aci_k{k}_a{alpha:g}"] = (preds, qs)
        # online rolling quantile tracker (W=1000)
        W = 1000
        preds = np.empty(len(scores), dtype=bool)
        qs = np.empty(len(scores))
        hist = list(src_cal[-W:])
        for i in range(len(scores)):
            t = float(np.quantile(hist[-W:], 1 - alpha))
            qs[i] = t
            preds[i] = bool(scores[i] > t)
            hist.append(scores[i])
        out[f"rollingq_a{alpha:g}"] = (preds, qs)
        # EVT/POT (GPD tail fit on source cal, static threshold)
        from scipy.stats import genpareto
        u = float(np.quantile(src_cal, 0.9))
        exc = src_cal[src_cal > u] - u
        if len(exc) > 30:
            c, loc, scale = genpareto.fit(exc, floc=0)
            p_tail = alpha / 0.1
            if c < -0.5:
                c = -0.5
            try:
                thr = u + genpareto.ppf(1 - p_tail, c, loc=0, scale=scale)
            except Exception:
                thr = q0
        else:
            thr = q0
        preds = scores > thr
        qs = np.full(len(scores), thr)
        out[f"evtpot_a{alpha:g}"] = (preds, qs)
    return out


def coverage_gap(preds, labels, alpha):
    normal = labels < 0.5
    cov = 1 - preds[normal].mean() if normal.any() else float("nan")
    return float(cov), float(abs(cov - (1 - alpha)))


def boot_ci(scores, labels, preds, alpha, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    gaps = np.empty(n_boot)
    normal = labels < 0.5
    idx_norm = np.where(normal)[0]
    for b in range(n_boot):
        take = idx_norm[rng.integers(0, len(idx_norm), len(idx_norm))]
        cov = 1 - preds[take].mean()
        gaps[b] = abs(cov - (1 - alpha))
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def det_delay_windows(scores, labels, preds):
    runs = []
    in_run = False
    for i, v in enumerate(labels > 0.5):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(labels) - 1))
    delays = []
    for a, b in runs:
        hit = np.where(preds[a:b + 1])[0]
        if len(hit):
            delays.append(int(hit[0]))
        else:
            delays.append(np.nan)
    return delays


def fa_per_window(preds, labels):
    normal = labels < 0.5
    return float(preds[normal].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--traces-dir", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    alphas = cfg["alphas"]
    gamma = cfg["conformal"]["aci_gamma"]
    seeds = cfg["seeds"]

    report = {}
    md = ["# Traces analysis (R2 calibration baselines + R5 ablations + CIs + operational metrics)", ""]
    for fold in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        report[fold] = {}
        md.append(f"## {fold} ({FOLD_TARGET[fold]})")
        for seed in seeds:
            z = load_traces(args.traces_dir, fold, seed)
            scores = z["scores"].astype(np.float64)
            labels = z["labels"]
            src_cal = z["src_cal_scores"].astype(np.float64)
            modes = eval_modes(scores, labels, src_cal, alphas, gamma)
            cov = {}
            for name, (preds, qs) in modes.items():
                alpha = float(name.split("_a")[-1]) if "_a" in name else None
                if alpha is None:
                    continue
                c, g = coverage_gap(preds, labels, alpha)
                cov[name] = {"coverage": c, "gap": g}
            report[fold][str(seed)] = {"coverage": cov}
            if seed == seeds[0]:
                md.append(f"- seed {seed} | " + " ".join(
                    f"{n}={cov[n]['coverage']:.3f}" for n in [
                        "static_a0.1", "aci_a0.1", "aci_k5_a0.1", "aci_k20_a0.1",
                        "rollingq_a0.1", "evtpot_a0.1"]))
                normal = labels < 0.5
                preds, _ = modes["aci_a0.1"]
                d = det_delay_windows(scores, labels, preds)
                valid = [x for x in d if not np.isnan(x)]
                md.append(f"  - ACI a0.1: FA/window={fa_per_window(preds, labels):.3f} "
                          f"(per 1000 windows {fa_per_window(preds, labels)*1000:.1f}); "
                          f"detection delay windows: median={np.median(valid) if valid else float('nan')} "
                          f"(runs={len(d)}, detected={len(valid)})")
        # bootstrap CI at alpha=0.1 for aci vs static (seed 0)
        z = load_traces(args.traces_dir, fold, seeds[0])
        scores = z["scores"].astype(np.float64)
        labels = z["labels"]
        src_cal = z["src_cal_scores"].astype(np.float64)
        modes = eval_modes(scores, labels, src_cal, alphas, gamma)
        for m in ["static_a0.1", "aci_a0.1"]:
            preds, _ = modes[m]
            lo, hi = boot_ci(scores, labels, preds, 0.1)
            report[fold].setdefault("boot_ci_a0.1", {})[m] = [lo, hi]
            md.append(f"- bootstrap CI gap {m}: [{lo:.3f}, {hi:.3f}] (seed {seeds[0]})")

    # operational table
    md.append("")
    md.append("## Operational metrics (seed 0, ACI alpha=0.1)")
    md.append("| fold | target | rate(Hz) | stride_s | FA/1000win | med delay (win) |")
    md.append("|---|---|---|---|---|---|")
    for fold in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        z = load_traces(args.traces_dir, fold, seeds[0])
        scores = z["scores"].astype(np.float64)
        labels = z["labels"]
        src_cal = z["src_cal_scores"].astype(np.float64)
        modes = eval_modes(scores, labels, src_cal, alphas, gamma)
        preds, _ = modes["aci_a0.1"]
        rate = NATIVE_RATE_HZ.get(FOLD_TARGET[fold], None)
        stride_s = (cfg["data"]["stride"] / rate) if rate else float("nan")
        d = det_delay_windows(scores, labels, preds)
        valid = [x for x in d if not np.isnan(x)]
        report[fold].setdefault("operational", {})["fa_per_1000"] = fa_per_window(preds, labels) * 1000
        report[fold]["operational"]["stride_s"] = stride_s
        report[fold]["operational"]["median_delay_windows"] = float(np.median(valid)) if valid else float("nan")
        md.append(f"| {fold} | {FOLD_TARGET[fold]} | {rate} | {stride_s:.4g} | "
                  f"{fa_per_window(preds, labels)*1000:.1f} | "
                  f"{np.median(valid) if valid else float('nan'):.0f} |")

    md.append("")
    md.append("## Runtime/memory (R6, from traces meta; per run)")
    out = os.path.join(args.traces_dir, "traces_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    md.append("(see meta.json: t_train_s, t_score_eval_s, t_scal_s, update_us_per_step, peak_mem_mb)")
    mdpath = os.path.join(args.traces_dir, "traces_analysis.md")
    with open(mdpath, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("saved:", out, mdpath, flush=True)


if __name__ == "__main__":
    main()
