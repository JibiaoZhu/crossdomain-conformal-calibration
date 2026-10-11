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
from metrics import auprc, auroc
from run_calibration import TRAIN_CONFIGS, score_model, set_seed, train_model
from run_main import base_metrics, inject_pv, load_target, mondrian_eval, online_eval, static_eval

SLUG = "scc-main-traces"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--contamination", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
        groups[name] = prepare_dataset(name, cfg, np.random.default_rng(offset))
        print(f"[data] {name}: {tuple(groups[name]['windows'].shape)}", flush=True)

    out_dir = os.path.join(cfg["repo_root"], "results", SLUG)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"slug": SLUG, "device": str(device), "torch": torch.__version__,
            "alphas": cfg["alphas"], "folds": {}}

    for fold in folds:
        fname = fold["name"]
        tgt = load_target(fold["target"], groups)
        src_train, src_cal_list = [], []
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
        is_synth = np.zeros(len(le), dtype=np.float32)
        is_pv = "pv" in tgt["name"].split("+")
        rng_inj = np.random.default_rng(cfg["seed"] + 23)
        if is_pv:
            n = xe.shape[0]
            k = int(n * cfg["data"]["pv"]["injection_fraction"])
            inj_idx = rng_inj.choice(n, size=k, replace=False)
            xe, le = inject_pv(xe, le, cfg, np.random.default_rng(cfg["seed"] + 23))
            is_synth[inj_idx] = 1.0
        e_ref = band_energies(src_cal[: cfg["conformal"]["d_ref_size"]], cfg["data"]["n_bands"])
        mu_ref, inv_ref = band_ref_stats(e_ref)
        dsrc = dsrc_mahalanobis(mu_ref, inv_ref, band_energies(xe, cfg["data"]["n_bands"]))
        d_band = relative_divergence(dsrc, cfg["conformal"]["cal_ratio"])

        meta["folds"][fname] = {"target": tgt["name"], "sources": fold["sources"],
                                "n_eval": int(len(le)),
                                "n_normal": int((le < 0.5).sum()),
                                "n_anom": int((le > 0.5).sum()),
                                "n_synth": int(is_synth.sum()),
                                "contamination": {}}
        for seed in cfg["seeds"]:
            cfg["seed"] = seed
            set_seed(seed)
            gen = np.random.default_rng(seed + 17)
            pool_idx = np.sort(gen.choice(Nt, size=min(8000, Nt), replace=False))
            tgt_pool = tgt["windows"][torch.from_numpy(pool_idx)]
            cfg["_gating_weights"] = [1.0] * len(src_train)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            print(f"[{fname}/seed{seed}] training ...", flush=True)
            model, tlog = train_model(TRAIN_CONFIGS["v2_plain"], cfg, src_train, tgt_pool, device)
            t_train = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

            t0 = time.time()
            src_cal_scores = score_model(model, src_cal, cfg, device)
            t_scal = time.time() - t0
            t0 = time.time()
            scores = score_model(model, xe, cfg, device)
            t_score = time.time() - t0
            auroc_v = auroc(scores, le)
            auprc_v = auprc(scores, le)

            q_traces = {}
            cov = {}
            t0 = time.time()
            n_updates = 0
            for alpha in cfg["alphas"]:
                q_source = float(np.quantile(src_cal_scores, 1 - alpha))
                preds, qs = static_eval(scores, q_source)
                q_traces[f"static_a{alpha:g}"] = qs
                cov[f"static_a{alpha:g}"] = base_metrics(preds, scores, le, qs, q_source, alpha)["coverage"]
                preds, qs = online_eval(scores, le, q_source, alpha, cfg["conformal"]["aci_gamma"])
                q_traces[f"aci_a{alpha:g}"] = qs
                cov[f"aci_a{alpha:g}"] = base_metrics(preds, scores, le, qs, q_source, alpha)["coverage"]
                preds, qs = online_eval(scores, le, q_source, alpha, cfg["conformal"]["aci_gamma"],
                                        d_norm=d_band, lam=0.1)
                q_traces[f"shiftband_a{alpha:g}"] = qs
                cov[f"shiftband_a{alpha:g}"] = base_metrics(preds, scores, le, qs, q_source, alpha)["coverage"]
                preds, qs, bin_id = mondrian_eval(scores, le, q_source, alpha,
                                                  cfg["conformal"]["aci_gamma"], d_band,
                                                  cfg["conformal"]["mondrian_bins"])
                q_traces[f"mondrian_a{alpha:g}"] = qs
                cov[f"mondrian_a{alpha:g}"] = base_metrics(preds, scores, le, qs, q_source, alpha)["coverage"]
                n_updates += 4 * len(scores)
            t_traces = time.time() - t0
            update_us = 1e6 * t_traces / max(n_updates, 1)

            payload = {"scores": scores.astype(np.float32), "labels": le.astype(np.float32),
                       "is_synth": is_synth, "d_band": d_band.astype(np.float32),
                       "src_cal_scores": src_cal_scores.astype(np.float32),
                       "auroc": np.float32(auroc_v), "auprc": np.float32(auprc_v),
                       "t_train": np.float32(t_train), "t_score_eval": np.float32(t_score),
                       "t_scal": np.float32(t_scal), "peak_mem_mb": np.float32(peak_mem),
                       "update_us": np.float32(update_us)}
            for k, v in q_traces.items():
                payload[f"q_{k}"] = v.astype(np.float32)
            np.savez_compressed(os.path.join(out_dir, f"{fname}_seed{seed}.npz"), **payload)
            meta["folds"][fname].setdefault("runs", {})[str(seed)] = {
                "auroc": auroc_v, "auprc": auprc_v, "coverage": cov,
                "t_train_s": t_train, "t_score_eval_s": t_score, "t_scal_s": t_scal,
                "peak_mem_mb": peak_mem, "update_us_per_step": update_us,
                "train_log": tlog}
            print(f"[{fname}/seed{seed}] AUROC={auroc_v:.3f} cov(aci,a0.1)={cov['aci_a0.1']:.3f} "
                  f"train={t_train:.0f}s score={t_score:.1f}s upd={update_us:.0f}us mem={peak_mem:.0f}MB", flush=True)

            if args.contamination and is_pv:
                flat = {}
                for frac in [0.05, 0.1, 0.2, 0.4]:
                    cfg2 = dict(cfg)
                    cfg2["data"] = dict(cfg["data"])
                    cfg2["data"]["pv"] = dict(cfg["data"]["pv"])
                    cfg2["data"]["pv"]["injection_fraction"] = frac
                    xe2, le2 = inject_pv(tgt["windows"][eval_mask], tgt["labels"][eval_mask].copy(),
                                         cfg2, np.random.default_rng(cfg["seed"] + 23))
                    t0 = time.time()
                    s2 = score_model(model, xe2, cfg, device)
                    flat[f"frac{frac:g}_scores"] = s2.astype(np.float32)
                    flat[f"frac{frac:g}_labels"] = le2.astype(np.float32)
                    flat[f"frac{frac:g}_t"] = np.float32(time.time() - t0)
                np.savez_compressed(os.path.join(out_dir, f"{fname}_seed{seed}_contamination.npz"), **flat)
                meta["folds"][fname]["contamination"][str(seed)] = "saved"

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)
    print("saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
