import argparse
import hashlib
import json
import os

import numpy as np
import torch
import yaml

from data import prepare_dataset
from losses import band_energies
from metrics import auprc, auroc, pearson
from run_calibration import TRAIN_CONFIGS, compute_gating, score_model, set_seed, train_model

FOLD_DEFS = [
    ("F5_mfpt", "mfpt", ["phm16", "pv", "xjtu"]),
    ("F6_xjtu", "xjtu", ["phm16", "pv", "mfpt", "cwru"]),
]


def window_features(x):
    rms = x.pow(2).mean(dim=(1, 2)).sqrt().numpy()
    e = band_energies(x, 8).numpy()
    low = e[:, [0, 8]].mean(axis=1)
    high = e[:, [7, 15]].mean(axis=1)
    return rms, low, high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        cfg["train"]["max_steps"] = 10
        cfg["train"]["batch_size"] = 8
        cfg["data"]["max_windows_per_dataset"] = 300
    cfg["seed"] = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    needed = {"mfpt", "xjtu", "phm16", "pv", "cwru"}
    groups = {}
    for name in sorted(needed):
        offset = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
        rng_d = np.random.default_rng(offset)
        groups[name] = prepare_dataset(name, cfg, rng_d)
        print(f"[data] {name}: {tuple(groups[name]['windows'].shape)}", flush=True)

    report = {}
    lines = ["# Fold diagnosis: F5 (MFPT, AUROC<0.5) vs F6 (XJTU, control)", ""]
    for tag, tgt_name, src_names in FOLD_DEFS:
        tgt = groups[tgt_name]
        src_groups = [groups[s] for s in src_names]
        src_train = []
        for g in src_groups:
            lab = g["labels"]
            nm = np.where(lab < 0.5)[0]
            wnorm = g["windows"][torch.from_numpy(nm)]
            nc = max(int(wnorm.shape[0] * cfg["conformal"]["cal_ratio"]), 10)
            src_train.append({"name": g["name"], "windows": wnorm[nc:]})
        Nt = tgt["windows"].shape[0]
        gen = np.random.default_rng(1000)
        pool_idx = np.sort(gen.choice(Nt, size=min(8000, Nt), replace=False))
        tgt_pool = tgt["windows"][torch.from_numpy(pool_idx)]
        cfg["_gating_weights"] = [1.0] * len(src_train)
        set_seed(cfg["seed"])
        print(f"[{tag}] training plain encoder ...", flush=True)
        model, tlog = train_model(TRAIN_CONFIGS["v2_plain"], cfg, src_train, tgt_pool, device)
        normal_idx = np.where(tgt["labels"] < 0.5)[0]
        cal_n = max(int(len(normal_idx) * cfg["conformal"]["cal_ratio"]), 10)
        eval_mask = np.ones(Nt, dtype=bool)
        eval_mask[normal_idx[:cal_n]] = False
        xe = tgt["windows"][eval_mask]
        le = tgt["labels"][eval_mask]
        scores = score_model(model, xe, cfg, device)
        rms, low, high = window_features(xe)
        normal = le < 0.5
        res = {
            "n_eval": int(len(le)), "n_normal": int(normal.sum()), "n_anom": int((~normal).sum()),
            "auroc_score": auroc(scores, le),
            "auprc_score": auroc(scores, le),
            "auroc_rms": auroc(rms, le), "auroc_lowband": auroc(low, le), "auroc_highband": auroc(high, le),
            "score_median_normal": float(np.median(scores[normal])),
            "score_median_anom": float(np.median(scores[~normal])),
            "rms_median_normal": float(np.median(rms[normal])),
            "rms_median_anom": float(np.median(rms[~normal])),
            "pearson_score_rms_normal": pearson(scores[normal], rms[normal])[0],
            "pearson_score_rms_anom": pearson(scores[~normal], rms[~normal])[0],
        }
        res["auprc_score"] = auprc(scores, le)
        report[tag] = res
        lines += [f"## {tag}", "", "```json", json.dumps(res, indent=2, default=float), "```", ""]
        print(f"[{tag}] auroc_score={res['auroc_score']:.3f} auroc_rms={res['auroc_rms']:.3f} "
              f"auroc_low={res['auroc_lowband']:.3f} auroc_high={res['auroc_highband']:.3f} "
              f"med_score_n={res['score_median_normal']:.4f} med_score_a={res['score_median_anom']:.4f}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mfpt_diagnosis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mfpt_diagnosis.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved:", out, md, flush=True)


if __name__ == "__main__":
    main()
