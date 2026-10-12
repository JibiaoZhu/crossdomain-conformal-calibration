import glob
import json
import os
import re

import numpy as np
import scipy.io as sio
from scipy import stats as sps

TR = r"F:\AutoSci_CrossDomain_Handoff\results\scc-main-traces"
OUT = r"F:\AutoSci_CrossDomain_Handoff\results\revision-batch2"
os.makedirs(OUT, exist_ok=True)
FOLDS = ["F1", "F2", "F3", "F4", "F5", "F6"]
SEEDS = [0, 1, 2]
GAMMA = 0.05
ALPHAS = [0.05, 0.1, 0.2, 0.3]
N_CAL = 512          # initial unlabeled calibration segment
PERIOD = 5000        # periodic recalibration interval
BLOCK = 2048         # streaming POT window
POT_REFIT = 64       # POT refit stride

report = {}


def aci_run(scores, q0, alpha, kappa=None):
    q = float(q0)
    preds = np.empty(len(scores), bool)
    for i, s in enumerate(scores):
        preds[i] = s > q
        if kappa is None or s <= kappa * q:
            q = q + GAMMA * max(abs(q), 1e-12) * ((1.0 if preds[i] else 0.0) - alpha)
    return preds


def cov_gap(preds, labels, alpha):
    normal = labels < 0.5
    if normal.sum() == 0:
        return float("nan")
    return abs(float(preds[~normal].mean()) - alpha)


# ---------------- B5: target-domain calibration baselines ----------------
b5 = {}
for fold in FOLDS:
    b5[fold] = {}
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
        s = z["scores"].astype(np.float64)
        lab = z["labels"]
        n = len(s)
        for alpha in ALPHAS:
            q_src = float(np.quantile(z["src_cal_scores"].astype(np.float64), 1 - alpha))
            # (a) first-N target quantile, frozen
            q_fn = float(np.quantile(s[:N_CAL], 1 - alpha))
            pred_fn = s > q_fn
            # (b) expanding-window quantile (recompute every step; fast enough)
            pred_exp = np.empty(n, bool)
            for t in range(n):
                pred_exp[t] = s[t] > (np.quantile(s[:t + 1], 1 - alpha) if t > 0 else q_fn)
            # (c) periodic recalibration every PERIOD windows
            pred_per = np.empty(n, bool)
            q_cur = q_fn
            for t in range(n):
                if t >= N_CAL and (t - N_CAL) % PERIOD == 0:
                    q_cur = float(np.quantile(s[:t], 1 - alpha))
                pred_per[t] = s[t] > q_cur
            # (d) streaming POT (DSPOT-like): sliding-window GPD tail
            pred_pot = np.empty(n, bool)
            q_pot = q_fn
            n_exc_min = 20
            for t in range(n):
                if t >= N_CAL and (t - N_CAL) % POT_REFIT == 0:
                    w = s[max(0, t - BLOCK):t]
                    if len(w) >= 128:
                        u = float(np.quantile(w, 0.90))
                        exc = w[w > u] - u
                        if len(exc) >= n_exc_min and exc.std() > 0:
                            try:
                                xi, _, sig = sps.genpareto.fit(exc, floc=0)
                                if -1 < xi < 0.5 and sig > 0:
                                    ratio = (len(w) * alpha) / len(exc)
                                    if abs(xi) > 1e-6:
                                        q_pot = float(u + sig / xi * (ratio ** (-xi) - 1))
                                    else:
                                        q_pot = float(u - sig * np.log(ratio))
                            except Exception:
                                pass
                pred_pot[t] = s[t] > q_pot
            row = {}
            for name, pred in [("firstN", pred_fn), ("expanding", pred_exp), ("periodic", pred_per), ("dspot", pred_pot)]:
                row[name] = {"gap": cov_gap(pred, lab, alpha), "alarm": float(pred.mean()),
                             "det": float(pred[lab > 0.5].mean()) if (lab > 0.5).any() else float("nan"),
                             "inflation": float(np.median(s[max(0, 0):]) if False else 0.0)}
            # inflation proxy: mean deployed threshold ratio
            row["firstN"]["q_ratio"] = q_fn / q_src if q_src else float("nan")
            row["q_src"] = q_src
            b5[fold].setdefault(f"{alpha:g}", []).append(row)
report["b5"] = b5

# aggregate B5 at alpha=0.10
print("=== B5 target-domain calibration baselines (alpha=0.10, mean over seeds) ===")
print(f"{'fold':4s} {'method':>10s} {'gap':>7s} {'alarm':>7s} {'det':>7s}")
for fold in FOLDS:
    rows = b5[fold]["0.1"]
    for name in ["firstN", "expanding", "periodic", "dspot"]:
        g = np.mean([r[name]["gap"] for r in rows])
        al = np.mean([r[name]["alarm"] for r in rows])
        dt = np.mean([r[name]["det"] for r in rows])
        print(f"{fold:4s} {name:>10s} {g:7.3f} {al:7.3f} {dt:7.3f}")
    qr = np.mean([r["firstN"]["q_ratio"] for r in rows])
    print(f"{fold:4s} {'q_N512/q_src':>10s} {qr:7.2f}")

# ---------------- D9: calibration-segment contamination ----------------
d9 = {}
print()
print("=== D9 calibration-segment contamination (alpha=0.10, N=512) ===")
for fold in FOLDS:
    d9[fold] = {}
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
        s = z["scores"].astype(np.float64)
        lab = z["labels"]
        q_src = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
        rng = np.random.default_rng(9000 + 100 * seed + FOLDS.index(fold))
        pool = s[lab > 0.5]
        for pi in [0.0, 0.01, 0.05, 0.10]:
            seg = s[:N_CAL].copy()
            k = int(round(pi * N_CAL))
            if k > 0 and len(pool) > 0:
                idx = rng.choice(N_CAL, size=k, replace=False)
                seg[idx] = rng.choice(pool, size=k, replace=True)
            q_init = float(np.quantile(seg, 0.9))
            pred_frozen = s > q_init
            pred_aci = aci_run(s, q_init, 0.10)
            pred_na3 = aci_run(s, q_init, 0.10, kappa=3.0)
            early = slice(N_CAL, N_CAL + 1000)
            early_lab = lab[early]
            d9[fold].setdefault(f"{pi:g}", []).append({
                "q_ratio": q_init / q_src if q_src else float("nan"),
                "frozen_gap": cov_gap(pred_frozen, lab, 0.10),
                "aci_gap": cov_gap(pred_aci, lab, 0.10),
                "na3_gap": cov_gap(pred_na3, lab, 0.10),
                "aci_early_gap": cov_gap(pred_aci[early], early_lab, 0.10),
                "na3_early_gap": cov_gap(pred_na3[early], early_lab, 0.10),
            })
report["d9"] = d9

# ---------------- D7/D8: F4 per-fault-type provenance and metrics ----------------
root = r"E:\AutosciOpencodeProcessionDocument\datasets"
CACHE = os.path.join(root, "_cache")
cwru_ids_n = [97, 98, 99, 100]
cwru_ids_f = [105, 106, 107, 108, 118, 130, 169, 185]


def cwru_n(fid):
    mat = sio.loadmat(os.path.join(root, "CWRU", f"{fid}.mat"))
    k = next(k for k in mat if k.endswith("_DE_time"))
    return (mat[k].size - 256) // 128 + 1


n_per = {fid: cwru_n(fid) for fid in cwru_ids_n + cwru_ids_f}
n_normal = sum(n_per[f] for f in cwru_ids_n)

# replicate the exact pipeline: cwru capped to 20000 via md5-offset rng, then target
# concat [cwru, paderborn]; eval removes the first 20% of normal windows (cal segment).
zc = np.load(os.path.join(CACHE, "cwru_w256_s128_v3.npz"))
zcw = zc["windows"]
zcl = zc["labels"]
off = int(__import__("hashlib").md5(b"cwru").hexdigest(), 16) % 100000
rng_cap = np.random.default_rng(off)
keep_idx = np.sort(rng_cap.choice(len(zcw), 20000, replace=False))
zp = np.load(os.path.join(CACHE, "paderborn_w256_s128_v3.npz"))
lab_concat = np.concatenate([zcl[keep_idx], zp["labels"]])
# original -> file mapping (concat order: normals by file, then faults by file)
orig_file = {}
pos0 = 0
for fid in cwru_ids_n + cwru_ids_f:
    orig_file[fid] = np.arange(pos0, pos0 + n_per[fid])
    pos0 += n_per[fid]
file_of_kept = np.full(len(keep_idx), -1)
for fid, rng_rr in orig_file.items():
    in_rr = np.isin(keep_idx, rng_rr)
    file_of_kept[in_rr] = fid
assert (file_of_kept >= 0).all()
file_of_concat = np.concatenate([file_of_kept, np.full(1280, -1)])
normal_idx_full = np.where(lab_concat < 0.5)[0]
cal_n = max(int(len(normal_idx_full) * 0.2), 10)
cal_idx = normal_idx_full[:cal_n]
eval_mask = np.ones(len(lab_concat), dtype=bool)
eval_mask[cal_idx] = False
file_of_eval = file_of_concat[eval_mask]      # -1 for paderborn windows
pb_eval_pos = np.where(file_of_eval == -1)[0]

print()
print("=== D7/D8 provenance replication ===")
print("cwru kept:", len(keep_idx), "| eval total:", int(eval_mask.sum()), "| normals:", int((lab_concat[eval_mask] < 0.5).sum()),
      "| anomalies:", int((lab_concat[eval_mask] > 0.5).sum()))

d78 = {}
for seed in SEEDS:
    z = np.load(os.path.join(TR, f"F4_seed{seed}.npz"))
    s = z["scores"].astype(np.float64)
    lab = z["labels"]
    assert len(s) == int(eval_mask.sum()), (len(s), int(eval_mask.sum()))
    assert int((lab > 0.5).sum()) == int((lab_concat[eval_mask] > 0.5).sum()), "anomaly count mismatch - cache newer than traces?"
    src_cal = z["src_cal_scores"].astype(np.float64)
    q0 = float(np.quantile(src_cal, 0.9))
    pred_aci = aci_run(s, q0, 0.10)
    pred_na3 = aci_run(s, q0, 0.10, kappa=3.0)
    normal_eval = np.where(lab < 0.5)[0]
    norm_scores = s[normal_eval]
    groups = {}
    for fid in cwru_ids_f:
        nm = {105: "CWRU-105 (IR007)", 106: "CWRU-106 (IR014)", 107: "CWRU-107 (IR021)", 108: "CWRU-108 (IR028)",
              118: "CWRU-118 (B007)", 130: "CWRU-130 (OR@6)", 169: "CWRU-169 (OR@3)", 185: "CWRU-185 (OR@12)"}[fid]
        groups[nm] = np.where(file_of_eval == fid)[0]
    # paderborn late-life = anomalous windows among paderborn eval positions
    pb_anom = pb_eval_pos[lab_concat[eval_mask][pb_eval_pos] > 0.5]
    groups["Paderborn late-life"] = pb_anom
    for gname, idx in groups.items():
        if len(idx) == 0:
            continue
        sc = s[idx]
        auc = float((sc[:, None] > norm_scores[None, :]).mean() + 0.5 * (sc[:, None] == norm_scores[None, :]).mean())
        d78.setdefault(gname, []).append({
            "n_windows": int(len(idx)),
            "auroc_vs_normals": auc,
            "aci_det": float(pred_aci[idx].mean()),
            "na3_det": float(pred_na3[idx].mean()),
        })
report["d78"] = d78
print()
print("=== D7/D8 F4 held-out-type metrics (alpha=0.10, mean seeds) ===")
for gname, rows in d78.items():
    print(f"{gname:36s} n={int(np.mean([r['n_windows'] for r in rows])):5d} "
          f"AUROC={np.mean([r['auroc_vs_normals'] for r in rows]):.3f} "
          f"det: ACI={np.mean([r['aci_det'] for r in rows]):.3f} NA3={np.mean([r['na3_det'] for r in rows]):.3f}")

# ---------------- D10: label-noise robustness ----------------
d10 = {}
print()
print("=== D10 symmetric label noise (alpha=0.10) ===")
for level, pi in [("0.05", 0.05), ("0.10", 0.10), ("0.20", 0.20)]:
    per = {}
    for fold in FOLDS:
        rows = []
        for seed in SEEDS:
            z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
            s = z["scores"].astype(np.float64)
            lab = z["labels"].astype(np.float64)
            rng = np.random.default_rng(7000 + 100 * seed + FOLDS.index(fold))
            mask = rng.random(len(lab)) < pi
            noisy = lab.copy()
            noisy[mask] = 1.0 - noisy[mask]
            q_src = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
            pred_static = s > float(z["q_static_a0.1"][0])
            pred_aci = aci_run(s, q_src, 0.10)
            pred_na3 = aci_run(s, q_src, 0.10, kappa=3.0)
            normal = noisy < 0.5
            anom = ~normal
            auc = float((s[anom][:, None] > s[normal][None, :]).mean()) if anom.any() and normal.any() else float("nan")
            rows.append({
                "auroc": auc,
                "static_gap": abs(float(pred_static[normal].mean()) - 0.10),
                "aci_gap": abs(float(pred_aci[normal].mean()) - 0.10),
                "na3_gap": abs(float(pred_na3[normal].mean()) - 0.10),
                "aci_det": float(pred_aci[anom].mean()) if anom.any() else float("nan"),
                "na3_det": float(pred_na3[anom].mean()) if anom.any() else float("nan"),
            })
        per[fold] = rows
    d10[level] = {fold: {k: float(np.mean([r[k] for r in rows])) for k in ["auroc", "static_gap", "aci_gap", "na3_gap", "aci_det", "na3_det"]} for fold, rows in per.items()}
report["d10"] = d10
for level in ["0.05", "0.10", "0.20"]:
    print(f"noise={level}: mean AUROC={np.mean([v['auroc'] for v in d10[level].values()]):.3f} "
          f"ACI gap={np.mean([v['aci_gap'] for v in d10[level].values()]):.3f} "
          f"NA3 gap={np.mean([v['na3_gap'] for v in d10[level].values()]):.3f} "
          f"static gap={np.mean([v['static_gap'] for v in d10[level].values()]):.3f}")

json.dump(report, open(os.path.join(OUT, "batch2.json"), "w"), indent=2, default=float)
print("\nsaved:", os.path.join(OUT, "batch2.json"))
