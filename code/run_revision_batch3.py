import hashlib
import json
import os

import numpy as np

TR = r"F:\AutoSci_CrossDomain_Handoff\results\scc-main-traces"
CACHE = r"E:\AutosciOpencodeProcessionDocument\datasets\_cache"
OUT = r"F:\AutoSci_CrossDomain_Handoff\results\revision-batch3"
os.makedirs(OUT, exist_ok=True)
FOLDS = ["F1", "F2", "F3", "F4", "F5", "F6"]
SEEDS = [0, 1, 2]
GAMMA = 0.05
ALPHAS = [0.05, 0.1, 0.2, 0.3]

report = {}


def aci_run(scores, q0, alpha, kappa=None, gamma=GAMMA):
    q = float(q0)
    preds = np.empty(len(scores), bool)
    for i, s in enumerate(scores):
        preds[i] = s > q
        if kappa is None or s <= kappa * q:
            q = q + gamma * max(abs(q), 1e-12) * ((1.0 if preds[i] else 0.0) - alpha)
    return preds


def gap_of(preds, labels, alpha):
    nrm = labels < 0.5
    return abs(float(preds[nrm].mean()) - alpha)


# ---------- C4: kappa sweep ----------
print("=== C4 kappa sweep (mean gap over 4 alphas / det@0.10, mean seeds) ===")
c4 = {}
for fold in FOLDS:
    c4[fold] = {}
    for kappa in [1.5, 2.0, 3.0, 5.0, 10.0, None]:
        gsum, dets = [], []
        for seed in SEEDS:
            z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
            s = z["scores"].astype(np.float64)
            lab = z["labels"]
            for alpha in ALPHAS:
                q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 1 - alpha))
                p = aci_run(s, q0, alpha, kappa)
                gsum.append(gap_of(p, lab, alpha))
            q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
            p1 = aci_run(s, q0, 0.1, kappa)
            dets.append(float(p1[lab > 0.5].mean()))
        c4[fold][str(kappa)] = {"mean_gap": float(np.mean(gsum)), "det10": float(np.mean(dets))}
    line = fold + "".join(f" k={k}:{v['mean_gap']:.3f}/{v['det10']:.2f}" for k, v in c4[fold].items())
    print(line)
report["c4"] = c4

# ---------- C5: gamma sweep ----------
print()
print("=== C5 gamma sweep (mean gap over alphas; det@0.10; F4 oscillation count) ===")
c5 = {}
for gamma in [0.01, 0.02, 0.05, 0.1, 0.2]:
    agg_aci, agg_na, dets, osc_f4 = [], [], [], []
    for fold in FOLDS:
        for seed in SEEDS:
            z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
            s = z["scores"].astype(np.float64)
            lab = z["labels"]
            for alpha in ALPHAS:
                q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 1 - alpha))
                agg_aci.append(gap_of(aci_run(s, q0, alpha, None, gamma), lab, alpha))
                agg_na.append(gap_of(aci_run(s, q0, alpha, 3.0, gamma), lab, alpha))
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"F4_seed{seed}.npz"))
        s = z["scores"].astype(np.float64)
        q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
        # oscillation count requires q trace; rerun with trace
        q = float(q0)
        lg = []
        for t in range(len(s)):
            a = s[t] > q
            q = q + gamma * max(abs(q), 1e-12) * ((1.0 if a else 0.0) - 0.9)
            lg.append(np.log(max(q, 1e-12)))
        lg = np.array(lg)
        osc_f4.append(int((np.diff(np.sign(np.diff(lg))) != 0).sum()))
    c5[str(gamma)] = {"aci_gap": float(np.mean(agg_aci)), "na3_gap": float(np.mean(agg_na)),
                      "f4_aci_osc": float(np.mean(osc_f4))}
    print(f"gamma={gamma:4.2f}  ACI gap={np.mean(agg_aci):.4f}  NA3 gap={np.mean(agg_na):.4f}  F4 osc={np.mean(osc_f4):.0f}")
report["c5"] = c5

# ---------- C6: warm-up ----------
print()
print("=== C6 warm-up @ alpha=0.10: first stable window index (running |cov-0.9|<=0.02 forever) ===")
c6 = {}
for fold in FOLDS:
    ws = []
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
        s = z["scores"].astype(np.float64)
        lab = z["labels"]
        q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
        p = aci_run(s, q0, 0.1)
        nrm = lab < 0.5
        idx_n = np.where(nrm)[0]
        run_cov = np.cumsum(~p[idx_n]) / np.arange(1, len(idx_n) + 1)
        out = np.where(np.abs(run_cov - 0.9) > 0.02)[0]
        ws.append(int(idx_n[out[-1] + 1]) if len(out) and out[-1] + 1 < len(idx_n) else -1)
    c6[fold] = [int(w) for w in ws]
    print(fold, ws)
report["c6"] = c6

# ---------- C10: P2 streaming quantile ----------
def p2_run(scores, p=0.9):
    n = len(scores)
    preds = np.empty(n, bool)
    if n < 6:
        q = float(np.quantile(scores, p)) if n else 0.0
        return scores > q
    qs = np.sort(scores[:5]).astype(np.float64)
    npos = np.array([1.0, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5.0])
    dn = np.array([0.0, p / 2, p, (1 + p) / 2, 1.0])
    for i in range(5):
        preds[i] = scores[i] > qs[2]
    for t in range(5, n):
        x = scores[t]
        k = 0 if x < qs[0] else (4 if x >= qs[4] else int(np.searchsorted(qs, x) - 1))
        k = min(max(k, 0), 3)
        for i in range(k + 1, 5):
            npos[i] += 1
        dp = np.array([0.0, p / 2, p, (1 + p) / 2, 1.0])
        for i in range(1, 4):
            d = dp[i]
            if (d >= 1 and npos[i + 1] - npos[i] > 1) or (d <= -1 and npos[i - 1] - npos[i] < -1):
                dd = 1 if d >= 0 else -1
                qi = qs[i] + dd / (npos[i + 1] - npos[i - 1]) * (
                    (npos[i] - npos[i - 1] + dd) * (qs[i + 1] - qs[i]) / (npos[i + 1] - npos[i])
                    + (npos[i + 1] - npos[i] - dd) * (qs[i] - qs[i - 1]) / (npos[i] - npos[i - 1]))
                qs[i] = qi if qs[i - 1] < qi < qs[i + 1] else (qs[i] + dd * (qs[i + 1 if dd > 0 else i - 1] - qs[i]) / max(abs(npos[i + 1 if dd > 0 else i - 1] - npos[i]), 1))
                npos[i] += dd
        if qs[0] > x:
            qs[0] = x
        if qs[4] < x:
            qs[4] = x
        preds[t] = x > qs[2]
    return preds


print()
print("=== C10 P2 streaming quantile (alpha=0.10) ===")
c10 = {}
for fold in FOLDS:
    gs, als, dets = [], [], []
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
        s = z["scores"].astype(np.float64)
        lab = z["labels"]
        p = p2_run(s, 0.9)
        gs.append(gap_of(p, lab, 0.1))
        als.append(float(p.mean()))
        dets.append(float(p[lab > 0.5].mean()))
    c10[fold] = {"gap": float(np.mean(gs)), "alarm": float(np.mean(als)), "det": float(np.mean(dets))}
    print(f"{fold}: gap={c10[fold]['gap']:.3f} alarm={c10[fold]['alarm']:.3f} det={c10[fold]['det']:.3f}")
report["c10"] = c10

# ---------- E1: alarm-frequency conversion ----------
print()
print("=== E1 operational alarm frequency @ alpha=0.10 ===")
DT = {"F1": 128.0, "F2": 128 * 600.0, "F3": 128 * 720.0, "F4a": 128 / 12000.0, "F4b": 128 / 128000.0,
      "F5": 128 / 48828.0, "F6": 128 / 25600.0}
e1 = {}
for k, dt in DT.items():
    per_h = 0.10 / dt * 3600
    alpha_1h = dt / 3600.0
    e1[k] = {"step_s": dt, "alarms_per_hour_at_a0.1": per_h, "alpha_for_1_per_hour": min(alpha_1h, 1.0)}
    print(f"{k}: step={dt:.4g}s -> {per_h:,.4g} alarms/h at alpha=0.1; alpha for <=1/h = {min(alpha_1h,1.0):.3g}")
report["e1"] = e1

# ---------- E3: extremal metrics all folds (alpha=0.10, ACI and NA3) ----------
def runs_of(mask):
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return s, e


print()
print("=== E3 extremal metrics @ alpha=0.10 (ACI | NA3), mean seeds ===")
e3 = {}
for fold in FOLDS:
    rows = []
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
        s = z["scores"].astype(np.float64)
        lab = z["labels"]
        q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
        out = {}
        for name, pred in [("aci", aci_run(s, q0, 0.1)), ("na3", aci_run(s, q0, 0.1, 3.0))]:
            sil = 0
            cur = 0
            for a in pred:
                cur = cur + 1 if not a else 0
                sil = max(sil, cur)
            miss = 0
            cur = 0
            for i in range(len(lab)):
                if lab[i] > 0.5 and not pred[i]:
                    cur += 1
                    miss = max(miss, cur)
                else:
                    cur = 0
            # per-run delays
            srun, erun = runs_of(lab > 0.5)
            delays = []
            for a, b in zip(srun, erun):
                w = np.where(pred[a:b])[0]
                if len(w):
                    delays.append(int(w[0]))
            delays = np.array(delays) if delays else np.array([-1])
            out[name] = {"max_silence": int(sil), "max_miss_streak": int(miss),
                         "delay_p50": float(np.percentile(delays, 50)), "delay_p95": float(np.percentile(delays, 95)),
                         "delay_p99": float(np.percentile(delays, 99)), "n_runs": int(len(srun)),
                         "n_detected_runs": int((delays >= 0).sum())}
        rows.append(out)
    e3[fold] = {k: {kk: float(np.mean([r[k][kk] for r in rows])) for kk in rows[0][k]} for k in ["aci", "na3"]}
    a, n3 = e3[fold]["aci"], e3[fold]["na3"]
    print(f"{fold} ACI: silence={a['max_silence']:.0f} miss={a['max_miss_streak']:.0f} delayP50/95/99={a['delay_p50']:.0f}/{a['delay_p95']:.0f}/{a['delay_p99']:.0f} runs={a['n_runs']:.0f} detected={a['n_detected_runs']:.0f}")
    print(f"{fold} NA3: silence={n3['max_silence']:.0f} miss={n3['max_miss_streak']:.0f} delayP50/95/99={n3['delay_p50']:.0f}/{n3['delay_p95']:.0f}/{n3['delay_p99']:.0f} runs={n3['n_runs']:.0f} detected={n3['n_detected_runs']:.0f}")
report["e3"] = e3

# ---------- E4: decision cost ----------
print()
print("=== E4 decision cost (sum over 6 folds, alpha=0.10; FP/FN x ratios) ===")
e4 = {}
for name, kappa in [("static", "static"), ("aci", None), ("na3", 3.0), ("na5", 5.0)]:
    fpn = 0
    tot = {"1": 0.0, "10": 0.0, "100": 0.0}
    fps, fns = 0, 0
    for fold in FOLDS:
        for seed in SEEDS:
            z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
            s = z["scores"].astype(np.float64)
            lab = z["labels"]
            if name == "static":
                p = s > float(z["q_static_a0.1"][0])
            else:
                q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
                p = aci_run(s, q0, 0.1, kappa)
            fp = int(((p) & (lab < 0.5)).sum())
            fn = int(((~p) & (lab > 0.5)).sum())
            fps += fp
            fns += fn
            for r in tot:
                tot[r] += fp + float(r) * fn
    e4[name] = {"FP": fps, "FN": fns, **{f"cost_ratio_{k}": v / (6 * 3) for k, v in tot.items()}}
    print(name, e4[name])
report["e4"] = e4

# ---------- E8: oracle quantile decomposition ----------
print()
print("=== E8 oracle decomposition @ alpha=0.10 (mean seeds) ===")
e8 = {}
for fold in FOLDS:
    rows = []
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
        s = z["scores"].astype(np.float64)
        lab = z["labels"]
        q_src = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
        q_oracle = float(np.quantile(s[lab < 0.5], 0.9))
        q0 = q_src
        q = q0
        qsum = 0.0
        for t in range(len(s)):
            a = s[t] > q
            qsum += q
            q = q + GAMMA * max(abs(q), 1e-12) * ((1.0 if a else 0.0) - 0.9)
        rows.append({"oracle_src_ratio": q_oracle / q_src, "aci_oracle_ratio": (qsum / len(s)) / q_oracle})
    e8[fold] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"{fold}: oracle/src={e8[fold]['oracle_src_ratio']:.2f}  ACI-mean/oracle={e8[fold]['aci_oracle_ratio']:.2f}")
report["e8"] = e8

# ---------- D4/D2: block bootstrap CI + equivalence ----------
print()
print("=== D4 block bootstrap 95% CI on ACI gap @0.10 (seed0, block L) ===")
d4 = {}
rngbb = np.random.default_rng(42)
for fold in FOLDS:
    z = np.load(os.path.join(TR, f"{fold}_seed0.npz"))
    s = z["scores"].astype(np.float64)
    lab = z["labels"]
    q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
    p = aci_run(s, q0, 0.1)
    nrm = lab < 0.5
    cov_series = (~p[nrm]).astype(np.float64)   # 1 = covered (not alarmed)
    n = len(cov_series)
    row = {}
    for L0 in [50, 200, 1000]:
        L = min(L0, max(n - 2, 20))
        if n <= L + 1:
            row[f"L{L0}"] = {"ci": None, "equiv_within_0.02": None, "note": "fold too short for block"}
            continue
        nb = int(np.ceil(n / L))
        stats = []
        for b in range(400):
            starts = rngbb.integers(0, n - L, size=nb)
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
            stats.append(cov_series[idx].mean())
        lo, hi = np.percentile(stats, [2.5, 97.5])
        row[f"L{L0}"] = {"ci": [float(lo), float(hi)],
                         "equiv_within_0.02": bool(abs(lo - 0.9) <= 0.0201 and abs(hi - 0.9) <= 0.0201)}
    d4[fold] = row
    print(f"{fold}: " + " | ".join(f"L{k[1:]} CI=[{v['ci'][0]:.3f},{v['ci'][1]:.3f}] eq0.02={v['equiv_within_0.02']}" if v['ci'] else f"L{k[1:]} short" for k, v in row.items()))
report["d4"] = d4

# ---------- D6: macro vs micro ----------
print()
print("=== D6 macro vs micro mean gap @0.10 (ACI / NA3) ===")
macro_aci, macro_na, micro_aci_num, micro_na_num, totw = [], [], 0.0, 0.0, 0.0
for fold in FOLDS:
    gA, gN, w = [], [], 0
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
        s = z["scores"].astype(np.float64)
        lab = z["labels"]
        q0 = float(np.quantile(z["src_cal_scores"].astype(np.float64), 0.9))
        gA.append(gap_of(aci_run(s, q0, 0.1), lab, 0.1))
        gN.append(gap_of(aci_run(s, q0, 0.1, 3.0), lab, 0.1))
        w = int((lab < 0.5).sum())
    macro_aci.append(np.mean(gA))
    macro_na.append(np.mean(gN))
    micro_aci_num += np.mean(gA) * w
    micro_na_num += np.mean(gN) * w
    totw += w
print(f"ACI: macro={np.mean(macro_aci):.3f}  micro={micro_aci_num / totw:.3f}")
print(f"NA3: macro={np.mean(macro_na):.3f}  micro={micro_na_num / totw:.3f}")
report["d6"] = {"aci_macro": float(np.mean(macro_aci)), "aci_micro": float(micro_aci_num / totw),
                "na3_macro": float(np.mean(macro_na)), "na3_micro": float(micro_na_num / totw)}

# ---------- D11: source pool composition ----------
print()
print("=== D11 source pool composition (normal windows after cap) ===")
d11 = {}
for name in ["care", "cwru", "mfpt", "phm16", "pv", "paderborn", "xjtu"]:
    files = [f for f in os.listdir(CACHE) if f.startswith(name + "_")]
    zz = np.load(os.path.join(CACHE, files[0]))
    w, l = zz["windows"], zz["labels"]
    off = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
    rng = np.random.default_rng(off)
    if len(w) > 20000:
        keep = np.sort(rng.choice(len(w), 20000, replace=False))
        l = l[keep]
    d11[name] = int((l < 0.5).sum())
print(d11)
report["d11"] = d11

json.dump(report, open(os.path.join(OUT, "batch3.json"), "w"), indent=2, default=float)
print("\nsaved:", os.path.join(OUT, "batch3.json"))
