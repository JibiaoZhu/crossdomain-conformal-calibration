import json
import os
import time

import numpy as np

TR = r"F:\AutoSci_CrossDomain_Handoff\results\scc-main-traces"
OUT = r"F:\AutoSci_CrossDomain_Handoff\results\revision-batch4"
os.makedirs(OUT, exist_ok=True)
FOLDS = ["F1", "F2", "F3", "F4", "F5", "F6"]
SEEDS = [0, 1, 2]
GAMMA = 0.05
report = {}


def gap_of(preds, labels, alpha=0.1):
    nrm = labels < 0.5
    return abs(float(preds[nrm].mean()) - alpha)


def load(fold, seed):
    z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
    return z["scores"].astype(np.float64), z["labels"], z["src_cal_scores"].astype(np.float64)


# ---------- C2: exclusion-rate monitor + fallback ----------
def monitored_run(scores, q0, alpha, kappa, hi=0.25, span=200.0, mode=None):
    q = float(q0)
    n = len(scores)
    preds = np.empty(n, bool)
    ewma = 0.0
    for i, s in enumerate(scores):
        preds[i] = s > q
        excl = 1.0 if s > kappa * q else 0.0
        ewma = ewma + (excl - ewma) / span
        use_na = (mode is None) or (mode == "na")
        if mode == "auto":
            use_na = ewma < hi
        if (not use_na) or (s <= kappa * q):
            q = q + GAMMA * max(abs(q), 1e-12) * ((1.0 if preds[i] else 0.0) - alpha)
    return preds


print("=== C2 monitored fallback (gap@0.10, mean seeds) ===")
c2 = {}
for fold in FOLDS:
    row = {}
    for name, mode, hi in [("NA3", "na", None), ("auto0.15", "auto", 0.15), ("auto0.25", "auto", 0.25), ("ACI", None, None)]:
        gs = []
        for seed in SEEDS:
            s, lab, sc = load(fold, seed)
            q0 = float(np.quantile(sc, 0.9))
            if name == "ACI":
                p = monitored_run(s, q0, 0.1, 10**9, mode=None)  # kappa huge ~ plain ACI update always
            else:
                p = monitored_run(s, q0, 0.1, 3.0, hi if hi else 0.25, mode=mode)
            gs.append(gap_of(p, lab))
        row[name] = float(np.mean(gs))
    c2[fold] = row
    print(fold, {k: round(v, 3) for k, v in row.items()})
report["c2"] = c2

# ---------- C7: prevalence estimation drives kappa ----------
def em2_prevalence(z, iters=60):
    z = np.asarray(z, dtype=np.float64)
    mu = np.array([np.percentile(z, 25), np.percentile(z, 75)])
    sd = np.array([z.std() * 0.8, z.std() * 0.8])
    pi = 0.3
    for _ in range(iters):
        p1 = pi * np.exp(-0.5 * ((z - mu[1]) / sd[1]) ** 2) / sd[1]
        p0 = (1 - pi) * np.exp(-0.5 * ((z - mu[0]) / sd[0]) ** 2) / sd[0]
        g = p1 / (p0 + p1 + 1e-300)
        pi = g.mean() + 1e-9
        mu[1] = (g * z).sum() / (g.sum() + 1e-12)
        mu[0] = ((1 - g) * z).sum() / ((1 - g).sum() + 1e-12)
        sd[1] = np.sqrt((g * (z - mu[1]) ** 2).sum() / (g.sum() + 1e-12)) + 1e-9
        sd[0] = np.sqrt(((1 - g) * (z - mu[0]) ** 2).sum() / ((1 - g).sum() + 1e-12)) + 1e-9
    return float(pi)


def na3_gap(s, lab, sc, kappa):
    q0 = float(np.quantile(sc, 0.9))
    q = q0
    preds = np.empty(len(s), bool)
    for i, x in enumerate(s):
        preds[i] = x > q
        if x <= kappa * q:
            q = q + GAMMA * max(abs(q), 1e-12) * ((1.0 if preds[i] else 0.0) - 0.1)
    return gap_of(preds, lab)


print()
print("=== C7 prevalence estimate -> kappa rule (true prev; p_hat; rule kappa; gap; best-kappa gap) ===")
c7 = {}
for fold in FOLDS:
    rows = []
    for seed in SEEDS:
        s, lab, sc = load(fold, seed)
        true_prev = float((lab > 0.5).mean())
        z = np.log(np.maximum(s, 1e-12))
        phat = em2_prevalence(z)
        k_rule = 2.0 if phat > 0.4 else (3.0 if phat > 0.2 else 5.0)
        g_rule = na3_gap(s, lab, sc, k_rule)
        gs = {k: na3_gap(s, lab, sc, k) for k in [1.5, 2.0, 3.0, 5.0, 10.0]}
        g_best = min(gs.values())
        rows.append({"true": true_prev, "phat": phat, "k_rule": k_rule, "g_rule": g_rule, "g_best": g_best})
    c7[fold] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"{fold}: true={c7[fold]['true']:.2f} phat={c7[fold]['phat']:.2f} k_rule={c7[fold]['k_rule']:.1f} "
          f"g_rule={c7[fold]['g_rule']:.3f} g_best={c7[fold]['g_best']:.3f}")
report["c7"] = c7

# ---------- C8: additive / DtACI-style / PID-style ----------
print()
print("=== C8 mechanisms (gap@0.10 mean seeds; F4/F6 gap; F4 osc) ===")


def custom_run(scores, q0, alpha, step_fn):
    q = float(q0)
    preds = np.empty(len(scores), bool)
    prev, prev2 = 0.0, 0.0
    logq = np.empty(len(scores))
    for i, s in enumerate(scores):
        preds[i] = s > q
        err = (1.0 if preds[i] else 0.0) - alpha
        q = max(q + step_fn(q, err, prev, prev2), 1e-12)
        prev2, prev = prev, err
        logq[i] = np.log(max(q, 1e-12))
    osc = int((np.diff(np.sign(np.diff(logq))) != 0).sum())
    return preds, osc


def make_step(kind, qsrc, gp, gd):
    if kind == "additive":
        return lambda q, e, p1, p2: GAMMA * qsrc * e
    if kind == "mixed":
        return lambda q, e, p1, p2: GAMMA * (0.5 * qsrc + 0.5 * abs(q)) * e
    if kind == "pid":
        return lambda q, e, p1, p2: GAMMA * abs(q) * (e + gp * (e - p1) + gd * (e - 2 * p1 + p2))
    raise ValueError


c8 = {}
configs = [("additive", None, None), ("mixed", None, None), ("pid", 0.05, 0.0), ("pid", 0.0, 0.05), ("pid", 0.05, 0.05)]
for kind, gp, gd in configs:
    name = kind if kind != "pid" else f"pid p={gp} d={gd}"
    gaps, f46, oscs = [], {}, []
    for fold in FOLDS:
        for seed in SEEDS:
            s, lab, sc = load(fold, seed)
            q0 = float(np.quantile(sc, 0.9))
            step = make_step(kind, q0, gp, gd)
            p, osc = custom_run(s, q0, 0.1, step)
            gaps.append(gap_of(p, lab))
            if fold in ("F4", "F6"):
                f46.setdefault(fold, []).append(gap_of(p, lab))
            if fold == "F4":
                oscs.append(osc)
    c8[name] = {"mean_gap": float(np.mean(gaps)), "F4": float(np.mean(f46["F4"])), "F6": float(np.mean(f46["F6"])),
                "F4_osc": float(np.mean(oscs))}
    print(f"{name:16s} mean_gap={c8[name]['mean_gap']:.3f} F4={c8[name]['F4']:.3f} F6={c8[name]['F6']:.3f} F4_osc={c8[name]['F4_osc']:.0f}")
report["c8"] = c8

# ---------- C9: weighted conformal (score-space density ratio) ----------
from scipy.stats import gaussian_kde

print()
print("=== C9 weighted conformal (gap@0.10, mean seeds) ===")
c9 = {}
for fold in FOLDS:
    rows = []
    for seed in SEEDS:
        s, lab, sc = load(fold, seed)
        zs = np.log(np.maximum(sc, 1e-12))
        zt = np.log(np.maximum(s[:512], 1e-12))
        try:
            ks = gaussian_kde(zs)
            kt = gaussian_kde(zt)
            w = kt.pdf(zs) / (ks.pdf(zs) + 1e-300)
            w = np.clip(w, 0.01, 100.0)
            order = np.argsort(sc)
            cw = np.cumsum(w[order])
            cw /= cw[-1]
            k = int(np.searchsorted(cw, 0.90))
            q_w = sc[order[min(k, len(sc) - 1)]]
        except Exception:
            q_w = float(np.quantile(sc, 0.9))
        p = s > q_w
        rows.append({"gap": gap_of(p, lab), "alarm": float(p.mean())})
    c9[fold] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"{fold}: gap={c9[fold]['gap']:.3f} alarm={c9[fold]['alarm']:.3f}")
report["c9"] = c9

# ---------- E2: debounce and hysteresis ----------
print()
print("=== E2 debounce (m) / hysteresis (h) on NA3, gap@0.10 & det ===")
e2 = {}
for fold in FOLDS:
    rows = []
    for seed in SEEDS:
        s, lab, sc = load(fold, seed)
        q0 = float(np.quantile(sc, 0.9))
        q = q0
        raw = np.empty(len(s), bool)
        qs = np.empty(len(s))
        for i, x in enumerate(s):
            raw[i] = x > q
            qs[i] = q
            if x <= 3.0 * q:
                q = q + GAMMA * max(abs(q), 1e-12) * ((1.0 if raw[i] else 0.0) - 0.1)
        anom = lab > 0.5
        base = {"alarm": float(raw.mean()), "det": float(raw[anom].mean()), "gap": gap_of(raw, lab)}
        for m in (3, 5):
            deb = np.zeros_like(raw)
            cum = np.cumsum(raw)
            for t in range(m - 1, len(raw)):
                deb[t] = (cum[t] - (cum[t - m] if t - m >= 0 else 0)) == m
            base[f"deb{m}_alarm"] = float(deb.mean())
            base[f"deb{m}_det"] = float(deb[anom].mean())
        for h in (0.05, 0.10):
            on = False
            hyst = np.empty_like(raw)
            for i, x in enumerate(s):
                if not on and x > qs[i]:
                    on = True
                elif on and x < qs[i] * (1 - h):
                    on = False
                hyst[i] = on
            base[f"hyst{h}_alarm"] = float(hyst.mean())
            base[f"hyst{h}_det"] = float(hyst[anom].mean())
        rows.append(base)
    e2[fold] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    b = e2[fold]
    print(f"{fold}: base {b['alarm']:.3f}/{b['det']:.3f} | deb3 {b['deb3_alarm']:.3f}/{b['deb3_det']:.3f} "
          f"| deb5 {b['deb5_alarm']:.3f}/{b['deb5_det']:.3f} | hyst.05 {b['hyst0.05_alarm']:.3f}/{b['hyst0.05_det']:.3f} "
          f"| hyst.10 {b['hyst0.1_alarm']:.3f}/{b['hyst0.1_det']:.3f}")
report["e2"] = e2

# ---------- D3/D13: seed spreads ----------
print()
print("=== D3/D13 seed spreads (max over folds; ACI/NA3 gap@0.1; AUROC) ===")
max_gap_spread, max_auc_spread = 0.0, 0.0
for fold in FOLDS:
    gaps_aci, gaps_na, aucs = [], [], []
    for seed in SEEDS:
        z = np.load(os.path.join(TR, f"{fold}_seed{seed}.npz"))
        s, lab, sc = load(fold, seed)
        q0 = float(np.quantile(sc, 0.9))
        p1 = monitored_run(s, q0, 0.1, 3.0, mode="na")
        gaps_na.append(gap_of(p1, lab))
        gaps_aci.append(gap_of(monitored_run(s, q0, 0.1, 10**9), lab))
        aucs.append(float(z["auroc"]))
    max_gap_spread = max(max_gap_spread, float(np.std(gaps_aci)), float(np.std(gaps_na)))
    max_auc_spread = max(max_auc_spread, float(np.std(aucs)))
    print(f"{fold}: gap SD ACI={np.std(gaps_aci):.3f} NA3={np.std(gaps_na):.3f} AUROC SD={np.std(aucs):.3f}")
report["spread"] = {"max_gap_sd": max_gap_spread, "max_auroc_sd": max_auc_spread}

# ---------- E5: CPU per-update cost (this x86 machine, calibration layer only) ----------
print()
print("=== E5 calibration-layer per-update CPU cost (this x86 host) ===")
s = np.linspace(0.01, 2.0, 200000)
q = 0.5
t0 = time.perf_counter()
for x in s:
    a = x > q
    q += GAMMA * max(abs(q), 1e-12) * ((1.0 if a else 0.0) - 0.1)
dt = (time.perf_counter() - t0) / len(s)
print(f"pure-python update: {dt * 1e6:.2f} us/update")
report["e5_cpu_us"] = dt * 1e6

json.dump(report, open(os.path.join(OUT, "batch4.json"), "w"), indent=2, default=float)
print("saved:", os.path.join(OUT, "batch4.json"))
