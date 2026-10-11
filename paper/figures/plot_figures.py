import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"G:\autosci-opencode"
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
CB = {"static": "#D55E00", "aci": "#0072B2", "shiftband": "#009E73", "mondrian": "#CC79A7"}
FOLD_TARGET = {"F1": "PHM16 (fab)", "F2": "CARE (wind)", "F3": "PVDAQ (solar)",
               "F4": "CWRU+PB (mach.)", "F5": "MFPT (mach.)", "F6": "XJTU-SY (mach.)"}

res = json.load(open(os.path.join(ROOT, "results", "scc-main", "all_results.json"), encoding="utf-8"))
alphas = res["diag_corr"] and [0.05, 0.1, 0.2, 0.3]

# ---- Fig 2: coverage vs alpha, small multiples
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True, sharey=True)
for ax, fold in zip(axes.flat, ["F1", "F2", "F3", "F4", "F5", "F6"]):
    fd = res["folds"][fold]
    for mode in ["static", "aci", "shiftband", "mondrian"]:
        ys = []
        for alpha in alphas:
            vals = []
            for sd in fd["seeds"].values():
                ev = sd["evals"].get(f"{mode}_a{alpha:g}")
                if ev:
                    vals.append(ev["coverage"])
            ys.append(np.mean(vals) if vals else np.nan)
        ax.plot(alphas, ys, "-o", ms=2.5, lw=1.1, color=CB[mode], label=mode)
    nominal = [1 - a for a in alphas]
    ax.plot(alphas, nominal, "--", lw=0.8, color="k", label="nominal")
    ax.set_title(FOLD_TARGET[fold])
    ax.set_ylim(0.05, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    if fold in ("F1", "F4"):
        ax.set_ylabel("normal-window coverage")
    if fold in ("F5", "F6", "F4"):
        ax.set_xlabel(r"nominal level $\alpha$ (target coverage $1-\alpha$)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.savefig(os.path.join(FIG, "fig_coverage_alpha.pdf"))
plt.close(fig)

# ---- Fig 3: operational (inflation + q_t traces)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
folds = ["F1", "F2", "F3", "F4", "F5", "F6"]
infl = []
for fold in folds:
    vals = [sd["evals"]["shiftband_a0.1"]["inflation"] for sd in res["folds"][fold]["seeds"].values()]
    infl.append(np.mean(vals))
ax1.bar(folds, infl, color="#0072B2", alpha=0.85)
ax1.set_yscale("log")
ax1.set_ylabel(r"interval inflation  $\overline{q_t}/q_{\mathrm{src}}$ (log)")
ax1.set_title("Transfer cost @ " + r"$\alpha=0.1$")
ax1.axhline(1.0, color="k", lw=0.7, ls="--")
ax1.grid(alpha=0.25, lw=0.4, axis="y")

tr = os.path.join(ROOT, "results", "scc-main-traces")
for fold, col in [("F2", "#0072B2"), ("F6", "#D55E00")]:
    z = np.load(os.path.join(tr, f"{fold}_seed0.npz"))
    n = len(z["scores"])
    k = max(n // 1500, 1)
    idx = np.arange(0, n, k)
    ax2.plot(idx, z["q_static_a0.1"][idx], lw=0.7, ls=":", color=col, label=f"{fold} static")
    ax2.plot(idx, z["q_aci_a0.1"][idx], lw=1.0, color=col, label=f"{fold} ACI")
ax2.set_xlabel("target window index")
ax2.set_ylabel(r"threshold $q_t$")
ax2.set_title(r"Threshold traces ($\alpha=0.1$, seed 0)")
ax2.legend(frameon=False)
ax2.grid(alpha=0.25, lw=0.4)
fig.savefig(os.path.join(FIG, "fig_operational.pdf"))
plt.close(fig)

# ---- Fig 4: decoupling scatter
fig, ax = plt.subplots(figsize=(3.4, 2.8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, fold in enumerate(folds):
    fd = res["folds"][fold]
    xs, ys = [], []
    for sd in fd["seeds"].values():
        gap = np.mean([sd["evals"][f"aci_a{a:g}"]["abs_coverage_gap"] for a in alphas])
        xs.append(sd["auroc"])
        ys.append(gap)
    ax.scatter(xs, ys, s=22, color=colors[i], label=fold, edgecolor="k", linewidth=0.3)
    ax.annotate(fold, (np.mean(xs), np.mean(ys)), fontsize=6.5,
                xytext=(3, 3), textcoords="offset points")
ax.axvline(0.5, color="k", lw=0.7, ls="--")
ax.text(0.505, ax.get_ylim()[1] * 0.9, "chance ranker", fontsize=6.5, rotation=90, va="top")
ax.set_xlabel("detection AUROC (per fold/seed)")
ax.set_ylabel(r"ACI mean $|\mathrm{coverage\ gap}|$ (4 $\alpha$)")
ax.set_title("Calibration vs discrimination")
ax.grid(alpha=0.25, lw=0.4)
fig.savefig(os.path.join(FIG, "fig_decoupling.pdf"))
plt.close(fig)

# ---- Fig 1: framework diagram (schematic)
fig, ax = plt.subplots(figsize=(7.0, 1.6))
ax.axis("off")
boxes = [
    (0.02, "Target stream\nwindows $x_t$"),
    (0.20, "Score $s_t$\n(recon. error)"),
    (0.38, "Calibration layer\nstatic | ACI | shift | Mondrian"),
    (0.62, "Decision\nalarm if $s_t>q_t$"),
    (0.80, "Report\ncoverage, FA rate,\ninflation"),
]
for x, label in boxes:
    ax.add_patch(plt.Rectangle((x, 0.25), 0.16, 0.5, fill=True, facecolor="#F0F0F0", edgecolor="k", lw=0.8))
    ax.text(x + 0.08, 0.5, label, ha="center", va="center", fontsize=7)
for x0, x1 in [(0.18, 0.20), (0.36, 0.38), (0.60, 0.62), (0.78, 0.80)]:
    ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="k"))
ax.annotate("", xy=(0.46, 0.25), xytext=(0.46, 0.25), arrowprops=dict(arrowstyle="-"))  # placeholder
ax.text(0.46, 0.08, "feedback: coverage error updates $q_t$ (ACI)", ha="center", fontsize=7)
ax.annotate("", xy=(0.46, 0.22), xytext=(0.46, 0.12), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#0072B2"))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.savefig(os.path.join(FIG, "fig_framework.pdf"))
plt.close(fig)

print("figures written to", FIG)
