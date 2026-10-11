import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = r"G:\autosci-opencode\survey\figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(7.0, 2.6))
ax.axis("off")
boxes = [
    (0.02, 0.55, "Score\nproducers", "#D9E8F5"),
    (0.20, 0.55, "Transfer\nmechanisms", "#DDEEDD"),
    (0.40, 0.55, "Calibration\n(reliability)", "#F7E2D0"),
    (0.60, 0.55, "Evaluation\nprotocol", "#EFE0EE"),
]
for x, y, label, c in boxes:
    ax.add_patch(plt.Rectangle((x, y), 0.16, 0.34, facecolor=c, edgecolor="k", lw=0.9))
    ax.text(x + 0.08, y + 0.17, label, ha="center", va="center", fontsize=8, weight="bold")

items = {
    0.20: ["feature alignment\n(DACAD)", "frequency-aware\ninvariance", "TTA / PEFT", "foundation models", "sensor relations"],
    0.40: ["static conformal", "ACI / DtACI / PID", "EVT thresholds", "marginal vs\nconditional coverage", "alarm budget"],
    0.60: ["range-F1 / VUS", "point-adjustment\ncritiques", "label provenance", "rankability audits", "transfer cost"],
}
for x, its in items.items():
    for i, it in enumerate(its):
        yy = 0.42 - i * 0.20
        ax.text(x + 0.01, yy + 0.07, "\u2022 " + it.replace("\n", " "), fontsize=6.4, va="center")
for x0 in [0.18, 0.38, 0.58]:
    ax.annotate("", xy=(x0 + 0.02, 0.72), xytext=(x0, 0.72), arrowprops=dict(arrowstyle="-|>", lw=0.9))
ax.set_xlim(0, 0.78)
ax.set_ylim(0, 1)
fig.savefig(os.path.join(FIG, "fig_taxonomy.pdf"))
print("taxonomy figure saved")
