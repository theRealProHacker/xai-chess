"""Bar chart: mean probe AUROC per feature family, layer 5 vs the input embedding.

  python lc0-probe/plot_square_families.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

enc = json.load(open(ROOT / "lc0-probe/results_square_encoder5_ln1.json"))["bits"]
emb = json.load(open(ROOT / "lc0-probe/results_square_attn_body_rehape.json"))["bits"]
fam = {}
for k, v in enc.items(): fam.setdefault(v["group"], []).append((v["auroc"], emb[k]["auroc"]))
rows = sorted(((g, *np.mean(v, 0), len(v)) for g, v in fam.items()), key=lambda r: -r[1])

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
n = len(rows)
fig, ax = plt.subplots(figsize=(9, 0.19 * n + 1.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
y = np.arange(n)
# two bars per row: layer 5 above, the embedding below
ax.barh(y - 0.12, [r[1] - 0.5 for r in rows], left=0.5, height=0.30, color=BLUE, label="Layer 5, after attention", zorder=2)
ax.barh(y + 0.15, [r[2] - 0.5 for r in rows], left=0.5, height=0.20, color=ORANGE, label="Input embedding", zorder=2)
for yi, r in zip(y, rows):   # number columns, right-aligned
    gain = f"{round(r[1], 3) - round(r[2], 3) + 0.0:+.3f}"   # from the printed values, so the row adds up
    gain = gain.replace("-0.000", "+0.000").replace("-", "\u2212")
    for x, v in [(1.050, f"{r[1]:.3f}"), (1.125, f"{r[2]:.3f}"), (1.180, gain)]:
        ax.text(x, yi, v, ha="right", va="center", fontsize=7.5, color=INK)
for x, h in [(1.050, "Layer 5"), (1.125, "Embedding"), (1.180, "Gain")]:
    ax.text(x, -1.0, h, ha="right", va="bottom", fontsize=7.5, color=INK, weight="bold")
ax.set_yticks(y, [f"{r[0]}  ({r[3]})" for r in rows], fontsize=7.5, color=INK)
ax.invert_yaxis()
ax.set_xlim(0.5, 1.185); ax.set_ylim(n - 0.4, -0.7)
ax.set_xticks(np.arange(0.5, 1.01, 0.05)); ax.set_xbound(0.5, 1.185)
ax.tick_params(axis="x", colors=MUTED, labelsize=8, top=False, labeltop=False)
ax.tick_params(axis="y", length=0)
ax.set_ylabel("Feature family (bits probed)", color=MUTED, fontsize=8)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlabel("Mean test AUROC per family (0.5 = chance)", color=MUTED, fontsize=8)
ax.set_title("Linear probe per square token, Leela t1-256x10, 20,000 positions", color=INK, fontsize=11, loc="left", pad=22)
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, borderaxespad=0.2)
fig.tight_layout()
fig.subplots_adjust(left=fig.subplotpars.left + 0.015)
fig.savefig(ROOT / "lc0-probe/square_families.png", dpi=160, facecolor=SURFACE)
