"""Two line charts of hybrid_layer_families.py's results: each layer's distribution over the feature families (share of
|signed family sum| per position), in L5 + L2 + L0 (one decoder path per layer) and in the single-layer runs; the lower chart also L5 + L2 + L0's whole-model attribution.
Families ordered by the L0 path of L5 + L2 + L0. Line charts, one above the other. -> lc0-probe/layer_families.png

  python3 lc0-probe/plot_layer_families.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
R = lambda s: json.load(open(ROOT / f"lc0-probe/results_hybrid_layer_families_games{s}.json"))
mag = lambda r: np.abs(np.array(r["signed"])).sum(0)                       # (layers, F)
deep = R("")
fams, D = deep["families"], mag(deep)
order = np.argsort(-D[deep["layers"].index("L0")], kind="stable")          # by the L0 path of L5 + L2 + L0
LAY = ["L5", "L2", "L0"]
dist = lambda M: M / M.sum(1, keepdims=True)
series = {"deep": dist(D)[[deep["layers"].index(L) for L in LAY]],
          "single": np.stack([dist(mag(R(f"_{L.lower()}")))[0] for L in LAY])}
T = np.abs(np.array(deep["signed"]).sum(1)).sum(0); T = T / T.sum()        # whole model: |family's total over layers| per position
TITLE = {"deep": "L5 + L2 + L0: each layer's path", "single": "Each layer's decoder alone"}

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"L5": "#2a78d6", "L2": "#eb6834", "L0": "#1baf7a"}
top = max(v.max() for v in series.values()) * 100 * 1.08
x = np.arange(len(order))
fig, axs = plt.subplots(2, 1, figsize=(8, 6.4), dpi=160, sharex=True, facecolor=SURFACE)
for ax, (k, S) in zip(axs, series.items()):
    ax.set_facecolor(SURFACE)
    for j, L in enumerate(LAY):
        ax.plot(x, S[j, order] * 100, color=COL[L], lw=2, label=L, zorder=3)
    if k == "single": ax.plot(x, T[order] * 100, color=INK, lw=1.5, ls=(0, (4, 2)), label="L5 + L2 + L0, whole model", zorder=4)
    ax.set_xlim(-0.5, len(order) - 0.5); ax.set_ylim(0, top)
    ax.set_ylabel("Share of the layer's attribution (%)", color=MUTED, fontsize=8)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(TITLE[k], color=INK, fontsize=10, loc="left", pad=6, y=1.0 if k == "deep" else 0.935)
axs[0].legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
axs[1].legend(*[[v[-1]] for v in axs[1].get_legend_handles_labels()], loc="upper right", frameon=False, fontsize=8)
axs[1].set_xticks(x, [fams[i] for i in order], rotation=55, ha="right", rotation_mode="anchor", fontsize=8, color=INK)
fig.tight_layout(); fig.subplots_adjust(hspace=0.12)
out = ROOT / "lc0-probe/layer_families.png"
fig.savefig(out, facecolor=SURFACE); print("wrote", out)
