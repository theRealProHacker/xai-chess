"""2x2 chart: held-out FVU of the per-square decoder at every tap, plus what the features add.

  python lc0-probe/plot_square_decoder.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

res = json.load(open(ROOT / "lc0-probe/results_square_decoder.json"))["taps"]
taps = list(res)
names = ["Emb." if t == "/attn_body/rehape" else t.split("/")[1].replace("encoder", "L") for t in taps]
test = {m: np.array([res[t][m]["test"] for t in taps]) for m in ["square only", "features only", "features + square"]}
series = [("Square only", test["square only"], "#2a78d6"),
          ("Features only", test["features only"], "#eb6834"),
          ("Features + square", test["features + square"], "#1baf7a"),
          ("Features add: square only − (features + square)",
           test["square only"].round(3) - test["features + square"].round(3), "#52514e")]   # adds up as printed

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TITLE = "Linear decoder from a square's features to its Leela token (t1-256x10), held-out positions"
XLABEL = "Embedding output, then each layer 0–9 halfway through (after attention)"
x = np.arange(len(taps))

def style(ax, right):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.9, len(taps) - 0.5 + right); ax.set_ylim(-0.04, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_xticks(x, names)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0, labelsize=8, colors=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)

def line(ax, y, c, label=None, z=0):
    ax.plot(x, y, color=c, lw=2, solid_joinstyle="round", solid_capstyle="round", label=label, zorder=2 + z)
    ax.plot(x, y, "o", ms=6, color=c, mec=SURFACE, mew=2, zorder=3 + z)

# 2x2, one each, same y scale
fig, axs = plt.subplots(2, 2, figsize=(9, 6.4), facecolor=SURFACE,
                        gridspec_kw={"hspace": 0.34, "wspace": 0.16})
for ax, (label, y, c) in zip(axs.flat, series):
    line(ax, y, c)
    style(ax, 0)
    ax.set_title(label, color=INK, fontsize=9, loc="left")
    ax.set_ylabel("Explained beyond the square (higher = more)" if c == "#52514e" else "Unexplained (FVU, lower = better)", color=INK, fontsize=8.5)
    for i in sorted({0, int(y.argmax()), len(y) - 1}):           # start, peak, end; on the side away from the line
        nb = [y[j] for j in (i - 1, i + 1) if 0 <= j < len(y)]
        dy, dx, ha = (0.07, 0, "center") if all(v < y[i] for v in nb) else (-0.08, 0, "center")
        if y[i] + dy < 0.05: dy, dx, ha = 0.06, -0.05, "right"   # no room below: above-left
        ax.text(x[i] + dx, y[i] + dy, f"{y[i]:.3f}", ha=ha, va="center", fontsize=7.5, color=INK)
fig.suptitle(TITLE, color=INK, fontsize=10.5, x=0.08, ha="left")
fig.supxlabel(XLABEL, color=MUTED, fontsize=8)
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.1)
fig.savefig(ROOT / "lc0-probe/square_decoder_grid.png", dpi=160, facecolor=SURFACE)
