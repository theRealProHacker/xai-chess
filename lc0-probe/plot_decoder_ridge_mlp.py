"""Two panels: MLP test gain over ridge, and each decoder's train-test gap, at every ln1 tap.

  python lc0-probe/plot_decoder_ridge_mlp.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ridge = json.load(open(ROOT / "lc0-probe/results_square_decoder.json"))["taps"]
mlp = json.load(open(ROOT / "lc0-probe/results_square_decoder_mlp.json"))["taps"]
taps = list(ridge)
names = ["Emb." if t == "/attn_body/rehape" else t.split("/")[1].replace("encoder", "L") for t in taps]
rt = np.array([ridge[t]["features + square"]["test"] for t in taps]); rr = np.array([ridge[t]["features + square"]["train"] for t in taps])
mt = np.array([mlp[t]["test"] for t in taps]); mr = np.array([mlp[t]["train"] for t in taps])
panels = [("MLP gain over ridge (test FVU, ridge \u2212 MLP)", [("MLP gain", rt - mt, "#52514e")]),
          ("Overfitting (test \u2212 train FVU)", [("Ridge", rt - rr, "#2a78d6"), ("MLP", mt - mr, "#eb6834")])]

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
fig, axs = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True, facecolor=SURFACE, gridspec_kw={"wspace": 0.1})
x = np.arange(len(taps))
for ax, (title, ss) in zip(axs, panels):
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for label, y, c in ss:
        ax.plot(x, y, color=c, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=2)
        ax.plot(x, y, "o", ms=5, color=c, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(x[-1] + 0.25, y[-1], label, va="center", fontsize=8, color=INK)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.set_xlim(-0.5, len(taps) + 0.9); ax.set_ylim(-0.005, 0.06)
    ax.set_xticks(x, names, fontsize=8, color=INK); ax.set_yticks(np.arange(0, 0.061, 0.01))
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for sp in ax.spines.values(): sp.set_visible(False)
axs[0].set_ylabel("FVU difference", color=INK, fontsize=8.5)
fig.suptitle("Ridge vs MLP decoder, features + square \u2192 Leela token (t1-256x10)", color=INK, fontsize=10.5, x=0.08, ha="left")
fig.supxlabel("Embedding output, then each layer 0\u20139 halfway through (after attention)", color=MUTED, fontsize=8)
fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.14)
fig.savefig(ROOT / "lc0-probe/square_decoder_ridge_mlp.png", dpi=160, facecolor=SURFACE)
