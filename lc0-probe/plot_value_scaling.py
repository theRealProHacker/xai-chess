"""Held-out FVU of the board-level value decoder against training positions, extrapolated to 100M.

Solid: measured (results_value_decoder_mlp_n*.json). Dashed: power-law fit FVU = b·N^−α on all sizes.
Brackets: range of the power-law fits that leave one measured size out. A log-linear fit is not drawn: it
crosses FVU 0 before 100M.

  python lc0-probe/plot_value_scaling.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

runs = sorted((json.load(open(p)) for p in (ROOT / "lc0-probe").glob("results_value_decoder_mlp_n*.json")), key=lambda r: r["n"])
N = np.array([r["n"] for r in runs], float)
taps = [("/value/dense1/mish", "dense1", "#2a78d6", -1), ("/value/dense2/add", "dense2", "#eb6834", 0),
        ("/output/wdl", "output (WDL)", "#1baf7a", 1)]                    # last: value labels above-right (1), right (0), below-left (-1)
END = 1e8
grid = np.geomspace(N[0], END, 200)                                      # fit drawn over the measured sizes too
power = lambda n, y: lambda x: np.exp(np.polyval(np.polyfit(np.log(n), np.log(y), 1), np.log(x)))

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axvspan(N[-1], 1e10, color="#f1f0ec", zorder=0, lw=0)
ends, meas = {}, []
for t, name, c, side in taps:
    y = np.array([r["taps"][t]["test"] for r in runs])
    pw = power(N, y)
    loo = np.array([power(np.delete(N, i), np.delete(y, i))(grid) for i in range(len(N))])
    ax.plot(N, y, color=c, lw=2, solid_capstyle="round", zorder=3)
    ax.plot(N, y, "o", ms=6, color=c, mec=SURFACE, mew=2, zorder=4)
    ins, out = grid <= N[-1], grid >= N[-1]
    ax.plot(grid[ins], pw(grid[ins]), color=c, lw=1, ls=(0, (2, 2)), alpha=0.6, zorder=2)   # fit under the data
    ax.plot(grid[out], pw(grid[out]), color=c, lw=1.5, ls=(0, (4, 3)), alpha=0.75, zorder=3)
    meas.append((name, c, y))
    ends[name] = (pw(END), loo.min(0)[-1], loo.max(0)[-1], c)
order = sorted(ends, key=lambda k: ends[k][0])
for i, name in enumerate(order):                                          # dense1, dense2 end 0.01 apart: spread them
    v, lo, hi, c = ends[name]
    ax.text(END * 1.15, v + ((-0.009, 0.009)[i] if i < 2 else 0), f"{name}  {v:.2f}  [{lo:.2f}–{hi:.2f}]",
            va="center", fontsize=8, color=c, fontweight="bold")

tx = [0.27, 0.42, 0.48, 0.54, 0.60]                                        # measured values, a table under the curves
T = dict(transform=ax.transAxes, fontsize=7.5, va="center", zorder=6)
ax.add_patch(plt.Rectangle((0.26, 0.06), 0.36, 0.22, transform=ax.transAxes, color=SURFACE, ec=GRID, lw=0.8, zorder=5))
ax.text(tx[0], 0.25, "Measured test FVU", ha="left", color=INK, **T)
for x, n in zip(tx[1:], N): ax.text(x, 0.21, f"{int(n) // 1000}k", ha="right", color=MUTED, **T)
for r, (name, c, y) in enumerate(meas[::-1]):
    ax.text(tx[0], 0.17 - 0.04 * r, name, ha="left", color=c, fontweight="bold", **T)
    for x, v in zip(tx[1:], y): ax.text(x, 0.17 - 0.04 * r, f"{v:.3f}", ha="right", color=INK, **T)
ticks = [2e4, 5e4, 1e5, N[-1], 1e6, 1e7, 1e8]
ax.set_xscale("log"); ax.set_xlim(1.15e4, 3.2e9); ax.set_ylim(0, 0.88)
ax.set_xticks(ticks, ["20k", "50k", "100k", "151k", "1M", "10M", "100M"]); ax.minorticks_off()
lb = ax.get_xticklabels(); lb[2].set_ha("right"); lb[3].set_ha("left")   # 100k and 151k sit close
ax.set_yticks(np.arange(0, 0.81, 0.1))
ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
ax.tick_params(axis="x", colors=INK, labelsize=8, length=0)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for s in ax.spines.values(): s.set_visible(False)
ax.text(np.sqrt(N[0] * N[-1]), 0.87, "Measured", ha="center", va="top", fontsize=8.5, color=MUTED)
ax.text(N[-1] * 1.3, 0.87, "Extrapolated with a power law FVU = b·N$^{−α}$ fitted to all 4 sizes, no floor\n[ ] = spread of the fits that leave one size out, not a confidence interval",
        ha="left", va="top", fontsize=8.5, color=MUTED)
ax.set_ylabel("Unexplained variance, held-out positions (FVU, lower = better)", color=INK, fontsize=8.5)
ax.set_xlabel("Training positions (log scale)", color=INK, fontsize=8)
fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.11)
fig.suptitle("MLP decoder from per-square features to Leela's value head: test FVU by training positions",
             color=INK, fontsize=10.5, x=ax.get_position().x0, ha="left")
fig.savefig(ROOT / "lc0-probe/value_decoder_scaling.png", dpi=160, facecolor=SURFACE)
