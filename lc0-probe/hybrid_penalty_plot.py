"""Held-out L5 FVU of the deepest-first stack over the size penalty lambda (linear): full stack on the left axis; on the right, L5 influence
(hybrid_endmlp_swap.py); the version without and with the prediction input (lambda = 0) and the gate as bars before the sweep, on the same axes. -> lc0-probe/hybrid_penalty_sweep.png

  python3 lc0-probe/hybrid_penalty_plot.py   (system python: matplotlib)
"""
import glob, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
R = {}
for f in glob.glob(str(ROOT / "lc0-probe/results_hybrid_endmlp_ablate_games_n100000*.json")):
    R.update(json.load(open(f)))
lam = {0.0: R["with prediction"]}
for k, v in R.items():
    if k.startswith("pred_pen"): lam[float(k[8:])] = v
SW = json.load(open(ROOT / "lc0-probe/results_hybrid_endmlp_swap_games_n100000.json"))   # L5 influence (hybrid_endmlp_swap.py)
inf = lambda name: SW[name]["L5 influence"]
xs = sorted(lam)
full = [lam[x]["full"]["L5 FVU"] for x in xs]
no5 = [inf("with prediction")] + [inf(f"pred_pen{x:g}") for x in xs[1:]]

ACC, GREEN, INK, MUTED, GRID, BG = "#55575c", "#1f9e5a", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
X0 = 2.1                                                                 # where lambda = 0 sits
fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=BG); ax.set_facecolor(BG)
ax2 = ax.twinx()                                                         # left: full stack, right: L5 influence
for ys, c, a_, lab in ((full, ACC, ax, "full stack (left axis)"), (no5, GREEN, ax2, "L5 influence (right axis)")):
    a_.plot([X0 + x for x in xs], ys, color=c, lw=2, zorder=3, label=lab)
    a_.scatter([X0 + x for x in xs], ys, s=36, color=c, edgecolor=BG, linewidth=2, zorder=4)
gate, nop = (R.get("gate") or R["pred_gate"]), R["without prediction"]
BX, W = (0.1, 0.7, 1.3), 0.14                                                  # the other types as bars before the sweep
for x, r, n_ in zip(BX, (nop, R["with prediction"], gate), ("without prediction", "with prediction", "pred_gate")):
    ax.bar(x - W / 2 - 0.01, r["full"]["L5 FVU"], W, color=ACC, zorder=3)
    ax2.bar(x + W / 2 + 0.01, inf(n_), W, color=GREEN, zorder=3)
ax.axvline(1.7, color=GRID, lw=1)
k = sum(a_ * b_ for a_, b_ in zip(full, no5)) / sum(a_ * a_ for a_ in full)     # right/left scale: least-squares fit of the
top = max(max(full), max(no5 + [inf("without prediction"), inf("pred_gate")]) / k) * 1.05   # without-L5 line onto the full one
ax.set_ylim(0, top); ax2.set_ylim(0, top * k)
ax.set_xlim(-0.25, X0 + 3.15)
T = [0, 0.5, 1, 1.5, 2, 2.5, 3]
ax.set_xticks([*BX, *(X0 + t for t in T)], ["no\nprediction\ninput", "with\nprediction\ninput", "gate", *(f"{t:g}" for t in T)])
ax.annotate("size penalty λ", (X0 + 1.5, 0), xycoords=("data", "axes fraction"), xytext=(0, -26), textcoords="offset points", color=INK, ha="center", va="top")
ax.set_ylabel("accuracy (L5 FVU)", color=INK); ax2.set_ylabel("L5 influence: output moved by swapping L5", color=INK)
ax.grid(True, axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for a_ in (ax, ax2):
    a_.spines["top"].set_visible(False)
    for s_ in ("left", "right", "bottom"): a_.spines[s_].set_color(GRID)
    a_.tick_params(colors=MUTED)
h = [ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
ax.legend(h[0][0] + h[1][0], h[0][1] + h[1][1], frameon=False, loc="upper right", labelcolor=INK)
ax.set_title("Deepest-first stack (L5 + L2 + L0) with the prediction input", color=INK, loc="left", fontsize=12)
fig.tight_layout(); out = ROOT / "lc0-probe/hybrid_penalty_sweep.png"; fig.savefig(out, facecolor=BG); print("wrote", out, xs)
