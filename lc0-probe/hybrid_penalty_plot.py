"""Held-out L5 FVU of the deepest-first stack over the size penalty lambda (linear), full and without L5, with the
version without the prediction input and the gate as reference lines. -> lc0-probe/hybrid_penalty_sweep.png

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
xs = sorted(lam)
full = [lam[x]["full"]["L5 FVU"] for x in xs]
no5 = [lam[x]["without L5"]["L5 FVU"] for x in xs]

BLUE, ORANGE, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=BG); ax.set_facecolor(BG)
for y, lab in ((R["without prediction"]["full"]["L5 FVU"], "no prediction input, full"),
               (R["gate"]["full"]["L5 FVU"] if "gate" in R else R["pred_gate"]["full"]["L5 FVU"], "gate, full"),
               (R["without prediction"]["without L5"]["L5 FVU"], "no prediction input, without L5"),
               ((R.get("gate") or R["pred_gate"])["without L5"]["L5 FVU"], "gate, without L5")):
    ax.axhline(y, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
for ys, c, lab in ((full, BLUE, "full stack"), (no5, ORANGE, "without L5")):
    ax.plot(xs, ys, color=c, lw=2, zorder=3, label=lab)
    ax.scatter(xs, ys, s=36, color=c, edgecolor=BG, linewidth=2, zorder=4)
ax.set_yscale("log"); T = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5]; ax.set_yticks(T, [f"{t:g}" for t in T]); ax.minorticks_off(); ax.set_xlim(-0.08, max(xs) + 0.08)
ax.set_xlabel("size penalty λ", color=INK); ax.set_ylabel("held-out L5 FVU (log)", color=INK)
ax.grid(True, axis="y", which="major", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED)
gate, nop = (R.get("gate") or R["pred_gate"]), R["without prediction"]
ax.text(max(xs), max(gate["full"]["L5 FVU"], nop["full"]["L5 FVU"]) * 1.08, "gate / no prediction input: full",
        color=MUTED, ha="right", va="bottom", fontsize=9)
ax.text(max(xs), min(gate["without L5"]["L5 FVU"], nop["without L5"]["L5 FVU"]) * 0.92, "gate / no prediction input: without L5",
        color=MUTED, ha="right", va="top", fontsize=9)
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
ax.set_title("Deepest-first stack (L5 + L2 + L0) with the prediction input", color=INK, loc="left", fontsize=12)
fig.tight_layout(); out = ROOT / "lc0-probe/hybrid_penalty_sweep.png"; fig.savefig(out, facecolor=BG); print("wrote", out, xs)
