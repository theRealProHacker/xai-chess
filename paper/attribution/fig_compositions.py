"""Fig. 3: the four cross-layer compositions on the tap axis (schematic, no data)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 9})

cols = ["e"] + [f"L{i}" for i in range(10)] + ["y"]
X = {c: i for i, c in enumerate(cols)}
fig, ax = plt.subplots(figsize=(7.2, 2.9))

rows = [
    (3, "M6 sweep", BLUE, "independent decoder at every tap; no arrows between taps"),
    (2, "M8 cascade", ORANGE, "erase or substitute G at L; re-read probes / decoders at every L' > L"),
    (1, "M8∘M2 paths", AQUA, "substitute G at L; regress change at L' on D_L'; push each part to y"),
    (0, "M7 graph", YELLOW, "skip transcoders per block; linear edges at frozen attention and LN"),
]
for y, name, c, note in rows:
    ax.text(-0.6, y, name, ha="right", va="center", color=INK, fontsize=9)
    ax.text(5.5, y - 0.36, note, ha="center", va="center", color=MUTED, fontsize=7.5)

# M6: a dot at every tap, value read separately at each
for t in cols[:-1]:
    ax.plot(X[t], 3, "o", color=BLUE, ms=6)
# M8: intervene at L2, read at all later taps
ax.plot(X["L2"], 2, "D", color=ORANGE, ms=7)
for t in cols[4:-1]:
    ax.plot(X[t], 2, "o", color=ORANGE, ms=5, mfc="white", mew=1.5)
ax.annotate("", xy=(X["L9"], 2), xytext=(X["L2"] + 0.2, 2),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
# M8∘M2: L2 -> L5 -> y, attention live
ax.plot(X["L2"], 1, "D", color=AQUA, ms=7)
ax.plot(X["L5"], 1, "s", color=AQUA, ms=7)
ax.annotate("", xy=(X["L5"] - 0.15, 1), xytext=(X["L2"] + 0.2, 1),
            arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.5))
ax.annotate("", xy=(X["y"] - 0.15, 1), xytext=(X["L5"] + 0.2, 1),
            arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.5, ls="--"))
ax.text(X["L3"] + 0.5, 1.13, "attention live", ha="center", fontsize=7.5, color=INK)
# M7: chain of nodes across all blocks, frozen attention
for t in cols[1:-1]:
    ax.plot(X[t], 0, "s", color=YELLOW, ms=6)
ax.annotate("", xy=(X["y"] - 0.15, 0), xytext=(X["L0"], 0),
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.5))
ax.text(X["L4"] + 0.5, 0.13, "attention frozen", ha="center", fontsize=7.5, color=INK)

ax.plot(X["y"], 3, alpha=0)
ax.set_xticks(range(len(cols)), cols)
ax.set_yticks([])
ax.set_xlim(-0.5, len(cols) - 0.5)
ax.set_ylim(-0.6, 3.4)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlabel("tap (y = value or policy logit)")
fig.tight_layout()
fig.savefig(HERE / "fig_compositions.png", dpi=200)
