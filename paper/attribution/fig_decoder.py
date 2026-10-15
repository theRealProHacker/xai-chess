"""Fig. 2: per-tap ridge decoder fits from lc0-probe/results_square_decoder.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "lc0-probe" / "results_square_decoder.json"

d = json.loads(RES.read_text())
taps = ["/attn_body/rehape"] + [f"/encoder{i}/ln1" for i in range(10)]
labels = ["e"] + [f"L{i}" for i in range(10)]
get = lambda row: [d["taps"][t][row]["test"] for t in taps]
sq, fo, fs = get("square only"), get("features only"), get("features + square")
pr2 = [(a - b) / a for a, b in zip(sq, fs)]

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
x = range(len(taps))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)

for y, c, m, name in [(sq, BLUE, "o", "square only"), (fo, ORANGE, "s", "features only"),
                      (fs, AQUA, "^", "features + square")]:
    a1.plot(x, y, color=c, marker=m, ms=5, lw=2, label=name)
a1.set_ylabel("test FVU")
a1.set_ylim(0, 1)
a1.set_title("(a) Fraction of token variance unexplained", fontsize=9, loc="left", color=INK)
a1.legend(frameon=False, fontsize=8, loc="lower right")

a2.plot(x, pr2, color=BLUE, marker="o", ms=5, lw=2)
for i in (0, 3, 10):  # e, L2 (the post-embedding peak), L9
    a2.annotate(f"{pr2[i]:.2f}", (i, pr2[i]), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8, color=INK)
a2.set_ylabel("partial $R^2$")
a2.set_ylim(0, 1.1)
a2.set_title("(b) Non-square variance explained by features", fontsize=9, loc="left", color=INK)

for a in (a1, a2):
    a.set_xticks(list(x), labels)
    a.set_xlabel("tap")
    a.grid(axis="y", color="#e4e3dc", lw=0.6)
    a.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(HERE / "fig_decoder.png", dpi=200)
print("partial R2:", [round(v, 2) for v in pr2])
