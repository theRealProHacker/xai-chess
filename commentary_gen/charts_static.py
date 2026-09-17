"""Charts for the reverted static-facts-block run (r5b_check_voice), for comparison with charts_r5.py.

The run is not in the repo; point STATIC at its directory (judge_dev/scores.jsonl with facts_used).
    python charts_static.py [path]  -> static_tool_use.png, static_tool_faithful_side_by_side.png,
                                       static_tool_cooccurrence_rownorm.png
"""
import collections, json, statistics, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATIC = sys.argv[1] if len(sys.argv) > 1 else "runs/r5b_check_voice"
BG, BLUE, LIGHT, ORANGE, GRAY, INK = "#fcfcfb", "#2a78d6", "#9ec5f4", "#eb6834", "#c9c8c3", "#52514e"
r5 = {json.loads(l)["id"]: json.loads(l) for l in open(f"{STATIC}/judge_dev/scores.jsonl")}
r4 = {json.loads(l)["id"]: json.loads(l) for l in open("runs/r4_best_lessons_think/judge_dev/coded.jsonl")}
ids = sorted(r5)
SEC = {"inventory": "inventory", "moved": "attack_map", "hanging": "hanging", "forcing": "forcing", "pawns": "pawn_structure"}
WISH = {"moved": ("per-piece attack",), "inventory": ("piece inventory",), "hanging": ("hanging",),
        "forcing": ("short forcing",), "pawns": ("pawn-structure", "file status")}  # same mapping as charts_r5.py


def used(i, s):
    return s in r5[i].get("facts_used", [])


def wished(i, s):
    return any(c.startswith(k) for c in r4[i]["tool_cats"] for k in WISH[s])


def style(ax):
    ax.set_facecolor(BG); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeede9"); ax.set_axisbelow(True)


secs = list(SEC); names = [SEC[s] for s in secs]; x = list(range(len(secs)))

# 1. intended x used x useful
tab = {}
for s in secs:
    c = collections.Counter()
    for i in ids:
        w, u, ok = wished(i, s), used(i, s), r5[i]["faithful"] >= 4
        c[(w, u)] += 1
        if u:
            c[("useful", w, ok)] += 1
    tab[s] = c
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)
bottom = [0] * len(secs)
for lab, key, col in (("intended, used", (True, True), BLUE), ("intended, not used", (True, False), LIGHT),
                      ("not intended, used", (False, True), ORANGE), ("not intended, not used", (False, False), GRAY)):
    v = [tab[s][key] for s in secs]
    a.bar(x, v, bottom=bottom, color=col, width=0.6, label=lab, edgecolor=BG, linewidth=2)
    bottom = [p + q for p, q in zip(bottom, v)]
a.set_xticks(x); a.set_xticklabels(names, rotation=15); a.set_ylabel("dev items (n=300)")
a.set_title("Intended by the r4 judge  x  used (static block, judge-marked)", fontsize=10, loc="left"); a.legend(frameon=False, fontsize=8)
w = 0.35
for off, wq, col, lab in ((-w / 2, True, BLUE, "intended"), (w / 2, False, ORANGE, "not intended")):
    u_ = [tab[s][(wq, True)] for s in secs]; g_ = [tab[s][("useful", wq, True)] for s in secs]
    bars = b.bar([i + off for i in x], [g / u if u else 0 for g, u in zip(g_, u_)], width=w, color=col, label=lab)
    for bar, g, u in zip(bars, g_, u_):
        b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{g}/{u}", ha="center", fontsize=7, color=INK)
b.set_ylim(0, 1.12); b.set_xticks(x); b.set_xticklabels(names, rotation=15)
b.set_ylabel("share of used items with faithful >= 4"); b.set_title("Useful when used (output faithful)", fontsize=10, loc="left"); b.legend(frameon=False, fontsize=8)
for ax in (a, b):
    style(ax)
fig.tight_layout(); fig.savefig("static_tool_use.png", dpi=150)

# 2. side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=BG, sharey=True)
req = [(SEC[s], [i for i in ids if wished(i, s)]) for s in secs]
use = [(SEC[s], [i for i in ids if used(i, s)]) for s in secs]
for ax, groups, title in ((axes[0], req, "By tool requested"), (axes[1], use, "By tool used (judge-marked)")):
    nm = [f"{t}\n(n={len(g)})" for t, g in groups]; xx = list(range(len(groups))); w = 0.36
    b4 = ax.bar([i - w / 2 for i in xx], [statistics.mean(r4[i]["faithful"] for i in g) for _, g in groups], w, color=LIGHT, label="without tools (r4)")
    b5 = ax.bar([i + w / 2 for i in xx], [statistics.mean(r5[i]["faithful"] for i in g) for _, g in groups], w, color=BLUE, label="with facts block")
    for bars in (b4, b5):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"{bar.get_height():.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xx); ax.set_xticklabels(nm, fontsize=8.5); ax.set_ylim(1, 5); ax.set_yticks([1, 2, 3, 4, 5]); ax.set_title(title, fontsize=11, loc="left"); style(ax)
axes[0].set_ylabel("mean faithful (1-5)"); axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig("static_tool_faithful_side_by_side.png", dpi=150)

# 3. co-use heat map, row-normalised, with margins
n = len(secs)
M = [[sum(1 for i in ids if used(i, secs[a_]) and used(i, secs[b_])) for b_ in range(n)] for a_ in range(n)]
rowcnt = [sum(M[a_][c] for c in range(n) if c != a_) for a_ in range(n)]
P = [[(M[a_][b_] / rowcnt[a_]) if a_ != b_ else float("nan") for b_ in range(n)] for a_ in range(n)]
colsum = [sum(P[a_][b_] for a_ in range(n) if a_ != b_) for b_ in range(n)]
fig, ax = plt.subplots(figsize=(7.4, 6), facecolor=BG)
cm = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#86b6ef", "#256abf", "#0d366b"]); cm.set_bad(BG)
grid = np.full((n + 1, n + 1), np.nan); grid[:n, :n] = np.array(P)
ax.imshow(grid, cmap=cm, vmin=0, vmax=0.6)
ax.set_xticks(range(n + 1)); ax.set_yticks(range(n + 1))
ax.set_xticklabels(names + ["sum"], rotation=30, ha="right"); ax.set_yticklabels([f"{SEC[s]} (n={M[i][i]})" for i, s in enumerate(secs)] + ["sum"])
for a_ in range(n):
    for b_ in range(n):
        if a_ != b_:
            ax.text(b_, a_, f"{P[a_][b_]:.0%}", ha="center", va="center", fontsize=9, color="#ffffff" if P[a_][b_] > 0.33 else "#0b0b0b")
    ax.text(n, a_, f"100%\n({rowcnt[a_]})", ha="center", va="center", fontsize=8.5, color=INK)
for b_ in range(n):
    ax.text(b_, n, f"{colsum[b_]:.0%}", ha="center", va="center", fontsize=9, color=INK)
ax.text(n, n, f"{sum(colsum):.0%}\n({sum(rowcnt)})", ha="center", va="center", fontsize=8.5, color=INK)
ax.plot([-0.5, n - 0.5], [n - 0.5, n - 0.5], color=GRAY, lw=1); ax.plot([n - 0.5, n - 0.5], [-0.5, n - 0.5], color=GRAY, lw=1)
ax.set_title("Co-use partners per row (static block); margins sum the percentages, counts in brackets", fontsize=8.5, loc="left")
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
fig.tight_layout(); fig.savefig("static_tool_cooccurrence_rownorm.png", dpi=150)
print("ok")
