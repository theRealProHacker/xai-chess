"""Charts for the r5_tools dev run, from the call log and JUDGE4 scores.

    python charts_r5.py   -> tool_requested_used.png, tool_faithful_side_by_side.png,
                             tool_cooccurrence_rownorm.png, faithful_by_calls.png
"""
import collections, json, statistics
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "r5_tools"
BG, BLUE, LIGHT, ORANGE, GRAY, INK = "#fcfcfb", "#2a78d6", "#9ec5f4", "#eb6834", "#c9c8c3", "#52514e"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"runs/{RUN}/dev.jsonl")}
r5 = {json.loads(l)["id"]: json.loads(l) for l in open(f"runs/{RUN}/judge_dev/scores.jsonl")}
r4 = {json.loads(l)["id"]: json.loads(l) for l in open("runs/r4_best_lessons_think/judge_dev/coded.jsonl")}
ids = sorted(r5)
TOOLS = ["inventory", "attack_map", "hanging", "threats", "forcing", "legal", "after", "pawn_structure"]
WISH = {"attack_map": ("per-piece attack",), "inventory": ("piece inventory",), "hanging": ("hanging",),
        "forcing": ("short forcing",), "pawn_structure": ("pawn-structure", "file status"), "legal": ("legal-move",)}


def called(i, t):
    return any(c["name"] == t for c in rows[i]["calls"])


def wished(i, t):
    return any(c.startswith(k) for c in r4[i]["tool_cats"] for k in WISH.get(t, ()))


def style(ax):
    ax.set_facecolor(BG); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeede9"); ax.set_axisbelow(True)


# 1. requested x called x useful
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)
tl = ["inventory", "attack_map", "hanging", "forcing", "pawn_structure", "legal"]
x = list(range(len(tl)))
tab = {}
for t in tl:
    c = collections.Counter()
    for i in ids:
        w, u, ok = wished(i, t), called(i, t), r5[i]["faithful"] >= 4
        c[(w, u)] += 1
        if u:
            c[("useful", w, ok)] += 1
    tab[t] = c
bottom = [0] * len(tl)
for lab, key, col in (("requested, called", (True, True), BLUE), ("requested, not called", (True, False), LIGHT),
                      ("not requested, called", (False, True), ORANGE), ("not requested, not called", (False, False), GRAY)):
    v = [tab[t][key] for t in tl]
    a.bar(x, v, bottom=bottom, color=col, width=0.6, label=lab, edgecolor=BG, linewidth=2)
    bottom = [p + q for p, q in zip(bottom, v)]
a.set_xticks(x); a.set_xticklabels(tl, rotation=15); a.set_ylabel("dev items (n=300)")
a.set_title("Requested by the r4 judge  x  called by the model (log)", fontsize=10, loc="left"); a.legend(frameon=False, fontsize=8)
w = 0.35
for off, wq, col, lab in ((-w / 2, True, BLUE, "requested"), (w / 2, False, ORANGE, "not requested")):
    used = [tab[t][(wq, True)] for t in tl]; good = [tab[t][("useful", wq, True)] for t in tl]
    bars = b.bar([i + off for i in x], [g / u if u else 0 for g, u in zip(good, used)], width=w, color=col, label=lab)
    for bar, g, u in zip(bars, good, used):
        b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{g}/{u}", ha="center", fontsize=7, color=INK)
b.set_ylim(0, 1.12); b.set_xticks(x); b.set_xticklabels(tl, rotation=15)
b.set_ylabel("share of items with faithful >= 4"); b.set_title("Useful when called (output faithful)", fontsize=10, loc="left"); b.legend(frameon=False, fontsize=8)
for ax in (a, b):
    style(ax)
fig.tight_layout(); fig.savefig("tool_requested_used.png", dpi=150)

# 2. faithful by tool requested / by tool called, with vs without tools
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=BG, sharey=True)
req = [(t, [i for i in ids if wished(i, t)]) for t in tl]
use = [(t, [i for i in ids if called(i, t)]) for t in TOOLS]
for ax, groups, title in ((axes[0], req, "By tool requested"), (axes[1], use, "By tool called")):
    names = [f"{t}\n(n={len(g)})" for t, g in groups]; xx = list(range(len(groups))); w = 0.36
    b4 = ax.bar([i - w / 2 for i in xx], [statistics.mean(r4[i]["faithful"] for i in g) for _, g in groups], w, color=LIGHT, label="without tools (r4)")
    b5 = ax.bar([i + w / 2 for i in xx], [statistics.mean(r5[i]["faithful"] for i in g) for _, g in groups], w, color=BLUE, label="with tools (r5)")
    for bars in (b4, b5):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"{bar.get_height():.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xx); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(1, 5); ax.set_yticks([1, 2, 3, 4, 5]); ax.set_title(title, fontsize=11, loc="left"); style(ax)
axes[0].set_ylabel("mean faithful (1-5)"); axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig("tool_faithful_side_by_side.png", dpi=150)

# 3. co-use heat map, row-normalised, with margins
n = len(TOOLS)
M = [[sum(1 for i in ids if called(i, TOOLS[a_]) and called(i, TOOLS[b_])) for b_ in range(n)] for a_ in range(n)]
rowcnt = [sum(M[a_][c] for c in range(n) if c != a_) for a_ in range(n)]
P = [[(M[a_][b_] / rowcnt[a_]) if a_ != b_ else float("nan") for b_ in range(n)] for a_ in range(n)]
colsum = [sum(P[a_][b_] for a_ in range(n) if a_ != b_) for b_ in range(n)]
fig, ax = plt.subplots(figsize=(8.6, 7), facecolor=BG)
cm = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#86b6ef", "#256abf", "#0d366b"]); cm.set_bad(BG)
grid = np.full((n + 1, n + 1), np.nan); grid[:n, :n] = np.array(P)
ax.imshow(grid, cmap=cm, vmin=0, vmax=0.4)
ax.set_xticks(range(n + 1)); ax.set_yticks(range(n + 1))
ax.set_xticklabels(TOOLS + ["sum"], rotation=30, ha="right"); ax.set_yticklabels([f"{t} (n={M[i][i]})" for i, t in enumerate(TOOLS)] + ["sum"])
for a_ in range(n):
    for b_ in range(n):
        if a_ != b_:
            ax.text(b_, a_, f"{P[a_][b_]:.0%}", ha="center", va="center", fontsize=8, color="#ffffff" if P[a_][b_] > 0.22 else "#0b0b0b")
    ax.text(n, a_, f"100%\n({rowcnt[a_]})", ha="center", va="center", fontsize=8, color=INK)
for b_ in range(n):
    ax.text(b_, n, f"{colsum[b_]:.0%}", ha="center", va="center", fontsize=8, color=INK)
ax.text(n, n, f"{sum(colsum):.0%}\n({sum(rowcnt)})", ha="center", va="center", fontsize=8, color=INK)
ax.plot([-0.5, n - 0.5], [n - 0.5, n - 0.5], color=GRAY, lw=1); ax.plot([n - 0.5, n - 0.5], [-0.5, n - 0.5], color=GRAY, lw=1)
ax.set_title("Tools called together (rows sum to 100%); margins sum the percentages, counts in brackets", fontsize=9, loc="left")
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
fig.tight_layout(); fig.savefig("tool_cooccurrence_rownorm.png", dpi=150)

# 4. faithful / overall / contradiction rate by number of calls
by = collections.defaultdict(list)
for i in ids:
    by[min(len(rows[i]["calls"]), 9)].append(i)
ks = sorted(k for k in by if len(by[k]) >= 5)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)
w = 0.36; xx = list(range(len(ks)))
a.bar([i - w / 2 for i in xx], [statistics.mean(r5[i]["faithful"] for i in by[k]) for k in ks], w, color=BLUE, label="faithful")
a.bar([i + w / 2 for i in xx], [statistics.mean(r5[i]["overall"] for i in by[k]) for k in ks], w, color=ORANGE, label="overall")
a.set_xticks(xx); a.set_xticklabels([f"{k}{'+' if k == 9 else ''}\n(n={len(by[k])})" for k in ks]); a.set_ylim(1, 5); a.set_yticks([1, 2, 3, 4, 5])
a.set_xlabel("tool calls per item"); a.set_ylabel("mean score (1-5)"); a.set_title("Scores by number of calls", fontsize=10, loc="left"); a.legend(frameon=False, fontsize=8)
contra = [sum(1 for i in by[k] if r5[i].get("calls_contradicted", "").strip().lower() not in ("", "none")) / len(by[k]) for k in ks]
bars = b.bar(xx, contra, 0.6, color=BLUE)
for bar, k in zip(bars, ks):
    b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.0%}", ha="center", fontsize=8, color=INK)
b.set_xticks(xx); b.set_xticklabels([f"{k}{'+' if k == 9 else ''}" for k in ks]); b.set_ylim(0, 0.6)
b.set_xlabel("tool calls per item"); b.set_ylabel("share of items"); b.set_title("Output contradicts a fetched result", fontsize=10, loc="left")
for ax in (a, b):
    style(ax)
fig.tight_layout(); fig.savefig("faithful_by_calls.png", dpi=150)
print("ok")
