"""Charts for the r6_algorithm train run: three callable tools and ten injected analyses.

Round 6 has two usage signals, and they are not the same measurement:
  called  — the call log, for the three argument-taking tools. Measurable on all 120 items.
  cited   — the [F..] ids the model quotes in STEP 1/2, mapped back through facts.py's index to
            the analysis that produced them. Only the 82 items that wrote an evaluation cite
            anything, so every injected number is out of 82, never 120.

The requested-x-called chart from charts_r5.py has no round-6 counterpart: its "requested" axis
is the r4 judge's tool_cats, which exist for dev items only, and round 6 was screened on train.

    python charts_r6.py   -> r6_tool_use.png, r6_tool_faithful_side_by_side.png,
                             r6_tool_cooccurrence_rownorm.png, r6_faithful_by_calls.png
"""
import collections, json, re, statistics, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import facts, program

RUN = "r6_algorithm"
BG, BLUE, LIGHT, ORANGE, GRAY, INK = "#fcfcfb", "#2a78d6", "#9ec5f4", "#eb6834", "#c9c8c3", "#52514e"
GREEN = "#2e8b6f"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"runs/{RUN}/train.jsonl")}
r6 = {json.loads(l)["id"]: json.loads(l) for l in open(f"runs/{RUN}/judge_train/scores.jsonl")}
r5 = {json.loads(l)["id"]: json.loads(l) for l in open("runs/r5_tools/judge_train/scores.jsonl")}
pool = program.by_id("train", "dev")
ids = sorted(r6)
shared = sorted(set(r6) & set(r5))
CALLED = ["attack_map", "after", "legal"]
INJECTED = ["hanging", "change", "piece_quality", "alternatives", "inventory", "pawn_structure",
            "forcing", "king_safety", "threats", "space", "status"]
ALL = CALLED + INJECTED
CITE = re.compile(r"F(\d+)")

# usage: called from the log, cited from the [F..] ids resolved through the fact index
cited = {i: set() for i in ids}
avail = collections.Counter()
for i in ids:
    _, idx = facts.for_example(pool[i])
    for v in idx.values():
        avail[v["src"]] += 1
    for num in set(CITE.findall(rows[i]["eval"])):
        v = idx.get(f"F{num}")
        if v:
            cited[i].add(v["src"])
EVAL = [i for i in ids if rows[i]["eval"].strip()]        # the 82 that can cite anything


def called(i, t):
    return any(c["name"] == t for c in rows[i]["calls"])


def used(i, t):
    return called(i, t) if t in CALLED else t in cited[i]


def style(ax):
    ax.set_facecolor(BG); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeede9"); ax.set_axisbelow(True)


# 1. how much each analysis is used, and what it costs in facts
fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=BG)
share = [(t, sum(1 for i in (ids if t in CALLED else EVAL) if used(i, t)) / len(ids if t in CALLED else EVAL))
         for t in ALL]
share.sort(key=lambda kv: -kv[1])
xx = list(range(len(share)))
cols = [ORANGE if t in CALLED else BLUE for t, _ in share]
bars = a.bar(xx, [v for _, v in share], 0.62, color=cols)
for bar, (t, v) in zip(bars, share):
    d = len(ids) if t in CALLED else len(EVAL)
    a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
           f"{round(v*d)}/{d}", ha="center", fontsize=7, color=INK)
a.set_xticks(xx); a.set_xticklabels([t for t, _ in share], rotation=35, ha="right", fontsize=8)
a.set_ylim(0, 1.0); a.set_ylabel("share of items that used it")
a.set_title("Used per item — called (orange, of 120) vs cited in STEP 1/2 (blue, of 82)",
            fontsize=9.5, loc="left")
inj = sorted(INJECTED, key=lambda t: -avail[t])
for t in inj:
    fx, cy = avail[t] / len(ids), sum(1 for i in EVAL if t in cited[i]) / len(EVAL)
    b.scatter([fx], [cy], s=70, color=BLUE, zorder=3)
    b.annotate(t, (fx, cy), textcoords="offset points", xytext=(7, 4), fontsize=8, color=INK)
b.set_xscale("log")
b.set_xticks([1, 2, 5, 10]); b.set_xticklabels(["1", "2", "5", "10"])
b.set_xlim(0.8, 22); b.set_ylim(0, 1.0)
b.set_xlabel("facts injected per item (log)"); b.set_ylabel("share of the 82 citing it")
b.set_title("Cost in facts vs how often it is cited — up and left is efficient",
            fontsize=9.5, loc="left")
for ax in (a, b):
    style(ax)
b.grid(axis="x", color="#eeede9")
fig.tight_layout(); fig.savefig("r6_tool_use.png", dpi=150)

# 2. faithful when the analysis was used vs not, within the 82 that wrote an evaluation
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=BG, sharey=True,
                           gridspec_kw={"width_ratios": [2.6, 10]})
w = 0.38
for ax, group, title in ((a, CALLED, "Three callable tools"), (b, INJECTED, "Ten injected analyses")):
    g = [t for t in group if 5 <= sum(1 for i in EVAL if used(i, t)) <= len(EVAL) - 5]
    xg = list(range(len(g)))
    yes = [statistics.mean(r6[i]["faithful"] for i in EVAL if used(i, t)) for t in g]
    no = [statistics.mean(r6[i]["faithful"] for i in EVAL if not used(i, t)) for t in g]
    ax.bar([i - w / 2 for i in xg], no, w, color=LIGHT, label="not used")
    ax.bar([i + w / 2 for i in xg], yes, w, color=BLUE, label="used")
    for i, (n_, y_) in enumerate(zip(no, yes)):
        ax.text(i - w / 2, n_ + 0.04, f"{n_:.2f}", ha="center", fontsize=7, color=INK)
        ax.text(i + w / 2, y_ + 0.04, f"{y_:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(xg)
    ax.set_xticklabels([f"{t}\n(n={sum(1 for i in EVAL if used(i, t))})" for t in g],
                       fontsize=7.5, rotation=35, ha="right")
    ax.set_ylim(1, 5.45); ax.set_yticks([1, 2, 3, 4, 5]); ax.set_title(title, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2); style(ax)
a.set_ylabel("mean faithful (1-5)")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.text(0.008, 0.015, "All bars are the 82 items that wrote an evaluation. A group is dropped when "
                       "either side holds fewer than 5 items, which is why legal and status are absent.",
         fontsize=8, color=INK)
fig.savefig("r6_tool_faithful_side_by_side.png", dpi=150)

# 3. co-use heat map over all thirteen, row-normalised, within the 82
T = [t for t in ALL if sum(1 for i in EVAL if used(i, t)) >= 5]
n = len(T)
M = [[sum(1 for i in EVAL if used(i, T[p]) and used(i, T[q])) for q in range(n)] for p in range(n)]
rowcnt = [sum(M[p][c] for c in range(n) if c != p) for p in range(n)]
P = [[(M[p][q] / rowcnt[p]) if p != q and rowcnt[p] else float("nan") for q in range(n)] for p in range(n)]
colsum = [sum(P[p][q] for p in range(n) if p != q and P[p][q] == P[p][q]) for q in range(n)]
fig, ax = plt.subplots(figsize=(11, 9.2), facecolor=BG)
cm = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#86b6ef", "#256abf", "#0d366b"]); cm.set_bad(BG)
grid = np.full((n + 1, n + 1), np.nan); grid[:n, :n] = np.array(P)
ax.imshow(grid, cmap=cm, vmin=0, vmax=0.25)
lab = [f"{t}*" if t in CALLED else t for t in T]
ax.set_xticks(range(n + 1)); ax.set_yticks(range(n + 1))
ax.set_xticklabels(lab + ["sum"], rotation=35, ha="right", fontsize=8)
ax.set_yticklabels([f"{l} (n={M[i][i]})" for i, l in enumerate(lab)] + ["sum"], fontsize=8)
for p in range(n):
    for q in range(n):
        if p != q:
            ax.text(q, p, f"{P[p][q]:.0%}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if P[p][q] > 0.14 else "#0b0b0b")
    ax.text(n, p, f"{rowcnt[p]/sum(rowcnt):.0%}", ha="center", va="center", fontsize=7, color=INK)
for q in range(n):
    ax.text(q, n, f"{colsum[q]:.0%}", ha="center", va="center", fontsize=7, color=INK)
ax.text(n, n, "100%", ha="center", va="center", fontsize=7, color=INK)
ax.plot([-0.5, n - 0.5], [n - 0.5, n - 0.5], color=GRAY, lw=1); ax.plot([n - 0.5, n - 0.5], [-0.5, n - 0.5], color=GRAY, lw=1)
ax.set_title("Used together in the same item, rows sum to 100%  (* = a called tool, the rest cited)\n"
             "the 82 items with an evaluation; right margin = share of all co-uses, bottom = column sums",
             fontsize=9, loc="left")
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
fig.tight_layout(); fig.savefig("r6_tool_cooccurrence_rownorm.png", dpi=150)

# 4. scores and contradiction by number of calls (the call log only)
by = collections.defaultdict(list)
for i in ids:
    by[min(len(rows[i]["calls"]), 9)].append(i)
ks = sorted(k for k in by if len(by[k]) >= 5)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)
w = 0.36; xx = list(range(len(ks)))
a.bar([i - w / 2 for i in xx], [statistics.mean(r6[i]["faithful"] for i in by[k]) for k in ks], w, color=BLUE, label="faithful")
a.bar([i + w / 2 for i in xx], [statistics.mean(r6[i]["overall"] for i in by[k]) for k in ks], w, color=ORANGE, label="overall")
a.set_xticks(xx); a.set_xticklabels([f"{k}{'+' if k == 9 else ''}\n(n={len(by[k])})" for k in ks])
a.set_ylim(1, 5); a.set_yticks([1, 2, 3, 4, 5])
a.set_xlabel("tool calls per item"); a.set_ylabel("mean score (1-5)")
a.set_title("Scores by number of calls", fontsize=10, loc="left"); a.legend(frameon=False, fontsize=8)
contra = [sum(1 for i in by[k] if str(r6[i].get("calls_contradicted", "")).strip().lower() not in ("", "none")) / len(by[k]) for k in ks]
bars = b.bar(xx, contra, 0.6, color=BLUE)
for bar in bars:
    b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.0%}", ha="center", fontsize=8, color=INK)
b.set_xticks(xx); b.set_xticklabels([f"{k}{'+' if k == 9 else ''}" for k in ks]); b.set_ylim(0, 0.6)
b.set_yticks([0, .1, .2, .3, .4, .5, .6]); b.set_yticklabels([f"{v:.0%}" for v in (0, .1, .2, .3, .4, .5, .6)])
b.set_xlabel("tool calls per item"); b.set_ylabel("share of items")
b.set_title("Output contradicts a fetched result", fontsize=10, loc="left")
for ax in (a, b):
    style(ax)
fig.tight_layout(); fig.savefig("r6_faithful_by_calls.png", dpi=150)
print("ok")
