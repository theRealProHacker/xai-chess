"""Per fact section: intended (r4 judge asked for it) x used (r5 judge marked facts_used) x useful (r5 faithful >= 4).

    python tool_use_chart.py  -> tool_use.png + table on stdout
"""
import json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

r5 = {json.loads(l)["id"]: json.loads(l) for l in open("runs/r5b_check_voice/judge_dev/scores.jsonl")}
r4 = {json.loads(l)["id"]: json.loads(l) for l in open("runs/r4_best_lessons_think/judge_dev/coded.jsonl")}
SEC = {"moved": "attack map", "inventory": "inventory", "hanging": "hanging scan", "forcing": "forcing list", "pawns": "pawn structure"}
WISH = {"per-piece attack": "moved", "piece inventory": "inventory", "hanging": "hanging", "short forcing": "forcing",
        "pawn-structure": "pawns", "file status": "pawns"}


def intended(i, sec):
    return any(c.startswith(k) and s == sec for c in r4[i]["tool_cats"] for k, s in WISH.items())


tab = {}
for sec in SEC:
    c = collections.Counter()
    for i in r5:
        it, us, ok = intended(i, sec), sec in r5[i].get("facts_used", []), r5[i]["faithful"] >= 4
        c[(it, us)] += 1
        if us:
            c[("useful", it, ok)] += 1
    tab[sec] = c
print(f"{'tool':15s} {'int+used':>9s} {'int+unused':>11s} {'unint+used':>11s} {'unint+unused':>13s} | useful when used: intended / not")
for sec, c in tab.items():
    iu, inu, nu, nnu = c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]
    ui, nn = c[("useful", True, True)], c[("useful", False, True)]
    print(f"{SEC[sec]:15s} {iu:9d} {inu:11d} {nu:11d} {nnu:13d} | {ui}/{iu} ({ui/max(iu,1):.0%})  {nn}/{nu} ({nn/max(nu,1):.0%})")

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c9c8c3"
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
names, x = [SEC[s] for s in SEC], list(range(len(SEC)))
bottom = [0] * len(names)
for lab, key, col in (("intended, used", (True, True), BLUE), ("intended, not used", (True, False), "#9ec5f4"),
                      ("not intended, used", (False, True), ORANGE), ("not intended, not used", (False, False), GRAY)):
    v = [tab[s][key] for s in SEC]
    a.bar(x, v, bottom=bottom, color=col, width=0.6, label=lab, edgecolor="#fcfcfb", linewidth=2)
    bottom = [b0 + v0 for b0, v0 in zip(bottom, v)]
a.set_xticks(x); a.set_xticklabels(names, rotation=15); a.set_ylabel("dev items (n=300)")
a.set_title("Intended by the r4 judge  x  used by the model in r5", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8)
w = 0.35
for off, it, col, lab in ((-w / 2, True, BLUE, "intended"), (w / 2, False, ORANGE, "not intended")):
    used = [tab[s][(it, True)] for s in SEC]
    good = [tab[s][("useful", it, True)] for s in SEC]
    bars = b.bar([i + off for i in x], [g / u if u else 0 for g, u in zip(good, used)], width=w, color=col, label=lab)
    for bar, g, u in zip(bars, good, used):
        b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{g}/{u}", ha="center", fontsize=7, color="#52514e")
b.set_ylim(0, 1.12); b.set_xticks(x); b.set_xticklabels(names, rotation=15)
b.set_ylabel("share of used items with faithful >= 4"); b.set_title("Useful when used (output faithful)", fontsize=10, loc="left")
b.legend(frameon=False, fontsize=8)
for ax in (a, b):
    ax.set_facecolor("#fcfcfb"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeede9"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("tool_use.png", dpi=150)
