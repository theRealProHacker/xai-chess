"""Round-6 dev score distribution: full-dev judges, with the 11 trade-fix items replaced by their
rerun and paired re-judgement (runs/r6_algorithm_tradefix/judge_pair)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RUN = HERE / "runs"


def merged():
    s = {json.loads(l)["id"]: json.loads(l) for l in open(RUN / "r6_algorithm/judge_dev/scores.jsonl")}
    pair = RUN / "r6_algorithm_tradefix/judge_pair"
    key = json.load(open(pair / "key.json"))
    for r in json.load(open(pair / "batch_000.scores.json")):
        i, arm = key[r["id"]]
        if arm == "new":
            s[i] = {**r, "id": i}
    return s


if __name__ == "__main__":
    s = merged()
    series = [("overall", "#2a78d6"), ("faithful", "#eb6834"), ("relevant", "#1baf7a"),
              ("human", "#eda100"), ("tool_use", "#e87ba4")]
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
    w, gap = 0.16, 0.012
    for k, (axis, col) in enumerate(series):
        counts = [sum(1 for r in s.values() if r[axis] == v) for v in range(1, 6)]
        xs = [v + (k - (len(series) - 1) / 2) * (w + gap) for v in range(1, 6)]
        bars = ax.bar(xs, counts, w, color=col, label=axis.replace("_", " "), zorder=2)
        for x, c in zip(xs, counts):
            ax.text(x, c + 2, str(c), ha="center", va="bottom", fontsize=7, color=muted)
    ax.set_xticks(range(1, 6)); ax.set_xlabel("judge score (1 = worst, 5 = best)", color=muted)
    ax.set_ylabel("items (of 300)", color=muted)
    ax.set_title("Round 6 on dev: items at each score", color=ink, loc="left", fontsize=12, pad=30)
    ax.grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2"); ax.tick_params(colors=muted, length=0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series), labelcolor=ink,
              borderaxespad=0.2, handlelength=1.2, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(HERE / "r6_dev_score_distribution.png", facecolor=surface)
    print({a: [sum(1 for r in s.values() if r[a] == v) for v in range(1, 6)] for a, _ in series})
