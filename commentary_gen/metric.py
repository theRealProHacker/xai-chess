"""Cheap local signals (chrF, length ratio) plus judge-batch packing/unpacking.

The real metric is a Sonnet judge (run as a subagent, not via API): judge_pack() writes
batches of (input, reference, generation) for it to score; judge_unpack() merges its JSON back.
Judge rubric, 1-5 each: FAITHFUL (claims about the position/moves are true), RELEVANT (talks about
the move just played and what matters now), HUMAN (reads like the corpus, not like an engine or
a textbook), plus OVERALL. Score = mean OVERALL; chrF is reported alongside.
"""
import argparse, collections, json, math, statistics
from pathlib import Path

from program import RUNS, by_id, fmt_input


def _ngrams(s, n):
    return collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp, ref, n=6, beta=2):
    hyp, ref = " ".join(hyp.split()), " ".join(ref.split())
    p = r = 0
    for k in range(1, n + 1):
        h, g = _ngrams(hyp, k), _ngrams(ref, k)
        ov = sum((h & g).values())
        p += ov / max(sum(h.values()), 1)
        r += ov / max(sum(g.values()), 1)
    p, r = p / n, r / n
    return 0 if p + r == 0 else (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def local(name, split="dev"):
    pool = by_id("train", "dev")
    rows = [json.loads(l) for l in open(RUNS / name / f"{split}.jsonl", encoding="utf-8")]
    c = [chrf(r["gen"], pool[r["id"]]["comment"]) for r in rows]
    lr = [len(r["gen"]) / max(len(pool[r["id"]]["comment"]), 1) for r in rows]
    return {"n": len(rows), "chrf": round(statistics.mean(c), 4), "len_ratio": round(statistics.median(lr), 2),
            "empty": sum(1 for r in rows if not r["gen"].strip())}


def judge_pack(name, split="dev", batch=25, ids=None):
    pool = by_id("train", "dev")
    rows = [json.loads(l) for l in open(RUNS / name / f"{split}.jsonl", encoding="utf-8")]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    out = RUNS / name / f"judge_{split}"
    out.mkdir(exist_ok=True)
    scored = set()
    if (out / "scores.jsonl").exists():
        scored = {json.loads(l)["id"] for l in open(out / "scores.jsonl")}
    rows = [r for r in rows if r["id"] not in scored]
    tools = bool(json.load(open(RUNS / name / "candidate.json")).get("tools"))
    paths = []
    for b in range(0, len(rows), batch):
        p = out / f"batch_{b // batch:03d}.md"
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows[b:b + batch]:
                e = pool[r["id"]]
                fh.write(f"### {r['id']}\n{fmt_input(e, tools)}\n\nREFERENCE (human):\n{e['comment']}\n\n")
                if r.get("thoughts"):
                    fh.write(f"MODEL THINKING (summary returned by the API, not shown to users):\n{r['thoughts']}\n\n")
                fh.write(f"GENERATED:\n{r['gen']}\n\n")
        paths.append(p)
    return paths


def judge_unpack(name, split="dev"):
    out = RUNS / name / f"judge_{split}"
    seen = {}
    for p in sorted(out.glob("batch_*.scores.json")):
        for r in json.load(open(p)):
            seen[r["id"]] = r
    with open(out / "scores.jsonl", "w") as fh:
        for r in seen.values():
            fh.write(json.dumps(r) + "\n")
    keys = ("faithful", "relevant", "human", "overall")
    summ = {k: round(statistics.mean(r[k] for r in seen.values()), 3) for k in keys} if seen else {}
    summ["n"] = len(seen)
    if any("thinking_faithful" in r for r in seen.values()):
        summ["thinking_faithful"] = round(statistics.mean(r["thinking_faithful"] for r in seen.values()
                                                          if "thinking_faithful" in r), 3)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--unpack", action="store_true")
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()
    if a.pack:
        for p in judge_pack(a.name, a.split, a.batch):
            print(p)
    if a.unpack:
        print(json.dumps(judge_unpack(a.name, a.split)))
    print(json.dumps(local(a.name, a.split)))


if __name__ == "__main__":
    main()
