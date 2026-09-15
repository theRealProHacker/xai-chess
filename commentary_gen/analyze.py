"""Join judge scores with generations: fail-tag counts, per-source means, worst/best items.

    python analyze.py c0_baseline [--split dev] [--worst 12] [--best 6] [--out file.md]
"""
import argparse, collections, json, statistics
from pathlib import Path

from program import RUNS, by_id, fmt_input


def load(name, split):
    pool = by_id("train", "dev")
    gens = {json.loads(l)["id"]: json.loads(l) for l in open(RUNS / name / f"{split}.jsonl", encoding="utf-8")}
    scores = [json.loads(l) for l in open(RUNS / name / f"judge_{split}" / "scores.jsonl")]
    return [(s, gens[s["id"]], pool[s["id"]]) for s in scores if s["id"] in gens]


def report(name, split="dev", worst=12, best=6, with_input=False):
    rows = load(name, split)
    keys = ("faithful", "relevant", "human", "overall")
    out = [f"# {name} / {split}  n={len(rows)}", ""]
    out.append("  ".join(f"{k} {statistics.mean(s[k] for s, _, _ in rows):.2f}" for k in keys))
    src = collections.defaultdict(list)
    for s, _, e in rows:
        src[e["source"]].append(s["overall"])
    out.append("per source overall: " + "  ".join(f"{k} {statistics.mean(v):.2f} (n={len(v)})" for k, v in sorted(src.items())))
    fails = collections.Counter(s.get("fail") for s, _, _ in rows if s["overall"] <= 3 and s.get("fail"))
    out.append("fail tags: " + ", ".join(f"{k} {n}" for k, n in fails.most_common()))
    out.append(f"overall distribution: {sorted(collections.Counter(s['overall'] for s, _, _ in rows).items())}")
    rows.sort(key=lambda r: (r[0]["overall"], r[0]["faithful"]))
    for title, sel in (("WORST", rows[:worst]), ("BEST", rows[-best:][::-1])):
        out += ["", f"## {title}"]
        for s, g, e in sel:
            out += ["", f"### {e['id']}  [{e['source']}]  F{s['faithful']} R{s['relevant']} H{s['human']} O{s['overall']}  {s.get('fail', '')}"]
            if with_input:
                out += [fmt_input(e), ""]
            else:
                out += [f"move: {e['move']} ({e['side']}), ply {e['ply']}"]
            out += ["REFERENCE: " + e["comment"], "", "GENERATED: " + g["gen"]]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--worst", type=int, default=12)
    ap.add_argument("--best", type=int, default=6)
    ap.add_argument("--input", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    text = report(a.name, a.split, a.worst, a.best, a.input)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text if not a.out else text.splitlines()[2])
