"""Tool use from the call log: which tools were called, how often, and (with judge scores) which
calls the judge saw used or contradicted.

    python tool_usage.py r5_tools [--split train] [--r4 r4_best_lessons_think]
"""
import argparse, collections, json, statistics
from pathlib import Path

from program import RUNS

TOOLS = ("inventory", "attack_map", "hanging", "threats", "forcing", "legal", "after", "pawn_structure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--r4", default="r4_best_lessons_think")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(RUNS / a.name / f"{a.split}.jsonl", encoding="utf-8")]
    n = len(rows)
    ncalls = [len(r.get("calls", [])) for r in rows]
    print(f"{a.name}/{a.split} n={n}: calls per item median {statistics.median(ncalls)}, max {max(ncalls)}, "
          f"none {sum(1 for c in ncalls if c == 0)}, MAX_ROUNDS {sum(1 for r in rows if r['finish'] == 'MAX_ROUNDS')}")
    items, total, errors = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        seen = set()
        for c in r.get("calls", []):
            total[c["name"]] += 1
            seen.add(c["name"])
            if c["result"].startswith("Error") or "not legal" in c["result"] or "is not a move" in c["result"]:
                errors[c["name"]] += 1
        items.update(seen)
    print(f"{'tool':15s} {'items':>6s} {'calls':>6s} {'errors':>7s}")
    for t in TOOLS:
        print(f"{t:15s} {items[t]:6d} {total[t]:6d} {errors[t]:7d}")
    fn_items, fn_total = collections.Counter(), collections.Counter()
    if any("functions" in c for r in rows for c in r.get("calls", [])):  # code sandbox: functions per cell
        for r in rows:
            seen = collections.Counter()
            for c in r.get("calls", []):
                seen.update(c.get("functions", {}))
            fn_total.update(seen)
            fn_items.update(seen.keys())
        print(f"\ncells: tracebacks {sum(1 for r in rows for c in r.get('calls', []) if 'Traceback' in c['result'])}")
        print(f"{'function':22s} {'items':>6s} {'calls':>6s}")
        for f, n in fn_total.most_common():
            print(f"{f:22s} {fn_items[f]:6d} {n:6d}")

    sp = RUNS / a.name / f"judge_{a.split}" / "scores.jsonl"
    if not sp.exists():
        return
    scores = {json.loads(l)["id"]: json.loads(l) for l in open(sp)}
    byid = {r["id"]: r for r in rows}
    used, contra = collections.Counter(), 0
    per_tool_used = collections.Counter()
    for i, s in scores.items():
        calls = byid[i].get("calls", [])
        for k in s.get("calls_used", []):
            if 1 <= k <= len(calls):
                per_tool_used[calls[k - 1]["name"]] += 1
        if s.get("calls_contradicted", "").strip():
            contra += 1
    print(f"\njudged: items with a used call {sum(1 for s in scores.values() if s.get('calls_used'))}/{len(scores)}, "
          f"items contradicting a call result {contra}")
    print(f"{'tool':15s} {'calls':>6s} {'judged used':>12s} {'share':>6s}")
    for t in TOOLS:
        print(f"{t:15s} {total[t]:6d} {per_tool_used[t]:12d} {per_tool_used[t]/max(total[t],1):6.0%}")
    faith = collections.defaultdict(list)
    for i, s in scores.items():
        faith[min(len(byid[i].get("calls", [])), 8)].append(s["faithful"])
    print("\nfaithful by number of calls:", {k: (len(v), round(statistics.mean(v), 2)) for k, v in sorted(faith.items())})

    r4p = RUNS / a.r4 / f"judge_{a.split}" / "coded.jsonl"
    if not r4p.exists():
        return
    r4 = {json.loads(l)["id"]: json.loads(l) for l in open(r4p)}
    section = {"per-piece attack": "attack_map", "piece inventory": "inventory", "hanging": "hanging",
               "short forcing": ("forcing", "threats", "after"), "pawn-structure": "pawn_structure", "file status": "pawn_structure",
               "legal-move": "legal"}
    print(f"\nr4 wish -> {a.name}: n, r4 faithful, r5 faithful, items that called the wished tool")
    for k, t in section.items():
        ids = [i for i, v in r4.items() if any(c.startswith(k) for c in v["tool_cats"]) and i in scores]
        if len(ids) < 5:
            continue
        ts = t if isinstance(t, tuple) else (t,)
        called = sum(1 for i in ids if any(c["name"] in ts for c in byid[i].get("calls", [])))
        print(f"{len(ids):4d}  {statistics.mean(r4[i]['faithful'] for i in ids):.2f} -> "
              f"{statistics.mean(scores[i]['faithful'] for i in ids):.2f}  called={called}  {k}")


if __name__ == "__main__":
    main()
