"""Did the model use the computed facts? Lexical check per item plus a join with the r4 judges' tool wishes.

    python tool_usage.py r5_recipe_tools [--split dev] [--r4 r4_best_lessons_think]

Per item: which fact sections the generation (and thinking) mention by name of square or move:
  hanging  a square the facts list as hanging appears in the text
  forcing  a listed check/capture (SAN) appears in the text
  pawns    a listed isolated/doubled/passed/backward square, or a file status, appears with its label
  moved    a square from the moved piece's attacks/attacked-by line appears in the text
If judge scores exist, prints the judges' facts_used counts and the r4 wish -> r5 faithful table.
"""
import argparse, collections, json, re, statistics
from pathlib import Path

import board_tools, chess
from program import RUNS, by_id, tool_block


def parse_facts(e):
    """Return the fact squares/moves per section, from the same tool_block the model saw."""
    b = chess.Board()
    for tok in e["moves"].split():
        if not tok[0].isdigit():
            b.push_san(tok)
    d = {}
    moved = b.piece_at(b.move_stack[-1].to_square)
    am = board_tools.attack_map_data(b, b.move_stack[-1].to_square) if moved else {}
    m = next(iter(am.values()), {})
    d["moved"] = set(m.get("attacks", []) + m.get("attacked_by", []))
    d["hanging"] = {r["square"] for r in board_tools.hanging_data(b)}
    d["forcing"] = {r["move"] for r in board_tools.forcing_data(b)}
    ps = board_tools.pawn_structure_data(b)
    d["pawns"] = {(lab, sq) for side in ("White", "Black") for lab, sqs in ps[side].items() for sq in sqs}
    d["files"] = {f for f, s in ps["files"].items() if s != "closed"}
    return d


def mentions(text, facts):
    t = text.replace("×", "x")
    sq = set(re.findall(r"\b([a-h][1-8])\b", t))
    san = set(re.findall(r"\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?)", t))
    san |= {s.rstrip("+#") for s in san}
    used = set()
    if sq & facts["moved"]:
        used.add("moved")
    if sq & facts["hanging"]:
        used.add("hanging")
    if san & facts["forcing"] or san & {s.rstrip("+#") for s in facts["forcing"]}:
        used.add("forcing")
    tl = t.lower()
    for lab, s in facts["pawns"]:
        if lab in tl and s in sq:
            used.add("pawns")
    if re.search(r"(open|half-open) [a-h]-?file", tl) or re.search(r"\b[a-h]-file", tl) and "open" in tl:
        used.add("pawns")
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--r4", default="r4_best_lessons_think")
    a = ap.parse_args()
    pool = by_id("train", "dev")
    rows = [json.loads(l) for l in open(RUNS / a.name / f"{a.split}.jsonl", encoding="utf-8")]
    gen_c, think_c, avail = collections.Counter(), collections.Counter(), collections.Counter()
    per = {}
    for r in rows:
        f = parse_facts(pool[r["id"]])
        for k in ("moved", "hanging", "forcing"):
            if f[k]:
                avail[k] += 1
        if f["pawns"] or f["files"]:
            avail["pawns"] += 1
        g, t = mentions(r["gen"], f), mentions(r.get("thoughts", ""), f)
        per[r["id"]] = {"gen": sorted(g), "think": sorted(t)}
        gen_c.update(g)
        think_c.update(t)
    n = len(rows)
    print(f"{a.name}/{a.split} n={n}: sections named in output / in thinking / available")
    for k in ("moved", "hanging", "forcing", "pawns"):
        print(f"  {k:8s} {gen_c[k]:4d} {think_c[k]:4d} {avail[k]:4d}")
    print(f"  any      {sum(1 for p in per.values() if p['gen']):4d} {sum(1 for p in per.values() if p['think']):4d}")
    with open(RUNS / a.name / f"tool_usage_{a.split}.json", "w") as fh:
        json.dump(per, fh, indent=0)

    sp = RUNS / a.name / f"judge_{a.split}" / "scores.jsonl"
    if not sp.exists():
        return
    scores = {json.loads(l)["id"]: json.loads(l) for l in open(sp)}
    ju = collections.Counter(x for s in scores.values() for x in s.get("facts_used", []))
    print("\njudge facts_used:", dict(ju.most_common()),
          f"| items with any: {sum(1 for s in scores.values() if s.get('facts_used'))}/{len(scores)}",
          f"| facts_contradicted: {sum(1 for s in scores.values() if s.get('facts_contradicted', '').strip())}")

    r4p = RUNS / a.r4 / f"judge_{a.split}" / "coded.jsonl"
    if not r4p.exists():
        return
    r4 = {json.loads(l)["id"]: json.loads(l) for l in open(r4p)}
    print(f"\nr4 judge wish -> {a.name}: n, r4 faithful, r5 faithful, r5 overall, r5 items naming that section")
    section = {"per-piece attack": "moved", "hanging": "hanging", "inventory": "moved", "forcing": "forcing",
               "pawn-structure": "pawns", "file status": "pawns"}
    for cat in sorted({c for v in r4.values() for c in v["tool_cats"]}):
        ids = [i for i, v in r4.items() if cat in v["tool_cats"] and i in scores]
        if len(ids) < 5:
            continue
        sec = next((s for k, s in section.items() if cat.startswith(k)), None)
        named = sum(1 for i in ids if sec and (sec in per[i]["gen"] or sec in per[i]["think"])) if sec else "-"
        print(f"{len(ids):4d}  {statistics.mean(r4[i]['faithful'] for i in ids):.2f} -> "
              f"{statistics.mean(scores[i]['faithful'] for i in ids):.2f}  o={statistics.mean(scores[i]['overall'] for i in ids):.2f}"
              f"  named={named}  {cat[:70]}")


if __name__ == "__main__":
    main()
