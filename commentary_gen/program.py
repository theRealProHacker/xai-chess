"""The 'DSPy program': one predictor  (moves, board) -> comment.

A candidate is {"name", "instruction", "demos": [example ids]} plus optional "context_file",
"thinking", "max_tokens", and "functions": true to expose tools.ToolBox to the model via function
calling (calls are logged per row). render() builds the prompt, run() executes it over a set of
examples with a thread pool and writes runs/<name>/<split>.jsonl.
"""
import argparse, concurrent.futures as cf, json, sys, threading, time
from pathlib import Path

HERE = Path(__file__).parent
DATA, RUNS = HERE / "data", HERE / "runs"


def load(split):
    return [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]


def by_id(*splits):
    return {e["id"]: e for s in splits for e in load(s)}


def fmt_input(e, with_facts=False, last_moves=None, board=True):
    moves = e["moves"]
    if last_moves:  # keep the last N move numbers; long games cost a 2048-token model its whole budget
        toks = moves.split()
        nums = [i for i, t in enumerate(toks) if t[0].isdigit()]
        moves = " ".join(toks[nums[-last_moves]:]) if len(nums) > last_moves else moves
        moves = ("... " if moves != e["moves"] else "") + moves
    s = f"Moves so far (the last move is the one to comment on):\n{moves}"
    if board:
        s += (f"\n\nBoard after {e['move']} ({e['side']} just moved; uppercase = White, lowercase = Black, "
              f"rank 8 at the top):\n{e['board']}")
    if with_facts:
        import facts
        s += "\n\n=== FACTS (computed from this position; exact) ===\n" + facts.for_example(e, with_facts if isinstance(with_facts, list) else None)[0]
    return s


def render_pgn(cand, e):
    """ChessGPT's own commentary format (annotated PGN, arXiv 2306.09200 §3.3): headers, the moves, an
    open brace after the last one. Facts, if any, go in a leading game comment."""
    s = f'[White "{e["white"]}"]\n[Black "{e["black"]}"]\n\n'
    if cand.get("facts"):
        import facts
        s += "{ " + facts.for_example(e, cand["facts"])[0].replace("\n", " ").replace("}", ")") + " } "
    moves = _annotated_moves(e, cand["history"]) if cand.get("history") else e["moves"]
    return s + moves + " {"


@__import__("functools").lru_cache(maxsize=None)
def _game(source, file, index):
    import sys
    sys.path.insert(0, str(HERE.parent / "cleaned_commentary"))
    import filter_games as fg
    path = next(p for p in fg.pgn_files(source) if p.name == file)
    with fg.open_pgn(path) as fh:
        for i, g in enumerate(fg.read_games(fh)):
            if i == index:
                return g, fg


def _annotated_moves(e, budget):
    """The movetext with the annotator's own earlier mainline comments put back (never the target's),
    newest first until `budget` characters of comment are spent."""
    game, fg = _game(e["source"], e["file"], e["game"])
    board, node, plies = game.board(), game, []
    while node.variations and len(plies) < e["ply"]:
        node = node.variations[0]
        plies.append([board.san(node.move), fg.prose(node.comment or "")])
        board.push(node.move)
    plies[-1][1] = ""
    for i in range(len(plies) - 1, -1, -1):  # keep the newest comments that fit
        budget -= len(plies[i][1])
        if budget < 0:
            plies[i][1] = ""
    out, after_comment = [], False
    for i, (san, text) in enumerate(plies):
        n = i // 2 + 1
        out.append(f"{n}. {san}" if i % 2 == 0 else (f"{n}... {san}" if after_comment else san))
        if text:
            out.append("{ " + text.replace("}", ")") + " }")
        after_comment = bool(text)
    return " ".join(out)


def render(cand, e, pool):
    if cand.get("format") == "pgn":
        return render_pgn(cand, e)
    parts = []
    if cand.get("context_file"):  # background reading, inserted verbatim before the instruction
        parts += [cand.get("context_intro", "The following chess lessons are background reading. Use their ideas "
                                            "and vocabulary where they apply to the position; do not quote or cite them."),
                  "", "=== BEGIN LESSONS ===", (HERE.parent / cand["context_file"]).read_text(encoding="utf-8").strip(),
                  "=== END LESSONS ===", ""]
    parts += [cand["instruction"].strip(), ""]
    if cand.get("worked_file"):  # one example of the whole procedure, run through
        parts += [(HERE.parent / cand["worked_file"]).read_text(encoding="utf-8").strip(), ""]
    for did in cand.get("demos", []):
        d = pool[did]
        parts += ["--- Example ---", fmt_input(d, cand.get("demo_facts")), "", f"Commentary:\n{d['comment']}", ""]
    parts += ["--- Your turn ---", fmt_input(e, cand.get("facts"), cand.get("last_moves"), cand.get("board", True)), "",
              cand.get("cue", "Commentary:")]
    return "\n".join(parts)


def split_note(text, marker):
    """Candidates that write an evaluation first: keep the part after the marker as the comment."""
    if not marker or marker not in text:
        return text.strip(), ""
    head, _, tail = text.rpartition(marker)
    return tail.strip(), head.strip()


def run(cand, examples, pool, out_path, workers=4, temperature=0.7):
    import llm
    done = {}
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
    todo = [e for e in examples if e["id"] not in done]
    lock = threading.Lock()
    t0 = time.time()

    def one(e):
        tb = None
        if cand.get("functions") or cand.get("code"):
            import tools
            fns = cand.get("functions")
            tb = tools.CodeToolBox(e["fen"]) if cand.get("code") else \
                tools.ToolBox(e["fen"], fns if isinstance(fns, list) else None)
        pre = cand.get("prefill", "").format(move=e["move"], side=e["side"])  # starts the model's own turn
        r = llm.generate(render(cand, e, pool), temperature=temperature, thinking=cand.get("thinking"),
                         max_tokens=cand.get("max_tokens", 600), tools=tb, max_rounds=cand.get("max_rounds", 10),
                         **({"prefill": pre} if pre else {}), **({"stop": cand["stop"]} if cand.get("stop") else {}),
                         **({"repetition_penalty": cand["repetition_penalty"]} if "repetition_penalty" in cand else {}))
        gen, evaluation = split_note(f"{pre} {r['text']}" if pre else r["text"], cand.get("split_on"))
        row = {"id": e["id"], "gen": gen, "eval": evaluation, "thoughts": r.get("thoughts", ""),
               "finish": r["finish"], "usage": r["usage"], "calls": r.get("calls", [])}
        with lock, open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 25 == 0:
                print(f"  {cand['name']}: {i}/{len(todo)}  {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="path to candidate json")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="file with one example id per line")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    cand = json.load(open(a.candidate))
    pool = by_id("train", "dev")
    exs = load(a.split)
    if a.ids:
        want = set(open(a.ids).read().split())
        exs = [e for e in exs if e["id"] in want]
    if a.limit:
        exs = exs[:a.limit]
    out = RUNS / cand["name"]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(cand, open(out / "candidate.json", "w"), indent=1)
    n = run(cand, exs, pool, out / f"{a.split}.jsonl", a.workers, a.temperature)
    print(f"{cand['name']}: generated {n} new, {len(exs)} total -> {out / (a.split + '.jsonl')}")


if __name__ == "__main__":
    main()
