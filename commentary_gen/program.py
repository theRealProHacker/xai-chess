"""The 'DSPy program': one predictor  (moves, board) -> comment.

A candidate is {"name", "instruction", "demos": [example ids]} plus optional "context_file" (lessons
prepended verbatim), "tools" (board_tools facts block after every board, demos included),
"thinking", "max_tokens". render() builds the prompt, run() executes it over a set of examples
with a thread pool and writes runs/<name>/<split>.jsonl.
"""
import argparse, concurrent.futures as cf, json, sys, threading, time
from pathlib import Path

HERE = Path(__file__).parent
DATA, RUNS = HERE / "data", HERE / "runs"


def load(split):
    return [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]


def by_id(*splits):
    return {e["id"]: e for s in splits for e in load(s)}


def fmt_input(e, tools=False):
    s = (f"Moves so far (the last move is the one to comment on):\n{e['moves']}\n\n"
         f"Board after {e['move']} ({e['side']} just moved; uppercase = White, lowercase = Black, "
         f"rank 8 at the top):\n{e['board']}")
    if tools:
        s += f"\n\nComputed facts (exact, from the board above):\n{tool_block(e)}"
    return s


_tool_cache = {}


def tool_block(e):
    """board_tools.report_after_move for the example's last move, computed from the move list."""
    if e["id"] not in _tool_cache:
        import chess, board_tools
        b = chess.Board()
        for tok in e["moves"].split():
            if not tok[0].isdigit():
                b.push_san(tok)
        b.pop()
        _tool_cache[e["id"]] = board_tools.report_after_move(b, e["move"])
    return _tool_cache[e["id"]]


def render(cand, e, pool):
    tools = bool(cand.get("tools"))
    parts = []
    if cand.get("context_file"):
        parts += [cand.get("context_intro", "Background lessons (use their ideas where they apply; never quote, cite "
                                            "or borrow their vocabulary):"),
                  "", "=== BEGIN LESSONS ===", (HERE.parent / cand["context_file"]).read_text(encoding="utf-8").strip(),
                  "=== END LESSONS ===", ""]
    parts += [cand["instruction"].strip(), ""]
    for did in cand.get("demos", []):
        d = pool[did]
        parts += ["--- Example ---", fmt_input(d, tools), "", f"Commentary:\n{d['comment']}", ""]
    parts += ["--- Your turn ---", fmt_input(e, tools), "", "Commentary:"]
    return "\n".join(parts)


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
        r = llm.generate(render(cand, e, pool), temperature=temperature, thinking=cand.get("thinking"),
                         max_tokens=cand.get("max_tokens", 600))
        row = {"id": e["id"], "gen": r["text"], "thoughts": r.get("thoughts", ""), "finish": r["finish"],
               "usage": r["usage"]}
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
