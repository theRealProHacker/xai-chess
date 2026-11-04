"""Picks mistakes and blunders from the sampled games for the agent run → annotate/data/cases.jsonl.

  python annotate/cases.py [--blunders 80] [--mistakes 40] [--seed 0]

Lichess grading: a move losing ≥0.3 winning chances is a blunder, ≥0.2 a mistake. Cases are taken
round-robin over the games, so every game is represented.
"""
import argparse, json, math, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / ".pylibs"))
import chess

ap = argparse.ArgumentParser()
ap.add_argument("--blunders", type=int, default=80)
ap.add_argument("--mistakes", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

games = [json.loads(l) for l in (HERE / "data/games.jsonl").open()]
evals = {}
for l in (HERE / "data/evals.jsonl").open():
    e = json.loads(l)
    if all("pv" in x for x in e["evals"]):
        evals[e["id"]] = e["evals"]


def chances(e, ply):
    """White's winning chances in [-1, 1] at the position after `ply` plies."""
    if e.get("mate") is not None:
        return (1 if ply % 2 else -1) if e["mate"] == 0 else math.copysign(1, e["mate"])
    return 2 / (1 + math.exp(-0.00368208 * e["cp"])) - 1


rng = random.Random(a.seed)
pools = {"Blunder": [], "Mistake": []}
for g in games:
    E = evals.get(g["id"])
    if not E:
        continue
    board = chess.Board()
    for p, mov in enumerate(g["moves"]):
        loss = (chances(E[p], p) - chances(E[p + 1], p + 1)) * (-1 if p % 2 else 1)
        kind = "Blunder" if loss >= 0.3 else "Mistake" if loss >= 0.2 else None
        if kind:
            pools[kind].append({"id": f"{g['id']}-{p}", "game": g["id"], "url": g["url"], "ply": p,
                                "fen": board.fen(), "mov": mov, "kind": kind, "loss": round(loss, 3),
                                "before": E[p], "after": E[p + 1],
                                "white": g["white"], "black": g["black"],
                                "white_elo": g["white_elo"], "black_elo": g["black_elo"]})
        board.push_san(mov)


def pick(pool, n):
    """Round-robin over games, each game's cases in random order."""
    by = {}
    for c in pool:
        by.setdefault(c["game"], []).append(c)
    for cs in by.values():
        rng.shuffle(cs)
    order, out = list(by), []
    rng.shuffle(order)
    while len(out) < n and any(by.values()):
        for gid in order:
            if by[gid] and len(out) < n:
                out.append(by[gid].pop())
    return out


cases = pick(pools["Blunder"], a.blunders) + pick(pools["Mistake"], a.mistakes)
cases.sort(key=lambda c: (c["game"], c["ply"]))
with (HERE / "data/cases.jsonl").open("w") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")
print(f"{len(cases)} cases from {len({c['game'] for c in cases})} games "
      f"({sum(c['kind'] == 'Blunder' for c in cases)} blunders)")
