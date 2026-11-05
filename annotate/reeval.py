"""Searches every sampled game again as server.py now does, in parallel, and rewrites data/evals.jsonl.

  python annotate/reeval.py [--jobs 8]
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.argv, args = sys.argv[:1], sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("server", HERE / "server.py")
server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)
import chess.engine


def run(game):
    e = chess.engine.SimpleEngine.popen_uci(str(server.STOCKFISH))
    e.configure({"Threads": 1})
    try:
        return {"id": game["id"], "search": server.SEARCH, "evals": server.search_game(e, game["moves"])}
    finally:
        e.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    jobs = ap.parse_args(args).jobs
    games = list(server.games().values())
    with ProcessPoolExecutor(jobs) as ex:
        out = list(ex.map(run, games))
    with server.EVALS.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} games")
