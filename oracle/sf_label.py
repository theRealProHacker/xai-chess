"""One Stockfish pass per position: depth-18 MultiPV-5, appended to <data>/sf.jsonl as it goes.

Each line: {"i", "lines": [{"uci", "cp" | "mate", "pv"}]}, scores from the side to move. Resumes by
skipping indices already in the file; val/test positions go first. Game-over positions get "lines": [].

  PYTHONPATH=.pylibs .venv/bin/python oracle/sf_label.py --data oracle/data_g50k [--workers 8]
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import numpy as np, chess, chess.engine

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "oracle/data_g50k"))
ap.add_argument("--depth", type=int, default=18)
ap.add_argument("--multipv", type=int, default=5)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--sf", default=str(ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2"))
a = ap.parse_args()
OUT = Path(a.data) / "sf.jsonl"
L = np.load(Path(a.data) / "labels.npz")
fens = L["fens"]
done = {json.loads(l)["i"] for l in OUT.open()} if OUT.exists() else set()
order = np.argsort(L["split"] == 0, kind="stable") if "split" in L else range(len(fens))   # val/test first
todo = [int(i) for i in order if int(i) not in done]
eng = None


def run(i):
    global eng
    if eng is None:
        eng = chess.engine.SimpleEngine.popen_uci(a.sf)
        eng.configure({"Threads": 1, "Hash": 64})
    b = chess.Board(str(fens[i]))
    if b.is_game_over():
        return dict(i=i, lines=[])
    eng.configure({"Clear Hash": None})
    lines = []
    for info in eng.analyse(b, chess.engine.Limit(depth=a.depth), multipv=a.multipv):
        s = info["score"].pov(b.turn)
        pv = [m.uci() for m in info["pv"]]
        lines.append(dict(uci=pv[0], pv=pv, **({"mate": s.mate()} if s.is_mate() else {"cp": s.score()})))
    return dict(i=i, lines=lines)


print(f"{len(done):,} done, {len(todo):,} to go", flush=True)
with Pool(a.workers) as p, OUT.open("a") as f:
    for n, r in enumerate(p.imap_unordered(run, todo, chunksize=4), 1):
        f.write(json.dumps(r) + "\n")
        if n % 100 == 0:
            f.flush(); print(f"\r{len(done) + n:,}/{len(fens):,}", end="", flush=True)
print()
