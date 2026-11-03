"""Data dir for the ChessQA structural curriculum (oracle/chessqa.py): Leela layer-9 tokens of
  train/val  Lichess puzzle positions (ChessQA's source for 8 tasks; its puzzle IDs and FENs left out) and broadcast-game
             positions after 30 half-moves with the next 15 moves (state tracking; broadcasts from after ChessQA's release)
  test       the 1,100 ChessQA structural items, prompts verbatim, gold answers as published
Writes acts.npy, labels.npz (wdl, black, split, fens, src 0 puzzle / 1 game / 2 ChessQA), moves.jsonl (game positions),
tasks_val.jsonl (one row per val position per task), tasks_test.jsonl (ChessQA), empty val.jsonl/test.jsonl.

  PYTHONPATH=.pylibs .venv/bin/python oracle/build_chessqa.py --chessqa structural.json --pgn b1.pgn.zst b2.pgn.zst
"""
import argparse, io, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe"), str(ROOT / "oracle")]
import numpy as np, onnx, onnxruntime as ort, chess, chess.pgn
from lc0_planes import encode_fens
import chessqa

ap = argparse.ArgumentParser()
ap.add_argument("--chessqa", required=True, help="ChessQA structural items as JSON (the HF parquet converted)")
ap.add_argument("--puzzles", default=str(Path.home() / "Downloads/lichess_db_puzzle.csv.zst"))
ap.add_argument("--pgn", nargs="+", required=True, help="Lichess broadcast PGN .zst files")
ap.add_argument("--n-puzzles", type=int, default=200000)
ap.add_argument("--val-puzzles", type=int, default=1000)
ap.add_argument("--val-games", type=int, default=300)
ap.add_argument("--start-ply", type=int, default=30, help="ChessQA: fragments start after the first 30 half-moves")
ap.add_argument("--track", type=int, default=15, help="moves kept after the start (ChessQA long = 11..15)")
ap.add_argument("--tap", default="/encoder9/ln2")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--out", default=str(ROOT / "oracle/data_chessqa"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

items = json.load(open(a.chessqa))
cq_rows = [chessqa.from_chessqa(r) for r in items]
cq_ids = {json.loads(r["metadata_json"]).get("puzzle_id") for r in items} - {None}
cq_fens = {r["fen"] for r in cq_rows}

# puzzles: CSV order (random IDs), skipping ChessQA's puzzles and repeated FENs
puz, seen = [], set(cq_fens)
p = subprocess.Popen(["zstdcat", a.puzzles], stdout=subprocess.PIPE, text=True)
next(p.stdout)
for l in p.stdout:
    pid, fen = l.split(",", 2)[:2]
    if pid in cq_ids or fen in seen: continue
    seen.add(fen); puz.append(fen)
    if len(puz) == a.n_puzzles + a.val_puzzles: break
p.kill()
print(f"puzzles: {len(puz):,}", flush=True)

# games: position after start-ply half-moves + the next moves; one fragment per distinct (fen, moves)
games, gseen = [], set()
for f in a.pgn:
    txt = io.StringIO(subprocess.run(["zstdcat", f], capture_output=True, text=True).stdout)
    while (g := chess.pgn.read_game(txt)) is not None:
        if "FEN" in g.headers: continue
        mv = list(g.mainline_moves())
        if len(mv) < a.start_ply + 1: continue
        b = g.board()
        for m in mv[:a.start_ply]: b.push(m)
        frag = [m.uci() for m in mv[a.start_ply:a.start_ply + a.track]]
        key = (b.fen(), " ".join(frag))
        if b.fen() in cq_fens or key in gseen: continue
        gseen.add(key); games.append((b.fen(), frag))
    print(f"games after {Path(f).name}: {len(games):,}", flush=True)

nP, nG = len(puz), len(games)
fens = puz + [g[0] for g in games] + [r["fen"] for r in cq_rows]
src = np.array([0] * nP + [1] * nG + [2] * len(cq_rows), np.int8)
split = np.zeros(len(fens), np.int8)
split[nP - a.val_puzzles:nP] = 1
split[nP + nG - a.val_games:nP + nG] = 1
split[src == 2] = 2
N = len(fens)
print(f"positions: {N:,} (train {(split == 0).sum():,}, val {(split == 1).sum():,}, test {(split == 2).sum():,})", flush=True)

with open(OUT / "moves.jsonl", "w") as f:
    for k, (_, frag) in enumerate(games): f.write(json.dumps(dict(i=nP + k, moves=frag)) + "\n")
for name in ("val.jsonl", "test.jsonl"): (OUT / name).write_text("")

# val rows: one per val position per task (state tracking on games, the rest on puzzles), as in training
rng = np.random.default_rng(1)
moves = {nP + k: g[1] for k, g in enumerate(games)}
with open(OUT / "tasks_val.jsonl", "w") as f:
    for i in np.flatnonzero(split == 1):
        for t in chessqa.TASKS:
            if (t in chessqa.TRACK) != (src[i] == 1): continue
            r = chessqa.sample(t, rng, fens[i], moves.get(int(i)))
            if r is not None: f.write(json.dumps(dict(i=int(i), **r)) + "\n")
with open(OUT / "tasks_test.jsonl", "w") as f:
    for k, r in enumerate(cq_rows): f.write(json.dumps(dict(i=nP + nG + k, **r)) + "\n")

m = onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx")
m.graph.output.append(onnx.ValueInfoProto(name=a.tap))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
acts = np.lib.format.open_memmap(OUT / "acts.npy", "w+", np.float16, (N, 64, 256))
wdl = np.zeros((N, 3), np.float32)
x = np.zeros((a.batch, 112, 8, 8), np.float32)
for s in range(0, N, a.batch):
    e = min(N, s + a.batch)
    encode_fens(fens[s:e], x[:e - s])
    w, t = sess.run(["/output/wdl", a.tap], {"/input/planes": x[:e - s]})
    wdl[s:e] = w; acts[s:e] = t.reshape(e - s, 64, 256)
    if s // a.batch % 20 == 0: print(f"acts {e:,}/{N:,}", flush=True)
acts.flush()
black = np.array([f.split()[1] == "b" for f in fens])
np.savez(OUT / "labels.npz", wdl=wdl, black=black, split=split, fens=np.array(fens), src=src, tap=a.tap)
print("done", flush=True)
