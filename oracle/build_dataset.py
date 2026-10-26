"""Eval-oracle dataset: Leela layer-9 residual (/encoder9/ln2, 64 square tokens) + value-head WDL.

Positions: the first N of --fens (default: sf-probe/data/fens.txt, pin puzzles), split by position into train/val/test.
Writes to oracle/data/:
  acts.npy        float16 [N, 64, 256]   layer-9 tokens, in the net's square order (side to move = us)
  labels.npz      wdl [N, 3] (side to move), black [N] bool, split [N] (0 train, 1 val, 2 test), fens
  val.jsonl, test.jsonl   one fixed (question, answer) per position; train resamples every epoch
  train_preview.jsonl     200 sampled train rows, for reading

  PYTHONPATH=.pylibs .venv/bin/python oracle/build_dataset.py [--tap /encoder5/ln2 --out oracle/data_l5]
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe"), str(ROOT / "oracle")]
import numpy as np, onnx, onnxruntime as ort
from lc0_planes import encode_fens
import qa

ap = argparse.ArgumentParser()
ap.add_argument("--fens", default=str(ROOT / "sf-probe/data/fens.txt"), help="fens_games.txt = one position per Lichess game")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--val", type=int, default=1000)
ap.add_argument("--test", type=int, default=1000)
ap.add_argument("--tap", default="/encoder9/ln2")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--out", default=str(ROOT / "oracle/data"))
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)

fens = [l for l in Path(a.fens).read_text().split("\n") if l][:a.n]
N = len(fens)
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
    wdl[s:e] = w
    acts[s:e] = t.reshape(e - s, 64, 256)
    print(f"\r{e:,}/{N:,}", end="", flush=True)
print()
acts.flush()

rng = np.random.default_rng(0)
split = np.zeros(N, np.int8)
perm = rng.permutation(N)
split[perm[:a.val]] = 1
split[perm[a.val:a.val + a.test]] = 2
black = np.array([f.split()[1] == "b" for f in fens])
np.savez(OUT / "labels.npz", wdl=wdl, black=black, split=split, fens=np.array(fens), tap=a.tap)


def rows(idx, seed):
    r = np.random.default_rng(seed)
    for i in idx:
        q, fmt, colour, qt, at = qa.sample(r, wdl[i], black[i])
        yield dict(i=int(i), q=q, fmt=fmt, colour=colour, question=qt, answer=at, fen=fens[i])


for name, k, seed in (("val", 1, 1), ("test", 2, 2)):
    with open(OUT / f"{name}.jsonl", "w") as f:
        for row in rows(np.flatnonzero(split == k), seed):
            f.write(json.dumps(row) + "\n")
with open(OUT / "train_preview.jsonl", "w") as f:
    for row in rows(np.flatnonzero(split == 0)[:200], 3):
        f.write(json.dumps(row) + "\n")

p = np.array([qa.pawns(w, l) for w, _, l in wdl])
print(f"{N:,} positions ({(split == 0).sum():,} train / {a.val:,} val / {a.test:,} test), tap {a.tap}")
print(f"pawns: mean {p.mean():+.2f}, |p| median {np.median(abs(p)):.2f}, clipped at ±{qa.PAWN_CLIP:g}: {(abs(p) >= qa.PAWN_CLIP).mean():.1%}")
print("bins -3..3:", np.bincount(np.array([qa.bin_of(v) for v in p]) + 3, minlength=7))
