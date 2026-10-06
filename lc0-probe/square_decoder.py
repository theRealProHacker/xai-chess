"""Linear decoder from each square's per-square features back to its Leela token.

The reverse of square_probe.py: token ~ W @ bits + b[square], ridge least squares, shared W across
squares. Same positions, labels and held-out split as the probe (run it first: it caches the labels).
Reported: fraction of variance unexplained (FVU) on held-out positions, against a decoder from the
square alone and from the features alone (with a global bias).

  python lc0-probe/square_decoder.py --taps /attn_body/rehape,/encoder0/ln1,...,/encoder9/ln1
"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import chess, numpy as np, onnx, onnxruntime as ort, scipy.linalg as la, scipy.sparse as sp
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--nodes", type=int, default=200000)
ap.add_argument("--taps", default="/encoder5/ln1,/attn_body/rehape")
ap.add_argument("--min", type=int, default=20, help="positive tokens in train needed to use a bit")
ap.add_argument("--ridge", type=float, default=1.0)
a = ap.parse_args()
rng = np.random.default_rng(0)

fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:a.n]
N = len(fens)
rows = [json.loads(l) for l in (ROOT / f"lc0-probe/data/square_bits_n{N}_nodes{a.nodes}.jsonl").read_text().splitlines()]
binary = ROOT / "features/square/target/release/square"
schema = json.loads(subprocess.run([binary, "--schema"], capture_output=True, text=True, check=True).stdout)
L = schema["bits_per_square"]
tok, bit = [], []
for i, r in enumerate(rows):
    for s, v in r["squares"].items():
        tok += [i * 64 + chess.parse_square(s)] * len(v); bit += v
tok, bit = np.array(tok), np.array(bit)
te = np.repeat(rng.random(N) < 0.2, 64)                                    # same split as square_probe.py

keep = np.flatnonzero(np.bincount(bit[~te[tok]], minlength=L) >= a.min)
col = np.full(L, -1); col[keep] = np.arange(len(keep))
k = col[bit] >= 0
F = sp.csr_matrix((np.ones(k.sum(), np.float32), (tok[k], col[bit[k]])), shape=(N * 64, len(keep)))
Q = sp.csr_matrix((np.ones(N * 64, np.float32), (np.arange(N * 64), np.tile(np.arange(64), N))), shape=(N * 64, 64))
B = sp.csr_matrix(np.ones((N * 64, 1), np.float32))                       # global bias, no square
print(f"{N:,} positions, {len(keep):,} bits used", flush=True)

m = onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx")
taps = a.taps.split(",")
for t in taps: m.graph.output.append(onnx.ValueInfoProto(name=t))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])

def tokens(t):
    """(N*64, D), token = pos*64 + sq; one tap at a time to bound memory."""
    xs = [sess.run([t], {"/input/planes": encode_fens(fens[s:s + 256])})[0] for s in range(0, N, 256)]
    return np.concatenate([x.reshape(-1, x.shape[-1]).astype(np.float32) for x in xs])

def fit(A, Y, free):
    """Ridge on sparse A; the last `free` columns (square one-hot or bias) are not penalised."""
    G = (A[~te].T @ A[~te]).toarray().astype(np.float64)
    pen = np.full(A.shape[1], a.ridge); pen[-free:] = 0
    G[np.diag_indices_from(G)] += pen + 1e-6
    return la.solve(G, np.asarray(A[~te].T @ Y[~te], np.float64), assume_a="pos")

def fvu(Y, Yh):
    return float(((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

res = {"n": N, "bits": int(len(keep)), "ridge": a.ridge, "taps": {}}
for t in taps:
    Y = tokens(t)
    r = {}
    for name, A, free in [("square only", Q, 64), ("features only", sp.hstack([F, B]).tocsr(), 1),
                          ("features + square", sp.hstack([F, Q]).tocsr(), 64)]:
        W = fit(A, Y, free)
        r[name] = {"train": fvu(Y[~te], A[~te] @ W), "test": fvu(Y[te], A[te] @ W)}
        print(f"{t:22} {name:18} FVU train {r[name]['train']:.3f}  test {r[name]['test']:.3f}", flush=True)
    res["taps"][t] = r
json.dump(res, open(ROOT / "lc0-probe/results_square_decoder.json", "w"), indent=1)
