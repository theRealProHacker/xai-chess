"""Linear probe from each square's Leela token to that square's per-square features.

Token: /encoder5/ln1 (the residual after block 5's attention), one 256-d vector per square.
Labels: features/square, with Stockfish lines for the tactics groups. Leela sees the board from
the side to move, so a black-to-move position is colour-mirrored before extraction: "white" in
a label is the side to move and squares are in Leela's token order.

One logistic probe per bit, all bits trained jointly; held out by position. Bits with fewer than
--min positives in either split are skipped. Reported: AUROC per bit, mean per group.

  python lc0-probe/square_probe.py --n 20000 --nodes 200000 --workers 8
"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import chess, numpy as np, onnx, onnxruntime as ort, scipy.sparse as sp
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000, help="positions (first n of sf-probe/data/fens.txt)")
ap.add_argument("--tap", default="/encoder5/ln1")
ap.add_argument("--nodes", type=int, default=200000)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--min", type=int, default=20, help="positive tokens needed in train and in test")
ap.add_argument("--steps", type=int, default=4000)
a = ap.parse_args()
rng = np.random.default_rng(0)

fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:a.n]
N = len(fens)
stm = [chess.Board(f).mirror().fen() if f.split()[1] == "b" else f for f in fens]

# --- labels, cached: the engine run is the slow part
cache = ROOT / f"lc0-probe/data/square_bits_n{N}_nodes{a.nodes}.jsonl"
binary = ROOT / "features/square/target/release/square"
subprocess.run(["cargo", "build", "--release", "-q"], cwd=ROOT / "features/square", check=True)
if not cache.exists():
    tmp = cache.with_suffix(".parts"); tmp.mkdir(exist_ok=True)
    procs = []
    for w in range(a.workers):                       # files, not pipes: a full pipe deadlocks the workers
        (tmp / f"{w}.in").write_text("\n".join(stm[w::a.workers]) + "\n")
        procs.append(subprocess.Popen([binary, "--indices", "--engine", ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2",
                                       "--nodes", str(a.nodes)], stdin=open(tmp / f"{w}.in"), stdout=open(tmp / f"{w}.out", "w")))
    assert all(p.wait() == 0 for p in procs), "square extractor failed"
    outs = [(tmp / f"{w}.out").read_text().splitlines() for w in range(a.workers)]
    rows = [outs[i % a.workers][i // a.workers] for i in range(N)]
    cache.write_text("\n".join(rows) + "\n")
    for f in tmp.iterdir(): f.unlink()
    tmp.rmdir()
rows = [json.loads(l) for l in cache.read_text().splitlines()]
assert len(rows) == N and all(r["fen"] == f for r, f in zip(rows, stm))

schema = json.loads(subprocess.run([binary, "--schema"], capture_output=True, text=True, check=True).stdout)
L = schema["bits_per_square"]
tok, bit = [], []
for i, r in enumerate(rows):
    for s, v in r["squares"].items():
        t = i * 64 + chess.parse_square(s)
        tok += [t] * len(v); bit += v
tok, bit = np.array(tok), np.array(bit)

# --- tokens
m = onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx")
m.graph.output.append(onnx.ValueInfoProto(name=a.tap))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
X = np.concatenate([sess.run([a.tap], {"/input/planes": encode_fens(fens[s:s + 256])})[0].reshape(-1, 256)
                    for s in range(0, N, 256)]).astype(np.float32)          # (N*64, D), token = pos*64 + sq
te = np.repeat(rng.random(N) < 0.2, 64)                                    # held-out positions

pos_tr = np.bincount(bit[~te[tok]], minlength=L)
pos_te = np.bincount(bit[te[tok]], minlength=L)
keep = np.flatnonzero((pos_tr >= a.min) & (pos_te >= a.min))
col = np.full(L, -1); col[keep] = np.arange(len(keep))
B = len(keep)
print(f"{N:,} positions, {L:,} bits per square, {B:,} with >= {a.min} positives in both splits", flush=True)

k = col[bit] >= 0
Y = sp.csr_matrix((np.ones(k.sum(), bool), (tok[k], col[bit[k]])), shape=(N * 64, B))

mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
X = (X - mu) / sd
Xtr, Ytr, Xte, Yte = X[~te], Y[~te], X[te], Y[te]
del X, Y

# --- joint logistic probes, numpy + Adam
W = np.zeros((Xtr.shape[1], B), np.float32)
p0 = np.asarray(Ytr.mean(0)).ravel().clip(1e-6, 1 - 1e-6)
b = np.log(p0 / (1 - p0)).astype(np.float32)
P = {"W": W, "b": b}
M = {n: np.zeros_like(v) for n, v in P.items()}; V = {n: np.zeros_like(v) for n, v in P.items()}
for t in range(1, a.steps + 1):
    i = rng.integers(0, len(Xtr), 8192)
    x, y = Xtr[i], Ytr[i].toarray()
    z = x @ P["W"] + P["b"]
    g = (1 / (1 + np.exp(-z)) - y) / len(x)
    G = {"W": x.T @ g + 1e-4 * P["W"], "b": g.sum(0)}
    for n in P:
        M[n] = 0.9 * M[n] + 0.1 * G[n]; V[n] = 0.999 * V[n] + 0.001 * G[n] ** 2
        P[n] -= 1e-2 * (M[n] / (1 - 0.9 ** t)) / (np.sqrt(V[n] / (1 - 0.999 ** t)) + 1e-8)
    if t % 500 == 0: print(f"step {t}", flush=True)

def auroc(s, y):
    r = np.empty(len(s)); r[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum(); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))

names = [g["group"] for g in schema["groups"]]
offs = np.array([g["offset"] for g in schema["groups"]])
label = lambda j: names[np.searchsorted(offs, j, "right") - 1]
per_bit = {}
Yte = Yte.tocsc()
for c0 in range(0, B, 256):
    S = Xte @ P["W"][:, c0:c0 + 256] + P["b"][c0:c0 + 256]
    for c in range(c0, min(c0 + 256, B)):
        j = keep[c]
        y = Yte[:, c].toarray().ravel()
        per_bit[schema["labels"][j]] = {"auroc": auroc(S[:, c - c0], y), "group": label(j),
                                        "pos_train": int(pos_tr[j]), "pos_test": int(pos_te[j])}
groups = {}
for v in per_bit.values(): groups.setdefault(v["group"], []).append(v["auroc"])
summary = {g: {"bits": len(v), "mean_auroc": float(np.mean(v)), "min": float(np.min(v))} for g, v in groups.items()}
slug = a.tap.strip("/").replace("/", "_")
out = ROOT / f"lc0-probe/results_square_{slug}.json"
json.dump({"tap": a.tap, "n": N, "nodes": a.nodes, "min_positives": a.min, "bits_probed": B,
           "groups": summary, "bits": per_bit}, open(out, "w"), indent=1)
np.savez(ROOT / f"lc0-probe/data/square_probe_{slug}.npz", W=P["W"], b=P["b"], keep=keep, mu=mu, sd=sd)
for g in names:
    if g in summary: print(f"{g:28} {summary[g]['bits']:>6} bits  mean AUROC {summary[g]['mean_auroc']:.3f}")
