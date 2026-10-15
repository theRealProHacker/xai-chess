"""Board-level MLP decoder from all 64 squares' features to Leela's value head (value/dense1, value/dense2).

These taps are one vector per position, not per square, so the decoder mirrors the head: a shared
per-square map (features + square one-hot -> 32, like value/embed), flattened over the 64 squares,
then one hidden layer. Same positions, bits and held-out split as square_probe.py; 10% of the train
positions are held back for early stopping. Reported: FVU on held-out positions.

  python lc0-probe/value_decoder_mlp.py
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import chess, numpy as np, onnx, onnxruntime as ort, torch
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--nodes", type=int, default=200000)
ap.add_argument("--taps", default="/value/dense1/mish,/value/dense2/add")
ap.add_argument("--min", type=int, default=20, help="positive tokens in train needed to use a bit")
ap.add_argument("--embed", type=int, default=32, help="per-square width, as value/embed")
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--batch", type=int, default=256, help="positions per step")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--threads", type=int, default=4, help="CPU threads, leaving the rest of the machine usable")
a = ap.parse_args()
rng = np.random.default_rng(0)
torch.manual_seed(0)
torch.set_num_threads(a.threads)

fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:a.n]
N = len(fens)
binary = ROOT / "features/square/target/release/square"
schema = json.loads(subprocess.run([binary, "--schema"], capture_output=True, text=True, check=True).stdout)
L = schema["bits_per_square"]
toks, bits = [], []
with open(ROOT / f"lc0-probe/data/square_bits_n{N}_nodes{a.nodes}.jsonl") as f:   # streamed: 150k rows don't fit as objects
    for i, l in enumerate(f):
        sq = json.loads(l)["squares"]
        toks.append(np.repeat(np.array([i * 64 + chess.parse_square(s) for s in sq], np.int32), [len(v) for v in sq.values()]))
        bits.append(np.fromiter((b for v in sq.values() for b in v), np.int32))
tok, bit = np.concatenate(toks), np.concatenate(bits); del toks, bits
te = rng.random(N) < 0.2                                                   # same split as square_probe.py
va = ~te & (np.random.default_rng(1).random(N) < 0.1)                      # early-stopping slice of train
tr = ~te & ~va

keep = np.flatnonzero(np.bincount(bit[~np.repeat(te, 64)[tok]], minlength=L) >= a.min)
col = np.full(L, -1); col[keep] = np.arange(len(keep))
k = col[bit] >= 0
col = col.astype(np.int32)                                                 # int32 throughout: 150k positions are ~350M entries
tok, idx = tok[k], col[bit[k]]; del bit, k
tok = np.concatenate([tok, np.arange(N * 64, dtype=np.int32)])
idx = np.concatenate([idx, (len(keep) + np.arange(N * 64) % 64).astype(np.int32)])
o = np.argsort(tok, kind="stable"); tok = tok[o]; idx = idx[o]; del o
start = np.searchsorted(tok, np.arange(N * 64 + 1)); n_idx = len(idx); del tok
D_in = len(keep) + 64
print(f"{N:,} positions, {len(keep):,} bits used", flush=True)

def bags(pos):
    """EmbeddingBag inputs for all 64 squares of each position in pos, position-major."""
    t = (pos[:, None] * 64 + np.arange(64)).ravel()
    lo, hi = start[t], start[t + 1]
    flat = np.concatenate([idx[l:h] for l, h in zip(lo, hi)]).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(hi - lo)[:-1]])
    return torch.from_numpy(flat), torch.from_numpy(off)

m = onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx")
taps = a.taps.split(",")
for t in taps: m.graph.output.append(onnx.ValueInfoProto(name=t))
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
X = {t: [] for t in taps}
for s in range(0, N, 256):
    for t, x in zip(taps, sess.run(taps, {"/input/planes": encode_fens(fens[s:s + 256])})):
        X[t].append(x.reshape(len(x), -1).astype(np.float32))
X = {t: np.concatenate(v) for t, v in X.items()}                           # (N, D): one vector per position

class Decoder(torch.nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.sq = torch.nn.EmbeddingBag(D_in, a.embed, mode="sum")         # shared per-square map, like value/embed
        torch.nn.init.normal_(self.sq.weight, std=(n_idx / (N * 64)) ** -0.5)
        self.sq_b = torch.nn.Parameter(torch.zeros(a.embed))
        self.hid = torch.nn.Linear(64 * a.embed, a.hidden)
        self.out = torch.nn.Linear(a.hidden, d_out)
    def forward(self, flat, off):
        e = torch.nn.functional.gelu(self.sq(flat, off) + self.sq_b).reshape(-1, 64 * a.embed)
        return self.out(torch.nn.functional.gelu(self.hid(e)))

def fvu(net, Z, pos):
    with torch.no_grad():
        P = torch.cat([net(*bags(pos[s:s + 1024])) for s in range(0, len(pos), 1024)])
    return float(((Z[pos] - P) ** 2).sum() / ((Z[pos] - Z[pos].mean(0)) ** 2).sum())

out = ROOT / f"lc0-probe/results_value_decoder_mlp_n{N}.json"
res = json.load(open(out)) if out.exists() else {"taps": {}}               # new taps merge into earlier ones
res.update({"n": N, "bits": int(len(keep)), "embed": a.embed, "hidden": a.hidden, "lr": a.lr, "wd": a.wd})
TR, VA, TE = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
for t in taps:
    Y = X[t]
    Z = torch.from_numpy((Y - Y[TR].mean(0)) / Y[TR].std())               # one scalar scale keeps FVU's weighting
    net = Decoder(Y.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(TR)
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            loss = torch.nn.functional.mse_loss(net(*bags(b)), Z[b])
            opt.zero_grad(); loss.backward(); opt.step()
        v = fvu(net, Z, VA)
        print(f"{t:20} epoch {ep + 1:3}  val FVU {v:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if v < best: best, bad, best_state = v, 0, {k_: p.clone() for k_, p in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    net.load_state_dict(best_state)
    r = {"dim": int(Y.shape[1]), "train": fvu(net, Z, TR), "val": best, "test": fvu(net, Z, TE), "epochs": ep + 1}
    print(f"{t:20} MLP FVU train {r['train']:.3f}  test {r['test']:.3f}", flush=True)
    res["taps"][t] = r
    json.dump(res, open(out, "w"), indent=1)
