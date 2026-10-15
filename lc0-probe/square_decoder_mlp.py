"""MLP decoder from each square's per-square features (+ square) back to its Leela token.

The nonlinear counterpart of square_decoder.py's "features + square" ridge fit: one wide hidden layer,
shared across squares. Same positions, bits and held-out split; 10% of the train positions are held
back for early stopping. Reported: FVU on held-out positions, next to the ridge FVU.

  python lc0-probe/square_decoder_mlp.py --taps /attn_body/rehape,/encoder0/ln1,...,/encoder9/ln1
  inside each layer also /encoderN/ffn/dense1/sqrrelu/sqr, /encoderN/ffn/dense2/b, /encoderN/ln2
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
ap.add_argument("--taps", default="/encoder5/ln1")
ap.add_argument("--min", type=int, default=20, help="positive tokens in train needed to use a bit")
ap.add_argument("--hidden", type=int, default=2048)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--threads", type=int, default=4, help="CPU threads, leaving the rest of the machine usable")
ap.add_argument("--save", action="store_true", help="write decoded tokens for every square to data/decoded_<tap>.npy")
a = ap.parse_args()
rng = np.random.default_rng(0)
torch.manual_seed(0)
torch.set_num_threads(a.threads)

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
te_pos = rng.random(N) < 0.2                                               # same split as square_probe.py
te = np.repeat(te_pos, 64)
va = np.repeat(~te_pos & (np.random.default_rng(1).random(N) < 0.1), 64)   # early-stopping slice of train
tr = ~te & ~va

keep = np.flatnonzero(np.bincount(bit[~te[tok]], minlength=L) >= a.min)
col = np.full(L, -1); col[keep] = np.arange(len(keep))
k = col[bit] >= 0
tok, idx = tok[k], col[bit[k]]
sq = np.arange(N * 64) % 64                                                # square one-hot as extra inputs
tok = np.concatenate([tok, np.arange(N * 64)]); idx = np.concatenate([idx, len(keep) + sq])
o = np.argsort(tok, kind="stable"); tok, idx = tok[o], idx[o]
start = np.searchsorted(tok, np.arange(N * 64 + 1))                       # token t's inputs: idx[start[t]:start[t+1]]
D_in = len(keep) + 64
print(f"{N:,} positions, {len(keep):,} bits used, {len(idx) / (N * 64):.1f} active inputs per token", flush=True)

def bags(rows_):
    """EmbeddingBag inputs for a batch of token ids."""
    lo, hi = start[rows_], start[rows_ + 1]
    flat = np.concatenate([idx[l:h] for l, h in zip(lo, hi)])
    off = np.concatenate([[0], np.cumsum(hi - lo)[:-1]])
    return torch.from_numpy(flat), torch.from_numpy(off)

m = onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx")
taps = a.taps.split(",")
for t in taps: m.graph.output.append(onnx.ValueInfoProto(name=t))
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

def tokens(t):
    """(N*64, D), token = pos*64 + sq; one tap at a time to bound memory."""
    Y = None
    for s in range(0, N, 256):                                             # filled in place: no second copy
        x = sess.run([t], {"/input/planes": encode_fens(fens[s:s + 256])})[0]
        x = x.reshape(-1, x.shape[-1])
        if Y is None: Y = np.empty((N * 64, x.shape[1]), np.float32)
        Y[s * 64:s * 64 + len(x)] = x
    return Y

class MLP(torch.nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.inp = torch.nn.EmbeddingBag(D_in, a.hidden, mode="sum")       # = Linear on the sparse 0/1 input
        torch.nn.init.normal_(self.inp.weight, std=(len(idx) / (N * 64)) ** -0.5)   # unit pre-activations
        self.bias = torch.nn.Parameter(torch.zeros(a.hidden))
        self.out = torch.nn.Linear(a.hidden, d_out)
    def forward(self, flat, off):
        return self.out(torch.nn.functional.gelu(self.inp(flat, off) + self.bias))

def fvu(net, Z, rows_):
    """FVU of net on Z[rows_], in chunks (dense1 taps are 1024 wide). Invariant to Z's shift and scale."""
    chunks = [rows_[s:s + a.batch] for s in range(0, len(rows_), a.batch)]
    mean = sum(Z[c].sum(0) for c in chunks) / len(rows_)
    sse = sst = 0.0
    with torch.no_grad():
        for c in chunks:
            sse += float(((Z[c] - net(*bags(c))) ** 2).sum()); sst += float(((Z[c] - mean) ** 2).sum())
    return sse / sst

ridge = json.load(open(ROOT / "lc0-probe/results_square_decoder.json"))["taps"]
out = ROOT / "lc0-probe/results_square_decoder_mlp.json"
res = json.load(open(out)) if out.exists() else {}
res.update({"n": N, "bits": int(len(keep)), "hidden": a.hidden, "lr": a.lr, "batch": a.batch})
res.setdefault("taps", {})
TR, VA, TE = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
for t in taps:
    Y = tokens(t)
    ch = [TR[s:s + 65536] for s in range(0, len(TR), 65536)]               # train mean and scalar std, in chunks
    mu = sum(Y[c].sum(0, dtype=np.float64) for c in ch) / len(TR)
    sd = np.sqrt(sum(((Y[c] - mu) ** 2).sum() for c in ch) / (len(TR) * Y.shape[1]))
    Y -= mu.astype(np.float32); Y /= np.float32(sd)                        # in place; one scalar scale keeps FVU's weighting
    Z = torch.from_numpy(Y)
    net = MLP(Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(TR)
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            loss = torch.nn.functional.mse_loss(net(*bags(b)), Z[b])
            opt.zero_grad(); loss.backward(); opt.step()
        v = fvu(net, Z, VA)
        print(f"{t:22} epoch {ep + 1:2}  val FVU {v:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if v < best: best, bad, best_state = v, 0, {k_: p.clone() for k_, p in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    net.load_state_dict(best_state)
    r = {"train": fvu(net, Z, TR), "val": best, "test": fvu(net, Z, TE), "epochs": ep + 1,
         "ridge test": ridge.get(t, {}).get("features + square", {}).get("test")}
    print(f"{t:22} MLP FVU train {r['train']:.3f}  test {r['test']:.3f}  (ridge test {r['ridge test']})", flush=True)
    res["taps"][t] = r
    json.dump(res, open(out, "w"), indent=1)                               # after every tap, so a crash keeps the rest
    if a.save:                                                             # decoded tokens, back in the tap's units
        P = np.empty(Y.shape, np.float16)
        with torch.no_grad():
            for s in range(0, N * 64, a.batch):
                r_ = np.arange(s, min(s + a.batch, N * 64))
                P[r_] = net(*bags(r_)).numpy() * sd + mu
        np.save(ROOT / f"lc0-probe/data/decoded{t.replace('/', '_')}.npy", P)
