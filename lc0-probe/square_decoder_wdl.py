"""square_decoder_mlp.py's decoder, scored on Leela's WDL instead of on the layer.

The decoder is trained as before (MSE on the tokens). For held-out positions all 64 decoded tokens
replace the real ones at the tap, the rest of the net runs, and the FVU is taken on /output/wdl
against the real net's WDL. Policy-only directions in the layer then cost nothing. Baseline: each
square's mean train token in place of the decoded one (the square alone). Only taps that carry the
whole state work: /attn_body/rehape and /encoderN/ln1, /encoderN/ln2 (inside the FFN the skip still
holds the real ln1).

  python lc0-probe/square_decoder_wdl.py --taps /attn_body/rehape,/encoder0/ln1,...,/encoder9/ln2
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
ap.add_argument("--taps", default="/attn_body/rehape," + ",".join(f"/encoder{i}/ln1" for i in range(10))
                + "," + ",".join(f"/encoder{i}/ln2" for i in range(10)))
ap.add_argument("--min", type=int, default=20, help="positive tokens in train needed to use a bit")
ap.add_argument("--hidden", type=int, default=2048)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--threads", type=int, default=4, help="CPU threads, leaving the rest of the machine usable")
ap.add_argument("--penalize", default="", help="feature families (label's first part, comma-separated) whose input "
                "weights get a group-lasso penalty: lam * sum of their rows' norms")
ap.add_argument("--pen-frac", type=float, default=0.1, help="lam such that the penalty starts at this fraction of the first batch's MSE")
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
te_pos = rng.random(N) < 0.2                                               # same split as square_probe.py
te = np.repeat(te_pos, 64)
va = np.repeat(~te_pos & (np.random.default_rng(1).random(N) < 0.1), 64)   # early-stopping slice of train
tr = ~te & ~va

keep = np.flatnonzero(np.bincount(bit[~te[tok]], minlength=L) >= a.min)
col = np.full(L, -1, np.int32); col[keep] = np.arange(len(keep))             # int32 throughout: 150k positions are ~350M entries
k = col[bit] >= 0
tok, idx = tok[k], col[bit[k]]; del bit, k
tok = np.concatenate([tok, np.arange(N * 64, dtype=np.int32)])             # square one-hot as extra inputs
idx = np.concatenate([idx, (len(keep) + np.arange(N * 64) % 64).astype(np.int32)])
o = np.argsort(tok, kind="stable"); tok = tok[o]; idx = idx[o]; del o
start = np.searchsorted(tok, np.arange(N * 64 + 1)); n_idx = len(idx); del tok   # token t's inputs: idx[start[t]:start[t+1]]
D_in = len(keep) + 64
fam = set(a.penalize.split(",")) - {""}
pen = torch.from_numpy(np.flatnonzero([schema["labels"][b].split(".")[0] in fam for b in keep]))   # decoder rows penalized
tag = "_pen_" + "_".join(sorted(fam)) if fam else ""
print(f"{N:,} positions, {len(keep):,} bits used, {n_idx / (N * 64):.1f} active inputs per token"
      + (f", {len(pen)} penalized ({a.penalize})" if fam else ""), flush=True)

def bags(rows_):
    """EmbeddingBag inputs for a batch of token ids."""
    lo, hi = start[rows_], start[rows_ + 1]
    flat = np.concatenate([idx[l:h] for l, h in zip(lo, hi)]).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(hi - lo)[:-1]])
    return torch.from_numpy(flat), torch.from_numpy(off)

net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
m = onnx.load(net_path)
taps = a.taps.split(",")
for t in taps + ["/output/wdl"]: m.graph.output.append(onnx.ValueInfoProto(name=t))
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
WDL = np.concatenate([sess.run(["/output/wdl"], {"/input/planes": encode_fens(fens[s:s + 256])})[0]
                      for s in range(0, N, 256)])                          # (N, 3), the real net's WDL

def tail(t):
    """Session running the net from tap t to /output/wdl; t must be the only input it needs."""
    sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(net_path))).extract_model([t], ["/output/wdl"])
    assert [i.name for i in sub.graph.input] == [t], [i.name for i in sub.graph.input]
    return ort.InferenceSession(sub.SerializeToString(), so, providers=["CPUExecutionProvider"])

def wdl(tail_sess, t, T):
    """(P, 64, D) tokens at tap t -> (P, 3) WDL. Some taps are (P*64, D) in the graph."""
    flat = len(tail_sess.get_inputs()[0].shape) == 2
    return np.concatenate([tail_sess.run(["/output/wdl"], {t: T[s:s + 256].reshape(-1, T.shape[2]) if flat else T[s:s + 256]})[0]
                           for s in range(0, len(T), 256)])

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
        torch.nn.init.normal_(self.inp.weight, std=(n_idx / (N * 64)) ** -0.5)   # unit pre-activations
        self.bias = torch.nn.Parameter(torch.zeros(a.hidden))
        self.out = torch.nn.Linear(a.hidden, d_out)
    def forward(self, flat, off):
        return self.out(torch.nn.functional.gelu(self.inp(flat, off) + self.bias))

def fvu(net, Z, rows_):
    """FVU of net on Z[rows_], in chunks. Invariant to Z's shift and scale."""
    chunks = [rows_[s:s + a.batch] for s in range(0, len(rows_), a.batch)]
    mean = sum(Z[c].sum(0) for c in chunks) / len(rows_)
    sse = sst = 0.0
    with torch.no_grad():
        for c in chunks:
            sse += float(((Z[c] - net(*bags(c))) ** 2).sum()); sst += float(((Z[c] - mean) ** 2).sum())
    return sse / sst

def fvu_wdl(P):
    """FVU of predicted WDL P on the held-out positions, over the 3 outputs."""
    Y = WDL[te_pos]
    return float(((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

out = ROOT / f"lc0-probe/results_square_decoder_wdl_n{N}.json"
res = json.load(open(out)) if out.exists() else {}
res.update({"n": N, "bits": int(len(keep)), "hidden": a.hidden, "lr": a.lr, "batch": a.batch})
res.setdefault("taps", {})
TR, VA, TE = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
for t in taps:
    ts = tail(t)
    Y = tokens(t)
    shape = (N, 64, Y.shape[1])
    err = np.abs(wdl(ts, t, Y.reshape(shape)[te_pos]) - WDL[te_pos]).max()   # the cut net matches the full one
    assert err < 1e-4, (t, err)
    ch = [TR[s:s + 65536] for s in range(0, len(TR), 65536)]               # train mean and scalar std, in chunks
    mu = sum(Y[c].sum(0, dtype=np.float64) for c in ch) / len(TR)
    sd = np.sqrt(sum(((Y[c] - mu) ** 2).sum() for c in ch) / (len(TR) * Y.shape[1]))
    sq_mean = np.stack([Y[TR[TR % 64 == s]].mean(0) for s in range(64)])  # (64, D), each square's mean train token
    Y -= mu.astype(np.float32); Y /= np.float32(sd)                        # in place; one scalar scale keeps FVU's weighting
    Z = torch.from_numpy(Y)
    net = MLP(Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    best, best_state, bad, t0, lam = np.inf, None, 0, time.time(), None
    norms = lambda: net.inp.weight[pen].norm(dim=1)                       # penalized rows' input-weight norms
    for ep in range(a.epochs):
        perm = rng.permutation(TR)
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            loss = torch.nn.functional.mse_loss(net(*bags(b)), Z[b])
            if fam:
                if lam is None: lam = a.pen_frac * float(loss) / float(norms().sum())
                loss = loss + lam * norms().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        v = fvu(net, Z, VA)
        print(f"{t:22} epoch {ep + 1:2}  val FVU {v:.3f}" + (f"  penalized row norm mean {float(norms().mean()):.3f}" if fam else "")
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        if v < best:
            best, bad, best_state = v, 0, {k_: p.clone() for k_, p in net.state_dict().items()}
            torch.save({"state": best_state, "mu": mu, "sd": sd, "keep": keep},     # decoded token = net(bits) * sd + mu;
                       ROOT / f"lc0-probe/data/square_decoder{t.replace('/', '_')}_n{N}{tag}.pt")   # every best epoch, so a kill keeps it
        else:
            bad += 1
            if bad >= a.patience: break
    net.load_state_dict(best_state)
    with torch.no_grad():                                                  # decoded test tokens, back in the tap's units
        D = np.concatenate([net(*bags(TE[s:s + a.batch])).numpy() for s in range(0, len(TE), a.batch)])
    D = (D * sd + mu).astype(np.float32).reshape(-1, 64, Y.shape[1])
    r = {"token test": fvu(net, Z, TE), "wdl test": fvu_wdl(wdl(ts, t, D)),
         "wdl square only": fvu_wdl(wdl(ts, t, np.broadcast_to(sq_mean.astype(np.float32), D.shape).copy())),
         "epochs": ep + 1}
    print(f"{t:22} FVU token {r['token test']:.3f}  WDL {r['wdl test']:.3f}  (square only {r['wdl square only']:.3f})", flush=True)
    if fam: r.update({"penalize": a.penalize, "lam": lam, "penalized row norm mean": float(norms().mean())})
    res["taps"][t + tag] = r
    json.dump(res, open(out, "w"), indent=1)                               # after every tap, so a crash keeps the rest
    del Y, Z, D
