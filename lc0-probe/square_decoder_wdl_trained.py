"""square_decoder_wdl.py's decoder, fine-tuned on the layer and the WDL together.

Same per-square MLP decoder (features + square -> token). Phase 1 trains it on the tokens alone, as
square_decoder_wdl.py does (kept in data/decoder_<tap>.pt). Phase 2 fine-tunes with
tok * token FVU + lam * WDL FVU, where the WDL comes from running all 64 decoded tokens of a position
through the frozen rest of the net (ONNX converted to torch). Reported: FVU on the tokens and on
/output/wdl, held-out positions, after each phase.

  python lc0-probe/square_decoder_wdl_trained.py --tap /encoder5/ln1 --lam 1
"""
import argparse, json, subprocess, sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, onnxruntime as ort, torch
from onnx2torch import convert
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--nodes", type=int, default=200000)
ap.add_argument("--tap", default="/encoder5/ln1")
ap.add_argument("--min", type=int, default=20, help="positive tokens in train needed to use a bit")
ap.add_argument("--hidden", type=int, default=2048)
ap.add_argument("--lam", type=float, default=1.0, help="weight of WDL FVU next to token FVU in phase 2")
ap.add_argument("--tok", type=float, default=1.0, help="weight of token FVU in phase 2; 0 = WDL alone after the pretraining")
ap.add_argument("--epochs", type=int, default=5, help="phase 2")
ap.add_argument("--pre-epochs", type=int, default=30, help="phase 1, tokens only")
ap.add_argument("--patience", type=int, default=3, help="phase 1 early stopping")
ap.add_argument("--batch", type=int, default=64, help="positions per step (x64 tokens)")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--threads", type=int, default=4, help="CPU threads, leaving the rest of the machine usable")
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
te = rng.random(N) < 0.2                                                   # same split as square_probe.py
va = ~te & (np.random.default_rng(1).random(N) < 0.1)                      # early-stopping slice of train
tr = ~te & ~va

keep = np.flatnonzero(np.bincount(bit[~np.repeat(te, 64)[tok]], minlength=L) >= a.min)
col = np.full(L, -1); col[keep] = np.arange(len(keep))
k = col[bit] >= 0
tok, idx = tok[k], col[bit[k]]
sq = np.arange(N * 64) % 64                                                # square one-hot as extra inputs
tok = np.concatenate([tok, np.arange(N * 64)]); idx = np.concatenate([idx, len(keep) + sq])
o = np.argsort(tok, kind="stable"); tok, idx = tok[o], idx[o]
start = np.searchsorted(tok, np.arange(N * 64 + 1))                       # token t's inputs: idx[start[t]:start[t+1]]
D_in = len(keep) + 64
print(f"{N:,} positions, {len(keep):,} bits used, {len(idx) / (N * 64):.1f} active inputs per token", flush=True)

def bags(pos):
    """EmbeddingBag inputs for all 64 squares of each position in pos, position-major."""
    t = (pos[:, None] * 64 + np.arange(64)).ravel()
    lo, hi = start[t], start[t + 1]
    flat = np.concatenate([idx[l:h] for l, h in zip(lo, hi)])
    off = np.concatenate([[0], np.cumsum(hi - lo)[:-1]])
    return torch.from_numpy(flat), torch.from_numpy(off)

net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
m = onnx.load(net_path)
for t in [a.tap, "/output/wdl"]: m.graph.output.append(onnx.ValueInfoProto(name=t))
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
Y, WDL = [], []
for s in range(0, N, 256):
    y, w = sess.run([a.tap, "/output/wdl"], {"/input/planes": encode_fens(fens[s:s + 256])})
    Y.append(y.reshape(-1, 64, y.shape[-1])); WDL.append(w)
Y, WDL = np.concatenate(Y), np.concatenate(WDL)                            # (N, 64, D), (N, 3)

sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(net_path))).extract_model([a.tap], ["/output/wdl"])
assert [i.name for i in sub.graph.input] == [a.tap]
flat_in = len(sub.graph.input[0].type.tensor_type.shape.dim) == 2          # some taps are (P*64, D) in the graph
tail = convert(sub).eval()
for p in tail.parameters(): p.requires_grad_(False)
def run_tail(T):
    """(P, 64, D) tokens -> (P, 3) WDL, differentiable."""
    return tail(T.reshape(-1, T.shape[2]) if flat_in else T)
with torch.no_grad():                                                      # the torch tail matches onnxruntime
    err = float((run_tail(torch.from_numpy(Y[:256])) - torch.from_numpy(WDL[:256])).abs().max())
assert err < 1e-4, err
print(f"tail check: max |diff| {err:.1e}", flush=True)

TR, VA, TE = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
mu = Y[TR].mean((0, 1)); sd = Y[TR].std()                                  # decoder works in standardised units
Z = torch.from_numpy((Y - mu) / sd)
mu_t, sd_t = torch.from_numpy(mu), float(sd)
W_t = torch.from_numpy(WDL)
w_var = float(((WDL[TR] - WDL[TR].mean(0)) ** 2).mean())                   # puts the WDL loss on FVU scale, like the token loss

class MLP(torch.nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.inp = torch.nn.EmbeddingBag(D_in, a.hidden, mode="sum")       # = Linear on the sparse 0/1 input
        torch.nn.init.normal_(self.inp.weight, std=(len(idx) / (N * 64)) ** -0.5)   # unit pre-activations
        self.bias = torch.nn.Parameter(torch.zeros(a.hidden))
        self.out = torch.nn.Linear(a.hidden, d_out)
    def forward(self, pos):
        """(P, 64, D) standardised tokens."""
        return self.out(torch.nn.functional.gelu(self.inp(*bags(pos)) + self.bias)).reshape(len(pos), 64, -1)

def wdl_of(z):
    """Standardised decoded tokens -> WDL through the rest of the net."""
    return run_tail(z * sd_t + mu_t)

def evaluate(net, pos):
    """Token FVU (as square_decoder_wdl.py: one mean over all tokens) and WDL FVU on positions pos."""
    P, T = [], []
    with torch.no_grad():
        for s in range(0, len(pos), 256):
            z = net(pos[s:s + 256]); T.append(z.numpy()); P.append(wdl_of(z).numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    W, Zp = WDL[pos], Z[pos].numpy()
    return (float(((Zp - T) ** 2).sum() / ((Zp - Zp.mean((0, 1))) ** 2).sum()),
            float(((W - P) ** 2).sum() / ((W - W.mean(0)) ** 2).sum()))

def train(net, epochs, tok_w, lam, patience, name):
    """Adam on tok_w * token MSE + lam * WDL MSE / w_var; keeps the epoch with the best val of the same sum of FVUs."""
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    for ep in range(epochs):
        perm = rng.permutation(TR)
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            z = net(b)
            loss = tok_w * torch.nn.functional.mse_loss(z, Z[b])
            if lam: loss = loss + lam * torch.nn.functional.mse_loss(wdl_of(z), W_t[b]) / w_var
            opt.zero_grad(); loss.backward(); opt.step()
        vt, vw = evaluate(net, VA)
        print(f"{a.tap:16} {name} epoch {ep + 1:2}  val FVU token {vt:.3f}  WDL {vw:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if tok_w * vt + lam * vw < best: best, bad, best_state = tok_w * vt + lam * vw, 0, {k_: p.clone() for k_, p in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(best_state)

net = MLP(Y.shape[2])
ckpt = ROOT / f"lc0-probe/data/decoder{a.tap.replace('/', '_')}_n{N}.pt"
if ckpt.exists(): net.load_state_dict(torch.load(ckpt))
else:
    train(net, a.pre_epochs, 1.0, 0.0, a.patience, "tokens")
    torch.save(net.state_dict(), ckpt)
r = {"n": N, "tap": a.tap, "tok": a.tok, "lam": a.lam, "epochs": a.epochs, "hidden": a.hidden, "lr": a.lr, "batch": a.batch}
r["tokens only"] = dict(zip(["token test", "wdl test"], evaluate(net, TE)))
print(f"{a.tap:16} tokens only   test FVU token {r['tokens only']['token test']:.3f}  WDL {r['tokens only']['wdl test']:.3f}", flush=True)
train(net, a.epochs, a.tok, a.lam, a.epochs, "tokens + WDL")
r["tokens + wdl"] = dict(zip(["token test", "wdl test"], evaluate(net, TE)))
print(f"{a.tap:16} tokens + WDL  test FVU token {r['tokens + wdl']['token test']:.3f}  WDL {r['tokens + wdl']['wdl test']:.3f}", flush=True)
name = f"{a.tap.replace('/', '_')}_tok{a.tok:g}_lam{a.lam:g}_ep{a.epochs}"
torch.save(net.state_dict(), ROOT / f"lc0-probe/data/decoder_wdl{name}_n{N}.pt")
json.dump(r, open(ROOT / f"lc0-probe/results_square_decoder_wdl_trained{name}.json", "w"), indent=1)
