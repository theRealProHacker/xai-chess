"""Reverse hybrid from the single-layer decoders: L5 = D5 + A(layers2-5(D2)), A fitted, the decoders frozen.

D2 and D5 are the side-to-move-frame square decoders of /encoder2/ln2 and /encoder5/ln1, each fitted to its own
layer's real stream. D2's output goes through Leela up to L5, and a combiner A maps it onto what D5 misses there.
With --no-d5 (control) A maps it to L5 itself, D5 left out. With --with-d5 A also reads D5's output, so D5 need not keep full weight.
A is a linear layer (ridge least squares), a Lasso, or an MLP, fitted on the validation positions, which no decoder
trained on (hyperparameters on a fifth of them); everything is reported on the test positions. The streams are
cached in --cache, so the combiners can be refitted without Leela.

  python lc0-probe/linear_combine.py
"""
import argparse, json, sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, torch
from onnx2torch import convert
from lc0_planes import encode_fens
torch.set_num_threads(2)
ap = argparse.ArgumentParser()
ap.add_argument("--with-d5", action="store_true", help="the combiner also reads D5's output")
ap.add_argument("--no-d5", action="store_true", help="control: A maps layers2-5(D2) to L5 itself, D5 left out")
ap.add_argument("--cache", default=str(Path(__file__).resolve().parent / "data/linear_combine_streams.pt"))
a = ap.parse_args()
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
N, K = 100000, 4000                                                        # cache size, test positions
T2, T5 = "/encoder2/ln2", "/encoder5/ln1"

te_pos = np.random.default_rng(0).random(N) < 0.2                          # the decoders' split
va_pos = ~te_pos & (np.random.default_rng(1).random(N) < 0.1)
VA, TE = np.flatnonzero(va_pos), np.flatnonzero(te_pos)[:K]   # all ~7,200 validation positions
want = set(VA) | set(TE)
fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:N]
white = [f if f.split()[1] == "w" else chess.Board(f).mirror().fen() for f in fens]   # the cache's frame
bits = {}
with open(ROOT / "lc0-probe/data/square_bits_n100000_nodes200000.jsonl") as f:
    for i, l in enumerate(f):
        if i in want: bits[i] = {chess.parse_square(s): np.array(b, np.int64) for s, b in json.loads(l)["squares"].items()}

class Dec:
    """A square decoder: features (side-to-move frame) plus the square one-hot -> its layer's token."""
    def __init__(self, path):
        c = torch.load(path, weights_only=False)
        self.n = len(c["keep"])
        self.col = np.full(30839, -1, np.int64); self.col[c["keep"]] = np.arange(self.n)
        s = c["state"]
        self.E, self.b, self.W, self.c = s["inp.weight"], s["bias"], s["out.weight"], s["out.bias"]
        self.mu, self.sd = torch.as_tensor(c["mu"]).float(), float(c["sd"])
    def __call__(self, P):
        flat, off = [], []
        for i in P:
            for t in range(64):
                off.append(sum(map(len, flat)))
                v = self.col[bits[i].get(t, np.empty(0, np.int64))]
                flat.append(np.append(v[v >= 0], self.n + t))
        h = torch.nn.functional.embedding_bag(torch.from_numpy(np.concatenate(flat)), self.E, torch.tensor(off), mode="sum")
        return (torch.nn.functional.gelu(h + self.b) @ self.W.T + self.c) * self.sd + self.mu

D2, D5 = [Dec(ROOT / f"lc0-probe/data/square_decoder_{t}.pt") for t in ("encoder2_ln2_n100000", "encoder5_ln1_n100000")]
inferred = onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))
seg = lambda i, o: convert(onnx.utils.Extractor(inferred).extract_model(i, o)).eval()
full, s25, tail = seg(["/input/planes"], [T5, "/output/wdl"]), seg([T2], [T5]), seg([T5], ["/output/wdl"])

def streams(P):
    """Per token: real L5, D5, layers2-5(D2); per board: real WDL."""
    out = {k: [] for k in ("y", "d5", "x2", "wdl")}
    with torch.no_grad():
        for s in range(0, len(P), 64):
            p = P[s:s + 64]
            y, w = full(torch.from_numpy(encode_fens([white[i] for i in p])))
            for k, x in (("y", y.reshape(-1, 256)), ("d5", D5(p)), ("x2", s25(D2(p))), ("wdl", w)):
                out[k].append(x.half())
            if s % 1280 == 0: log(s, "/", len(P))
    return {k: torch.cat(x) for k, x in out.items()}

if Path(a.cache).exists(): va, te = torch.load(a.cache)
else:
    va, te = streams(VA), streams(TE)
    torch.save((va, te), a.cache)
va, te = [{k: x.float() for k, x in d.items()} for d in (va, te)]
log(f"streams: {len(VA):,} validation, {len(TE):,} test positions")
inp = lambda d: torch.cat([d["d5"], d["x2"]], 1) if a.with_d5 else d["x2"]
base = lambda d: 0 if a.no_d5 else d["d5"]                                # what A adds onto
X, R = inp(va), va["y"] - base(va)
cut = len(X) * 4 // 5                                                      # inner split for hyperparameters (by board:
fvu = lambda Y, P: float(((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())   # 64 tokens per board, in order)

def ridge(X, R, lam):
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1).double()
    W = torch.linalg.solve(X1.T @ X1 + lam * len(X) * torch.eye(X1.shape[1], dtype=torch.double), X1.T @ R.double()).float()
    return lambda x: torch.cat([x, torch.ones(len(x), 1)], 1) @ W, float((W[:-1].abs() > 1e-8).float().mean())

def lasso(X, R, alpha):
    """min 1/(2n) |R - XW - b|^2 + alpha |W|_1, by FISTA on the 256x256 Gram matrix (centred, so b is the mean)."""
    xm, rm = X.mean(0), R.mean(0)
    Xc = (X - xm).double(); G, C = Xc.T @ Xc / len(X), Xc.T @ (R - rm).double() / len(X)
    step = 1 / float(torch.linalg.eigvalsh(G)[-1])
    W = Z = torch.zeros(X.shape[1], 256, dtype=torch.double); t = 1.0
    for _ in range(3000):
        U = Z - step * (G @ Z - C)
        Wn = U.sign() * (U.abs() - step * alpha).clamp(min=0)
        tn = (1 + (1 + 4 * t * t) ** 0.5) / 2
        Z, W, t = Wn + (t - 1) / tn * (Wn - W), Wn, tn
    W = W.float(); b = rm - xm @ W
    return lambda x: x @ W + b, float((W != 0).float().mean())

def mlp(X, R, hidden):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 256))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    c = len(X) * 4 // 5; Xt, Rt, Xv, Rv = X[:c], R[:c], X[c:], R[c:]           # early stopping on the last fifth
    best, state, bad = float("inf"), None, 0
    for ep in range(100):
        for i in torch.randperm(c).split(1024):
            loss = torch.nn.functional.mse_loss(net(Xt[i]), Rt[i]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): v = float(torch.nn.functional.mse_loss(net(Xv), Rv))
        if v < best: best, bad, state = v, 0, {k: p.clone() for k, p in net.state_dict().items()}
        else: bad += 1
        if bad == 3: break
    net.load_state_dict(state); net.eval()
    return lambda x: net(x).detach(), None

res = {"n val": len(VA), "n test": len(TE), "taps": [T2, T5], "with d5": a.with_d5, "no d5": a.no_d5, "combiners": {}}
grid = {"linear": (ridge, [1e-5, 1e-3, 1e-1]), "lasso": (lasso, [1e-4, 1e-3, 1e-2]), "mlp": (mlp, [512, 2048])}
fA = {}
for name, (fn, hs) in grid.items():
    tune = {}
    for h in hs:                                                          # pick on the inner split, refit on all of it
        f, _ = fn(X[:cut], R[:cut], h); tune[h] = fvu(R[cut:], f(X[cut:])); log(f"{name} {h:g}: inner FVU of D5's miss {tune[h]:.4f}")
    h = min(tune, key=tune.get)
    fA[name], dens = fn(X, R, h)
    res["combiners"][name] = {"picked": h, "inner FVU of D5's miss": tune, "nonzero weights": dens}

v = lambda w: (w[:, 0] - w[:, 2]).numpy()
cand = {"D5": te["d5"], "layers2-5(D2) alone": te["x2"]} | {f"{'' if a.no_d5 else 'D5 + '}A·D2 ({k})": base(te) + f(inp(te)) for k, f in fA.items()}
with torch.no_grad():
    for name, P in cand.items():
        w = tail(P)
        r = {"token FVU": fvu(te["y"], P), "WDL FVU": fvu(te["wdl"], w),
             "value r": float(np.corrcoef(v(te["wdl"]), v(w))[0, 1]), "median |dy|": float(np.median(np.abs(v(te["wdl"]) - v(w))))}
        if name.startswith("D5 +") and not a.no_d5: r["FVU of D5's miss"] = fvu(te["y"] - te["d5"], P - te["d5"])
        res[name] = r
        log(f"{name:24} " + "  ".join(f"{k} {x:.4f}" for k, x in r.items()))
json.dump(res, open(ROOT / f"lc0-probe/results_linear_combine{'_with_d5' if a.with_d5 else '_no_d5' if a.no_d5 else ''}.json", "w"), indent=1)
