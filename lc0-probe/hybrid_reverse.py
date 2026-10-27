"""Hybrid decoder trained in reverse: L5 = D5 + layers3-5(D2 + layers1-2(D0)), fitted D5 first, then D2, then D0.

The same model as hybrid_decoder.py (S0 = D0, S2 = layers1-2(S0) + D2, S5 = layers3-5(S2) + D5); only the order
differs, so the order decides which layer gets the credit. Leela's layers end in a layer norm, so what goes through
them is never small: with D0 and D2 at zero, layers3-5(layers1-2(0)) is a constant c per token (norm 11.6 against
the real L5's 12.2). D5 is therefore fitted to the real L5 minus c; D2 and D0 start at exactly zero output (zeroed
out layer, mu 0), so each stage starts where the one before it ended. Stages 2 and 3 are fitted to the real L5 in
stage 1's scale, backpropagating through layers 3-5 (D2) and layers 1-5 (D0).

Cross-fitting as in hybrid_decoder.py: D5 and D2 each get --folds fold models, and a train board's frozen D5 / D2
output in the next stage comes from the model that did not see its fold. Splits, colourings, inputs and the saved
checkpoint format are hybrid_decoder.py's, so its readers (square_shap.py --hybrid) load the result unchanged.

--stages 1 fits only the final D5 and reports its curve: the optimizer sweep.

  python lc0-probe/hybrid_reverse.py --n 20000 --opt adam --lr 1e-3
"""
import argparse, json, shutil, sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, torch
from onnx2torch import convert
from lc0_planes import encode_fens
from colour_frame import label_map
import optim_extra

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000, help="positions, a prefix of the 100000-position feature cache")
ap.add_argument("--taps", default="/encoder0/ln2,/encoder2/ln2,/encoder5/ln2")
ap.add_argument("--hidden", default="2048,1024,1024", help="per tap, as in hybrid_decoder.py")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--stages", type=int, default=3)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--batch", type=int, default=256, help="boards per batch")
ap.add_argument("--opt", default="adam", help="adam, adamw, nadam, radam, adamax, adagrad, rmsprop, lion, adabound")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--min", type=int, default=20, help="positive train tokens needed to use a bit (in either colouring)")
ap.add_argument("--tag", default="", help="suffix for the run's files")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
rng = np.random.default_rng(0)
torch.manual_seed(0)
torch.set_num_threads(a.threads)
dev = torch.device(a.device)
TAPS, HID = a.taps.split(","), [int(h) for h in a.hidden.split(",")]
assert len(TAPS) == 3
N = a.n
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
mse = torch.nn.functional.mse_loss

# --- boards, splits, inputs: as hybrid_decoder.py
fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:N]
white = [f if f.split()[1] == "w" else chess.Board(f).mirror().fen() for f in fens]
boards = white + [chess.Board(f).mirror().fen() for f in white]
M = np.array(label_map(), np.int32)
L = len(M)
te_pos = (np.random.default_rng(0).random(100000) < 0.2)[:N]
va_pos = ~te_pos & (np.random.default_rng(1).random(100000) < 0.1)[:N]
tr_pos = ~te_pos & ~va_pos
fold = np.random.default_rng(2).integers(0, max(a.folds, 1), N)
toks, bits = [], []
with open(ROOT / "lc0-probe/data/square_bits_n100000_nodes200000.jsonl") as f:
    for i, l in enumerate(f):
        if i == N: break
        sq = json.loads(l)["squares"]
        toks.append(np.repeat(np.array([i * 64 + chess.parse_square(s) for s in sq], np.int32), [len(v) for v in sq.values()]))
        bits.append(np.fromiter((b for v in sq.values() for b in v), np.int32))
tok, bit = np.concatenate(toks), np.concatenate(bits); del toks, bits
cnt = np.bincount(bit[tr_pos[tok // 64]], minlength=L)
keep = np.flatnonzero((cnt + cnt[M]) >= a.min)
col = np.full(L, -1, np.int32); col[keep] = np.arange(len(keep))
SQ, STM = len(keep), len(keep) + 64
c0, c1 = col[bit], col[M[bit]]; del bit
t1 = (tok // 64 + N) * 64 + tok % 64
m0, m1 = c0 >= 0, c1 >= 0
all_t = np.arange(2 * N * 64, dtype=np.int32)
real_sq = np.where(all_t < N * 64, all_t % 64, (all_t % 64) ^ 56)
T_ = np.concatenate([tok[m0], t1[m1], all_t, all_t[N * 64:]])
I_ = np.concatenate([c0[m0], c1[m1], (SQ + real_sq).astype(np.int32), np.full(N * 64, STM, np.int32)])
del tok, t1, c0, c1, m0, m1, real_sq
o = np.argsort(T_, kind="stable")
start = np.concatenate([[0], np.cumsum(np.bincount(T_, minlength=2 * N * 64))]); del T_
idx = torch.from_numpy(I_[o]); del o, I_
D_in = STM + 1
log(f"{N:,} positions x 2 colourings, {len(keep):,} bits, {len(idx) / (2 * N * 64):.1f} inputs per token; "
    f"{a.opt} lr {a.lr:g} wd {a.wd:g}")

def bags(E):
    flat = torch.cat([idx[l:h] for l, h in zip(start[E * 64], start[E * 64 + 64])])
    t = (E[:, None] * 64 + np.arange(64)).ravel()
    off = np.cumsum(np.concatenate([[0], (start[t + 1] - start[t])[:-1]])).astype(np.int32)
    return flat.to(dev, non_blocking=True), torch.from_numpy(off).to(dev, non_blocking=True)

class MLP(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.inp = torch.nn.EmbeddingBag(D_in, hidden, mode="sum")
        torch.nn.init.normal_(self.inp.weight, std=(len(idx) / (2 * N * 64)) ** -0.5)
        self.bias = torch.nn.Parameter(torch.zeros(hidden))
        self.out = torch.nn.Linear(hidden, 256)
    def forward(self, flat, off):
        return self.out(torch.nn.functional.gelu(self.inp(flat, off) + self.bias))

# --- Leela: the real L5 stream (target), values, the scales of L0 and L2, and the zero-decoder constants
net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
inferred = onnx.shape_inference.infer_shapes(onnx.load(net_path))
def seg(i, o_):
    m = convert(onnx.utils.Extractor(inferred).extract_model(i, o_)).eval().to(dev)
    for p in m.parameters(): p.requires_grad_(False)
    return m
s12, s35, tail = [seg([t], [u]) for t, u in zip(TAPS, TAPS[1:] + ["/output/wdl"])]
v_of = lambda w: w[:, 0] - w[:, 2]
ST = ROOT / f"lc0-probe/data/hybrid_rev_streams_n{N}"; ST.mkdir(exist_ok=True)   # shared by the sweep's runs
MM = ROOT / f"lc0-probe/data/hybrid_rev_tmp_n{N}{a.tag}"; MM.mkdir(exist_ok=True)
big = lambda d, name, n, mode="w+": np.lib.format.open_memmap(d / f"{name}.npy", mode, np.float16, (n, 64, 256))
if (ST / "V.npy").exists():
    Y5, V = big(ST, "Y5", N, "r"), np.load(ST / "V.npy")
    sd0, sd2 = torch.load(ST / "scales.pt")
else:
    full = seg(["/input/planes"], TAPS + ["/output/wdl"])
    Y5, V = big(ST, "Y5", N), np.empty(2 * N, np.float32)
    ss = np.zeros((2, 2))                                                  # sum, sum of squares of L0 and L2
    with torch.no_grad():
        for s in range(0, N, 512):
            e = min(s + 512, N)
            y0, y2, y5, w = full(torch.from_numpy(encode_fens(white[s:e])).to(dev))
            Y5[s:e] = y5.reshape(-1, 64, 256).half().cpu().numpy()
            V[s:e] = v_of(w).cpu().numpy()
            V[N + s:N + e] = v_of(full(torch.from_numpy(encode_fens(boards[N + s:N + e])).to(dev))[-1]).cpu().numpy()
            for j, y in enumerate((y0, y2)): ss[j] += [float(y.sum()), float((y.double() ** 2).sum())]
    n_el = N * 64 * 256
    sd0, sd2 = [float(np.sqrt(q / n_el - (p / n_el) ** 2)) for p, q in ss]
    Y5.flush(); torch.save((sd0, sd2), ST / "scales.pt")
    np.save(ST / "V.npy", V)                                               # written last: marks the streams complete
with torch.no_grad():
    Z2 = s12(torch.zeros(64, 256, device=dev))                            # L2 with D0 at zero
    C = s35(Z2)                                                            # L5 with D0 and D2 at zero
log(f"real streams read; L0 sd {sd0:.3f}, L2 sd {sd2:.3f}; constant c: token norm {float(C.norm(dim=-1).mean()):.2f}")
tile = lambda X, B: X.repeat(B, 1)                                         # (64, 256) -> (B*64, 256)
y5c = lambda E: torch.from_numpy(Y5[E % N].astype(np.float32)).to(dev).reshape(-1, 256) - tile(C, len(E))

both = lambda P: np.concatenate([P, P + N])
E_va, TE = both(np.flatnonzero(va_pos)), np.flatnonzero(te_pos)
P_all = np.flatnonzero(tr_pos)
res = {"n": N, "taps": TAPS, "hidden": HID, "folds": a.folds, "opt": a.opt, "lr": a.lr, "wd": a.wd, "curves": {}}

nk = ST / "norm5.pt"                                                       # stage 1's scale; all stages' loss uses it
if not nk.exists():
    Zs = y5c(both(rng.choice(P_all, min(4000, len(P_all)), replace=False)))
    torch.save((Zs.mean(0).cpu(), float(Zs.std())), nk); del Zs
mu5, sd5 = torch.load(nk); mu5 = mu5.to(dev)
log(f"target L5 - c: sd {sd5:.4f}")

def fit(h, P_tr, pred, label, zero=False):
    """Train one decoder D on positions P_tr (both colourings): pred(D, E) is the model's L5 minus c for boards E,
    in stage 1's scale. Early stopping on E_va; checkpointed every epoch."""
    D = MLP(h).to(dev)
    if zero: torch.nn.init.zeros_(D.out.weight); torch.nn.init.zeros_(D.out.bias)
    loss_of = lambda E: mse(pred(D, E), (y5c(E) - mu5) / sd5)
    opt = optim_extra.make(a.opt, D.parameters(), a.lr, a.wd)
    def val():
        with torch.no_grad():
            return float(np.mean([float(loss_of(E_va[s:s + 256])) for s in range(0, len(E_va), 256)]))
    E_tr = both(P_tr)
    ck = MM / f"fit_{label.replace(' ', '_')}.pt"
    if ck.exists():
        c = torch.load(ck, weights_only=False)
        D.load_state_dict(c["D"]); opt.load_state_dict(c["opt"])
        best, state, bad, ep0, done, curve = c["best"], c["state"], c["bad"], c["ep"], c["done"], c["curve"]
        log(f"{label} resumed after epoch {ep0}, best val MSE {best:.4f}" + (", done" if done else ""))
    else:
        best, state, bad, ep0, done = val(), {q: p.clone() for q, p in D.state_dict().items()}, 0, 0, False
        curve = [best]
        log(f"{label} start val MSE {best:.4f}")
    t0 = time.time()
    save = lambda ep, done: torch.save({"D": D.state_dict(), "opt": opt.state_dict(), "best": best, "state": state,
                                        "bad": bad, "ep": ep, "done": done, "curve": curve}, ck)
    for ep in range(ep0, 0 if done else a.epochs):
        perm = rng.permutation(E_tr)
        for s in range(0, len(perm), a.batch):
            loss = loss_of(perm[s:s + a.batch])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep == ep0 and s // a.batch == 2: log(f"{label}: {(time.time() - t0) / 3:.1f}s per batch of {a.batch}")
        v = val(); curve.append(v)
        log(f"{label} epoch {ep + 1}  val MSE {v:.4f}  ({time.time() - t0:.0f}s)")
        if v < best: best, bad, state = v, 0, {q: p.clone() for q, p in D.state_dict().items()}
        else: bad += 1
        save(ep + 1, bad >= a.patience or ep + 1 == a.epochs)
        if bad >= a.patience: break
    D.load_state_dict(state)
    for p in D.parameters(): p.requires_grad_(False)
    res["curves"][label] = curve
    return D, best

def crossfit(h, pred, raw, name, label):
    """Final model plus one per fold; writes memmap `name` holding each board's raw decoder output from the model that
    did not see its fold (val/test boards: the final model). Returns the final model."""
    D, best = fit(h, P_all, pred, f"{label} final", zero=label != "D5")
    res[label] = {"val MSE": best}
    owner = np.full(2 * N, -1); owner[both(P_all)] = np.concatenate([fold[P_all]] * 2)
    Ds = {-1: D}
    for j in range(a.folds):
        Ds[j] = fit(h, P_all[fold[P_all] != j], pred, f"{label} fold {j}", zero=label != "D5")[0]
    if not (MM / f"{name}.done").exists():
        G = big(MM, name, 2 * N)
        with torch.no_grad():
            for s in range(0, 2 * N, 512):
                E = np.arange(s, min(s + 512, 2 * N))
                out = torch.empty(len(E) * 64, 256, device=dev)
                for j in np.unique(owner[E]):
                    m = np.flatnonzero(owner[E] == j); rows = torch.from_numpy((m[:, None] * 64 + np.arange(64)).ravel()).to(dev)
                    out[rows] = raw(Ds[j], E[m])
                G[E] = out.reshape(-1, 64, 256).half().cpu().numpy()
        G.flush(); (MM / f"{name}.done").touch()
    return D, big(MM, name, 2 * N, "r")

g = lambda G, E: torch.from_numpy(G[E].astype(np.float32)).to(dev).reshape(-1, 256)
raw5 = lambda D, E: D(*bags(E)) * sd5 + mu5                                # L5 minus c
raw2 = lambda D, E: D(*bags(E)) * sd2                                      # zero at init
raw0 = lambda D, E: D(*bags(E)) * sd0
norm = lambda X: (X - mu5) / sd5

# --- stage 1: D5 from the features alone
pred1 = lambda D, E: D(*bags(E))
if a.stages == 1:
    D5, best = fit(HID[2], P_all, pred1, "D5 final")
    res["D5"] = {"val MSE": best}
else:
    D5, G5 = crossfit(HID[2], pred1, raw5, "G5", "D5")
    # --- stage 2: D2 through layers 3-5, onto the frozen D5
    pred2 = lambda D, E: norm(g(G5, E) + s35(raw2(D, E) + tile(Z2, len(E))) - tile(C, len(E)))
    if a.stages == 2:
        D2, best = fit(HID[1], P_all, pred2, "D2 final", zero=True); res["D2"] = {"val MSE": best}
    else:
        D2, G2 = crossfit(HID[1], pred2, raw2, "G2", "D2")
        # --- stage 3: D0 through layers 1-5, onto the frozen D5 and D2 (final model only, as the forward run's last stage)
        pred3 = lambda D, E: norm(g(G5, E) + s35(g(G2, E) + s12(raw0(D, E))) - tile(C, len(E)))
        D0, best = fit(HID[0], P_all, pred3, "D0 final", zero=True)
        res["D0"] = {"val MSE": best}

# --- held-out value, both colourings: D5 alone, D5 + D2, D5 + D2 + D0 (each with the others at zero)
dec = {"D5": D5} | ({"D2": D2} if a.stages >= 2 else {}) | ({"D0": D0} if a.stages >= 3 else {})
with torch.no_grad():
    for upto in range(1, len(dec) + 1):
        on = list(dec)[:upto]
        vals = []
        for E in (TE, TE + N):
            out = []
            for s in range(0, len(E), 256):
                e = E[s:s + 256]
                S0 = raw0(D0, e) if "D0" in on else torch.zeros(len(e) * 64, 256, device=dev)
                S2 = s12(S0) + (raw2(D2, e) if "D2" in on else 0)
                out.append(v_of(tail(s35(S2) + raw5(D5, e))).cpu())
            vals.append(torch.cat(out).numpy())
        name = "value " + " + ".join(on)                                     # not res["D5"], the fit's entry
        r = {"r": float(np.corrcoef(V[TE], vals[0])[0, 1]), "median |dy|": float(np.median(np.abs(V[TE] - vals[0]))),
             "r Black to move": float(np.corrcoef(V[TE + N], vals[1])[0, 1]),
             "median |dy| Black to move": float(np.median(np.abs(V[TE + N] - vals[1]))),
             "colour gap median": float(np.median(np.abs(vals[0] - vals[1])))}
        res[name] = r
        log(f"{name:20} r {r['r']:.4f} (Black to move {r['r Black to move']:.4f})  median |dy| {r['median |dy|']:.4f}  "
            f"colour gap {r['colour gap median']:.4f}")
json.dump(res, open(ROOT / f"lc0-probe/results_hybrid_reverse_n{N}{a.tag}.json", "w"), indent=1)
if a.stages == 3:                                                          # hybrid_decoder.py's format, taps in order
    z = lambda D, mu, sd, h, tap: {"state": {q: p.cpu() for q, p in D.state_dict().items()}, "mu": mu, "sd": sd,
                                   "hidden": h, "tap": tap}
    torch.save({"keep": keep, "sq row": SQ, "stm row": STM, "taps": TAPS,
                "decoders": [z(D0, np.zeros(256, np.float32), sd0, HID[0], TAPS[0]),
                             z(D2, np.zeros(256, np.float32), sd2, HID[1], TAPS[1]),
                             z(D5, mu5.cpu().numpy(), sd5, HID[2], TAPS[2])]},
               ROOT / f"lc0-probe/data/hybrid_reverse_n{N}{a.tag}.pt")
shutil.rmtree(MM)
