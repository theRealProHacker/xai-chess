"""Deepest first, earlier layers through end MLPs: L5 = D5 + M2(F2) + M0(F0).

D5 is a hybrid_reverse.py run's final L5 decoder (output cached by its --d0-from refit, plus the constant c). F2 is
Leela's L5 computed by layers 3-5 from the L2-only decoder's L2 (hybrid_decoder.py, frozen); F0 is Leela's L5 computed
by layers 1-5 from the forward run's L0 decoder (frozen). M2 reads F2 per token and is fitted to what D5 leaves of the
real L5; M0 reads F0 and is fitted to what D5 + M2 leave (loss in hybrid_reverse.py's L5 scale). Each M also reads
the prediction it corrects (D5, then D5 + M2), so it can cancel that prediction's errors; --no-pred: F alone. Held-out table as
hybrid_fvu.py: L5, WDL and value FVU, value r, for D5, D5 + M2, D5 + M2 + M0.

  python lc0-probe/hybrid_endmlp.py --games --n 100000
"""
import argparse, json, sys, time, types
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
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--games", action="store_true")
ap.add_argument("--fwd", default="lc0-probe/data/hybrid_decoder_games_n100000_adamw.pt", help="forward run: its D0")
ap.add_argument("--l2", default="lc0-probe/data/hybrid_decoder_games_n100000_adamw_l2only.pt", help="L2-only decoder")
ap.add_argument("--cache", default="lc0-probe/data/hybrid_rev_tmp_games_n100000_adamw_d0", help="cached G5")
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--opt", default="adamw")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--no-pred", action="store_true", help="the end MLPs read F only, not the prediction they correct")
ap.add_argument("--gate", action="store_true", help="M = sigmoid(g(F, prediction)) * h(F): the prediction only scales the term")
ap.add_argument("--penalty", type=float, default=0, help="lambda * mean square of the correction (L5 scale), added to the loss")
ap.add_argument("--resume", action="store_true", help="continue from the checkpoint written after each epoch")
a = ap.parse_args()
torch.manual_seed(0); rng = np.random.default_rng(0)
torch.set_num_threads(a.threads)
N, G = a.n, "_games" if a.games else ""
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)

# --- boards, splits, decoder inputs: as hybrid_reverse.py
fens = [l for l in (ROOT / f"sf-probe/data/fens{G}.txt").read_text().split("\n") if l][:N]
white = [f if f.split()[1] == "w" else chess.Board(f).mirror().fen() for f in fens]
M = np.array(label_map(), np.int32); L = len(M)
te_pos = (np.random.default_rng(0).random(100000) < 0.2)[:N]
va_pos = ~te_pos & (np.random.default_rng(1).random(100000) < 0.1)[:N]
tr_pos = ~te_pos & ~va_pos
toks, bits = [], []
with open(ROOT / f"lc0-probe/data/square_bits{G}_n100000_nodes200000.jsonl") as f:
    for i, l in enumerate(f):
        if i == N: break
        sq = json.loads(l)["squares"]
        toks.append(np.repeat(np.array([i * 64 + chess.parse_square(s) for s in sq], np.int32), [len(v) for v in sq.values()]))
        bits.append(np.fromiter((b for v in sq.values() for b in v), np.int32))
tok, bit = np.concatenate(toks), np.concatenate(bits); del toks, bits
cnt = np.bincount(bit[tr_pos[tok // 64]], minlength=L)
keep = np.flatnonzero((cnt + cnt[M]) >= 20)
ck = torch.load(ROOT / a.fwd, weights_only=False)
ck2 = torch.load(ROOT / a.l2, weights_only=False)
assert np.array_equal(ck["keep"], keep) and np.array_equal(ck2["keep"], keep), "decoders trained on other input bits"
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

def bags(E):
    flat = torch.cat([idx[l:h] for l, h in zip(start[E * 64], start[E * 64 + 64])])
    t = (E[:, None] * 64 + np.arange(64)).ravel()
    off = np.cumsum(np.concatenate([[0], (start[t + 1] - start[t])[:-1]])).astype(np.int32)
    return flat, torch.from_numpy(off)

def decoder(d):                                                            # as hybrid_fvu.py
    s = d["state"]
    inp = torch.nn.EmbeddingBag(*s["inp.weight"].shape, mode="sum"); inp.weight.data = s["inp.weight"]
    o_ = torch.nn.Linear(*s["out.weight"].shape[::-1]); o_.load_state_dict({"weight": s["out.weight"], "bias": s["out.bias"]})
    mu, sd = torch.as_tensor(d["mu"]).float(), float(d["sd"])
    return lambda flat, off: o_(torch.nn.functional.gelu(inp(flat, off) + s["bias"])) * sd + mu
d0 = next(d for d in ck["decoders"] if d["tap"] == "/encoder0/ln2")
D0, D2 = decoder(d0), decoder(ck2["decoders"][0])

net = onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))
def seg(i, o_):
    m = convert(onnx.utils.Extractor(net).extract_model(i, o_)).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m
s12, s35, s05, tail = (seg(["/encoder0/ln2"], ["/encoder2/ln2"]), seg(["/encoder2/ln2"], ["/encoder5/ln2"]),
                       seg(["/encoder0/ln2"], ["/encoder5/ln2"]), seg(["/encoder5/ln2"], ["/output/wdl"]))
with torch.no_grad(): C = s35(s12(torch.zeros(64, 256)))                  # L5 with D0, D2 at zero: D5 fits L5 - C

ST, MM = ROOT / f"lc0-probe/data/hybrid_rev_streams{G}_n{N}", ROOT / a.cache
Y5 = np.load(ST / "Y5.npy", mmap_mode="r")
mu5, sd5 = torch.load(ST / "norm5.pt")
big = lambda name, mode="w+": np.lib.format.open_memmap(MM / f"{name}.npy", mode, np.float16, (2 * N, 64, 256))
f32 = lambda X: torch.from_numpy(X.astype(np.float32)).reshape(-1, 256)
G5 = big("G5", "r")
for name, D, sg in (("F2", D2, s35), ("F0", D0, s05)):                    # each layer's decoder, forwarded to L5
    if (MM / f"{name}.done").exists(): continue
    X = big(name)
    with torch.no_grad():
        for s in range(0, 2 * N, 512):
            E = np.arange(s, min(s + 512, 2 * N))
            X[E] = sg(D(*bags(E))).reshape(-1, 64, 256).half().numpy()
            if s % 51200 == 0: log(f"{name} {s:,} / {2 * N:,}")
    X.flush(); (MM / f"{name}.done").touch()
F = {"L2": big("F2", "r"), "L0": big("F0", "r")}
log("F2, F0 cached")

class MLP(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.a, self.b = torch.nn.Linear(256 if a.no_pred or a.gate else 512, h), torch.nn.Linear(h, 256)
        torch.nn.init.zeros_(self.b.weight); torch.nn.init.zeros_(self.b.bias)   # starts at the model before it
        if a.gate: self.g = torch.nn.Sequential(torch.nn.Linear(512, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    def forward(self, x):
        if not a.gate: return self.b(torch.nn.functional.gelu(self.a(x)))
        return torch.sigmoid(self.g(x)) * self.b(torch.nn.functional.gelu(self.a(x[:, :256])))   # h reads F only
both = lambda P: np.concatenate([P, P + N])
E_tr, E_va, TE = both(np.flatnonzero(tr_pos)), both(np.flatnonzero(va_pos)), np.flatnonzero(te_pos)
Ms = {}
src = lambda E: (G5[E], {k: F[k][E] for k in F}, Y5[E % N])             # E sorted: reads stay sequential
def model(g5, fs, upto):                                                   # L5 from D5 and the end MLPs before `upto`
    x = f32(g5) + C.repeat(len(g5), 1)
    for k, Mo in Ms.items():
        if k == upto: break
        f = f32(fs[k])
        m = Mo(f if a.no_pred else torch.cat([f, (x - C.repeat(len(x) // 64, 1) - mu5) / sd5], 1))
        model.out[k] = m; x = x + m * sd5
    return x
model.out = {}
def blocks(E, n=8192):                                                     # random blocks of consecutive boards, in RAM
    for j in rng.permutation(-(-len(E) // n)):
        e = E[j * n:(j + 1) * n]
        yield np.asarray(G5[e]), {k: np.asarray(F[k][e]) for k in Ms}, np.asarray(Y5[e % N])
res = {"args": vars(a)}
P = ("" if a.no_pred else "_pred") + ("_gate" if a.gate else "") + (f"_pen{a.penalty:g}" if a.penalty else "")
CK = MM / f"endmlp{P}_ckpt.pt"
ck = torch.load(CK, weights_only=False) if a.resume and CK.exists() else None
if ck: res = ck["res"]; log(f"resume: M{ck['stage'][1]} after epoch {ck['ep']}")
for k in ("L2", "L0"):
    Ms[k] = Mo = MLP(a.hidden)
    if ck and k != ck["stage"] and f"M{k[1]}" in ck["res"]:                    # stage finished before the stop
        Mo.load_state_dict(ck["done"][k])
        for p in Mo.parameters(): p.requires_grad_(False)
        continue
    def loss_of(g5, fs, y, pen=a.penalty):
        l = torch.nn.functional.mse_loss(model(g5, fs, None), f32(y)) / sd5 ** 2
        return l + pen * model.out[k].pow(2).mean() if pen else l
    def val():
        with torch.no_grad(): return float(np.mean([float(loss_of(*src(E_va[s:s + 256]))) for s in range(0, len(E_va), 256)]))
    opt = optim_extra.make(a.opt, Mo.parameters(), a.lr, a.wd)
    for q in Ms.values():
        for p in q.parameters(): p.requires_grad_(q is Mo)
    ep0 = 0
    if ck and k == ck["stage"]:
        Mo.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        best, bad, curve, state, ep0 = ck["best"], ck["bad"], ck["curve"], ck["state"], ck["ep"]
    else:
        best, bad = val(), 0
        curve, state = [best], {q: p.clone() for q, p in Mo.state_dict().items()}
        log(f"M{k[1]} start val MSE {best:.4f}")
    t0 = time.time()
    for ep in range(ep0, a.epochs if bad < a.patience else ep0):
        for g5, fs, y in blocks(E_tr):
            perm = rng.permutation(len(g5))
            for s in range(0, len(perm), a.batch):
                b = perm[s:s + a.batch]
                loss = loss_of(g5[b], {q: X[b] for q, X in fs.items()}, y[b])
                opt.zero_grad(); loss.backward(); opt.step()
        v = val(); curve.append(v)
        log(f"M{k[1]} epoch {ep + 1}  val MSE {v:.4f}  ({time.time() - t0:.0f}s)")
        if v < best: best, bad, state = v, 0, {q: p.clone() for q, p in Mo.state_dict().items()}
        else: bad += 1
        torch.save({"stage": k, "ep": ep + 1, "model": Mo.state_dict(), "opt": opt.state_dict(), "best": best, "bad": bad,
                    "curve": curve, "state": state, "res": res,
                    "done": {q: m.state_dict() for q, m in Ms.items() if q != k}}, CK)
        if bad >= a.patience: break
    Mo.load_state_dict(state)
    for p in Mo.parameters(): p.requires_grad_(False)
    res[f"M{k[1]}"] = {"curve": curve, "val MSE": best}
    torch.save({"stage": k, "ep": a.epochs, "model": Mo.state_dict(), "opt": opt.state_dict(), "best": best, "bad": a.patience,
                "curve": curve, "state": state, "res": res, "done": {q: m.state_dict() for q, m in Ms.items()}}, CK)

# --- held-out table, both colourings against Leela on the White-to-move board (as hybrid_fvu.py)
full = seg(["/input/planes"], ["/encoder5/ln2", "/output/wdl"])
with torch.no_grad():
    Y, W = map(torch.cat, zip(*[full(torch.from_numpy(encode_fens([white[i] for i in TE[s:s + 256]])))
                               for s in range(0, len(TE), 256)]))
Y = Y.reshape(-1, 256)
fvu = lambda P, R: float(((P - R) ** 2).sum() / ((R - R.mean(0)) ** 2).sum())
v_of = lambda w: w[:, 0] - w[:, 2]
for name, upto in (("just L5", "L2"), ("L5+L2", "L0"), ("L5+L2+L0", None)):
    S, w = [], []
    with torch.no_grad():
        for E in (TE, TE + N):
            for s in range(0, len(E), 256):
                g5, fs, _ = src(E[s:s + 256]); x = model(g5, fs, upto); S.append(x); w.append(tail(x))
    S, w = torch.cat(S), torch.cat(w)
    r = {"L5 FVU": fvu(S, torch.cat([Y, Y])), "WDL FVU": fvu(w, torch.cat([W, W])), "value FVU": fvu(v_of(w), v_of(torch.cat([W, W]))),
         "value r": float(np.corrcoef(v_of(W).numpy(), v_of(w[:len(TE)]).numpy())[0, 1])}
    res[name] = r
    log(f"{name:10} L5 FVU {r['L5 FVU']:.4f}   WDL FVU {r['WDL FVU']:.4f}   value FVU {r['value FVU']:.4f}   value r {r['value r']:.4f}")
json.dump(res, open(ROOT / f"lc0-probe/results_hybrid_endmlp{G}_n{N}{P}.json", "w"), indent=1)
torch.save({k: Mo.state_dict() for k, Mo in Ms.items()} | {"hidden": a.hidden, "fwd": a.fwd, "l2": a.l2, "gate": a.gate, "penalty": a.penalty}, MM / f"endmlp{P}.pt")
