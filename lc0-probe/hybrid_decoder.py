"""Hybrid decoder: features decoded into Leela at L0, then added onto the running stream at later layers.

D0 maps a square's features to its L0 token (/encoder0/ln2), which replaces the real one; Leela runs on. At each
later tap (--taps, default L2 and L5) a decoder adds what Leela did not recompute: its target is the real stream
minus the stream forwarded from the decoders before it. Every decoder reads every feature; which layer a feature
acts in is left to training.

Cross-fitting: a later decoder must learn the earlier ones' errors on boards they did not train on, else it learns
their memorised train errors and nothing that holds out of sample. The train boards are split into --folds folds;
for each stage, one model per fold is trained without it, and the stream forwarded to the next tap for a board
comes from the model that did not see its fold. The stage's final model, kept for Shapley, trains on all of them.

Colour: inputs are the features in real colours, the real square's one-hot and a side-to-move bit. Each position
is trained in both colourings, once with White to move and once colour-swapped with Black to move (features
flipped by colour_frame.label_map, squares mirrored), both against Leela's tokens for the White-to-move board.
The tokens are in Leela's side-to-move frame (Leela sees the colour only through one input plane; its own colour
gap is ~0.003), so the decoder has to learn the symmetry. --inv optionally adds the squared gap between the two
colourings' values to the loss from epoch --inv-from.

--combine H replaces the plain sum at the added taps with S = F + D + C([F, D]): C a per-token MLP (hidden H) on the
forwarded stream F and the decoder's output D, trained jointly with D. C's last layer starts at zero, so each stage
starts where the plain sum does. Readers that add the decoders (hybrid_fvu.py, square_shap.py --hybrid) ignore C.

  python lc0-probe/hybrid_decoder.py --n 100000 --device cuda
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
ap.add_argument("--n", type=int, default=100000, help="positions, a prefix of the 100000-position feature cache")
ap.add_argument("--taps", default="/encoder0/ln2,/encoder2/ln2,/encoder5/ln2", help="the first replaces, the rest add")
ap.add_argument("--hidden", default="2048,1024,1024")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--batch", type=int, default=256, help="boards per batch")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--opt", default="adam", help="any of optim_extra.make's names")
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--tag", default="", help="suffix for the run's files")
ap.add_argument("--games", action="store_true", help="the 100k Lichess game positions (fens_games.txt) instead of the puzzles")
ap.add_argument("--inv", type=float, default=0.0, help="weight of the colour-invariance loss")
ap.add_argument("--inv-from", type=int, default=10, help="first epoch with the invariance loss")
ap.add_argument("--inv-pos", type=int, default=32, help="positions per batch the invariance loss runs on")
ap.add_argument("--combine", type=int, default=0, help="hidden size of the combiner at the added taps (0: plain sum)")
ap.add_argument("--min", type=int, default=20, help="positive train tokens needed to use a bit (in either colouring)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
rng = np.random.default_rng(0)
torch.manual_seed(0)
torch.set_num_threads(a.threads)
dev = torch.device(a.device)
TAPS, HID = a.taps.split(","), [int(h) for h in a.hidden.split(",")]
N = a.n
G = "_games" if a.games else ""
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)

# --- boards: e < N White to move (the cache's side-to-move frame), e >= N the same board colour-swapped
fens = [l for l in (ROOT / f"sf-probe/data/fens{G}.txt").read_text().split("\n") if l][:N]
white = [f if f.split()[1] == "w" else chess.Board(f).mirror().fen() for f in fens]
boards = white + [chess.Board(f).mirror().fen() for f in white]
M = np.array(label_map(), np.int32)
L = len(M)
te_pos = (np.random.default_rng(0).random(100000) < 0.2)[:N]               # the L0 decoder's split, per position
va_pos = ~te_pos & (np.random.default_rng(1).random(100000) < 0.1)[:N]
tr_pos = ~te_pos & ~va_pos
fold = np.random.default_rng(2).integers(0, a.folds, N)                     # cross-fitting folds
toks, bits = [], []
with open(ROOT / f"lc0-probe/data/square_bits{G}_n100000_nodes200000.jsonl") as f:
    for i, l in enumerate(f):
        if i == N: break
        sq = json.loads(l)["squares"]
        toks.append(np.repeat(np.array([i * 64 + chess.parse_square(s) for s in sq], np.int32), [len(v) for v in sq.values()]))
        bits.append(np.fromiter((b for v in sq.values() for b in v), np.int32))
tok, bit = np.concatenate(toks), np.concatenate(bits); del toks, bits
cnt = np.bincount(bit[tr_pos[tok // 64]], minlength=L)
keep = np.flatnonzero((cnt + cnt[M]) >= a.min)                             # closed under the colour swap
col = np.full(L, -1, np.int32); col[keep] = np.arange(len(keep))
SQ, STM = len(keep), len(keep) + 64                                        # one-hot rows, side-to-move row
c0, c1 = col[bit], col[M[bit]]; del bit
t1 = (tok // 64 + N) * 64 + tok % 64                                       # Leela's token index is the same in both
m0, m1 = c0 >= 0, c1 >= 0                                                  # colourings (its side-to-move frame)
all_t = np.arange(2 * N * 64, dtype=np.int32)
real_sq = np.where(all_t < N * 64, all_t % 64, (all_t % 64) ^ 56)         # the token's real square
T_ = np.concatenate([tok[m0], t1[m1], all_t, all_t[N * 64:]])
I_ = np.concatenate([c0[m0], c1[m1], (SQ + real_sq).astype(np.int32), np.full(N * 64, STM, np.int32)])
del tok, t1, c0, c1, m0, m1, real_sq
o = np.argsort(T_, kind="stable")
start = np.concatenate([[0], np.cumsum(np.bincount(T_, minlength=2 * N * 64))]); del T_
idx = torch.from_numpy(I_[o]); del o, I_                                   # int32: 475M inputs at n=100k
D_in = STM + 1
log(f"{N:,} positions x 2 colourings, {len(keep):,} bits, {len(idx) / (2 * N * 64):.1f} inputs per token")

def bags(E):
    """EmbeddingBag inputs for all 64 tokens of boards E (a board's tokens are contiguous in idx)."""
    flat = torch.cat([idx[l:h] for l, h in zip(start[E * 64], start[E * 64 + 64])])
    t = (E[:, None] * 64 + np.arange(64)).ravel()
    off = np.cumsum(np.concatenate([[0], (start[t + 1] - start[t])[:-1]])).astype(np.int32)
    return flat.to(dev, non_blocking=True), torch.from_numpy(off).to(dev, non_blocking=True)

class MLP(torch.nn.Module):
    def __init__(self, hidden, comb=0):
        super().__init__()
        if comb:                                                           # [forwarded stream, decoder output] -> correction
            self.comb = torch.nn.Sequential(torch.nn.Linear(512, comb), torch.nn.GELU(), torch.nn.Linear(comb, 256))
            torch.nn.init.zeros_(self.comb[2].weight); torch.nn.init.zeros_(self.comb[2].bias)
        self.inp = torch.nn.EmbeddingBag(D_in, hidden, mode="sum")
        torch.nn.init.normal_(self.inp.weight, std=(len(idx) / (2 * N * 64)) ** -0.5)   # unit pre-activations
        self.bias = torch.nn.Parameter(torch.zeros(hidden))
        self.out = torch.nn.Linear(hidden, 256)
    def forward(self, flat, off):
        return self.out(torch.nn.functional.gelu(self.inp(flat, off) + self.bias))

# --- Leela, in torch: the real stream at every tap, and the segments between taps
net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
inferred = onnx.shape_inference.infer_shapes(onnx.load(net_path))
def seg(i, o_):
    m = convert(onnx.utils.Extractor(inferred).extract_model(i, o_)).eval().to(dev)
    for p in m.parameters(): p.requires_grad_(False)
    return m
full = seg(["/input/planes"], TAPS + ["/output/wdl"])
segs = [seg([t], [u]) for t, u in zip(TAPS, TAPS[1:] + ["/output/wdl"])]
MM = ROOT / f"lc0-probe/data/hybrid_tmp{G}_n{N}{a.tag}"; MM.mkdir(exist_ok=True)   # streams and checkpoints on disk: 25+ GB at
big = lambda name, n, mode="w+": np.lib.format.open_memmap(MM / f"{name}.npy", mode, np.float16, (n, 64, 256))  # n=100k
v_of = lambda w: w[:, 0] - w[:, 2]
if (MM / "V.npy").exists():                                                # a rerun resumes from what is on disk
    Y, V = [big(f"Y{k}", N, "r") for k in range(len(TAPS))], np.load(MM / "V.npy")
else:
    Y = [big(f"Y{k}", N) for k in range(len(TAPS))]                       # White-to-move boards: both colourings' target
    V = np.empty(2 * N, np.float32)
    with torch.no_grad():
        for s in range(0, N, 512):
            e = min(s + 512, N)
            r = full(torch.from_numpy(encode_fens(white[s:e])).to(dev))
            for y, x in zip(Y, r[:-1]): y[s:e] = x.reshape(-1, 64, 256).half().cpu().numpy()
            V[s:e] = v_of(r[-1]).cpu().numpy()
            V[N + s:N + e] = v_of(full(torch.from_numpy(encode_fens(boards[N + s:N + e])).to(dev))[-1]).cpu().numpy()
    for y in Y: y.flush()
    np.save(MM / "V.npy", V)                                               # written last: marks the streams complete
log(f"real streams read; Leela's own colour gap: median {np.median(np.abs(V[:N] - V[N:])):.4f}")

def run_from(k, S):
    """Stream S (B*64, 256) at TAPS[k] -> value, through the remaining segments (no later decoders)."""
    for g in segs[k:]: S = g(S)
    return v_of(S)

both = lambda P: np.concatenate([P, P + N])
E_va, TE = both(np.flatnonzero(va_pos)), np.flatnonzero(te_pos)
P_all = np.flatnonzero(tr_pos)
F = None                                                                   # stream forwarded to the current tap
res = {"n": N, "taps": TAPS, "hidden": HID, "folds": a.folds, "combine": a.combine, "inv": a.inv, "opt": a.opt, "lr": a.lr, "wd": a.wd}
final = []

def fit(k, h, P_tr, Zf, mu, sd, fwd, label):
    """Train one decoder for stage k on positions P_tr (both colourings); early stopping on E_va.
    Checkpointed every epoch, so a rerun picks up where it stopped."""
    D = MLP(h, a.combine if k else 0).to(dev)
    if k: torch.nn.init.zeros_(D.out.weight)                               # added decoders start at the mean residual
    def raw(E, f=None):
        """D's output (normalised) plus the combiner's correction; f: the forwarded stream, default fwd(E)."""
        r = D(*bags(E))
        if hasattr(D, "comb"): r = r + D.comb(torch.cat([fwd(E) if f is None else f, r * sd + mu], 1))
        return r
    dec = lambda E, f=None: raw(E, f) * sd + mu
    target = lambda E: (raw(E), (torch.from_numpy(Zf(E)).to(dev).reshape(-1, 256) - mu) / sd)
    opt = optim_extra.make(a.opt, D.parameters(), a.lr, a.wd)
    def val():
        with torch.no_grad():
            return float(np.mean([float(torch.nn.functional.mse_loss(*target(E_va[s:s + 512]))) for s in range(0, len(E_va), 512)]))
    E_tr = both(P_tr)
    ck = MM / f"fit_{label.replace('/', '_').replace(' ', '_')}.pt"
    if ck.exists():
        c = torch.load(ck, weights_only=False)
        D.load_state_dict(c["D"]); opt.load_state_dict(c["opt"])
        best, state, bad, ep0, done = c["best"], c["state"], c["bad"], c["ep"], c["done"]
        log(f"{label} resumed after epoch {ep0}, best val MSE {best:.4f}" + (", done" if done else ""))
    else:
        best, state, bad, ep0, done = val(), {q: p.clone() for q, p in D.state_dict().items()}, 0, 0, False
        log(f"{label} start val MSE {best:.4f}")
    t0 = time.time()
    save = lambda ep, done: torch.save({"D": D.state_dict(), "opt": opt.state_dict(), "best": best, "state": state,
                                        "bad": bad, "ep": ep, "done": done}, ck)
    for ep in range(ep0, 0 if done else a.epochs):
        perm, gap, nb = rng.permutation(E_tr), 0.0, 0
        for s in range(0, len(perm), a.batch):
            loss = torch.nn.functional.mse_loss(*target(perm[s:s + a.batch]))
            if a.inv and ep + 1 >= a.inv_from:
                q = rng.choice(P_tr, a.inv_pos, replace=False)
                g = ((run_from(k, fwd(q) + dec(q)) - run_from(k, fwd(q + N) + dec(q + N))) ** 2).mean()
                loss = loss + a.inv * g; gap += float(g.detach()); nb += 1
            opt.zero_grad(); loss.backward(); opt.step()
        v = val()
        log(f"{label} epoch {ep + 1}  val MSE {v:.4f}" + (f"  colour gap rms {np.sqrt(gap / nb):.4f}" if nb else "") + f"  ({time.time() - t0:.0f}s)")
        if v < best: best, bad, state = v, 0, {q: p.clone() for q, p in D.state_dict().items()}
        else: bad += 1
        save(ep + 1, bad >= a.patience or ep + 1 == a.epochs)
        if bad >= a.patience: break
    D.load_state_dict(state)
    for p in D.parameters(): p.requires_grad_(False)
    return D, dec, best

for k, (tap, h) in enumerate(zip(TAPS, HID)):
    Yk = Y[k]                                                              # target: real stream minus forwarded one
    if F is None:
        Zf, fwd = (lambda E: Yk[E % N].astype(np.float32)), (lambda E: 0)
    else:
        Zf = lambda E: Yk[E % N].astype(np.float32) - F[E].astype(np.float32)
        fwd = lambda E: torch.from_numpy(F[E].astype(np.float32)).to(dev).reshape(-1, 256)
    nk = MM / f"norm{k}.pt"                                                # kept, so resumed decoders see the same scale
    if not nk.exists():
        Zs = Zf(both(rng.choice(P_all, min(4000, len(P_all)), replace=False)))
        torch.save((torch.from_numpy(Zs.reshape(-1, 256).mean(0)), float(Zs.std())), nk); del Zs
    mu, sd = torch.load(nk); mu = mu.to(dev)
    log(f"{tap}: target sd {sd:.4f}")
    D, dec, best = fit(k, h, P_all, Zf, mu, sd, fwd, f"{tap} final")
    final.append({"state": {q: p.cpu() for q, p in D.state_dict().items()}, "mu": mu.cpu().numpy(), "sd": sd,
                  "hidden": h, "tap": tap, "fn": dec})
    res[tap] = {"val MSE": best, "target sd": sd}
    if k + 1 == len(TAPS): break
    # stream forwarded to the next tap: train boards from the model that did not see their fold, the rest from the final one
    owner = np.full(2 * N, -1); owner[both(P_all)] = np.concatenate([fold[P_all]] * 2)
    decs = {-1: dec}
    for j in range(a.folds):
        decs[j] = fit(k, h, P_all[fold[P_all] != j], Zf, mu, sd, fwd, f"{tap} fold {j}")[1]
    if (MM / f"F{k}.done").exists():
        F = big(f"F{k}", 2 * N, "r")
        continue
    Fn = big(f"F{k}", 2 * N)
    with torch.no_grad():
        for s in range(0, 2 * N, 512):
            E = np.arange(s, min(s + 512, 2 * N))
            out = torch.empty(len(E) * 64, 256, device=dev)
            for j in np.unique(owner[E]):
                m = np.flatnonzero(owner[E] == j); rows = torch.from_numpy((m[:, None] * 64 + np.arange(64)).ravel()).to(dev)
                out[rows] = segs[k](fwd(E[m]) + decs[j](E[m]))
            Fn[E] = out.reshape(-1, 64, 256).half().cpu().numpy()
    Fn.flush(); (MM / f"F{k}.done").touch()
    F = Fn

# --- fidelity on held-out positions, both colourings: Leela's value vs D0, D0 + D2, ...
with torch.no_grad():
    for upto in range(1, len(TAPS) + 1):
        vals = []
        for E in (TE, TE + N):
            out = []
            for s in range(0, len(E), 512):
                e, S = E[s:s + 512], None
                for k in range(len(TAPS)):
                    f = None if S is None else segs[k - 1](S)
                    S = (0 if f is None else f) + (final[k]["fn"](e, f) if k < upto else 0)
                out.append(run_from(len(TAPS) - 1, S).cpu())
            vals.append(torch.cat(out).numpy())
        name = " + ".join("D" + t.split("encoder")[1].split("/")[0] for t in TAPS[:upto])
        r = {"r": float(np.corrcoef(V[TE], vals[0])[0, 1]), "median |dy|": float(np.median(np.abs(V[TE] - vals[0]))),
             "r Black to move": float(np.corrcoef(V[TE + N], vals[1])[0, 1]),
             "median |dy| Black to move": float(np.median(np.abs(V[TE + N] - vals[1]))),
             "colour gap median": float(np.median(np.abs(vals[0] - vals[1])))}
        res[name] = r
        log(f"{name:14} value r {r['r']:.4f} (Black to move {r['r Black to move']:.4f})  median |dy| {r['median |dy|']:.4f}  "
            f"colour gap {r['colour gap median']:.4f}")
torch.save({"keep": keep, "sq row": SQ, "stm row": STM, "taps": TAPS, "combine": a.combine,
            "decoders": [{q: d[q] for q in ("state", "mu", "sd", "hidden", "tap")} for d in final]},
           ROOT / f"lc0-probe/data/hybrid_decoder{G}_n{N}{a.tag}.pt")
json.dump(res, open(ROOT / f"lc0-probe/results_hybrid_decoder{G}_n{N}{a.tag}.json", "w"), indent=1)
shutil.rmtree(MM)
