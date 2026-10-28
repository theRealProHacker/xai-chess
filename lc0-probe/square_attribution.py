"""Gradient x input on the per-square features, through the embedding (or --tap) decoder and Leela.

The decoder square_decoder_wdl.py saved for the embedding turns each square's feature bits into
its token. With all 64 decoded tokens in place of the real embedding, the rest of the net (ONNX
converted to torch) gives y = P(win) - P(loss) for the side to move. A bit's attribution is
dy/dx * x on the decoder's 0/1 input: the gradient at the square's hidden pre-activation, dotted
with the bit's input weights. First order, so the attributions do not sum to y. The square's
one-hot input gets one too, kept apart as "square". Written for the first --positions held-out
positions (with --gap, only quiet ones: Stockfish's three best moves are close), with each bit's mean and spread over the tokens where it is on ("usual"), so
square_attribution.html can show an attribution relative to the bit's usual one. With --ig, integrated
gradients instead: from the square-only token (all features off) to the position, so a square's attributions
sum to y - y(baseline). With --ig-base mean the baseline is instead each square's mean bits for its occupant
(training rows), so a bit that is off gets (0 - mean) x gradient too ("absent"); with --ig-base square, each square's
mean bits over any occupant, so the occupant and its fixed geometry get attributions as well. Either way a deletion test switches off the top-k features by attribution and prints
how far y moves, against gradient x input and a random order.

  python lc0-probe/square_attribution.py --n 100000
  python lc0-probe/square_attribution.py --tap /encoder0/ln2 --ig 32 --pgn cleaned_commentary/pgnlib --gap 50
"""
import argparse, json, subprocess, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, chess.engine, chess.pgn, numpy as np, onnx, onnxruntime as ort, torch
from onnx2torch import convert
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--nodes", type=int, default=200000)
ap.add_argument("--positions", type=int, default=300)
ap.add_argument("--pgn", help="take the positions from the games in this file or directory instead of the held-out "
                "ones: main lines, moves 12-40, every fifth ply, shuffled")
ap.add_argument("--fens", help="take the positions from this file instead, one FEN per line ('#' lines skipped), "
                "in order; the output file name ends in the file's stem")
ap.add_argument("--gap", type=int, default=0, help="keep a position only if Stockfish's best and third-best moves "
                "(depth 12) differ by at most this many centipawns; 0 = keep all")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tap", default="/attn_body/rehape", help="layer the decoder was trained on, e.g. /encoder2/ln2")
ap.add_argument("--ig", type=int, default=0, help="integrated gradients with this many steps from the square-only "
                "token instead of gradient x input; 0 = gradient x input")
ap.add_argument("--ig-base", choices=["off", "mean", "square", "expected", "start"], default="off", help="IG baseline: every "
                "feature off, the mean bits of the square's occupant on that square, the square's mean bits over any "
                "occupant (means over the first --mean training rows), or expected gradients: --ig / 8 random training boards "
                "(of the first --mean) as baselines, --eg-steps midpoint steps each; or the starting position's bits")
ap.add_argument("--mean", type=int, default=20000)
ap.add_argument("--eg-steps", type=int, default=8, help="expected gradients: midpoint steps per baseline board")
ap.add_argument("--ig-only", help="IG path over the bits with this label prefix only (e.g. square.control.); every "
                "other bit stays at its value in the position, so the attributions sum to y - y(those bits at baseline)")
ap.add_argument("--target", choices=["wl", "logit", "atanh", "win"], default="wl", help="attributed output: P(win) - P(loss), "
                "log P(win) - log P(loss), atanh(P(win) - P(loss)), or P(win)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
TAP = a.tap

n_fens = sum(1 for l in open(ROOT / "sf-probe/data/fens.txt") if l.strip())
N = min(a.n, n_fens)
te_pos = np.random.default_rng(0).random(N) < 0.2                          # same split as square_probe.py
ck = torch.load(ROOT / f"lc0-probe/data/square_decoder{TAP.replace('/', '_')}_n{N}.pt", weights_only=False)
keep, mu, sd = ck["keep"], torch.from_numpy(ck["mu"]).float(), float(ck["sd"])
schema = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                   capture_output=True, text=True, check=True).stdout)
labels = [schema["labels"][i] for i in keep] + ["square"] * 64             # decoder input -> name
col = np.full(schema["bits_per_square"], -1); col[keep] = np.arange(len(keep))
E, bias = ck["state"]["inp.weight"], ck["state"]["bias"]                   # (D_in, hidden), (hidden,)
out = torch.nn.Linear(*ck["state"]["out.weight"].shape[::-1])
out.load_state_dict({"weight": ck["state"]["out.weight"], "bias": ck["state"]["out.bias"]})

net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(net_path))).extract_model([TAP], ["/output/wdl"])
assert [i.name for i in sub.graph.input] == [TAP]
tail = convert(sub).eval()                                                 # (64, 256) tokens at the tap -> (1, 3) WDL
for p in tail.parameters(): p.requires_grad_(False)
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
full = ort.InferenceSession(str(net_path), so, providers=["CPUExecutionProvider"])

sf = chess.engine.SimpleEngine.popen_uci(str(ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2")) if a.gap else None

def gap(fen):
    """Centipawns between Stockfish's best and third-best move; None with a mate score or under three moves."""
    info = sf.analyse(chess.Board(fen), chess.engine.Limit(depth=12), multipv=3)
    cp = [x["score"].relative.score() for x in info]
    return None if len(cp) < 3 or None in cp else cp[0] - cp[2]

seen = 0
def rows():
    """Feature rows (fen with the side to move mirrored to White, bits per square) passing --gap, with the gap."""
    global seen
    def quiet(fen):
        global seen
        seen += 1
        d = gap(fen) if sf else None
        return (not sf or (d is not None and d <= a.gap)), d
    if not a.pgn and not a.fens:
        with open(ROOT / f"lc0-probe/data/square_bits_n{N}_nodes{a.nodes}.jsonl") as f:
            for i, l in enumerate(f):
                if not te_pos[i]: continue
                r = json.loads(l)
                ok, d = quiet(r["fen"])
                if ok: yield r, d
        return
    fens = []
    if a.fens:
        for l in open(a.fens):
            if l.strip() and not l.startswith("#"):
                b = chess.Board(l.strip()); fens.append((b if b.turn else b.mirror()).fen())
    else:
        for path in sorted(Path(a.pgn).glob("*.pgn")) if Path(a.pgn).is_dir() else [Path(a.pgn)]:
            with open(path, errors="replace") as f:
                while (g := chess.pgn.read_game(f)) is not None:
                    b = g.board()
                    if b.fen() != chess.STARTING_FEN: continue
                    for k, m in enumerate(g.mainline_moves()):
                        b.push(m)
                        if 24 <= k < 80 and k % 5 == 0: fens.append((b if b.turn else b.mirror()).fen())
    fens = list(dict.fromkeys(fens))
    if not a.fens: np.random.default_rng(0).shuffle(fens)
    picked = []
    for fen in fens:
        ok, d = quiet(fen)
        if ok: picked.append((fen, d))
        if len(picked) == a.positions: break
    out_ = subprocess.run([ROOT / "features/square/target/release/square", "--indices", "--engine",
                           ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2", "--nodes", str(a.nodes)],
                          input="\n".join(f for f, _ in picked) + "\n", capture_output=True, text=True, check=True).stdout
    for l, (fen, d) in zip(out_.splitlines(), picked):
        r = json.loads(l); assert r["fen"] == fen
        yield r, d

KEYS = ".PNBRQKpnbrqk"
occ = lambda b, s: KEYS.index(b.piece_at(s).symbol()) if b.piece_at(s) else 0
if a.ig_base != "off":                                                      # XM[occupant * 64 + square] = mean decoder input
    mpath = ROOT / f"lc0-probe/data/ig_means_n{N}_{a.mean}{'' if len(keep) == 22498 else f'_k{len(keep)}'}.npz"   # per kept bit set
    if mpath.exists(): z = np.load(mpath); XM, XC = z["XM"], z["XC"]
    else:
        XM = np.zeros((13 * 64, len(keep)), np.float32); XC = np.zeros(13 * 64)
        with open(ROOT / f"lc0-probe/data/square_bits_n{N}_nodes{a.nodes}.jsonl") as f:
            for i, l in enumerate(f):
                if i >= a.mean: break
                if te_pos[i]: continue
                r = json.loads(l); b = chess.Board(r["fen"])
                for s in range(64): XC[occ(b, s) * 64 + s] += 1
                rr, cc = [], []
                for q, v in r["squares"].items():
                    c = col[v]; c = c[c >= 0]; rr.append(np.full(len(c), occ(b, chess.parse_square(q)) * 64 + chess.parse_square(q))); cc.append(c)
                np.add.at(XM, (np.concatenate(rr), np.concatenate(cc)), 1)
        XM /= np.maximum(XC, 1)[:, None]
        np.savez(mpath, XM=XM, XC=XC)
    SM = (XM * XC[:, None]).reshape(13, 64, -1).sum(0) / np.maximum(XC.reshape(13, 64).sum(0), 1)[:, None]   # any occupant
    print(f"ig baseline: means over {int(XC.reshape(13, 64).sum(0)[0])} positions", flush=True)
fallback = 0
if a.ig_base == "start":                                                   # the starting position's bits
    r0 = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--indices", "--engine",
                                    ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2", "--nodes", str(a.nodes)],
                                   input=chess.STARTING_FEN + "\n", capture_output=True, text=True, check=True).stdout.splitlines()[0])
    XS = np.zeros((64, len(keep)), np.float32)
    for q, v in r0["squares"].items(): c = col[v]; XS[chess.parse_square(q), c[c >= 0]] = 1
    np.save(ROOT / f"lc0-probe/data/ig_start_bits{TAP.replace('/', '_')}.npy", XS)
if a.ig_base == "expected":                                                # pool of training boards: per square, decoder inputs
    POOL = []
    with open(ROOT / f"lc0-probe/data/square_bits_n{N}_nodes{a.nodes}.jsonl") as f:
        for i, l in enumerate(f):
            if i >= a.mean: break
            if te_pos[i]: continue
            sq = json.loads(l)["squares"]; c = [col[sq.get(chess.square_name(s), [])] for s in range(64)]
            POOL.append([x[x >= 0] for x in c])
    erng = np.random.default_rng(1)
    print(f"expected gradients: {len(POOL)} training boards", flush=True)

def value(P):
    """(B, 64, hidden) decoder pre-activations -> (B,) P(win) - P(loss)."""
    w = tail((out(torch.nn.functional.gelu(P)) * sd + mu).reshape(-1, mu.shape[0]))
    q = w[:, 0] - w[:, 2]
    return w[:, 0].log() - w[:, 2].log() if a.target == "logit" else q.clamp(-0.9999, 0.9999).atanh() if a.target == "atanh" else w[:, 0] if a.target == "win" else q

KS = [1, 2, 5, 10, 20, 50, 100]                                            # deletion test: features switched off
rng = np.random.default_rng(0)
dele = {m: np.zeros((2, len(KS))) for m in ["gradient x input", "integrated gradients", "random"]}   # summed dy, positive-first / negative-first
res = []
cnt, tot, tot2 = np.zeros(len(keep)), np.zeros(len(keep)), np.zeros(len(keep))   # per bit, over the tokens where it is on
ab = np.zeros(len(keep), bool)                                             # bits with an "absent" attribution somewhere
for r, d in rows():
    if len(res) == a.positions: break
    by_sq = {chess.parse_square(s): v for s, v in r["squares"].items()}
    ins = []                                                           # each square's decoder inputs
    for s in range(64):
        b = col[by_sq.get(s, [])]
        ins.append(np.append(b[b >= 0], len(keep) + s))
    base = E[len(keep):] + bias                                        # (64, hidden): square one-hot only
    feat = torch.stack([E[torch.from_numpy(x[:-1])].sum(0) for x in ins])   # each square's features, summed
    pre = (base + feat)[None].clone().requires_grad_(True)
    y = value(pre)[0]
    y.backward()
    g1 = pre.grad[0]
    if a.ig:                                                           # midpoint rule from the baseline to the position
        xm = torch.zeros(64, len(keep))
        if a.ig_base == "square": xm = torch.from_numpy(SM).float()
        elif a.ig_base == "start": xm = torch.from_numpy(XS)
        elif a.ig_base == "mean":                                      # the occupant's mean; any occupant's if under 10 seen
            bd = chess.Board(r["fen"])
            for s in range(64):
                k = occ(bd, s) * 64 + s
                if XC[k] < 10: fallback += 1
                xm[s] = torch.from_numpy(XM[k] if XC[k] >= 10 else SM[s])
        if a.ig_only:                                                  # hold every other bit at the position's value
            fix = torch.from_numpy(np.array([not labels[j].startswith(a.ig_only) for j in range(len(keep))]))
            for s in range(64):
                x = torch.zeros(len(keep)); x[torch.from_numpy(col[by_sq.get(s, [])][col[by_sq.get(s, [])] >= 0])] = 1
                xm[s, fix] = x[fix]
        if a.ig_base == "expected":                                    # x * E g(mean) - mean_k x'_k * E g_k
            m = a.eg_steps; ks = erng.integers(len(POOL), size=a.ig // m)   # boards x m midpoint steps each
            feat0 = torch.stack([torch.stack([E[torch.from_numpy(POOL[k][s])].sum(0) for s in range(64)]) for k in ks])
            al = ((torch.arange(m) + 0.5) / m)[None, :, None, None]
            P = (base + feat0[:, None] + al * (feat - feat0[:, None])).reshape(-1, 64, feat.shape[-1]).requires_grad_(True)
            value(P).sum().backward()
            g = P.grad.reshape(len(ks), m, 64, -1).mean(1); gi = g.mean(0)
            EG = (E[:len(keep)] @ gi.T).numpy().astype(np.float64)
            SUB = np.zeros((len(keep), 64))                            # mean_k x'_k * (E g_k)
            for n, k in enumerate(ks):
                sq = np.concatenate([np.full(len(POOL[k][s]), s) for s in range(64)]); j = np.concatenate(POOL[k])
                v = (E[torch.from_numpy(j)] * g[n, torch.from_numpy(sq)]).sum(1).numpy()
                np.add.at(SUB, (j, sq), v / len(ks))
            with torch.no_grad(): y0 = float(value(base + feat0).mean())
        else:
            feat0 = xm @ E[:len(keep)]
            al = ((torch.arange(a.ig) + 0.5) / a.ig)[:, None, None]
            P = (base + feat0 + al * (feat - feat0)).requires_grad_(True)
            value(P).sum().backward()
            gi = P.grad.mean(0)
            EG = (E[:len(keep)] @ gi.T).numpy().astype(np.float64)    # (inputs, 64): input row x mean gradient
            with torch.no_grad(): y0 = float(value((base + feat0)[None])[0])
    wr = full.run(["/output/wdl"], {"/input/planes": encode_fens([r["fen"]])})[0][0]
    squares, absent, flat = {}, {}, {"gradient x input": [], "integrated gradients": []}   # (square, input row, attribution)
    for s in range(64):
        rows_ = E[torch.from_numpy(ins[s])]
        att = (rows_ @ g1[s]).numpy().astype(np.float64)                  # dy/dx * x, x = 1 on these inputs
        b, ident = ins[s][:-1], float(att[-1])                         # the one-hot is last
        flat["gradient x input"] += [(s, j, v) for j, v in zip(b, att[:-1])]
        if a.ig:
            on = np.zeros(len(keep), bool); on[b] = True
            if a.ig_base == "expected":
                att = EG[b, s] - SUB[b, s]
                z = np.flatnonzero((np.abs(SUB[:, s]) > 1e-6) & ~on); av = -SUB[z, s]
            else:
                xs = xm[s].numpy().astype(np.float64)
                att = (1 - xs[b]) * EG[b, s]                            # (x - baseline) * mean gradient along the path
                z = np.flatnonzero((xs > 0) & ~on)                     # off here, on in the baseline
                av = -xs[z] * EG[z, s]
            flat["integrated gradients"] += [(s, j, v) for j, v in zip(b, att)]
            absent[s] = (z, av); ab[z] = True
            ident = None                                               # the one-hot is in the baseline
        else: att = att[:-1]
        cnt[b] += 1; tot[b] += att; tot2[b] += att ** 2
        squares[chess.square_name(s)] = {"square": None if ident is None else round(ident, 5), "bits": [b, att]}
        if s in absent: squares[chess.square_name(s)]["absent"] = list(absent.pop(s))
    flat["random"] = [(s, j, rng.random()) for s, j, _ in flat["gradient x input"]]
    with torch.no_grad():                                              # switch off the top-k features of each order
        for m, f in flat.items():
            if not f: continue
            for o, order in enumerate([sorted(f, key=lambda x: -x[2]), sorted(f, key=lambda x: x[2])]):
                P = []
                for k in KS:
                    q = (base + feat).clone()
                    for s_, j, _ in order[:k]: q[s_] -= E[j]
                    P.append(q)
                dele[m][o] += (value(torch.stack(P)) - y.detach()).numpy()
    res.append({"fen": r["fen"], "y": float(wr[0] - wr[2]), "y decoded": float(y.detach()), "squares": squares})
    if a.ig: res[-1]["y baseline"] = y0
    if sf: res[-1]["gap"] = d
    if len(res) % 50 == 0: print(f"{len(res)} positions", flush=True)

if sf: sf.quit(); print(f"kept {len(res)} of {seen} positions with gap <= {a.gap} cp")
Y = np.array([[r["y"], r["y decoded"]] for r in res])
if a.ig:                                                                   # completeness: attributions sum to y - y(baseline)
    gap_ = [abs(sum(v for q in r["squares"].values() for v in [*q["bits"][1], *q.get("absent", [[], []])[1]])
                - (r["y decoded"] - r["y baseline"])) for r in res]
    if a.ig_base != "off":
        sh = [sum(abs(v) for q in r["squares"].values() for v in q["absent"][1]) /
              sum(abs(v) for q in r["squares"].values() for v in [*q["bits"][1], *q["absent"][1]]) for r in res]
        print(f"absent bits: {np.mean(sh):.1%} of |attribution| (mean over positions); {fallback} squares on the any-occupant mean")
    print(f"integrated gradients, {a.ig} steps: |sum - (y - y baseline)| mean {np.mean(gap_):.4f}, max {np.max(gap_):.4f}")
print("deletion test, mean change of y decoded after switching off the k features with the highest (lowest) attribution:")
print("  k                     " + "".join(f"{k:>8}" for k in KS))
for m, v in dele.items():
    if m == "integrated gradients" and not a.ig: continue
    for o, name in enumerate(["highest", "lowest"]):
        print(f"  {m[:20]:20} {name:8}" + "".join(f"{x / len(res):8.3f}" for x in v[o]))
print(f"{len(res)} positions, y decoded vs real: FVU {((Y[:, 0] - Y[:, 1]) ** 2).sum() / ((Y[:, 0] - Y[:, 0].mean()) ** 2).sum():.3f}")
used = np.flatnonzero((cnt > 0) | ab)                                      # bits that occur; the file indexes into these
new = np.full(len(keep), -1); new[used] = np.arange(len(used))
mean = tot[used] / np.maximum(cnt[used], 1)
std = np.sqrt(np.maximum(tot2[used] / np.maximum(cnt[used], 1) - mean ** 2, 0))
for r in res:
    for q in r["squares"].values():
        q["bits"] = [[int(new[j]), round(float(v), 5)] for j, v in zip(*q["bits"])]
        if "absent" in q: q["absent"] = [[int(new[j]), round(float(v), 5)] for j, v in zip(*q["absent"]) if round(float(v), 5)]
tag = "" if TAP == "/attn_body/rehape" else TAP.replace("/", "_")             # e.g. _encoder2_ln2
out_path = ROOT / f"lc0-probe/square_attribution_n{N}{tag}{'_ig' if a.ig else ''}{'_' + a.ig_base if a.ig_base != 'off' else ''}{'_s' + str(a.eg_steps) if a.ig_base == 'expected' and a.eg_steps != 8 else ''}{'_only_' + a.ig_only.strip('.').replace('.', '_') if a.ig_only else ''}{'_' + a.target if a.target != 'wl' else ''}{'_games' if a.pgn else ''}{'_' + Path(a.fens).stem if a.fens else ''}.js"                  # a script, so the demo opens from file://
out_path.write_text("window.ATTRIBUTION = " + json.dumps({
    "n": N, "tap": TAP, "method": f"integrated gradients, {a.ig} steps{', ' + a.ig_base + ' mean baseline' if a.ig_base != 'off' else ''}{', path over ' + a.ig_only + '* only' if a.ig_only else ''}" if a.ig else "gradient x input", "gap": a.gap, "source": a.pgn or a.fens or "held-out puzzles", "labels": [labels[j] for j in used],
    "usual": [[round(float(m), 6), round(float(d), 6), int(c)] for m, d, c in zip(mean, std, cnt[used])],   # mean, spread, tokens
    "positions": res}, separators=(",", ":")) + ";\n")
print("wrote", out_path)
