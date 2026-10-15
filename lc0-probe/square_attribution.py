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
sum to y - y(baseline). Either way a deletion test switches off the top-k features by attribution and prints
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
ap.add_argument("--gap", type=int, default=0, help="keep a position only if Stockfish's best and third-best moves "
                "(depth 12) differ by at most this many centipawns; 0 = keep all")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tap", default="/attn_body/rehape", help="layer the decoder was trained on, e.g. /encoder2/ln2")
ap.add_argument("--ig", type=int, default=0, help="integrated gradients with this many steps from the square-only "
                "token instead of gradient x input; 0 = gradient x input")
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
    if not a.pgn:
        with open(ROOT / f"lc0-probe/data/square_bits_n{N}_nodes{a.nodes}.jsonl") as f:
            for i, l in enumerate(f):
                if not te_pos[i]: continue
                r = json.loads(l)
                ok, d = quiet(r["fen"])
                if ok: yield r, d
        return
    fens = []
    for path in sorted(Path(a.pgn).glob("*.pgn")) if Path(a.pgn).is_dir() else [Path(a.pgn)]:
        with open(path, errors="replace") as f:
            while (g := chess.pgn.read_game(f)) is not None:
                b = g.board()
                if b.fen() != chess.STARTING_FEN: continue
                for k, m in enumerate(g.mainline_moves()):
                    b.push(m)
                    if 24 <= k < 80 and k % 5 == 0: fens.append((b if b.turn else b.mirror()).fen())
    fens = list(dict.fromkeys(fens))
    np.random.default_rng(0).shuffle(fens)
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

def value(P):
    """(B, 64, hidden) decoder pre-activations -> (B,) P(win) - P(loss)."""
    w = tail((out(torch.nn.functional.gelu(P)) * sd + mu).reshape(-1, mu.shape[0]))
    return w[:, 0] - w[:, 2]

KS = [1, 2, 5, 10, 20, 50, 100]                                            # deletion test: features switched off
rng = np.random.default_rng(0)
dele = {m: np.zeros((2, len(KS))) for m in ["gradient x input", "integrated gradients", "random"]}   # summed dy, positive-first / negative-first
res = []
cnt, tot, tot2 = np.zeros(len(keep)), np.zeros(len(keep)), np.zeros(len(keep))   # per bit, over the tokens where it is on
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
    if a.ig:                                                           # midpoint rule from the square-only token to the position
        al = ((torch.arange(a.ig) + 0.5) / a.ig)[:, None, None]
        P = (base + al * feat).requires_grad_(True)
        value(P).sum().backward()
        gi = P.grad.mean(0)
        with torch.no_grad(): y0 = float(value(base[None])[0])
    wr = full.run(["/output/wdl"], {"/input/planes": encode_fens([r["fen"]])})[0][0]
    squares, flat = {}, {"gradient x input": [], "integrated gradients": []}   # (square, input row, attribution)
    for s in range(64):
        rows_ = E[torch.from_numpy(ins[s])]
        att = (rows_ @ g1[s]).numpy().astype(np.float64)                  # dy/dx * x, x = 1 on these inputs
        b, ident = ins[s][:-1], float(att[-1])                         # the one-hot is last
        flat["gradient x input"] += [(s, j, v) for j, v in zip(b, att[:-1])]
        if a.ig:
            att = (rows_[:-1] @ gi[s]).numpy().astype(np.float64)        # x * mean gradient along the path
            flat["integrated gradients"] += [(s, j, v) for j, v in zip(b, att)]
            ident = None                                               # the one-hot is in the baseline
        else: att = att[:-1]
        cnt[b] += 1; tot[b] += att; tot2[b] += att ** 2
        squares[chess.square_name(s)] = {"square": None if ident is None else round(ident, 5), "bits": [b, att]}
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
    gap_ = [abs(sum(v for q in r["squares"].values() for v in q["bits"][1]) - (r["y decoded"] - r["y baseline"])) for r in res]
    print(f"integrated gradients, {a.ig} steps: |sum - (y - y baseline)| mean {np.mean(gap_):.4f}, max {np.max(gap_):.4f}")
print("deletion test, mean change of y decoded after switching off the k features with the highest (lowest) attribution:")
print("  k                     " + "".join(f"{k:>8}" for k in KS))
for m, v in dele.items():
    if m == "integrated gradients" and not a.ig: continue
    for o, name in enumerate(["highest", "lowest"]):
        print(f"  {m[:20]:20} {name:8}" + "".join(f"{x / len(res):8.3f}" for x in v[o]))
print(f"{len(res)} positions, y decoded vs real: FVU {((Y[:, 0] - Y[:, 1]) ** 2).sum() / ((Y[:, 0] - Y[:, 0].mean()) ** 2).sum():.3f}")
used = np.flatnonzero(cnt)                                                 # bits that occur; the file indexes into these
new = np.full(len(keep), -1); new[used] = np.arange(len(used))
mean = tot[used] / cnt[used]
std = np.sqrt(np.maximum(tot2[used] / cnt[used] - mean ** 2, 0))
for r in res:
    for q in r["squares"].values():
        q["bits"] = [[int(new[j]), round(float(v), 5)] for j, v in zip(*q["bits"])]
tag = "" if TAP == "/attn_body/rehape" else TAP.replace("/", "_")             # e.g. _encoder2_ln2
out_path = ROOT / f"lc0-probe/square_attribution_n{N}{tag}{'_ig' if a.ig else ''}{'_games' if a.pgn else ''}.js"                  # a script, so the demo opens from file://
out_path.write_text("window.ATTRIBUTION = " + json.dumps({
    "n": N, "tap": TAP, "method": f"integrated gradients, {a.ig} steps" if a.ig else "gradient x input", "gap": a.gap, "source": a.pgn or "held-out puzzles", "labels": [labels[j] for j in used],
    "usual": [[round(float(m), 6), round(float(d), 6), int(c)] for m, d, c in zip(mean, std, cnt[used])],   # mean, spread, tokens
    "positions": res}, separators=(",", ":")) + ";\n")
print("wrote", out_path)
