"""Data for hybrid_ig_demo.html: per-square integrated gradients for 10 book positions through two hybrid stacks.

Forward: L5 = layers3-5(layers1-2(D0) + D2) + D5 (hybrid_decoder.py, L0 fitted first). Deepest first: L5 = D5 + M2 + M0,
the end MLPs reading the L2-only and the forward D0 decoder forwarded to L5 (hybrid_endmlp.py). Both read the same
feature bits; IG of W - L for the side to move (Leela's tail on the stack's L5) from each square's any-occupant mean bits to the position,
128 midpoint steps. A square's attribution = the sum over its bits, the absent ones included; written on the real board, signed for White.

  python lc0-probe/hybrid_ig_demo_data.py [--pred]
"""
import argparse, json, subprocess, sys, types
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
ap.add_argument("--pred", action="store_true", help="the end MLPs that also read the prediction they correct")
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
CLAIMS = [1, 3, 14, 11, 22, 23, 30, 31, 33, 42]                             # one per topic: space, pieces, pawns, files, ranks
D = ROOT / "lc0-probe/data"
fwd = torch.load(D / "hybrid_decoder_games_n100000_adamw.pt", weights_only=False)
rev = torch.load(D / "hybrid_reverse_games_n100000_adamw.pt", weights_only=False)
l2 = torch.load(D / "hybrid_decoder_games_n100000_adamw_l2only.pt", weights_only=False)
MM = D / "hybrid_rev_tmp_games_n100000_adamw_d0"
em = torch.load(MM / f"endmlp{'_pred' if a.pred else ''}.pt", weights_only=False)
keep = fwd["keep"]
assert all(np.array_equal(c["keep"], keep) for c in (rev, l2))
K = len(keep); D_in = K + 65

def dense(d):
    """Decoder on dense inputs: X (B, 64, D_in) -> (B*64, 256), the bag sum as a matmul."""
    s = d["state"]; W, b = s["inp.weight"], s["bias"]
    o = torch.nn.Linear(*s["out.weight"].shape[::-1]); o.load_state_dict({"weight": s["out.weight"], "bias": s["out.bias"]})
    for p in o.parameters(): p.requires_grad_(False)
    mu, sd = torch.as_tensor(d["mu"]).float(), float(d["sd"])
    return lambda X: o(torch.nn.functional.gelu((X @ W + b).reshape(-1, W.shape[1]))) * sd + mu
tap = lambda ck, t: dense(next(d for d in ck["decoders"] if d["tap"] == t))
F0d, F2d, F5d = tap(fwd, "/encoder0/ln2"), tap(fwd, "/encoder2/ln2"), tap(fwd, "/encoder5/ln2")
R5, L2d = tap(rev, "/encoder5/ln2"), l2["decoders"][0]
L2d = dense(L2d)

net = onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))
def seg(i, o_):
    m = convert(onnx.utils.Extractor(net).extract_model([i], [o_])).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m
s12, s35, s05, tail = seg("/encoder0/ln2", "/encoder2/ln2"), seg("/encoder2/ln2", "/encoder5/ln2"), \
                      seg("/encoder0/ln2", "/encoder5/ln2"), seg("/encoder5/ln2", "/output/wdl")
with torch.no_grad(): C = s35(s12(torch.zeros(64, 256)))
mu5, sd5 = torch.load(D / "hybrid_rev_streams_games_n100000/norm5.pt")

class MLP(torch.nn.Module):
    def __init__(self, n_in, h):
        super().__init__(); self.a, self.b = torch.nn.Linear(n_in, h), torch.nn.Linear(h, 256)
    def forward(self, x): return self.b(torch.nn.functional.gelu(self.a(x)))
Ms = {}
for k in ("L2", "L0"):
    st = em[k]; Ms[k] = MLP(st["a.weight"].shape[1], em["hidden"]); Ms[k].load_state_dict(st)
    for p in Ms[k].parameters(): p.requires_grad_(False)

def forward_stack(X):
    B = X.shape[0]
    S2 = s12(F0d(X)) + F2d(X)
    return tail(s35(S2) + F5d(X)).reshape(B, 3)
def deep_stack(X):
    B = X.shape[0]; Cb = C.repeat(B, 1)
    x = R5(X) + Cb
    for k, F in (("L2", s35(L2d(X))), ("L0", s05(F0d(X)))):
        inp = F if not a.pred else torch.cat([F, (x - Cb - mu5) / sd5], 1)
        x = x + Ms[k](inp) * sd5
    return tail(x).reshape(B, 3)
wl = lambda w: w[:, 0] - w[:, 2]

# --- the positions' bits (White to move) and the any-occupant baseline
games = {tuple(g["claims"]): g for g in json.load(open(ROOT / "lc0-probe/results_claim_concepts.json"))["games"]}
pick = [next(g for c, g in games.items() if n in c) for n in CLAIMS]
fens = [g["orig_fen"] for g in pick]
white = [(b if b.turn else b.mirror()).fen() for b in map(chess.Board, fens)]
outp = subprocess.run([ROOT / "features/square/target/release/square", "--indices", "--engine",
                       ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2", "--nodes", "200000"],
                      input="\n".join(white) + "\n", capture_output=True, text=True, check=True).stdout.splitlines()
col = np.full(max(keep) + 1 + 100000, -1); col[keep] = np.arange(K)
def xs(squares):
    X = torch.zeros(64, D_in)
    for q, v in squares.items():
        c = col[np.array(v)]; X[chess.parse_square(q), torch.from_numpy(c[c >= 0])] = 1
    X[torch.arange(64), K + torch.arange(64)] = 1                             # square one-hot, White to move: no STM row
    return X
te = np.random.default_rng(0).random(100000) < 0.2
va = np.random.default_rng(1).random(100000) < 0.1
SM, n = torch.zeros(64, D_in), 0
with open(D / "square_bits_games_n100000_nodes200000.jsonl") as f:          # mean bits per square over training boards
    for i, l in enumerate(f):
        if n == 20000: break
        if te[i] or va[i]: continue
        SM += xs(json.loads(l)["squares"]); n += 1
SM /= n

full = ort.InferenceSession(str(ROOT / "vendor/lc0-nets/t1-256x10.onnx"), providers=["CPUExecutionProvider"])
NTOP = 30
LAB = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                capture_output=True, text=True, check=True).stdout)["labels"]
import re
SWAP = {"white": "black", "black": "white", "n": "s", "s": "n", "ne": "se", "se": "ne", "nw": "sw", "sw": "nw",
        "a1h8": "h1a8", "h1a8": "a1h8"}
def unmirror(lab):
    """A label of the mirrored board (Black to move shown as White) named on the real board."""
    def tok(t):
        if re.fullmatch(r"[a-h][1-8]", t): return t[0] + str(9 - int(t[1]))
        if m := re.fullmatch(r"rank([1-8])", t): return f"rank{9 - int(m[1])}"
        if m := re.fullmatch(r"(white|black)(\w+)", t): return SWAP[m[1]] + m[2]
        return SWAP.get(t, t)
    return ".".join("_".join(tok(u) for u in part.split("_")) for part in lab.split("."))
res = []
for g, fen, wf, l in zip(pick, fens, white, outp):
    r = json.loads(l); assert r["fen"] == wf
    X = xs(r["squares"])
    w = full.run(["/output/wdl"], {"/input/planes": encode_fens([wf])})[0][0]; y_real = float(w[0] - w[2])
    mir = not chess.Board(fen).turn                                        # shown for White: Black to move is mirrored back
    sg, sq_of = (-1, chess.square_mirror) if mir else (1, lambda s: s)
    row = {"fen": fen, "claim": g["claim"], "source": g["source"], "y": sg * y_real}; As = {}
    for name, f in (("fwd", forward_stack), ("deep", deep_stack)):
        al = ((torch.arange(a.steps) + 0.5) / a.steps)[:, None, None]
        P = (SM + al * (X - SM)).requires_grad_(True)
        wl(f(P)).sum().backward()
        A = (X - SM) * P.grad.mean(0)                                         # per square and bit
        att = A.sum(1)                                                        # per square, absent bits included
        with torch.no_grad(): y, y0 = float(wl(f(X[None]))[0]), float(wl(f(SM[None]))[0])
        print(f"{name:4} claim {g['claims']}: y {y:+.3f} (Leela {y_real:+.3f}), baseline {y0:+.3f}, "
              f"completeness gap {abs(float(att.sum()) - (y - y0)):.4f}", flush=True)
        sq = [0.0] * 64
        for s in range(64): sq[sq_of(s)] = round(sg * float(att[s]), 4)     # real board, for White
        row[name] = {"y": sg * y, "y0": sg * y0, "sq": sq}
        As[name] = sg * A[:, :K]
    # top bits by either setup or by their difference, named on the real board
    top = set()
    for v in (As["fwd"], As["deep"], As["fwd"] - As["deep"]):
        top |= set(map(tuple, np.argwhere(v.abs().numpy() >= np.sort(v.abs().numpy(), None)[-NTOP])))
    row["feat"] = sorted(({"sq": int(sq_of(int(s))), "f": unmirror(LAB[keep[b]]) if mir else LAB[keep[b]], "on": bool(X[s, b] > 0),
                           "fwd": round(float(As["fwd"][s, b]), 4), "deep": round(float(As["deep"][s, b]), 4)}
                          for s, b in top), key=lambda r: -abs(r["fwd"] - r["deep"]))
    res.append(row)
out = ROOT / f"lc0-probe/hybrid_ig_demo{'_pred' if a.pred else ''}.js"
out.write_text("window.HYB = " + json.dumps(res) + ";\n"); print("wrote", out)
