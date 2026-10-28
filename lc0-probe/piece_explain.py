"""Which named features of a piece carry its placement, through the per-square decoder at /encoder0/ln2.

For each claim piece of results_piece_placement.json, two boards: the piece on its real square and on its median
safe square (the one its ceiling uses). Both are mirrored so the side to move is White and get their feature bits
from features/square --engine. Players are groups of bits about the piece; each player switches from the
median-square board's bits to the real board's:
  occupant   its occupant bit
  secure     on its square: pawn_cover, stability, strong, weak
  activity   on its square: control, mobility
  relations  on its square: role, bishop_pair, unlike_bishops
  line       on its square: blocks, blocked_by, xray, rook
  route      on its square: route
  square     on its square: square, centralisation, open_file, half_open_file, long_diagonal
  elsewhere  role bits on other squares that name its square (other pieces defending, attacking, needing it)
  rest       every other bit (tactics, material, the other squares), switched with its board as one block
Game value: decoder win - loss for the piece's owner. Exact Shapley over the 9 players (512 coalitions); the values
sum to the decoder's difference real - median. Fidelity: that difference against real Leela's v(real) - v(median).
Named share: the sum over secure..elsewhere divided by real Leela's difference (|difference| >= --min only).
Corpus fidelity: the same two-board difference, decoder against Leela, for every corpus piece with a safe alternative.

  python lc0-probe/piece_explain.py --stage bits --jobs 2
  python lc0-probe/piece_explain.py --stage explain
"""
import argparse, itertools, json, subprocess, sys, types
from math import factorial
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["bits", "explain"])
ap.add_argument("--nodes", type=int, default=200000)
ap.add_argument("--jobs", type=int, default=2, help="bits: parallel square processes (one engine thread each)")
ap.add_argument("--tap", default="/encoder0/ln2")
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--min", type=float, default=0.05, help="named share only where |real Leela difference| >= this")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
PLACE = ROOT / "lc0-probe/results_piece_placement.json"
NORMS = ROOT / "lc0-probe/data/piece_placement_norms.json"
CORPUS = ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_games.js"
BITS = ROOT / "lc0-probe/data/piece_explain_bits.jsonl"
OUT = ROOT / "lc0-probe/results_piece_explain.json"
JS = ROOT / "lc0-probe/piece_explain_textbook.js"
SQUARE = ROOT / "features/square/target/release/square"
SF = ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2"
GROUPS = ["occupant", "secure", "activity", "relations", "line", "route", "square", "elsewhere", "rest"]
NAMED = GROUPS[1:8]
FAM = {"occupant": "occupant", "pawn_cover": "secure", "stability": "secure", "strong": "secure", "weak": "secure",
       "control": "activity", "mobility": "activity", "role": "relations", "bishop_pair": "relations",
       "unlike_bishops": "relations", "blocks": "line", "blocked_by": "line", "xray": "line", "rook": "line",
       "route": "route", "square": "square", "centralisation": "square", "open_file": "square",
       "half_open_file": "square", "long_diagonal": "square"}

def clean(c):
    c.castling_rights = c.clean_castling_rights(); c.ep_square = None
    return c

def pair(fen, sq, med):
    """(real board, median-square board) mirrored to White to move, the piece's square on each, owner sign."""
    b = chess.Board(fen); s0, t = chess.parse_square(sq), chess.parse_square(med)
    c = b.copy(stack=False); p = c.remove_piece_at(s0); c.set_piece_at(t, p); clean(c)
    own = 1 if p.color == b.turn else -1
    if b.turn: return b.fen(), c.fen(), s0, t, own
    return b.mirror().fen(), c.mirror().fen(), chess.square_mirror(s0), chess.square_mirror(t), own

def jobs():
    """Every (key, fen, sq, median square) to explain: claims, then corpus pieces."""
    P = json.loads(PLACE.read_text())["claims"]
    out = [(f"claim {i}", r["fen"], r["square"], r["median square"]) for i, r in enumerate(P) if r["median square"]]
    fens = [p["fen"] for p in json.loads(CORPUS.read_text().split("=", 1)[1].rstrip().rstrip(";"))["positions"]]
    for i, r in enumerate(json.loads(NORMS.read_text())["pieces"]):
        if r["median square"]: out.append((f"corpus {i}", fens[r["pos"]], r["square"], r["median square"]))
    return out

if a.stage == "bits":
    have = {json.loads(l)["in"] for l in open(BITS)} if BITS.exists() else set()
    need = list(dict.fromkeys(f for _, fen, sq, med in jobs() for f in pair(fen, sq, med)[:2] if f not in have))
    print(f"{len(need)} boards to featurise", flush=True)
    chunks = [need[k::a.jobs] for k in range(a.jobs)]
    tmp = [BITS.with_name(f"{BITS.stem}.part{k}") for k in range(a.jobs)]
    for t, c in zip(tmp, chunks): t.with_name(t.name + ".in").write_text("\n".join(c) + "\n")
    procs = [subprocess.Popen([SQUARE, "--indices", "--engine", SF, "--nodes", str(a.nodes)],
                              stdin=open(t.with_name(t.name + ".in")), stdout=open(t, "w")) for t in tmp]
    for p in procs: p.wait()
    with open(BITS, "a") as f:
        for t in tmp:
            for l in open(t):
                r = json.loads(l); f.write(json.dumps({"in": r["fen"], "squares": r["squares"]}) + "\n")
    sys.exit()

import onnx, torch
from onnx2torch import convert
torch.set_num_threads(a.threads)
tag = a.tap.replace("/", "_")
ck = torch.load(ROOT / f"lc0-probe/data/square_decoder{tag}_n{a.n}.pt", weights_only=False)
keep, mu, sd = ck["keep"], torch.from_numpy(ck["mu"]).float(), float(ck["sd"])
schema = json.loads(subprocess.run([SQUARE, "--schema"], capture_output=True, text=True, check=True).stdout)
labels = schema["labels"]
col = np.full(schema["bits_per_square"], -1); col[keep] = np.arange(len(keep))
E, bias = ck["state"]["inp.weight"], ck["state"]["bias"]
out = torch.nn.Linear(*ck["state"]["out.weight"].shape[::-1])
out.load_state_dict({"weight": ck["state"]["out.weight"], "bias": ck["state"]["out.bias"]})
sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))) \
    .extract_model([a.tap], ["/output/wdl"])
tail = convert(sub).eval()
for p in tail.parameters(): p.requires_grad_(False)
base = E[len(keep):] + bias

@torch.no_grad()
def value(P):
    ys = []
    for s in range(0, len(P), 128):
        w = tail((out(torch.nn.functional.gelu(P[s:s + 128])) * sd + mu).reshape(-1, mu.shape[0]))
        ys.append(w[:, 0] - w[:, 2])
    return torch.cat(ys).numpy()

BIT = {}
for l in open(BITS):
    r = json.loads(l); BIT[r["in"]] = r["squares"]

def parts(fen, p):
    """(groups, 64, hidden): the board's decoder rows summed per player, for the piece on square p."""
    G = torch.zeros(len(GROUPS), 64, base.shape[1]); n = np.zeros(len(GROUPS), int)
    name = chess.square_name(p)
    for sq, idx in BIT[fen].items():
        s = chess.parse_square(sq)
        for i in idx:
            if col[i] < 0: continue
            L = labels[i]; f = L.split(".")[0]
            g = FAM.get(f, "rest") if s == p else "elsewhere" if f == "role" and L.rsplit(".", 1)[1] == name else "rest"
            k = GROUPS.index(g); G[k, s] += E[col[i]]; n[k] += 1
    return G, n

W = {k: factorial(k) * factorial(len(GROUPS) - k - 1) / factorial(len(GROUPS)) for k in range(len(GROUPS))}
COAL = list(itertools.product([0, 1], repeat=len(GROUPS)))
def explain(fen, sq, med):
    fr, fm, p, t, own = pair(fen, sq, med)
    Gr, nr = parts(fr, p); Gm, nm = parts(fm, t)
    M = torch.tensor(COAL, dtype=torch.float32)                            # 1 = the real board's bits
    P = base + torch.einsum("cg,ghd->chd", M, Gr) + torch.einsum("cg,ghd->chd", 1 - M, Gm)
    v = dict(zip(COAL, own * value(P)))
    phi = np.zeros(len(GROUPS))
    for S in COAL:
        for i in range(len(GROUPS)):
            if S[i]: continue
            T = S[:i] + (1,) + S[i + 1:]; phi[i] += W[sum(S)] * (v[T] - v[S])
    return float(v[COAL[-1]] - v[COAL[0]]), dict(zip(GROUPS, phi.round(5).tolist())), \
           {"bits real": dict(zip(GROUPS, nr.tolist())), "bits median": dict(zip(GROUPS, nm.tolist()))}

def two(fen, sq, med):
    fr, fm, p, t, own = pair(fen, sq, med)
    P = torch.stack([base + parts(fr, p)[0].sum(0), base + parts(fm, t)[0].sum(0)])
    y = own * value(P); return float(y[0] - y[1])

def fid(x, y):
    x, y = np.asarray(x), np.asarray(y); rk = lambda z: np.argsort(np.argsort(z))
    big = np.abs(y) >= a.min
    return {"n": len(x), "pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(np.corrcoef(rk(x), rk(y))[0, 1]),
            "sign agreement (|leela| >= min)": float(np.mean(np.sign(x[big]) == np.sign(y[big]))),
            "mean |decoder - leela|": float(np.abs(x - y).mean()), "slope": float(np.polyfit(y, x, 1)[0])}

P = json.loads(PLACE.read_text())["claims"]
res = []
for r in P:
    e = {k: r[k] for k in ("src", "game", "piece", "square", "meaning", "sign", "verdict", "median square")}
    if r["median square"]:
        real = r["v"] - r["v median"]
        d, phi, nb = explain(r["fen"], r["square"], r["median square"])
        top = sorted(NAMED, key=lambda g: -abs(phi[g]))[:3]
        e.update({"leela delta": real, "decoder delta": d, "phi": phi, "top": [[g, phi[g]] for g in top],
                  "named share": sum(phi[g] for g in NAMED) / real if abs(real) >= a.min else None, **nb})
    res.append(e)
    print(e["src"], e["piece"] + e["square"], {k: e.get(k) for k in ("leela delta", "decoder delta", "named share")}, flush=True)
N = json.loads(NORMS.read_text())["pieces"]
fens = [p["fen"] for p in json.loads(CORPUS.read_text().split("=", 1)[1].rstrip().rstrip(";"))["positions"]]
cx, cy = [], []
for r in N:
    if r["median square"]: cx.append(two(fens[r["pos"]], r["square"], r["median square"])); cy.append(r["v"] - r["v median"])
ok = [e for e in res if "decoder delta" in e]
out_ = {"tap": a.tap, "groups": {g: [f for f, h in FAM.items() if h == g] for g in GROUPS[:7]},
        "fidelity corpus": fid(cx, cy), "fidelity claims": fid([e["decoder delta"] for e in ok], [e["leela delta"] for e in ok]),
        "claims": res}
OUT.write_text(json.dumps(out_, indent=1))
JS.write_text("window.PIECE_EXPLAIN = " + json.dumps(out_) + ";\n")
print(json.dumps({k: out_[k] for k in ("fidelity corpus", "fidelity claims")}, indent=1))
