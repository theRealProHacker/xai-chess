"""Colour-swap Shapley values of concepts, through the per-square decoder and Leela: who has more of X.

Baseline: the same board with the colours swapped (mirrored, White still to move). v(S) = P(win) - P(loss) of the
rest of the net on decoded tokens whose concept-S bits come from the real board and all other bits from the swapped
one. Both endpoints are real positions, so the values sum to v(real) - v(swapped), and phi_c > 0 means the asymmetry
in c favours White (the side to move). Six concepts, 2^6 coalitions, exact. With --kingzone two more players, one
per king: the square, pawn_cover, pawn_shield and king.castle bits on the 3x3 around it, and enemy control.reach
bits into it, taken out of their concepts (2^8 coalitions).

Features come from the square binary with --engine for both boards, cached in data/. Positions are the 300 quiet
game positions of square_attribution.py --pgn; a position is skipped when its swapped board is illegal (the side to
move is in check, so after the swap the side not to move is). Per position: phi, leave-one-out from the real board,
Leela's y on both boards, and board facts (White minus Black) to check the signs against. A row is said if |phi|
clears --min-phi and --min-share and leave-one-out agrees in sign without saturating; "unexplained" is the decoded
asymmetry no said row covers, kept apart from the decoder's gap to the real one.

  python lc0-probe/square_swap_shap.py
  python lc0-probe/square_swap_shap.py --kingzone
  python lc0-probe/square_swap_shap.py --kingzone --summary      # redo the summary only
"""
import argparse, itertools, json, math, subprocess, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe"), str(ROOT / "commentary_gen")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, onnxruntime as ort, torch
from onnx2torch import convert
from lc0_planes import encode_fens
import positional

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--tap", default="/encoder0/ln2")
ap.add_argument("--nodes", type=int, default=200000, help="Stockfish nodes per line, as for the corpus features")
ap.add_argument("--positions", type=int, default=300)
ap.add_argument("--kingzone", action="store_true", help="add a king-zone player per king")
ap.add_argument("--min-phi", type=float, default=0.2, help="|phi| below this gets no sentence")
ap.add_argument("--min-share", type=float, default=0.15, help="nor below this share of |v(real) - v(swapped)|")
ap.add_argument("--summary", action="store_true", help="only redo the summary of an existing run")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
tag = a.tap.replace("/", "_")
SQ = ROOT / "features/square/target/release/square"
SF = ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2"

ck = torch.load(ROOT / f"lc0-probe/data/square_decoder{tag}_n{a.n}.pt", weights_only=False)
keep, mu, sd = ck["keep"], torch.from_numpy(ck["mu"]).float(), float(ck["sd"])
schema = json.loads(subprocess.run([SQ, "--schema"], capture_output=True, text=True, check=True).stdout)
col = {schema["labels"][b]: i for i, b in enumerate(keep)}                 # label -> decoder input row
E, bias = ck["state"]["inp.weight"], ck["state"]["bias"]
out = torch.nn.Linear(*ck["state"]["out.weight"].shape[::-1])
out.load_state_dict({"weight": ck["state"]["out.weight"], "bias": ck["state"]["out.bias"]})
net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(net_path))).extract_model([a.tap], ["/output/wdl"])
tail = convert(sub).eval()
base = E[len(keep):] + bias                                                # (64, hidden): square one-hot only
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
full = ort.InferenceSession(str(net_path), so, providers=["CPUExecutionProvider"])

@torch.no_grad()
def value(P):
    """(B, 64, hidden) pre-activations -> (B,) P(win) - P(loss), in chunks."""
    ys = []
    for s in range(0, len(P), 128):
        w = tail((out(torch.nn.functional.gelu(P[s:s + 128])) * sd + mu).reshape(-1, mu.shape[0]))
        ys.append(w[:, 0] - w[:, 2])
    return torch.cat(ys)

CONCEPT = {**dict.fromkeys(["occupant", "material", "bishop_pair", "unlike_bishops"], "material"),
           **dict.fromkeys(["control", "square", "centralisation", "long_diagonal"], "control"),
           **dict.fromkeys(["mobility", "route", "blocked_by", "blocks", "xray"], "activity"),
           **dict.fromkeys(["pawn", "pawn_cover", "weak", "strong", "open_file", "half_open_file", "rook"], "pawns"),
           **dict.fromkeys(["pawn_shield", "king", "check"], "king"),
           **dict.fromkeys(["role", "threat", "material_win", "stability", "mate"], "tactics")}
missing = {l.split(".")[0] for l in schema["labels"]} - set(CONCEPT)
assert not missing, f"families in no concept: {missing}"
PLAYERS = sorted(set(CONCEPT.values())) + (["king zone W", "king zone B"] if a.kingzone else [])
ZONE_FAM = ("square", "pawn_cover", "pawn_shield", "king")

def zones(board):
    """Square -> zone player, for the 3x3 around each king (as positional._zone; White's first on overlap)."""
    z = {}
    for c, name in ((chess.BLACK, "king zone B"), (chess.WHITE, "king zone W")):
        for s in positional._zone(board, c): z[s] = name
    return z

def player(label, s, board, z):
    """The player a bit (label on square s) belongs to."""
    f = label.split(".")[0]
    if z:
        if f in ZONE_FAM and s in z: return z[s]
        if label.startswith("control.reach."):
            t, pc = chess.parse_square(label.split(".")[2]), board.piece_at(s)
            if t in z and pc and (z[t] == "king zone W") != pc.color: return z[t]
    return CONCEPT[f]

def parts(squares, board):
    """(players, 64, hidden): each player's decoder rows, summed per square."""
    z = zones(board) if a.kingzone else {}
    pi, sq, rw = [], [], []
    for name, ls in squares.items():
        s = chess.parse_square(name)
        for l in ls:
            if l in col: pi.append(PLAYERS.index(player(l, s, board, z))); sq.append(s); rw.append(col[l])
    P = torch.zeros(len(PLAYERS) * 64, base.shape[1])
    P.index_add_(0, torch.tensor(pi) * 64 + torch.tensor(sq), E[torch.tensor(rw)])
    return P.view(len(PLAYERS), 64, -1)

def facts(b):
    """Board facts, White minus Black (king: White's safety minus Black's)."""
    def side(c):
        att = chess.SquareSet()
        for s in chess.SQUARES:
            if b.color_at(s) == c: att |= b.attacks(s)
        bb = b.copy(stack=False); bb.turn = c; bb.ep_square = None
        moves = sum(1 for m in bb.pseudo_legal_moves if bb.piece_type_at(m.from_square) not in (chess.PAWN, chess.KING))
        pw = b.pieces(chess.PAWN, c)
        files = [chess.square_file(s) for s in pw]
        iso = sum(1 for f in files if f - 1 not in files and f + 1 not in files)
        dbl = len(files) - len(set(files))
        passed = sum(1 for s in pw if not any(
            b.piece_at(t) == chess.Piece(chess.PAWN, not c) for t in chess.SQUARES
            if abs(chess.square_file(t) - chess.square_file(s)) <= 1
            and (chess.square_rank(t) > chess.square_rank(s) if c else chess.square_rank(t) < chess.square_rank(s))))
        mat = sum(len(b.pieces(t, c)) * v for t, v in ((1, 1), (2, 3), (3, 3), (4, 5), (5, 9)))
        return {"attacked": len(att), "moves": moves, "material": mat, "pawns": passed - iso - dbl}
    W, B = side(chess.WHITE), side(chess.BLACK)
    sp = positional.space_data(b)
    ks = positional.king_safety_data(b)
    k = {s: (len(ks[s]["shield"]), len(ks[s]["enemy_pieces_bearing_on_the_zone"]), len(ks[s]["flight_squares"]))
         for s in ("White", "Black")}
    return {"control": W["attacked"] - B["attacked"],
            "space": sum(sp["White"]["squares_held_in_the_far_half"].values()) - sum(sp["Black"]["squares_held_in_the_far_half"].values()),
            "activity": W["moves"] - B["moves"], "material": W["material"] - B["material"], "pawns": W["pawns"] - B["pawns"],
            "king shield": k["White"][0] - k["Black"][0], "king attackers": k["Black"][1] - k["White"][1],
            "king flight": k["White"][2] - k["Black"][2]}

src = ROOT / f"lc0-probe/square_attribution_n{a.n}{tag}_ig_games.js"      # the 300 quiet game positions
A = json.loads(src.read_text().split("=", 1)[1].rstrip().rstrip(";"))
pos, skipped = [], []
for p in A["positions"][:a.positions]:
    b = chess.Board(p["fen"]); sw = b.mirror(); sw.turn = chess.WHITE; sw.ep_square = None
    (pos if sw.is_valid() else skipped).append((p, b, sw))
cache = ROOT / f"lc0-probe/data/square_swap_bits_nodes{a.nodes}.jsonl"   # engine lines for both boards
F = {r["fen"]: r for r in map(json.loads, cache.read_text().splitlines())} if cache.exists() else {}
todo = [f for _, b, sw in pos for f in (b.fen(), sw.fen()) if f not in F]
if todo:
    o = subprocess.run([SQ, "--engine", SF, "--nodes", str(a.nodes)], input="\n".join(todo) + "\n",
                       capture_output=True, text=True, check=True).stdout
    with open(cache, "a") as f: f.write(o)
    F.update({r["fen"]: r for r in map(json.loads, o.splitlines())})
assert not any("error" in F[f] for _, b, sw in pos for f in (b.fen(), sw.fen()))

n = len(PLAYERS)
masks = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float32)
key = {tuple(r): k for k, r in enumerate(masks.astype(int).tolist())}
w = [math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)]
M = torch.from_numpy(masks)[:, :, None, None]
ALL, NONE = key[(1,) * n], key[(0,) * n]
name = f"square_swap_shap{tag}{'_kingzone' if a.kingzone else ''}"
if a.summary:
    res = json.load(open(ROOT / f"lc0-probe/results_{name}.json"))["positions"]
else:
    res = []
    for i, (p, b, sw) in enumerate(pos):
        X, Y = parts(F[b.fen()]["squares"], b), parts(F[sw.fen()]["squares"], sw)
        V = torch.cat([value(base + (M[s:s + 64] * X + (1 - M[s:s + 64]) * Y).sum(1)) for s in range(0, len(M), 64)]).numpy()
        phi = np.zeros(n)
        for k, r in enumerate(masks.astype(int).tolist()):
            for j in range(n):
                if not r[j]: phi[j] += w[sum(r)] * (V[key[tuple(r[:j] + [1] + r[j + 1:])]] - V[k])
        loo = [V[key[tuple(int(j != c) for j in range(n))]] for c in range(n)]  # real board with player c swapped
        wr = full.run(["/output/wdl"], {"/input/planes": encode_fens([b.fen(), sw.fen()])})[0]
        res.append({"fen": b.fen(), "swapped": sw.fen(), "y": float(wr[0, 0] - wr[0, 2]), "y swapped": float(wr[1, 0] - wr[1, 2]),
                    "v real": float(V[ALL]), "v swapped": float(V[NONE]),
                    "phi": dict(zip(PLAYERS, phi.round(5).tolist())),
                    "loo": {c: round(float(V[ALL] - loo[j]), 5) for j, c in enumerate(PLAYERS)},
                    "v loo": {c: round(float(loo[j]), 5) for j, c in enumerate(PLAYERS)},
                    "facts": facts(b)})
        if (i + 1) % 25 == 0: print(f"{i + 1} positions", flush=True)

# A row gets a sentence if |phi| clears both floors and leave-one-out agrees in sign without saturating (|v| > SAT
# is off the board's manifold). The board fact only picks the phrasing: agrees, or Leela's view against the count.
SAT, BANDS = 0.98, ((0.8, "clearly"), (0.4, ""), (0.0, "slightly"))
FACT = {"control": "control", "activity": "activity", "material": "material", "pawns": "pawns",
        "king": "king", "king zone W": "king", "king zone B": "king"}
KF = ("king shield", "king attackers", "king flight")
for r in res:
    f, d = r["facts"], r["v real"] - r["v swapped"]
    f["king"] = int(sum(np.sign(f[k]) for k in KF))                       # White's king safer: votes of the three
    r["said"] = []
    for c in PLAYERS:
        p = r["phi"][c]
        if abs(p) < max(a.min_phi, a.min_share * abs(d)) or np.sign(r["loo"][c]) != np.sign(p) or abs(r["v loo"][c]) > SAT:
            continue
        fv = f.get(FACT.get(c))
        r["said"].append({"concept": c, "side": "White" if p > 0 else "Black", "phi": p,
                          "band": next(b for t, b in BANDS if abs(p) >= t),
                          "board": None if fv is None else "level" if fv == 0 else "agrees" if np.sign(fv) == np.sign(p) else "disagrees"})
    r["unexplained"] = round(d - sum(s["phi"] for s in r["said"]), 5)     # asymmetry no sentence covers
    r["decoder gap"] = round(r["y"] - r["y swapped"] - d, 5)              # real minus decoded asymmetry

def agree(ph, fv, m):
    m = m & (fv != 0)
    return {"rate": round(float(np.mean(np.sign(ph[m]) == np.sign(fv[m]))), 3) if m.any() else None, "n": int(m.sum())}
dec = np.array([r["v real"] - r["v swapped"] for r in res]); real = np.array([r["y"] - r["y swapped"] for r in res])
un = np.array([r["unexplained"] for r in res]); big = np.abs(dec) >= 0.1; sh = np.abs(un[big]) / np.abs(dec[big])
summ = {"min phi": a.min_phi, "min share": a.min_share, "saturated": SAT, "bands": BANDS, "concepts": {},
        "sentences per position": np.bincount([len(r["said"]) for r in res]).tolist(),
        "unexplained |.|": np.percentile(np.abs(un), [25, 50, 75]).round(3).tolist(),
        "unexplained share of |v real - v swapped| (>= 0.1)": {"n": int(big.sum()), "quartiles": np.percentile(sh, [25, 50, 75]).round(3).tolist(),
                                                          "over 0.5": round(float(np.mean(sh > 0.5)), 3)},
        "decoder gap |.|": np.percentile(np.abs(real - dec), [25, 50, 75]).round(3).tolist()}
print(f"{len(res)} positions, {len(skipped)} skipped (swapped board illegal); asymmetry decoded vs real: "
      f"r {np.corrcoef(dec, real)[0, 1]:.3f}, median |gap| {summ['decoder gap |.|'][1]:.3f}, median |real| {np.median(np.abs(real)):.3f}")
for c in PLAYERS:
    ph = np.array([r["phi"][c] for r in res])
    said = np.array([any(s["concept"] == c for s in r["said"]) for r in res])
    sat = np.array([abs(r["v loo"][c]) > SAT for r in res])
    s = {"phi quantiles 5 25 50 75 95": np.percentile(ph, [5, 25, 50, 75, 95]).round(3).tolist(),
         "|phi| median": round(float(np.median(np.abs(ph))), 3), "said": round(float(said.mean()), 3),
         "loo saturated": round(float(sat.mean()), 3)}
    if c in FACT:
        for k in [FACT[c]] + (["space"] if c == "control" else list(KF) if FACT[c] == "king" else []):
            fv = np.array([r["facts"][k] for r in res])
            s[f"agrees with {k}"] = {"all": agree(ph, fv, np.ones(len(res), bool)), "said": agree(ph, fv, said)}
    summ["concepts"][c] = s
    print(f"  {c:12} |phi| median {s['|phi| median']:.3f}  said {s['said']:.2f}  loo saturated {s['loo saturated']:.2f}  " +
          "  ".join(f"{k[12:]} {v['all']['rate']}/{v['said']['rate']} (n {v['all']['n']}/{v['said']['n']})" for k, v in s.items() if k.startswith("agrees")))
if a.kingzone:
    ph = np.array([r["phi"]["king zone W"] + r["phi"]["king zone B"] for r in res])
    summ["king zones summed"] = {k: agree(ph, np.array([r["facts"][k] for r in res]), np.ones(len(res), bool)) for k in ("king",) + KF}
    print("  both king zones, agreement:", {k: v["rate"] for k, v in summ["king zones summed"].items()})
print(f"  sentences per position {summ['sentences per position']}; unexplained share quartiles "
      f"{summ['unexplained share of |v real - v swapped| (>= 0.1)']['quartiles']}, over 0.5 {summ['unexplained share of |v real - v swapped| (>= 0.1)']['over 0.5']}")

out_ = {"tap": a.tap, "nodes": a.nodes, "players": PLAYERS, "skipped": [p["fen"] for p, _, _ in skipped], "summary": summ, "positions": res}
json.dump(out_, open(ROOT / f"lc0-probe/results_{name}.json", "w"))
(ROOT / f"lc0-probe/{name}.js").write_text("window.SWAP = " + json.dumps(out_, separators=(",", ":")) + ";\n")
