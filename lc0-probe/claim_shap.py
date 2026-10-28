"""Shapley over the features each annotator claim names (claim_players.md), through the /encoder0/ln2 decoder.

Players are the feature bits of the squares a claim's quote names. A player is never removed: switching it off
replaces its bits by the average bits of the same occupant (or emptiness) on the same square, over --mean training
positions; everything else stays on. Per claim: exact Shapley over its players (the decomposition) and, per player,
phi = v(all) - v(player averaged), ranked against the same kind of player in the 300 corpus positions. Squares in SPEC are real-board squares; positions are stored White to move,
so a Black-to-move claim is mirrored (squares ^ 56, colours swapped).

  python lc0-probe/claim_shap.py
"""
import argparse, itertools, json, math, subprocess, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, torch
from onnx2torch import convert

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--empty", type=int, default=6, help="empty squares sampled per corpus position for the 'empty' kind")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--mean", type=int, default=20000, help="training positions averaged for the off state")
ap.add_argument("--claims-only", action="store_true", help="print the claim games (pieces on the players' squares, phi) and stop")
a = ap.parse_args()
torch.set_num_threads(a.threads)
rng = np.random.default_rng(0)

# claim (1-based index into square_shap_textbook.js) -> players (name, selector, kind, rule); see claim_players.md
# selectors: sq "e4 f4" | centre | shield "g8" | pieces "w" | occ "c4 c6" | threat "d6"; rules: gen, low, neg, notpos, zero
P = lambda name, sel, arg, kind, rule="gen": (name, sel, arg, kind, rule)
SPEC = {
    (1,): [P("pawns e4 f4", "sq", "e4 f4", "advanced"), P("knight d4", "sq", "d4", "N")],
    (2,): [P("pawns e4 f5", "sq", "e4 f5", "advanced"), P("knight e5", "sq", "e5", "N")],
    (3,): [P("pawns d5 f4", "sq", "d5 f4", "advanced"), P("queen d4", "sq", "d4", "Q")],
    (4,): [P("pawns b4 c3", "sq", "b4 c3", "advanced")],
    (5,): [P("W pawns a3 b4 c4 d5", "sq", "a3 b4 c4 d5", "advanced"), P("B queenside b6 c6 d6 b7 d7", "sq", "b6 c6 d6 b7 d7", "pieces")],
    (6,): [P("centre", "centre", "", "centre")],
    (7,): [P("centre", "centre", "", "centre")],
    (8, 43): [P("d5", "sq", "d5", "empty")],
    (9, 12): [P("bishops c8 f8", "sq", "c8 f8", "pair"), P("knight e4", "sq", "e4", "N")],
    (10,): [P("bishops e3 b3", "sq", "e3 b3", "pair"), P("knights d7 e5", "sq", "d7 e5", "pieces")],
    (11,): [P("bishops a6 b6", "sq", "a6 b6", "pair"), P("bishop b4 knight c6", "sq", "b4 c6", "pieces")],
    (13,): [P("knight c3", "sq", "c3", "N"), P("pawn b4", "sq", "b4", "pawn")],
    (14, 40): [P("knight f4", "sq", "f4", "N"), P("bishops c8 g7", "sq", "c8 g7", "pair")],
    (15,): [P("W king e1", "shield", "e1", "shield")],
    (16,): [P("W king c1", "shield", "c1", "shield")],
    (17,): [P("B king g8", "shield", "g8", "shield")],
    (18,): [P("W king g2", "shield", "g2", "shield"), P("B king g8", "shield", "g8", "shield")],
    (19,): [P("B king g8", "shield", "g8", "shield")],
    (20,): [P("W king d2", "shield", "d2", "shield")],
    (21,): [P("passed e4", "sq", "e4", "passed"), P("king e3", "sq", "e3", "K")],
    (22,): [P("passed d6", "threat", "d6", "passed+threat")],
    (23, 32): [P("passed g5", "sq", "g5", "passed"), P("rook h7", "sq", "h7", "R")],
    (24,): [P("isolated a5 d6 f5", "sq", "a5 d6 f5", "isolated")],
    (25,): [P("isolated a5 c7 e5 e6", "sq", "a5 c7 e5 e6", "isolated")],
    (26,): [P("doubled c6 c7", "sq", "c6 c7", "doubled")],
    (27,): [P("doubled c4 c6", "occ", "c4 c6", "doubled+occ", "zero")],
    (28,): [P("isolated d5", "sq", "d5", "isolated1")],
    (29,): [P("isolated h7", "sq", "h7", "isolated1")],
    (30,): [P("rook d8", "sq", "d8", "R"), P("king g8", "sq", "g8", "K")],
    (31,): [P("rook a7", "sq", "a7", "R"), P("W pawns e3 g2", "sq", "e3 g2", "isolated", "neg")],
    (33,): [P("rook a2", "sq", "a2", "R", "low")],
    (34,): [P("W pieces", "pieces", "w", "pieces"), P("B pieces", "pieces", "b", "pieces")],
    (35,): [P("W pieces", "pieces", "w", "pieces"), P("B pieces", "pieces", "b", "pieces")],
    (36,): [P("B pieces", "pieces", "b", "pieces"), P("W pieces", "pieces", "w", "pieces")],
    (37,): [P("W pieces", "pieces", "w", "pieces"), P("B pieces", "pieces", "b", "pieces")],
    (38,): [P("W pieces", "pieces", "w", "pieces"), P("B pieces", "pieces", "b", "pieces")],
    (39,): [P("pawns b4 c5 a6", "sq", "b4 c5 a6", "advanced")],
    (41,): [P("bishop b7", "sq", "b7", "B", "low"), P("bishops b7 e7", "sq", "b7 e7", "pair", "notpos")],
    (42,): [P("knight d5", "sq", "d5", "N")],
}
COMPARE = {(11,): (0, 1), (14, 40): (0, 1)}                    # joint-game s(A) > s(B), both signed for their owner
STANDIN = {34, 35, 36, 37, 38}                                 # development: piece-activity stand-in

L = lambda f: json.loads(Path(f).read_text().split("=", 1)[1].rstrip().rstrip(";"))
TB = L(ROOT / "lc0-probe/square_shap_textbook.js")["positions"]
srcs = [L(ROOT / f"lc0-probe/square_attribution_n{a.n}_encoder0_ln2_ig_textbook{s}.js") for s in ("", "2")]
by_fen = {p["fen"]: (p, A["labels"]) for A in srcs for p in A["positions"]}
G = L(ROOT / f"docs/square_attribution_n{a.n}_encoder0_ln2_ig_games.js")
assert all(p["fen"].split()[1] == "w" for p in G["positions"])

# --- decoder and Leela, as square_shap.py
schema = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                   capture_output=True, text=True, check=True).stdout)
inferred = onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))
tail = convert(onnx.utils.Extractor(inferred).extract_model(["/encoder0/ln2"], ["/output/wdl"])).eval()
for q in tail.parameters(): q.requires_grad_(False)
ck = torch.load(ROOT / f"lc0-probe/data/square_decoder_encoder0_ln2_n{a.n}.pt", weights_only=False)
keep, E, bias = ck["keep"], ck["state"]["inp.weight"], ck["state"]["bias"]
out = torch.nn.Linear(*ck["state"]["out.weight"].shape[::-1])
out.load_state_dict({"weight": ck["state"]["out.weight"], "bias": ck["state"]["out.bias"]})
mu, sd = torch.as_tensor(ck["mu"]).float(), float(ck["sd"])
base = E[len(keep):len(keep) + 64] + bias                                  # (64, hidden): square one-hot only
col = {schema["labels"][b]: i for i, b in enumerate(keep)}

@torch.no_grad()
def value(Ps):
    """(B, 64, hidden) pre-activations -> (B,) P(win) - P(loss) for the side to move."""
    ys = []
    for s in range(0, len(Ps), 128):
        w = tail((out(torch.nn.functional.gelu(Ps[s:s + 128])) * sd + mu).reshape(-1, 256))
        ys.append(w[:, 0] - w[:, 2])
    return torch.cat(ys)

TACTIC = ("occupant", "material", "mate", "check", "threat")
NAMING = ("role.defends.", "role.needed_for.", "role.attacks.", "role.protected_by.", "role.needs.")
KEYS = ".PNBRQKpnbrqk"
def owner(s, l, threat=False):
    """Square whose player a bit belongs to (-1: always on). Naming bits belong to the square they name."""
    if l.startswith(NAMING):
        t = chess.parse_square(l.rsplit(".", 1)[1])
        if t != s: return t
    if l.startswith(TACTIC) and not (threat and l.startswith("threat")): return -1
    return s
key = lambda b, s: KEYS.index(b.piece_at(s).symbol()) if b.piece_at(s) else 0

# --- off state: mean bits per (occupant, square), as a (64, hidden) block; threat bits kept apart (claim 22)
mcache = ROOT / f"lc0-probe/data/claim_shap_means_n{a.mean}.pt"
if mcache.exists(): M, MT, cnt = torch.load(mcache)
else:
    lab = schema["labels"]
    c_of = torch.tensor([col.get(l, -1) for l in lab])
    own0 = [[owner(s, l) for s in range(64)] if l.startswith(NAMING) else None for l in lab]
    tac = torch.tensor([l.startswith(TACTIC) for l in lab]); thr = torch.tensor([l.startswith("threat") for l in lab])
    M = torch.zeros(13 * 64 * 64, E.shape[1]); MT = torch.zeros_like(M); cnt = torch.zeros(13 * 64)
    te = np.random.default_rng(0).random(100000) < 0.2                      # held out by the n100000 decoder
    with open(ROOT / f"lc0-probe/data/square_bits_n{a.n}_nodes200000.jsonl") as f:
        for i, l in enumerate(f):
            if i >= a.mean: break
            if te[i]: continue
            r = json.loads(l); b = chess.Board(r["fen"])
            ks = torch.tensor([key(b, s) * 64 + s for s in range(64)]); cnt[ks] += 1
            S, J = zip(*[(chess.parse_square(q), j) for q, v in r["squares"].items() for j in v])
            S, J = torch.tensor(S), torch.tensor(J)
            o = torch.where(tac[J], -1, S)
            for n, (s, j) in enumerate(zip(S.tolist(), J.tolist())):
                if own0[j] is not None: o[n] = own0[j][s]
            m = (c_of[J] >= 0) & ((o >= 0) | thr[J])
            oo = torch.where(o >= 0, o, S)
            idx = ks[oo[m]] * 64 + S[m]
            M.index_add_(0, idx[~thr[J][m]], E[c_of[J][m][~thr[J][m]]])
            MT.index_add_(0, idx[thr[J][m]], E[c_of[J][m][thr[J][m]]])
            if i % 2000 == 0: print(f"means: {i} positions", flush=True)
    M = (M.view(13 * 64, 64, -1) / cnt.clamp(min=1)[:, None, None]); MT = MT.view_as(M) / cnt.clamp(min=1)[:, None, None]
    torch.save((M, MT, cnt), mcache)

class Pos:
    """One stored position: its bits (square, label), their decoder rows, and selectors over them."""
    def __init__(self, p, labels):
        self.board = chess.Board(p["fen"])
        bits = [(chess.parse_square(sq), labels[j]) for sq, q in p["squares"].items() for j, _ in q["bits"]]
        self.bits = [b for b in bits if b[1] in col]
        self.sq = torch.tensor([s for s, _ in self.bits])
        self.rows = torch.stack([E[col[l]] for _, l in self.bits])
        self.full = base.index_add(0, self.sq, self.rows)
        self.y, self.y_dec = p["y"], p["y decoded"]
    def pos(self, sqs, occ=False, threat=False):
        """Player = the bits owned by sqs (occ: the occupant is never switched, so it's ignored); a frozenset of
        bit indices tagged with its squares, whose off state is the mean block of each square's occupant."""
        m = Player(i for i, (s, l) in enumerate(self.bits) if owner(s, l, threat) in sqs)
        m.sqs, m.threat = list(sqs), threat
        return m
    def shield(self, k):
        c = self.board.piece_at(k).color
        f, r, d = chess.square_file(k), chess.square_rank(k), 1 if c else -1
        cover = [s for s in self.board.pieces(chess.PAWN, c)
                 if abs(chess.square_file(s) - f) <= 1 and 1 <= (chess.square_rank(s) - r) * d <= 3]
        return self.pos([k] + cover)
    def off(self, ms):
        """Pre-activations with the players in off replaced by their mean blocks."""
        P = self.full.clone()
        for m in ms:
            ix = list(m)
            if ix: P.index_add_(0, self.sq[ix], -self.rows[ix])
            for s in m.sqs:
                k = key(self.board, s) * 64 + s
                P += M[k] + (MT[k] if m.threat else 0)
        return P

class Player(frozenset): pass

def two_player(pos, sets):
    """phi of each player against its average, everything else on, for the side to move (stored White)."""
    v = value(torch.stack([pos.full] + [pos.off([m]) for m in sets])).numpy()
    return [v[0] - x for x in v[1:]]

def joint(pos, sets):
    """Exact Shapley over the named players, the rest always on; returns phi per player, v(all), v(all averaged)."""
    k = len(sets)
    coal = list(itertools.product([0, 1], repeat=k))
    v = value(torch.stack([pos.off([sets[i] for i in range(k) if not c[i]]) for c in coal])).numpy()
    V = {c: x for c, x in zip(coal, v)}
    phi = np.zeros(k)
    for c in coal:
        for i in range(k):
            if c[i]: continue
            w = math.factorial(sum(c)) * math.factorial(k - sum(c) - 1) / math.factorial(k)
            phi[i] += w * (V[c[:i] + (1,) + c[i + 1:]] - V[c])
    return phi, V[(1,) * k], V[(0,) * k]

# --- corpus norms: phi (2-player) of every instance of each kind, signed for its owner (pooled +- for no owner)
WC = lambda c: 1 if c == chess.WHITE else -1
def instances(pos):
    b = pos.board
    out = []
    for t, kind in ((chess.KNIGHT, "N"), (chess.BISHOP, "B"), (chess.ROOK, "R"), (chess.QUEEN, "Q"), (chess.KING, "K")):
        for c in (chess.WHITE, chess.BLACK):
            out += [(kind, WC(c), pos.pos([s])) for s in b.pieces(t, c)]
    for c in (chess.WHITE, chess.BLACK):
        pw = list(b.pieces(chess.PAWN, c))
        files = [chess.square_file(s) for s in pw]
        iso = [s for s in pw if not any(abs(f - chess.square_file(s)) == 1 for f in files)]
        dbl = [s for s in pw if files.count(chess.square_file(s)) > 1]
        adv = [s for s in pw if (chess.square_rank(s) if c else 7 - chess.square_rank(s)) >= 3]
        passed = [s for s in pw if not any(x for x in b.pieces(chess.PAWN, not c)
                  if abs(chess.square_file(x) - chess.square_file(s)) <= 1 and (chess.square_rank(x) - chess.square_rank(s)) * WC(c) > 0)]
        out += [("isolated1", WC(c), pos.pos([s])) for s in iso]            # single isolated pawns
        if iso: out.append(("isolated", WC(c), pos.pos(iso)))                # and the side's set
        if dbl: out += [("doubled", WC(c), pos.pos(dbl)), ("doubled+occ", WC(c), pos.pos(dbl, occ=True))]
        if adv: out.append(("advanced", WC(c), pos.pos(adv)))
        for s in passed: out += [("passed", WC(c), pos.pos([s])), ("passed+threat", WC(c), pos.pos([s], threat=True))]
        for s in rng.permutation(pw)[:4]: out.append(("pawn", WC(c), pos.pos([int(s)])))
        bs = list(b.pieces(chess.BISHOP, c))
        if any(chess.BB_SQUARES[s] & chess.BB_LIGHT_SQUARES for s in bs) and any(chess.BB_SQUARES[s] & chess.BB_DARK_SQUARES for s in bs):
            out.append(("pair", WC(c), pos.pos(bs)))
        out.append(("pieces", WC(c), pos.pos([s for t in (chess.KNIGHT, chess.BISHOP, chess.QUEEN) for s in b.pieces(t, c)])))
        out += [("shield", WC(c), pos.shield(k)) for k in b.pieces(chess.KING, c)]
    out.append(("centre", 0, pos.pos([chess.D4, chess.E4, chess.D5, chess.E5])))
    empty = [s for s in chess.SQUARES if not b.piece_at(s)]
    out += [("empty", 0, pos.pos([int(s)])) for s in rng.permutation(empty)[:a.empty]]
    return [x for x in out if x[2]]

# --- claims
games = []
for ckey, players in SPEC.items():
    tb = TB[ckey[0] - 1]
    assert all(TB[k - 1]["fen"] == tb["fen"] for k in ckey)
    p, labels = by_fen[tb["fen"]]
    pos = Pos(p, labels)
    mir = tb["side_to_move"] == "black"                                  # stored board = real board mirrored
    st = lambda s: chess.square_mirror(s) if mir else s
    fav = WC(chess.WHITE if (tb["favours"] == "white") != mir else chess.BLACK)   # stored colour sign
    sets, owners, desc = [], [], []
    for name, sel, arg, kind, rule in players:
        sqs = [st(chess.parse_square(x)) for x in arg.split()] if sel in ("sq", "occ", "threat", "shield") else []
        if sel == "centre": m, own = pos.pos([chess.D4, chess.E4, chess.D5, chess.E5]), 0
        elif sel == "pieces":
            c = (arg == "w") != mir
            sqs = [s for t in (chess.KNIGHT, chess.BISHOP, chess.QUEEN) for s in pos.board.pieces(t, c)]
            m, own = pos.pos(sqs), WC(c)
        elif sel == "shield": m, own = pos.shield(sqs[0]), WC(pos.board.piece_at(sqs[0]).color)
        else:
            m = pos.pos(sqs, occ=sel == "occ", threat=sel == "threat")
            cs = {pos.board.piece_at(s).color for s in sqs if pos.board.piece_at(s)}
            own = WC(cs.pop()) if len(cs) == 1 else 0
        sym = lambda x: x.symbol().swapcase() if mir else x.symbol()      # real colours
        on = " ".join(f"{sym(pos.board.piece_at(s)) if pos.board.piece_at(s) else '.'}{chess.square_name(st(s))}" for s in sqs)
        sets.append(m); owners.append(own); desc.append(on)
    phi2 = two_player(pos, sets)
    phij, v_all, v_none = joint(pos, sets)
    assert abs(phij.sum() - (v_all - v_none)) < 1e-4
    games.append((ckey, players, tb, mir, fav, sets, owners, desc, phi2, phij))
    print(f"{'+'.join(map(str, ckey)):6} y {tb['y']:+.2f} dec {tb['y decoded']:+.2f}  " + "  ".join(
        f"{nm[0]} [{d}] {len(m)} bits phi2 {fav * f:+.3f}" for nm, d, m, f in zip(players, desc, sets, phi2)), flush=True)
if a.claims_only: sys.exit()

norms = {}
for i, p in enumerate(G["positions"]):
    pos = Pos(p, G["labels"])
    inst = instances(pos)
    for (kind, sgn, _), phi in zip(inst, two_player(pos, [m for _, _, m in inst])):
        norms.setdefault(kind, []).extend([sgn * phi] if sgn else [phi, -phi])
    if (i + 1) % 50 == 0: print(f"norms: {i + 1} positions", {k: len(v) for k, v in norms.items()}, flush=True)
norms = {k: np.array(v) for k, v in norms.items()}
pct = lambda x, ref: float(100 * ((ref < x).sum() + 0.5 * (ref == x).sum()) / len(ref))

# --- verdicts
res = []
for ckey, players, tb, mir, fav, sets, owners, desc, phi2, phij in games:
    rows, ok = [], True
    for (name, sel, arg, kind, rule), m, own, d, f2, fj in zip(players, sets, owners, desc, phi2, phij):
        ref = norms[kind]
        s = (own or fav) * f2
        q = pct(abs(s), np.abs(ref)) if rule == "zero" else pct(s, ref)
        passed = {"gen": (s > 0 and q >= 75) if (own or fav) == fav else (s < 0 and q <= 25),
                  "low": q <= 25, "neg": s < 0 and q <= 25, "notpos": s <= 0, "zero": q <= 25}[rule]
        ok &= passed
        rows.append({"player": name, "on": d, "kind": kind, "rule": rule, "bits": len(m), "owner": own * (-1 if mir else 1),
                     "s": round(float(s), 4), "pct": round(q, 1), "pass": bool(passed),
                     "phi joint (favoured)": round(float(fav * fj), 4)})
    if ckey in COMPARE:
        i, j = COMPARE[ckey]
        ok &= (owners[i] * phij[i]) > (owners[j] * phij[j])
    r = {"claims": list(ckey), "fen": tb["fen"], "orig_fen": tb["orig_fen"], "claim": [TB[k - 1]["claim"] for k in ckey],
         "quote": tb["quote"], "source": tb["source"], "topic": tb["topic"], "favours": tb["favours"],
         "y": tb["y"], "y decoded": tb["y decoded"],
         "players": rows, "verdict": "supported" if ok else "not supported",
         **({"stand-in": "piece activity, not development"} if ckey[0] in STANDIN else {})}
    res.append(r)
    print(f"{'+'.join(map(str, ckey)):6} {r['verdict']:14} y {r['y']:+.2f} dec {r['y decoded']:+.2f}  " +
          "  ".join(f"{x['player']} [{x['on']}] s {x['s']:+.3f} p{x['pct']:.0f}{'*' if x['pass'] else ''}" for x in rows), flush=True)

n_ok = sum(r["verdict"] == "supported" for r in res)
print(f"{n_ok} of {len(res)} games supported", flush=True)
outd = {"tap": "/encoder0/ln2", "criterion": (ROOT / "lc0-probe/claim_players.md").read_text(),
        "norms": {k: {"n": len(v), "q25": float(np.percentile(v, 25)), "median": float(np.median(v)),
                      "q75": float(np.percentile(v, 75))} for k, v in norms.items()}, "games": res}
json.dump(outd, open(ROOT / "lc0-probe/results_claim_shap.json", "w"), indent=1)
(ROOT / "lc0-probe/claim_shap.js").write_text("window.CLAIM_SHAP = " + json.dumps(outd, separators=(",", ":")) + ";\n")
