"""Annotators' piece claims against real Leela T1-256x10, by counterfactual boards, without removing pieces.

v = P(win) - P(loss), always for the piece's owner. For one piece:
  gap     = v(on its square) - v(best reachable square)  (<= 0; 0 when it already stands on its best square)
  rank    = place of its square among its reachable squares by v (1 = best)
  ceiling = v(best reachable square) - v(median safe square), the median taken over the squares other than its own
  swap    = v(knight <-> same-colour bishop on the same square) - v(on its square)
reachable: squares the piece reaches in <= 3 quiet moves with the rest of the board frozen, every landing square
safe. Safe: empty, board legal, no check given or uncovered (checkers unchanged), the piece not hanging (attacked,
undefended) and not attacked by a lower-value piece. Norms: the same numbers for every non-pawn, non-king piece of
the 300 corpus positions, pooled by type. (Removing a piece sends v to about -1 in 98% of the corpus, so no
number here uses removal.)

Placement Shapley: players = the non-pawn, non-king pieces; absent = the piece on a square of its background set
(its reachable squares minus its own). Per antithetic pair each piece draws a preference order over its set; on
each coalition board an absent piece takes the first square that is empty and safe there, else the first empty
one giving no check (an unsafe draw). Positions where more than --unsafe of the absent draws are unsafe are
flagged unreliable and left out of the disagreement list. Pieces with no safe alternative stay on their square
(phi 0, "fixed"). Compared with the single-piece marginal v(all) - mean over the set of v(piece moved, rest on).

  python lc0-probe/piece_placement.py --stage norms
  python lc0-probe/piece_placement.py --stage claims --perms 128
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import chess, numpy as np, onnxruntime as ort
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["norms", "claims"])
ap.add_argument("--depth", type=int, default=3, help="moves to reach a square")
ap.add_argument("--perms", type=int, default=128, help="Shapley permutations per position (even: antithetic pairs)")
ap.add_argument("--unsafe", type=float, default=0.10, help="Shapley unreliable above this share of unsafe absent draws")
ap.add_argument("--disagree", type=float, default=0.10, help="|phi - single-piece marginal| flagged as interaction")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
NORMS = ROOT / "lc0-probe/data/piece_placement_norms.json"
OUT = ROOT / "lc0-probe/results_piece_placement.json"
JS = ROOT / "lc0-probe/piece_placement_textbook.js"
CORPUS = ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_games.js"
TEXT = ROOT / "lc0-probe/square_shap_textbook.js"
EXAMPLES = ROOT / "features/examples.json"
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
TYPES = {chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q"}
jsload = lambda p: json.loads(p.read_text().split("=", 1)[1].rstrip().rstrip(";"))

so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
SESS = ort.InferenceSession(str(ROOT / "vendor/lc0-nets/t1-256x10.onnx"), so, providers=["CPUExecutionProvider"])
CACHE = {}
def leela(fens):
    """P(win) - P(loss) for the side to move, cached by FEN."""
    new = list(dict.fromkeys(f for f in fens if f not in CACHE))
    for s in range(0, len(new), 256):
        w = SESS.run(["/output/wdl"], {"/input/planes": encode_fens(new[s:s + 256])})[0]
        CACHE.update(zip(new[s:s + 256], (w[:, 0] - w[:, 2]).tolist()))
    return np.array([CACHE[f] for f in fens])

def clean(c):
    c.castling_rights = c.clean_castling_rights(); c.ep_square = None
    return c

def legal(c, b):
    return c.is_valid() and c.checkers() == b.checkers()

def safe(c, s, b=None):
    """Piece on s of board c is safe; b: the reference board for legality (None: skip that test)."""
    p = c.piece_at(s)
    if (b is not None and not legal(c, b)) or c.attacks(s) & c.pieces(chess.KING, not p.color): return False
    att = c.attackers(not p.color, s)
    if att and not c.attackers(p.color, s): return False
    return not any(VAL[c.piece_type_at(x)] < VAL[p.piece_type] for x in att)

def moved(b, s0, t, p=None):
    """b with the piece on s0 (or p) moved to t; t None removes it."""
    c = b.copy(stack=False)
    if s0 is not None: p = c.remove_piece_at(s0)
    if t is not None: c.set_piece_at(t, p)
    return clean(c)

def reach(b, s0):
    """{square: moves} reachable in <= depth quiet moves through safe landing squares (s0 itself: 0)."""
    p, base = b.piece_at(s0), b.copy(stack=False)
    base.remove_piece_at(s0)
    seen, front = {s0: 0}, [s0]
    for d in range(1, a.depth + 1):
        nxt = []
        for s in front:
            c = base.copy(stack=False); c.set_piece_at(s, p)
            for t in c.attacks(s):
                if t in seen or base.piece_at(t): continue
                if safe(moved(base, None, t, p), t, b): seen[t] = d; nxt.append(t)
        front = nxt
    return seen

def piece(b, s0):
    """Counterfactual numbers for the piece on s0, v for its owner."""
    p = b.piece_at(s0); own = 1 if p.color == b.turn else -1
    R = reach(b, s0)
    sq = list(R)
    boards = [moved(b, s0, t) if t != s0 else b for t in sq]
    swap = None
    if p.piece_type in (chess.KNIGHT, chess.BISHOP):
        sw = b.copy(stack=False); sw.set_piece_at(s0, chess.Piece(chess.BISHOP + chess.KNIGHT - p.piece_type, p.color))
        if legal(clean(sw), b): swap = sw; boards.append(sw)
    v = own * leela([c.fen() for c in boards])
    vs = dict(zip(sq, v[:len(sq)])); von = vs[s0]
    best = max(sq, key=lambda t: vs[t]); alt = [vs[t] for t in sq if t != s0]
    order = sorted((t for t in sq if t != s0), key=lambda t: vs[t]); med = order[(len(order) - 1) // 2] if order else None
    r = {"square": chess.square_name(s0), "piece": p.symbol(), "type": TYPES[p.piece_type],
         "colour": "white" if p.color else "black", "v": float(von), "gap": float(von - vs[best]),
         "ceiling": None if med is None else float(vs[best] - vs[med]),
         "best": chess.square_name(best), "best moves": R[best], "reachable": len(sq) - 1,
         "median square": None if med is None else chess.square_name(med),
         "v median": None if med is None else float(vs[med]),
         "rank": int(1 + sum(x > von for x in alt)),
         "marginal": float(von - np.mean(alt)) if alt else 0.0,
         "swap": None if swap is None else float(v[-1] - von),
         "squares": {chess.square_name(t): [R[t], round(float(vs[t]), 4)] for t in sq}}
    return r, [t for t in sq if t != s0]

def players(b):
    return [s for s in chess.SQUARES if b.piece_at(s) and b.piece_type_at(s) not in (chess.PAWN, chess.KING)]

def shapley(b, bg, rng):
    """Permutation Shapley over the pieces in bg (square -> background squares), antithetic pairs."""
    P = list(bg); n = len(P); phi = np.zeros(n); pairs = []; fall = draws = bad = boards = 0; ends = []
    for k in range(a.perms // 2):
        pref = {s: list(rng.permutation(bg[s])) for s in P}
        def board(S):
            nonlocal fall, draws, bad, boards
            c = b.copy(stack=False); absent = [s for s in P if s not in S]
            pcs = {s: c.remove_piece_at(s) for s in absent}; at = {}
            def place(s):                                      # first square empty and safe on this board,
                for strict in (True, False):                   # else the first empty one giving no check
                    for t in pref[s]:
                        if c.piece_at(t): continue
                        c.set_piece_at(t, pcs[s])
                        if safe(c, t) if strict else not c.attacks(t) & c.pieces(chess.KING, not pcs[s].color):
                            at[s] = t; return strict
                        c.remove_piece_at(t)
                at[s] = None; return False
            for s in absent: place(s)
            for s in absent:                                   # re-check once all are placed
                if at[s] is None or not safe(c, at[s]):
                    if at[s] is not None: c.remove_piece_at(at[s])
                    fall += not place(s)
            boards += 1; draws += len(absent)
            bad += not legal(clean(c), b)
            return c.fen()
        perm = rng.permutation(n); one = np.zeros(n)
        for order in (perm, perm[::-1]):
            S, fens = set(), [board(set())]
            for i in order: S.add(P[i]); fens.append(board(S))
            v = leela(fens); d = np.diff(v); one[order] += d; ends.append((v[0], v[-1]))
        phi += one; pairs.append(one / 2)
    own = np.array([1 if b.piece_at(s).color == b.turn else -1 for s in P])
    m = a.perms // 2 * 2
    se = np.std(pairs, 0) / np.sqrt(len(pairs)) if len(pairs) > 1 else np.zeros(n)
    return {chess.square_name(s): {"phi": float(own[i] * phi[i] / m), "se": float(se[i])} for i, s in enumerate(P)}, \
           {"unsafe draws": fall / max(draws, 1), "illegal boards": bad / max(boards, 1),
            "efficiency residual": float(phi.sum() / m - np.mean([e[1] - e[0] for e in ends]))}

def pct(x, ref):
    if x is None or not len(ref): return None
    ref = np.asarray(ref); return round(float(100 * ((ref < x).sum() + 0.5 * (ref == x).sum()) / len(ref)), 1)

# Claims: (source, row, piece square, meaning, sign). Sign +1: the piece is called good, -1: bad, 0: ambiguous.
# T = square_shap_textbook.js positions[row]; E = features/examples.json[row]. Quotes are copied from the files.
CLAIMS = [
    ("T", 8, "c8", "general", 1), ("T", 8, "f8", "general", 1),            # "his two Bishops"
    ("T", 11, "e4", "placement", 1),                                         # "strongly-posted Knight"
    ("T", 9, "b3", "general", 1), ("T", 9, "e3", "general", 1),             # "possession of the two Bishops"
    ("T", 9, "d7", "ambiguous", -1), ("T", 9, "e5", "ambiguous", -1),       # "his Knights lack bases in the centre"
    ("T", 10, "a6", "general", 1), ("T", 10, "b6", "general", 1),           # "two Bishops more powerful than B+N"
    ("T", 12, "c3", "placement", 1),                                         # "support his knight on c3"
    ("T", 13, "f4", "both", 1),                                              # "strength of the unassailable knight"
    ("T", 13, "c8", "general", -1), ("T", 13, "g7", "general", -1),         # "two bishops have little to say"
    ("T", 29, "d8", "placement", 1),                                         # "dominating position ... open file"
    ("T", 30, "a7", "placement", 1),                                         # "commands the 7th Row"
    ("T", 31, "h7", "placement", 1),                                         # "his Rook cuts the Black King off"
    ("T", 32, "a2", "placement", -1),                                        # "the Rook is quite out of play"
    ("T", 40, "b7", "general", -1),                                          # "inactive and cannot come into the game"
    ("T", 41, "d5", "ambiguous", -1),                                        # "cannot be maintained in this position"
    ("E", 15, "d4", "ambiguous", -1), ("E", 15, "g2", "ambiguous", -1),     # "advantage of the two Bishops is illusory"
    ("E", 32, "h8", "ambiguous", -1), ("E", 32, "a8", "ambiguous", -1),     # "a whole Rook ahead ... out of play"
    ("E", 33, "g7", "ambiguous", 1),                                         # "becomes a very powerful attacking piece"
    ("E", 34, "a2", "general", -1),                                          # "hemmed in permanently ... a piece up"
    ("E", 36, "d5", "placement", 1),                                         # "No outpost, yet strong"
    ("E", 39, "a7", "ambiguous", 1),                                         # "would no longer command the long diagonal"
    ("E", 41, "e3", "placement", -1),                                        # "much better posted on the open diagonal"
    ("E", 42, "e7", "ambiguous", 1),                                         # "defensive Bishop becomes an attacking piece"
    ("E", 43, "c8", "general", -1),                                          # "cannot get his c8 bishop into play"
    ("E", 44, "a5", "placement", 1), ("E", 44, "d3", "placement", -1),      # "especially favourable" / "unfavourable position"
    ("E", 45, "g2", "general", -1),                                          # "shuts in the White g2 bishop altogether"
    ("E", 47, "e7", "ambiguous", 1),                                         # "prevent the Black Rook from controlling the open file"
    ("E", 50, "f4", "ambiguous", 1),                                         # "dare not allow the Rook to penetrate into the seventh"
    ("E", 56, "e8", "placement", -1),                                        # "standing there with 'nothing doing'"
    ("E", 58, "d1", "ambiguous", 1), ("E", 58, "e1", "ambiguous", 1),       # rooks as over-protectors of d4/e5
    ("E", 59, "d4", "placement", 1),                                         # "his Queen better placed"
    ("E", 61, "a2", "placement", -1),                                        # "will momentarily be out of play"
    ("E", 62, "h7", "placement", 1),                                         # "she will be excellently placed"
    ("E", 68, "a1", "placement", -1), ("E", 68, "d3", "placement", -1),     # "blocked by the Bishop" / "unnaturally placed"
]
PAIRS = {("T", 8), ("T", 9), ("T", 10), ("T", 13)}                          # bishop-pair claims: verdict on the pair
DISAGREE = (f"Placement Shapley disagrees with the single-piece marginal when |phi - single| > {a.disagree}, or the signs "
            f"differ with both |values| > 0.05.")
CRITERION = ("Fixed before the claim runs. Percentiles: mid-rank among the corpus pieces of the same type (both colours, "
             "v for the owner). Rank fraction f = (rank - 1) / reachable (0 = its square is the best, 1 = the worst); "
             "a piece with no safe alternative gets no placement verdict. placement+: gap pct >= 75 and f <= 0.25. "
             "placement-: gap pct <= 25 and f >= 0.5. general score: for a knight or bishop the mean of ceiling pct and "
             "(100 - swap pct) (a swap that loses value is a point for the piece; ceiling pct alone if the swap board is "
             "illegal), for a rook or queen ceiling pct. general+: score >= 75; general-: score <= 25. both: the "
             "placement and the general rule in the claim's direction. Bishop pairs: supported iff both bishops pass. "
             "Ambiguous: no verdict.")

def gscore(r):
    if r["pct ceiling"] is None: return None
    return r["pct ceiling"] if r["pct swap"] is None else (r["pct ceiling"] + 100 - r["pct swap"]) / 2

def verdict(r, meaning, sign):
    if meaning == "ambiguous": return None
    f = r["rank fraction"]
    pl = None if f is None else (r["pct gap"] >= 75 and f <= 0.25) if sign > 0 else (r["pct gap"] <= 25 and f >= 0.5)
    g = r["general score"]
    ge = None if g is None else g >= 75 if sign > 0 else g <= 25
    return {"placement": pl, "general": ge, "both": None if pl is None or ge is None else pl and ge}[meaning]

rng = np.random.default_rng(a.seed)
if a.stage == "norms":
    fens = [p["fen"] for p in jsload(CORPUS)["positions"]]
    rows, t0 = [], time.time()
    for i, f in enumerate(fens):
        b = chess.Board(f)
        for s in players(b):
            r, _ = piece(b, s); r.pop("squares"); r["pos"] = i; rows.append(r)
        if i % 10 == 0: print(f"{i}/{len(fens)} {len(rows)} pieces {time.time() - t0:.0f}s", flush=True)
    NORMS.write_text(json.dumps({"positions": len(fens), "depth": a.depth, "pieces": rows}))
    sys.exit()

N = json.loads(NORMS.read_text())["pieces"]
ref = {t: {k: [x[k] for x in N if x["type"] == t and x[k] is not None] for k in ("gap", "ceiling", "swap")}
       for t in TYPES.values()}
T, E = jsload(TEXT)["positions"], json.loads(EXAMPLES.read_text())
def meta(src, i):
    x = T[i] if src == "T" else E[i]
    return {"fen": x["orig_fen"] if src == "T" else x["fen"], "source": x["source"], "quote": x["quote"],
            "claim": x.get("claim"), "game": x.get("game"), "url": x.get("url"), "topic": x.get("topic", x.get("family"))}

out, shap, t0 = [], {}, time.time()
for src, i, sq, meaning, sign in CLAIMS:
    m = meta(src, i); b = chess.Board(m["fen"]); s0 = chess.parse_square(sq)
    assert b.piece_at(s0) and b.piece_type_at(s0) in TYPES, (src, i, sq)
    if m["fen"] not in shap:
        bg, single = {}, {}
        for s in players(b):
            r, alt = piece(b, s); single[chess.square_name(s)] = r["marginal"]
            if alt: bg[s] = alt                                # no safe alternative: always on its square
        phi, fall = shapley(b, bg, rng)
        for k in single: phi.setdefault(k, {"phi": 0.0, "se": 0.0, "fixed": True})
        for k, x in phi.items():
            x.update(single=single[k], piece=b.piece_at(chess.parse_square(k)).symbol())
            x["disagree"] = not x.get("fixed") and (abs(x["phi"] - x["single"]) > a.disagree or
                                                    (x["phi"] * x["single"] < 0 and min(abs(x["phi"]), abs(x["single"])) > 0.05))
        fall["unreliable"] = fall["unsafe draws"] > a.unsafe
        if fall["unreliable"]:
            for x in phi.values(): x["disagree"] = None
        shap[m["fen"]] = {"phi": phi, "fallback": fall}
        print(f"{src}{i} {len(bg)} players {fall} {time.time() - t0:.0f}s", flush=True)
    r, _ = piece(b, s0); t = r["type"]
    r.update({"pct " + k: pct(r[k], ref[t][k]) for k in ("gap", "ceiling", "swap")})
    r["rank fraction"] = (r["rank"] - 1) / r["reachable"] if r["reachable"] else None
    r["general score"] = gscore(r)
    r.update(m, src=f"{src}{i}", meaning=meaning, sign=sign, verdict=verdict(r, meaning, sign),
             shapley=shap[m["fen"]]["phi"], **{"shapley fallback": shap[m["fen"]]["fallback"]})
    out.append(r)
for key in PAIRS:
    rs = [r for r in out if r["src"] == f"{key[0]}{key[1]}" and r["type"] == "B" and r["meaning"] != "ambiguous"]
    v = None if any(r["verdict"] is None for r in rs) else all(r["verdict"] for r in rs)
    for r in rs: r["pair verdict"] = v
norms = {t: {k: {"n": len(x), "q25": float(np.quantile(x, .25)), "median": float(np.median(x)),
                 "q75": float(np.quantile(x, .75))} for k, x in d.items() if x} for t, d in ref.items()}
res = {"net": "t1-256x10", "depth": a.depth, "perms": a.perms, "unsafe": a.unsafe, "disagree": a.disagree,
       "criterion": CRITERION, "disagreement": DISAGREE, "norms": norms, "claims": out}
OUT.write_text(json.dumps(res, indent=1))
JS.write_text("window.PIECES = " + json.dumps(res) + ";\n")
print(f"wrote {OUT} and {JS}")
