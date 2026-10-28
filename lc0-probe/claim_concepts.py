"""Annotator claims as concept attribution (claim_concepts.md): integrated gradients through the /encoder0/ln2 decoder,
summed over the claim's concept bits, signed for the side named; with / without the credited pieces, or descriptive.

Builds the real boards and their counterfactuals, runs square_attribution.py --fens on them (cached), scores.
"Approx 0" = |x| at or below the 25th percentile of |the same concept| over the 300 corpus game positions.

  python lc0-probe/claim_concepts.py
"""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import chess, numpy as np

L = lambda f: json.loads(Path(f).read_text().split("=", 1)[1].rstrip().rstrip(";"))
TB = L(ROOT / "lc0-probe/square_shap_textbook.js")["positions"]
W, B = chess.WHITE, chess.BLACK
sq = chess.parse_square
CENTRE = [sq(x) for x in ("d4", "e4", "d5", "e5")]

class View:
    """One board's IG bits, read in real-board squares and colours (stored boards are White to move)."""
    def __init__(self, r, labels, mirrored):
        self.mir, self.b = mirrored, chess.Board(r["fen"])
        self.bits = [(sq(s), labels[j], v) for s, q in r["squares"].items() for j, v in q["bits"]]
        self.total = sum(abs(v) for _, _, v in self.bits)
    def s(self, x): return chess.square_mirror(x) if self.mir else x                   # real square -> stored
    def c(self, x): return (not x) if self.mir else x                                 # real colour -> stored
    def sign(self, X): return 1 if self.c(X) == W else -1
    def name(self, x): return chess.square_name(self.s(x))
    rec = None
    def sum(self, X, pred):
        hit = [(s, l, v) for s, l, v in self.bits if pred(s, l)]
        if self.rec is not None: self.rec += [(s, l, self.sign(X) * v) for s, l, v in hit]
        return self.sign(X) * sum(v for _, _, v in hit)
    def real(self, s, l):
        """A recorded bit as [real square, label with real square names, value]."""
        f = (lambda x: chess.square_name(chess.square_mirror(sq(x)))) if self.mir else (lambda x: x)
        return [f(chess.square_name(s)), ".".join(f(t) if len(t) == 2 and t[0] in "abcdefgh" and t[1] in "12345678" else t
                                                 for t in l.split("."))]
    def work(self, fn):
        """fn's value and the bits it summed, largest first."""
        self.rec = []; x = fn(self); bits, self.rec = self.rec, None
        return x, sorted(([*self.real(s, l), round(v, 5)] for s, l, v in bits), key=lambda b: -abs(b[2]))
    def col(self, s): p = self.b.piece_at(s); return None if p is None else p.color
    def own(self, X): return [s for s in chess.SQUARES if self.col(s) == self.c(X)]   # stored squares
    def tgt(self, l): return sq(l.rsplit(".", 1)[1])
    # --- concepts (real squares in, stored inside)
    def control_into(self, X, targets):
        T, cn = {self.s(t) for t in targets}, "white" if self.c(X) == W else "black"
        mine = set(self.own(X))
        return self.sum(X, lambda s, l: (s in mine and l.startswith(("control.reach.", "control.geometry.")) and self.tgt(l) in T)
                        or (s in T and l.startswith(f"square.control.{cn}")))
    def space(self, X, files="abcdefgh"):
        half = range(4, 8) if X == W else range(0, 4)
        return self.control_into(X, [chess.square(f, r) for f in range(8) if "abcdefgh"[f] in files for r in half])
    def centre(self, X): return self.control_into(X, CENTRE)
    def hold(self, X, t):
        cn = "white" if self.c(X) == W else "black"
        return self.control_into(X, [t]) + self.sum(X, lambda s, l: s == self.s(t) and l.startswith(f"strong.{cn}"))
    def outpost(self, t):
        X = self.owner(t)
        return self.sum(X, lambda s, l: s == self.s(t) and (l.startswith(("strong.", "stability.", "role.protected_by."))))
    def owner(self, t): c = self.col(self.s(t)); return c if c is None or not self.mir else (not c)
    def pieces(self, X, files=range(8)):
        """Real squares of X's knights, bishops, rooks and queens on these files."""
        return [t for t in chess.SQUARES if chess.square_file(t) in files and self.owner(t) == X
                and self.b.piece_at(self.s(t)).piece_type not in (chess.PAWN, chess.KING)]
    def king(self, X): return chess.square_mirror(self.b.king(self.c(X))) if self.mir else self.b.king(self.c(X))
    def king_safety(self, X):
        k = self.b.king(self.c(X)); zone = {k} | set(chess.SquareSet(chess.BB_KING_ATTACKS[k]))
        kn, cn, enemy = chess.square_name(k), "white" if self.c(X) == W else "black", set(self.own(not X))
        return self.sum(X, lambda s, l: l.startswith(f"pawn_shield.{kn}.") or l == f"role.attacks.king.{kn}"
                        or (s in zone and l.startswith(("king.castle", f"pawn_cover.{cn}", "square.control")))
                        or (s in enemy and l.startswith(("control.reach.", "control.geometry.")) and self.tgt(l) in zone))
    def king_mobility(self, X):
        k = self.b.king(self.c(X))
        return self.sum(X, lambda s, l: s == k and l.startswith(("mobility.", "control.", "blocked_by.own.")))
    def pair(self, X):
        bs = set(self.b.pieces(chess.BISHOP, self.c(X)))
        return self.sum(X, lambda s, l: s in bs and l.startswith(("bishop_pair.", "unlike_bishops.")))
    def activity(self, t, mobility_only=False):
        u, X = self.s(t), self.owner(t)
        pre = ("mobility.",) if mobility_only else ("control.", "mobility.", "xray.", "role.")
        return self.sum(X, lambda s, l: s == u and l.startswith(pre))
    def passer(self, t):
        u, X = self.s(t), self.owner(t)
        return self.sum(X, lambda s, l: (s == u and l.startswith(("pawn.passed", "pawn.rank4_6", "pawn.rank7", "role.protected_by.")))
                        or l == f"rook.behind_passer.{chess.square_name(u)}")
    def threat(self, t): return self.sum(self.owner(t), lambda s, l: s == self.s(t) and l.startswith("threat."))
    def weakness(self, ts, defenders=False):
        U = {self.s(t) for t in ts}; X = self.owner(ts[0]); cn = "white" if self.c(X) == W else "black"
        names = {chess.square_name(u) for u in U}
        return self.sum(X, lambda s, l: (s in U and l.startswith(("pawn.isolated", "pawn.doubled_with", "pawn.hanging", f"weak.{cn}", "stability.")))
                        or (l.startswith("role.attacks.pawn.") and l.rsplit(".", 1)[1] in names)
                        or (defenders and l.startswith("role.defends.pawn.") and l.rsplit(".", 1)[1] in names))
    def material_up(self, X): return self.sum(X, lambda s, l: l.startswith("material.up") and self.col(s) == self.c(X))
    def rook_seventh(self, t): return self.sum(self.owner(t), lambda s, l: s == self.s(t) and l == "rook.seventh.on")
    def stability(self, t): return self.sum(self.owner(t), lambda s, l: s == self.s(t) and l.startswith("stability."))

S = lambda *xs: [sq(x) for x in xs]
def off(*xs): return ("off " + " ".join(xs), lambda b: [b.remove_piece_at(sq(x)) for x in xs])
def back(desc, moves):
    """Take a move back: moves = [(from, to)] applied as piece relocations; castling rights added back by hand."""
    def f(b):
        for a, c in moves: b.set_piece_at(sq(c), b.remove_piece_at(sq(a)))
        b.turn = not b.turn
    return (desc, f)

def edited(tb, key, f):
    b = chess.Board(tb["orig_fen"]); f(b)
    if key == (16,): b.castling_rights |= chess.BB_A1 | chess.BB_E1          # before O-O-O: White could castle long
    b.castling_rights = b.clean_castling_rights(); b.ep_square = None
    assert b.is_valid(), (key, b.fen(), b.status())
    return b.fen()

# with / without: claim -> (concept name, concept fn, [edits], expected "drop"/"rise")
WW = {
    (1,): ("Space(W)", lambda v: v.space(W), [off("e4"), off("f4"), off("d4")], "drop"),
    (2,): ("Space(B)", lambda v: v.space(B), [off("e4"), off("f5"), off("e5")], "drop"),
    (3,): ("Space(W)", lambda v: v.space(W), [off("d5"), off("f4"), off("d4")], "drop"),
    (4,): ("Space(B, a-d)", lambda v: v.space(B, "abcd"), [off("b4"), off("c3")], "drop"),
    (5,): ("B mobility on a-d", lambda v: sum(v.activity(t, True) for t in v.pieces(B, range(4))),
           [off("a3"), off("b4"), off("c4"), off("d5")], "rise"),
    (13,): ("Outpost(c3)", lambda v: v.outpost(sq("c3")), [off("b4")], "drop"),
    (14, 40): ("Pair(B)", lambda v: v.pair(B), [off("f4")], "rise"),
    (15,): ("KingSafety(W)", lambda v: v.king_safety(W), [off("d7")], "rise"),
    (16,): ("KingSafety(W)", lambda v: v.king_safety(W), [back("take back O-O-O", [("c1", "e1"), ("d1", "a1")])], "drop"),
    (19,): ("KingSafety(B)", lambda v: v.king_safety(B), [back("take back ...h6", [("h6", "h7")])], "rise"),
    (22,): ("Threat(d6) + Passer(d6)", lambda v: v.threat(sq("d6")) + v.passer(sq("d6")), [off("d3")], "drop"),
    (23,): ("Passer(g5)", lambda v: v.passer(sq("g5")), [off("f4")], "drop"),
    (30,): ("KingMobility(W)", lambda v: v.king_mobility(W), [off("f2", "g2", "h2")], "rise"),
    (32,): ("KingMobility(B)", lambda v: v.king_mobility(B), [off("h7")], "rise"),
}
# "drop" = the concept is lower without the credited piece; for take-backs the edited board is "before" the move:
# 16: safety lower before O-O-O (drop), 19: safety higher before ...h6 (rise)

# descriptive: claim -> [(concept name, fn, test)], test in ">0", "<0", "~0", "<=0" or ("gt", other fn) / kind for ~0
DESC = {
    (6,): [("Centre(W) - Centre(B)", lambda v: v.centre(W) - v.centre(B), ">0"), ("Centre(W)", lambda v: v.centre(W), ">0")],
    (7,): [("Centre(B) - Centre(W)", lambda v: v.centre(B) - v.centre(W), ">0"), ("Centre(B)", lambda v: v.centre(B), ">0")],
    (8, 43): [("Hold(B, d5)", lambda v: v.hold(B, sq("d5")), ">0")],
    (9, 12): [("Pair(B)", lambda v: v.pair(B), ">0"), ("Outpost(e4)", lambda v: v.outpost(sq("e4")), ">0")],
    (10,): [("Pair(W)", lambda v: v.pair(W), ">0"), ("Outpost(d7)+Outpost(e5)", lambda v: v.outpost(sq("d7")) + v.outpost(sq("e5")), "<=0")],
    (11,): [("Pair(B)+Act(a6,b6) - Act(b4,c6)", lambda v: v.pair(B) + v.activity(sq("a6")) + v.activity(sq("b6"))
             - v.activity(sq("b4")) - v.activity(sq("c6")), ">0")],
    (17,): [("KingSafety(B)", lambda v: v.king_safety(B), "<0")],
    (18,): [("KingSafety(W)", lambda v: v.king_safety(W), "<0"), ("KingSafety(B)", lambda v: v.king_safety(B), ">0")],
    (20,): [("KingSafety(W)", lambda v: v.king_safety(W), "<0")],
    (21,): [("Passer(e4)", lambda v: v.passer(sq("e4")), ">0"), ("Activity(Ke3)", lambda v: v.activity(sq("e3")), ">0")],
    (24,): [("Weakness(a5 d6 f5)", lambda v: v.weakness(S("a5", "d6", "f5")), "<0")],
    (25,): [("Weakness(a5 c7 e5 e6)+defenders", lambda v: v.weakness(S("a5", "c7", "e5", "e6"), True), "<0")],
    (26,): [("Weakness(c6 c7)", lambda v: v.weakness(S("c6", "c7")), "<0")],
    (27,): [("material.up(B) + Weakness(c4 c6)", lambda v: v.material_up(B) + v.weakness(S("c4", "c6")), ("~0", "up+doubled"))],
    (28,): [("Weakness(d5)", lambda v: v.weakness(S("d5")), "<0")],
    (29,): [("Weakness(h7)", lambda v: v.weakness(S("h7")), "<0")],
    (31,): [("rook.seventh + Activity(Ra7)", lambda v: v.rook_seventh(sq("a7")) + v.activity(sq("a7")), ">0"),
            ("Weakness(e3 g2)", lambda v: v.weakness(S("e3", "g2")), "<0")],
    (33,): [("Activity(ra2)", lambda v: v.activity(sq("a2")), ("~0", "R"))],
    (39,): [("Space(B, a-c) + Weakness(b4 c5 a6)", lambda v: v.space(B, "abc") + v.weakness(S("b4", "c5", "a6")), "<=0")],
    (41,): [("Activity(bb7)", lambda v: v.activity(sq("b7")), ("~0", "B")), ("Pair(B)", lambda v: v.pair(B), "<=0")],
    (42,): [("stability(d5), for White", lambda v: v.stability(sq("d5")), "<0")],
}

# --- boards: real FENs, edited in the real frame
fens, jobs = [], []
for key in list(WW) + list(DESC):
    tb = TB[key[0] - 1]
    fens.append(tb["orig_fen"])
    fens += [edited(tb, key, f) for _, f in (WW[key][2] if key in WW else [])]
fens = list(dict.fromkeys(fens))
FF = ROOT / "lc0-probe/data/claim_concepts.fens"
FF.write_text("\n".join(fens) + "\n")
OUT = ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_claim_concepts.js"
if not OUT.exists() or json.loads(OUT.read_text().split("=", 1)[1].rstrip().rstrip(";"))["source"] != str(FF) \
        or len(L(OUT)["positions"]) != len(fens):
    subprocess.run([sys.executable, ROOT / "lc0-probe/square_attribution.py", "--n", "100000", "--tap", "/encoder0/ln2",
                    "--ig", "256", "--fens", FF, "--positions", str(len(fens)), "--threads", "4"], check=True)
A = L(OUT)
assert len(A["positions"]) == len(fens)
views = {}
for real, r in zip(fens, A["positions"]):
    views[real] = View(r, A["labels"], chess.Board(real).turn == B)

# --- corpus norms for "~0": |concept| over the 300 game positions (stored = real, White to move)
G = L(ROOT / "docs/square_attribution_n100000_encoder0_ln2_ig_games.js")
norm = {"R": [], "B": [], "up+doubled": []}
for r in G["positions"]:
    v = View(r, G["labels"], False)
    for t in chess.SQUARES:
        p = v.b.piece_at(t)
        if p and p.piece_type == chess.ROOK: norm["R"].append(abs(v.activity(t)))
        if p and p.piece_type == chess.BISHOP: norm["B"].append(abs(v.activity(t)))
    for X in (W, B):
        pw = list(v.b.pieces(chess.PAWN, X)); fl = [chess.square_file(s) for s in pw]
        dbl = [s for s in pw if fl.count(chess.square_file(s)) > 1]
        if dbl and v.material_up(X) != 0: norm["up+doubled"].append(abs(v.material_up(X) + v.weakness(dbl)))
norm = {k: np.array(x) for k, x in norm.items()}
pct = lambda x, ref: float(100 * (ref < x).mean())

res = []
for key in list(WW) + list(DESC):
    tb = TB[key[0] - 1]; v0 = views[tb["orig_fen"]]
    r = {"claims": list(key), "claim": [TB[k - 1]["claim"] for k in key], "orig_fen": tb["orig_fen"],
         "y": tb["y"], "y decoded": tb["y decoded"], "tests": [],
         **{k: tb[k] for k in ("quote", "source", "url", "topic", "favours", "side_to_move") if k in tb}}
    ok = True
    if key in WW:
        name, fn, edits, exp = WW[key]
        x0, sh0 = fn(v0), fn(v0) / v0.total
        for desc, f in edits:
            e = edited(tb, key, f)
            v1 = views[e]; x1, sh1 = fn(v1), fn(v1) / v1.total
            d, dsh = x1 - x0, sh1 - sh0
            p = (d < 0 and dsh < 0) if exp == "drop" else (d > 0 and dsh > 0)
            (_, b0), (_, b1) = v0.work(fn), v1.work(fn)
            ok &= p
            r["tests"].append({"kind": "with/without", "concept": name, "edit": desc, "expected": exp, "with": round(x0, 4),
                               "without": round(x1, 4), "change": round(d, 4), "share change": round(dsh, 5),
                               "y decoded without (stm)": round(float(A["positions"][fens.index(e)]["y decoded"]), 3),
                               "fen without": e, "total with": round(v0.total, 4), "total without": round(v1.total, 4),
                               "bits with": b0, "bits without": b1, "pass": bool(p)})
    for name, fn, test in DESC.get(key, []):
        x = fn(v0); extra = {}
        if isinstance(test, tuple):
            q = pct(abs(x), norm[test[1]]); p = q <= 25; extra = {"|x| pct": round(q, 1)}; test = "~0"
        else: p = {">0": x > 0, "<0": x < 0, "<=0": x <= 0}[test]
        ok &= p
        r["tests"].append({"kind": "descriptive", "concept": name, "expected": test, "value": round(x, 4), **extra,
                           "bits": v0.work(fn)[1], "pass": bool(p)})
    r["verdict"] = "supported" if ok else "not supported"
    res.append(r)
    print(f"{'+'.join(map(str, key)):6} {r['verdict']:14} " + "  ".join(
        (f"{t['concept']} {t['edit']}: {t['with']:+.3f} -> {t['without']:+.3f} (share {t['share change']:+.4f})"
         if t["kind"] == "with/without" else f"{t['concept']} {t['value']:+.3f}{' p%.0f' % t['|x| pct'] if '|x| pct' in t else ''}")
        + ("*" if t["pass"] else "") for t in r["tests"]), flush=True)

n = sum(r["verdict"] == "supported" for r in res)
print(f"{n} of {len(res)} games supported")
out = {"tap": "/encoder0/ln2", "method": A["method"], "spec": (ROOT / "lc0-probe/claim_concepts.md").read_text(),
       "norms": {k: {"n": len(x), "q25": float(np.percentile(x, 25)), "median": float(np.median(x))} for k, x in norm.items()},
       "games": res}
json.dump(out, open(ROOT / "lc0-probe/results_claim_concepts.json", "w"), indent=1)
(ROOT / "lc0-probe/claim_concepts.js").write_text("window.CONCEPTS = " + json.dumps(out, separators=(",", ":")) + ";\n")
