"""Space claims (1-4, 39) scored with Stockfish 11's space term, on integrated gradients from the mean baseline.

SF11 space for side X: zone = files c-f (here: the claim's files), X's ranks 2-4; safe = zone squares with no X pawn
and no enemy pawn attack; bonus = |safe| + |safe squares at most three behind an X pawn that no enemy piece attacks|;
space = bonus * (X's pieces - 1)^2 / 16. Each feature bit the formula reads gets weight w = its effect on space:
  pawn_cover.<enemy>.none on a zone square: +m, pawn_cover.<enemy>.one/two: -m, m = 1 + [behind, no enemy piece
  attacks it] (one enemy pawn bit more or less makes the square safe or not);
  an enemy piece's control.reach.t, t safe and behind: -1 when t has at most one such attacker (else 0).
The bit's space attribution against the same baseline is (x - mean) * w * (pieces - 1)^2 / 16 (exact: the formula is
linear in each bit it reads). Score = sum over bits of IG * that, IG signed for X; also over sum |attribution|.
--base mean (same occupant): occupant and pawn-geometry bits are always on in the baseline, so pawn placement gets no
attribution. --base square (any occupant): X's pawn on a square of the claim's files (occupant.<X>_pawn) gets
w = space with it - space without it (or with one added, if the square is empty), holding enemy attacks and the piece
count fixed.

  python lc0-probe/space_sf.py --base square
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import chess, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--base", choices=["mean", "square"], default="mean"); BASE = ap.parse_args().base
L = lambda f: json.loads(Path(f).read_text().split("=", 1)[1].rstrip().rstrip(";"))
A = L(ROOT / f"lc0-probe/square_attribution_n100000_encoder0_ln2_ig_{BASE}_claim_concepts.js")
fens = [l for l in (ROOT / "lc0-probe/data/claim_concepts.fens").read_text().split("\n") if l]
assert len(fens) == len(A["positions"])
TB = L(ROOT / "lc0-probe/square_shap_textbook.js")["positions"]
CC = {tuple(g["claims"]): g for g in json.load(open(ROOT / "lc0-probe/results_claim_concepts.json"))["games"]}
W, B = chess.WHITE, chess.BLACK
Z = np.load(ROOT / "lc0-probe/data/ig_means_n100000_20000.npz"); XM, XC = Z["XM"], Z["XC"]
SM = (XM * XC[:, None]).reshape(13, 64, -1).sum(0) / np.maximum(XC.reshape(13, 64).sum(0), 1)[:, None]   # any occupant
keep = __import__("torch").load(ROOT / "lc0-probe/data/square_decoder_encoder0_ln2_n100000.pt", weights_only=False)["keep"]
SCH = json.loads(__import__("subprocess").run([ROOT / "features/square/target/release/square", "--schema"],
                                               capture_output=True, text=True, check=True).stdout)["labels"]
COL = {SCH[i]: n for n, i in enumerate(keep)}
KEYS = ".PNBRQKpnbrqk"
NPM = {chess.KNIGHT: 781, chess.BISHOP: 825, chess.ROOK: 1276, chess.QUEEN: 2538}   # SF11 midgame values

def space(fen, X, files):
    """(formula space, score, score / sum|w|, bits [real square, label, IG, w]) on the board fen, for real side X."""
    real = chess.Board(fen); mir = real.turn == B
    b = real.mirror() if mir else real                                     # stored frame, as the IG file
    C = (not X) if mir else X                                              # X's stored colour
    r = A["positions"][fens.index(fen)]; assert r["fen"].split()[0] == b.fen().split()[0]
    them, up = not C, 1 if C == W else -1
    ranks = [1, 2, 3] if C == W else [6, 5, 4]
    zone = {chess.square("abcdefgh".index(f), k) for f in files for k in ranks}       # mirroring keeps files
    own = set(b.pieces(chess.PAWN, C))
    pa = set().union(*[b.attacks(p) for p in b.pieces(chess.PAWN, them)])
    att = {t: [a for a in chess.SQUARES if b.color_at(a) == them and b.piece_type_at(a) != chess.PAWN and t in b.attacks(a)]
           for t in chess.SQUARES}
    def bon(own):
        behind = {chess.square(chess.square_file(p), chess.square_rank(p) - up * d) for p in own for d in range(4)
                  if 0 <= chess.square_rank(p) - up * d < 8}
        safe = {t for t in zone if t not in own and t not in pa}
        return len(safe) + len({t for t in safe & behind if not att[t]}), safe, behind
    bonus, safe, behind = bon(own)
    k = (len(chess.SquareSet(b.occupied_co[C])) - 1) ** 2 / 16
    tn = "black" if C == W else "white"
    w = {}                                                                 # (stored square, label) -> weight
    for t in zone - own:
        m = 1 + (t in behind and not att[t]); n = chess.square_name(t)
        w[(t, f"pawn_cover.{tn}.none")] = m
        for x in ("one", "two"): w[(t, f"pawn_cover.{tn}.{x}")] = -m
    for t in safe & behind:
        if len(att[t]) <= 1:
            for a in chess.SQUARES:
                if b.color_at(a) == them and b.piece_type_at(a) != chess.PAWN:
                    w[(a, f"control.reach.{chess.square_name(t)}")] = -1
    pn = f"occupant.{'white' if C == W else 'black'}_pawn"
    for t in chess.SQUARES:
        if chess.square_file(t) in [("abcdefgh".index(f)) for f in files] and 1 <= chess.square_rank(t) <= 6:
            if t in own: w[(t, pn)] = bonus - bon(own - {t})[0]
            elif b.piece_at(t) is None: w[(t, pn)] = bon(own | {t})[0] - bonus
    ig, on = {}, {}
    for s, q in r["squares"].items():
        for j, v in q["bits"] + q.get("absent", []): ig[(chess.parse_square(s), A["labels"][j])] = v
        on[chess.parse_square(s)] = {A["labels"][j] for j, _ in q["bits"]}
    sg = 1 if C == W else -1
    occ = lambda s: KEYS.index(b.piece_at(s).symbol()) if b.piece_at(s) else 0
    xm = lambda s, l: 0.0 if l not in COL else float(SM[s, COL[l]] if BASE == "square" else XM[occ(s) * 64 + s, COL[l]])
    bits = [(s, l, sg * ig.get((s, l), 0.0), k * ((l in on.get(s, ())) - xm(s, l)) * x)
            for (s, l), x in w.items()]
    score = sum(v * x for _, _, v, x in bits)
    f = (lambda z: chess.square_name(chess.square_mirror(chess.parse_square(z)))) if mir else (lambda z: z)
    rl = lambda l: ".".join(f(p) if len(p) == 2 and p[0] in "abcdefgh" and p[1] in "12345678" else p for p in l.split("."))
    out = sorted(([f(chess.square_name(s)), rl(l), round(v, 5), round(x, 3)] for s, l, v, x in bits if v and x),
                 key=lambda z: -abs(z[2] * z[3]))
    npm = sum(NPM.get(p.piece_type, 0) for p in b.piece_map().values())
    return bonus * k, score, score / max(sum(abs(x) for *_, x in bits), 1e-9), out, npm

TESTS = {1: (W, "cdef"), 2: (B, "cdef"), 3: (W, "cdef"), 4: (B, "abcd"), 39: (B, "abc")}
res = []
for c, (X, files) in TESTS.items():
    key = next(k for k in CC if c in k); g = CC[key]
    sp0, s0, n0, b0, npm = space(g["orig_fen"], X, files)
    r = {"claim": c, "quote": g["quote"], "side": "white" if X == W else "black", "files": files, "orig_fen": g["orig_fen"],
         "space": round(sp0, 2), "score": round(s0, 5), "score/|w|": round(n0, 6), "npm": npm, "bits": b0[:40], "tests": []}
    line = f"{c:3} {r['side']:5} space {sp0:6.2f} score {s0:+.4f} ({n0:+.5f}){' npm<12222' if npm < 12222 else ''}"
    if c == 39:
        r["pass"] = s0 <= 0; line += f"  expect <= 0: {'pass' if r['pass'] else 'fail'}"
    else:
        for t in g["tests"]:
            sp1, s1, n1, b1, _ = space(t["fen without"], X, files)
            p = s1 < s0 and n1 < n0
            r["tests"].append({"edit": t["edit"], "space": round(sp1, 2), "score": round(s1, 5), "score/|w|": round(n1, 6),
                               "bits": b1[:40], "pass": bool(p)})
            line += f"\n      {t['edit']:7} space {sp1:6.2f} score {s1:+.4f} ({n1:+.5f})  {'pass' if p else 'fail'}"
        r["pass"] = all(t["pass"] for t in r["tests"])
    print(line, flush=True); res.append(r)
json.dump(res, open(ROOT / f"lc0-probe/results_space_sf_{BASE}.json", "w"), indent=1)
print(f"{sum(r['pass'] for r in res)} of {len(res)} supported")
