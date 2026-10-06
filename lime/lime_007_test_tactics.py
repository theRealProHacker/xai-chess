"""Tactic columns as extra surrogate inputs, Capablanca - Janowski, two-push rule. They are held at the base
value when a family is ablated and are not reported as families. Per side (us = side to move, them = other side,
read as if it were to move):
  win_N/B/R/Q   an enemy piece of that type is attacked by something cheaper, or attacked and unguarded
  hanging_pawn  an enemy pawn is attacked and unguarded
  mate          mate in one
  check         a check is available
  prot_passer   a passed pawn guarded by an own pawn

  python lime/lime_007_test_tactics.py ["<fen>"]
"""
import importlib.util, sys
ARGV = sys.argv[1:]
from pathlib import Path
import numpy as np
s = importlib.util.spec_from_file_location("l", Path(__file__).with_name("lime_001.py"))
l = importlib.util.module_from_spec(s); sys.argv = ["x"]; s.loader.exec_module(l)
import chess
from sklearn.linear_model import Ridge

FEN = ARGV[0] if ARGV else "1k2r3/1ppr2pp/p1p2p2/5R2/4P3/1P1P3P/P1P3P1/5RK1 w - - 1 27"
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
TAGS = ["win_N", "win_B", "win_R", "win_Q", "hanging_pawn", "mate", "check", "prot_passer"]

def passed(b, sq, c):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ahead = range(r + 1, 8) if c == chess.WHITE else range(0, r)
    return not any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead)

def tags(b):
    """The TAGS for the side to move of b."""
    c, t = b.turn, {k: 0 for k in TAGS}
    for sq, p in b.piece_map().items():
        if p.color == c or p.piece_type == chess.KING:
            continue
        att = b.attackers(c, sq)
        if not att:
            continue
        guarded = bool(b.attackers(not c, sq))
        cheap = min(VAL[b.piece_type_at(a)] for a in att) < VAL[p.piece_type]
        if p.piece_type == chess.PAWN:
            t["hanging_pawn"] |= not guarded
        elif cheap or not guarded:
            t["win_" + chess.piece_symbol(p.piece_type).upper()] = 1
    for m in b.legal_moves:
        t["check"] |= b.gives_check(m)
        if not t["mate"] and b.gives_check(m):
            b.push(m); t["mate"] = int(b.is_checkmate()); b.pop()
    t["prot_passer"] = int(any(passed(b, sq, c) and b.attackers(c, sq) & b.pieces(chess.PAWN, c)
                               for sq in b.pieces(chess.PAWN, c)))
    return [int(t[k]) for k in TAGS]

def both(f):
    b = chess.Board(f); o = b.copy(stack=False); o.turn = not o.turn; o.ep_square = None
    return tags(b) + tags(o)

fens = l.perturb(FEN, 3000, np.random.default_rng(0)); rows = l.features(fens)
keep = [i for i, r in enumerate(rows) if r]
fens, rows = [fens[i] for i in keep], [rows[i] for i in keep]
base = chess.Board(FEN)
deep = [any(chess.square_rank(s) == (5 if p.color else 2) for s, p in chess.Board(f).piece_map().items()
            if p.piece_type == chess.PAWN and base.piece_at(s) != p) for f in fens]
fens, rows = [f for f, d in zip(fens, deep) if not d], [r for r, d in zip(rows, deep) if not d]
q = l.leela_q(fens)
cols = sorted({(f["id"], f["side"]) for r in rows for f in r if l.family_of(f["id"]) != "initiative"},
              key=lambda c: (c[0], c[1] or ""))
ci = {c: i for i, c in enumerate(cols)}
Y = np.zeros((len(rows), len(cols)), np.float32)
for j, r in enumerate(rows):
    for f in r:
        if f["on"] and (f["id"], f["side"]) in ci:
            Y[j, ci[(f["id"], f["side"])]] = 1
T = np.array([both(f) for f in fens], np.float32)
tnames = [f"{k}.{s}" for s in ("us", "them") for k in TAGS]
g = l.groups(cols)

print(f"{len(q)} samples, Q variance {q.var():.3f}")
print("tag        share us  share them  (base us/them)")
for i, k in enumerate(TAGS):
    print(f"  {k:13} {T[:, i].mean():.2f}      {T[:, i + len(TAGS)].mean():.2f}      ({int(T[0, i])}/{int(T[0, i + len(TAGS)])})")

fid0, rel0 = l.fit(Y, q, g)
YT = np.hstack([Y, T])
fid1, rel1 = l.fit(YT, q, g)
print(f"\nfeatures only      R2 mlp {fid0['mlp']:.2f} ridge {fid0['ridge']:.2f}")
print(f"features + tactics R2 mlp {fid1['mlp']:.2f} ridge {fid1['ridge']:.2f}")
TT = T[:, T.min(0) != T.max(0)]
te = np.random.default_rng(0).random(len(q)) < 0.2; te[0] = False
print(f"tactics only       R2 ridge {Ridge(1.0).fit(TT[~te], q[~te]).score(TT[te], q[te]):.2f}")

var = YT.min(0) != YT.max(0)
w = Ridge(1.0).fit(YT[:, var], q).coef_
names = [f"{a}.{s}" if s else a for a, s in cols] + tnames
print("\nRidge weights of the tactic columns (Q for the side to move):")
for n, x in sorted(zip([n for n, v in zip(names, var) if v], w), key=lambda x: -abs(x[1])):
    if n in tnames:
        print(f"  {n:20} {x:+.2f}")

nm = lambda k: k.replace(".us", " (us)").replace(".them", " (them)")
m = lambda r: {k: (np.mean(v), (np.array(v) > 0).all() or (np.array(v) < 0).all()) for k, v in r.items() if v}
r0, r1 = m(rel0), m(rel1)
print(f"\n{'family':34} {'features':>9} {'+tactics':>9}")
for k in sorted(r0, key=lambda k: -abs(r0[k][0]))[:12]:
    print(f"{nm(k):34} {r0[k][0]:+8.2f}{'' if r0[k][1] else '?'} {r1[k][0]:+8.2f}{'' if r1[k][1] else '?'}")
