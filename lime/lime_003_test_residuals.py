"""Are lime_001's big held-out errors tactical samples? Same n=3000 sample, split and seed-0 MLP as run()/fit();
prints the 20 worst residuals with tags, then per-tag shares among the top-10% residuals and MSE without them.

  python lime/lime_003_test_residuals.py ["<fen>"]
"""
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lime_001", HERE / "lime_001.py")
L = importlib.util.module_from_spec(spec); argv, sys.argv = sys.argv, ["x"]; spec.loader.exec_module(L); sys.argv = argv
import chess, chess.engine, numpy as np
from sklearn.neural_network import MLPRegressor

FEN = sys.argv[1] if len(sys.argv) > 1 else "1k2r3/1ppr2pp/p1p2p2/5R2/4P3/1P1P3P/P1P3P1/5RK1 w - - 1 27"
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
SF = L.ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2"


def wins_material(b, c):
    """Side c attacks an enemy non-king piece that is undefended or hit by something cheaper."""
    for s, p in b.piece_map().items():
        if p.color == c or p.piece_type == chess.KING:
            continue
        att = b.attackers(c, s)
        if att and (not b.attackers(not c, s) or min(VAL[b.piece_type_at(a)] for a in att) < VAL[p.piece_type]):
            return True
    return False

def passers(b):
    out = set()
    for s in b.pieces(chess.PAWN, chess.WHITE) | b.pieces(chess.PAWN, chess.BLACK):
        c, f, r = b.color_at(s), chess.square_file(s), chess.square_rank(s)
        ahead = range(r + 1, 8) if c == chess.WHITE else range(0, r)
        if not any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead):
            out.add((s, c))
    return out

def tags(b, base, eng):
    t = {"stm_wins": wins_material(b, b.turn), "opp_wins": wins_material(b, not b.turn),
         "check": any(b.gives_check(m) for m in b.legal_moves), "new_passer": bool(passers(b) - passers(base))}
    if eng:  # Stockfish segfaults on >8 pawns / too much material: those count as untagged
        sc = lambda d: eng.analyse(b, chess.engine.Limit(depth=d), game=object())["score"].relative.score(mate_score=3000)
        t["sf_swing"] = b.is_valid() and abs(sc(8) - sc(1)) > 150  # fresh hash each call, else d1 reads d8
    return t

def diff(b, base):
    rm = [chess.square_name(s) for s, p in base.piece_map().items() if b.piece_at(s) != p]
    add = [("P" if b.color_at(s) else "p") + chess.square_name(s) for s in chess.SQUARES if b.piece_at(s) and not base.piece_at(s)]
    return rm, add


res, (fens, Y, cols, g) = L.run(FEN, 3000, np.random.default_rng(0))
q = L.leela_q(fens)
te = np.random.default_rng(0).random(len(q)) < 0.2; te[0] = False
X = Y[:, np.flatnonzero(Y.min(0) != Y.max(0))]
m = MLPRegressor(hidden_layer_sizes=(64, 64, 64, 64), early_stopping=True, max_iter=500, random_state=0).fit(X[~te], q[~te])
idx = np.flatnonzero(te)
pred = m.predict(X[idx]); err = np.abs(q[idx] - pred)
print(f"base Q {q[0]:+.3f}  n {len(q)}  held-out {len(idx)}  MSE {np.mean(err ** 2):.4f}  var Q {np.var(q[idx]):.4f}")

base = chess.Board(FEN)
eng = chess.engine.SimpleEngine.popen_uci(str(SF)) if SF.exists() else None
B = [chess.Board(fens[i]) for i in idx]
T = [tags(b, base, eng) for b in B]
if eng:
    eng.quit()
D = [diff(b, base) for b in B]
names = list(T[0])
tag = {k: np.array([t[k] for t in T]) for k in names}
tag["any"] = np.any([tag[k] for k in names], 0)
tag["over8"] = np.array([not b.is_valid() for b in B])  # not tactical: >8 pawns a side, off Leela's training data
tag["deep_pawn"] = np.array([any(chess.square_rank(s) == (5 if b.color_at(s) else 2) for s in chess.SQUARES
                                 if b.piece_at(s) and not base.piece_at(s)) for b in B])  # added pawn two pushes from queening
names += ["over8", "deep_pawn"]
removed = np.array([bool(rm) for rm, _ in D])

print(f"stockfish-searchable (valid) held-out boards: {np.mean([b.is_valid() for b in B]):.2f}")
print("\n=== 20 largest |Q - pred| ===")
for j in np.argsort(-err)[:20]:
    rm, add = D[j]
    print(f"\nQ {q[idx[j]]:+.3f}  pred {pred[j]:+.3f}  |err| {err[j]:.3f}  {'white' if B[j].turn else 'black'} to move"
          f"\nremoved {' '.join(rm) or '-'}  added {' '.join(add) or '-'}"
          f"\ntags {' '.join(k for k in names if tag[k][j]) or '-'}\n{B[j]}\n{fens[idx[j]]}")

top = err >= np.quantile(err, 0.9)
print("\n=== tags: share in top-10% |err| vs all held-out; MSE held-out without / with the tag, Q variance without ===")
for lab, sel in (("all", np.ones_like(removed)), ("removals", removed), ("addition-only", ~removed)):
    e2, qs = err[sel] ** 2, q[idx][sel]
    print(f"\n{lab}: n {sel.sum()}  MSE {e2.mean():.4f}  top-10% n {(top & sel).sum()}")
    print(f"  {'tag':12} {'top10%':>7} {'all':>7} {'MSE w/o':>8} {'MSE tag':>8} {'varQ w/o':>9}")
    for k in names + ["any"]:
        tk = tag[k][sel]; tt = tag[k][top & sel]
        print(f"  {k:12} {tt.mean() if len(tt) else float('nan'):7.2f} {tk.mean():7.2f} {e2[~tk].mean():8.4f} "
              f"{e2[tk].mean() if tk.any() else float('nan'):8.4f} {qs[~tk].var():9.4f}")
