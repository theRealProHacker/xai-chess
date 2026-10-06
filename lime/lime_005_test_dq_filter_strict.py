"""lime_004 with stricter |dQ| cuts: drop samples whose Leela Q moved more than 0.7, 0.5 or 0.3 from the
base, alone and together with the two-push rule.

  python lime/lime_005_test_dq_filter_strict.py
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
s = importlib.util.spec_from_file_location("l", Path(__file__).with_name("lime_001.py"))
l = importlib.util.module_from_spec(s); sys.argv = ["x"]; s.loader.exec_module(l)
import chess
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

FEN = "1k2r3/1ppr2pp/p1p2p2/5R2/4P3/1P1P3P/P1P3P1/5RK1 w - - 1 27"
fens = l.perturb(FEN, 3000, np.random.default_rng(0)); rows = l.features(fens)
keep = [i for i, r in enumerate(rows) if r]
fens, rows = [fens[i] for i in keep], [rows[i] for i in keep]
q = l.leela_q(fens)
cols = sorted({(f["id"], f["side"]) for r in rows for f in r if l.family_of(f["id"]) != "initiative"},
              key=lambda c: (c[0], c[1] or ""))
ci = {c: i for i, c in enumerate(cols)}
Y = np.zeros((len(rows), len(cols)), np.float32)
for j, r in enumerate(rows):
    for f in r:
        if f["on"] and (f["id"], f["side"]) in ci:
            Y[j, ci[(f["id"], f["side"])]] = 1
dbl = np.array([next(f["id"] for f in r if f["on"] and f["id"].startswith("pawns.doubled") and f["side"] == "them")
                for r in rows])
base = chess.Board(FEN)
def deep(f):
    b = chess.Board(f)
    return any(chess.square_rank(s) == (5 if p.color == chess.WHITE else 2)
               for s, p in b.piece_map().items() if p.piece_type == chess.PAWN and base.piece_at(s) != p)
deep = np.array([deep(f) for f in fens])

def ev(m, name):
    Yk, qk, dk = Y[m], q[m], dbl[m]
    te = np.random.default_rng(0).random(len(qk)) < 0.2; te[0] = False
    X = Yk[:, Yk.min(0) != Yk.max(0)]
    v = qk[te].var(); mse = lambda p: float(((p - qk[te]) ** 2).mean())
    rid = mse(Ridge(1.0).fit(X[~te], qk[~te]).predict(X[te]))
    mlp = np.mean([mse(MLPRegressor(hidden_layer_sizes=(64,) * 4, early_stopping=True, max_iter=500, random_state=sd)
                       .fit(X[~te], qk[~te]).predict(X[te])) for sd in range(3)])
    by = " ".join(f"{qk[dk == f'pawns.doubled.{x}'].mean():+.2f}({(dk == f'pawns.doubled.{x}').sum()})"
                  for x in ("0", "1", "2", "3plus"))
    print(f"{name:14} kept {m.mean():.2f}  varQ {v:.3f}  ridge {rid:.3f} R2 {1 - rid / v:.2f}"
          f"  mlp {mlp:.3f} R2 {1 - mlp / v:.2f}  | Q by black doubled 0/1/2/3+: {by}")

ev(np.ones(len(q), bool), "all")
ev(~deep, "two-push rule")
for t in (0.7, 0.5, 0.3):
    ev(np.abs(q - q[0]) <= t, f"|dQ| <= {t}")
    ev(~deep & (np.abs(q - q[0]) <= t), f"both, {t}")
