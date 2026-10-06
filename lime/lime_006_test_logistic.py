"""Multinomial logistic surrogate on Leela's W/D/L against Ridge on Q, Capablanca - Janowski, two-push rule.
Logistic is fit on soft labels: each sample three times, labelled W, D, L, weighted by Leela's probabilities.

  python lime/lime_006_test_logistic.py
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
s = importlib.util.spec_from_file_location("l", Path(__file__).with_name("lime_001.py"))
l = importlib.util.module_from_spec(s); sys.argv = ["x"]; s.loader.exec_module(l)
import chess
from sklearn.linear_model import LogisticRegression, Ridge

FEN = "1k2r3/1ppr2pp/p1p2p2/5R2/4P3/1P1P3P/P1P3P1/5RK1 w - - 1 27"
fens = l.perturb(FEN, 3000, np.random.default_rng(0)); rows = l.features(fens)
keep = [i for i, r in enumerate(rows) if r]
fens, rows = [fens[i] for i in keep], [rows[i] for i in keep]
base = chess.Board(FEN)
deep = np.array([any(chess.square_rank(s) == (5 if p.color else 2) for s, p in chess.Board(f).piece_map().items()
                     if p.piece_type == chess.PAWN and base.piece_at(s) != p) for f in fens])
fens, rows = [f for f, d in zip(fens, deep) if not d], [r for r, d in zip(rows, deep) if not d]

l.leela_q(fens[:1])                                                          # opens the session
W = np.concatenate([l._sess.run(["/output/wdl"], {"/input/planes": l.encode_fens(fens[i:i + 1024])})[0]
                    for i in range(0, len(fens), 1024)]).astype(np.float64)
q = W[:, 0] - W[:, 2]
cols = sorted({(f["id"], f["side"]) for r in rows for f in r if l.family_of(f["id"]) != "initiative"},
              key=lambda c: (c[0], c[1] or ""))
ci = {c: i for i, c in enumerate(cols)}
Y = np.zeros((len(rows), len(cols)), np.float32)
for j, r in enumerate(rows):
    for f in r:
        if f["on"] and (f["id"], f["side"]) in ci:
            Y[j, ci[(f["id"], f["side"])]] = 1
X = Y[:, Y.min(0) != Y.max(0)]
te = np.random.default_rng(0).random(len(q)) < 0.2; te[0] = False

def report(name, p):
    v, e = q[te].var(), (p - q[te]) ** 2
    ext = np.abs(q[te]) > 0.6
    print(f"{name:22} MSE {e.mean():.3f}  R2 {1 - e.mean() / v:.2f}  |  |Q|>0.6 ({ext.sum()}): MSE {e[ext].mean():.3f}"
          f"  |  |Q|<=0.6: MSE {e[~ext].mean():.3f}")

print(f"{len(q)} samples kept, {te.sum()} held out, Q variance {q[te].var():.3f}")
for a in (0.1, 1.0, 10.0):
    report(f"ridge alpha {a}", Ridge(a).fit(X[~te], q[~te]).predict(X[te]))
tr = np.flatnonzero(~te)
for C in (0.1, 1.0, 10.0):
    lr = LogisticRegression(C=C, max_iter=2000).fit(np.repeat(X[tr], 3, 0), np.tile([0, 1, 2], len(tr)),
                                                    sample_weight=W[tr].ravel())
    P = lr.predict_proba(X[te])
    report(f"logistic C {C}", P[:, 0] - P[:, 2])
