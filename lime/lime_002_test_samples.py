"""Is lime_001's surrogate data-limited or overfitting? Learning curve and train/test MSE on one position.

Samples once at the largest n; perturb() with a fixed seed draws the same sequence, so each smaller n is a prefix.
"Irreducible" = within-group variance of Q across identical feature rows on the training set, a floor for any
model of these features.

  python lime/lime_002_test_samples.py ["<fen>"] [--n 1000 3000 10000 20000]
"""
import argparse, importlib.util, sys, time
from pathlib import Path
sys.argv, argv = ["x"], sys.argv
spec = importlib.util.spec_from_file_location("lime_001", Path(__file__).resolve().parent / "lime_001.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
sys.argv = argv
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

FEN = "1k2r3/1ppr2pp/p1p2p2/5R2/4P3/1P1P3P/P1P3P1/5RK1 w - - 1 27"


def matrix(fen, n):
    """run()'s Y and Q without the relevance part."""
    fens = L.perturb(fen, n, np.random.default_rng(0))
    rows = L.features(fens)
    keep = [i for i, r in enumerate(rows) if r is not None]
    fens, rows = [fens[i] for i in keep], [rows[i] for i in keep]
    cols = sorted({(f["id"], f["side"]) for r in rows for f in r if L.family_of(f["id"]) != "initiative"},
                  key=lambda c: (c[0], c[1] or ""))
    ci = {c: i for i, c in enumerate(cols)}
    Y = np.zeros((len(rows), len(cols)), np.float32)
    for j, r in enumerate(rows):
        for f in r:
            if f["on"] and (f["id"], f["side"]) in ci:
                Y[j, ci[(f["id"], f["side"])]] = 1
    return np.array(keep), Y, L.leela_q(fens)

def floor(X, q):
    """Mean squared deviation of q from its group mean over identical rows of X; number of groups."""
    _, g = np.unique(X, axis=0, return_inverse=True)
    g = g.ravel()
    mu = np.bincount(g, q) / np.bincount(g)
    return float(np.mean((q - mu[g]) ** 2)), int(g.max() + 1)

def mse(m, X, q):
    return float(np.mean((m.predict(X) - q) ** 2))

def curve(Y, q):
    """lime_001.fit()'s split and models, reporting MSE instead of R²."""
    n = len(q)
    te = np.random.default_rng(0).random(n) < 0.2; te[0] = False
    X = Y[:, Y.min(0) != Y.max(0)]
    irr, groups = floor(X[~te], q[~te])
    r = Ridge(1.0).fit(X[~te], q[~te])
    tr, ts = [], []
    for seed in range(L.SEEDS):
        m = MLPRegressor(hidden_layer_sizes=(64, 64, 64, 64), early_stopping=True, max_iter=500, random_state=seed)
        m.fit(X[~te], q[~te]); tr.append(mse(m, X[~te], q[~te])); ts.append(mse(m, X[te], q[te]))
    return dict(n=n, cols=X.shape[1], distinct=len(np.unique(X, axis=0)), groups_tr=groups, var=float(q[te].var()),
                irr=irr, r_tr=mse(r, X[~te], q[~te]), r_te=mse(r, X[te], q[te]), m_tr=tr, m_te=ts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fen", nargs="?", default=FEN)
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 3000, 10000, 20000])
    a = ap.parse_args()
    t0 = time.time()
    keep, Y, q = matrix(a.fen, max(a.n))
    print(f"sampled {max(a.n)} -> {len(q)} rows after CLI errors in {time.time() - t0:.0f}s", flush=True)
    rng = lambda v: f"{np.mean(v):.4f} [{min(v):.4f}-{max(v):.4f}]"
    print(f"{'n':>6} {'cols':>4} {'distinct':>8} {'Qvar(te)':>8} {'irr(tr)':>7} {'ridge tr':>8} {'ridge te':>8}"
          f"  {'mlp tr mean [range]':>26}  {'mlp te mean [range]':>26}  time")
    for n in sorted(a.n):
        t = time.time()
        s = keep < n
        c = curve(Y[s], q[s])
        print(f"{c['n']:>6} {c['cols']:>4} {c['distinct']:>8} {c['var']:>8.4f} {c['irr']:>7.4f} {c['r_tr']:>8.4f}"
              f" {c['r_te']:>8.4f}  {rng(c['m_tr']):>26}  {rng(c['m_te']):>26}  {time.time() - t:.0f}s", flush=True)
    print(f"total {time.time() - t0:.0f}s")
