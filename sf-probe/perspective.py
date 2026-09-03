#!/usr/bin/env python3
"""The perspective control for the layer sweep.

`probe_mlp.ipynb` took `acc` and `tacc` from the side-to-move perspective only —
1024 dims each — while `l1` is stored as both perspectives (2 x 512 pairs). So the
pre-`l1` ladder was handed strictly less of the position than `l1` was, and the
sweep's "l1 clears acc||tacc" gap was confounded with that.

This re-probes the three pre-`l1` columns at BOTH perspectives: acc 2048, tacc 2048,
acc||tacc 4096. Same probe, same folds, same seed as the sweep, so the numbers drop
straight into its table. Writes data/perspective.json, which the notebook reads.

    PYTHONPATH=../.pylibs ../.venv/bin/python perspective.py
"""
import gc, json, os, sys, time, warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
_pylibs = HERE.parent / ".pylibs"
if _pylibs.is_dir():
    sys.path.insert(0, str(_pylibs))
sys.path.insert(0, str(HERE))
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from nnue_np import load, probe_matrix

DUMP  = Path(os.environ.get("SF_PROBE_DUMP", HERE / "data" / "dump.bin"))
META  = Path(os.environ.get("SF_PROBE_META", HERE / "data" / "meta.tsv"))
OUT   = HERE / "data" / "perspective.json"

SEED, FOLDS = 0, 5
MLP = dict(hidden_layer_sizes=(512, 128), activation="relu", solver="adam", alpha=1e-4,
           batch_size=512, learning_rate_init=1e-3, max_iter=200, early_stopping=True,
           n_iter_no_change=15, validation_fraction=0.1, random_state=SEED)

# (row name, how to build it) — "both" is every perspective the dump carries.
COLUMNS = [
    ("acc (both)",       lambda a: probe_matrix(a, "acc", "both")),
    ("tacc (both)",      lambda a: probe_matrix(a, "tacc", "both")),
    ("acc||tacc (both)", lambda a: np.hstack([probe_matrix(a, "acc", "both"),
                                              probe_matrix(a, "tacc", "both")])),
]


def probe(X, y, splits):
    """5-fold CV MLP, transforms in place to keep a 4096-dim matrix affordable."""
    au, ac, it = [], [], []
    for tr, te in splits:
        Xtr = X[tr]                                  # fancy index: already a copy
        sc = StandardScaler().fit(Xtr)
        m = MLPClassifier(**MLP).fit(sc.transform(Xtr, copy=False), y[tr])
        del Xtr; gc.collect()
        p = m.predict_proba(sc.transform(X[te], copy=False))[:, 1]
        au.append(roc_auc_score(y[te], p))
        ac.append(accuracy_score(y[te], p >= 0.5))
        it.append(m.n_iter_)
        del m, sc, p; gc.collect()
    return float(np.mean(au)), float(np.std(au)), float(np.mean(ac)), float(np.mean(it))


def main():
    acts = load(str(DUMP))
    y = np.loadtxt(META, delimiter="\t", skiprows=1, usecols=(2,), dtype=np.int8)
    assert len(acts) == len(y), f"{len(acts)} records vs {len(y)} labels"
    N = len(y)
    # Identical to the sweep's SPLITS: same call, same seed, same y.
    splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(N), y))
    print(f"N = {N:,}   positive {y.mean()*100:.2f}%   folds {FOLDS}   seed {SEED}", flush=True)

    rows, t0 = [], time.time()
    print(f"{'row':18} {'dims':>6} {'AUROC':>8} {'sd':>7} {'acc':>7} {'epochs':>7} {'secs':>8}",
          flush=True)
    for name, build in COLUMNS:
        X = build(acts)
        t = time.time()
        auroc, sd, acc, epochs = probe(X, y, splits)
        dt = time.time() - t
        rows.append(dict(layer=name, dims=int(X.shape[1]), auroc=auroc, sd=sd, acc=acc,
                         epochs=epochs, secs=dt, batch=MLP["batch_size"],
                         note="both perspectives — the fair comparison against l1"))
        print(f"{name:18} {X.shape[1]:>6} {auroc:>8.4f} {sd:>7.4f} {acc:>7.4f} "
              f"{epochs:>7.1f} {dt:>8.1f}", flush=True)
        del X; gc.collect()

    meta = dict(n=int(N), positive=int(y.sum()), folds=FOLDS, seed=SEED,
                probe="MLPClassifier", mlp={k: str(v) for k, v in MLP.items()},
                perspective="both", total_min=(time.time() - t0) / 60)
    json.dump(dict(meta=meta, rows=rows), open(OUT, "w"), indent=1)
    print(f"\ntotal {meta['total_min']:.1f} min   wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
