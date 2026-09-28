"""The probe, the folds and the row cache, shared by sweep_labels.py and auto_labels.ipynb.

Held identical to probe_mlp.ipynb's config, so a difference between rows is a difference
in the representation or the label, never in the classifier.

One deliberate departure: the folds here are a plain KFold, not stratified. Stratified
folds depend on `y`, so six labels would get six different partitions and the ladders
would no longer be measured on the same split. Fold sizes are ~30k, so stratification
was cosmetic anyway.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED, FOLDS = 0, 5
MLP = dict(hidden_layer_sizes=(512, 128), activation="relu", solver="adam", alpha=1e-4,
           batch_size=512, learning_rate_init=1e-3, max_iter=200, early_stopping=True,
           n_iter_no_change=15, validation_fraction=0.1, random_state=SEED)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def labels():
    """(y [N, L] int8, names, metadata features, metadata feature names)."""
    z = np.load(DATA / "labels.npz", allow_pickle=False)
    return z["y"], list(z["names"]), z["meta_feat"], list(z["meta_names"])


def splits(n):
    """One partition for every label — that is the point of not stratifying."""
    return list(KFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n)))


def probe(X, y, sp, cfg=None, scale=True):
    cfg = cfg or MLP
    au, ac, it = [], [], []
    for tr, te in sp:
        if scale:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
        m = MLPClassifier(**cfg).fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        au.append(roc_auc_score(y[te], p))
        ac.append(accuracy_score(y[te], (p >= 0.5).astype(np.int8)))
        it.append(m.n_iter_)
        del Xtr, Xte, m, p
        gc.collect()
    return float(np.mean(au)), float(np.std(au)), float(np.mean(ac)), float(np.mean(it))


def cached():
    """Every row any worker has measured. Keys are "<label> <layer>"."""
    store = {}
    for f in sorted(DATA.glob("labels_*.json")):        # NOT sweep*.json — that is the
        store.update(json.loads(f.read_text()))         # published lichess run's cache
    return store


def measure(key, build_X, y, sp, cache, cfg=None, scale=True, **extra):
    """Train one row unless some worker already has it. Returns the row dict."""
    done = cached()
    if key in done:
        r = done[key]
        print(f"{key:26} {r['dims']:>6} {r['auroc']:>8.4f} {r['sd']:>7.4f}   cached", flush=True)
        return r
    X = build_X()
    t = time.time()
    auroc, sd, acc, epochs = probe(X, y, sp, cfg, scale)
    r = dict(key=key, dims=int(X.shape[1]), auroc=auroc, sd=sd, acc=acc, epochs=epochs,
             secs=time.time() - t, batch=(cfg or MLP)["batch_size"], **extra)
    del X
    gc.collect()
    print(f"{key:26} {r['dims']:>6} {auroc:>8.4f} {sd:>7.4f} {acc:>7.4f} "
          f"{r['secs'] / 60:>6.1f} min", flush=True)
    own = json.loads(cache.read_text()) if cache.exists() else {}
    own[key] = r
    cache.write_text(json.dumps(own, indent=1))
    return r
