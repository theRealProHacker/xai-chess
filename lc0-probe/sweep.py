"""Train the probe rows for one worker, caching each to data/sweep_<name>.json.

usage: OMP_NUM_THREADS=3 python sweep.py <name> <tap>:<view> [<tap>:<view> ...]

The notebook holds the same probe and reads every data/sweep_*.json, so rows
measured here are the notebook's rows; three workers x 3 threads finish the
ladder about twice as fast as one kernel x 8 threads (sklearn's 512-row
batches do not fill 8 cores).
"""
import os, sys, json, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".pylibs"))
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

HERE = Path(__file__).resolve().parent
name, jobs = sys.argv[1], [j.split(":") for j in sys.argv[2:]]
CACHE = HERE / "data" / f"sweep_{name}.json"
SEED, FOLDS = 0, 5
VIEWS = {"all": [0, 1, 2, 3], "pin": [1, 2, 3], "mean": [0]}
MLP = dict(hidden_layer_sizes=(512, 128), activation="relu", solver="adam", alpha=1e-4,
           batch_size=512, learning_rate_init=1e-3, max_iter=200, early_stopping=True,
           n_iter_no_change=15, validation_fraction=0.1, random_state=SEED)

acts = np.load(HERE / "data" / "acts.npy", mmap_mode="r")
TAPS = json.load(open(HERE / "data" / "taps.json"))["taps"]
y = np.loadtxt(HERE.parent / "sf-probe" / "data" / "meta.tsv", delimiter="\t", skiprows=1,
               usecols=(2,), dtype=np.int8)
N = len(y)
SPLITS = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(N), y))

def probe(X):
    au, ac, it = [], [], []
    for tr, te in SPLITS:
        sc = StandardScaler().fit(X[tr])
        m = MLPClassifier(**MLP).fit(sc.transform(X[tr]), y[tr])
        p = m.predict_proba(sc.transform(X[te]))[:, 1]
        au.append(roc_auc_score(y[te], p)); ac.append(accuracy_score(y[te], p >= 0.5)); it.append(m.n_iter_)
        del m, p; gc.collect()
    return float(np.mean(au)), float(np.std(au)), float(np.mean(ac)), float(np.mean(it))

store = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for tap, view in jobs:
    key = f"{tap} {view}"
    if key in store:
        print(f"{key:16} cached", flush=True); continue
    X = np.ascontiguousarray(acts[:, TAPS.index(tap), VIEWS[view], :]).reshape(N, -1).astype(np.float32)
    t = time.time()
    auroc, sd, acc, epochs = probe(X)
    store[key] = dict(layer=key, dims=int(X.shape[1]), auroc=auroc, sd=sd, acc=acc, epochs=epochs,
                      secs=time.time() - t, batch=MLP["batch_size"], tap=tap, view=view)
    del X; gc.collect()
    CACHE.write_text(json.dumps(store, indent=1))
    print(f"{key:16} {store[key]['dims']:>6} {auroc:.4f} ± {sd:.4f}  {(time.time()-t)/60:.1f} min", flush=True)
print("worker done", flush=True)
