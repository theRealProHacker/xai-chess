"""Fixed val/test rows for the supervised tasks in tasks.py, on an existing build_dataset.py dir (same positions,
split and acts.npy). One row per position per task; train resamples on the fly.
best and pv need <data>/sf.jsonl (oracle/sf_label.py); positions it lacks are left out of those two, so rerun once
it is complete.

  PYTHONPATH=.pylibs .venv/bin/python oracle/build_tasks.py --data oracle/data_g50k
"""
import argparse, collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "oracle")]
import numpy as np
import tasks

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "oracle/data_g50k"))
ap.add_argument("--tasks", nargs="+", default=tasks.TASKS, help="only these tasks (other row draws change with it)")
a = ap.parse_args()
D = Path(a.data)
L = np.load(D / "labels.npz")
fens, split = L["fens"], L["split"]
sf = {}
if (D / "sf.jsonl").exists():
    for l in (D / "sf.jsonl").open():
        r = json.loads(l); sf[r["i"]] = r["lines"]


def rows(idx, seed):
    rng = np.random.default_rng(seed)
    for i in idx:
        for t in a.tasks:
            r = tasks.sample(t, rng, str(fens[i]), sf.get(int(i)))
            if r is not None:
                yield dict(i=int(i), **r)


for name, k, seed in (("val", 1, 1), ("test", 2, 2), ("train_preview", 0, 3)):
    idx = np.flatnonzero(split == k)[:200 if k == 0 else None]
    n = collections.Counter()
    with open(D / f"tasks_{name}.jsonl", "w") as f:
        for r in rows(idx, seed):
            f.write(json.dumps(r) + "\n"); n[r["task"]] += 1
    print(f"tasks_{name}: {len(idx):,} positions,", " ".join(f"{t} {n[t]}" for t in a.tasks))

# base rates on val, the numbers a model has to beat
V = [json.loads(l) for l in open(D / "tasks_val.jsonl")]
by = collections.defaultdict(list)
for r in V: by[r["task"]].append(r)
ph = collections.Counter(r["target"] for r in by["phase"])
if not ph: sys.exit()                                   # base rates only for a full build
print("phase:", dict(ph), f"majority {max(ph.values()) / max(1, sum(ph.values())):.1%}")
for t, empty in (("attacks", "Nothing."), ("move_facts", "Nothing special."), ("move_attacks", "Nothing changes.")):
    print(f"{t}: empty answer {np.mean([r['answer'] == empty for r in by[t]]):.1%}")
print(f"control: mean {np.mean([len(r['target']) for r in by['control']]):.1f} squares")
