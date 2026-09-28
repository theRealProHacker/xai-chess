#!/usr/bin/env python3
"""Train every label on one layer, for one worker.

usage: OMP_NUM_THREADS=3 python sweep_labels.py <name> <layer> [<layer> ...]

The layer matrix is the expensive thing to build (acc||tacc is 2.5 GB), so a worker
builds it once and trains all six labels on it. Rows go to data/labels_<name>.json;
the notebook reads every labels_*.json and trains whatever is missing.

Layers come from probe_mlp.ipynb's `layer_matrix`, imported rather than re-derived:
nnue_np.probe_matrix defaults to perspective="stm", which reads half the position and
is the error §7 of that notebook documents (acc 0.760 against 0.821).
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".pylibs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import probe_lib as P
from nnue_np import load, probe_matrix

WIDE = {"acc": 2048, "tacc": 2048, "acc||tacc": 4096}
acts = load(str(P.DATA / "dump.bin"))


def layer_matrix(name):
    """(N, D) float32 for one layer, both perspectives — probe_mlp.ipynb cell 9."""
    if name.startswith("acc||tacc"):
        return np.hstack([probe_matrix(acts, "acc", "both"),
                          probe_matrix(acts, "tacc", "both")])
    if name in ("acc", "tacc"):
        return probe_matrix(acts, name, "both")
    return probe_matrix(acts, name)


def main():
    name, layers = sys.argv[1], sys.argv[2:]
    cache = P.DATA / f"labels_{name}.json"
    y, names, _, _ = P.labels()
    n = len(y)
    assert n == len(acts), f"{len(acts)} dump records vs {n} labels — misaligned"
    sp = P.splits(n)

    for layer in layers:
        todo = [j for j, lb in enumerate(names) if f"{lb} {layer}" not in P.cached()]
        if not todo:
            print(f"{layer:12} all labels cached", flush=True)
            continue
        X = layer_matrix(layer)
        if layer in WIDE:
            assert X.shape[1] == WIDE[layer], f"{layer}: {X.shape[1]}d, expected {WIDE[layer]}"
        for j in todo:
            P.measure(f"{names[j]} {layer}", lambda: X, y[:, j], sp, cache,
                      layer=layer, label=names[j])
        del X
        gc.collect()
    print("worker done", flush=True)


main()
