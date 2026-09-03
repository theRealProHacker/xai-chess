# sf-probe

A nonlinear (MLP) probe on **every layer** of the Stockfish 18 NNUE forward pass,
asking: *does this pin carry the Lichess `pin` theme tag?* — pin **relevance**, not
pin presence.

Layers only. No null projections, no handcrafted-geometry baseline.

```
probe_mlp.ipynb       layer sweep — which layer carries pin relevance
distance_mlp.ipynb    how far ahead that relevance is readable (l1 only), on a
                      50:50 set whose negatives are matched on puzzle length
three_class.ipynb     one classifier over {negative, d=0, d=1}, max-samples and
                      uniform arms, as a pair of confusion matrices
perspective.py        the perspective control for the layer sweep: acc/tacc re-probed
                      at BOTH perspectives, the fair comparison against l1
nnue_np.py            numpy reader for the XAINNUE1 dump record
build_dataset.py      Lichess puzzle CSV -> fens.txt + meta.tsv
pin_relevance.py      ... -> relevance.tsv: which solver move the pin becomes load-bearing on
dumpnet.patch         Stockfish 18 patch adding the `nnuedump` UCI command
DUMPNET.md            build + usage notes for the patched engine
data/                 dump.bin, meta.tsv, fens.txt, relevance.tsv, oof_l1_matched.npz,
                      oof_l1_sweep.npz, experiments.json, oof_3class.npz,
                      perspective.json  (git-ignored, 1.6 GB)
results_mlp.json      per-layer AUROC, sd, accuracy, epochs, runtime
results_mlp.csv       same, flat
layers_mlp.png        the layer plot
results_distance.*    per-distance AUROC, sd, recall, precision, accuracy, rating
distance_shared.png   AUROC vs distance, one model sliced by distance (part A)
distance_own.png      AUROC vs distance, a separate model per distance (part B)
                      — all three on `acc||tacc`, the default layer
results_3class.*      per-arm confusion matrix, per-class recall and precision,
                      accuracy, macro-F1
confusion_3class.png  the two confusion matrices, side by side
*_l1.*                the same three analyses on `l1`, kept for comparison. `acc||tacc`
                      is the default and holds the unsuffixed names; `LAYER` at the
                      top of each notebook picks the layer, and artifacts are named
                      after it so neither run overwrites the other
```

## Run it

```bash
PYTHONPATH=../.pylibs jupyter notebook probe_mlp.ipynb      # or
PYTHONPATH=../.pylibs python -c "
import nbformat; from nbclient import NotebookClient
nb = nbformat.read('probe_mlp.ipynb', as_version=4)
NotebookClient(nb, timeout=7200, resources={'metadata':{'path':'.'}}).execute()
nbformat.write(nb, 'probe_mlp.ipynb')"
```

Deps in `requirements.txt`. This repo keeps its packages in `../.pylibs`, hence the
`PYTHONPATH`; the notebook adds that directory itself if it exists.

Point at a dump elsewhere with `SF_PROBE_DUMP` / `SF_PROBE_META`.

`distance_mlp.ipynb` caches its fold predictions in `data/oof_l1_matched.npz` and the
resampled experiments in `data/experiments.json`, so re-running it to change the slicing
or a plot takes seconds instead of the ~8 minutes of training. Delete those to retrain.

## Rebuilding `data/`

`data/` is git-ignored — `dump.bin` is 1.6 GB. To regenerate:

```bash
# 1. patched engine
git -C ../vendor/stockfish apply ../../sf-probe/dumpnet.patch
make -C ../vendor/stockfish/src -j8 build ARCH=x86-64-avx2
# 2. dataset: 151,614 puzzles with exactly one sustained absolute pin
#    (https://database.lichess.org/lichess_db_puzzle.csv.zst)
zstdcat lichess_db_puzzle.csv.zst | python build_dataset.py
# 3. activations for every layer
printf 'nnuedump fens.txt data/dump.bin\nquit\n' | ../vendor/stockfish/src/stockfish
# 4. which solver move the pin becomes load-bearing on (for distance_mlp.ipynb)
zstdcat lichess_db_puzzle.csv.zst | python pin_relevance.py > data/relevance.tsv
```

## Method

| | |
|---|---|
| Positions | 151,614 Lichess puzzles, exactly one absolute pin sustained through every move |
| Sample | P0 — the position after `Moves[0]`, the one the solver sees |
| Label | the published Lichess `pin` theme, verbatim (28.48% positive) |
| Probe | `MLPClassifier(512, 128)`, adam, early stopping — the same config on every layer |
| CV | 5-fold stratified, identical splits across layers, scaler fit on train folds only |
| Metric | ROC AUC (chance 0.500), mean ± sd over folds |

Holding the probe fixed is what makes the columns comparable: a difference between
layers is a difference in the **representation**, not in the classifier.

`acc` and `tacc` are read at **both perspectives**, which is the form `l1` is stored in
(2 x 512 pairs, one per side); `acc||tacc` is their concatenation, the actual input to
the first nonlinearity. The first pass read them at the side-to-move perspective only,
which handed `l1` more of the position than its own inputs and made it look like the
best layer by +0.022; the fair comparison reverses that to −0.023. Both readings are in
`results_mlp.json` — the one-perspective rows are suffixed `@ stm`.
