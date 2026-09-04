# lc0-probe

`sf-probe`'s layer sweep on a Leela transformer: an MLP probe at every tap of the
residual stream, asking *does this pin carry the Lichess `pin` theme tag?* Same
151,614 positions, labels, probe and folds as `sf-probe`, so the ladders compare.

```
probe_mlp.ipynb    the sweep: 21 taps x 3 views, plus the raw input, against the Stockfish ladder
lc0_planes.py      INPUT_CLASSICAL_112_PLANE encoder (fen-only history fill), token index helper
check_encoding.py  our planes through onnxruntime vs lc0's own V and policy on 9 positions
pin_tokens.py      king / pinned / pinner squares as side-to-move token indices -> data/tokens.npy
dump_acts.py       runs the ONNX net, stores mean + 3 pin tokens per tap -> data/acts.npy (6.5 GB)
sweep.py           one worker's rows -> data/sweep_<name>.json; the notebook reads every sweep*.json
run_nb.py          headless notebook execution
results_mlp.*      per-row AUROC, sd, accuracy; layers_mlp.png the plot
data/              git-ignored
```

## Net and taps

`t1-256x10-distilled-swa-2432500` (networks-contrib): 10 post-LN encoder blocks, 8 heads,
d=256, smolgen. Taps: the embedding and the residual after each block's attention LN and
FFN LN, 21 in all. The residual is 64 tokens x 256; the probe sees per tap the mean over
squares (`mean`, 256d), the three tokens the pin lives on (`pin`, 768d), or both (`all`, 1024d).
Storing the full residual at every tap would be 150 GB.

## Rebuild

```bash
# lc0 (CPU, eigen) — needs meson + ninja
git clone --depth 1 --recurse-submodules https://github.com/LeelaChessZero/lc0 vendor/lc0
meson setup vendor/lc0/build/release --buildtype release -Dgtest=false -Dblas=true -Dopenblas=false -Dmkl=false -Ddnnl=false -Donnx=false -Dopencl=false -Dcudnn=false -Dplain_cuda=false -Ddx=false
ninja -C vendor/lc0/build/release
# net -> ONNX
curl -o vendor/lc0-nets/t1-256x10-distilled-swa-2432500.pb.gz https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz
vendor/lc0/build/release/lc0 leela2onnx --input=vendor/lc0-nets/t1-256x10-distilled-swa-2432500.pb.gz --output=vendor/lc0-nets/t1-256x10.onnx
# activations (~16 min on 8 cores)
python pin_tokens.py && python dump_acts.py
```

`check_encoding.py` wants lc0's own numbers: run `position fen ...` / `go nodes 1` with
`VerboseMoveStats` on, pausing between positions (lc0 exits on `quit` while a search runs).
Value agrees to 1e-5; policy to 5e-4 after lc0's default softmax temperature 1.359.

## Run

A 1024-d row is ~14 min on one 8-thread kernel, ~11 min per worker with three workers at
3 threads each, so the grid is trained by workers and the notebook only assembles:

```bash
OMP_NUM_THREADS=3 python sweep.py all embed:all enc0.ffn:all ...     # <tap>:<view> per row
OMP_NUM_THREADS=8 PYTHONPATH=../.pylibs python run_nb.py probe_mlp.ipynb
```

Rows: `all` at every tap, `pin` and `mean` at `embed`, `enc2.ffn`, `enc5.ffn`, `enc9.ffn`.
