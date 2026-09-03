# NNUE forward-pass dump patch

`dumpnet.patch` against pristine Stockfish 18 (`sf_18`, commit
`cb3d4ee9b47d0c5aae855b12379378ea1439675c`) — the same base `tools/stockfish-xai/build.sh`
pins. 252 insertions across 8 files, no deletions.

## Verification

| check | result |
|---|---|
| bench, dump inactive | **2050811** — exact match to `build.sh`'s `EXPECTED_BENCH` |
| `positional` recomputed in Python from dumped `fc0`/`fc2` | **3/3 exact** |
| record size | `static_assert(sizeof(DumpRecord) == 10780)` in C++, asserted again in the reader |
| in-check positions | dumped (calls `networks.big.evaluate` directly, below `Eval::evaluate`'s `assert(!pos.checkers())`) |
| `l1` reproduced from `acc`+`tacc` | **multiset-exact** per record, both perspectives (see l1 ordering below) |

Search behaviour is unchanged — the hooks are a null-pointer test on a `thread_local`
that is never set during search. Not binary-identical; the code does differ.

## Usage

    nnuedump <fen-list-file> <output-file>

One FEN per line, one 10780-byte record out per FEN. Batch, not per-position UCI
round-trips. Read with `nnue_dump.py`.

## What each record holds (big net only)

| field | shape | note |
|---|---|---|
| `idx_halfka` | ≤32 per perspective | HalfKAv2_hm active features |
| `idx_threats` | ≤128 per perspective | FullThreats active features |
| `acc` | 2 × 1024 int16 | accumulation — **linear** in the input |
| `tacc` | 2 × 1024 int16 | threatAccumulation — also linear |
| `l1` | 1024 uint8 | transformer output: clip + pairwise product, **the first nonlinearity** |
| `fc0` | 16 int32 | index 15 is the skip term feeding `fwdOut` |
| `ac0` | 30 uint8 | `ac_sqr_0 ‖ ac_0`, the `fc_1` input width |
| `fc1` | 32 int32 |
| `ac1` | 32 uint8 |
| `fc2` | int32 |
| `psqt`, `positional` | int32 | the engine's own returns |
| `bucket`, `stm`, `in_check`, `small_net_would_run`, `simple_eval` | scalars | |

Perspective index 0 is always the side to move, matching the net's own ordering.

## l1 is in SIMD packing order

On a vectorised build the transformer writes output through `vec_packus_16`, which
interleaves per 128-bit lane. `l1[j]` is therefore **not** the product of `acc[j]` and
`acc[j+512]` — it is that product under a fixed permutation of the 512 pair indices,
per perspective.

Verified from the dump alone, no engine and no weights: the multiset of `l1` equals
the multiset of `clip(acc+tacc)_lo * clip(acc+tacc)_hi // 512` exactly, while
elementwise agreement is only ~83%. That is the signature of a permutation, and it
jointly validates `acc`, `tacc` and `l1`.

For probing it is harmless: a fixed permutation of input dimensions changes no probe's
accuracy. It matters only for mapping an individual `l1` unit back to accumulator
units — recover the permutation empirically from a few thousand records first.

## Design notes

- No `nnueweights` command: `export_net` already exists (`uci.cpp:153`) and
  `write_parameters()` only re-emits the `.nnue` format. What is missing is a Python
  `.nnue` *parser*, not another C++ exporter.
- `small_net_would_run` is recorded rather than acted on. The dump always runs the big
  net; this flag marks positions where the engine itself would have used the 128-dim
  small net, which has `UseThreats == false` and therefore no `FullThreats` input at all.
- Deadlock fixed during development: `engine.nnue_dump()` must be called *before* taking
  the `sync_cout` lock, because `verify_networks()` prints through the same non-recursive
  mutex.
