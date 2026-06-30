# Rule-level Stockfish — reproducible build

`vendor/stockfish/stockfish-xai-rulelevel` is **not** checked into git (the
Stockfish binaries and nets are >100 MB each — over GitHub's limit — and live
under the git-ignored `vendor/`). This directory is the recipe to regenerate it
deterministically from source.

## What it is

Pristine **Stockfish 18** plus a small movegen patch that adds a `MaskPinner`
UCI option. When `MaskPinner` is set to the square index (0–63) of a pinning
slider, the pin from that slider is *suspended*: the pinned piece is allowed to
move off the pin ray (and the enemy king may be left "exposed" along it). This
is the rule-level intervention used by the causal-faithfulness experiments. The
option defaults to `SQ_NONE`, so default behaviour is byte-for-byte stock
Stockfish 18.

## Pinned inputs

| Input | Value |
|---|---|
| Upstream | `github.com/official-stockfish/Stockfish`, tag `sf_18` |
| Commit | `cb3d4ee9b47d0c5aae855b12379378ea1439675c` |
| Patch | `rulelevel.patch` (4 files, +49 / −3) |
| Big net | `nn-c288c895ea92.nnue` (`EvalFileDefaultNameBig`) |
| Small net | `nn-37f18f62d772.nnue` (`EvalFileDefaultNameSmall`) |
| Arch | `x86-64-avx2` |
| Reference compiler | `g++ (Ubuntu) 13.3.0` |

## Build

```sh
tools/stockfish-xai/build.sh
# -> writes vendor/stockfish/stockfish-xai-rulelevel and self-verifies
```

The script clones the pinned commit, applies `rulelevel.patch`, downloads the
pinned nets (`make net`), builds, and fails loudly unless both verification
checks below pass.

## Verification (what "reproducible" means here)

Reproduction is verified **functionally**, not by binary hash — `incbin` embeds
the net and the build embeds compiler/path strings, so the SHA-256 varies across
toolchains and is not a reliable equality test. The reference binary built with
g++ 13.3.0 was `2b567ac77f243d301d1ed099589b9f5201b3b8edfb1cdeddef32fb6812d2f7f7`,
recorded for information only.

Two checks pin the behaviour and are arch/compiler-independent:

1. **Default bench** (`MaskPinner` inactive) = `2050811` nodes — identical to
   pristine Stockfish 18, confirming the patch is inert when unused.
2. **Intervention behaviour** on a pinned-knight position
   (`4k3/4n3/8/8/8/8/8/4R1K1 b - - 0 1`, black Ne7 pinned by Re1 to Ke8;
   `e1` = square index 4):

   | | `perft 1` |
   |---|---|
   | `MaskPinner` off | 4 (knight is pinned, king moves only) |
   | `MaskPinner` = 4 (e1) | 10 (pin suspended, knight gains 6 moves) |

Both the committed reference binary and a fresh rebuild produce exactly these
numbers.
