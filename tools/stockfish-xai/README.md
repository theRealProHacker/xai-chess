# Rule-level Stockfish — reproducible build

`vendor/stockfish/stockfish-xai-rulelevel` is **not** checked into git (the
Stockfish binaries and nets are >100 MB each — over GitHub's limit — and live
under the git-ignored `vendor/`). This directory is the recipe to regenerate it
deterministically from source.

## What it is

Pristine **Stockfish 18** plus a small movegen/legality patch that adds a
`MaskPinner` UCI option. When `MaskPinner` is set to the square index (0–63) of a
pinning slider, the pin from that slider is *suspended*: the pinned piece is
allowed to move off the pin ray (and the enemy king may be left "exposed" along
it). This is the rule-level intervention used by the causal-faithfulness
experiments. The option defaults to `SQ_NONE`, so default behaviour is
byte-for-byte stock Stockfish 18.

The suspension is bound to the specific pinning piece, not the bare square. The
mask is *armed* only when its square genuinely pins one enemy piece to that
enemy's king (verified against the real board, not Stockfish's `pinners()`,
which counts a slider stacked behind the real pinner). Armed state is carried
per node through `do_move`/`undo_move`: it follows the pinner as it slides along
its own ray, and latches off (never to re-arm in that search branch) the instant
the pin breaks. Consequences that a naive bare-square mask got wrong, all now
covered by build-time regression checks: a non-slider check from the mask square
is never erased; a slider that pins nothing never frees a piece; a slider that
transits or re-occupies the square mid-search is never mistaken for the pinner;
and the king may step onto the severed ray (the suspended slider does not veto
it), symmetric with the check suspension.

## Pinned inputs

| Input | Value |
|---|---|
| Upstream | `github.com/official-stockfish/Stockfish`, tag `sf_18` |
| Commit | `cb3d4ee9b47d0c5aae855b12379378ea1439675c` |
| Patch | `rulelevel.patch` (4 files, +162 / −2) |
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

Six checks pin the behaviour and are arch/compiler-independent:

1. **Default bench** (`MaskPinner` inactive) = `2050811` nodes — identical to
   pristine Stockfish 18, confirming the patch is inert when unused.
2. **Intervention behaviour** on a pinned-knight position
   (`4k3/4n3/8/8/8/8/8/4R1K1 b - - 0 1`, black Ne7 pinned by Re1 to Ke8;
   `e1` = square index 4):

   | | `perft 1` |
   |---|---|
   | `MaskPinner` off | 4 (knight is pinned, king moves only) |
   | `MaskPinner` = 4 (e1) | 10 (pin suspended, knight gains 6 moves) |

3–5. **Soundness regressions.** Masking a square that is *not* a genuine pinner
   must be inert — `perft 1` identical to mask-off. Three cases, each of which a
   naive bare-square mask got wrong (erasing a real check or freeing a piece it
   does not pin): a non-slider check on the mask square
   (`R3k3/8/8/8/8/3n4/8/4K3 w`, mask `d3`); a slider giving a direct check that
   pins nothing (`4k3/8/8/4r3/8/8/R7/4K3 w`, mask `e5`); and a slider stacked
   behind the real pinner (`4r2k/8/4r3/8/4N3/8/8/4K3 w`, mask `e8`).

6. **No masked-search crash.** A full search under an active mask must return a
   move, not die. The mask can leave a king exposed; if the king-capture guard
   were gated on the mask being armed, a branch that latches the mask off could
   capture the exposed king and reach a kingless board that crashes NNUE
   (`make_index` on `SQ_NONE`). Perft cannot catch this (it does no eval), so a
   real search runs on `6k1/6pp/4n1P1/8/2B5/1r6/8/4R1K1 w` with mask `c4`.

All checks pass on a fresh rebuild from the pinned inputs.
