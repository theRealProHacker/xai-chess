#!/usr/bin/env bash
# Reproducible build of the xai-chess rule-level Stockfish.
#
# Reconstructs vendor/stockfish/stockfish-xai-rulelevel from source:
#   pristine Stockfish 18 (tag sf_18, pinned commit) + rulelevel.patch,
#   built for ARCH=x86-64-avx2 with the pinned NNUE nets, then self-verified
#   against a known bench signature and the MaskPinner intervention behaviour.
#
# Usage:  ./build.sh [OUTPUT_PATH]
#   OUTPUT_PATH defaults to  <repo>/vendor/stockfish/stockfish-xai-rulelevel
#   ARCH can be overridden:  ARCH=x86-64-bmi2 ./build.sh
#
# Requires: git, g++ (or clang), make, and network access to clone Stockfish
# and download the pinned nets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- pinned inputs -----------------------------------------------------------
SF_REPO="https://github.com/official-stockfish/Stockfish.git"
SF_TAG="sf_18"
SF_COMMIT="cb3d4ee9b47d0c5aae855b12379378ea1439675c"
ARCH="${ARCH:-x86-64-avx2}"
NET_BIG="nn-c288c895ea92.nnue"      # EvalFileDefaultNameBig
NET_SMALL="nn-37f18f62d772.nnue"    # EvalFileDefaultNameSmall
EXPECTED_BENCH="2050811"            # default bench, MaskPinner inactive
PATCH="$SCRIPT_DIR/rulelevel.patch"

OUTPUT="${1:-$REPO_ROOT/vendor/stockfish/stockfish-xai-rulelevel}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ">> cloning Stockfish $SF_TAG"
git clone --quiet --depth 1 --branch "$SF_TAG" "$SF_REPO" "$WORK/sf"
got="$(git -C "$WORK/sf" rev-parse HEAD)"
[ "$got" = "$SF_COMMIT" ] || { echo "!! commit mismatch: got $got, want $SF_COMMIT"; exit 1; }

echo ">> applying rulelevel.patch"
git -C "$WORK/sf" apply "$PATCH"

echo ">> building (ARCH=$ARCH); 'net' downloads the pinned nets"
make -C "$WORK/sf/src" -j"$(nproc)" ARCH="$ARCH" net build >/dev/null

for n in "$NET_BIG" "$NET_SMALL"; do
  [ -f "$WORK/sf/src/$n" ] || { echo "!! expected net $n not present after build"; exit 1; }
done

BIN="$WORK/sf/src/stockfish"

echo ">> verifying default bench == $EXPECTED_BENCH"
bench="$(printf 'bench\nquit\n' | "$BIN" 2>&1 | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1)"
[ "$bench" = "$EXPECTED_BENCH" ] || { echo "!! bench mismatch: $bench != $EXPECTED_BENCH"; exit 1; }

echo ">> verifying MaskPinner intervention (pinned knight perft1: 4 -> 10)"
FEN="4k3/4n3/8/8/8/8/8/4R1K1 b - - 0 1"   # black Ne7 pinned by Re1 to Ke8; e1 = square index 4
off="$(printf "position fen %s\ngo perft 1\nquit\n" "$FEN" | "$BIN" 2>&1 | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1)"
on="$( printf "setoption name MaskPinner value 4\nposition fen %s\ngo perft 1\nquit\n" "$FEN" | "$BIN" 2>&1 | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1)"
{ [ "$off" = "4" ] && [ "$on" = "10" ]; } || { echo "!! MaskPinner check failed: off=$off on=$on (want 4 / 10)"; exit 1; }

# Soundness regressions: the mask must NEVER erase a check that is not the
# suspended pin's revealed check, nor free a piece it does not genuinely pin.
# For each, masking the given square must leave perft1 IDENTICAL to mask-off
# (the mask arms nothing, because the square is not a genuine pinner).
echo ">> verifying mask soundness (non-pinner squares must be inert)"
perft1() { printf "$2\ngo perft 1\nquit\n" | "$BIN" 2>&1 | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1; }
check_inert() { # label, fen, maskSquare
  local o m
  o="$(perft1 x "position fen $2")"
  m="$(perft1 x "setoption name MaskPinner value $3\nposition fen $2")"
  [ "$o" = "$m" ] || { echo "!! soundness check '$1' failed: off=$o mask=$m (want equal)"; exit 1; }
  echo "   ok: $1 (perft1=$o, mask inert)"
}
# knight checker on the mask square (not a slider): check must stand
check_inert "non-slider check"   "R3k3/8/8/8/8/3n4/8/4K3 w - - 0 1"      19
# slider checking directly (pins nothing): check must stand
check_inert "direct-slider check" "4k3/8/8/4r3/8/8/R7/4K3 w - - 0 1"     36
# a slider stacked behind the real pinner (pins nothing): must not free the piece
check_inert "stacked non-pinner"  "4r2k/8/4r3/8/4N3/8/8/4K3 w - - 0 1"   60

# A full search under an active mask must not crash. The mask can leave a king
# exposed; if the king-capture guard is gated on the mask being armed, a branch
# that latches the mask off can capture the exposed king and reach a kingless
# board that crashes NNUE. Perft can't catch this (no eval); a real search must.
# stdin is held open (no trailing 'quit') so the search runs to the node limit.
echo ">> verifying a masked search does not crash (kingless-board regression)"
CRASH_FEN="6k1/6pp/4n1P1/8/2B5/1r6/8/4R1K1 w - - 0 1"   # bishop pin; once reached a kingless node
bm="$( ( printf 'setoption name Hash value 64\nsetoption name MaskPinner value 26\nposition fen %s\ngo nodes 300000\n' "$CRASH_FEN"; sleep 6 ) | "$BIN" 2>&1 | grep -i '^bestmove' | awk '{print $2}' )"
[ -n "$bm" ] || { echo "!! masked search crashed (no bestmove returned)"; exit 1; }
echo "   ok: masked search completed (bestmove=$bm)"

install -D "$BIN" "$OUTPUT"
echo ">> OK: reproduced -> $OUTPUT"
echo "   bench=$bench  MaskPinner perft1 off=$off on=$on  arch=$ARCH"
