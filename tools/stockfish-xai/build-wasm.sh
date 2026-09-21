#!/usr/bin/env bash
# Reproducible build of the xai-chess rule-level Stockfish WASM (the engine docs/pin.html loads).
#
# Produces docs/vendor/stockfish/stockfish-18-lite-single.{js,wasm} from source:
#   nmrugg/stockfish.js (pinned) + rulelevel.patch, compiled to WASM with the pinned Emscripten
#   the JS project requires. rulelevel.patch carries two intervention operators, each a UCI
#   option that is inert at its default, so the engine is stock Stockfish until one is set:
#     MaskPinner=<sq>   suspend the pin created by the slider on <sq>  (absolute and relative)
#     LineBlocker=<sq>  drop a magic blocker on <sq>: it obstructs every ray through the square
#                       and nothing may move onto it, but it is not a piece (no colour, no
#                       material, no Zobrist key, no NNUE feature). The dual of MaskPinner.
#   Squares are a1=0 .. h8=63; 64 (=SQ_NONE) is off. Clear Hash when changing either option --
#   neither is in the Zobrist key, so stale TT entries would answer for the wrong engine.
#
# The vendored .js/.wasm in docs/ were built by this script from this patch. Rebuilding is only
# needed when the patch changes; the browser demo (docs/pin.html) loads the vendored files.
#
# NOTE: the initial-research branch carries tools/stockfish-xai/build.sh, which builds the NATIVE
# binary for the Python research from its own, older rulelevel.patch — absolute pins only. That
# patch and this one are not interchangeable. Point build.sh at this patch to get an engine whose
# behaviour matches the demo.
#
# Requires: git, curl, node, and Emscripten 3.1.7 (via emsdk). ~15 min cold.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"      # adjust if relocated

# --- pinned inputs -----------------------------------------------------------
SFJS_REPO="https://github.com/nmrugg/stockfish.js.git"
SFJS_COMMIT="6a2a60eb3e3cd20bc1d6ad2f32f592c35233511c"
EMSCRIPTEN_VER="3.1.7"                              # nmrugg/stockfish.js build.js pins this exactly
LITE_NET="nn-9067e33176e8.nnue"                    # embedded in the lite variant
NET_URL="https://tests.stockfishchess.org/api/nn/${LITE_NET}"
PATCH="${1:-$SCRIPT_DIR/rulelevel.patch}"          # MaskPinner (pins) + LineBlocker (obstruct)
OUT_DIR="${2:-$REPO_ROOT/docs/vendor/stockfish}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

echo ">> Emscripten $EMSCRIPTEN_VER must be active (emcc --version)"
emcc --version | head -1 | grep -q "$EMSCRIPTEN_VER" || { echo "!! need emcc $EMSCRIPTEN_VER (install via emsdk: ./emsdk install $EMSCRIPTEN_VER && ./emsdk activate $EMSCRIPTEN_VER && source ./emsdk_env.sh)"; exit 1; }

echo ">> cloning nmrugg/stockfish.js @ $SFJS_COMMIT"
git clone --quiet "$SFJS_REPO" "$WORK/sfjs"
git -C "$WORK/sfjs" checkout --quiet "$SFJS_COMMIT"

echo ">> applying rulelevel.patch (MaskPinner pins + LineBlocker obstruct)"
git -C "$WORK/sfjs" apply "$PATCH"

echo ">> fetching lite net $LITE_NET"
curl -fsSL "$NET_URL" -o "$WORK/sfjs/src/$LITE_NET"

echo ">> building lite single-threaded WASM"
( cd "$WORK/sfjs" && node build.js --lite --single-threaded --basename stockfish-18-lite-single --do-not-verify-nets --force )

# VERIFY THE RESULT. Passing verify.sh against a native binary is NOT enough: the WASM comes from
# a different source tree, and a patch that is correct natively can be wrong here. nmrugg compiles
# out pos_is_ok()'s `Fast` early return and calls pos_is_ok() for real before every search, which
# once made an armed LineBlocker answer `bestmove (none)` for every square while native verify.sh
# stayed green. ./verify-wasm.sh drives the vendored file through node and exits non-zero on a
# failure -- run it after every re-vendor.

install -D "$WORK/sfjs/src/stockfish-18-lite-single.js"   "$OUT_DIR/stockfish-18-lite-single.js"
install -D "$WORK/sfjs/src/stockfish-18-lite-single.wasm" "$OUT_DIR/stockfish-18-lite-single.wasm"
echo ">> OK: re-vendored -> $OUT_DIR/stockfish-18-lite-single.{js,wasm}"

echo ">> verifying the vendored engine"
"$SCRIPT_DIR/verify-wasm.sh" "$OUT_DIR/stockfish-18-lite-single.js"
