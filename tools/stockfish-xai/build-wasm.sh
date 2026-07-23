#!/usr/bin/env bash
# Reproducible build of the xai-chess rule-level Stockfish WASM (the engine docs/pin.html loads).
#
# Produces docs/vendor/stockfish/stockfish-18-lite-single.{js,wasm} from source:
#   nmrugg/stockfish.js (pinned) + rulelevel.patch, compiled to WASM with the pinned Emscripten
#   the JS project requires. This rulelevel.patch handles both absolute pins (to the king) and
#   RELATIVE pins (to a more valuable backstop) via MaskPinner.
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
PATCH="${1:-$SCRIPT_DIR/rulelevel.patch}"          # absolute + relative pin MaskPinner
OUT_DIR="${2:-$REPO_ROOT/docs/vendor/stockfish}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

echo ">> Emscripten $EMSCRIPTEN_VER must be active (emcc --version)"
emcc --version | head -1 | grep -q "$EMSCRIPTEN_VER" || { echo "!! need emcc $EMSCRIPTEN_VER (install via emsdk: ./emsdk install $EMSCRIPTEN_VER && ./emsdk activate $EMSCRIPTEN_VER && source ./emsdk_env.sh)"; exit 1; }

echo ">> cloning nmrugg/stockfish.js @ $SFJS_COMMIT"
git clone --quiet "$SFJS_REPO" "$WORK/sfjs"
git -C "$WORK/sfjs" checkout --quiet "$SFJS_COMMIT"

echo ">> applying rulelevel.patch (absolute + relative pin MaskPinner)"
git -C "$WORK/sfjs" apply "$PATCH"

echo ">> fetching lite net $LITE_NET"
curl -fsSL "$NET_URL" -o "$WORK/sfjs/src/$LITE_NET"

echo ">> building lite single-threaded WASM"
( cd "$WORK/sfjs" && node build.js --lite --single-threaded --basename stockfish-18-lite-single --do-not-verify-nets --force )

# Not verified here: the WASM speaks UCI over a Web Worker, so there is no CLI to pipe into.
# The mask semantics are checked by verify.sh against a native binary built from this same patch;
# the WASM itself is checked by hand in docs/pin.html.

install -D "$WORK/sfjs/src/stockfish-18-lite-single.js"   "$OUT_DIR/stockfish-18-lite-single.js"
install -D "$WORK/sfjs/src/stockfish-18-lite-single.wasm" "$OUT_DIR/stockfish-18-lite-single.wasm"
echo ">> OK: re-vendored -> $OUT_DIR/stockfish-18-lite-single.{js,wasm}"
