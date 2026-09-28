#!/usr/bin/env bash
# Vendor the tagger that produced the Lichess `pin` theme. Pinned: cook_pin.py's
# predicates are copied from this commit and checked against it.
set -euo pipefail
COMMIT=8d9faff694ba3a8598abc5465347209af3f90a82
DEST="$(git rev-parse --show-toplevel)/vendor/lichess-puzzler"
[ -d "$DEST" ] || git clone https://github.com/ornicar/lichess-puzzler "$DEST"
git -C "$DEST" fetch origin "$COMMIT" 2>/dev/null || git -C "$DEST" fetch origin
git -C "$DEST" checkout -q "$COMMIT"
echo "vendored $DEST @ $COMMIT"
