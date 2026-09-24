#!/usr/bin/env bash
# Verify the VENDORED WASM engine -- the file the browser demos actually load.
#
# verify.sh checks the operators against a NATIVE binary. That is not enough, and this script
# exists because it was not: the WASM is built from a DIFFERENT source tree
# (nmrugg/stockfish.js), and a patch that is correct natively can still be wrong there.
# It happened. nmrugg compiles out pos_is_ok()'s `Fast` early return AND calls pos_is_ok() for
# real from uci.cpp before every search, so the blocker's extra occupancy bit made the engine
# refuse to search and answer `bestmove (none)` for every armed square. Native verify.sh passed
# the whole time. Run this after every re-vendor.
#
# Usage: ./verify-wasm.sh [PATH_TO_ENGINE_JS]
#   Exits 0 if every check passes, 1 otherwise. Needs node.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${1:-$HERE/../../docs/vendor/stockfish/stockfish-18-lite-single.js}"
[ -f "$ENGINE" ] || { echo "!! no engine at $ENGINE"; exit 1; }
command -v node >/dev/null || { echo "!! node is required"; exit 1; }
DIR="$(cd "$(dirname "$ENGINE")" && pwd)"; JS="$(basename "$ENGINE")"

FAILED=0
# One node session per call. The engine needs a few seconds to instantiate the wasm before it
# reads stdin, and it answers on stdout; `quit` sent too early exits before anything is printed.
run(){ ( printf '%b\n' "$1"; sleep "${2:-7}"; printf 'quit\n' ) | ( cd "$DIR" && timeout 90 node "./$JS" 2>/dev/null ); }
perft1(){ run "$1\ngo perft 1" 6 | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1; }
evalcp(){ run "$1\ngo nodes 200000" 9 | grep -oE 'score (cp|mate) -?[0-9]+' | tail -1; }
check(){ # name  got  want
  if [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"
  else printf 'FAIL  %s\n        got  %s\n        want %s\n' "$1" "$2" "$3"; FAILED=$((FAILED+1)); fi; }
checklike(){ # name  got  regex  want-description
  if printf '%s' "$2" | grep -qE "$3"; then printf 'ok    %s\n' "$1"
  else printf 'FAIL  %s\n        got  %s\n        want %s\n' "$1" "$2" "$4"; FAILED=$((FAILED+1)); fi; }

echo "engine: $ENGINE"

OPTS="$(run 'uci' 6)"
checklike "all three options are registered" \
  "$(printf '%s' "$OPTS" | grep -cE 'option name (MaskPinner|LineBlocker|LineBlockerAxis) ')" '^3$' "3 options"

# The bug that shipped: with a blocker on an EMPTY square the engine returned no legal move at
# all. The cheapest position with a legal move catches it.
BARE='4k3/8/8/8/8/8/8/4K3 w - - 0 1'
check "bare kings, blocker off"      "$(perft1 "position fen $BARE")" "5"
check "blocker on an EMPTY square does not empty the move list" \
      "$(perft1 "setoption name LineBlocker value 0\nposition fen $BARE")" "5"

# G/H/O, same positions verify.sh uses: the blocker cuts the ray of the matching slider class
# and nothing else.
R='4k3/8/8/8/8/8/8/R3K3 w - - 0 1'
check "rook on the open a-file, blocker off" "$(perft1 "position fen $R")" "15"
check "blocker cuts the rook ray (a4, orthogonal)" \
      "$(perft1 "setoption name LineBlocker value 24\nsetoption name LineBlockerAxis value 0\nposition fen $R")" "11"
check "the wrong axis leaves the rook alone (a4, diagonal)" \
      "$(perft1 "setoption name LineBlocker value 24\nsetoption name LineBlockerAxis value 1\nposition fen $R")" "15"
check "a ray FROM the blocker's own square is not cut (a1)" \
      "$(perft1 "setoption name LineBlocker value 0\nsetoption name LineBlockerAxis value 0\nposition fen $R")" "15"

# The false positive the axis split removes: a pawn is not a rook, so a blocker on its file
# must not stop the push. Under the old one-OR-into-pieces() blocker it did, and every file a
# pawn marched down read as load-bearing.
PAWN='4k3/8/8/8/8/8/P7/4K3 w - - 0 1'
check "pawn file, blocker off" "$(perft1 "position fen $PAWN")" "7"
check "an orthogonal blocker does not stop a pawn push (a3)" \
      "$(perft1 "setoption name LineBlocker value 16\nsetoption name LineBlockerAxis value 0\nposition fen $PAWN")" "7"

AFILE='6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1'

# L: the operator's point -- cut the line that carries the mate, with a placebo off it.
checklike "mate stands with the line open" "$(evalcp "position fen $AFILE")" 'mate 1$' "score mate 1"
checklike "cutting the line kills the mate" \
      "$(evalcp "setoption name LineBlocker value 24\nsetoption name LineBlockerAxis value 0\nposition fen $AFILE")" '^score cp' "a cp score"
checklike "a placebo square off the line keeps the mate" \
      "$(evalcp "setoption name LineBlocker value 25\nsetoption name LineBlockerAxis value 0\nposition fen $AFILE")" 'mate 1$' "score mate 1"
checklike "the wrong axis cannot cut a file, so the mate stands" \
      "$(evalcp "setoption name LineBlocker value 24\nsetoption name LineBlockerAxis value 1\nposition fen $AFILE")" 'mate 1$' "score mate 1"

# REGRESSION (the bug that shipped in the first axis build): a blocker on the PINNER's own
# square must not cut that pinner's own ray. between_bb() includes the far endpoint, so
# update_slider_blockers counted a blocker on h5 as a second occupant of the a5-h5 ray,
# cancelled the pin, and generated b5b6 -- leaving the white king in check from that rook.
# Legal movegen producing an illegal move, which the assert sweep cannot see.
PINPOS='8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'
check "pin stands, blocker off" \
      "$(perft1 "position fen $PINPOS")" "14"
check "a blocker on the PINNER's own square does not cut its own ray (h5)" \
      "$(perft1 "setoption name LineBlocker value 39\nsetoption name LineBlockerAxis value 0\nposition fen $PINPOS")" "14"
check "a blocker ON the ray cuts the pin even though a piece stands there (b5)" \
      "$(perft1 "setoption name LineBlocker value 33\nsetoption name LineBlockerAxis value 0\nposition fen $PINPOS")" "15"
check "a blocker strictly between cuts the pin (c5)" \
      "$(perft1 "setoption name LineBlocker value 34\nsetoption name LineBlockerAxis value 0\nposition fen $PINPOS")" "15"
check "the wrong axis leaves the pin alone (c5 diagonal)" \
      "$(perft1 "setoption name LineBlocker value 34\nsetoption name LineBlockerAxis value 1\nposition fen $PINPOS")" "14"

# MaskPinner still works here too: Be5 pins g7 to Kh8, so the mate needs the pin.
PIN='3r3k/6p1/4Q3/4B3/1p3P2/4PKP1/3q4/8 w - - 18 52'
checklike "MaskPinner: mate with the pin intact" "$(evalcp "position fen $PIN")" 'mate 2$' "score mate 2"
checklike "MaskPinner: no mate once suspended" \
      "$(evalcp "setoption name MaskPinner value 36\nposition fen $PIN")" '^score cp' "a cp score"

if [ "$FAILED" -eq 0 ]; then echo; echo "all checks passed"; exit 0
else echo; echo "$FAILED check(s) failed"; exit 1; fi
