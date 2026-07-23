#!/usr/bin/env bash
# Verify the relative-pin extension: absolute pins unregressed, relative pin now suspendable, sound.
#
# Usage: ./verify.sh [PATH_TO_NATIVE_BINARY]
#   The binary must be Stockfish 18 built with this directory's rulelevel.patch (see build.sh on the
#   initial-research branch, pointed at that patch). This prints a report with the expected value on
#   each line; it does not assert, so read the output.
BIN="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/vendor/stockfish/stockfish-xai-rulelevel}"
[ -x "$BIN" ] || { echo "!! no engine at $BIN — pass one as \$1"; exit 1; }
say(){ printf '\n=== %s ===\n' "$1"; }

perft1(){ printf "$1\ngo perft 1\nquit\n" | "$BIN" 2>/dev/null | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1; }
evalcp(){ # $1 = setoption+position lines, prints "cp N" or "mate N"
  # No `quit`: it races the search and the engine exits before printing a score. Give it a window.
  ( printf "$1\ngo nodes 400000\n"; sleep 6 ) | "$BIN" 2>/dev/null | grep -oE 'score (cp|mate) -?[0-9]+' | tail -1; }

say "A. absolute-pin perft (build.sh golden): off=4 on=10"
FEN='4k3/4n3/8/8/8/8/8/4R1K1 b - - 0 1'
echo "off: $(perft1 "position fen $FEN")   on(e1=4): $(perft1 "setoption name MaskPinner value 4
position fen $FEN")   (want 4 / 10)"

say "B. demo absolute pin (Be5 pins g7 to Kh8): off=mate, on(e5=36)=no mate"
D='3r3k/6p1/4Q3/4B3/1p3P2/4PKP1/3q4/8 w - - 18 52'
echo "off:      $(evalcp "setoption name MaskPinner value 64
position fen $D")"
echo "on(e5=36): $(evalcp "setoption name MaskPinner value 36
position fen $D")   (want mate / ~0)"

say "C. RELATIVE pin (user FEN, Bg5 pins Nf6 to Qd8): off vs on(g5=38) should now DIFFER"
U='rnbqkb1r/pppp1pp1/4pn2/6Bp/3PP3/8/PPP2PPP/RN1QKBNR w KQkq - 0 4'
echo "off:       $(evalcp "setoption name MaskPinner value 64
position fen $U")"
echo "on(g5=38):  $(evalcp "setoption name MaskPinner value 38
position fen $U")   (want LOWER than off if pin load-bearing)"

say "D. soundness: non-pinner squares must stay INERT (perft1 unchanged)"
# knight checker on the mask square (not a slider): must be inert
echo "non-slider check: off=$(perft1 "position fen R3k3/8/8/8/8/3n4/8/4K3 w - - 0 1") mask19=$(perft1 "setoption name MaskPinner value 19
position fen R3k3/8/8/8/8/3n4/8/4K3 w - - 0 1")  (want equal)"
# direct slider check (pins nothing): must be inert
echo "direct-slider:    off=$(perft1 "position fen 4k3/8/8/4r3/8/8/R7/4K3 w - - 0 1") mask=$(perft1 "setoption name MaskPinner value 32
position fen 4k3/8/8/4r3/8/8/R7/4K3 w - - 0 1")  (want equal)"

say "E. soundness: masked relative search does not crash (bestmove returned)"
bm=$( ( printf 'setoption name Hash value 64\nsetoption name MaskPinner value 38\nposition fen %s\ngo nodes 500000\n' "$U"; sleep 6 ) | "$BIN" 2>&1 | grep -i '^bestmove' | awk '{print $2}' )
echo "masked relative search bestmove: ${bm:-<none/CRASH>}"

say "F. default bench unchanged (mask off == stock)"
printf 'bench\nquit\n' | "$BIN" 2>&1 | grep -i 'Nodes searched' | tail -1
