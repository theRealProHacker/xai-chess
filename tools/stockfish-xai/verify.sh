#!/usr/bin/env bash
# Verify both operators in rulelevel.patch:
#   MaskPinner  (A-F) -- absolute pins unregressed, relative pin suspendable, sound.
#   LineBlocker (G-L) -- the magic blocker obstructs rays, is unreachable, and is inert when off.
#
# Usage: ./verify.sh [PATH_TO_NATIVE_BINARY]
#   The binary must be Stockfish 18 built with this directory's rulelevel.patch (see build.sh on the
#   initial-research branch, pointed at that patch). This prints a report with the expected value on
#   each line; it does not assert, so read the output.
#
# Square indices are a1=0, b1=1 ... h8=63 (file + 8*rank); 64 = SQ_NONE = operator off.
#
# This runs against a RELEASE binary and checks semantics. It is not the soundness gate:
# sweep-blocker.sh sweeps all 64 squares under an assert-enabled build, and that is what
# caught update_piece_threats reading piece_on() of the blocker square -- a bug this file
# cannot see, because under NDEBUG it never crashes, it just evaluates wrong. Run both.
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

say "F. default bench unchanged (both operators off == stock)"
printf 'bench\nquit\n' | "$BIN" 2>&1 | grep -i 'Nodes searched' | tail -1

# --- LineBlocker (obstruct) ---------------------------------------------------
# perft with the blocker armed; $1 = fen, $2 = square index
lbperft(){ printf "setoption name LineBlocker value $2\nposition fen $1\ngo perft 1\nquit\n" | "$BIN" 2>/dev/null | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1; }
lbeval(){ ( printf "setoption name Clear Hash\nsetoption name LineBlocker value $2\nposition fen $1\ngo nodes 600000\n"; sleep 7 ) | "$BIN" 2>/dev/null | grep -oE 'score (cp|mate) -?[0-9]+' | tail -1; }

say "G. blocker obstructs a ray (Ra1 on the open a-file, blocker a4=24)"
R='4k3/8/8/8/8/8/8/R3K3 w - - 0 1'
echo "off=$(lbperft "$R" 64) on(a4)=$(lbperft "$R" 24)   (want 15 / 10: rook loses a4-a8)"

say "H. nothing may land on the blocker (no move to a4)"
echo "moves to a4: $(printf "setoption name LineBlocker value 24\nposition fen $R\ngo perft 1\nquit\n" | "$BIN" 2>/dev/null | grep -c 'a1a4')   (want 0)"

say "I. the blocker interposes a check (Black Ra1 checks Ke1; blocker c1=2)"
C='4k3/8/8/8/8/8/8/r3K3 w - - 0 1'
echo "off=$(lbperft "$C" 64) on(c1)=$(lbperft "$C" 2)   (want 3 in check / 5 check blocked)"

say "J. soundness: a blocker on an OCCUPIED square is inert (e1=4 holds the king)"
echo "off=$(lbperft "$R" 64) on(e1)=$(lbperft "$R" 4)   (want equal -- the OR into pieces() is a no-op)"

say "K. blocker stops castling and pawn pushes"
E='4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1'
echo "castling: off=$(lbperft "$E" 64) f1=$(lbperft "$E" 5) b1=$(lbperft "$E" 1)   (want 26 / 23 / 22)"
P='4k3/8/8/8/8/8/4P3/4K3 w - - 0 1'
echo "pushes:   off=$(lbperft "$P" 64) e3=$(lbperft "$P" 20) e4=$(lbperft "$P" 28)   (want 6 / 4 / 5)"

say "L. the operator's point: obstruct the file that carries the mate, with a placebo"
B='6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1'          # Ra8# -- the open a-file IS the mate
echo "no blocker:        $(lbeval "$B" 64)   (want mate 1)"
echo "on-file  a4 (24):  $(lbeval "$B" 24)   (want NO mate -- the file was load-bearing)"
echo "on-file  a6 (40):  $(lbeval "$B" 40)   (want NO mate)"
echo "placebo  b4 (25):  $(lbeval "$B" 25)   (want mate 1 -- off the file, nothing changes)"
echo "placebo  d4 (27):  $(lbeval "$B" 27)   (want mate 1)"

say "M. obstruct CUTS A PIN -- the blocker frees a piece, so it can ADD moves"
# Pb5 is pinned to Ka5 by Rh5 along the fifth rank, so b5-b6 is illegal. A blocker
# anywhere on c5-g5 cuts the pinning line and the push becomes legal. This is the
# "dual of MaskPinner" claim in OPERATORS.md made concrete: MaskPinner suspends the
# pin by rule, obstruct dissolves it by geometry -- and it is why no sweep over this
# operator may assume a blocker can only ever remove moves.
PIN='8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'
haspush(){ printf "setoption name LineBlocker value $1\nposition fen $PIN\ngo perft 1\nquit\n" | "$BIN" 2>/dev/null | grep -c '^b5b6:'; }
echo "off:        perft1=$(lbperft "$PIN" 64)  b5b6 legal=$(haspush 64)   (want 14 / 0 -- pinned)"
echo "on c5 (34): perft1=$(lbperft "$PIN" 34)  b5b6 legal=$(haspush 34)   (want 15 / 1 -- pin cut)"
echo "on g5 (38): perft1=$(lbperft "$PIN" 38)  b5b6 legal=$(haspush 38)   (want 15 / 1 -- pin cut)"
echo "on b6 (41): perft1=$(lbperft "$PIN" 41)  b5b6 legal=$(haspush 41)   (want 14 / 0 -- inert: b5b6 was already pinned, and Kb6 is covered by the c7 pawn, so there was nothing on b6 to take away)"
