#!/usr/bin/env bash
# Verify both operators in rulelevel.patch:
#   MaskPinner  (A-F) -- absolute pins unregressed, relative pin suspendable, sound.
#   LineBlocker (G-P) -- the magic blocker cuts the marked line for the matching slider class,
#                        leaves every other piece alone, and is inert when off.
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
# The blocker obstructs a LINE, not a square: axis 0 cuts rook and queen rays on ranks and
# files, axis 1 cuts bishop and queen rays on diagonals. Pawns, knights, kings and castling
# are untouched, and any piece -- including the sliders it cuts -- may still move onto the
# square. A piece standing there does not lift the obstruction, so a blocker on an OCCUPIED
# square is not inert either. Sections N-P are the selectivity the operator exists for.
#
# $1 = fen, $2 = square index, $3 = axis (0 = rank/file, 1 = diagonal; default 0)
lbperft(){ printf "setoption name LineBlocker value $2\nsetoption name LineBlockerAxis value ${3:-0}\nposition fen $1\ngo perft 1\nquit\n" | "$BIN" 2>/dev/null; }
lbn(){ lbperft "$1" "$2" "$3" | grep -i 'Nodes searched' | grep -oE '[0-9]+' | tail -1; }
lbhas(){ lbperft "$1" "$2" "$3" | grep -c "^$4:"; }
lbeval(){ ( printf "setoption name Clear Hash\nsetoption name LineBlocker value $2\nsetoption name LineBlockerAxis value ${3:-0}\nposition fen $1\ngo nodes 600000\n"; sleep 7 ) | "$BIN" 2>/dev/null | grep -oE 'score (cp|mate) -?[0-9]+' | tail -1; }

say "G. blocker cuts a ray (Ra1 on the open a-file, blocker a4=24)"
R='4k3/8/8/8/8/8/8/R3K3 w - - 0 1'
echo "off=$(lbn "$R" 64) orth=$(lbn "$R" 24 0) diag=$(lbn "$R" 24 1)   (want 15 / 11 / 15 -- rook loses a5-a8; a diagonal blocker does not touch a rook)"

say "H. the square stays open -- the rook may stop ON the blocker, never past it"
echo "a1a4: $(lbhas "$R" 24 0 a1a4)  a1a5: $(lbhas "$R" 24 0 a1a5)   (want 1 / 0)"

say "I. the blocker interposes a check (Black Ra1 checks Ke1; blocker c1=2)"
C='4k3/8/8/8/8/8/8/r3K3 w - - 0 1'
echo "off=$(lbn "$C" 64) orth(c1)=$(lbn "$C" 2 0) diag(c1)=$(lbn "$C" 2 1)   (want 3 in check / 5 check blocked / 3 -- wrong axis, still in check)"

say "J. a blocker on an OCCUPIED square still cuts the ray (it is not inert any more)"
# A ray FROM a square is never cut by a blocker sitting on it -- only rays THROUGH it are.
echo "on its own square a1(0): $(lbn "$R" 0 0)   (want 15 -- a blocker cuts rays THROUGH a square, not FROM it)"
echo "on the king's square e1(4): $(lbn "$R" 4 0)   (want 15 -- the king already stopped that ray)"

say "K. castling and pawn pushes are NOT the blocker's business any more"
E='4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1'
echo "castling: off=$(lbn "$E" 64) f1=$(lbn "$E" 5 0) b1=$(lbn "$E" 1 0)   (want 26 / 26 / 24 -- both castles legal in all three; b1 only costs Ra1 its c1/d1 rank moves)"
echo "O-O / O-O-O under b1: $(lbhas "$E" 1 0 e1g1) / $(lbhas "$E" 1 0 e1c1)   (want 1 / 1)"
P='4k3/8/8/8/8/8/4P3/4K3 w - - 0 1'
echo "pushes:   off=$(lbn "$P" 64) e3=$(lbn "$P" 20 0) e4=$(lbn "$P" 28 0)   (want 6 / 6 / 6 -- a pawn is not a rook)"
N='4k3/8/8/8/8/8/8/N3K3 w - - 0 1'
echo "knight:   off=$(lbn "$N" 64) b3=$(lbn "$N" 17 0)   (want equal -- and Nb3 itself stays legal: $(lbhas "$N" 17 0 a1b3))"

say "L. the operator's point: obstruct the file that carries the mate, with a placebo"
B='6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1'          # Ra8# -- the open a-file IS the mate
echo "no blocker:        $(lbeval "$B" 64)   (want mate 1)"
echo "on-file  a4 (24):  $(lbeval "$B" 24 0)   (want NO mate -- the file was load-bearing)"
echo "on-file  a6 (40):  $(lbeval "$B" 40 0)   (want NO mate)"
echo "wrong axis a4 (24, diag): $(lbeval "$B" 24 1)   (want mate 1 -- a diagonal blocker cannot cut a file)"
echo "placebo  b4 (25):  $(lbeval "$B" 25 0)   (want mate 1 -- off the file, nothing changes)"
echo "placebo  d4 (27):  $(lbeval "$B" 27 0)   (want mate 1)"

say "M. obstruct CUTS A PIN -- the blocker frees a piece, so it can ADD moves"
# Pb5 is pinned to Ka5 by Rh5 along the fifth rank, so b5-b6 is illegal. An ORTHOGONAL blocker
# anywhere on c5-g5 cuts the pinning line and the push becomes legal. This is the "dual of
# MaskPinner" claim in OPERATORS.md made concrete: MaskPinner suspends the pin by rule, obstruct
# dissolves it by geometry -- and it is why no sweep over this operator may assume a blocker can
# only ever remove moves. A diagonal blocker on the same square does nothing: the pin is on a rank.
PIN='8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'
echo "off:             perft1=$(lbn "$PIN" 64)  b5b6 legal=$(lbhas "$PIN" 64 0 b5b6)   (want 14 / 0 -- pinned)"
echo "orth c5 (34):    perft1=$(lbn "$PIN" 34 0)  b5b6 legal=$(lbhas "$PIN" 34 0 b5b6)   (want 15 / 1 -- pin cut)"
echo "orth g5 (38):    perft1=$(lbn "$PIN" 38 0)  b5b6 legal=$(lbhas "$PIN" 38 0 b5b6)   (want 15 / 1 -- pin cut)"
echo "diag c5 (34):    perft1=$(lbn "$PIN" 34 1)  b5b6 legal=$(lbhas "$PIN" 34 1 b5b6)   (want 14 / 0 -- wrong axis, pin stands)"
# REGRESSION: a blocker on the PINNER's own square must not cut its own ray. between_bb()
# includes the far endpoint, so an earlier update_slider_blockers counted the blocker on h5 as
# a second occupant of the a5-h5 ray, cancelled the pin, and generated b5b6 -- which leaves the
# white king on a5 in check from that very rook. Legal movegen producing an illegal move; the
# assert sweep cannot see it, because nothing crashes.
echo "orth h5 (39):    perft1=$(lbn "$PIN" 39 0)  b5b6 legal=$(lbhas "$PIN" 39 0 b5b6)   (want 14 / 0 -- the pinner sits on h5; a ray is never cut at its own end)"
echo "orth b5 (33):    perft1=$(lbn "$PIN" 33 0)  b5b6 legal=$(lbhas "$PIN" 33 0 b5b6)   (want 15 / 1 -- the blocker IS on the ray, so the pin is cut even though a piece stands there)"

say "N. SELECTIVITY: a queen keeps the rays the blocker is not aimed at"
Q='4k3/8/8/8/8/8/8/Q3K3 w - - 0 1'
echo "off=$(lbn "$Q" 64) orth a4=$(lbn "$Q" 24 0) diag d4=$(lbn "$Q" 27 1)   (want 22 / 18 / 18)"
echo "under orth a4 -- Qa5 (file, cut): $(lbhas "$Q" 24 0 a1a5)  Qd4 (diagonal, intact): $(lbhas "$Q" 24 0 a1d4)   (want 0 / 1)"
echo "under diag d4 -- Qe5 (diagonal, cut): $(lbhas "$Q" 27 1 a1e5)  Qa5 (file, intact): $(lbhas "$Q" 27 1 a1a5)   (want 0 / 1)"

say "O. SELECTIVITY: a bishop ignores an orthogonal blocker, a rook ignores a diagonal one"
BI='4k3/8/8/8/8/8/8/B3K3 w - - 0 1'
echo "bishop: off=$(lbn "$BI" 64) orth c3=$(lbn "$BI" 18 0) diag c3=$(lbn "$BI" 18 1)   (want 12 / 12 / 7)"
echo "rook:   off=$(lbn "$R" 64) diag a4=$(lbn "$R" 24 1) orth a4=$(lbn "$R" 24 0)   (want 15 / 15 / 11)"

say "P. the false positive this axis split removes: a pawn's file"
# A pawn is not a rook. Under the old blocker -- one OR into pieces(), which denied the square
# to everything -- an orthogonal blocker in front of a pawn stopped the push, so ANY file a pawn
# marched down read as load-bearing. It must now be inert.
PF='4k3/8/8/8/8/8/P7/4K3 w - - 0 1'
echo "off=$(lbn "$PF" 64) orth a3=$(lbn "$PF" 16 0) orth a4=$(lbn "$PF" 24 0)   (want 7 / 7 / 7)"
echo "a2a3: $(lbhas "$PF" 16 0 a2a3)  a2a4: $(lbhas "$PF" 16 0 a2a4)   (want 1 / 1 -- the blocker is not in the pawn's way)"
