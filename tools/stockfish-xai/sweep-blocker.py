#!/usr/bin/env python3
"""Soundness sweep for the LineBlocker (obstruct) operator: every square, both axes.

    ./sweep-blocker.py PATH_TO_DEBUG_BINARY [--depth N]

Build the binary with ASSERTIONS ON -- that is the point of this script:

    make -j8 build ARCH=x86-64-avx2 debug=yes optimize=no

A release build does not catch the failure this was written for. The bug it found
(update_piece_threats reading piece_on() of the blocker square) only trips an assert;
under NDEBUG the engine runs on and feeds NO_PIECE into the NNUE threat features, so it
looks fine and evaluates wrong. verify.sh cannot see that class of bug.

Invariants, per (position, square, axis):

  1. no assert / abort / segfault, and perft reports a number
  2. cross-class inertness: an orthogonal blocker changes nothing in a position whose
     only sliders are bishops, and a diagonal blocker changes nothing in one whose only
     sliders are rooks                                       [CLASS_FENS only]

There is deliberately no "a blocker can only remove moves" invariant. It sounds obvious
and it is false: a blocker cuts lines for BOTH sides, so it can hand the side to move
extra options. Two ways, both found by earlier versions of this sweep reporting them as
failures:

  * it interposes a CHECK, so the side to move stops generating evasions and generates
    everything instead (verify.sh section I asserts this on purpose), and
  * it cuts a PIN, freeing a piece that could not legally move before (section M).

Move counts are therefore printed as deltas for the reader, never failed on.

Nor is there a "nothing may land on the blocker" invariant any more. The blocker used to
own its square; it does not. It obstructs a LINE -- rook and queen rays on a rank or file
(axis 0), bishop and queen rays on a diagonal (axis 1) -- and leaves the square itself
open to every piece, including the sliders it cuts. A piece standing on the square does
not lift the obstruction: the ray is cut there whether or not anything occupies it, which
is also why a blocker on an OCCUPIED square is no longer inert.

The real gate is the assert build: perft walks the whole tree and Stockfish's own internal
assertions fire on any inconsistency the blocker introduces.
"""
import argparse
import re
import subprocess
import sys

FENS = [
    "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2N1P/PP1PBPP1/R1BQ1RK1 w - - 0 9",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
]

# Single-class positions for the selectivity invariant: the only sliders are rooks in the
# first and bishops in the second, so the OTHER axis must be a no-op on every square.
# (axis that must be inert, fen)
CLASS_FENS = [
    (1, "4k3/8/r7/8/8/7R/8/4K3 w - - 0 1"),   # rooks only  -> a diagonal blocker is inert
    (0, "4k3/8/b7/8/8/7B/8/4K3 w - - 0 1"),   # bishops only -> an orthogonal blocker is inert
]
NAME = [f"{'abcdefgh'[s % 8]}{s // 8 + 1}" for s in range(64)]
NODES = re.compile(r"Nodes searched\s*:\s*(\d+)", re.I)
BAD = re.compile(r"assert|abort|segmentation|core dumped", re.I)


def run(binary, cmds, timeout=120):
    p = subprocess.run([binary], input="\n".join(cmds) + "\nquit\n",
                       capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


def perft(binary, fen, sq, depth, axis=0):
    """(nodes|None, root move strings, raw output) with LineBlocker on `sq` (64 = off)."""
    out = run(binary, [f"setoption name LineBlocker value {sq}",
                       f"setoption name LineBlockerAxis value {axis}",
                       f"position fen {fen}", f"go perft {depth}"])
    m = NODES.search(out)
    moves = re.findall(r"^([a-h][1-8][a-h][1-8][qrbn]?)\s*:", out, re.M)
    return (int(m.group(1)) if m else None), moves, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()

    runs = fails = 0
    gained = []
    for fen in FENS:
        base_deep, _, _ = perft(a.binary, fen, 64, a.depth)
        base_one, _, _ = perft(a.binary, fen, 64, 1)
        if base_deep is None or base_one is None:
            print(f"FATAL: no baseline for {fen}", flush=True)
            return 1
        for axis in (0, 1):
            tag = "orth" if axis == 0 else "diag"
            for sq in range(64):
                runs += 1
                n, _, out = perft(a.binary, fen, sq, a.depth, axis)
                if BAD.search(out):
                    line = next((l for l in out.splitlines() if BAD.search(l)), "")
                    print(f"CRASH     {tag} {NAME[sq]:3} {fen}\n            {line.strip()}", flush=True)
                    fails += 1
                    continue
                if n is None:
                    print(f"NO NUMBER {tag} {NAME[sq]:3} {fen}", flush=True)
                    fails += 1
                    continue
                # Root-move gains are reader information, not an invariant (see the docstring),
                # so only pay for the extra probe on squares that changed anything at all.
                if n != base_deep:
                    n1, _, _ = perft(a.binary, fen, sq, 1, axis)
                    if n1 is not None and n1 > base_one:
                        gained.append((f"{tag} {NAME[sq]}", base_one, n1, fen))
            print(f"... {tag} pass done for {fen}", flush=True)

    # Selectivity: the wrong axis must not touch a position that has no slider of that class.
    for inert_axis, fen in CLASS_FENS:
        base_deep, _, _ = perft(a.binary, fen, 64, a.depth)
        if base_deep is None:
            print(f"FATAL: no baseline for {fen}")
            return 1
        tag = "orth" if inert_axis == 0 else "diag"
        for sq in range(64):
            runs += 1
            n, _, _ = perft(a.binary, fen, sq, a.depth, inert_axis)
            if n != base_deep:
                print(f"NOT INERT {tag} {NAME[sq]:3} {fen}  perft{a.depth} {base_deep} -> {n}", flush=True)
                fails += 1
    if gained:
        print(f"\n{len(gained)} square(s) where the blocker ADDED root moves "
              f"(expected -- it cuts enemy lines too):")
        for name, b, n, fen in gained:
            print(f"  {name:3} perft1 {b} -> {n}   {fen}")
    print(f"\nsweep depth {a.depth}, both axes: {runs} runs, {fails} failures")
    print("CLEAN" if not fails else "FAILURES ABOVE")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
