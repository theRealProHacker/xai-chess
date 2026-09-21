#!/usr/bin/env python3
"""Soundness sweep for the LineBlocker (obstruct) operator: every square, several positions.

    ./sweep-blocker.py PATH_TO_DEBUG_BINARY [--depth N]

Build the binary with ASSERTIONS ON -- that is the point of this script:

    make -j8 build ARCH=x86-64-avx2 debug=yes optimize=no

A release build does not catch the failure this was written for. The bug it found
(update_piece_threats reading piece_on() of the blocker square) only trips an assert;
under NDEBUG the engine runs on and feeds NO_PIECE into the NNUE threat features, so it
looks fine and evaluates wrong. verify.sh cannot see that class of bug.

Invariants, per (position, square):

  1. no assert / abort / segfault, and perft reports a number      [every square]
  2. no root move lands on the blocker square             [empty squares only]

There is deliberately no "a blocker can only remove moves" invariant. It sounds obvious
and it is false: a blocker cuts lines for BOTH sides, so it can hand the side to move
extra options. Two ways, both found by earlier versions of this sweep reporting them as
failures:

  * it interposes a CHECK, so the side to move stops generating evasions and generates
    everything instead (verify.sh section I asserts this on purpose), and
  * it cuts a PIN, freeing a piece that could not legally move before (section M).

Move counts are therefore printed as deltas for the reader, never failed on. The real
gate is the assert build: perft walks the whole tree and Stockfish's own internal
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
NAME = [f"{'abcdefgh'[s % 8]}{s // 8 + 1}" for s in range(64)]
NODES = re.compile(r"Nodes searched\s*:\s*(\d+)", re.I)
BAD = re.compile(r"assert|abort|segmentation|core dumped", re.I)


def occupied_squares(fen):
    """Square indices (a1=0..h8=63) holding a piece, read straight off the FEN board."""
    out, ranks = set(), fen.split()[0].split("/")
    for r, row in enumerate(ranks):            # ranks[0] is rank 8
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                out.add((7 - r) * 8 + f)
                f += 1
    return out


def run(binary, cmds, timeout=120):
    p = subprocess.run([binary], input="\n".join(cmds) + "\nquit\n",
                       capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


def perft(binary, fen, sq, depth):
    """(nodes|None, root move strings, raw output) with LineBlocker on `sq` (64 = off)."""
    out = run(binary, [f"setoption name LineBlocker value {sq}",
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
        occ = occupied_squares(fen)
        base_deep, _, _ = perft(a.binary, fen, 64, a.depth)
        base_one, _, _ = perft(a.binary, fen, 64, 1)
        if base_deep is None or base_one is None:
            print(f"FATAL: no baseline for {fen}")
            return 1
        for sq in range(64):
            runs += 1
            n, moves, out = perft(a.binary, fen, sq, a.depth)
            if BAD.search(out):
                line = next((l for l in out.splitlines() if BAD.search(l)), "")
                print(f"CRASH     {NAME[sq]:3} {fen}\n            {line.strip()}")
                fails += 1
                continue
            if n is None:
                print(f"NO NUMBER {NAME[sq]:3} {fen}")
                fails += 1
                continue
            if sq in occ:
                continue                       # blocker is inert on an occupied square
            landed = [m for m in moves if m[2:4] == NAME[sq]]
            if landed:
                print(f"LANDS ON  {NAME[sq]:3} {fen}  ({' '.join(landed)})")
                fails += 1
            n1, _, _ = perft(a.binary, fen, sq, 1)
            if n1 is not None and n1 > base_one:   # legitimate: see the docstring
                gained.append((NAME[sq], base_one, n1, fen))
    if gained:
        print(f"\n{len(gained)} square(s) where the blocker ADDED root moves "
              f"(expected -- it cuts enemy lines too):")
        for name, b, n, fen in gained:
            print(f"  {name:3} perft1 {b} -> {n}   {fen}")
    print(f"\nsweep depth {a.depth}: {runs} runs, {fails} failures")
    print("CLEAN" if not fails else "FAILURES ABOVE")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
