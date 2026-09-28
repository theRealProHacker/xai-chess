#!/usr/bin/env python3
"""Reproduce Lichess's own `pin` tag with vendored cook.py, on our 151,614 puzzles.

The probe label is `"pin" in Themes` from the puzzle export. cook.py is the program
that put it there (ornicar/lichess-puzzler, tagger/cook.py). This runs its two pin
predicates on the same puzzles and reports how often the export agrees.

  zstdcat ~/Downloads/lichess_db_puzzle.csv.zst | python check_cook.py
"""
import sys, csv, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".pylibs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor",
                                "lichess-puzzler", "tagger"))
import chess
from chess import Board
from chess.pgn import Game
from multiprocessing import Pool
import cook
from model import Puzzle

csv.field_size_limit(10 ** 7)
KEEP = set()


def run(row):
    pid = row[0]
    if pid not in KEEP:
        return None
    try:
        node = Game.from_board(Board(row[1]))
        for uci in row[2].split():
            node = node.add_main_variation(chess.Move.from_uci(uci))
        pz = Puzzle(pid, node.game(), 0)
        a = cook.pin_prevents_attack(pz)
        e = cook.pin_prevents_escape(pz)
    except Exception:
        return None
    return pid, int("pin" in row[7].split()), int(a), int(e)


def main():
    with open(os.path.join(os.path.dirname(__file__), "data", "meta.tsv")) as fh:
        r = csv.reader(fh, delimiter="\t"); next(r)
        for row in r:
            KEEP.add(row[1])

    n = 0
    cells = {}          # (export, cook) -> count
    only_a = only_e = 0
    rdr = csv.reader(sys.stdin); next(rdr, None)
    with Pool(8) as p:
        for res in p.imap(run, (x for x in rdr if len(x) > 8), chunksize=2000):
            if res is None:
                continue
            _, exp, a, e = res
            ck = int(a or e)
            cells[(exp, ck)] = cells.get((exp, ck), 0) + 1
            only_a += a and not e
            only_e += e and not a
            n += 1

    agree = cells.get((0, 0), 0) + cells.get((1, 1), 0)
    print(f"rows={n}")
    print(f"export pin+ = {cells.get((1,0),0)+cells.get((1,1),0)}")
    print(f"cook   pin+ = {cells.get((0,1),0)+cells.get((1,1),0)}"
          f"   (attack-only {only_a}, escape-only {only_e})")
    print(f"\n{'':14}{'cook -':>10}{'cook +':>10}")
    for exp in (1, 0):
        print(f"export {'+' if exp else '-':<7}"
              f"{cells.get((exp,0),0):>10}{cells.get((exp,1),0):>10}")
    print(f"\nagreement = {agree}/{n} ({agree/n*100:.2f}%)")


main()
