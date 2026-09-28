#!/usr/bin/env python3
"""Ten labels for the same 151,614 positions, aligned to meta.tsv.

Three are cook.py's own -- the program that produced the Lichess `pin` theme, run
per position by ../cook_pin.py (verified identical to the vendored source on all
6,100,960 puzzles in the export). Three are pin_relevance.py's.

  cook.pin       the tag itself: attack or escape, over the boards after each solver move
  cook.attack    a pinned enemy piece may not take a pov piece worth more, or hanging
  cook.escape    an attacker on the pin ray hits a pinned piece that may not step aside
  rel.defence        the solver's move lands on a square the pinned piece may not contest
  rel.defence_sole   ...and nothing else guards that square
  rel.attacks        the solver's move attacks the pinned piece

cook's are properties of a POSITION and never mention the solver's move;
pin_relevance.py's are properties of a MOVE. That is the comparison the sweep is for.

Also stores the metadata baseline (rating, solver moves, in check, simple eval) --
the floor each ladder has to clear, and the control for the puzzle-length confound:
"fires on at least one solver move" is monotone in the number of solver moves.

  zstdcat lichess_db_puzzle.csv.zst | python auto_labels.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".pylibs"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from multiprocessing import Pool

import cook_pin

csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["cook.pin", "cook.attack", "cook.escape",
         "cook.attack.greater", "cook.attack.hanging",
         "cook.escape.greater", "cook.escape.hanging",
         "rel.defence", "rel.defence_sole", "rel.attacks"]
KEEP = {}


def run(row):
    """(puzzle_id, cook.attack, cook.escape) for one CSV row we keep."""
    if row[0] not in KEEP:
        return None
    try:
        a, e = cook_pin.pin_branches(row[1], row[2])
        lf = cook_pin.pin_leaf_flags(row[1], row[2])
    except Exception:
        return None
    return (row[0], int(a), int(e), int("pin" in row[7].split()),
            [int(k in lf) for k in cook_pin.LEAVES])


def main():
    order, meta = [], {}
    with open(os.path.join(HERE, "data", "meta.tsv")) as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for i, x in enumerate(r):
            KEEP[x["puzzle_id"]] = i
            order.append(x["puzzle_id"])
            meta[x["puzzle_id"]] = x
    n = len(order)

    # pin_relevance.py's per-move flags, "fires on at least one solver move"
    rel = {}
    with open(os.path.join(HERE, "data", "relevance.tsv")) as fh:
        for x in csv.DictReader(fh, delimiter="\t"):
            t = x["trace"].split(";")
            rel[x["puzzle_id"]] = [int(any(c in s for s in t)) for c in "DSA"]
    assert set(rel) == set(KEEP), "relevance.tsv and meta.tsv describe different puzzles"

    y = np.zeros((n, len(NAMES)), np.int8)
    export = np.zeros(n, np.int8)
    seen = 0
    rdr = csv.reader(sys.stdin)
    next(rdr, None)
    with Pool(8) as p:
        for res in p.imap(run, (x for x in rdr if len(x) > 8), chunksize=2000):
            if res is None:
                continue
            pid, a, e, tag, leaves = res
            i = KEEP[pid]
            y[i] = [int(a or e), a, e] + leaves + rel[pid]
            export[i] = tag
            seen += 1
    assert seen == n, f"{seen} of {n} puzzles found in the CSV"

    # The label the published sweep used was the export's tag; cook.pin is that tag
    # recomputed. They differ only where cook.py moved on since the export was cut.
    drift = int((y[:, 0] != export).sum())

    # Nesting the rules guarantee, asserted rather than trusted.
    J = {nm: k for k, nm in enumerate(NAMES)}
    assert not (y[:, J["rel.defence_sole"]] & ~y[:, J["rel.defence"]]).any(), \
        "defence_sole must imply defence"
    for br in ("attack", "escape"):
        leaf = y[:, J[f"cook.{br}.greater"]] | y[:, J[f"cook.{br}.hanging"]]
        assert (leaf == y[:, J[f"cook.{br}"]]).all(), f"cook.{br} leaves do not sum to the branch"
    assert not (y[:, J["cook.attack"]] & ~y[:, J["cook.pin"]]).any()

    feat = np.array([[float(meta[p]["rating"]), float(meta[p]["solver_moves"]),
                      float(meta[p]["in_check"]), float(meta[p]["simple_eval"])]
                     for p in order], np.float32)

    out = os.path.join(HERE, "data", "labels.npz")
    np.savez(out, y=y, names=np.array(NAMES), meta_feat=feat,
             meta_names=np.array(["rating", "solver_moves", "in_check", "simple_eval"]),
             export=export, puzzle_id=np.array(order),
             source_mtime=os.path.getmtime(os.path.join(HERE, "data", "relevance.tsv")),
             puzzler_commit=cook_pin.PUZZLER_COMMIT)

    print(f"n = {n:,}   -> {out}")
    print(f"{'label':20}{'positives':>11}{'rate':>8}")
    for j, nm in enumerate(NAMES):
        print(f"{nm:20}{y[:, j].sum():>11,}{y[:, j].mean() * 100:>7.2f}%")
    print(f"\ncook.pin vs the export's tag: {n - drift:,}/{n:,} "
          f"({(n - drift) / n * 100:.2f}%), {drift} rows of cook.py drift")


main()
