#!/usr/bin/env python3
"""Tag WHEN the pin becomes load-bearing, for every puzzle in the probe dataset.

The probe dataset (build_dataset.py) keeps puzzles carrying exactly one absolute
enemy pin, the same one, present in every position. This script walks each puzzle's
solver moves and asks at each one: does the solver's move USE that pin?

Load-bearing test (`defence_sole`): the solver lands on a square the pinned piece
attacks but may not legally contest -- the pin is holding the guard off -- and no
other enemy piece legally guards that square. That is the pin's paralysis doing the
work, and it is the cheapest static stand-in for the rule-level counterfactual.

Two conditions from lichess_ingest.exploited_pin are deliberately NOT used:
  - "captures the pinned piece": structurally impossible here. The dataset requires
    the pin to survive every move, so the pinned piece is never taken. (This is also
    why the cook.py mainline[1::2] off-by-one does not bias this set -- the cases it
    drops were already excluded.)
  - "the only pin on the board": true by construction for every record, so it tags
    everything at ply 0 and measures nothing.
  - "lands attacking the pinned piece" is measured but excluded: 37.5% Lichess-pin
    rate against a 28.5% base is near chance. `defence_sole` runs at 89.8%.

distance = how many solver moves past P0 (the position the probe sees) the first
move that passes the test sits. It counts the solver's own moves, not plies: the
solver's k-th move is 2k plies out, and only even ply offsets are reachable, so plies
would leave every odd value empty. Distance 0 means the very first solver move uses
the pin.

Output: relevance.tsv, keyed by puzzle_id (join to meta.tsv on that, not on order).

  zstdcat lichess_db_puzzle.csv.zst | python pin_relevance.py > data/relevance.tsv
"""
import sys, csv, chess
from multiprocessing import Pool
csv.field_size_limit(10**7)

META = {}


def flags(b, move, psq, ray):
    """Which relevance conditions the solver's `move` satisfies against the pin on psq."""
    mover, opp = b.turn, not b.turn
    out = set()

    voided = b.attacks(psq) & ~ray            # guarded on paper, forbidden by the pin
    if move.to_square in voided:
        out.add("defence")
        others = b.attackers(opp, move.to_square) & ~chess.BB_SQUARES[psq]
        if not any(not b.is_pinned(opp, sq) or move.to_square in b.pin(opp, sq)
                   for sq in chess.SquareSet(others)):
            out.add("defence_sole")

    nb = b.copy(stack=False)
    nb.push(move)
    if psq in nb.attacks(move.to_square):
        out.add("attacks")
        trapped = True
        for m in nb.legal_moves:
            if m.from_square != psq:
                continue
            t = nb.copy(stack=False)
            t.push(m)
            if not t.is_attacked_by(mover, m.to_square):
                trapped = False
                break
        if trapped:
            out.add("attacks_trapped")
    return out


def run(row):
    pid = row[0]
    if pid not in META:
        return None
    moves = row[2].split()
    try:
        b = chess.Board(row[1])
        b.push(chess.Move.from_uci(moves[0]))
    except Exception:
        return None
    opp = not b.turn
    pinned = [sq for sq in chess.SquareSet(b.occupied_co[opp]) if b.is_pinned(opp, sq)]
    if len(pinned) != 1:
        return None                      # should not happen; build_dataset guarantees it
    psq = pinned[0]

    per_ply = []
    for k in range(len(moves) // 2):
        i = 2 * k + 1
        if i >= len(moves):
            break
        m = chess.Move.from_uci(moves[i])
        if m not in b.legal_moves:
            break
        per_ply.append(flags(b, m, psq, b.pin(opp, psq)))
        b.push(m)
        if i + 1 < len(moves):
            b.push(chess.Move.from_uci(moves[i + 1]))

    first = next((k for k, f in enumerate(per_ply) if "defence_sole" in f), -1)
    first_loose = next((k for k, f in enumerate(per_ply) if "defence" in f), -1)
    trace = ";".join("".join(sorted(c for c, n in
                                    (("C", "capture"), ("D", "defence"),
                                     ("S", "defence_sole"), ("A", "attacks"),
                                     ("T", "attacks_trapped")) if n in f)) or "-"
                     for f in per_ply)
    return (pid, first, first_loose, len(per_ply), trace)


def main():
    with open("data/meta.tsv") as fh:
        r = csv.reader(fh, delimiter="\t")
        next(r)
        for row in r:
            META[row[1]] = int(row[2])

    w = sys.stdout
    w.write("puzzle_id\tlabel\tdistance\tdistance_loose\tn_solver_moves\tagree\ttrace\n")
    rdr = csv.reader(sys.stdin)
    next(rdr, None)
    n = agree = 0
    with Pool(8) as p:
        for res in p.imap(run, (x for x in rdr if len(x) > 8), chunksize=2000):
            if res is None:
                continue
            pid, first, loose, nk, trace = res
            lab = META[pid]
            # agreement: tagger fires and Lichess says pin, or neither ever does
            ok = int((first >= 0) == bool(lab))
            agree += ok
            n += 1
            w.write(f"{pid}\t{lab}\t{first}\t{loose}\t{nk}\t{ok}\t{trace}\n")
    print(f"# rows={n}  agree={agree} ({agree / n * 100:.2f}%)", file=sys.stderr)


main()
