#!/usr/bin/env python3
"""Build the probe dataset.

Selection: exactly one absolute pin, on an opponent piece, present in EVERY position
of the puzzle (P0 and after every move). Same filter that produced 151,614 puzzles.

Sample  : P0 only -- the board after Moves[0], which is the position the solver sees.
Label   : the published Lichess `pin` theme, verbatim. No exclusions, no relabelling.

Writes fens.txt (one FEN per line, dump input) and meta.tsv (aligned by line number).
"""
import sys, csv, chess
from multiprocessing import Pool
csv.field_size_limit(10**7)

SF_PAWN = 208
SF_NPM = {chess.KNIGHT: 781, chess.BISHOP: 825, chess.ROOK: 1276, chess.QUEEN: 2538}


def pins(b, opp):
    return {(sq, b.pin(opp, sq).mask) for sq in chess.SquareSet(b.occupied_co[opp])
            if b.pin(opp, sq).mask != chess.BB_ALL}


def simple_eval(b):
    stm = b.turn
    v = SF_PAWN * (len(b.pieces(chess.PAWN, stm)) - len(b.pieces(chess.PAWN, not stm)))
    for pt, val in SF_NPM.items():
        v += val * (len(b.pieces(pt, stm)) - len(b.pieces(pt, not stm)))
    return v


def run(row):
    try:
        moves = row[2].split()
        if len(moves) < 2:
            return None
        b = chess.Board(row[1])
        b.push(chess.Move.from_uci(moves[0]))
    except Exception:
        return None
    opp = not b.turn
    p0_fen = b.fen()
    per_pos = [pins(b, opp)]
    try:
        for uci in moves[1:]:
            b.push(chess.Move.from_uci(uci))
            per_pos.append(pins(b, opp))
    except Exception:
        return None
    union = set().union(*per_pos)
    if len(union) != 1 or min(len(s) for s in per_pos) != 1:
        return None

    b0 = chess.Board(p0_fen)
    pinned_sq, ray = next(iter(union))
    se = simple_eval(b0)
    try:
        rating, nplays = int(row[3]), int(row[6])
    except Exception:
        rating, nplays = 0, 0
    return (p0_fen, row[0], int("pin" in row[7].split()), rating, nplays,
            (len(moves) - 1 + 1) // 2,                       # solver moves
            (chess.popcount(b0.occupied) - 1) // 4,          # NNUE bucket
            int(abs(se) > 962), se,
            chess.square_name(pinned_sq),
            (b0.piece_at(pinned_sq).symbol() if b0.piece_at(pinned_sq) else "?"),
            int(bool(b0.checkers())))


def main():
    rdr = csv.reader(sys.stdin); next(rdr, None)
    fens = open("fens.txt", "w")
    meta = open("meta.tsv", "w")
    meta.write("idx\tpuzzle_id\tlabel\trating\tnbplays\tsolver_moves\tbucket\t"
               "small_net\tsimple_eval\tpinned_sq\tpinned_piece\tin_check\n")
    n = pos = 0
    with Pool(8) as p:
        for r in p.imap(run, (x for x in rdr if len(x) > 8), chunksize=2000):
            if r is None:
                continue
            fens.write(r[0] + "\n")
            meta.write(f"{n}\t" + "\t".join(str(x) for x in r[1:]) + "\n")
            pos += r[2]; n += 1
    fens.close(); meta.close()
    print(f"positions={n}  positive={pos} ({pos/n*100:.2f}%)  negative={n-pos}")


main()
