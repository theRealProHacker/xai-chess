#!/usr/bin/env python3
"""Convert Lichess puzzles (the scientific-standard chess corpus, CC0) into xai-chess
explanation records.

Why this corpus: ~4M natural positions from real games, theme-tagged (`pin`, `fork`, ...)
and Elo-rated. It gives the meta-eval BOTH arms for free and on natural positions --
fixing BonaFide's named #1 limitation (sparse/imbalanced/constructed ground truth):

  faithful     : `pin`-themed puzzles where the solution move exploits a pre-existing
                 absolute enemy pin  -> the pin IS the driver (ground-truth load-bearing).
  red herring  : non-`pin` puzzles that nonetheless contain an absolute enemy pin the
                 solution ignores     -> the pin is present but NOT the driver.

Lichess convention: the FEN is one ply early; the opponent plays Moves[0] (the setup),
then the SOLVER plays Moves[1] -- that is the justified move, made on (FEN after Moves[0]).

Our pin-verifier tests a narrower thing than Lichess's broad `pin` tag (which also covers
*creating* a pin): we keep only puzzles where the solver's move provably EXPLOITS an
already-absolute enemy pin. That selectivity is fine -- the corpus is huge.

Usage:
  python3 lichess_ingest.py puzzles.csv --mode faithful    --limit 40 > faithful.jsonl
  python3 lichess_ingest.py puzzles.csv --mode red-herring --limit 40 > placebo.jsonl
Theme filters (e.g. the non-saturated band) via --require / --exclude on Lichess themes:
  ... --mode faithful --require advantage --exclude crushing,mate
"""
import csv
import sys
import json
import argparse
import chess

# CSV columns (Lichess puzzle export header).
COLS = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
        "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags"]


def position_and_move(fen, moves_uci):
    """Apply the setup move (Moves[0]); return (board with solver to move, the justified move)."""
    board = chess.Board(fen)
    moves = moves_uci.split()
    if len(moves) < 2:
        return None, None
    board.push(chess.Move.from_uci(moves[0]))     # opponent's setup move
    return board, chess.Move.from_uci(moves[1])   # the solver's justified move


def enemy_absolute_pins(board):
    """Squares of the side-not-to-move's pieces that are absolutely pinned to their king."""
    enemy = not board.turn
    return [sq for sq in chess.SquareSet(board.occupied_co[enemy]) if board.is_pinned(enemy, sq)]


def exploited_pin(board, move):
    """The enemy pinned square the solver's `move` exploits, or None if none clearly does.
    Exploited == the move captures the pinned piece, or lands attacking it (it can't flee),
    or it is the only enemy pin on the board."""
    pins = enemy_absolute_pins(board)
    if not pins:
        return None
    if move.to_square in pins:                       # captures the pinned piece outright
        return move.to_square
    nb = board.copy(stack=False)
    nb.push(move)
    for sq in pins:                                  # lands attacking a piece that can't flee
        if move.to_square in nb.attackers(board.turn, sq):
            return sq
    if len(pins) == 1:                               # unambiguous single pin
        return pins[0]
    return None


def themes_ok(themes, require, exclude):
    tset = set(themes.split())
    if any(r and r not in tset for r in require):
        return False
    if any(e in tset for e in exclude):
        return False
    return True


def record(row, mode, require, exclude):
    """Lichess CSV row -> one explanation record, or None if it doesn't cleanly fit `mode`."""
    themes = row["Themes"]
    is_pin = "pin" in themes.split()
    if mode == "faithful" and not is_pin:
        return None
    if mode == "red-herring" and is_pin:
        return None
    if not themes_ok(themes, require, exclude):
        return None

    board, move = position_and_move(row["FEN"], row["Moves"])
    if board is None or move not in board.legal_moves:
        return None

    pin_sq = exploited_pin(board, move)
    if mode == "faithful":
        if pin_sq is None:               # the pin the tag refers to isn't an exploited absolute pin
            return None
        label = "faithful"
    else:  # red-herring: a pin must be PRESENT but NOT exploited by the move
        pins = enemy_absolute_pins(board)
        if not pins or exploited_pin(board, move) is not None:
            return None
        pin_sq = pins[0]
        label = "unfaithful"

    return {
        "fen": board.fen(),
        "justified_move_uci": move.uci(),
        "cited_factors": [{"type": "pin", "args": {"pinned": chess.square_name(pin_sq)}}],
        "source": f"{label}: lichess {row['PuzzleId']} rating {row['Rating']} [{themes}]",
    }


def main():
    ap = argparse.ArgumentParser(description="Lichess puzzles -> xai-chess pin records.")
    ap.add_argument("csv", help="Lichess puzzle CSV (decompressed)")
    ap.add_argument("--mode", choices=["faithful", "red-herring"], default="faithful")
    ap.add_argument("--limit", type=int, default=40, help="max records to emit")
    ap.add_argument("--require", default="", help="comma-separated themes that MUST be present")
    ap.add_argument("--exclude", default="", help="comma-separated themes that must be ABSENT")
    args = ap.parse_args()

    require = [t for t in args.require.split(",") if t]
    exclude = [t for t in args.exclude.split(",") if t]

    kept = scanned = 0
    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh, fieldnames=COLS)
        for row in reader:
            if row["PuzzleId"] == "PuzzleId":
                continue  # header
            scanned += 1
            rec = record(row, args.mode, require, exclude)
            if rec is None:
                continue
            print(json.dumps(rec))
            kept += 1
            if kept >= args.limit:
                break
    print(f"# kept {kept} / scanned {scanned} ({args.mode})", file=sys.stderr)


if __name__ == "__main__":
    main()
