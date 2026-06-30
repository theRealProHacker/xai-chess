#!/usr/bin/env python3
"""Build a forced-mate-VERIFIED labeled set -- replacing Lichess theme-tag labels with
exact game-theoretic ground truth.

For a Lichess mate puzzle with a pin, the move m forces mate. The cited pin is load-bearing
iff, with that pin suspended (rule-level, in the patched engine's variant search), m NO
LONGER forces mate -- i.e. the freed piece defends. "Forced mate exists" is provable (a
reported mate is a proof), so the label is verified ground truth, NOT a proxy tag. It is
also BINARY (mate / no-mate) while the eff metric is CONTINUOUS win-prob, so grading eff
against it is a real test: a saturated eff can disagree with the exact label.

  faithful (pin necessary)  : m forces mate; suspending the pin BREAKS the mate.
  red herring (pin not nec.): m forces mate; the mate SURVIVES suspension.

Independence note: the label is computed at high node budget as a binary forced-mate fact;
the metric (corpus_eval) is a continuous win-prob delta at a fixed low budget. They share the
rule-level counterfactual, so rule-level eff is expected to track the label well -- that part
is a heuristic-recovers-exact (efficiency) check; the board-level eff is the cross-operator
test.

Usage:
  python3 verified_labels.py puzzles.csv --engine <patched-sf> --limit 40 > verified.jsonl
"""
import csv
import sys
import json
import argparse
import logging
import chess
import chess.engine

logging.getLogger("chess.engine").setLevel(logging.CRITICAL)
from pin_verify import score_of_move, pin_suspended, find_attacker
from lichess_ingest import COLS, position_and_move, enemy_absolute_pins, exploited_pin


def forces_mate(score):
    """True if `score` (mover's POV) is a forced mate FOR the mover."""
    return score.is_mate() and score.mate() > 0


def main():
    ap = argparse.ArgumentParser(description="Lichess mate puzzles -> forced-mate-verified pin labels.")
    ap.add_argument("csv")
    ap.add_argument("--engine", required=True, help="the movegen-patched Stockfish (MaskPinner)")
    ap.add_argument("--nodes", type=int, default=1000000, help="nodes for the mate verification")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 64})
    if "MaskPinner" not in engine.options:
        sys.exit("error: --engine must be the movegen-patched build (no MaskPinner option)")
    limit = chess.engine.Limit(nodes=args.nodes)

    kept = scanned = faithful = herring = 0
    with open(args.csv, newline="") as fh:
        for row in csv.DictReader(fh, fieldnames=COLS):
            if row["PuzzleId"] == "PuzzleId":
                continue
            if "mate" not in row["Themes"]:           # forced-mate puzzles only
                continue
            scanned += 1

            board, move = position_and_move(row["FEN"], row["Moves"])
            if board is None or move not in board.legal_moves:
                continue
            pins = enemy_absolute_pins(board)
            if not pins:
                continue

            # Precondition: m must force mate with the DEFENDER still to move (mate#>=2; a
            # mate#1 is delivered on the spot, so suspending a pin is vacuous). Keep short
            # mates so the forced-mate verification is reliable at this node budget.
            sc_norm = score_of_move(engine, board, move, limit)
            if not forces_mate(sc_norm) or not (2 <= sc_norm.mate() <= 4):
                continue

            # Verified label: the cited pin is load-bearing iff suspending it BREAKS the
            # forced mate. Test each enemy pin directly (don't guess which one matters);
            # cite the first whose suspension breaks the mate. None breaks -> red herring.
            cite_sq, broke = pins[0], False
            for psq in pins:
                pinner = find_attacker(board, board.piece_at(psq).color, psq)
                if pinner is None:
                    continue
                with pin_suspended(engine, pinner):
                    sc_susp = score_of_move(engine, board, move, limit)
                if not forces_mate(sc_susp):       # mate broke -> this pin is load-bearing
                    cite_sq, broke = psq, True
                    break
            label = "faithful" if broke else "unfaithful"
            faithful += broke
            herring += not broke

            verdict = "breaks" if broke else "survives"
            print(json.dumps({
                "fen": board.fen(),
                "justified_move_uci": move.uci(),
                "cited_factors": [{"type": "pin", "args": {"pinned": chess.square_name(cite_sq)}}],
                "source": f"{label}: verified mate#{sc_norm.mate()} {verdict} pin-suspension "
                          f"| lichess {row['PuzzleId']} rating {row['Rating']}",
            }))
            kept += 1
            if kept >= args.limit:
                break
    engine.quit()
    print(f"# verified {kept} (faithful={faithful} red-herring={herring}) / scanned {scanned} mate puzzles",
          file=sys.stderr)


if __name__ == "__main__":
    main()
