#!/usr/bin/env python3
"""xai-chess factor verifier (factor-typed DSL).

Question it answers: given an explanation that says "move m is good because of
factor X on square s", is that factor actually causally responsible for m being
good, or is it a red herring the explainer name-dropped?

Method (one cited factor at a time):
  1. detect the cited factor on the board       -> python-chess (faithful by construction)
  2. neutralize it, material-neutral             -> a small, local board edit
  3. read the oracle engine's change for m        -> Stockfish, win-probability of playing m

  eff = outcome(s, m) - outcome(s', m)
      large positive  => removing the factor makes m worse => it was load-bearing (citation holds)
      ~ zero          => removing the factor changes nothing => the cited factor is a red herring

Factor DSL (plan section 3). Each factor type is a small spec:
  - to_nl(board, args) -> str                              human description
  - remove_board(board, move, args) -> (board|None, desc, extra)   the material-neutral
        BOARD-LEVEL counterfactual (the universal surface; every type has one)
  - rule_level(engine, board, move, args, limit) | None    the RULE-LEVEL surface, only
        for types whose factor is rule-enforced (pins are; most motifs are not)

Tiering (plan section 3): types split into rule-clean (pin: has both surfaces) and
board-only (hanging: board-level only). Two types are shipped:
  - pin     : absolute pin; board-level = relocate the pinned-to king off the ray;
              rule-level = suspend the pin in a movegen-patched Stockfish (MaskPinner).
  - hanging : an undefended enemy piece the move wins; board-level = relocate that
              piece one square to safety; no rule-level surface (nothing "hanging" is
              a rule, it is the absence of a defender).

Still deferred (plan section 3): detect(board)->[Factor] for placebo matching, and
validate_factor_type() running the edit-shock diagnostic before a new type may score.

Two outcomes are reported per surface:
  eff_wp     : delta of win-prob AFTER playing m (mover's POV). Primary, tests the claim.
  eff_margin : delta of policy margin wp(m) - wp(best legal alternative); grades the
               *choice* of m, not just the position's value.

Reuse, no mechanics reinvented: python-chess (pins, attacks, legality, cp->win-prob via
Score.wdl) + Stockfish (the oracle). Error contract: a cited factor that isn't real, or a
position with no legal material-neutral neutralization, is QUARANTINED with a reason --
never a silent float.
"""
import sys
import math
import json
import logging
import argparse
import contextlib
import chess
import chess.engine

MASK_OFF = 64  # MaskPinner == SQ_NONE on the patched engine -> stock behavior

# Under the rule-level intervention the engine legitimately reports a pin-violating
# ponder move (the suspended-pin reply); python-chess logs an error trying to validate it
# but recovers fine. We only read scores, so quiet that non-fatal noise.
logging.getLogger("chess.engine").setLevel(logging.CRITICAL)

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


MATE_CP = 10000  # mate scores clamped to this magnitude on the centipawn scale


def win_prob(score):
    """chess.engine.Score (mover's POV) -> win expectation in [0, 1]. Bounded, so it
    SATURATES near already-won/lost positions: a factor worth a whole piece can move it
    by <0.05 when the side is already winning."""
    return score.wdl(model="lichess").expectation()


def centipawns(score):
    """Score (mover's POV) -> centipawns, mate clamped to +-MATE_CP. The engine-native,
    non-saturating scale: a factor worth a knight shows ~+300 regardless of who's winning."""
    return score.score(mate_score=MATE_CP)


def log_odds(score):
    """logit(win_prob) -- the de-saturated probability scale. Expands the [0,1] tails that
    `win_prob` compresses, so a real effect near the ceiling stays visible, while staying a
    monotone function of the same WP the threshold is defined on."""
    p = min(max(win_prob(score), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def effects(score_present, score_removed):
    """The three eff scales for one operator: (eff_wp, eff_cp, eff_logit) = present - removed."""
    return (win_prob(score_present) - win_prob(score_removed),
            centipawns(score_present) - centipawns(score_removed),
            log_odds(score_present) - log_odds(score_removed))


@contextlib.contextmanager
def pin_suspended(engine, pinner_sq):
    """Rule-level intervention: tell the patched engine to suspend the pin cast by the
    slider on `pinner_sq`. Inside this block the pinned piece may move and that pinner can
    neither check nor capture the king along its ray -- the king is NOT moved, so there is
    no king-safety confound.

    The transposition table is cleared on entry and exit: it is keyed by Zobrist hash, which
    does NOT encode the mask, so entries are invalid across a mask change (the same position
    has different legal moves). Sharing them corrupts the eval and -- via the TT-move path --
    can feed the search an illegal king capture. Clearing keeps eff a clean function of
    (position, move, mask) and is a correctness requirement, not an optimization."""
    engine.configure({"Clear Hash": None})
    engine.configure({"MaskPinner": int(pinner_sq)})
    try:
        yield
    finally:
        engine.configure({"MaskPinner": MASK_OFF})
        engine.configure({"Clear Hash": None})


def score_after_move(engine, board, move, limit):
    """Engine Score (mover's POV) of the position after the mover plays `move`."""
    mover = board.turn
    nb = board.copy(stack=False)
    nb.push(move)
    info = engine.analyse(nb, limit)
    return info["score"].pov(mover)


def score_of_move(engine, board, move, limit):
    """Engine Score (mover's POV) of playing `move` from `board`, evaluated WITHOUT pushing
    it onto a python-chess board. Under the rule-level intervention the engine answers with
    pin-violating replies that python-chess would reject if it had to validate them; by
    keeping `move` (always legal on `board`) as the only root move and reading only the
    score, those replies stay safely inside the engine's PV. Used for the rule-level
    surface so it never hands python-chess an "illegal" move to parse."""
    mover = board.turn
    info = engine.analyse(board, limit, root_moves=[move], info=chess.engine.INFO_SCORE)
    return info["score"].pov(mover)


def policy_margin(engine, board, move, limit):
    """wp(move) - wp(best legal alternative), mover's POV. None if no alternative exists."""
    mover = board.turn
    info_m = engine.analyse(board, limit, root_moves=[move], info=chess.engine.INFO_SCORE)
    wp_m = win_prob(info_m["score"].pov(mover))
    alts = [mv for mv in board.legal_moves if mv != move]
    if not alts:
        return None
    info_alt = engine.analyse(board, limit, root_moves=alts, info=chess.engine.INFO_SCORE)
    wp_alt = win_prob(info_alt["score"].pov(mover))
    return wp_m - wp_alt


def material(board):
    return sum(PIECE_VALUE[p.piece_type] * (1 if p.color == chess.WHITE else -1)
               for p in board.piece_map().values())


def find_attacker(board, color, pinned_sq):
    """The enemy sliding piece doing the pinning (for reporting). Call only on a real pin."""
    for sq in board.pin(color, pinned_sq):
        p = board.piece_at(sq)
        if p and p.color != color and p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            return sq
    return None


# --------------------------------------------------------------------------- pin

def neutralize_pin(board, pinned_sq):
    """Remove an absolute pin by relocating the king it pins to, off the ray.
    Material and side-to-move are preserved. Returns (new_board, description) or (None, reason)."""
    piece = board.piece_at(pinned_sq)
    if piece is None:
        return None, f"no piece on {chess.square_name(pinned_sq)}"
    color = piece.color
    if not board.is_pinned(color, pinned_sq):
        return None, (f"{chess.square_name(pinned_sq)} is not absolutely pinned to its king "
                      f"-> there is no causal pin to remove")
    king_sq = board.king(color)
    for target in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        if board.piece_at(target) is not None:
            continue
        nb = board.copy(stack=False)
        nb.remove_piece_at(king_sq)
        nb.set_piece_at(target, chess.Piece(chess.KING, color))
        if not nb.is_valid():
            continue
        if nb.is_pinned(color, pinned_sq):
            continue  # king stayed on the ray; still pinned
        side = "white" if color == chess.WHITE else "black"
        return nb, f"{side} king {chess.square_name(king_sq)}->{chess.square_name(target)}"
    return None, "no legal material-neutral king relocation off the pin ray"


def remove_pin_board(board, move, args):
    """Factor-DSL board-level remover for `pin`. Adapts neutralize_pin to the (board, move,
    args)->(board|None, desc, extra) contract and reports the pinning slider."""
    pinned_sq = chess.parse_square(args["pinned"])
    nb, desc = neutralize_pin(board, pinned_sq)
    if nb is None:
        return None, desc, {}
    piece = board.piece_at(pinned_sq)
    attacker = find_attacker(board, piece.color, pinned_sq) if piece else None
    extra = {"attacker": chess.square_name(attacker) if attacker is not None else None,
             "relation": f"pinned by {chess.square_name(attacker)}" if attacker is not None else None}
    return nb, desc, extra


def pin_rule_level_scores(engine, board, move, args, limit):
    """Rule-level surface for `pin`: score m with the pin present (normal rules) and with the
    pinner's ray suspended (king stays put -> no confound). Returns (sc_norm, sc_rule, m_rule)
    or None when there is no slider pinner to suspend. The same root-moves protocol is used
    for both terms so the rule-level eff is internally consistent and never feeds python-chess
    a pin-violating move."""
    pinned_sq = chess.parse_square(args["pinned"])
    piece = board.piece_at(pinned_sq)
    if piece is None:
        return None
    attacker = find_attacker(board, piece.color, pinned_sq)
    if attacker is None:
        return None
    sc_norm = score_of_move(engine, board, move, limit)  # pin present
    with pin_suspended(engine, attacker):
        sc_rule = score_of_move(engine, board, move, limit)  # pin suspended
        m_rule = policy_margin(engine, board, move, limit)
    return sc_norm, sc_rule, m_rule


def pin_to_nl(board, args):
    return f"pin on {args.get('pinned', '?')}"


# ----------------------------------------------------------------------- hanging

def neutralize_hanging(board, move, args):
    """Board-level remover for `hanging`: an enemy piece that the side to move attacks and
    that its own side does not defend (a free piece the move wins). Remove the factor by
    relocating that piece one square to a safe neighbour -- material-neutral and side-to-move
    preserved, the minimal-disruption parallel to neutralize_pin's one-square king hop.

    Faithful by construction: if the cited square is empty, holds the mover's own piece, is
    not attacked by the mover, or is actually defended, there is no free piece to win and the
    factor is QUARANTINED. If no safe adjacent square exists, the position is quarantined too
    (counted against the retention rate). Returns (board|None, desc, extra)."""
    sq = chess.parse_square(args["square"])
    piece = board.piece_at(sq)
    if piece is None:
        return None, f"no piece on {chess.square_name(sq)}", {}
    mover = board.turn
    if piece.color == mover:
        return None, (f"{chess.square_name(sq)} holds the side-to-move's own piece, "
                      f"not an enemy hanging piece"), {}
    if not board.is_attacked_by(mover, sq):
        return None, (f"{chess.square_name(sq)} is not attacked by the side to move "
                      f"-> nothing is hanging to win"), {}
    if board.is_attacked_by(not mover, sq):
        return None, (f"{chess.square_name(sq)} is defended -> not hanging "
                      f"(no free material to win)"), {}

    for target in chess.SquareSet(chess.BB_KING_ATTACKS[sq]):  # the 8 neighbours
        if board.piece_at(target) is not None:
            continue
        if target in (move.from_square, move.to_square):
            continue  # don't relocate onto the move's own squares; keep m's geometry intact
        nb = board.copy(stack=False)
        nb.remove_piece_at(sq)
        nb.set_piece_at(target, piece)
        if not nb.is_valid():
            continue
        # the piece must no longer be hanging on its new square (safe = unattacked or defended)
        if nb.is_attacked_by(mover, target) and not nb.is_attacked_by(not mover, target):
            continue
        side = "white" if piece.color == chess.WHITE else "black"
        name = chess.piece_name(piece.piece_type)
        desc = (f"{side} {name} {chess.square_name(sq)}->{chess.square_name(target)} "
                f"(to safety)")
        extra = {"relation": "attacked by the mover, undefended"}
        return nb, desc, extra
    return None, ("no safe material-neutral relocation for the hanging piece "
                  "(all neighbours occupied or still hanging)"), {}


def hanging_to_nl(board, args):
    sq = args.get("square", "?")
    name = "piece"
    try:
        p = board.piece_at(chess.parse_square(sq))
        if p is not None:
            name = chess.piece_name(p.piece_type)
    except (ValueError, AttributeError):
        pass
    return f"hanging {name} on {sq}"


# --------------------------------------------------------------------- registry

# Each factor type: human description + the material-neutral board-level counterfactual,
# plus an optional rule-level surface (None = board-level only; the tier choice).
FACTOR_TYPES = {
    "pin": {
        "to_nl": pin_to_nl,
        "remove_board": remove_pin_board,
        "rule_level": pin_rule_level_scores,
    },
    "hanging": {
        "to_nl": hanging_to_nl,
        "remove_board": neutralize_hanging,
        "rule_level": None,
    },
}


def verify_record(engine, rec, limit, threshold, supports_rule_level=False):
    out = {"source": rec.get("source"), "fen": rec["fen"],
           "move": rec["justified_move_uci"], "factors": []}
    board = chess.Board(rec["fen"])
    move = chess.Move.from_uci(rec["justified_move_uci"])
    if move not in board.legal_moves:
        out["status"] = "quarantined"
        out["reason"] = "justified move is not legal in this position"
        return out

    for f in rec.get("cited_factors", []):
        ftype = f["type"]
        args = f.get("args", {})
        fo = {"type": ftype, "args": args}
        spec = FACTOR_TYPES.get(ftype)
        if spec is None:
            fo["status"] = "quarantined"
            fo["reason"] = (f"unsupported factor type '{ftype}' "
                            f"(supported: {', '.join(FACTOR_TYPES)})")
            out["factors"].append(fo)
            continue

        sp_board, desc, extra = spec["remove_board"](board, move, args)
        if sp_board is None:
            fo["status"] = "quarantined"
            fo["reason"] = desc
            out["factors"].append(fo)
            continue
        if move not in sp_board.legal_moves:
            fo["status"] = "quarantined"
            fo["reason"] = (f"the counterfactual ({desc}) makes the justified move illegal "
                            f"-> not a clean material-neutral comparison")
            out["factors"].append(fo)
            continue

        fo.update(extra)
        fo["neutralization"] = desc
        fo["nl"] = spec["to_nl"](board, args)

        # Board-level operator (universal surface): the move's value in the original
        # position vs the factor-removed counterfactual.
        sc_s = score_after_move(engine, board, move, limit)
        m_s = policy_margin(engine, board, move, limit)
        sc_sp = score_after_move(engine, sp_board, move, limit)
        m_sp = policy_margin(engine, sp_board, move, limit)
        eff_wp, eff_cp, eff_logit = effects(sc_s, sc_sp)
        eff_margin = None if (m_s is None or m_sp is None) else (m_s - m_sp)

        fo.update(
            status="scored",
            material_delta=material(sp_board) - material(board),
            eff_wp=round(eff_wp, 4),
            eff_cp=round(eff_cp),
            eff_logit=round(eff_logit, 3),
            eff_margin=None if eff_margin is None else round(eff_margin, 4),
            threshold=threshold,
            load_bearing=bool(eff_wp >= threshold),
        )

        # Rule-level operator: only the types that have one, only on the patched engine.
        # Same board, re-scored under suspended-factor rules; no piece moves -> no
        # board-edit confound. Reported alongside board-level so their agreement can be
        # read off (the plan's gate for the "confound-free" claim).
        rule_level = spec["rule_level"]
        if rule_level is not None and supports_rule_level:
            rl = rule_level(engine, board, move, args, limit)
            if rl is None:
                fo["rule_level_note"] = "no rule-level target to suspend"
            else:
                sc_norm, sc_rule, m_rule = rl
                eff_wp_rule, eff_cp_rule, eff_logit_rule = effects(sc_norm, sc_rule)
                eff_margin_rule = None if (m_s is None or m_rule is None) else (m_s - m_rule)
                lb_rule = bool(eff_wp_rule >= threshold)
                fo.update(
                    eff_wp_rule=round(eff_wp_rule, 4),
                    eff_cp_rule=round(eff_cp_rule),
                    eff_logit_rule=round(eff_logit_rule, 3),
                    eff_margin_rule=None if eff_margin_rule is None else round(eff_margin_rule, 4),
                    load_bearing_rule=lb_rule,
                    surfaces_agree=bool(lb_rule == fo["load_bearing"]),
                )
        elif rule_level is None and supports_rule_level:
            fo["rule_level_note"] = f"{ftype}: board-level only (no rule-level surface)"

        out["factors"].append(fo)

    out["status"] = "scored"
    return out


def main():
    ap = argparse.ArgumentParser(description="Verify whether cited factors are causally load-bearing.")
    ap.add_argument("records", help="path to a JSONL file of explanation records")
    ap.add_argument("--engine", required=True, help="path to the UCI engine (Stockfish)")
    ap.add_argument("--nodes", type=int, default=400000, help="fixed nodes per search (determinism)")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="eff_wp >= threshold => load-bearing (placeholder; pre-register via a power calc)")
    args = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 64})  # single-thread fixed-nodes => deterministic
    limit = chess.engine.Limit(nodes=args.nodes)

    # The rule-level operator needs the movegen-patched engine (exposes MaskPinner).
    # With a stock engine we silently fall back to the board-level operator only.
    supports_rule_level = "MaskPinner" in engine.options

    with open(args.records) as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    surfaces = "board+rule" if supports_rule_level else "board-only (stock engine)"
    print(f"# xai-chess factor verifier | types={','.join(FACTOR_TYPES)} "
          f"engine={args.engine.split('/')[-1]} nodes={args.nodes} "
          f"threshold(eff_wp)>={args.threshold} surfaces={surfaces}\n")
    for rec in records:
        r = verify_record(engine, rec, limit, args.threshold, supports_rule_level)
        if r["status"] == "quarantined":
            print(f"QUARANTINED [{r['source']}] {r['move']}: {r['reason']}")
            continue
        for fo in r["factors"]:
            key = fo["args"].get("pinned") or fo["args"].get("square") or "?"
            tag = f"{fo['type']}@{key}"
            if fo["status"] != "scored":
                print(f"QUARANTINED [{r['source']}] {r['move']} {tag}: {fo['reason']}")
                continue
            verdict = "LOAD-BEARING" if fo["load_bearing"] else "not load-bearing (red herring)"
            rel = f" ({fo['relation']})" if fo.get("relation") else ""
            print(f"[{r['source']}]")
            print(f"  move {r['move']} cites {fo['nl']}{rel}")
            print(f"  board-level ({fo['neutralization']}, material delta {fo['material_delta']:+d}):")
            print(f"    eff_wp={fo['eff_wp']:+.3f}  eff_logit={fo['eff_logit']:+.2f}  "
                  f"eff_cp={fo['eff_cp']:+d}  (margin "
                  f"{'n/a' if fo['eff_margin'] is None else format(fo['eff_margin'], '+.3f')})"
                  f"   => {verdict}")
            if "eff_wp_rule" in fo:
                v_rule = "LOAD-BEARING" if fo["load_bearing_rule"] else "not load-bearing (red herring)"
                print(f"  rule-level (factor suspended in the rules; board untouched):")
                print(f"    eff_wp={fo['eff_wp_rule']:+.3f}  eff_logit={fo['eff_logit_rule']:+.2f}  "
                      f"eff_cp={fo['eff_cp_rule']:+d}  (margin "
                      f"{'n/a' if fo['eff_margin_rule'] is None else format(fo['eff_margin_rule'], '+.3f')})"
                      f"   => {v_rule}")
                agree = "AGREE" if fo["surfaces_agree"] else "DISAGREE"
                gap = abs(fo["eff_wp"] - fo["eff_wp_rule"])
                print(f"  surfaces {agree}  (board vs rule |Δeff_wp|={gap:.3f})")
            elif "rule_level_note" in fo:
                print(f"  rule-level: n/a ({fo['rule_level_note']})")
            print(f"  (threshold eff_wp >= {fo['threshold']})\n")

    engine.quit()


if __name__ == "__main__":
    main()
