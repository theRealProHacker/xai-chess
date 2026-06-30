"""tempo operator ("nullzug") -- grade zugzwang / initiative by flipping side-to-move.

A POSITION-level intervention: the side to move passes (a null move). The board is
untouched and no engine patch is needed, so -- like the rule-level pin surface -- it is
confound-free. One signed number characterizes both reasons:

    eff_tempo = wp(pass) - wp(best move)        (side-to-move's POV)
       >> 0  : the side to move would rather pass  -> ZUGZWANG (move-obligation hurts)
       << 0  : having the move is a big asset       -> INITIATIVE / tempo is load-bearing
       ~  0  : tempo is not decisive here

Legality: you cannot pass out of check, so a position where the side to move is in
check has no null move (reported n/a).

Reuses pin_verify's scale helpers (win_prob saturates; eff_cp / eff_logit do not)."""
import argparse
import chess
import chess.engine
from pin_verify import win_prob, centipawns, log_odds


def pos_score(engine, board, color, limit):
    """Engine Score from `color`'s POV with the side to move playing best."""
    info = engine.analyse(board, limit)
    return info["score"].pov(color)


def null_score(engine, board, color, limit):
    """Engine Score from `color`'s POV after the side to move passes (null move).
    None if the side to move is in check (cannot pass out of check)."""
    if board.is_check():
        return None
    nb = board.copy(stack=False)
    nb.push(chess.Move.null())  # flips turn, clears en-passant, board otherwise untouched
    info = engine.analyse(nb, limit)
    return info["score"].pov(color)


# Golden cases: a clean zugzwang, a tactical control where the move is an asset, and a
# roughly-tempo-neutral opening position.
CASES = [
    ("zugzwang: Black to move loses the opposition (wKe5/Pe4 vs bKe7); White-to-move here is a draw",
     "8/4k3/8/4K3/4P3/8/8/8 b - - 0 1", "expect >0 (zugzwang)"),
    ("tactical: White to move wins the pinned e5 knight with d4",
     "4k3/8/5p2/4n3/8/8/3P4/4R1K1 w - - 0 1", "expect <0 (tempo is an asset)"),
    ("opening start position: tempo roughly neutral",
     chess.STARTING_FEN, "expect ~0"),
]


def main():
    ap = argparse.ArgumentParser(description="tempo (nullzug) operator: zugzwang / initiative via side-to-move flip")
    ap.add_argument("--engine", required=True, help="path to a UCI engine (stock Stockfish is fine -- no patch needed)")
    ap.add_argument("--nodes", type=int, default=400000, help="fixed nodes per search (determinism)")
    a = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(a.engine)
    engine.configure({"Threads": 1, "Hash": 64})  # single-thread fixed-nodes => deterministic
    limit = chess.engine.Limit(nodes=a.nodes)

    print(f"# tempo operator (nullzug) | engine={a.engine.split('/')[-1]} nodes={a.nodes}")
    print("# eff_tempo = pass - move  (side-to-move POV):  >0 zugzwang | <0 tempo-is-asset\n")
    try:
        for desc, fen, expect in CASES:
            board = chess.Board(fen)
            stm = board.turn
            sotm = "White" if stm == chess.WHITE else "Black"
            mv = pos_score(engine, board, stm, limit)
            nl = null_score(engine, board, stm, limit)
            print(desc)
            print(f"  FEN: {fen}")
            if nl is None:
                print("  pass: n/a (side to move is in check -- cannot null-move)\n")
                continue
            eff_wp = win_prob(nl) - win_prob(mv)
            eff_cp = centipawns(nl) - centipawns(mv)
            eff_lg = log_odds(nl) - log_odds(mv)
            print(f"  {sotm} to move:  move wp={win_prob(mv):.3f} cp={centipawns(mv):+d}"
                  f"   |   pass wp={win_prob(nl):.3f} cp={centipawns(nl):+d}")
            print(f"  eff_tempo:  wp={eff_wp:+.3f}  logit={eff_lg:+.2f}  cp={eff_cp:+d}    ({expect})\n")
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
