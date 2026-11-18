"""Why must preparation move B come before move A?

Search A now: its refutation line. Search B, the opponent's best reply R, then A:
replay the refutation there and report the first ply where it stops working.
"""
import sys
import chess, chess.engine

SF = "vendor/stockfish/stockfish-ubuntu-x86-64-avx2"
DEPTH = 20


def score(info, pov):
    return info["score"].pov(pov).score(mate_score=10000)


def search(eng, board, pov):
    info = eng.analyse(board, chess.engine.Limit(depth=DEPTH))
    return score(info, pov), info.get("pv", [])


def san_line(board, moves):
    return board.copy().variation_san(moves)


def explain(fen, a, b, eng):
    p = chess.Board(fen)
    me = p.turn
    A, B = p.parse_san(a), p.parse_san(b)

    now = p.copy(); now.push(A)
    e_now, refut = search(eng, now, me)

    prep = p.copy(); prep.push(B)
    _, pv = search(eng, prep, me)
    R = pv[0]
    prep.push(R)
    after = prep.copy()
    after.push(after.parse_san(a))
    e_prep, line_prep = search(eng, after, me)

    print(f"{a} now:            {e_now:+5d}  {san_line(now, refut[:8])}")
    print(f"{b} then {a}:  {e_prep:+5d}  {san_line(after, line_prep[:8])}")

    # Replay the refutation in the prepared position: first ply that is illegal or
    # whose value for me is >= 100 cp better than in the original line.
    x, y = now.copy(), after.copy()
    for k, m in enumerate(refut[:10]):
        if m not in y.legal_moves:
            print(f"break at ply {k+1}: {x.san(m)} is illegal after {b}")
            return
        x.push(m); y.push(m)
        ex, _ = search(eng, x, me); ey, _ = search(eng, y, me)
        if ey - ex >= 100:
            print(f"break at ply {k+1}: after {san_line(now, refut[:k+1])} "
                  f"{ex:+d} without {b}, {ey:+d} with it")
            return
    print("no break found")


if __name__ == "__main__":
    with chess.engine.SimpleEngine.popen_uci(SF) as eng:
        explain(sys.argv[1], sys.argv[2], sys.argv[3], eng)
