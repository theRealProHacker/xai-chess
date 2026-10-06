"""Targeted test of Capablanca's isolated d5 (Capablanca - Teichmann, Berlin 1913, after 13...Nxc5).
Add one Black and one White pawn, so material stays level. Repair boards put the Black pawn next to d5
(e6, c6, c7); control boards put it away from it (g6, a5, h5); the White pawn goes to the same neutral squares
in both. --mode remove instead takes d5 off against taking another Black pawn off (a6, b6, h7), with the
same White pawn removed in both (a2, b2, h2). --mode white adds a White d-pawn (d2, d3) against a White
flank pawn (a3, h3, g3), with the same Black flank pawn (g6, a5, h5) in both. Each board: Leela raw net, Leela with search, Stockfish, and the
families that changed from the base. Q is White's view (W - L).

  python lime/lime_008_test_d5.py [--mode add|remove|white] [--nodes 800] [--depth 18]
"""
import argparse, importlib.util, itertools, sys
from pathlib import Path
import numpy as np
s = importlib.util.spec_from_file_location("l", Path(__file__).with_name("lime_001.py"))
l = importlib.util.module_from_spec(s); argv, sys.argv = sys.argv, ["x"]; s.loader.exec_module(l); sys.argv = argv
import chess, chess.engine

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["add", "remove", "white"], default="add")
ap.add_argument("--nodes", type=int, default=800)
ap.add_argument("--depth", type=int, default=18)
a = ap.parse_args()

FEN = "2rq1rk1/1b2bppp/pp3n2/2np2B1/B7/2N1PN2/PP2QPPP/2R2RK1 w - - 0 14"
if a.mode == "add":
    REPAIR, CONTROL, WHITE = ["e6", "c6", "c7"], ["g6", "a5", "h5"], ["a3", "h3", "g3"]
elif a.mode == "remove":
    REPAIR, CONTROL, WHITE = ["d5"], ["a6", "b6", "h7"], ["a2", "b2", "h2"]
else:                                   # "black" squares are the paired ones; repair/control is the White pawn
    REPAIR, CONTROL, WHITE = ["d2", "d3"], ["a3", "h3", "g3"], ["g6", "a5", "h5"]
LC0 = [str(l.ROOT / "vendor/lc0/build/release/lc0"), f"--weights={l.ROOT / 'vendor/lc0-nets/t1-256x10-distilled-swa-2432500.pb.gz'}"]
SF = str(l.ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2")

def board(bsq, wsq):
    b = chess.Board(FEN)
    if a.mode == "remove":
        for sq in (bsq, wsq):
            assert b.piece_type_at(chess.parse_square(sq)) == chess.PAWN, sq
            b.remove_piece_at(chess.parse_square(sq))
        assert b.is_valid() and not b.is_check(), b.fen()
        return b
    b.set_piece_at(chess.parse_square(bsq), chess.Piece(chess.PAWN, chess.BLACK))
    b.set_piece_at(chess.parse_square(wsq), chess.Piece(chess.PAWN, chess.WHITE))
    for sq in (bsq, wsq):
        p = chess.parse_square(sq)
        assert not l.tactical(b, p, b.color_at(p)), (bsq, wsq, sq)
    assert b.is_valid() and not b.is_check(), b.fen()
    return b

boards = [("base", None, None, chess.Board(FEN))] + [
    ("repair" if bsq in REPAIR else "control", bsq, wsq, board(wsq, bsq) if a.mode == "white" else board(bsq, wsq))
    for bsq, wsq in itertools.product(REPAIR + CONTROL, WHITE)]
fens = [b.fen() for *_, b in boards]

# d5 weak per our features? and which families changed from the base
rows = l.features(fens)
d5 = [any(f["on"] and f["id"].startswith("pawns.weak.") and f["side"] == "them" and "d5" in f["squares"] for f in r)
      for r in rows]
def fams(r):
    on = {}
    for f in r:
        on.setdefault((l.family_of(f["id"]), f["side"] if f["side"] else ""), set())
        if f["on"]:
            on[(l.family_of(f["id"]), f["side"] if f["side"] else "")].add(f["id"])
    return on
F0 = fams(rows[0])
changed = [sorted(f"{k[0]}{'.' + k[1] if k[1] else ''}" for k, v in fams(r).items() if F0.get(k) != v) for r in rows]
raw = l.leela_q(fens)

def search(cmd, limit, opts):
    out = []
    with chess.engine.SimpleEngine.popen_uci(cmd) as e:
        e.configure(opts)
        for f in fens:
            info = e.analyse(chess.Board(f), limit, game=object())          # fresh game: no hash carried over
            w = info["wdl"].white()
            out.append(((w.wins - w.losses) / w.total(), info["score"].white().score(mate_score=10000)))
    return np.array(out)

lc0 = search(LC0, chess.engine.Limit(nodes=a.nodes), {"UCI_ShowWDL": True})
sf = search(SF, chess.engine.Limit(depth=a.depth), {"UCI_ShowWDL": True, "Threads": 4})

print(f"{'board':8} {'black':5} {'white':5} {'d5 weak':7} {'raw Q':>7} {'lc0 Q':>7} {'lc0 cp':>7} {'SF Q':>7} {'SF cp':>6}")
for (kind, bsq, wsq, _), w, r, (lq, lcp), (sq, scp) in zip(boards, d5, raw, lc0, sf):
    print(f"{kind:8} {bsq or '-':5} {wsq or '-':5} {str(w):7} {r:+7.2f} {lq:+7.2f} {lcp:+7.0f} {sq:+7.2f} {scp:+6.0f}")
print("\nfamilies changed from the base:")
for (kind, bsq, wsq, _), c in zip(boards[1:], changed[1:]):
    print(f"  {kind:8} {bsq:3} {wsq:3}  {', '.join(c)}")

kinds = np.array([k for k, *_ in boards])
print("\nrepair - control, paired by White pawn square (Q for White; negative = d5 change helps Black)")
for name, v in (("raw net", raw), ("lc0 search", lc0[:, 0]), ("Stockfish Q", sf[:, 0]), ("Stockfish cp", sf[:, 1])):
    diffs = [v[(kinds == "repair") & np.array([w == wsq for *_, w, _ in boards])].mean()
             - v[(kinds == "control") & np.array([w == wsq for *_, w, _ in boards])].mean() for wsq in WHITE]
    print(f"  {name:13} mean {np.mean(diffs):+.3f}   per White square {dict(zip(WHITE, np.round(diffs, 3)))}")
