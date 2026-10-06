"""2x2 on Capablanca - Teichmann's isolated d5: White gets a d-pawn (d2, d3) or a pawn elsewhere (c2), crossed
with Black getting a pawn that protects d5 (e6, c6) or one elsewhere (--black-no, default g6 a5 h5). One pawn
added per side on every board, so material stays level. Each board: Leela raw net, Leela with search, Stockfish.
Q is White's view (W - L).

--all averages over every legal square instead: White d-pawn on any d-file square, White elsewhere on any other
empty square, Black elsewhere on any empty square but e6/c6; pairs that break the sampler's filters are skipped
(added pawn one or two pushes from promotion, hitting a piece, or giving check).

  python lime/lime_009_test_d5_2x2.py [--black-no g6,a5,h5] [--all] [--no-search] [--nodes 800] [--depth 18]
"""
import argparse, importlib.util, itertools, sys
from pathlib import Path
import numpy as np
s = importlib.util.spec_from_file_location("l", Path(__file__).with_name("lime_001.py"))
l = importlib.util.module_from_spec(s); argv, sys.argv = sys.argv, ["x"]; s.loader.exec_module(l); sys.argv = argv
import chess, chess.engine

ap = argparse.ArgumentParser()
ap.add_argument("--black-no", default="g6,a5,h5")
ap.add_argument("--all", action="store_true")
ap.add_argument("--no-search", action="store_true")
ap.add_argument("--nodes", type=int, default=800)
ap.add_argument("--depth", type=int, default=18)
a = ap.parse_args()

FEN = "2rq1rk1/1b2bppp/pp3n2/2np2B1/B7/2N1PN2/PP2QPPP/2R2RK1 w - - 0 14"
W = {"d-pawn": ["d2", "d3"], "elsewhere": ["c2"]}
B = {"protects d5": ["e6", "c6"], "elsewhere": a.black_no.split(",")}
if a.all:
    empty = lambda ranks: [chess.square_name(q) for q in chess.SQUARES
                           if chess.square_rank(q) in ranks and chess.Board(FEN).piece_at(q) is None]
    W = {"d-pawn": [q for q in empty(range(1, 5)) if q[0] == "d"], "elsewhere": [q for q in empty(range(1, 5)) if q[0] != "d"]}
    B = {"protects d5": ["e6", "c6"], "elsewhere": [q for q in empty(range(3, 7)) if q not in ("e6", "c6")]}
LC0 = [str(l.ROOT / "vendor/lc0/build/release/lc0"), f"--weights={l.ROOT / 'vendor/lc0-nets/t1-256x10-distilled-swa-2432500.pb.gz'}"]
SF = str(l.ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2")

def board(wsq, bsq):
    """The base plus both pawns, or None if the pair breaks a filter (asserted when the squares are hand-picked)."""
    b = chess.Board(FEN)
    for sq, c in ((wsq, chess.WHITE), (bsq, chess.BLACK)):
        assert b.piece_at(chess.parse_square(sq)) is None, sq
        b.set_piece_at(chess.parse_square(sq), chess.Piece(chess.PAWN, c))
    deep = lambda q: chess.square_rank(q) == (5 if b.color_at(q) == chess.WHITE else 2)
    bad = [sq for sq in (wsq, bsq) if l.tactical(b, chess.parse_square(sq), b.color_at(chess.parse_square(sq)))
           or deep(chess.parse_square(sq))]
    if bad or not b.is_valid() or b.is_check():
        assert a.all, (wsq, bsq, bad, b.fen())
        return None
    return b

cells = [(wk, bk, wsq, bsq, board(wsq, bsq)) for wk, bk in itertools.product(W, B)
         for wsq, bsq in itertools.product(W[wk], B[bk]) if wsq != bsq]
cells = [(wk, bk, wsq, bsq) for wk, bk, wsq, bsq, b in cells if b is not None]
fens = [FEN] + [board(wsq, bsq).fen() for *_, wsq, bsq in cells]

def fams(r):
    on = {}
    for f in r:
        k = (l.family_of(f["id"]), f["side"] or "")
        on.setdefault(k, set())
        if f["on"]:
            on[k].add(f["id"])
    return on
rows = l.features(fens)
F0 = fams(rows[0])
changed = [sorted(f"{k[0]}{'.' + k[1] if k[1] else ''}" for k, v in fams(r).items() if F0.get(k) != v) for r in rows]
d5 = [any(f["on"] and f["id"].startswith("pawns.weak.") and f["side"] == "them" and "d5" in f["squares"] for f in r)
      for r in rows]
raw = l.leela_q(fens)

def search(cmd, limit, opts):
    out = []
    with chess.engine.SimpleEngine.popen_uci(cmd) as e:
        e.configure(opts)
        for f in fens:
            info = e.analyse(chess.Board(f), limit, game=object())
            w = info["wdl"].white()
            out.append(((w.wins - w.losses) / w.total(), info["score"].white().score(mate_score=10000)))
    return np.array(out)

if a.no_search:
    lc0 = sf = np.full((len(fens), 2), np.nan)
else:
    lc0 = search(LC0, chess.engine.Limit(nodes=a.nodes), {"UCI_ShowWDL": True})
    sf = search(SF, chess.engine.Limit(depth=a.depth), {"UCI_ShowWDL": True, "Threads": 4})

print(f"{'white':5} {'black':5} {'d5 weak':7} {'raw Q':>6} {'lc0 Q':>6} {'SF Q':>6} {'SF cp':>6}  changed families")
print(f"{'-':5} {'-':5} {str(d5[0]):7} {raw[0]:+6.2f} {lc0[0, 0]:+6.2f} {sf[0, 0]:+6.2f} {sf[0, 1]:+6.0f}  (base)")
for i, (wk, bk, wsq, bsq) in enumerate(cells, 1):
    if a.all:
        continue
    print(f"{wsq:5} {bsq:5} {str(d5[i]):7} {raw[i]:+6.2f} {lc0[i, 0]:+6.2f} {sf[i, 0]:+6.2f} {sf[i, 1]:+6.0f}  {', '.join(changed[i])}")

for wk in W:
    for bk in B:
        ix = [i for i, c in enumerate(cells, 1) if c[:2] == (wk, bk)]
        print(f"White {wk:10} Black {bk:12} {len(ix):4} boards, d5 weak in {np.mean([d5[i] for i in ix]):.0%}")
for name, v in (("Leela raw Q", raw), ("Leela search Q", lc0[:, 0]), ("Stockfish Q", sf[:, 0]), ("Stockfish cp", sf[:, 1])):
    m = {(wk, bk): np.mean([v[i] for i, c in enumerate(cells, 1) if c[:2] == (wk, bk)]) for wk in W for bk in B}
    print(f"\n{name}{'':14}" + "".join(f"{'Black ' + bk:>20}" for bk in B) + f"{'difference':>14}")
    for wk in W:
        print(f"  White {wk:20}" + "".join(f"{m[wk, bk]:+20.2f}" for bk in B) + f"{m[wk, 'protects d5'] - m[wk, 'elsewhere']:+14.2f}")
    print(f"  {'difference':26}" + "".join(f"{m['d-pawn', bk] - m['elsewhere', bk]:+20.2f}" for bk in B)
          + f"{(m['d-pawn', 'protects d5'] - m['d-pawn', 'elsewhere']) - (m['elsewhere', 'protects d5'] - m['elsewhere', 'elsewhere']):+14.2f}")
