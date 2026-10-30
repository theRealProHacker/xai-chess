"""Space by piece, real board only: each piece's IG over its own control.geometry.t bits, weighted by t's forward
rank for its side (sign for X: + own pieces). The credited pieces together vs every same-size set of the other
non-king pieces of that side. Pass: together > 0; strict: also above 90% of those sets. 39: Black's pieces on files
a-c together <= 0.

  python lc0-probe/space_piece.py [ig suffix, default square] [tap, default encoder0_ln2]
"""
import sys
SUF = sys.argv[1] if len(sys.argv) > 1 else "square"
TAP = sys.argv[2] if len(sys.argv) > 2 else "encoder0_ln2"
sys.argv[1:] = [SUF, "pg+pp", TAP]
import itertools, numpy as np, chess
import space_lab as S

def pieces(fen, X):
    """{real square: space value} for X's non-king pieces and pawns, from their own control.geometry bits."""
    real = chess.Board(fen); mir = real.turn == chess.BLACK
    b = real.mirror() if mir else real
    C = (not X) if mir else X
    r = next(p for p in S.A["positions"] if p["fen"].split()[0] == b.fen().split()[0])
    sg = 1 if C == chess.WHITE else -1
    out = {}
    for s, pc in b.piece_map().items():
        if pc.color != C or pc.piece_type == chess.KING: continue
        q = r["squares"].get(chess.square_name(s), {"bits": []})
        ig = {S.A["labels"][j]: v for j, v in q["bits"] + q.get("absent", [])}
        v = sum(sg * ig.get(S.GEO[t], 0) * S.FWD[C][t] for t in chess.SQUARES)
        out[chess.square_mirror(s) if mir else s] = v
    return out

ok, st = 0, 0
for c, X in ((1, chess.WHITE), (2, chess.BLACK), (3, chess.WHITE), (4, chess.BLACK), (39, chess.BLACK)):
    g = S.CC[next(k for k in S.CC if c in k)]
    v = pieces(g["orig_fen"], X); bd = chess.Board(g["orig_fen"])
    if c == 39:
        tot = sum(x for s, x in v.items() if chess.square_file(s) <= 2); p = q = tot <= 0
        print(f"{c:3} Black a-c pieces {tot:+.4f}  {'pass' if p else 'fail'}  (all Black pieces: " +
              " ".join(f"{bd.piece_at(s).symbol()}{chess.square_name(s)} {x:+.4f}" for s, x in sorted(v.items(), key=lambda z: -z[1])) + ")")
    else:
        cred = [chess.parse_square(t["edit"].split()[1]) for t in g["tests"]]
        tot = sum(v[s] for s in cred)
        null = np.array([sum(v[s] for s in k) for k in itertools.combinations([s for s in v if s not in cred], len(cred))])
        f = np.mean(null >= tot); p, q = tot > 0, tot > 0 and f <= .1
        print(f"{c:3} " + " + ".join(f"{bd.piece_at(s).symbol()}{chess.square_name(s)} {v[s]:+.4f}" for s in cred)
              + f" = {tot:+.4f}  {'pass' if p else 'fail'} | {f:.0%} of {len(null)} other sets >=  strict {'pass' if q else 'fail'}")
    ok += p; st += q
print(f"{TAP} {SUF} credited pieces together: {ok}/5, strict {st}/5")
