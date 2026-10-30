"""Noise floor for one space setup (space_lab.py: encoder0_ln2, square baseline, W-L, sc+net).

Zone: claim 39's queenside score next to the same score on every 3-file window, both sides, every position.
Removal: for claims 1-4, the credited pieces' drops next to removing each other non-king piece of that side;
strict pass = the drop beats 90% of the random removals' drops
(data/space_null.fens, IG in square_attribution_n100000_encoder0_ln2_ig_square_space_null.js).

  python lc0-probe/space_null.py
"""
import sys
sys.argv[1:] = ["square", "sc+net"]
import numpy as np, chess
import space_lab as S

B = S.L(S.ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_square_space_null.js")
null_fens = [l for l in (S.ROOT / "lc0-probe/data/space_null.fens").read_text().split("\n") if l]
W, Bk = chess.WHITE, chess.BLACK

# --- zone
g39 = S.CC[next(k for k in S.CC if 39 in k)]
s39 = S.run(g39["orig_fen"], Bk, "abc")[1]
win = ["abcdefgh"[i:i + 3] for i in range(6)]
z = []
for A, fens in ((S.A, [p["fen"] for p in S.A["positions"]]), (B, null_fens)):
    for f in fens:
        for X in (W, Bk):
            for w in win: z.append(S.run(f, X, w, A)[1])
z = np.array(z)
print(f"zone: claim 39 score {s39:+.4f}; {len(z)} window scores, |score| median {np.median(abs(z)):.4f}, "
      f"90% {np.quantile(abs(z), .9):.4f}; {np.mean(abs(z) <= abs(s39)):.0%} have |score| <= |claim 39|")

# --- removal
for c, X in ((1, W), (2, Bk), (3, W), (4, Bk)):
    g = S.CC[next(k for k in S.CC if c in k)]
    _, s0, z0, _, _ = S.run(g["orig_fen"], X)
    d = lambda f, A: (lambda r: (r[1] - s0, r[2] - z0))(S.run(f, X, "abcdefgh", A))
    cred = [(t["edit"], *d(t["fen without"], S.A)) for t in g["tests"]]
    o, rnd = chess.Board(g["orig_fen"]), []
    for f in null_fens:                                                    # boards one piece away from this claim's
        b = chess.Board(f)
        diff = [s for s in chess.SQUARES if o.piece_at(s) != b.piece_at(s)]
        if len(diff) == 1 and b.turn == o.turn: rnd.append((chess.square_name(diff[0]), *d(f, B)))
    ds = np.array([x[1] for x in rnd])
    rate = np.mean([x[1] < 0 and x[2] < 0 for x in rnd])
    print(f"claim {c}: real {s0:+.4f}; random removals n={len(rnd)} pass rate {rate:.0%}, drop median {np.median(ds):+.4f}")
    for e, ds_, dz in cred:
        q = np.mean(ds <= ds_)
        print(f"    {e:7} drop {ds_:+.4f}  pass {'yes' if ds_ < 0 and dz < 0 else 'no '}  "
              f"{q:.0%} of random removals drop as much  strict {'pass' if ds_ < 0 and dz < 0 and q <= .1 else 'fail'}")
    print(f"  claim {c} strict: {'pass' if all(x[1] < 0 and x[2] < 0 and np.mean(ds <= x[1]) <= .1 for x in cred) else 'fail'}")
