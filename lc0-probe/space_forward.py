"""Space claims (1-4, 39) as forward control: integrated gradients (any-occupant baseline) weighted by each control
bit's attribution to a space score.

Space(X) = sum over X's control.geometry.t bits of fwd_X(t), minus the same for the enemy, fwd = the square's rank seen
from that side - 1 (0 on your back rank, 7 on the enemy's). Whole board. A bit's space attribution against the
baseline: its value now (x * +-fwd by the colour standing there) minus its baseline mean over the occupants of that
square (the means behind --ig-base square). Score = sum of IG * that, IG signed for X; also over sum |attribution|.

  python lc0-probe/space_forward.py
"""
import json, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import chess, numpy as np, torch

L = lambda f: json.loads(Path(f).read_text().split("=", 1)[1].rstrip().rstrip(";"))
A = L(ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_square_claim_concepts.js")
fens = [l for l in (ROOT / "lc0-probe/data/claim_concepts.fens").read_text().split("\n") if l]
assert len(fens) == len(A["positions"])
CC = {tuple(g["claims"]): g for g in json.load(open(ROOT / "lc0-probe/results_claim_concepts.json"))["games"]}
W, B = chess.WHITE, chess.BLACK
Z = np.load(ROOT / "lc0-probe/data/ig_means_n100000_20000.npz"); XM, XC = Z["XM"], Z["XC"]
keep = torch.load(ROOT / "lc0-probe/data/square_decoder_encoder0_ln2_n100000.pt", weights_only=False)["keep"]
SCH = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                capture_output=True, text=True, check=True).stdout)["labels"]
COL = {SCH[i]: n for n, i in enumerate(keep)}
GEO = [f"control.geometry.{chess.square_name(t)}" for t in chess.SQUARES]
GI = np.array([COL[l] for l in GEO])
rank = np.array([chess.square_rank(t) for t in chess.SQUARES])
FWD = {W: rank.astype(float), B: (7 - rank).astype(float)}
P = XC.reshape(13, 64) / np.maximum(XC.reshape(13, 64).sum(0), 1)          # occupant frequencies per square
KEYS = ".PNBRQKpnbrqk"

def space(fen, X):
    """(space, score, score / sum|attribution|, bits [real square, label, IG, space attribution]) for real side X."""
    real = chess.Board(fen); mir = real.turn == B
    b = real.mirror() if mir else real                                     # stored frame, as the IG file
    C = (not X) if mir else X
    r = A["positions"][fens.index(fen)]; assert r["fen"].split()[0] == b.fen().split()[0]
    wt = lambda c: FWD[c] if c == C else -FWD[c]                           # (64,) weight of a c bit into each target
    ig, on = {}, {}
    for s, q in r["squares"].items():
        s_ = chess.parse_square(s)
        for j, v in q["bits"] + q.get("absent", []): ig[(s_, A["labels"][j])] = v
        on[s_] = {A["labels"][j] for j, _ in q["bits"]}
    sg = 1 if C == W else -1
    total, bits = 0.0, []
    for s in chess.SQUARES:
        p = b.piece_at(s)
        now = np.zeros(64) if p is None else np.array([l in on.get(s, ()) for l in GEO]) * wt(p.color)
        base = sum(P[k, s] * XM[k * 64 + s, GI] * wt(KEYS[k].isupper()) for k in range(1, 13))
        total += now.sum()
        for t in np.flatnonzero(np.abs(now - base) > 1e-9):
            v = ig.get((s, GEO[t]), 0.0)
            if v: bits.append((s, GEO[t], sg * v, now[t] - base[t]))
    score = sum(v * a for _, _, v, a in bits)
    f = (lambda z: chess.square_name(chess.square_mirror(chess.parse_square(z)))) if mir else (lambda z: z)
    out = sorted(([f(chess.square_name(s)), "control.geometry." + f(l.rsplit(".", 1)[1]), round(float(v), 5), round(float(a), 3)]
                  for s, l, v, a in bits), key=lambda z: -abs(z[2] * z[3]))
    return float(total), float(score), float(score / max(sum(abs(a) for *_, a in bits), 1e-9)), out

TESTS = {1: W, 2: B, 3: W, 4: B, 39: B}
res = []
for c, X in TESTS.items():
    g = CC[next(k for k in CC if c in k)]
    n0, s0, z0, b0 = space(g["orig_fen"], X)
    r = {"claim": c, "quote": g["quote"], "side": "white" if X == W else "black", "orig_fen": g["orig_fen"],
         "space": n0, "score": round(s0, 5), "score/|a|": round(z0, 6), "bits": b0[:40], "tests": []}
    line = f"{c:3} {r['side']:5} space {n0:+5.0f} score {s0:+.4f} ({z0:+.5f})"
    if c == 39:
        r["pass"] = bool(s0 <= 0); line += f"  expect <= 0: {'pass' if r['pass'] else 'fail'}"
    else:
        for t in g["tests"]:
            n1, s1, z1, b1 = space(t["fen without"], X)
            p = s1 < s0 and z1 < z0
            r["tests"].append({"edit": t["edit"], "fen": t["fen without"], "space": n1, "score": round(s1, 5),
                               "score/|a|": round(z1, 6), "bits": b1[:40], "pass": bool(p)})
            line += f"\n      {t['edit']:7} space {n1:+5.0f} score {s1:+.4f} ({z1:+.5f})  {'pass' if p else 'fail'}"
        r["pass"] = bool(all(t["pass"] for t in r["tests"]))
    print(line, flush=True); res.append(r)
json.dump(res, open(ROOT / "lc0-probe/results_space_forward.json", "w"), indent=1)
print(f"{sum(r['pass'] for r in res)} of {len(res)} supported")

# --- own share: each credited piece's part of the score on the real board (no removal); pass if > 0.
# 39: Black's pieces and pawns on files a-c together, pass if <= 0.
share = []
for r in res:
    X = W if r["side"] == "white" else B
    _, s0, _, bits = space(r["orig_fen"], X)
    by = {}
    for q, _, v, a in bits: by[q] = by.get(q, 0.0) + v * a
    tot = sum(abs(v * a) for _, _, v, a in bits)
    b = chess.Board(r["orig_fen"])
    sqs = ([chess.square_name(s) for s in chess.SQUARES if b.color_at(s) == X and chess.square_file(s) <= 2]
           if r["claim"] == 39 else [t["edit"].split()[1] for t in r["tests"]])
    rows = [{"square": q, "piece": b.piece_at(chess.parse_square(q)).symbol(), "share": round(by.get(q, 0.0), 5),
             "of |score|": round(by.get(q, 0.0) / tot, 4)} for q in sqs]
    ok = (sum(x["share"] for x in rows) <= 0) if r["claim"] == 39 else all(x["share"] > 0 for x in rows)
    share.append({"claim": r["claim"], "score": r["score"], "pieces": rows, "pass": bool(ok)})
    print(f"{r['claim']:3} {r['side']:5} score {s0:+.3f}  " + "  ".join(
        f"{x['piece']}{x['square']} {x['share']:+.4f} ({x['of |score|']:+.1%})" for x in rows) + f"  {'pass' if ok else 'fail'}")
json.dump(share, open(ROOT / "lc0-probe/results_space_forward_share.json", "w"), indent=1)
print(f"own share: {sum(x['pass'] for x in share)} of {len(share)} supported")
