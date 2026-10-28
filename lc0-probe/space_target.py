"""Space claims (1-4, 39) on the controlled squares: integrated gradients weighted by each square.control bit's
attribution to a space score.

Space(X) = sum over squares t of margin_X(t) * fwd of the side holding t; fwd = t's rank seen from that side - 1; margin
from the square.control one-hot on t (X1 +1, X2 +2, X3plus +3, enemy the same negative, equal 0), so pawns and pieces count through the squares
they control, not through standing somewhere. Whole board for 1-4, files a-c for 39 (Queen's wing). A bit's space
attribution against the baseline: (x - mean) * w. Score = sum of IG * that, IG signed for X; also over sum |attribution|.
--base square: any-occupant baseline (IG and means); --base mean: same occupant. --only: IG path over the
square.control bits only, every other bit at its value in the position.

  python lc0-probe/space_target.py --base square
"""
import argparse, json, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import chess, numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument("--base", choices=["mean", "square"], default="square")
ap.add_argument("--only", action="store_true", help="IG with the path over square.control bits only")
BASE, ONLY = (lambda z: (z.base, z.only))(ap.parse_args())
SUF = BASE + ("_only_square_control" if ONLY else "")
L = lambda f: json.loads(Path(f).read_text().split("=", 1)[1].rstrip().rstrip(";"))
A = L(ROOT / f"lc0-probe/square_attribution_n100000_encoder0_ln2_ig_{SUF}_claim_concepts.js")
fens = [l for l in (ROOT / "lc0-probe/data/claim_concepts.fens").read_text().split("\n") if l]
assert len(fens) == len(A["positions"])
CC = {tuple(g["claims"]): g for g in json.load(open(ROOT / "lc0-probe/results_claim_concepts.json"))["games"]}
W, B = chess.WHITE, chess.BLACK
Z = np.load(ROOT / "lc0-probe/data/ig_means_n100000_20000.npz"); XM, XC = Z["XM"], Z["XC"]
SM = (XM * XC[:, None]).reshape(13, 64, -1).sum(0) / np.maximum(XC.reshape(13, 64).sum(0), 1)[:, None]   # any occupant
keep = torch.load(ROOT / "lc0-probe/data/square_decoder_encoder0_ln2_n100000.pt", weights_only=False)["keep"]
SCH = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                capture_output=True, text=True, check=True).stdout)["labels"]
COL = {SCH[i]: n for n, i in enumerate(keep)}
LEV = {"white1": 1, "white2": 2, "white3plus": 3, "black1": -1, "black2": -2, "black3plus": -3, "equal": 0}
rank = np.array([chess.square_rank(t) for t in chess.SQUARES])
FWD = {W: rank.astype(float), B: (7 - rank).astype(float)}
KEYS = ".PNBRQKpnbrqk"

def space(fen, X, files):
    """(space, score, score / sum|attribution|, bits [real square, label, IG, space attribution]) for real side X."""
    real = chess.Board(fen); mir = real.turn == B
    b = real.mirror() if mir else real                                     # stored frame, as the IG file
    C = (not X) if mir else X
    r = A["positions"][fens.index(fen)]; assert r["fen"].split()[0] == b.fen().split()[0]
    sg = 1 if C == W else -1
    occ = lambda s: KEYS.index(b.piece_at(s).symbol()) if b.piece_at(s) else 0
    mean = lambda t, j: float(SM[t, j] if BASE == "square" else XM[occ(t) * 64 + t, j])
    total, bits = 0.0, []
    for t in chess.SQUARES:
        if "abcdefgh"[chess.square_file(t)] not in files: continue        # mirroring keeps files
        q = r["squares"].get(chess.square_name(t), {"bits": []})
        ig = {A["labels"][j]: v for j, v in q["bits"] + q.get("absent", [])}
        on = {A["labels"][j] for j, _ in q["bits"]}
        for lv, k in LEV.items():
            l = f"square.control.{lv}"; w = k * sg * FWD[C if k * sg > 0 else not C][t]
            x = l in on; total += x * w
            a = (x - mean(t, COL[l])) * w
            if a and ig.get(l): bits.append((t, l, sg * ig[l], a))
    score = sum(v * a for _, _, v, a in bits)
    f = (lambda z: chess.square_name(chess.square_mirror(chess.parse_square(z)))) if mir else (lambda z: z)
    out = sorted(([f(chess.square_name(t)), l, round(float(v), 5), round(float(a), 3)] for t, l, v, a in bits),
                 key=lambda z: -abs(z[2] * z[3]))
    tig = sum(abs(v) for q in r["squares"].values() for _, v in q["bits"] + q.get("absent", []))
    return float(total), float(score), float(score / max(sum(abs(a) for *_, a in bits), 1e-9)), out, float(score / tig)

TESTS = {1: W, 2: B, 3: W, 4: B, 39: B}
res = []
for c, X in TESTS.items():
    g = CC[next(k for k in CC if c in k)]
    files = "abc" if c == 39 else "abcdefgh"
    n0, s0, z0, b0, u0 = space(g["orig_fen"], X, files)
    r = {"claim": c, "quote": g["quote"], "side": "white" if X == W else "black", "files": files, "orig_fen": g["orig_fen"],
         "space": n0, "score": round(s0, 5), "score/|a|": round(z0, 6), "score/|IG|": round(u0, 6), "bits": b0[:40], "tests": []}
    line = f"{c:3} {r['side']:5} space {n0:+5.0f} score {s0:+.4f} ({z0:+.5f}) /|IG| {u0:+.5f}"
    if c == 39:
        r["pass"] = bool(s0 <= 0); line += f"  expect <= 0: {'pass' if r['pass'] else 'fail'}"
    else:
        for t in g["tests"]:
            n1, s1, z1, b1, u1 = space(t["fen without"], X, files)
            p = s1 < s0 and z1 < z0
            r["tests"].append({"edit": t["edit"], "fen": t["fen without"], "space": n1, "score": round(s1, 5),
                               "score/|a|": round(z1, 6), "score/|IG|": round(u1, 6), "bits": b1[:40],
                               "pass": bool(p), "pass |IG|": bool(u1 < u0)})
            line += f"\n      {t['edit']:7} space {n1:+5.0f} score {s1:+.4f} ({z1:+.5f}) /|IG| {u1:+.5f}  {'pass' if p else 'fail'}  /|IG| {'pass' if u1 < u0 else 'fail'}"
        r["pass"] = bool(all(t["pass"] for t in r["tests"])); r["pass |IG|"] = bool(all(t["pass |IG|"] for t in r["tests"]))
    print(line, flush=True); res.append(r)
json.dump(res, open(ROOT / f"lc0-probe/results_space_target_{SUF}.json", "w"), indent=1)
print(f"{sum(r['pass'] for r in res)} of {len(res)} supported; score / total |IG|: {sum(r.get('pass |IG|', r['pass']) for r in res)} of {len(res)}")

# --- own share on target squares, no removal: each square's space attribution (sum of IG * a over its square.control
# bits) split equally among X's pieces and pawns that control it (control.geometry.t on their square). Pass if every
# credited piece's share > 0; 39: Black's pieces and pawns on files a-c together <= 0.
def shares(fen, X):
    real = chess.Board(fen); mir = real.turn == B
    b = real.mirror() if mir else real
    C = (not X) if mir else X
    r = A["positions"][fens.index(fen)]
    on = {chess.parse_square(s): {A["labels"][j] for j, _ in q["bits"]} for s, q in r["squares"].items()}
    _, _, _, bits, _ = space(fen, X, "abcdefgh")
    f = (lambda s: chess.square_mirror(s)) if mir else (lambda s: s)       # stored <-> real
    att = {}
    for q, _, v, a in bits: att[chess.parse_square(q)] = att.get(chess.parse_square(q), 0.0) + v * a   # real squares
    by, tot = {}, sum(abs(x) for x in att.values())
    for t, x in att.items():
        ctl = [s for s in chess.SQUARES if b.color_at(s) == C and f"control.geometry.{chess.square_name(f(t))}" in on.get(s, ())]
        for s in ctl: by[f(s)] = by.get(f(s), 0.0) + x / len(ctl)
    return by, tot

share = []
for r in res:
    X = W if r["side"] == "white" else B
    by, tot = shares(r["orig_fen"], X)
    b = chess.Board(r["orig_fen"])
    sqs = ([s for s in chess.SQUARES if b.color_at(s) == X and chess.square_file(s) <= 2] if r["claim"] == 39
           else [chess.parse_square(t["edit"].split()[1]) for t in r["tests"]])
    rows = [{"square": chess.square_name(s), "piece": b.piece_at(s).symbol(), "share": round(by.get(s, 0.0), 5),
             "of |score|": round(by.get(s, 0.0) / tot, 4)} for s in sqs]
    ok = (sum(x["share"] for x in rows) <= 0) if r["claim"] == 39 else all(x["share"] > 0 for x in rows)
    share.append({"claim": r["claim"], "pieces": rows, "pass": bool(ok)})
    print(f"{r['claim']:3} {r['side']:5}  " + "  ".join(
        f"{x['piece']}{x['square']} {x['share']:+.4f} ({x['of |score|']:+.1%})" for x in rows) + f"  {'pass' if ok else 'fail'}")
json.dump(share, open(ROOT / f"lc0-probe/results_space_target_share_{SUF}.json", "w"), indent=1)
print(f"own share: {sum(x['pass'] for x in share)} of {len(share)} supported")
