"""Space claims (1-4, 39) as control: integrated gradients (any-occupant baseline) weighted by each control bit's effect
on a space count.

Space(X) = zone squares (the claim's files, default c-f; X's ranks 2-4) that X controls (more control.geometry bits
of X's pieces and pawns on it than the enemy's, the rule of square.control) and that no enemy pawn attacks. Pieces and
pawns count only through their control; standing somewhere is not space. A control.geometry.t bit on a piece's square,
t in the zone, gets w = count with the bit on - count with it off (holding every other bit); its space attribution
against the baseline is (x - mean) * w. Score = sum of IG * that, IG signed for X; also over sum |attribution|.

  python lc0-probe/space_control.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import chess, numpy as np, torch, subprocess

L = lambda f: json.loads(Path(f).read_text().split("=", 1)[1].rstrip().rstrip(";"))
A = L(ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_square_claim_concepts.js")
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
mism = [0, 0]                                                              # geometry count vs square.control bit

def space(fen, X, files):
    """(count, score, score / sum|attribution|, bits [real square, label, IG, space attribution]) for real side X."""
    real = chess.Board(fen); mir = real.turn == B
    b = real.mirror() if mir else real                                     # stored frame, as the IG file
    C = (not X) if mir else X
    r = A["positions"][fens.index(fen)]; assert r["fen"].split()[0] == b.fen().split()[0]
    ranks = [1, 2, 3] if C == W else [6, 5, 4]
    zone = {chess.square("abcdefgh".index(f), k) for f in files for k in ranks}       # mirroring keeps files
    ig, on = {}, {}
    for s, q in r["squares"].items():
        s_ = chess.parse_square(s)
        for j, v in q["bits"] + q.get("absent", []): ig[(s_, A["labels"][j])] = v
        on[s_] = {A["labels"][j] for j, _ in q["bits"]}
    n = {t: [0, 0, 0] for t in zone}                                       # X's control, enemy's, enemy pawns'
    for s, ls in on.items():
        p = b.piece_at(s)
        for l in ls:
            if l.startswith("control.geometry.") and (t := chess.parse_square(l.rsplit(".", 1)[1])) in zone:
                n[t][p.color != C] += 1; n[t][2] += p.color != C and p.piece_type == chess.PAWN
    for t in zone:                                                         # the count agrees with square.control
        sc = next((l for l in on.get(t, ()) if l.startswith("square.control.")), "")
        d = (n[t][0] - n[t][1]) * (1 if C == W else -1)
        mism[0] += 1; mism[1] += sc != "square.control." + ("equal" if d == 0 else f"{'white' if d > 0 else 'black'}{min(abs(d), 3)}{'plus' if abs(d) >= 3 else ''}")
    held = lambda x, y, pw: x > y and pw == 0
    count = sum(held(*n[t]) for t in zone)
    sg = 1 if C == W else -1
    bits = []
    for t in zone:
        x, y, pw = n[t]; lab = f"control.geometry.{chess.square_name(t)}"
        for s in chess.SQUARES:
            p = b.piece_at(s)
            if p is None or s == t: continue
            mine, pawn = p.color == C, p.piece_type == chess.PAWN and p.color != C
            bit = lab in on.get(s, ())
            d = 1 if bit else -1                                           # the other state: bit off (on)
            alt = (x - d, y, pw) if mine else (x, y - d, pw - d * pawn)
            w = (held(x, y, pw) - held(*alt)) * d                          # count(on) - count(off)
            dx = bit - float(SM[s, COL[lab]])
            if w and dx: bits.append((s, lab, sg * ig.get((s, lab), 0.0), dx * w))
    score = sum(v * a for _, _, v, a in bits)
    f = (lambda z: chess.square_name(chess.square_mirror(chess.parse_square(z)))) if mir else (lambda z: z)
    out = sorted(([f(chess.square_name(s)), "control.geometry." + f(l.rsplit(".", 1)[1]), round(v, 5), round(a, 3)]
                  for s, l, v, a in bits if v), key=lambda z: -abs(z[2] * z[3]))
    return count, score, score / max(sum(abs(a) for *_, a in bits), 1e-9), out

TESTS = {1: (W, "cdef"), 2: (B, "cdef"), 3: (W, "cdef"), 4: (B, "abcd"), 39: (B, "abc")}
res = []
for c, (X, files) in TESTS.items():
    g = CC[next(k for k in CC if c in k)]
    n0, s0, z0, b0 = space(g["orig_fen"], X, files)
    r = {"claim": c, "quote": g["quote"], "side": "white" if X == W else "black", "files": files, "orig_fen": g["orig_fen"],
         "count": n0, "score": round(s0, 5), "score/|a|": round(z0, 6), "bits": b0[:40], "tests": []}
    line = f"{c:3} {r['side']:5} squares {n0:2} score {s0:+.4f} ({z0:+.5f})"
    if c == 39:
        r["pass"] = s0 <= 0; line += f"  expect <= 0: {'pass' if r['pass'] else 'fail'}"
    else:
        for t in g["tests"]:
            n1, s1, z1, b1 = space(t["fen without"], X, files)
            p = s1 < s0 and z1 < z0
            r["tests"].append({"edit": t["edit"], "fen": t["fen without"], "count": n1, "score": round(s1, 5),
                               "score/|a|": round(z1, 6), "bits": b1[:40], "pass": bool(p)})
            line += f"\n      {t['edit']:7} squares {n1:2} score {s1:+.4f} ({z1:+.5f})  {'pass' if p else 'fail'}"
        r["pass"] = all(t["pass"] for t in r["tests"])
    print(line, flush=True); res.append(r)
print(f"geometry count disagrees with square.control on {mism[1]} of {mism[0]} zone squares")
json.dump(res, open(ROOT / "lc0-probe/results_space_control.json", "w"), indent=1)
print(f"{sum(r['pass'] for r in res)} of {len(res)} supported")
