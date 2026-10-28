"""Space claims (1-4, 39): one IG file x one space concept -> with/without and own-share verdicts.

IG file (suffix): off, mean, square, start (the starting position), square_only_square_control, mean_only_square_control, expected.
Concept, "+"-joined: sc = square.control margin on each square; pc = pawn_cover margin (pawns' control, on the
squares they cover); pg = non-pawn pieces' control.geometry bits on their own square (pp adds pawns, k the king).
Each side's control weighted by its own forward rank (flat: weight 1; bin: a square counts 1 for the side
holding it, whatever the margin). net = score is the value X gets from its space minus the value
the enemy gets from its space (sum of IG_X * (x - mean) * fwd, no sign flip by owner); without net the enemy's
term is flipped, so enemy space that helps the enemy counts for X. Own share: X's own control only, a square's sc part
split among X's controllers, pc part among X's pawns covering it, pg bits to their piece.

  python lc0-probe/space_lab.py square sc+net [tap, default encoder0_ln2] [claim 39 files, default abc]
"""

import json, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import chess, numpy as np, torch

SUF, CON = sys.argv[1], sys.argv[2].split("+")
NET = "net" in CON                                                         # net: value of X's space - value of the enemy's space
BASE = SUF.split("_")[0]
L = lambda f: json.loads(Path(f).read_text().split("=", 1)[1].rstrip().rstrip(";"))
TAP = sys.argv[3] if len(sys.argv) > 3 else "encoder0_ln2"
Q39 = sys.argv[4] if len(sys.argv) > 4 else "abc"                          # claim 39's Queen's wing files
A = next(L(f) for x in ("claim_concepts", "space_claims")
         if (f := ROOT / f"lc0-probe/square_attribution_n100000{'' if TAP == 'attn_body_rehape' else '_' + TAP}_ig{'' if SUF == 'off' else '_' + SUF}_{x}.js").exists())
fens = [l for l in (ROOT / "lc0-probe/data/claim_concepts.fens").read_text().split("\n") if l]
CC = {tuple(g["claims"]): g for g in json.load(open(ROOT / "lc0-probe/results_claim_concepts.json"))["games"]}
W, B = chess.WHITE, chess.BLACK

keep = torch.load(ROOT / f"lc0-probe/data/square_decoder_{TAP}_n100000.pt", weights_only=False, mmap=True)["keep"]
SCH = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                capture_output=True, text=True, check=True).stdout)["labels"]
COL = {SCH[i]: n for n, i in enumerate(keep)}
Z = np.load(ROOT / f"lc0-probe/data/ig_means_n100000_20000{'' if len(keep) == 22498 else f'_k{len(keep)}'}.npz"); XM, XC = Z["XM"], Z["XC"]
SM = (XM * XC[:, None]).reshape(13, 64, -1).sum(0) / np.maximum(XC.reshape(13, 64).sum(0), 1)[:, None]
rank = np.array([chess.square_rank(t) for t in chess.SQUARES])
FWD = {W: rank.astype(float), B: (7 - rank).astype(float)}
KEYS = ".PNBRQKpnbrqk"
XS = np.load(ROOT / f"lc0-probe/data/ig_start_bits_{TAP}.npy") if BASE == "start" else None
LAB = []                                                                   # (label, colour, level, kind)
if "sc" in CON:
    LAB += [(f"square.control.{c}{k}", c, n, "sc") for c in ("white", "black") for k, n in (("1", 1), ("2", 2), ("3plus", 3))]
    LAB += [("square.control.equal", None, 0, "sc")]
if "pc" in CON:
    LAB += [(f"pawn_cover.{c}.{k}", c, n, "pc") for c in ("white", "black") for k, n in (("none", 0), ("one", 1), ("two", 2))]
LAB = [x for x in LAB if x[0] in COL]
GEO = [f"control.geometry.{chess.square_name(t)}" for t in chess.SQUARES]

def run(fen, X, files="abcdefgh"):
    real = chess.Board(fen); mir = real.turn == B
    b = real.mirror() if mir else real
    C = (not X) if mir else X
    r = next(p for p in A["positions"] if p["fen"].split()[0] == b.fen().split()[0])
    sg = 1 if C == W else -1
    occ = lambda s: KEYS.index(b.piece_at(s).symbol()) if b.piece_at(s) else 0
    mean = lambda t, j: 0.0 if BASE == "off" else float(XS[t, j]) if BASE == "start" else float(SM[t, j] if BASE in ("square", "expected") else XM[occ(t) * 64 + t, j])
    on = {chess.parse_square(s): {A["labels"][j] for j, _ in q["bits"]} for s, q in r["squares"].items()}
    total, bits = 0.0, []
    for t in chess.SQUARES:
        if "abcdefgh"[chess.square_file(t)] not in files: continue
        q = r["squares"].get(chess.square_name(t), {"bits": []})
        ig = {A["labels"][j]: v for j, v in q["bits"] + q.get("absent", [])}
        for l, c, k, kind in LAB:
            if c is None: continue
            mine = (c == "white") == (C == W)
            k_ = min(k, 1) if "bin" in CON else k
            w = k_ * (1 if mine else -1) * (1.0 if "flat" in CON else FWD[C if mine else not C][t])
            x = l in on.get(t, ()); total += x * w
            a = (x - mean(t, COL[l])) * w
            z = sg * ig.get(l, 0) * (abs(a) * np.sign(x - mean(t, COL[l])) if NET else a)
            if a and ig.get(l): bits.append((t, kind if mine else "e", z, a))
    if "pg" in CON:                                                        # non-pawn pieces' own control bits
        for s, pc in b.piece_map().items():
            if pc.piece_type == chess.PAWN and "pp" not in CON or pc.piece_type == chess.KING and "k" not in CON: continue
            if "abcdefgh"[chess.square_file(s)] not in files: continue
            q = r["squares"].get(chess.square_name(s), {"bits": []})
            ig = {A["labels"][j]: v for j, v in q["bits"] + q.get("absent", [])}
            mine = pc.color == C
            for t in chess.SQUARES:
                l = GEO[t]
                if not ig.get(l): continue
                w = (1 if mine else -1) * FWD[pc.color][t]
                x = l in on.get(s, ()); total += x * w
                a = (x - mean(s, COL[l])) * w
                z = sg * ig[l] * (abs(a) * np.sign(x - mean(s, COL[l])) if NET else a)
                if a: bits.append((s, "pg" if mine else "e", z, a))
    score = sum(z for _, _, z, _ in bits)
    norm = score / max(sum(abs(a) for *_, a in bits), 1e-9)
    # own share: sc part split among C's controllers of t, pc part among C's pawns controlling t
    f = (lambda s: chess.square_mirror(s)) if mir else (lambda s: s)
    by = {}
    for t, kind, z, _ in bits:
        if kind == "e": continue                                           # only X's own control goes to X's pieces
        if kind == "pg":
            if b.color_at(t) == C: by[f(t)] = by.get(f(t), 0.0) + z
            continue
        g = f"control.geometry.{chess.square_name(t)}"
        ctl = [s for s in chess.SQUARES if b.color_at(s) == C and g in on.get(s, ())
               and (kind == "sc" or b.piece_type_at(s) == chess.PAWN)]
        for s in ctl: by[f(s)] = by.get(f(s), 0.0) + z / len(ctl)
    return total, score, norm, by, sum(abs(z) for _, _, z, _ in bits)

TESTS = {1: W, 2: B, 3: W, 4: B, 39: B}
ww, sh = 0, 0
for c, X in TESTS.items():
    g = CC[next(k for k in CC if c in k)]
    n0, s0, z0, by, tot = run(g["orig_fen"], X, Q39 if c == 39 else "abcdefgh")
    bd = chess.Board(g["orig_fen"])
    if c == 39:
        p = s0 <= 0
        _, _, _, by, tot = run(g["orig_fen"], X)
        sq = [s for s in chess.SQUARES if bd.color_at(s) == X and chess.square_file(s) <= 2]
        q = sum(by.get(s, 0) for s in sq) <= 0
        print(f"{c:3} space {n0:+4.0f} score {s0:+.4f}  {'pass' if p else 'fail'} | share a-c {sum(by.get(s, 0) for s in sq):+.4f} {'pass' if q else 'fail'}")
    else:
        line, p = f"{c:3} space {n0:+4.0f} score {s0:+.4f}", True
        for t in g["tests"]:
            n1, s1, z1, _, _ = run(t["fen without"], X)
            ok = s1 < s0 and z1 < z0; p &= ok
            line += f" | {t['edit']} {n1:+4.0f} {s1:+.4f} {'P' if ok else 'F'}"
        sq = [chess.parse_square(t["edit"].split()[1]) for t in g["tests"]]
        q = all(by.get(s, 0) > 0 for s in sq)
        line += " || share " + " ".join(f"{bd.piece_at(s).symbol()}{chess.square_name(s)} {by.get(s, 0) / max(tot, 1e-9):+.1%}" for s in sq) + f" {'pass' if q else 'fail'}"
        print(line)
    ww += p; sh += q
print(f"{TAP} {SUF} {'+'.join(CON)}: with/without {ww}/5, own share {sh}/5")
