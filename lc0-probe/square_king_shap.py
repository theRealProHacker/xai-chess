"""King safety as a Shapley player: every feature bit about one king's zone, for each king, through the per-square
decoder and the rest of Leela.

Players: White's king safety K_w and Black's K_b; everything else stays on. v(S) = P(win) - P(loss) for the side to
move (White: positions are mirrored). Exact over the 4 coalitions. Values are for White, so the king-safety edge is
phi_w + phi_b: K_w's bits helping White means White's king is safe, K_b's bits helping White means Black's is not.

A king's zone: its square and neighbours, plus the three squares two ranks in front of it. Its bits:
  on zone squares    pawn_shield, pawn_cover, square, weak, strong, open_file, half_open_file
  on the king square stability, king.castle
  into the zone      control.reach/geometry, role.attacks/restricts (enemy pieces), role.defends (own)
  aimed at the king  xray and threat.xray through or to a king (the piece's enemy king), check.on
  mating lines       mate.* (Black's king: White mates), threat.mate.* (White's king)
Compared with a pawn_shield-only player per king (the zone's pawn_shield bits), the read the family game gave.

  python lc0-probe/square_king_shap.py --tap /encoder0/ln2
"""
import argparse, json, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe"), str(ROOT / "commentary_gen")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, torch
from onnx2torch import convert
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--tap", default="/encoder0/ln2")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
tag = a.tap.replace("/", "_")

ck = torch.load(ROOT / f"lc0-probe/data/square_decoder{tag}_n{a.n}.pt", weights_only=False)
mu, sd = torch.from_numpy(ck["mu"]).float(), float(ck["sd"])
E, bias = ck["state"]["inp.weight"], ck["state"]["bias"]
out = torch.nn.Linear(*ck["state"]["out.weight"].shape[::-1])
out.load_state_dict({"weight": ck["state"]["out.weight"], "bias": ck["state"]["out.bias"]})
sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))).extract_model([a.tap], ["/output/wdl"])
tail = convert(sub).eval()
base = E[len(ck["keep"]):] + bias                                          # (64, hidden): square one-hot only

@torch.no_grad()
def value(P):
    w = tail((out(torch.nn.functional.gelu(P)) * sd + mu).reshape(-1, mu.shape[0]))
    return (w[:, 0] - w[:, 2]).numpy()

def zone(k, c):
    f, r, fwd = chess.square_file(k), chess.square_rank(k), 1 if c == chess.WHITE else -1
    z = {s for s in chess.SQUARES if chess.square_distance(s, k) <= 1}
    return z | {chess.square(ff, r + 2 * fwd) for ff in (f - 1, f, f + 1) if 0 <= ff < 8 and 0 <= r + 2 * fwd < 8}

ON_ZONE = ("pawn_shield", "pawn_cover", "square", "weak", "strong", "open_file", "half_open_file")
INTO = ("control.reach", "control.geometry", "role.attacks", "role.restricts", "role.defends")

def owner(board, s, l, Z):
    """Which king's player the bit at s with label l belongs to (chess colour), or None. Full and shield-only."""
    p, parts = board.piece_at(s), l.split(".")
    fam = parts[0]
    hits = [c for c in (chess.WHITE, chess.BLACK) if s in Z[c]]
    if fam in ON_ZONE: return hits[0] if hits else None             # zones of the two kings rarely overlap
    if fam in ("stability", "king") and p and p.piece_type == chess.KING: return p.color
    if fam == "king" and p: return p.color                          # castling right on the rook
    if l.startswith(INTO) and p:
        t = chess.parse_square(parts[-1])
        for c in (chess.WHITE, chess.BLACK):
            if t in Z[c] and (fam == "control" or (parts[1] == "defends") == (p.color == c)): return c
    if (fam == "xray" or l.startswith("threat.xray")) and p and "king" in parts[1:4]: return not p.color
    if fam == "check" and p: return not p.color
    if fam == "mate": return chess.BLACK
    if l.startswith("threat.mate"): return chess.WHITE
    return None

def run(src):
    A = json.loads((ROOT / "lc0-probe" / src).read_text().split("=", 1)[1].rstrip().rstrip(";"))
    res = []
    for pos in A["positions"]:
        b = chess.Board(pos["fen"])
        if b.king(chess.WHITE) is None or b.king(chess.BLACK) is None: continue
        Z = {c: zone(b.king(c), c) for c in (chess.WHITE, chess.BLACK)}
        full = base.clone()
        part = {(k, c): torch.zeros_like(base) for k in ("all", "shield") for c in (chess.WHITE, chess.BLACK)}
        nbits = {c: 0 for c in (chess.WHITE, chess.BLACK)}
        for name, q in pos["squares"].items():
            s = chess.parse_square(name)
            for j, _ in q["bits"]:
                l, row = A["labels"][j], E[ckcol[A["labels"][j]]]
                full[s] += row
                c = owner(b, s, l, Z)
                if c is None: continue
                part["all", c][s] += row; nbits[c] += 1
                if l.startswith("pawn_shield"): part["shield", c][s] += row
        r = {"fen": pos["fen"], "y": pos["y"], "bits w": nbits[True], "bits b": nbits[False]}
        for k in ("all", "shield"):
            W, B = part[k, chess.WHITE], part[k, chess.BLACK]
            v = value(torch.stack([full - W - B, full - B, full - W, full]))   # {}, {w}, {b}, {w,b}
            r[k] = {"w": float((v[1] - v[0] + v[3] - v[2]) / 2), "b": float((v[2] - v[0] + v[3] - v[1]) / 2),
                    "v none": float(v[0]), "v all": float(v[3])}
        r["facts"] = facts(b)
        res.append(r)
    return res

from positional import king_safety_data
def facts(b):
    """Board facts per king, signed so larger = safer: shield pawns, -attackers on the zone, flight squares."""
    d = king_safety_data(b)
    return {side: [len(d[side]["shield"]), -len(d[side]["enemy_pieces_bearing_on_the_zone"]), len(d[side]["flight_squares"])]
            for side in ("White", "Black")}

schema = json.loads(__import__("subprocess").run([ROOT / "features/square/target/release/square", "--schema"],
                                                 capture_output=True, text=True, check=True).stdout)
ckcol = {schema["labels"][bi]: i for i, bi in enumerate(ck["keep"])}

corpus = run(f"square_attribution_n{a.n}{tag}_ig_games.js")
books = run(f"square_attribution_n{a.n}{tag}_ig_textbook.js") + run(f"square_attribution_n{a.n}{tag}_ig_textbook2.js")
json.dump({"tap": a.tap, "corpus": corpus, "textbook": books}, open(ROOT / f"lc0-probe/results_square_king_shap{tag}.json", "w"))

print(f"{len(corpus)} corpus positions; bits per king: median {np.median([r['bits w'] for r in corpus]):.0f} (White), "
      f"{np.median([r['bits b'] for r in corpus]):.0f} (Black)")
for k in ("shield", "all"):
    w, bl = np.array([r[k]["w"] for r in corpus]), np.array([r[k]["b"] for r in corpus])
    edge = w + bl
    print(f"\n[{k}] |phi| median: White king {np.median(abs(w)):.3f}, Black king {np.median(abs(bl)):.3f}; "
          f"edge median |.| {np.median(abs(edge)):.3f}; saturated v(all) {np.mean([abs(r[k]['v all']) > .98 for r in corpus]):.0%}")
    for i, name in enumerate(["shield pawns", "-attackers", "flight squares"]):
        fw = np.array([r["facts"]["White"][i] for r in corpus]); fb = np.array([r["facts"]["Black"][i] for r in corpus])
        rw, rb, re = spearmanr(w, fw)[0], spearmanr(-bl, fb)[0], spearmanr(edge, fw - fb)[0]
        nz = (fw - fb) != 0
        agree = np.mean(np.sign(edge[nz]) == np.sign((fw - fb)[nz]))
        print(f"  {name:15s} rho White king {rw:+.2f}, Black king {rb:+.2f}, edge vs difference {re:+.2f}, sign agreement {agree:.2f} (n {nz.sum()})")
    vote = np.array([sum(np.sign(np.array(r["facts"]["White"]) - np.array(r["facts"]["Black"]))) for r in corpus])
    nz = vote != 0
    print(f"  vote of the three: sign agreement {np.mean(np.sign(edge[nz]) == np.sign(vote[nz])):.2f} (n {nz.sum()})")
