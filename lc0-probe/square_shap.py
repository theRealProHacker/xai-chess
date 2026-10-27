"""Shapley values of the feature families, through a per-square decoder and Leela, with high-value targets held fixed.

Players are the feature families (a label's first part, e.g. pawn_cover), each switched on or off on all 64
squares at once. v(S) = P(win) - P(loss) of the rest of the net on the decoded tokens with only the families
in S on (the square one-hot always on). Sampled by permutations, in antithetic pairs; the values sum to
v(all) - v(none).

High-value targets: single features whose removal alone moves the value by more than --tau, such as an
occupant bit, which deletes a piece. Candidates are each position's --cand features with the largest
|integrated gradient| (from square_attribution.py --ig); each is switched off alone and kept if |dy| > tau.
The targets then stay on in every coalition, so the Shapley values split only what is left: v(all) - v(targets).
Both runs, with and without the targets held, are written, so they compare.

  python lc0-probe/square_shap.py --tap /encoder0/ln2 --perms 32
"""
import argparse, json, subprocess, sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, torch
from onnx2torch import convert

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--tap", default="/encoder0/ln2")
ap.add_argument("--perms", type=int, default=32, help="permutations per position and run (even: antithetic pairs)")
ap.add_argument("--tau", type=float, default=0.2, help="|dy| from removing one feature that makes it a high-value target")
ap.add_argument("--cand", type=int, default=64, help="features per position tested as targets, by |integrated gradient|")
ap.add_argument("--positions", type=int, default=300)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--src", help="positions from this square_attribution.py --ig output instead of the pgnlib games one")
ap.add_argument("--suffix", default="", help="appended to the output file names")
ap.add_argument("--hybrid", help="hybrid_decoder.py checkpoint: features enter at L0 and are added at its later taps")
ap.add_argument("--colour", default="white", choices=["white", "black"], help="with --hybrid: run each (White-to-move) "
                "position as stored, or colour-swapped with Black to move, features in real colours")
ap.add_argument("--split", default="", help="comma-separated label prefixes, e.g. pawn.centre: each value under one is its own player")
a = ap.parse_args()
torch.set_num_threads(a.threads)
rng = np.random.default_rng(0)
tag = a.tap.replace("/", "_")

schema = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                   capture_output=True, text=True, check=True).stdout)
net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
inferred = onnx.shape_inference.infer_shapes(onnx.load(net_path))
def seg(i, o):
    m = convert(onnx.utils.Extractor(inferred).extract_model([i], [o])).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m
def head(d):
    """Decoder output layer, normalized units -> the tap's units."""
    out = torch.nn.Linear(*d["state"]["out.weight"].shape[::-1])
    out.load_state_dict({"weight": d["state"]["out.weight"], "bias": d["state"]["out.bias"]})
    mu_, sd_ = torch.as_tensor(d["mu"]).float(), float(d["sd"])
    return lambda h: out(torch.nn.functional.gelu(h)) * sd_ + mu_
swap = lambda l: l                                                         # stored label -> real-colour label
if a.hybrid:                                                               # all decoders' rows side by side: a bit
    ck = torch.load(a.hybrid, weights_only=False)                          # switches in every decoder at once
    keep, ds = ck["keep"], ck["decoders"]
    E = torch.cat([d["state"]["inp.weight"] for d in ds], 1)
    bias = torch.cat([d["state"]["bias"] for d in ds])
    W, heads = np.cumsum([0] + [d["hidden"] for d in ds]), [head(d) for d in ds]
    segs = [seg(t, u) for t, u in zip(ck["taps"], ck["taps"][1:] + ["/output/wdl"])]
    def decode(P):
        S = None
        for k, h in enumerate(heads):
            x = h(P[..., W[k]:W[k + 1]]).reshape(-1, 256)
            S = x if S is None else segs[k - 1](S) + x
        return segs[-1](S)
    t = np.arange(64)
    if a.colour == "black":                                                # token t (Leela's frame) is real square t ^ 56
        from colour_frame import label_map
        M = label_map()
        lab = {l: schema["labels"][M[i]] for i, l in enumerate(schema["labels"])}
        swap = lambda l: lab[l]
        base = E[ck["sq row"] + (t ^ 56)] + E[ck["stm row"]] + bias
    else:
        base = E[ck["sq row"] + t] + bias
    otag = f"{tag}_hybrid" + ("_black" if a.colour == "black" else "")
else:
    ck = torch.load(ROOT / f"lc0-probe/data/square_decoder{tag}_n{a.n}.pt", weights_only=False)
    keep, E, bias = ck["keep"], ck["state"]["inp.weight"], ck["state"]["bias"]
    h0, tail = head(ck), seg(a.tap, "/output/wdl")
    decode = lambda P: tail(h0(P).reshape(-1, 256))
    base = E[len(keep):len(keep) + 64] + bias                              # (64, hidden): square one-hot only
    otag = tag
col = {schema["labels"][b]: i for i, b in enumerate(keep)}                 # label -> decoder input row

@torch.no_grad()
def value(P):
    """(B, 64, hidden) pre-activations -> (B,) P(win) - P(loss), in chunks."""
    ys = []
    for s in range(0, len(P), 128):
        w = decode(P[s:s + 128])
        ys.append(w[:, 0] - w[:, 2])
    return torch.cat(ys)

src = Path(a.src) if a.src else ROOT / f"lc0-probe/square_attribution_n{a.n}{tag}_ig_games.js"      # positions, their features and IG
A = json.loads(src.read_text().split("=", 1)[1].rstrip().rstrip(";"))
SPLIT = [x for x in a.split.split(",") if x]
def family(l):
    """A label's player: its first part, or for a --split prefix the prefix plus the next part (pawn.centre.open)."""
    for x in SPLIT:
        if l.startswith(x + "."): return ".".join(l.split(".")[:x.count(".") + 2])
    return l.split(".")[0]
fams = sorted({family(l) for l in A["labels"]} | {family(swap(l)) for l in A["labels"]})
F = len(fams)

def shapley(parts, fixed):
    """Permutation-sampled Shapley values of the F families; parts[f] = (64, hidden) family f's summed rows."""
    phi = np.zeros(F)
    for k in range(a.perms // 2):
        p = rng.permutation(F)
        for order in (p, p[::-1]):                                         # antithetic pair
            steps = torch.cumsum(torch.stack([torch.zeros_like(base)] + [parts[f] for f in order]), 0)
            v = value(fixed + steps).numpy()
            phi[order] += np.diff(v)
    return phi / (a.perms // 2 * 2)

res, t0 = [], time.time()
for i, pos in enumerate(A["positions"][:a.positions]):
    bits = [(chess.parse_square(sq), swap(A["labels"][j]), ig) for sq, q in pos["squares"].items() for j, ig in q["bits"]]
    bits = [b for b in bits if b[1] in col]                                # a hybrid's keep can differ from the source's
    rows = torch.stack([E[col[l]] for _, l, _ in bits])                    # (bits, hidden); placed at Leela's token
    sqs = torch.tensor([s for s, _, _ in bits])
    full = base.index_add(0, sqs, rows)
    y = float(value(full[None])[0])
    cand = np.argsort([-abs(ig) for _, _, ig in bits])[:a.cand]            # targets: candidates by |IG|, tested alone
    P = full[None].repeat(len(cand), 1, 1)
    P[torch.arange(len(cand)), sqs[cand]] -= rows[cand]
    dy = value(P).numpy() - y
    hit = [c for c, d in zip(cand, dy) if abs(d) > a.tau]
    hv = set(hit)
    fam = np.array([fams.index(family(l)) for _, l, _ in bits])
    def parts(skip):
        return [torch.zeros_like(base).index_add(0, sqs[m], rows[m]) for f in range(F)
                for m in [torch.from_numpy(np.flatnonzero((fam == f) & ~np.isin(np.arange(len(bits)), list(skip))))]]
    fixed = base.index_add(0, sqs[list(hit)], rows[list(hit)]) if hit else base
    r = {"fen": pos["fen"] if a.colour == "white" else chess.Board(pos["fen"]).mirror().fen(), "y": pos["y"], "y decoded": y,
         "v none": float(value(base[None])[0]), "v targets": float(value(fixed[None])[0]),
         "targets": [[chess.square_name(bits[c][0] ^ (56 if a.colour == "black" else 0)), bits[c][1], round(float(d), 4)] for c, d in zip(cand, dy) if c in hv],
         "phi": shapley(parts(set()), base).round(5).tolist(),
         "phi targets held": shapley(parts(hv), fixed).round(5).tolist()}
    res.append(r)
    if (i + 1) % 10 == 0: print(f"{i + 1} positions ({time.time() - t0:.0f}s)", flush=True)

out_ = {"tap": a.tap, "hybrid": a.hybrid, "colour": a.colour, "perms": a.perms, "tau": a.tau, "cand": a.cand, "families": fams, "positions": res}
json.dump(out_, open(ROOT / f"lc0-probe/results_square_shap{otag}{a.suffix}.json", "w"))
(ROOT / f"lc0-probe/square_shap{otag}{a.suffix}.js").write_text("window.SHAP = " + json.dumps(out_, separators=(",", ":")) + ";\n")   # for square_shap.html
tg = [t for r in res for t in r["targets"]]
print(f"{len(res)} positions, {len(tg)} targets ({len(tg) / len(res):.1f} per position), by family:",
      dict(sorted(((f, sum(t[1].split('.')[0] == f for t in tg)) for f in fams if any(t[1].split('.')[0] == f for t in tg)),
                  key=lambda x: -x[1])))
share = np.mean([abs(r["v targets"] - r["v none"]) / max(abs(r["y decoded"] - r["v none"]), 1e-6) for r in res])
print(f"mean |v(targets) - v(none)| / |v(all) - v(none)|: {share:.2f}")
for key in ["phi", "phi targets held"]:
    m = np.mean([np.abs(r[key]) for r in res], 0)
    print(f"mean |{key}| by family:", ", ".join(f"{fams[f]} {m[f]:.3f}" for f in np.argsort(-m)[:12]))
