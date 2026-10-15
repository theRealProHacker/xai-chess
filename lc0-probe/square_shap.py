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
a = ap.parse_args()
torch.set_num_threads(a.threads)
rng = np.random.default_rng(0)
tag = a.tap.replace("/", "_")

ck = torch.load(ROOT / f"lc0-probe/data/square_decoder{tag}_n{a.n}.pt", weights_only=False)
keep, mu, sd = ck["keep"], torch.from_numpy(ck["mu"]).float(), float(ck["sd"])
schema = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                   capture_output=True, text=True, check=True).stdout)
col = {schema["labels"][b]: i for i, b in enumerate(keep)}                 # label -> decoder input row
E, bias = ck["state"]["inp.weight"], ck["state"]["bias"]
out = torch.nn.Linear(*ck["state"]["out.weight"].shape[::-1])
out.load_state_dict({"weight": ck["state"]["out.weight"], "bias": ck["state"]["out.bias"]})
net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(net_path))).extract_model([a.tap], ["/output/wdl"])
tail = convert(sub).eval()
for p in tail.parameters(): p.requires_grad_(False)
base = E[len(keep):] + bias                                                # (64, hidden): square one-hot only

@torch.no_grad()
def value(P):
    """(B, 64, hidden) pre-activations -> (B,) P(win) - P(loss), in chunks."""
    ys = []
    for s in range(0, len(P), 128):
        w = tail((out(torch.nn.functional.gelu(P[s:s + 128])) * sd + mu).reshape(-1, mu.shape[0]))
        ys.append(w[:, 0] - w[:, 2])
    return torch.cat(ys)

src = ROOT / f"lc0-probe/square_attribution_n{a.n}{tag}_ig_games.js"      # positions, their features and IG
A = json.loads(src.read_text().split("=", 1)[1].rstrip().rstrip(";"))
fams = sorted({l.split(".")[0] for l in A["labels"]})
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
    bits = [(chess.parse_square(sq), A["labels"][j], ig) for sq, q in pos["squares"].items() for j, ig in q["bits"]]
    rows = torch.stack([E[col[l]] for _, l, _ in bits])                    # (bits, hidden)
    sqs = torch.tensor([s for s, _, _ in bits])
    full = base.index_add(0, sqs, rows)
    y = float(value(full[None])[0])
    cand = np.argsort([-abs(ig) for _, _, ig in bits])[:a.cand]            # targets: candidates by |IG|, tested alone
    P = full[None].repeat(len(cand), 1, 1)
    P[torch.arange(len(cand)), sqs[cand]] -= rows[cand]
    dy = value(P).numpy() - y
    hit = [c for c, d in zip(cand, dy) if abs(d) > a.tau]
    hv = set(hit)
    fam = np.array([fams.index(l.split(".")[0]) for _, l, _ in bits])
    def parts(skip):
        return [torch.zeros_like(base).index_add(0, sqs[m], rows[m]) for f in range(F)
                for m in [torch.from_numpy(np.flatnonzero((fam == f) & ~np.isin(np.arange(len(bits)), list(skip))))]]
    fixed = base.index_add(0, sqs[list(hit)], rows[list(hit)]) if hit else base
    r = {"fen": pos["fen"], "y": pos["y"], "y decoded": y,
         "v none": float(value(base[None])[0]), "v targets": float(value(fixed[None])[0]),
         "targets": [[chess.square_name(bits[c][0]), bits[c][1], round(float(d), 4)] for c, d in zip(cand, dy) if c in hv],
         "phi": shapley(parts(set()), base).round(5).tolist(),
         "phi targets held": shapley(parts(hv), fixed).round(5).tolist()}
    res.append(r)
    if (i + 1) % 10 == 0: print(f"{i + 1} positions ({time.time() - t0:.0f}s)", flush=True)

out_ = {"tap": a.tap, "perms": a.perms, "tau": a.tau, "cand": a.cand, "families": fams, "positions": res}
json.dump(out_, open(ROOT / f"lc0-probe/results_square_shap{tag}.json", "w"))
(ROOT / f"lc0-probe/square_shap{tag}.js").write_text("window.SHAP = " + json.dumps(out_, separators=(",", ":")) + ";\n")   # for square_shap.html
tg = [t for r in res for t in r["targets"]]
print(f"{len(res)} positions, {len(tg)} targets ({len(tg) / len(res):.1f} per position), by family:",
      dict(sorted(((f, sum(t[1].split('.')[0] == f for t in tg)) for f in fams if any(t[1].split('.')[0] == f for t in tg)),
                  key=lambda x: -x[1])))
share = np.mean([abs(r["v targets"] - r["v none"]) / max(abs(r["y decoded"] - r["v none"]), 1e-6) for r in res])
print(f"mean |v(targets) - v(none)| / |v(all) - v(none)|: {share:.2f}")
for key in ["phi", "phi targets held"]:
    m = np.mean([np.abs(r[key]) for r in res], 0)
    print(f"mean |{key}| by family:", ", ".join(f"{fams[f]} {m[f]:.3f}" for f in np.argsort(-m)[:12]))
