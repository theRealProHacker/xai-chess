"""Which feature families the base deepest-first stack (L5 = C + D5 + M2 + M0, no prediction input) credits through
each layer's decoder. IG of W - L (side to move) from the any-occupant mean bits, on held-out game positions.

The path runs in the decoders' first layer Z = X @ Wcat, so dy/dX splits exactly into the D5, L2 and L0 decoders'
parts: A_k = (X - SM) * (dy/dZ_k @ W_k^T), sum_k A_k = the full attribution. Per position, layer and family: sum |A_k|
over the family's bits (absent ones included) and the signed sum.

  python lc0-probe/hybrid_layer_families.py [--n 300] [--steps 128]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lc0-probe")]
from hybrid_stacks import build
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300, help="held-out game positions")
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--stacks", default="deep", help="comma-separated: deep (L5+L2+L0), fwd (L0+L2+L5), gate (the gate run), l0, l2, l5")
a = ap.parse_args()
torch.set_num_threads(a.threads)
GATE = "gate" in a.stacks.split(",")
assert not (GATE and "deep" in a.stacks.split(",")), "gate and deep need different end MLPs: run them separately"
H = build(GATE, gate=GATE)                                                   # gate: endmlp_pred_gate's end MLPs in deep_h
LAB = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                capture_output=True, text=True, check=True).stdout)["labels"]
fams = sorted({LAB[k].split(".")[0] for k in H.keep})
fam = torch.tensor([fams.index(LAB[k].split(".")[0]) for k in H.keep])
F = len(fams)
one = lambda f: (lambda Z: H.tail(f(H.split(Z))).reshape(Z.shape[0], 3))
STACKS = {"deep": (H.deep_h, {"L5": "R5", "L2": "L2", "L0": "F0"}),         # f(Z), layer: its decoder
          "fwd": (H.forward_h, {"L5": "F5", "L2": "F2", "L0": "F0"}),
          "l0": (one(lambda d: H.s05(d["F0"])), {"L0": "F0"}),
          "l2": (one(lambda d: H.s35(d["L2"])), {"L2": "L2"}),
          "gate": (H.deep_h, {"L5": "R5", "L2": "L2", "L0": "F0"}),            # endmlp_pred_gate: same decoders, gated end MLPs
          "l5": (one(lambda d: d["R5"] + H.C.repeat(len(d["R5"]) // 64, 1)), {"L5": "R5"})}
names = list(H.dec)
te = np.random.default_rng(0).random(100000) < 0.2
Xs = []
with open(ROOT / "lc0-probe/data/square_bits_games_n100000_nodes200000.jsonl") as f:
    for i, l in enumerate(f):
        if te[i]: Xs.append(H.xs(json.loads(l)["squares"]))
        if len(Xs) == a.n: break
def run(name):
    fn, LAYERS = STACKS[name]
    absA, sgnA, gaps, out = np.zeros((a.n, len(LAYERS), F)), np.zeros((a.n, len(LAYERS), F)), [], []
    t0 = time.time()
    for p, X in enumerate(Xs):
        al = ((torch.arange(a.steps) + 0.5) / a.steps)[:, None, None]
        Z0, dZ = H.SM @ H.Wcat, (X - H.SM) @ H.Wcat
        P = (Z0 + al * dZ).requires_grad_(True)
        H.wl(fn(P)).sum().backward()
        g = P.grad.mean(0)
        with torch.no_grad(): y, y0 = float(H.wl(fn((X @ H.Wcat)[None]))[0]), float(H.wl(fn(Z0[None]))[0])
        tot = 0.0
        for j, (L, k) in enumerate(LAYERS.items()):
            i = names.index(k); lo, hi = H.cuts[i], H.cuts[i + 1]
            A = ((X - H.SM) * (g[:, lo:hi] @ H.Wcat[:, lo:hi].T))[:, :H.K]        # (64, K)
            tot += float(A.sum())
            absA[p, j] = torch.zeros(F).index_add_(0, fam, A.abs().sum(0)).numpy()
            sgnA[p, j] = torch.zeros(F).index_add_(0, fam, A.sum(0)).numpy()
        gaps.append(abs(tot - (y - y0)))
        if p % 25 == 0: print(f"{p + 1}/{a.n} gap {gaps[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    S = absA.sum(0)                                                              # (layers, F)
    res = {"n": a.n, "steps": a.steps, "families": fams, "layers": list(LAYERS), "max gap": max(gaps),
           "abs": absA.tolist(), "signed": sgnA.tolist()}
    json.dump(res, open(ROOT / f"lc0-probe/results_hybrid_layer_families_games{'' if name == 'deep' else '_' + name}.json", "w"))
    print(f"\n== {name}: {a.n} positions, max completeness gap {max(gaps):.4f}, median {np.median(gaps):.4f}")
    print("layer share of all |A|: " + ", ".join(f"{L} {S[j].sum() / S.sum():.1%}" for j, L in enumerate(LAYERS)))
    print(f"\n{'family':16}{'all':>8}" + "".join(f"{L + ' col':>9}" for L in LAYERS) + "".join(f"{'in ' + L:>8}" for L in LAYERS)
          + f"{'net/abs':>9}")
    for f_ in np.argsort(-S.sum(0)):
        row = S[:, f_]
        if row.sum() == 0: continue
        print(f"{fams[f_]:16}{row.sum() / S.sum():8.1%}" + "".join(f"{row[j] / S[j].sum():9.1%}" for j in range(len(LAYERS)))
              + "".join(f"{row[j] / row.sum():8.1%}" for j in range(len(LAYERS)))
              + f"{np.abs(sgnA[:, :, f_].sum(1)).sum() / max(absA[:, :, f_].sum(), 1e-12):9.2f}")
    N_ = np.abs(sgnA).sum(0)                                                     # (layers, F): |family's signed sum| per position
    print(f"\nby |signed family sum| per position; layer share: " + ", ".join(f"{L} {N_[j].sum() / N_.sum():.1%}" for j, L in enumerate(LAYERS)))
    print(f"{'family':16}{'all':>8}" + "".join(f"{L + ' col':>9}" for L in LAYERS) + "".join(f"{'in ' + L:>8}" for L in LAYERS))
    for f_ in np.argsort(-N_.sum(0)):
        row = N_[:, f_]
        if row.sum() == 0: continue
        print(f"{fams[f_]:16}{row.sum() / N_.sum():8.1%}" + "".join(f"{row[j] / N_[j].sum():9.1%}" for j in range(len(LAYERS)))
              + "".join(f"{row[j] / row.sum():8.1%}" for j in range(len(LAYERS))))

for name in a.stacks.split(","): run(name)
