"""hybrid_ig_demo.js from 10 random held-out game positions (test split, White to move): per-feature integrated gradients
of W - L through L0 + L2 + L5 and both L5 + L2 + L0 (end MLPs without and with the prediction input), from each square's mean bits over all
training boards, or over the boards where it is empty when it is empty here.

  python lc0-probe/hybrid_ig_demo_random.py [--n 10] [--seed 2] [--steps 128]
"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lc0-probe")]
from hybrid_stacks import build
import numpy as np, onnxruntime as ort, torch
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
H, Hp = build(pred=False), build(pred=True)
te = np.flatnonzero(np.random.default_rng(0).random(100000) < 0.2)       # hybrid_endmlp.py's test split
pick = set(np.random.default_rng(a.seed).choice(te, a.n, replace=False).tolist())
rows = {}
with open(ROOT / "lc0-probe/data/square_bits_games_n100000_nodes200000.jsonl") as f:
    for i, l in enumerate(f):
        if i in pick: rows[i] = json.loads(l)
        if len(rows) == a.n: break
LAB = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                capture_output=True, text=True, check=True).stdout)["labels"]
full = ort.InferenceSession(str(ROOT / "vendor/lc0-nets/t1-256x10.onnx"), providers=["CPUExecutionProvider"])
occ = [j for j, k in enumerate(H.keep) if LAB[k].startswith("occupant.")]
te_m, va_m = np.random.default_rng(0).random(100000) < 0.2, np.random.default_rng(1).random(100000) < 0.1
SE, ne, n = torch.zeros(64, H.D_in), torch.zeros(64), 0                  # per square: mean bits where it is empty
with open(ROOT / "lc0-probe/data/square_bits_games_n100000_nodes200000.jsonl") as f:   # the same boards as H.SM
    for i, l in enumerate(f):
        if n == 20000: break
        if te_m[i] or va_m[i]: continue
        Xt = H.xs(json.loads(l)["squares"]); e = Xt[:, occ].sum(1) == 0
        SE[e] += Xt[e]; ne += e; n += 1
SE /= ne[:, None]
NTOP, res, dist = 30, [], []
for i in sorted(rows):
    r = rows[i]; X = H.xs(r["squares"])
    SM = torch.where((X[:, occ].sum(1) == 0)[:, None], SE, H.SM)
    w = full.run(["/output/wdl"], {"/input/planes": encode_fens([r["fen"]])})[0][0]
    row, As = {"fen": r["fen"], "y": float(w[0] - w[2])}, {}
    for name, f in (("fwd", H.forward_h), ("deep", H.deep_h), ("pred", Hp.deep_h)):            # IG over Z = X @ Wcat equals IG over X
        al = ((torch.arange(a.steps) + 0.5) / a.steps)[:, None, None]
        Z0, dZ = SM @ H.Wcat, (X - SM) @ H.Wcat
        P = (Z0 + al * dZ).requires_grad_(True)
        H.wl(f(P)).sum().backward()
        A = ((X - SM) * (P.grad.mean(0) @ H.Wcat.T))[:, :H.K]
        with torch.no_grad(): y, y0 = float(H.wl(f((X @ H.Wcat)[None]))[0]), float(H.wl(f(Z0[None]))[0])
        print(f"{name:4} {i}: y {y:+.3f} (Leela {row['y']:+.3f}) baseline {y0:+.3f} gap {abs(float(A.sum()) - (y - y0)):.4f}", flush=True)
        row[name] = {"y": y, "y0": y0}; As[name] = A
    d = {}
    for p_, q_ in (("fwd", "deep"), ("fwd", "pred"), ("deep", "pred")):    # all features, not just the shown ones
        u, v = As[p_], As[q_]
        d[f"{p_}-{q_}"] = {"L1": float((u - v).abs().sum()), "rel L1": float((u - v).abs().sum() / (.5 * (u.abs().sum() + v.abs().sum()))),
                           "cos": float(torch.nn.functional.cosine_similarity(u.flatten(), v.flatten(), 0))}
    dist.append(d)
    top = set()
    St = torch.stack([As[k] for k in ("fwd", "deep", "pred")])
    for v in (*St, St.amax(0) - St.amin(0)):
        m = v.abs().numpy(); top |= set(map(tuple, np.argwhere(m >= np.sort(m, None)[-NTOP])))
    row["feat"] = sorted(({"sq": int(s), "f": LAB[H.keep[b]], "on": bool(X[s, b] > 0),
                           "fwd": round(float(As["fwd"][s, b]), 4), "deep": round(float(As["deep"][s, b]), 4), "pred": round(float(As["pred"][s, b]), 4)}
                          for s, b in top), key=lambda q: -(max(q["fwd"], q["deep"], q["pred"]) - min(q["fwd"], q["deep"], q["pred"])))
    res.append(row)
out = ROOT / "lc0-probe/hybrid_ig_demo.js"
out.write_text("window.HYB = " + json.dumps(res) + ";\n"); print("wrote", out)
mean = {k: {m: float(np.mean([d[k][m] for d in dist])) for m in dist[0][k]} for k in dist[0]}
for k, v in mean.items(): print(f"{k:10} " + "  ".join(f"{m} {x:.3f}" for m, x in v.items()))
json.dump({"positions": sorted(rows), "per position": dist, "mean": mean},
          open(ROOT / "lc0-probe/results_hybrid_ig_distance.json", "w"), indent=1)
