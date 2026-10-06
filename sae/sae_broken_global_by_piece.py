"""Named + free dictionary for one Leela layer.

Per square token x (256-d) of a position:  x ~ W_named[sq] @ y + W_dec @ topk(W_enc @ r)
  y      ~1000 named bits of the position: the board (12 piece planes x 64 squares, castling,
         side to move, as Leela sees them) and the binary features from features/ (Rust)
  r      the residual x - W_named[sq] @ y, encoded by a TopK SAE with k free latents

Stage 1 fits W_named by least squares (per square, so the board planes can map to the right
token). Stage 2 trains the free SAE on what is left. Reported: fraction of variance unexplained
(FVU) on held-out positions, for named only and for named + k free latents.

  python sae/sae_broken_global_by_piece.py --n 20000 --tap /encoder5/ln2 --k 0,4,16,64
"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import numpy as np, onnx, onnxruntime as ort
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000, help="positions (first n of sf-probe/data/fens.txt)")
ap.add_argument("--tap", default="/encoder5/ln2", help="ONNX node whose output is the residual")
ap.add_argument("--k", default="0,4,16,64", help="free latent counts to sweep")
ap.add_argument("--active", default="4", help="free latents active per token (TopK), comma list to sweep")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--out", default=str(ROOT / "sae" / "out"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:a.n]
N = len(fens)

# --- named bits, part 1: the board as Leela's input planes see it (side to move = "us")
planes = encode_fens(fens)                                       # (N, 112, 8, 8)
board = np.concatenate([planes[:, :12].reshape(N, -1), planes[:, 104:109, 0, 0]], 1)

# --- named bits, part 2: the Rust features, sides turned into us/them like the board
run = subprocess.run(["cargo", "run", "--release", "-q", "--bin", "features", "--", "--all"],
                     cwd=ROOT / "features", input="\n".join(fens) + "\n", capture_output=True, text=True, check=True)
rows = [json.loads(l) for l in run.stdout.splitlines()]
assert len(rows) == N and not any("error" in r for r in rows)
def key(f, fen):
    if f["side"] is None: return f["id"]
    us = "white" if fen.split()[1] == "w" else "black"
    return f["id"] + (".us" if f["side"] == us else ".them")
# ponytail: initiative is never on (no engine line passed); add PVs to the stdin lines if wanted
on = [{key(f, r["fen"]) for f in r["features"] if f["on"]} for r in rows]
names = sorted({key(f, r["fen"]) for r in rows for f in r["features"]})
col = {n: i for i, n in enumerate(names)}
feats = np.zeros((N, len(names)), np.float32)
for i, s in enumerate(on):
    feats[i, [col[n] for n in s]] = 1
Y = np.concatenate([board, feats, np.ones((N, 1), np.float32)], 1)   # last column: bias
print(f"{N:,} positions, {Y.shape[1] - 1} named bits ({board.shape[1]} board + {len(names)} features)", flush=True)

# --- the layer's residual, all 64 tokens
m = onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx")
m.graph.output.append(onnx.ValueInfoProto(name=a.tap))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
X = np.concatenate([sess.run([a.tap], {"/input/planes": planes[s:s + 256]})[0].reshape(-1, 64, 256)
                    for s in range(0, N, 256)])                     # (N, 64, D)
D = X.shape[-1]
del planes

te = rng.random(N) < 0.2                                             # held-out positions
def fvu(x, xh):
    return float(((x - xh) ** 2).sum() / ((x - x.mean((0, 1))) ** 2).sum())

# --- stage 1: named decoder, least squares per square (ridge for the never-on bits)
lam = 1e-2 * len(Y[~te])
G = Y[~te].T @ Y[~te] + lam * np.eye(Y.shape[1])
W = np.stack([np.linalg.solve(G, Y[~te].T @ X[~te, s]) for s in range(64)])   # (64, named+1, D)
named = np.matmul(Y[None], W).transpose(1, 0, 2)                    # (N, 64, D)
R = (X - named).reshape(N, 64, D)
print(f"named only   FVU train {fvu(X[~te], named[~te]):.3f}  test {fvu(X[te], named[te]):.3f}", flush=True)

# --- stage 2: TopK SAE on the residual, numpy + Adam (no torch here)
def topk(h, k):
    if k >= h.shape[1]: return h
    idx = np.argpartition(-h, k - 1, 1)[:, k:]
    h = h.copy(); np.put_along_axis(h, idx, 0, 1); return h

def train(Rtr, k, act):
    Wd = Rtr[rng.choice(len(Rtr), k)].T.copy()
    Wd /= np.linalg.norm(Wd, axis=0) + 1e-8
    P = {"We": Wd.T.copy(), "be": np.zeros(k), "Wd": Wd, "bd": Rtr.mean(0)}
    M = {n: np.zeros_like(v) for n, v in P.items()}; V = {n: np.zeros_like(v) for n, v in P.items()}
    for t in range(1, a.steps + 1):
        x = Rtr[rng.integers(0, len(Rtr), 4096)]
        pre = (x - P["bd"]) @ P["We"].T + P["be"]
        z = topk(np.maximum(pre, 0), act)
        err = z @ P["Wd"].T + P["bd"] - x                            # d loss / d xhat (up to 2/B)
        dz = (err @ P["Wd"]) * (z > 0)
        # ponytail: bd's gradient through the encoder input is dropped; the decoder path fits it
        g = {"Wd": err.T @ z, "bd": err.sum(0), "We": dz.T @ (x - P["bd"]), "be": dz.sum(0)}
        for n in P:
            gn = g[n] / len(x)
            M[n] = 0.9 * M[n] + 0.1 * gn; V[n] = 0.999 * V[n] + 0.001 * gn ** 2
            P[n] -= 1e-3 * (M[n] / (1 - 0.9 ** t)) / (np.sqrt(V[n] / (1 - 0.999 ** t)) + 1e-8)
        P["Wd"] /= np.linalg.norm(P["Wd"], axis=0) + 1e-8
    return P

def recon(P, r, act):
    return topk(np.maximum((r - P["bd"]) @ P["We"].T + P["be"], 0), act) @ P["Wd"].T + P["bd"]

res = {"tap": a.tap, "n": N, "named": int(Y.shape[1] - 1),
       "fvu": {"0": fvu(X[te], named[te])}}
for k in [int(s) for s in a.k.split(",") if int(s) > 0]:
    for act in sorted({min(k, int(s)) for s in a.active.split(",")}):
        P = train(R[~te].reshape(-1, D), k, act)
        xh = named[te] + recon(P, R[te].reshape(-1, D), act).reshape(-1, 64, D)
        res["fvu"][f"{k}/{act}"] = fvu(X[te], xh)
        print(f"named + {k:>4} free (top {act:>3})   FVU test {res['fvu'][f'{k}/{act}']:.3f}", flush=True)
        np.savez(out / f"free_k{k}_a{act}.npz", **P)
np.savez(out / "named.npz", W=W, names=np.array([f"board{i}" for i in range(board.shape[1])] + names + ["bias"]))
json.dump(res, open(out / "results.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
for k in sorted({int(key.split("/")[0]) for key in res["fvu"] if "/" in key}):
    pts = sorted((int(key.split("/")[1]), v) for key, v in res["fvu"].items() if key.startswith(f"{k}/"))
    ax.plot(*zip(*pts), "o-", label=f"{k} free latents")
ax.axhline(res["fvu"]["0"], color="grey", ls="--", label="named only")
ax.set(xscale="log", xlabel="free latents active per token", ylabel="unexplained variance (held out)",
       title=f"{a.tap}, {N:,} positions"); ax.legend(); fig.tight_layout(); fig.savefig(out / "fvu.png", dpi=150)

# ponytail: FVU only; patching the reconstruction back into Leela (policy KL) needs the graph cut at
# the tap, add when the FVU curve says the free part is worth a causal check
