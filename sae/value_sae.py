"""Named + free dictionary for Leela's value bottleneck.

Per position, h = /value/dense1/mish (128-d): the last vector before the WDL logits (dense2, linear,
then softmax). h ~ W_named @ y + W_dec @ topk(W_enc @ r)
  y      the binary features from features/ (Rust), sides turned into us/them; one row per position
  r      the residual h - W_named @ y, encoded by a TopK SAE with k free latents

Reported on held-out positions: FVU of h, and FVU of Q = P(win) - P(loss) after pushing the
reconstruction through dense2 + softmax.

  python sae/value_sae.py --n 150000 --k 16,64,256 --active 1,2,4,8
"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper
from lc0_planes import encode_fens

TAP = "/value/dense1/mish"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150000, help="positions (first n of sf-probe/data/fens.txt)")
ap.add_argument("--k", default="16,64,256", help="free latent counts to sweep")
ap.add_argument("--active", default="1,2,4,8", help="free latents active per position (TopK), comma list")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--out", default=str(ROOT / "sae" / "out" / "value"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:a.n]
N = len(fens)

cache = out.parent / f"value_data_{N}.npz"
m = onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx")
w2 = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
W2, b2 = w2["/value/dense2/matmul/w"], w2["/value/dense2/add/w"]           # (128, 3), (3,)
if cache.exists():
    c = np.load(cache); H, WDL, Y, names = c["H"], c["WDL"], c["Y"], list(c["names"])
else:
    # --- named bits: the Rust features, sides turned into us/them (Leela sees side to move as "us")
    run = subprocess.run(["cargo", "run", "--release", "-q", "--bin", "features", "--", "--all"],
                         cwd=ROOT / "features", input="\n".join(fens) + "\n", capture_output=True, text=True, check=True)
    on, names = [], set()
    for l in run.stdout.splitlines():
        r = json.loads(l); assert "error" not in r
        us = "white" if r["fen"].split()[1] == "w" else "black"
        ks = [f["id"] if f["side"] is None else f["id"] + (".us" if f["side"] == us else ".them") for f in r["features"]]
        names.update(ks); on.append([k for k, f in zip(ks, r["features"]) if f["on"]])
    del run
    assert len(on) == N
    names = sorted(names); col = {n: i for i, n in enumerate(names)}
    Y = np.zeros((N, len(names) + 1), np.float32); Y[:, -1] = 1                # last column: bias
    for i, s in enumerate(on):
        Y[i, [col[n] for n in s]] = 1

    # --- the value bottleneck, one 128-d vector per position
    m.graph.output.append(onnx.ValueInfoProto(name=TAP))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    H, WDL = [], []
    for s in range(0, N, 1024):
        h, wdl = sess.run([TAP, "/output/wdl"], {"/input/planes": encode_fens(fens[s:s + 1024])})
        H.append(h); WDL.append(wdl)
    H, WDL = np.concatenate(H).astype(np.float64), np.concatenate(WDL)
    np.savez(cache, H=H, WDL=WDL, Y=Y, names=np.array(names))
print(f"{N:,} positions, {len(names)} named bits", flush=True)
D = H.shape[1]

def q(h):
    z = h @ W2 + b2; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return p[:, 0] - p[:, 2]
assert np.allclose(q(H), WDL[:, 0] - WDL[:, 2], atol=1e-4)                  # dense2 + softmax reproduced

te = rng.random(N) < 0.2                                                     # held-out positions
def fvu(x, xh):
    return float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
def score(xh):
    qt, qh = q(H[te]), q(xh)
    return {"h": fvu(H[te], xh), "q": fvu(qt, qh),
            "mse_h": float(((H[te] - xh) ** 2).mean()), "var_h": float(H[te].var(0).mean()),
            "mae_q": float(np.abs(qt - qh).mean()), "mad_q": float(np.abs(qt - qt.mean()).mean())}
def show(label, sc):
    print(f"{label:<28} FVU h {sc['h']:.3f}  Q {sc['q']:.3f}   MSE h {sc['mse_h']:.4f} (var {sc['var_h']:.4f})"
          f"   mean |dQ| {sc['mae_q']:.3f} (mean |Q - mean| {sc['mad_q']:.3f})", flush=True)

# --- stage 1: named decoder, least squares (ridge for the never-on bits)
lam = 1e-2 * (~te).sum()
W = np.linalg.solve(Y[~te].T @ Y[~te] + lam * np.eye(Y.shape[1]), Y[~te].T @ H[~te])   # (named+1, D)
named = Y @ W
R = H - named
res = {"tap": TAP, "n": N, "named": len(names), "fvu": {"0": score(named[te])}}
show("named only", res["fvu"]["0"])

# --- the other direction, no new fit: the named bits read back out of h through the stage-1 decoder
Yh = (H[te] - W[-1]) @ np.linalg.pinv(W[:-1])                                # (test, named)
err = np.abs(Yh - Y[te, :-1])
res["readout"] = {"mae": float(err.mean()), "mae_on": float(err[Y[te, :-1] == 1].mean()),
                  "mae_off": float(err[Y[te, :-1] == 0].mean()), "acc": float(((Yh > .5) == Y[te, :-1]).mean())}
print(f"bits read from h  mean |error| {res['readout']['mae']:.3f}  (on bits {res['readout']['mae_on']:.3f}, "
      f"off bits {res['readout']['mae_off']:.3f})   thresholded at 0.5: {res['readout']['acc']:.1%} correct", flush=True)

# --- stage 2: TopK SAE on the residual, numpy + Adam
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
        z = topk(np.maximum((x - P["bd"]) @ P["We"].T + P["be"], 0), act)
        err = z @ P["Wd"].T + P["bd"] - x
        dz = (err @ P["Wd"]) * (z > 0)
        g = {"Wd": err.T @ z, "bd": err.sum(0), "We": dz.T @ (x - P["bd"]), "be": dz.sum(0)}
        for n in P:
            gn = g[n] / len(x)
            M[n] = 0.9 * M[n] + 0.1 * gn; V[n] = 0.999 * V[n] + 0.001 * gn ** 2
            P[n] -= 1e-3 * (M[n] / (1 - 0.9 ** t)) / (np.sqrt(V[n] / (1 - 0.999 ** t)) + 1e-8)
        P["Wd"] /= np.linalg.norm(P["Wd"], axis=0) + 1e-8
    return P

def recon(P, r, act):
    return topk(np.maximum((r - P["bd"]) @ P["We"].T + P["be"], 0), act) @ P["Wd"].T + P["bd"]

for k in [int(s) for s in a.k.split(",") if int(s) > 0]:
    for act in sorted({min(k, int(s)) for s in a.active.split(",")}):
        P = train(R[~te], k, act)
        sc = res["fvu"][f"{k}/{act}"] = score(named[te] + recon(P, R[te], act))
        show(f"named + {k} free (top {act})", sc)
        # each latent's push on the W and L logits: decoder direction through dense2
        np.savez(out / f"free_k{k}_a{act}.npz", **P, logit=P["Wd"].T @ W2)
np.savez(out / "named.npz", W=W, names=np.array(names + ["bias"]), logit=W @ W2)
json.dump(res, open(out / "results.json", "w"), indent=1)
