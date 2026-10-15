"""Leela's value head applied to layer-9 tokens decoded from the per-square features.

Takes data/decoded_encoder9_ln2.npy (square_decoder_mlp.py --taps /encoder9/ln2 --save), runs it through
the value head (weights read from the ONNX net, checked against onnxruntime on the real tokens) and
reports FVU of value/dense1 and value/dense2 on held-out positions. Baseline: each square's mean
train token, i.e. the square alone.

  python lc0-probe/value_via_head.py
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import numpy as np, onnx, onnx.numpy_helper as nh, onnxruntime as ort
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:a.n]
N = len(fens)
te = np.random.default_rng(0).random(N) < 0.2                              # same split as square_probe.py
TE = np.flatnonzero(te)

m = onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx")
W = {i.name: nh.to_array(i).astype(np.float64) for i in m.graph.initializer if i.name.startswith("/value/")}
taps = ["/encoder9/ln2", "/value/dense1/mish", "/value/dense2/add"]
for t in taps: m.graph.output.append(onnx.ValueInfoProto(name=t))
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
X = {t: [] for t in taps}
for s in range(0, N, 256):
    for t, x in zip(taps, sess.run(taps, {"/input/planes": encode_fens(fens[s:s + 256])})):
        X[t].append(x.astype(np.float32))
X = {t: np.concatenate(v) for t, v in X.items()}
tok = X["/encoder9/ln2"].reshape(N, 64, -1)

mish = lambda z: z * np.tanh(np.logaddexp(0, z))
def head(T):
    """(P, 64, 256) layer-9 tokens -> value/dense1 (after mish), value/dense2 (logits)."""
    e = mish(T.astype(np.float64) @ W["/value/embed/matmul/w"] + W["/value/embed/add/w"]).reshape(len(T), -1)
    d1 = mish(e @ W["/value/dense1/matmul/w"] + W["/value/dense1/add/w"])
    return d1, d1 @ W["/value/dense2/matmul/w"] + W["/value/dense2/add/w"]

real = dict(zip(taps[1:], head(tok)))
for t in taps[1:]:                                                         # the reimplementation matches onnxruntime
    err = np.abs(real[t] - X[t]).max()
    assert err < 1e-3, (t, err)
    print(f"head check {t}: max |diff| {err:.1e}")

def fvu(Y, P):
    return float(((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

dec = np.load(ROOT / "lc0-probe/data/decoded_encoder9_ln2.npy").reshape(N, 64, -1)
sq_mean = tok[~te].mean(0)                                                 # (64, 256)
res = {"n": N, "test positions": int(len(TE)), "token FVU": fvu(tok[TE].reshape(-1, tok.shape[2]), dec[TE].reshape(-1, tok.shape[2])),
       "taps": {}}
for name, T in [("decoded from features + square", dec[TE]), ("square only", np.broadcast_to(sq_mean, tok[TE].shape))]:
    for t, P in zip(taps[1:], head(T)):
        res["taps"].setdefault(t, {})[name] = fvu(X[t][TE], P)
        print(f"{t:20} {name:32} test FVU {res['taps'][t][name]:.3f}")
json.dump(res, open(ROOT / "lc0-probe/results_value_via_head.json", "w"), indent=1)
