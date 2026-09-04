"""Run the ONNX net over fens.txt and store the residual stream at every tap.

Per position and tap, four 256-vectors: the mean over the 64 square tokens, and
the tokens on the pin's three squares (king, pinned piece, pinner). The full
(64 x 256) residual at 21 taps would be 150 GB; this is 6.5 GB.

  data/acts.npy    float16 [N, taps, 4, D]     (numpy memmap)
  data/wdl.npy     float32 [N, 3]              value head, a sanity check
  data/taps.json   tap names, in forward order

env: LC0_ONNX (model), LC0_BATCH
"""
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".pylibs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, onnx, onnxruntime as ort
from lc0_planes import encode_fens

HERE = Path(__file__).resolve().parent
SF = HERE.parent / "sf-probe" / "data"
MODEL = Path(os.environ.get("LC0_ONNX", HERE.parent / "vendor/lc0-nets/t1-256x10.onnx"))
BATCH = int(os.environ.get("LC0_BATCH", 256))

m = onnx.load(MODEL)
n_enc = 1 + max(int(n.name.split("/")[1][7:]) for n in m.graph.node if n.name.startswith("/encoder"))
taps = ["embed"] + [f"enc{i}.{s}" for i in range(n_enc) for s in ("attn", "ffn")]
nodes = ["/attn_body/rehape"] + [f"/encoder{i}/ln{j}" for i in range(n_enc) for j in (1, 2)]
for t in nodes:
    m.graph.output.append(onnx.ValueInfoProto(name=t))
so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count()
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

fens = [l for l in (SF / "fens.txt").read_text().split("\n") if l]
tok = np.load(HERE / "data" / "tokens.npy").astype(np.int64)
N, T = len(fens), len(taps)
D = sess.get_outputs()[-1].shape[-1]
print(f"{MODEL.name}: {n_enc} encoders, D={D}, {T} taps, N={N:,}", flush=True)

acts = np.lib.format.open_memmap(HERE / "data" / "acts.npy", "w+", np.float16, (N, T, 4, D))
wdl = np.zeros((N, 3), np.float32)
x = np.zeros((BATCH, 112, 8, 8), np.float32)
t0 = time.time()
for s in range(0, N, BATCH):
    e = min(N, s + BATCH); b = e - s
    encode_fens(fens[s:e], x[:b])
    outs = sess.run(["/output/wdl"] + nodes, {"/input/planes": x[:b]})
    wdl[s:e] = outs[0]
    ar = np.arange(b)
    for t, o in enumerate(outs[1:]):
        o = o.reshape(b, 64, D)
        acts[s:e, t, 0] = o.mean(1)
        for k in range(3):
            acts[s:e, t, 1 + k] = o[ar, tok[s:e, k]]
    if (s // BATCH) % 50 == 0:
        done = e / N; el = time.time() - t0
        print(f"{e:>7,}/{N:,}  {el/60:5.1f} min  eta {el/done*(1-done)/60:5.1f} min", flush=True)
acts.flush(); del acts
np.save(HERE / "data" / "wdl.npy", wdl)
json.dump(dict(model=MODEL.name, taps=taps, nodes=nodes, views=["mean", "king", "pinned", "pinner"],
               d=D, n=N), open(HERE / "data" / "taps.json", "w"), indent=1)
print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
