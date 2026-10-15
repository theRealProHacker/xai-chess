"""How the embedding decoder's error grows through the net.

Takes the decoder square_decoder_wdl.py saved for one tap (default the embedding, /attn_body/rehape),
puts its decoded tokens in place of the real ones for the held-out positions, runs the rest of the
net and reports the token FVU at every later /encoderN/ln1 and /encoderN/ln2 against the real net,
and the FVU on /output/wdl. Next to it: the FVU of the decoder trained directly on that layer.

  python lc0-probe/square_decoder_forward.py
"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import chess, numpy as np, onnx, onnxruntime as ort, torch
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--nodes", type=int, default=200000)
ap.add_argument("--tap", default="/attn_body/rehape", help="where the decoded tokens go in")
ap.add_argument("--positions", type=int, default=0, help="score only the first this many held-out positions; 0 = all")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)

fens = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:a.n]
N = len(fens)
te_pos = np.random.default_rng(0).random(N) < 0.2                          # same split as square_probe.py
TE = np.flatnonzero(te_pos)
if a.positions: TE = TE[:a.positions]
ck = torch.load(ROOT / f"lc0-probe/data/square_decoder{a.tap.replace('/', '_')}_n{N}.pt", weights_only=False)
keep, mu, sd = ck["keep"], ck["mu"], ck["sd"]
L = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                              capture_output=True, text=True, check=True).stdout)["bits_per_square"]
col = np.full(L, -1); col[keep] = np.arange(len(keep))
flat, off = [], [0]                                                        # EmbeddingBag inputs, held-out tokens in order
with open(ROOT / f"lc0-probe/data/square_bits_n{N}_nodes{a.nodes}.jsonl") as f:
    for i, l in enumerate(f):
        if i > TE[-1]: break
        if not te_pos[i]: continue
        by_sq = {chess.parse_square(s): v for s, v in json.loads(l)["squares"].items()}
        for s in range(64):
            b = col[by_sq.get(s, [])]; b = b[b >= 0]
            flat += [b, [len(keep) + s]]; off.append(off[-1] + len(b) + 1)   # the square one-hot comes last
flat, off = torch.from_numpy(np.concatenate(flat).astype(np.int64)), torch.tensor(off[:-1])

hidden, d_out = ck["state"]["out.weight"].shape[::-1]
inp = torch.nn.EmbeddingBag(len(keep) + 64, hidden, mode="sum"); out = torch.nn.Linear(hidden, d_out)
inp.load_state_dict({"weight": ck["state"]["inp.weight"]})
out.load_state_dict({"weight": ck["state"]["out.weight"], "bias": ck["state"]["out.bias"]})
with torch.no_grad():                                                      # decoded tokens, in the tap's units
    D = (out(torch.nn.functional.gelu(inp(flat, off) + ck["state"]["bias"])).numpy() * sd + mu).astype(np.float32)

net_path = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
order = ["/attn_body/rehape"] + [f"/encoder{i}/ln{j}" for i in range(10) for j in (1, 2)]
later = order[order.index(a.tap) + 1:] + ["/output/wdl"]
m = onnx.load(net_path)
for t in [a.tap] + later[:-1]: m.graph.output.append(onnx.ValueInfoProto(name=t))
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
full = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(net_path))).extract_model([a.tap], later)
assert [i.name for i in sub.graph.input] == [a.tap]
tail = ort.InferenceSession(sub.SerializeToString(), so, providers=["CPUExecutionProvider"])
flat_in = len(tail.get_inputs()[0].shape) == 2                             # some taps are (P*64, D) in the graph

sse = {t: 0.0 for t in ["decoded " + a.tap] + later}; real = {t: [] for t in sse}; check = 0.0
for s in range(0, len(TE), 256):
    R = full.run([a.tap] + later, {"/input/planes": encode_fens([fens[i] for i in TE[s:s + 256]])})
    d = D[s * 64:(s + 256) * 64]
    feed = lambda x: x.reshape(R[0].shape) if flat_in else x.reshape(-1, 64, x.shape[-1])
    check = max(check, max(np.abs(p - r).max() for p, r in zip(tail.run(later, {a.tap: feed(R[0])}), R[1:])))
    P = [d] + tail.run(later, {a.tap: feed(d)})
    for t, p, r in zip(sse, P, R):
        r = r.reshape(-1, r.shape[-1]); sse[t] += float(((r - p.reshape(r.shape)) ** 2).sum()); real[t].append(r)
assert check < 1e-3, check                                                 # the cut net on real tokens matches the full one
print(f"tail check: max |diff| {check:.1e}")

direct = json.load(open(ROOT / f"lc0-probe/results_square_decoder_wdl_n{N}.json"))["taps"]
res = {"n": N, "tap": a.tap, "test positions": int(len(TE)), "taps": {}}
suffix = f"_first{a.positions}" if a.positions else ""
for t in sse:
    r = np.concatenate(real[t]); name = t.replace("decoded ", "")
    res["taps"][name] = {"forward": sse[t] / float(((r - r.mean(0)) ** 2).sum()),
                         "direct": direct.get(name, {}).get("wdl test" if name == "/output/wdl" else "token test")}
    if name == "/output/wdl": res["taps"][name]["direct"] = None           # no decoder is trained on the WDL itself
    print(f"{name:20} FVU forward {res['taps'][name]['forward']:.3f}   direct {res['taps'][name]['direct']}")
json.dump(res, open(ROOT / f"lc0-probe/results_square_decoder_forward{a.tap.replace('/', '_')}_n{N}{suffix}.json", "w"), indent=1)
