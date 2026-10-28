"""Fraction of variance unexplained at L5 (/encoder5/ln2, all tokens and channels), at the WDL head and in the value
(W - L), on the held-out positions in both colourings, for hybrid decoders and subsets of them.

A model is a decoders checkpoint (hybrid_decoder.py / hybrid_reverse.py format) or a hybrid_reverse.py temp dir
(its fit_D*_final.pt, for a run still going). The first tap's decoder replaces the stream, later ones add; a decoder
left out of --use outputs zero. FVU = sum (pred - real)^2 / sum (real - mean)^2, means per channel (WDL: per class),
both colourings against Leela on the White-to-move board.

  python lc0-probe/hybrid_fvu.py "just L5=lc0-probe/data/hybrid_rev_tmp_n20000_adamw:/encoder5/ln2" ...
"""
import argparse, json, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, torch
from onnx2torch import convert
from lc0_planes import encode_fens
from colour_frame import label_map

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", metavar="NAME=PATH[:TAP,TAP]", help="taps to use (default: all of PATH's)")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--out", default=str(ROOT / "lc0-probe/results_hybrid_fvu_n20000.json"))
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--games", action="store_true", help="the 100k Lichess game positions (fens_games.txt) instead of the puzzles")
a = ap.parse_args()
torch.set_num_threads(a.threads)
N = a.n
G = "_games" if a.games else ""
REV_TAPS = ["/encoder0/ln2", "/encoder2/ln2", "/encoder5/ln2"]

def load(path):
    p = Path(path)
    if p.suffix == ".pt":
        ck = torch.load(p, weights_only=False)
        return ck["keep"], ck["decoders"]
    # a hybrid_reverse.py temp dir: its finished stages, in tap order, with that script's scales
    st = ROOT / f"lc0-probe/data/hybrid_rev_streams{G}_n{N}"
    mu5, sd5 = torch.load(st / "norm5.pt"); sd0, sd2 = torch.load(st / "scales.pt")
    ds = []
    for name, tap, mu, sd in (("D0", REV_TAPS[0], 0, sd0), ("D2", REV_TAPS[1], 0, sd2), ("D5", REV_TAPS[2], mu5, sd5)):
        f = p / f"fit_{name}_final.pt"
        c = torch.load(f, weights_only=False) if f.exists() else None
        ds.append({"state": c["state"] if c and c["done"] else None, "mu": mu, "sd": sd, "tap": tap})
    return None, ds

fens = [l for l in (ROOT / f"sf-probe/data/fens{G}.txt").read_text().split("\n") if l][:N]
white = [f if f.split()[1] == "w" else chess.Board(f).mirror().fen() for f in fens]
te = np.flatnonzero((np.random.default_rng(0).random(100000) < 0.2)[:N])                # the decoders' test split
tr_pos = ~(np.random.default_rng(0).random(100000) < 0.2)[:N] & ~(np.random.default_rng(1).random(100000) < 0.1)[:N]
Mp = np.array(label_map(), np.int32)
rows = {}
with open(ROOT / f"lc0-probe/data/square_bits{G}_n100000_nodes200000.jsonl") as f:
    for i, l in enumerate(f):
        if i == N: break
        rows[i] = json.loads(l)["squares"]
sel = set(te.tolist())
cnt = np.bincount(np.fromiter((b for i in range(N) if tr_pos[i] for v in rows[i].values() for b in v), np.int32),
                  minlength=len(Mp))
keep0 = np.flatnonzero((cnt + cnt[Mp]) >= 20)                              # the decoders' default --min
rows = {i: rows[i] for i in sel}

def inputs(keep, black, B=256):
    """Bags of input rows per test token, as the decoders were trained, in chunks of B positions."""
    col = np.full(len(Mp), -1, np.int32); col[keep] = np.arange(len(keep))
    SQ, STM = len(keep), len(keep) + 64
    chunks = []
    for s in range(0, len(te), B):
        flat, off, n = [], [], 0
        for i in te[s:s + B]:
            sq = {chess.parse_square(q): np.array(v, np.int32) for q, v in rows[i].items()}
            for t in range(64):
                v = sq.get(t, np.array([], np.int32))
                c = col[Mp[v]] if black else col[v]
                b = np.concatenate([c[c >= 0], [SQ + (t ^ 56 if black else t)] + ([STM] if black else [])])
                off.append(n); n += len(b); flat.append(b.astype(np.int64))
        chunks.append((torch.from_numpy(np.concatenate(flat)), torch.tensor(off)))
    return chunks

net = onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))
def seg(i, o):
    m = convert(onnx.utils.Extractor(net).extract_model(i, o)).eval()
    for q in m.parameters(): q.requires_grad_(False)
    return m
segs = {}
def seg_c(i, o):
    if (i, o) not in segs: segs[i, o] = seg([i], [o])
    return segs[i, o]

full = seg(["/input/planes"], ["/encoder5/ln2", "/output/wdl"])
with torch.no_grad():
    Y5, W = map(torch.cat, zip(*[full(torch.from_numpy(encode_fens([white[i] for i in te[s:s + 256]])))
                                for s in range(0, len(te), 256)]))
Y5 = Y5.reshape(-1, 256)
fvu = lambda P, R: float(((P - R) ** 2).sum() / ((R - R.mean(0)) ** 2).sum())

def decoder(d):
    s = d["state"]
    inp = torch.nn.EmbeddingBag(*s["inp.weight"].shape, mode="sum"); inp.weight.data = s["inp.weight"]
    o = torch.nn.Linear(*s["out.weight"].shape[::-1]); o.load_state_dict({"weight": s["out.weight"], "bias": s["out.bias"]})
    mu, sd = torch.as_tensor(d["mu"]).float(), float(d["sd"])
    return lambda flat, off: o(torch.nn.functional.gelu(inp(flat, off) + s["bias"])) * sd + mu

cache, res = {}, {}
for spec in a.models:
    name, rest = spec.split("=", 1)
    path, use = rest.rsplit(":", 1) if ":/" in rest else (rest, "")
    keep, ds = load(path)
    keep = keep0 if keep is None else keep
    use = use.split(",") if use else [d["tap"] for d in ds]
    assert all(d["state"] is not None for d in ds if d["tap"] in use), f"{name}: a used stage is not finished"
    taps = [d["tap"] for d in ds]
    fns = [decoder(d) if d["tap"] in use else None for d in ds]
    S5, Wp = [], []
    for black in (False, True):
        k = (tuple(keep), black)
        if k not in cache: cache[k] = inputs(keep, black)
        with torch.no_grad():
            for flat, off in cache[k]:
                S = fns[0](flat, off) if fns[0] else torch.zeros(len(off), 256)
                for j in range(1, len(ds)):
                    S = seg_c(taps[j - 1], taps[j])(S)
                    if fns[j]: S = S + fns[j](flat, off)
                if taps[-1] != "/encoder5/ln2": S = seg_c(taps[-1], "/encoder5/ln2")(S)
                S5.append(S); Wp.append(seg_c("/encoder5/ln2", "/output/wdl")(S))
    S, w = torch.cat(S5), torch.cat(Wp)
    v = lambda x: x[:, 0] - x[:, 2]
    res[name] = {"L5 FVU": fvu(S, torch.cat([Y5, Y5])), "WDL FVU": fvu(w, torch.cat([W, W])),
                 "value FVU": fvu(v(w), v(torch.cat([W, W]))), "uses": use,
                 "value r": float(np.corrcoef((W[:, 0] - W[:, 2]).numpy(), (w[:len(te), 0] - w[:len(te), 2]).numpy())[0, 1])}
    print(f"{name:14} L5 FVU {res[name]['L5 FVU']:.4f}   WDL FVU {res[name]['WDL FVU']:.4f}   value FVU {res[name]['value FVU']:.4f}   value r {res[name]['value r']:.4f}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
