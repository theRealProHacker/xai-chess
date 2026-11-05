"""The two hybrid stacks, Leela's W - L read off feature bits, for integrated gradients.

Forward: L5 = layers3-5(layers1-2(D0) + D2) + D5 (hybrid_decoder.py, L0 fitted first). Deepest first: L5 = C + D5 + M2 + M0,
the end MLPs reading the L2-only and the forward D0 decoder forwarded to L5 (hybrid_endmlp.py; pred=True: they also read
the prediction they correct). Inputs X (B, 64, D_in): per-square bits + square one-hot, boards White to move.
"""
import json, subprocess, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")   # onnx2torch imports it for ops
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None                             # this net doesn't use
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, torch
from onnx2torch import convert

def build(pred=False):
    """-> namespace with forward_stack, deep_stack (on X), forward_h, deep_h (on Z = X @ Wcat), wl, keep, K, D_in, bits(white_fens) -> [X], SM (any-occupant mean)."""
    a = types.SimpleNamespace(pred=pred)
    D = ROOT / "lc0-probe/data"
    fwd = torch.load(D / "hybrid_decoder_games_n100000_adamw.pt", weights_only=False)
    rev = torch.load(D / "hybrid_reverse_games_n100000_adamw.pt", weights_only=False)
    l2 = torch.load(D / "hybrid_decoder_games_n100000_adamw_l2only.pt", weights_only=False)
    MM = D / "hybrid_rev_tmp_games_n100000_adamw_d0"
    em = torch.load(MM / f"endmlp{'_pred' if a.pred else ''}.pt", weights_only=False)
    keep = fwd["keep"]
    assert all(np.array_equal(c["keep"], keep) for c in (rev, l2))
    K = len(keep); D_in = K + 65

    def dense(d):
        """Decoder split at its linear first layer: (W, post), decoder(X) = post(X @ W), (B, 64, h) -> (B*64, 256)."""
        s = d["state"]; W, b = s["inp.weight"], s["bias"]
        o = torch.nn.Linear(*s["out.weight"].shape[::-1]); o.load_state_dict({"weight": s["out.weight"], "bias": s["out.bias"]})
        for p in o.parameters(): p.requires_grad_(False)
        mu, sd = torch.as_tensor(d["mu"]).float(), float(d["sd"])
        return W, lambda H: o(torch.nn.functional.gelu((H + b).reshape(-1, W.shape[1]))) * sd + mu
    tap = lambda ck, t: dense(next(d for d in ck["decoders"] if d["tap"] == t))
    dec = {"F0": tap(fwd, "/encoder0/ln2"), "F2": tap(fwd, "/encoder2/ln2"), "F5": tap(fwd, "/encoder5/ln2"),
           "R5": tap(rev, "/encoder5/ln2"), "L2": dense(l2["decoders"][0])}
    Wcat = torch.cat([W for W, _ in dec.values()], 1)                     # all first layers: one matmul, X @ Wcat
    cuts = np.cumsum([0] + [W.shape[1] for W, _ in dec.values()])
    def split(Z): return {k: post(Z[..., cuts[i]:cuts[i + 1]]) for i, (k, (_, post)) in enumerate(dec.items())}

    net = onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))
    def seg(i, o_):
        m = convert(onnx.utils.Extractor(net).extract_model([i], [o_])).eval()
        for p in m.parameters(): p.requires_grad_(False)
        return m
    s12, s35, s05, tail = seg("/encoder0/ln2", "/encoder2/ln2"), seg("/encoder2/ln2", "/encoder5/ln2"), \
                          seg("/encoder0/ln2", "/encoder5/ln2"), seg("/encoder5/ln2", "/output/wdl")
    with torch.no_grad(): C = s35(s12(torch.zeros(64, 256)))
    mu5, sd5 = torch.load(D / "hybrid_rev_streams_games_n100000/norm5.pt")

    class MLP(torch.nn.Module):
        def __init__(self, n_in, h):
            super().__init__(); self.a, self.b = torch.nn.Linear(n_in, h), torch.nn.Linear(h, 256)
        def forward(self, x): return self.b(torch.nn.functional.gelu(self.a(x)))
    Ms = {}
    for k in ("L2", "L0"):
        st = em[k]; Ms[k] = MLP(st["a.weight"].shape[1], em["hidden"]); Ms[k].load_state_dict(st)
        for p in Ms[k].parameters(): p.requires_grad_(False)

    def forward_h(Z):                                                     # Z = X @ Wcat, (B, 64, sum h)
        B, d = Z.shape[0], split(Z)
        return tail(s35(s12(d["F0"]) + d["F2"]) + d["F5"]).reshape(B, 3)
    def deep_h(Z):
        B, d = Z.shape[0], split(Z); Cb = C.repeat(B, 1)
        x = d["R5"] + Cb
        for k, F in (("L2", s35(d["L2"])), ("L0", s05(d["F0"]))):
            inp = F if not a.pred else torch.cat([F, (x - Cb - mu5) / sd5], 1)
            x = x + Ms[k](inp) * sd5
        return tail(x).reshape(B, 3)
    forward_stack, deep_stack = (lambda X: forward_h(X @ Wcat)), (lambda X: deep_h(X @ Wcat))
    wl = lambda w: w[:, 0] - w[:, 2]

    col = np.full(max(keep) + 1 + 100000, -1); col[keep] = np.arange(K)
    def xs(squares):
        X = torch.zeros(64, D_in)
        for q, v in squares.items():
            c = col[np.array(v)]; X[chess.parse_square(q), torch.from_numpy(c[c >= 0])] = 1
        X[torch.arange(64), K + torch.arange(64)] = 1                             # square one-hot, White to move: no STM row
        return X
    te = np.random.default_rng(0).random(100000) < 0.2
    va = np.random.default_rng(1).random(100000) < 0.1
    SM, n = torch.zeros(64, D_in), 0
    with open(D / "square_bits_games_n100000_nodes200000.jsonl") as f:          # mean bits per square over training boards
        for i, l in enumerate(f):
            if n == 20000: break
            if te[i] or va[i]: continue
            SM += xs(json.loads(l)["squares"]); n += 1
    SM /= n
    def bits(white):
        out = subprocess.run([ROOT / "features/square/target/release/square", "--indices", "--engine",
                              ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2", "--nodes", "200000"],
                             input="\n".join(white) + "\n", capture_output=True, text=True, check=True).stdout.splitlines()
        rs = [json.loads(l) for l in out]
        assert [r["fen"] for r in rs] == list(white)
        return [xs(r["squares"]) for r in rs]
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if k in
        ("forward_stack", "deep_stack", "forward_h", "deep_h", "Wcat", "wl", "keep", "K", "D_in", "bits", "SM", "xs")})
