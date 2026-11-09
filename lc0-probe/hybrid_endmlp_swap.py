"""How much the trained L5 + L2 + L0 (hybrid_endmlp.py) follows L5, with no baseline: on each held-out board, L5's decoder
output is swapped for another held-out board's (fixed random permutation), L2 and L0 stay. L5 influence =
sum |out(swapped) - out(real)|^2 / sum |Leela L5 - mean|^2 (0: L5 ignored); also the L5 FVU with the swap.

  python lc0-probe/hybrid_endmlp_swap.py [suffixes of endmlp{suffix}.pt, default: all]
"""
import json, sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")
tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None
sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
import chess, numpy as np, onnx, torch
from onnx2torch import convert
from lc0_planes import encode_fens

torch.set_num_threads(6)
N = 100000
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
fens = [l for l in (ROOT / "sf-probe/data/fens_games.txt").read_text().split("\n") if l][:N]
white = [f if f.split()[1] == "w" else chess.Board(f).mirror().fen() for f in fens]
TE = np.flatnonzero((np.random.default_rng(0).random(100000) < 0.2)[:N])
net = onnx.shape_inference.infer_shapes(onnx.load(ROOT / "vendor/lc0-nets/t1-256x10.onnx"))
def seg(i, o_):
    m = convert(onnx.utils.Extractor(net).extract_model(i, o_)).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m
s12, s35, tail = seg(["/encoder0/ln2"], ["/encoder2/ln2"]), seg(["/encoder2/ln2"], ["/encoder5/ln2"]), seg(["/encoder5/ln2"], ["/output/wdl"])
with torch.no_grad(): C = s35(s12(torch.zeros(64, 256)))
D = ROOT / "lc0-probe/data"; ST, MM = D / "hybrid_rev_streams_games_n100000", D / "hybrid_rev_tmp_games_n100000_adamw_d0"
mu5, sd5 = torch.load(ST / "norm5.pt")
mm = lambda n: np.lib.format.open_memmap(MM / f"{n}.npy", "r")
G5, F = mm("G5"), {"L2": mm("F2"), "L0": mm("F0")}
f32 = lambda X: torch.from_numpy(np.asarray(X).astype(np.float32)).reshape(-1, 256)

full = seg(["/input/planes"], ["/encoder5/ln2", "/output/wdl"])
with torch.no_grad():
    Y, W = map(torch.cat, zip(*[full(torch.from_numpy(encode_fens([white[i] for i in TE[s:s + 256]])))
                               for s in range(0, len(TE), 256)]))
Y = Y.reshape(-1, 256); log("Leela on the test boards")

class MLP(torch.nn.Module):
    def __init__(self, n_in, h, gate):
        super().__init__(); self.a, self.b, self.gate = torch.nn.Linear(n_in, h), torch.nn.Linear(h, 256), gate
        if gate: self.g = torch.nn.Sequential(torch.nn.Linear(512, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    def forward(self, x):
        if not self.gate: return self.b(torch.nn.functional.gelu(self.a(x)))
        return torch.sigmoid(self.g(x)) * self.b(torch.nn.functional.gelu(self.a(x[:, :256])))
fvu = lambda P, R: float(((P - R) ** 2).sum() / ((R - R.mean(0)) ** 2).sum())
v_of = lambda w: w[:, 0] - w[:, 2]
PERM = np.random.default_rng(0).permutation(len(TE))
res = {}
SUF = sys.argv[1:] or ["", "_pred", "_pred_gate"] + [f"_pred_pen{l:g}" for l in (0.03, 0.1, 0.3, 1, 2, 3)]
var = float(((Y - Y.mean(0)) ** 2).sum())
for P in SUF:
    em = torch.load(MM / f"endmlp{P}.pt", weights_only=False); pred, gate = P.startswith("_pred"), em.get("gate", False); Ms = {}
    for k in ("L2", "L0"):
        Ms[k] = MLP(em[k]["a.weight"].shape[1], em["hidden"], gate); Ms[k].load_state_dict(em[k])
    def model(E, E5):                                                     # L5 from boards E5, L2 and L0 from boards E
        Cb = C.repeat(len(E), 1)
        x = Cb + f32(G5[E5])
        for k, Mo in Ms.items():
            f = f32(F[k][E])
            x = x + Mo(torch.cat([f, (x - Cb - mu5) / sd5], 1) if pred else f) * sd5
        return x
    mv, err, n = 0.0, 0.0, 0
    with torch.no_grad():
        for off in (0, N):                                                # both colourings
            for s in range(0, len(TE), 256):
                E, E5 = TE[s:s + 256] + off, TE[PERM[s:s + 256]] + off
                a_, b_ = model(E, E), model(E, E5)
                mv += float(((b_ - a_) ** 2).sum()); err += float(((b_ - Y[64 * s:64 * (s + 256)]) ** 2).sum())
    name = {"": "without prediction", "_pred": "with prediction"}.get(P, P[1:])
    res[name] = {"L5 influence": mv / (2 * var), "L5 FVU, L5 swapped": err / (2 * var)}
    log(f"{name:18} " + "  ".join(f"{m} {v:.4f}" for m, v in res[name].items()))
json.dump(res, open(ROOT / "lc0-probe/results_hybrid_endmlp_swap_games_n100000.json", "w"), indent=1)
