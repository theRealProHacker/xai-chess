"""What each layer gives the trained L5 + L2 + L0 (hybrid_endmlp.py, both versions): held-out L5, WDL and value FVU of the
full stack and of the stack with one layer's term dropped (D5 -> 0, i.e. L5 = C + ...; M2 or M0 -> 0). With the prediction
input, M0 then reads the prediction without the dropped term.

  python lc0-probe/hybrid_endmlp_ablate.py [suffixes of endmlp{suffix}.pt, default: "" _pred]
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
res = {}
SUF = sys.argv[1:] or ["", "_pred"]
for P in SUF:
    em = torch.load(MM / f"endmlp{P}.pt", weights_only=False); pred, gate = P.startswith("_pred"), em.get("gate", False); Ms = {}
    for k in ("L2", "L0"):
        Ms[k] = MLP(em[k]["a.weight"].shape[1], em["hidden"], gate); Ms[k].load_state_dict(em[k])
    def model(E, drop):
        Cb = C.repeat(len(E), 1)
        x = Cb + (0 if drop == "L5" else f32(G5[E]))
        for k, Mo in Ms.items():
            if k == drop: continue
            f = f32(F[k][E])
            x = x + Mo(torch.cat([f, (x - Cb - mu5) / sd5], 1) if pred else f) * sd5
        return x
    name = {"": "without prediction", "_pred": "with prediction"}.get(P, P[1:]); res[name] = {}
    for drop in (None, "L5", "L2", "L0"):
        S, w = [], []
        with torch.no_grad():
            for E in (TE, TE + N):
                for s in range(0, len(E), 256):
                    x = model(E[s:s + 256], drop); S.append(x); w.append(tail(x))
        S, w = torch.cat(S), torch.cat(w)
        r = {"L5 FVU": fvu(S, torch.cat([Y, Y])), "WDL FVU": fvu(w, torch.cat([W, W])),
             "value FVU": fvu(v_of(w), v_of(torch.cat([W, W])))}
        res[name]["full" if drop is None else f"without {drop}"] = r
        log(f"{name:18} {'full' if drop is None else 'without ' + drop:11} " + "  ".join(f"{m} {v:.4f}" for m, v in r.items()))
out = ROOT / f"lc0-probe/results_hybrid_endmlp_ablate_games_n100000{'' if SUF == ['', '_pred'] else '_' + '_'.join(s.strip('_') for s in SUF)}.json"
json.dump(res, open(out, "w"), indent=1)
