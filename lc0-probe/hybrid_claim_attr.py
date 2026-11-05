"""Integrated gradients through a hybrid stack (hybrid_stacks.py) on the claim boards, in square_attribution.py's format,
so claim_concepts.py --attr can score them. W - L for the side to move, from each square's any-occupant mean bits.

  python lc0-probe/hybrid_claim_attr.py [--stack fwd|deep] [--pred] [--steps 256]   (both stacks by default, one load)
"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lc0-probe")]
from hybrid_stacks import build
import chess, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--stack", choices=["fwd", "deep"], nargs="*", default=["fwd", "deep"])
ap.add_argument("--pred", action="store_true")
ap.add_argument("--steps", type=int, default=256)
ap.add_argument("--fens", default="lc0-probe/data/claim_concepts.fens")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
H = build(a.pred)
LAB = json.loads(subprocess.run([ROOT / "features/square/target/release/square", "--schema"],
                                capture_output=True, text=True, check=True).stdout)["labels"]
real = [l for l in (ROOT / a.fens).read_text().splitlines() if l and not l.startswith("#")]
white = [(b if b.turn else b.mirror()).fen() for b in map(chess.Board, real)]
from lc0_planes import encode_fens
import onnxruntime as ort
full = ort.InferenceSession(str(ROOT / "vendor/lc0-nets/t1-256x10.onnx"), providers=["CPUExecutionProvider"])
Xs = H.bits(white)
for st in a.stack:
    f = H.forward_h if st == "fwd" else H.deep_h                         # the path runs in the decoders' first layer:
                                                                         # it is linear, so IG over Z equals IG over X
    SM, res, gaps = H.SM, [], []
    for wf, X in zip(white, Xs):
        al = ((torch.arange(a.steps) + 0.5) / a.steps)[:, None, None]
        Z0, dZ = SM @ H.Wcat, (X - SM) @ H.Wcat
        P = (Z0 + al * dZ).requires_grad_(True)
        H.wl(f(P)).sum().backward()
        A = ((X - SM) * (P.grad.mean(0) @ H.Wcat.T))[:, :H.K].numpy()            # dy/dX = dy/dZ @ Wcat^T
        with torch.no_grad(): y, y0 = float(H.wl(f((X @ H.Wcat)[None]))[0]), float(H.wl(f(Z0[None]))[0])
        gaps.append(abs(float(A.sum()) - (y - y0)))
        w = full.run(["/output/wdl"], {"/input/planes": encode_fens([wf])})[0][0]
        on, sm = X[:, :H.K].numpy() > 0, SM[:, :H.K].numpy()
        squares = {}
        for s in range(64):
            b, z = np.flatnonzero(on[s]), np.flatnonzero(~on[s] & (sm[s] > 0))
            squares[chess.square_name(s)] = {"square": None, "bits": [[int(j), round(float(A[s, j]), 6)] for j in b],
                                             "absent": [[int(j), round(float(A[s, j]), 6)] for j in z]}
        res.append({"fen": wf, "y": float(w[0] - w[2]), "y decoded": y, "y baseline": y0, "squares": squares})
        print(f"{len(res)}/{len(white)} y {y:+.3f} (Leela {res[-1]['y']:+.3f}) baseline {y0:+.3f} gap {gaps[-1]:.4f}", flush=True)
    name = f"{st}{'_pred' if a.pred else ''}"
    out = ROOT / f"lc0-probe/square_attribution_hybrid_{name}_claim_concepts.js"
    out.write_text("window.ATTRIBUTION = " + json.dumps({
        "tap": f"hybrid {name}", "method": f"integrated gradients, {a.steps} steps, square mean baseline, hybrid {name}",
        "source": a.fens, "labels": [LAB[k] for k in H.keep], "positions": res}, separators=(",", ":")) + ";\n")
    print(f"wrote {out}; max completeness gap {max(gaps):.4f}")
