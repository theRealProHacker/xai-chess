"""Compare onnxruntime(policy, value) on our planes against lc0's own numbers.

usage: python check_encoding.py ref_fens.txt ref_lc0.txt model.onnx
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".pylibs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import onnxruntime as ort
from lc0_planes import encode_fens

fens = Path(sys.argv[1]).read_text().split("\n")
fens = [f for f in fens if f.strip()]
ref = Path(sys.argv[2]).read_text()

# lc0 verbose output: per position, "info string <move> (<idx>) ... (P: x%)" lines,
# the root line "info string node ... (V: v) (D: d)", then "bestmove".
blocks = ref.split("bestmove")[:-1]
assert len(blocks) == len(fens), (len(blocks), len(fens))
P_RE = re.compile(r"info string (?!node)(\S+)\s+\((\s*\d+)\s*\).*?\(P:\s*([\d.]+)%\)")
V_RE = re.compile(r"info string node .*?\(D:\s*([\d.]+)\).*?\(V:\s*([-\d.]+)\)")

sess = ort.InferenceSession(sys.argv[3], providers=["CPUExecutionProvider"])
x = encode_fens(fens)
pol, wdl, mlh = sess.run(["/output/policy", "/output/wdl", "/output/mlh"], {"/input/planes": x})

worst_p, worst_v = 0, 0
for i, (f, b) in enumerate(zip(fens, blocks)):
    moves = P_RE.findall(b)
    idx = np.array([int(m[1]) for m in moves])
    p_ref = np.array([float(m[2]) for m in moves]) / 100
    z = pol[i, idx] / 1.359          # lc0's default PolicySoftmaxTemp in classic search
    p = np.exp(z - z.max()); p /= p.sum()
    d_ref, v_ref = map(float, V_RE.search(b).groups())
    v = wdl[i, 0] - wdl[i, 2]
    dp = np.abs(p - p_ref).max()
    worst_p, worst_v = max(worst_p, dp), max(worst_v, abs(v - v_ref))
    print(f"{i}: {len(moves):2d} moves  max|dP| {dp:.5f}   V {v:+.4f} vs lc0 {v_ref:+.4f}   "
          f"D {wdl[i,1]:.3f} vs {d_ref:.3f}   {f[:40]}")
print(f"\nworst |dP| {worst_p:.5f} (lc0 prints 2 decimals of %, so <=0.0001 is exact)"
      f"   worst |dV| {worst_v:.5f}")
ok = worst_p < 1e-3 and worst_v < 2e-4    # P: lc0 eigen fp32 vs ort differ ~4e-4; a wrong plane costs >0.05
print("ENCODING", "MATCHES" if ok else "DIFFERS")
sys.exit(0 if ok else 1)
