"""PoC for dsl_attribution.md: book claim 28, "Black's isolated d5 pawn is weak".

DSL verdict of Isolated(Squares([d5])) (dsl_verifier check, after the quiet 14.h3, since check needs a move), then IG
on the board (square baseline, 256 steps) and the Weakness bit sum on each Black pawn, signed for White.

  python lc0-probe/dsl_poc.py
"""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs")]
import chess

FEN = "2rq1rk1/1b2bppp/pp3n2/2np2B1/B7/2N1PN2/PP2QPPP/2R2RK1 w - - 0 14"
DSL = "Isolated(Squares([d5]))"

# claim_concepts.py runs its whole study on import, so take only its View class
src = (ROOT / "lc0-probe/claim_concepts.py").read_text()
W, B, sq = chess.WHITE, chess.BLACK, chess.parse_square
exec(src[src.index("class View:"):src.index("\nS = lambda")])

q = json.dumps({"fen": FEN, "mov": "h3", "dsl": DSL})
dsl = json.loads(subprocess.run([ROOT / "dsl_verifier/target/release/dsl_verifier", "check"], input=q + "\n",
                                capture_output=True, text=True, check=True).stdout)

FF = ROOT / "lc0-probe/data/dsl_poc.fens"
FF.write_text(FEN + "\n")
OUT = ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_square_dsl_poc.js"
if not OUT.exists():
    subprocess.run([sys.executable, ROOT / "lc0-probe/square_attribution.py", "--n", "100000", "--tap", "/encoder0/ln2",
                    "--ig", "256", "--ig-base", "square", "--fens", FF, "--positions", "1", "--threads", "4"], check=True)
A = json.loads(OUT.read_text().split("=", 1)[1].rstrip().rstrip(";"))
r = A["positions"][0]
v = View(r, A["labels"], False)            # White to move: stored = real

print(f"DSL {DSL}: {dsl['v']}  ({dsl['english']})")
print(f"y {float(r['y']):+.3f}  y baseline {float(r['y baseline']):+.3f}  total |IG| {v.total:.3f}\n")
print("pawn   weakness (for White)   top bits")
rows = []
for t in sorted(v.b.pieces(chess.PAWN, B)):
    x, bits = v.work(lambda u: -u.weakness([t]))        # weakness() signs for the pawn's owner; flip to White
    rows.append((chess.square_name(t), x, bits))
for name, x, bits in sorted(rows, key=lambda z: -z[1]):
    top = ", ".join(f"{b[1]} {-b[2]:+.3f}" for b in bits[:3])
    print(f"{name:5}  {x:+.4f}               {top}")
