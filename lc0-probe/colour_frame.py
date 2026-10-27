"""Square features in real colours from the side-to-move-frame cache: for a Black-to-move position, flip each
square (and every square, rank and colour named in a label) back. Checked against the extractor run on the real board."""
import json, re, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

COMPASS = {"n": "s", "s": "n", "ne": "se", "se": "ne", "nw": "sw", "sw": "nw", "e": "e", "w": "w"}

def _swap(l):
    parts = [COMPASS.get(x, x) for x in l.split(".")]
    diag = re.fullmatch(r"([a-h][1-8])([a-h][1-8])", parts[-1]) if parts[0] in ("blocks", "long_diagonal") else None
    if diag: parts[-1] = {flip_square(diag[1]), flip_square(diag[2])}       # diagonals are named by their ends
    l = ".".join(x if isinstance(x, str) else "{}" for x in parts) if diag else ".".join(parts)
    if diag: return l, parts[-1]
    l = re.sub(r"\b([a-h])([1-8])\b", lambda m: m[1] + str(9 - int(m[2])), l.replace(".", " ")).replace(" ", ".")
    l = re.sub(r"(line|doubled_with)\.rank([1-8])$", lambda m: f"{m[1]}.rank{9 - int(m[2])}", l)   # board ranks; pawn.rank7 is the pawn's own
    return re.sub(r"white|black", lambda m: "black" if m[0] == "white" else "white", l)

def label_map():
    """bit index -> the same feature with colours and ranks flipped (an involution)."""
    binary, cached = ROOT / "features/square/target/release/square", ROOT / "lc0-probe/data/square_schema.json"
    schema = (subprocess.run([binary, "--schema"], capture_output=True, text=True, check=True).stdout if binary.exists()
              else cached.read_text())                                     # the cluster has no Rust build
    labels = json.loads(schema)["labels"]
    ix = {l: i for i, l in enumerate(labels)}
    ends = {(l.rsplit(".", 1)[0], frozenset((m[1], m[2]))): i for i, l in enumerate(labels)
            if (m := re.fullmatch(r".*\.([a-h][1-8])([a-h][1-8])", l))}
    out = []
    for l in labels:
        t = _swap(l)
        out.append(ends[t[0].rsplit(".", 1)[0], frozenset(t[1])] if isinstance(t, tuple) else ix[t])
    return out

def flip_square(name):
    return name[0] + str(9 - int(name[1]))

def to_real(squares, black_to_move, M):
    """A cache row's {square: bits} (side-to-move frame) -> real colours."""
    if not black_to_move: return squares
    return {flip_square(s): sorted(M[b] for b in v) for s, v in squares.items()}
