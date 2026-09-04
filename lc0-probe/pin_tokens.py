"""fens.txt + meta.tsv -> data/tokens.npy: per position the three squares the pin
lives on, as token indices in the network's side-to-move sequence.

columns: king (of the pinned piece's colour), pinned piece, pinner.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".pylibs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import chess
from lc0_planes import stm_square

HERE = Path(__file__).resolve().parent
SF = HERE.parent / "sf-probe" / "data"

fens = [l for l in (SF / "fens.txt").read_text().split("\n") if l]
meta = [l.split("\t") for l in (SF / "meta.tsv").read_text().split("\n") if l]
hdr, meta = meta[0], meta[1:]
assert len(fens) == len(meta), (len(fens), len(meta))
col = hdr.index("pinned_sq")

out = np.zeros((len(fens), 3), np.int8)
bad = 0
for i, (fen, row) in enumerate(zip(fens, meta)):
    b = chess.Board(fen)
    sq = chess.parse_square(row[col])
    pc = b.piece_at(sq)
    assert pc is not None and b.is_pinned(pc.color, sq), (i, fen, row[col])
    king = b.king(pc.color)
    ray = int(b.pin(pc.color, sq))
    # pin() is the whole line through king and pinner; the pinner is the enemy
    # piece on it with nothing between it and the pinned piece, away from the king.
    cands = [c for c in chess.scan_forward(ray & b.occupied_co[not pc.color])
             if not (chess.between(sq, c) & b.occupied)
             and chess.square_distance(c, king) > chess.square_distance(sq, king)]
    assert len(cands) == 1, (i, fen, row[col], [chess.square_name(c) for c in cands])
    pinner = cands[0]
    flip = b.turn == chess.BLACK
    out[i] = [stm_square(king, flip), stm_square(sq, flip), stm_square(pinner, flip)]

(HERE / "data").mkdir(exist_ok=True)
np.save(HERE / "data" / "tokens.npy", out)
print(f"wrote {len(out):,} rows -> data/tokens.npy")
