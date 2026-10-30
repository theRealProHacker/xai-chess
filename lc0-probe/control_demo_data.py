"""Data for the Control view of docs/features.html: how much each piece's control is worth to Leela.

Value of a piece = W-L(real board) - W-L(that piece's control.geometry bits at the any-occupant square mean, everything
else real), from one IG run per piece (square_attribution.py --ig-alone control.geometry.), signed for the piece's side.

  python lc0-probe/control_demo_data.py
"""
import json, re, sys
sys.argv[1:] = ["square", "sc"]
import chess
import space_lab as S

AL = S.L(S.ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_square_alone_control_geometry_space_demo.js")
GEO = set(S.GEO)
rows = [l.split("\t") for l in (S.ROOT / "lc0-probe/data/space_demo.tsv").read_text().split("\n") if l]
out = []
for fen, game, event, after in rows:
    event = " ".join(w for n, w in enumerate(event.split()) if "?" not in w and w not in event.split()[:n])   # drop unknown and repeated fields
    if re.fullmatch(r"W\d+ – B\d+", game): game = "Anonymous game"
    real = chess.Board(fen); mir = real.turn == chess.BLACK
    b = real.mirror() if mir else real
    r = next(p for p in AL["positions"] if p["fen"].split()[0] == b.fen().split()[0])
    pieces = []
    for s, pc in b.piece_map().items():
        if pc.piece_type == chess.KING: continue
        q = r["squares"].get(chess.square_name(s), {"bits": []})
        v = sum(x for j, x in q["bits"] + q.get("absent", []) if AL["labels"][j] in GEO)
        sg = 1 if pc.color == chess.WHITE else -1                          # IG is for White in the stored frame
        pieces.append([chess.square_name(chess.square_mirror(s) if mir else s), round(sg * v, 4)])
    out.append({"fen": fen, "game": game, "event": event, "after": after, "pieces": sorted(pieces, key=lambda x: -x[1])})
(S.ROOT / "docs/control_demo.js").write_text("window.CONTROL = " + json.dumps(out, separators=(",", ":"), ensure_ascii=False) + ";\n")
print(len(out), "positions")
