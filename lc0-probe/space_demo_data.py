"""Data for space_demo.html: each piece's space value in 100 game positions (one per shuffled game).

Space value of a piece = its IG over its own control.geometry.t bits, each weighted by t's forward rank for the
piece's side, signed so positive helps that side. IG: encoder0_ln2 decoder, any-occupant baseline, W-L, path over
square.control and control.geometry bits only; second value: one IG run per piece moving only its own
control.geometry bits (--ig-alone).

  python lc0-probe/space_demo_data.py
"""
import json, re, sys
sys.argv[1:] = ["square_only_square_control_control_geometry", "pg+pp"]
import chess, numpy as np
import space_lab as S

A = S.L(S.ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_square_only_square_control_control_geometry_space_demo.js")
AL = S.L(S.ROOT / "lc0-probe/square_attribution_n100000_encoder0_ln2_ig_square_alone_control_geometry_space_demo.js")   # each piece moved alone
rows = [l.split("\t") for l in (S.ROOT / "lc0-probe/data/space_demo.tsv").read_text().split("\n") if l]
igs = {}                                                                   # (row, square, file) -> {label: IG}
for i, (fen, *_) in enumerate(rows):
    real = chess.Board(fen); b = real.mirror() if real.turn == chess.BLACK else real
    for s, pc in b.piece_map().items():
        if pc.piece_type == chess.KING: continue
        for k, F in enumerate((A, AL)):
            r = next(p for p in F["positions"] if p["fen"].split()[0] == b.fen().split()[0])
            q = r["squares"].get(chess.square_name(s), {"bits": []})
            igs[i, s, k] = {F["labels"][j]: v for j, v in q["bits"] + q.get("absent", [])}
out = []
for i, (fen, game, event, after) in enumerate(rows):
    event = " ".join(w for n, w in enumerate(event.split()) if "?" not in w and w not in event.split()[:n])   # drop unknown and repeated fields
    if re.fullmatch(r"W\d+ – B\d+", game): game = "Anonymous game"
    real = chess.Board(fen); mir = real.turn == chess.BLACK
    b = real.mirror() if mir else real
    pieces = []
    for s, pc in b.piece_map().items():
        if pc.piece_type == chess.KING: continue
        sg = 1 if pc.color == chess.WHITE else -1                          # IG is for White in the stored frame
        w = sg * S.FWD[pc.color]
        x = [igs[i, s, k].get(S.GEO[t], 0) for k in (0, 1) for t in chess.SQUARES]
        vs = [float(np.dot(w, x[:64])), float(np.dot(w, x[64:]))]
        pieces.append([chess.square_name(chess.square_mirror(s) if mir else s), *(round(v, 4) for v in vs)])   # [square, shared, alone]
    out.append({"fen": fen, "game": game, "event": event, "after": after, "pieces": sorted(pieces, key=lambda x: -abs(x[1]))})
V = np.array([p[2] for g in out for p in g["pieces"]])                 # z-score: alone values standardised over all pieces
for g in out:
    for p in g["pieces"]: p.append(round(float((p[2] - V.mean()) / V.std()), 4))
(S.ROOT / "lc0-probe/space_demo.js").write_text("window.SPACE = " + json.dumps(out) + ";\n")
print(len(out), "positions")
