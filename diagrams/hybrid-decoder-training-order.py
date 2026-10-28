"""Writes hybrid-decoder-training-order.excalidraw: forward vs reverse training order, side by side."""
import json, random
from pathlib import Path

random.seed(0)
els = []
INK, LEELA, FEAT, DEC = "#1e1e1e", "#e9ecef", "#fff3bf", "#1971c2"

def base(kind, x, y, w, h, **kw):
    e = {"type": kind, "id": f"e{len(els)}", "x": x, "y": y, "width": w, "height": h, "angle": 0,
         "strokeColor": INK, "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 1,
         "strokeStyle": "solid", "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
         "roundness": None, "seed": random.randint(1, 2**31), "version": 1, "versionNonce": random.randint(1, 2**31),
         "isDeleted": False, "boundElements": None, "updated": 1, "link": None, "locked": False}
    e.update(kw); els.append(e); return e

def text(x, y, s, size=18, w=None, align="left", container=None, color=INK):
    lines = s.split("\n")
    w = w or max(len(l) for l in lines) * size * 0.55
    h = len(lines) * size * 1.25
    return base("text", x, y, w, h, text=s, originalText=s, fontSize=size, fontFamily=2, textAlign=align,
                verticalAlign="middle", containerId=container, lineHeight=1.25, autoResize=True, strokeColor=color)

def box(x, y, w, h, label, fill):
    b = base("rectangle", x, y, w, h, backgroundColor=fill, roundness={"type": 3})
    t = text(x, y + h / 2 - 11, label, w=w, align="center", container=b["id"])
    b["boundElements"] = [{"type": "text", "id": t["id"]}]
    return b

def arrow(x0, y0, x1, y1, color=INK, width=1):
    return base("arrow", x0, y0, x1 - x0, y1 - y0, points=[[0, 0], [x1 - x0, y1 - y0]], strokeColor=color,
                strokeWidth=width, startBinding=None, endBinding=None, startArrowhead=None, endArrowhead="arrow",
                roundness={"type": 2}, lastCommittedPoint=None)

def panel(ox, title, labels, reverse=False):
    """labels: decoder arrow captions for L0, L2, L5 (top to bottom). reverse: layers 1–2 and 3–5 outputs are added to the next decoder."""
    text(ox, 0, title, size=26)
    col, w, h, gap = ox + 480, 220, 50, 34
    names = ["L0 stream", "Leela layers 1–2", "L2 stream", "Leela layers 3–5", "L5 stream", "Leela layers 6–10", "Value"]
    ys = [70 + i * (h + gap) for i in range(len(names))]
    for i, (n, y) in enumerate(zip(names, ys)):
        box(col, y, w, h, n, LEELA)
        if not i: continue
        if reverse and i in (2, 4):
            base("arrow", col + w / 2, ys[i - 1] + h, 0, y - ys[i - 1] - h, points=[[0, 0], [0, y - ys[i - 1] - h]],
                 strokeColor=DEC, strokeWidth=2, strokeStyle="dashed", startBinding=None, endBinding=None,
                 startArrowhead=None, endArrowhead="arrow", roundness={"type": 2}, lastCommittedPoint=None)
            text(col + w / 2 + 10, ys[i - 1] + h + 6, "+ added", size=16, color=DEC)
        else: arrow(col + w / 2, ys[i - 1] + h, col + w / 2, y)
    if reverse: text(col - 20, ys[-1] + h + 30, "L5 = D5 + layers₃₋₅(D2 + layers₁₋₂(D0))", size=16)
    fy = ys[2]
    box(ox, fy, 150, h, "Features", FEAT)
    for lab, i in zip(labels, (0, 2, 4)):
        y = ys[i] + h / 2
        arrow(ox + 150, fy + h / 2, col, y, color=DEC, width=2)
        if i == 2: text(ox + 225, y - 30, lab, size=16, color=DEC)          # above the straight arrow
        else: text(ox, ys[i] + (0 if i == 0 else 8), lab, size=16, color=DEC)  # in the empty corner beside the slope

panel(0, "Forward (trained)", ["① D0 replaces", "② D2 adds L2 leftover", "③ D5 adds L5 leftover"])
panel(820, "Reverse (proposed)", ["③ D0 last", "② D2 next", "① D5 first: L5 base"], reverse=True)

scene = {"type": "excalidraw", "version": 2, "source": "xai-chess", "elements": els,
         "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}
Path(__file__).with_suffix(".excalidraw").write_text(json.dumps(scene, ensure_ascii=False, indent=1))
