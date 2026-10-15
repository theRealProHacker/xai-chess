"""Concatenate src parts, number [@key] citations IEEE-style by first appearance, append references."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from refs import R

here = Path(__file__).parent
src = "\n".join((here / f"part{i}.md").read_text() for i in (1, 2, 3, 4))
order = {}
for m in re.finditer(r"\[(@[^\]]+)\]", src):
    for k in re.findall(r"@([\w]+)", m.group(1)):
        if k not in R:
            sys.exit(f"unknown key {k}")
        order.setdefault(k, len(order) + 1)

def fmt(m):
    ns = sorted({order[k] for k in re.findall(r"@([\w]+)", m.group(1))})
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"[{ns[i]}]–[{ns[j]}]" if j - i >= 2 else ", ".join(f"[{n}]" for n in ns[i:j + 1]))
        i = j + 1
    return ", ".join(out)

body = re.sub(r"\[(@[^\]]+)\]", fmt, src)
refs = "\n".join(f"[{n}] {R[k]}\n" for k, n in sorted(order.items(), key=lambda kv: kv[1]))
body = body.replace("<!--REFERENCES-->", refs)
Path(sys.argv[1]).write_text(body)
print(len(order), "references cited;", len(R) - len(order), "unused:", sorted(set(R) - set(order)))
