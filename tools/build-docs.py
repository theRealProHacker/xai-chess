#!/usr/bin/env python3
"""Regenerate docs/papers.html (and the old pages' redirect stubs) from Markdown.

    uv run --no-project --with markdown==3.5.2 python3 tools/build-docs.py          # write
    uv run --no-project --with markdown==3.5.2 python3 tools/build-docs.py --check  # verify only

papers.html holds the survey, the landscape and the deep-research summaries,
one view at a time, switched by #survey / #landscape / #summaries.
`toc` supplies the heading slugs, `smarty` the &rsquo;/&ldquo; entities.
"""
import argparse
import pathlib
import re
import sys

import markdown

EXTENSIONS = ["extra", "toc", "smarty"]

# (view hash, button label, source markdown, old page now redirecting here)
VIEWS = [
    ("survey", "Survey", "paper/survey.md", "docs/survey.html"),
    ("landscape", "Landscape", "paper/landscape.md", "docs/landscape.html"),
    ("summaries", "Summaries", "RESEARCH_SUMMARIES.md", "docs/RESEARCH_SUMMARIES.html"),
]
OUT = "docs/papers.html"

ICONS = """<link rel="icon" href="favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="32x32" href="favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="favicon-16x16.png">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<link rel="manifest" href="site.webmanifest">"""

STYLE = """:root{
  --shadow-s: inset 0 1px 2px #ffffff30, 0 1px 2px #00000030, 0 2px 4px #00000015;
  --shadow-m: inset 0 1px 2px #ffffff50, 0 2px 4px #00000030, 0 4px 8px #00000015;
  --ink:#1a1a1a; --muted:#6b6b6b; --line:#e4e4e4; --accent:#0b5fa5;
}
*{box-sizing:border-box}
body{
  max-width:820px; margin:2.5rem auto 4rem; padding:0 1.5rem; background:#fff; color:var(--ink);
  font:17px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}
h1{font-size:1.6rem;font-weight:600;letter-spacing:-.01em;line-height:1.25;margin:0 0 .4rem}
a.back{color:#8a8a8a;text-decoration:none;font-size:.9rem}
a.back:hover{color:var(--accent)}
button{
  font:inherit;padding:.4rem .9rem;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink);
  cursor:pointer;box-shadow:var(--shadow-s);transition:background .12s ease,box-shadow .12s ease;
}
button:hover{background:#f4f4f4;box-shadow:var(--shadow-m)}
.mode{display:flex;flex-wrap:wrap;gap:.4rem;margin:.6rem 0 1.6rem}
.mode button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.mode button.on:hover{background:#333}
@media (max-width:480px){.mode button{padding:.4rem .6rem}}
.ai{color:#aaa;font-size:.85rem;margin:0 0 .3rem}
.doc{overflow-wrap:anywhere}
.doc h1{font-size:1.35rem;margin:2.4rem 0 .5rem}
.doc > h1:first-of-type{margin-top:0}
.doc h2{font-size:1.15rem;font-weight:600;margin:2.2rem 0 .5rem;padding-bottom:.2rem;border-bottom:1px solid var(--line)}
.doc h3{font-size:1rem;font-weight:600;margin:1.6rem 0 .4rem}
.doc p{margin:.7rem 0}
.doc em{color:#444}
.doc a{color:var(--accent)}
.doc hr{border:none;border-top:1px solid var(--line);margin:2rem 0}
.doc ol,.doc ul{padding-left:1.6rem}
.doc li{margin:.3rem 0}
.doc code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px;font-size:.88em}
.tw{overflow-x:auto;margin:1.2rem 0}
.doc table{border-collapse:collapse;font-size:.92rem;width:100%;overflow-wrap:normal}
.doc th,.doc td{border:1px solid var(--line);padding:.45rem .7rem;text-align:left;vertical-align:top}
.doc th{background:#f4f4f4;font-weight:600}
"""

SCRIPT = """const views = [...document.querySelectorAll(".doc")].map(d => d.id.slice(4));
function show() {
  const h = decodeURIComponent(location.hash.slice(1));
  const el = h && !views.includes(h) && document.getElementById(h);
  const doc = el && el.closest(".doc");
  const v = doc ? doc.id.slice(4) : views.includes(h) ? h : views[0];
  for (const d of views) {
    document.getElementById("doc-" + d).hidden = d !== v;
    document.getElementById("btn-" + d).classList.toggle("on", d === v);
  }
  if (doc) el.scrollIntoView(); else window.scrollTo(0, 0);
}
for (const v of views) document.getElementById("btn-" + v).onclick = () => { location.hash = v; };
addEventListener("hashchange", show);
show();
"""

STUB = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Papers &middot; xai-chess</title>
<meta http-equiv="refresh" content="0; url=papers.html#{view}">
<script>location.replace("papers.html" + (location.hash || "#{view}"));</script>
</head><body><a href="papers.html#{view}">Papers</a></body></html>
"""

TABLE = re.compile(r"<table>.*?</table>", re.S)


def body(src: pathlib.Path) -> str:
    html = markdown.markdown(src.read_text(encoding="utf-8"), extensions=EXTENSIONS).strip()
    return TABLE.sub(lambda m: f'<div class="tw">{m.group(0)}</div>', html)


def page(root: pathlib.Path) -> str:
    buttons = "".join(f'<button id="btn-{v}">{label}</button>' for v, label, _, _ in VIEWS)
    docs = "\n".join(
        f'<section class="doc" id="doc-{v}" hidden>\n<p class="ai">AI-generated</p>\n{body(root / src)}\n</section>'
        for v, _, src, _ in VIEWS
    )
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Papers &middot; xai-chess</title>\n{ICONS}\n<style>\n{STYLE}</style>\n</head>\n<body>\n"
        '<a class="back" href="index.html">&larr; xai-chess</a>\n<h1>Papers</h1>\n'
        f'<div class="mode">{buttons}</div>\n{docs}\n<script>\n{SCRIPT}</script>\n</body>\n</html>\n'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify without writing")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    targets = {OUT: page(root)}
    targets.update({old: STUB.format(view=v) for v, _, _, old in VIEWS})
    stale = []
    for rel, html in targets.items():
        out = root / rel
        if out.exists() and out.read_text(encoding="utf-8") == html:
            print(f"ok     {rel}")
            continue
        stale.append(rel)
        if args.check:
            print(f"STALE  {rel}")
        else:
            out.write_text(html, encoding="utf-8")
            print(f"wrote  {rel}  ({len(html):,} bytes)")

    if args.check and stale:
        print(f"\n{len(stale)} page(s) stale; run without --check", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
