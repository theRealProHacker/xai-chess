#!/usr/bin/env python3
"""Regenerate the published docs/ pages from their Markdown sources.

    python3 tools/build-docs.py          # write
    python3 tools/build-docs.py --check  # verify only, non-zero exit if stale

Reproduces the existing pages byte-for-byte with python-markdown 3.5.2.
`toc` supplies the heading slugs, `smarty` the &rsquo;/&ldquo; entities.

Each page's <head> is hand-tuned (survey and summaries differ in max-width,
heading sizes, and table striping), so an existing page's head is preserved
verbatim and only its <body> is replaced. DEFAULT_STYLE applies to new pages.
"""
import argparse
import pathlib
import re
import sys

import markdown

EXTENSIONS = ["extra", "toc", "smarty"]

SPLIT = re.compile(r"^(.*?</style></head>\n<body>)(.*)(</body></html>\s*)$", re.S)

DEFAULT_STYLE = """body{max-width:820px;margin:2.5rem auto;padding:0 1.3rem;
 font:18px/1.65 Georgia,'Times New Roman',serif;color:#1a1a1a;background:#fafafa}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 .3rem;border-bottom:3px solid #333;padding-bottom:.4rem}
h2{font-size:1.4rem;margin:2.2rem 0 .6rem;border-bottom:1px solid #ccc;padding-bottom:.2rem}
h3{font-size:1.15rem;margin:1.6rem 0 .4rem;color:#333}
p{margin:.7rem 0}em{color:#444}
table{border-collapse:collapse;margin:1.2rem 0;font-size:.95rem;width:100%}
th,td{border:1px solid #bbb;padding:.45rem .7rem;text-align:left}
th{background:#eee}tr:nth-child(even) td{background:#f3f3f3}
code{background:#eee;padding:.1rem .3rem;border-radius:3px;font-size:.9em}
a{color:#0b5fa5}hr{border:none;border-top:1px solid #ccc;margin:2rem 0}
ol,ul{padding-left:1.6rem}ol li{margin:.35rem 0}
"""

# (source markdown, output page, <title> used only when creating the page)
PAGES = [
    ("paper/survey.md", "docs/survey.html", "xai-chess survey"),
    ("RESEARCH_SUMMARIES.md", "docs/RESEARCH_SUMMARIES.html", "xai-chess deep-research summaries"),
    ("paper/landscape.md", "docs/landscape.html", "xai-chess move-explanation landscape"),
]


def default_head(title: str) -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{title}</title><style>\n{DEFAULT_STYLE}</style></head>\n<body>"
    )


def render(src: pathlib.Path, out: pathlib.Path, title: str) -> str:
    """Markdown -> full page, preserving `out`'s existing head if it has one."""
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=EXTENSIONS).strip()
    head, tail = default_head(title), "</body></html>"
    if out.exists():
        m = SPLIT.match(out.read_text(encoding="utf-8"))
        if m:
            head, tail = m.group(1), m.group(3)
        else:
            print(f"warn   {out}: unrecognised layout, rewriting head", file=sys.stderr)
    return head + body + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify without writing")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    stale = []
    for src_rel, out_rel, title in PAGES:
        src, out = root / src_rel, root / out_rel
        if not src.exists():
            print(f"skip   {out_rel}  (no {src_rel})")
            continue
        html = render(src, out, title)
        if out.exists() and out.read_text(encoding="utf-8") == html:
            print(f"ok     {out_rel}")
            continue
        stale.append(out_rel)
        if args.check:
            print(f"STALE  {out_rel}")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
            print(f"wrote  {out_rel}  ({len(html):,} bytes)")

    if args.check and stale:
        print(f"\n{len(stale)} page(s) stale; run without --check", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
