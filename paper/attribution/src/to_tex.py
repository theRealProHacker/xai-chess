"""Convert src/part1-4.md (with [@key] citations) to ../paper.tex (IEEEtran, references.bib).

Handles the markdown subset the paper uses: headings, paragraphs, numbered and bulleted
lists, pipe tables with a bold "Table X" label + italic caption + optional "*Note.*",
images with a bold "Figure N." caption, $...$ / $$...$$ math, `code`, **bold**, *italic*.
Layout-only choices (table widths, equation line breaks) live in TABLES and EQ_BREAKS.
"""
import re, sys
from pathlib import Path

here = Path(__file__).parent
src = "\n".join((here / f"part{i}.md").read_text() for i in (1, 2, 3, 4))

# Table layout: id -> (environment, column spec, font size). X = tabularx wrapping column.
P = lambda w: f">{{\\raggedright\\arraybackslash}}p{{{w}}}"
X = r">{\raggedright\arraybackslash}X"
TABLES = {
    "A":  ("table*", f"l{P('2.2cm')}{X}{P('3.2cm')}{P('1.8cm')}{P('2.6cm')}", r"\footnotesize"),
    "1":  ("table*", f"{P('4.6cm')}{X}", r"\footnotesize"),
    "2":  ("table*", "l" + "r" * 11, r"\footnotesize"),
    "3":  ("table*", f"{P('2.4cm')}{X}{X}{X}", r"\footnotesize"),
    "4":  ("table*", f"l{X}{X}{P('3.4cm')}", r"\footnotesize"),
    "5":  ("table", "lrr", r"\footnotesize"),
    "6":  ("table*", f"{X}{X}{X}", r"\footnotesize"),
    "B1": ("table", f"l{X}{P('3.6cm')}", r"\small"),
    "C1": ("landscape", "l" + X * 10, r"\scriptsize\setlength{\tabcolsep}{3pt}"),
    "D1": ("table", "l" + "c" * 7, r"\small"),
    "D2": ("table", "l" + "c" * 5 + X, r"\small"),
}
# Appendices typeset on a landscape page (their single wide table closes the environment).
LANDSCAPE_APPENDICES = {"C"}
# Display equations too wide for their column: first characters -> replacement body.
EQ_BREAKS = {
    r"\phi^{(L)}_n=\tfrac12": r"""\begin{aligned}\phi^{(L)}_n&=\tfrac12\Big[g_L(q+n'+\varepsilon)-g_L(q+n+\varepsilon)\\
&\qquad+g_L(q+n'+\varepsilon')-g_L(q+n+\varepsilon')\Big],\\
\phi^{(L)}_\varepsilon&=\Delta^{\rm true}-\phi^{(L)}_n.\end{aligned}""",
    r"h^{(S)}_{L,s}=": r"""\begin{aligned}h^{(S)}_{L,s}=q_{L,s}&+\sum_{G\in S}n^G_{L,s}(x)+\sum_{G\notin S}n^G_{L,s}(x_0)\\
&+\mathbf 1[\varepsilon\in S]\,\varepsilon_{L,s}(x)+\mathbf 1[\varepsilon\notin S]\,\varepsilon_{L,s}(x_0),\end{aligned}""",
    r"\mathrm{IG}_G=\int_0^1": r"""\begin{gathered}\mathrm{IG}_G=\int_0^1\sum_s\big\langle\nabla_{h_{e,s}}y(\gamma(\alpha)),\ W^G_e(\tilde f^G_s-\tilde f^G_{0,s})\big\rangle\,d\alpha,\\
\sum_G\mathrm{IG}_G+\mathrm{IG}_\varepsilon=y(x)-y(x_0).\end{gathered}""",
}

placeholders = []


def hold(s):
    placeholders.append(s)
    return f"\x00{len(placeholders) - 1}\x00"


def code(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("{", r"\{"), ("}", r"\}"), ("_", r"\_"), ("#", r"\#"), ("%", r"\%"),
                 ("&", r"\&"), ("$", r"\$"), ("~", r"\textasciitilde{}")]:
        s = s.replace(a, b)
    s = s.replace(r"\textbackslash\{\}", r"\textbackslash{}")
    s = re.sub(r"([/._-])", r"\1\\allowbreak{}", s)
    s = s.replace("\\\\allowbreak", "\\allowbreak")
    return r"\texttt{" + s + "}"


def math(s):
    return s.replace(r"\mathbb 1", r"\mathbf 1")


def inl(s):
    s = re.sub(r"`([^`]+)`", lambda m: hold(code(m.group(1))), s)
    s = re.sub(r"\$([^$]+)\$", lambda m: hold("$" + math(m.group(1)) + "$"), s)
    s = re.sub(r"\s*\[@(\w+)\]", lambda m: hold(r"~\cite{" + m.group(1) + "}"), s)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("~", r"$\sim$")]:
        s = s.replace(a, b)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    s = s.replace("/", r"/\allowbreak{}")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    for a, b in [("–", "--"), ("—", "---"), ("∘", r"$\circ$"), ("→", r"$\to$\allowbreak{}"), ("…", r"\ldots{}"),
                 ("−", "$-$"), ("÷", r"$\div$"), ("×", r"$\times$")]:
        s = s.replace(a, b)
    while "\x00" in s:
        s = re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], s)
    return s


def display(eq):
    for k, v in EQ_BREAKS.items():
        if eq.startswith(k):
            return "\\[\n" + v + "\n\\]"
    eq = math(eq)
    parts = re.split(r"(?<=[,;])\\q?quad\s*", eq)
    if len(parts) > 1:
        return "\\[\n\\begin{gathered}" + "\\\\\n".join(parts) + "\\end{gathered}\n\\]"
    return "\\[\n" + eq + "\n\\]"


def cells(row):
    return [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]


def table(tid, caption, rows, note):
    env, spec, size = TABLES[tid]
    landscape = env == "landscape"
    env = "table" if landscape else env
    head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    width = (r"\linewidth" if env == "landscape" else
             r"\textwidth" if env == "table*" or appendix else r"\columnwidth")
    tab = "tabularx" if "X" in spec else "tabular"
    out = [rf"\begin{{{env}}}" + ("[H]" if appendix else "[!t]"), r"\centering", rf"\renewcommand{{\thetable}}{{{tid}}}",
           rf"\caption{{{inl(caption)}}}", rf"\label{{tab:{tid}}}", size]
    out.append(rf"\begin{{tabularx}}{{{width}}}{{{spec}}}" if tab == "tabularx" else rf"\begin{{tabular}}{{{spec}}}")
    out += [r"\toprule", " & ".join(inl(c) for c in head) + r" \\", r"\midrule"]
    out += [" & ".join(inl(c) for c in r) + r" \\" for r in body]
    out += [r"\bottomrule", rf"\end{{{tab}}}"]
    if note:
        out += [r"\par\smallskip", rf"\parbox{{{width}}}{{\footnotesize {inl(note)}}}"]
    out.append(rf"\end{{{env}}}")
    if landscape:
        out.append(r"\end{landscape}")
    return "\n".join(out)


lines = src.split("\n")
out, i, appendix, counters = [], 0, False, [0, 0]
title = inl(lines[0][2:])
author = lines[2].strip()
i = 3
bib_preface = ""
while i < len(lines):
    ln = lines[i]
    if not ln.strip():
        i += 1
        continue
    if ln.startswith("## Abstract"):
        i += 1
        abstract, kw = [], ""
        while not lines[i].startswith("## "):
            t = lines[i].strip()
            if t.startswith("*Keywords:*"):
                kw = t[len("*Keywords:*"):].strip()
            elif t:
                abstract.append(inl(t))
            i += 1
        out += [r"\begin{abstract}", "\n\n".join(abstract), r"\end{abstract}", "",
                r"\begin{IEEEkeywords}", inl(kw), r"\end{IEEEkeywords}", ""]
        continue
    if ln.startswith("## ") or ln.startswith("### "):
        lvl = 2 if ln.startswith("## ") else 3
        text = ln[lvl + 1:].strip()
        if lvl == 2 and text.startswith("Appendix "):
            m = re.match(r"Appendix ([A-Z])\. (.+)", text)
            if not appendix:
                out += ["", r"\onecolumn", r"\appendices"]
                appendix = True
            if m.group(1) in LANDSCAPE_APPENDICES:
                out.append(r"\begin{landscape}")
            out.append(rf"\section{{{inl(m.group(2))}}}\label{{app:{m.group(1)}}}")
        elif text == "References":
            j = i + 1
            while not lines[j].strip():
                j += 1
            bib_preface = inl(lines[j].strip())
            i = j + 1
            continue
        elif m := re.match(r"(\d+)(?:\.(\d+))?\.? (.+)", text):
            if lvl == 2:
                counters = [counters[0] + 1, 0]
                assert int(m.group(1)) == counters[0], text
                out.append(rf"\section{{{inl(m.group(3))}}}\label{{sec:{m.group(1)}}}")
            else:
                counters[1] += 1
                assert (int(m.group(1)), int(m.group(2))) == tuple(counters), text
                out.append(rf"\subsection{{{inl(m.group(3)).replace('$\\circ$', '\\texorpdfstring{$\\circ$}{o}')}}}\label{{sec:{m.group(1)}.{m.group(2)}}}")
        else:
            out.append(rf"\section*{{{inl(text)}}}")
        i += 1
        continue
    if ln.strip() == "<!--REFERENCES-->":
        out += [r"\bibliographystyle{IEEEtran}", r"\bibliography{references}"]
        i += 1
        continue
    if m := re.fullmatch(r"\*\*Table (\w+)\*\*", ln.strip()):
        tid = m.group(1)
        caption = lines[i + 1].strip().strip("*")
        j = i + 2
        while not lines[j].startswith("|"):
            j += 1
        rows = []
        while j < len(lines) and lines[j].startswith("|"):
            rows.append(lines[j])
            j += 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        note = None
        if lines[j].startswith("*Note.*"):
            note, j = lines[j].strip(), j + 1
        out += [table(tid, caption, rows, note), ""]
        i = j
        continue
    if m := re.fullmatch(r"!\[Figure (\d+)\]\((.+?)\)", ln.strip()):
        num, path = m.groups()
        j = i + 1
        while not lines[j].strip():
            j += 1
        cap = re.sub(rf"^\*\*Figure {num}\.\*\*\s*", "", lines[j].strip())
        out += [r"\begin{figure*}[!t]", r"\centering", rf"\renewcommand{{\thefigure}}{{{num}}}",
                rf"\includegraphics[width=0.92\textwidth]{{{path}}}", rf"\caption{{{inl(cap)}}}",
                rf"\label{{fig:{num}}}", r"\end{figure*}", ""]
        i = j + 1
        continue
    if ln.startswith("$$"):
        out += [display(ln.strip()[2:-2]), ""]
        i += 1
        continue
    if re.match(r"(\d+\. |- )", ln):
        env = "enumerate" if ln[0].isdigit() else "itemize"
        out.append(rf"\begin{{{env}}}")
        while i < len(lines) and re.match(r"(\d+\. |- )", lines[i]):
            out.append(r"\item " + inl(re.sub(r"^(\d+\. |- )", "", lines[i])))
            i += 1
        out += [rf"\end{{{env}}}", ""]
        continue
    para = []
    while i < len(lines) and lines[i].strip() and not re.match(r"(#|\$\$|\*\*Table|!\[|\d+\. |- |<!--)", lines[i]):
        para.append(lines[i].strip())
        i += 1
    if not para:
        sys.exit(f"stuck at line {i}: {ln!r}")
    out += [inl(" ".join(para)), ""]

preamble = rf"""% Generated by src/to_tex.py from src/part1-4.md. Compile: latexmk -pdf paper.tex
\documentclass[conference]{{IEEEtran}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs,tabularx,array}}
\usepackage{{cite}}
\usepackage{{pdflscape,float}}
\usepackage{{url}}
\usepackage[hidelinks]{{hyperref}}
% Section numbers as in the markdown source ("Section 5.2"), not IEEE Roman numerals.
\renewcommand{{\thesection}}{{\arabic{{section}}}}
\renewcommand{{\thesectiondis}}{{\arabic{{section}}.}}
\renewcommand{{\thesubsection}}{{\thesection.\arabic{{subsection}}}}
\renewcommand{{\thesubsectiondis}}{{\thesection.\arabic{{subsection}}}}
\renewcommand{{\tablename}}{{Table}}
% Tier legend as the first, unnumbered line of the reference list.
\makeatletter
\let\origthebibliography\thebibliography
\renewcommand{{\thebibliography}}[1]{{\origthebibliography{{#1}}\item[]\hskip-\labelsep\hskip-\leftmargin {bib_preface}}}
\makeatother

\begin{{document}}
\title{{{title}}}
\author{{\IEEEauthorblockN{{{author}}}}}
\bstctlcite{{IEEEexample:BSTcontrol}}
\maketitle

"""
Path(here.parent / "paper.tex").write_text(preamble + "\n".join(out) + "\n\\end{document}\n")
print("wrote paper.tex")
