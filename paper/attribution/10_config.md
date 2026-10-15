# Paper Configuration Record (academic-paper Phase 0)

Decided without an interview (the user's goal directive says proceed without pausing); defaults copied from the project's earlier paper/survey.* and the user's standing preferences.

- Mode: full. Phase 1 (literature_strategist) SKIPPED: research done in deep-research (01-07 in this folder).
- Paper type: methods survey + position paper with a preregistered (not run) evaluation protocol. Hybrid: surveyed evidence vs the project's own measured numbers kept separate.
- Discipline: machine learning interpretability (mechanistic interpretability, XAI), chess networks.
- Target venue style: workshop/journal-length ML paper (e.g. TMLR / NeurIPS workshop). ~8,000-10,000 words main text + appendix.
- Citation format: IEEE numeric (as PAPERS.md and paper/survey.tex). Only sources from 02_biblio_T*.md / 07_report_final.md reference list.
- Output: markdown (paper.md) + LaTeX IEEEtran (paper.tex) + BibTeX (references.bib), in paper/attribution/; pdflatex available.
- Language: English. Abstract: English only (as the earlier survey; the zh-TW abstract is omitted, noted as a deviation).
- Author line: "xai-chess project" (as earlier survey). Mandatory statements: data availability, ethics, CRediT, COI, funding (none), AI disclosure (Claude, Anthropic, via Claude Code).
- Style: terse. User calls verbose prose "AI slop": every sentence a claim; no throat-clearing; minimal em dashes; no hype. User pushes back on over-claiming: nothing beyond the numbers in 00_context.md has been run; all method outcomes are hypotheses.
- Must answer the user's question directly: several approaches for feature attribution (a) through the decoder at the embedding, (b) through encoders at any layer, (c) combining several decoders/encoders across layers; plus a recommendation of what to try first.
- Core source: 07_report_final.md (revised after editor/ethics/DA review; see 07_response.md). Do not re-introduce issues fixed there.
