# Phase 4b: writer paper-visible drafting and self-scoring

Contract: `academic_paper/writer_full/v1` (re-injected). Phase 4a output: `13_phase4a.md`. Inputs: `10_config.md`, `11_outline.md`, `12_argument.md`, `07_report_final.md`, `07_response.md`, `02_biblio_T*.md`, `lc0-probe/results_square_decoder.json`.

## Draft Body

Draft body: `paper.md`, 9,165 words main text (Sections 1–13), 4,047 words appendices A–E, 254-word abstract, 89 numbered references.

Word-count convention: whitespace split (`shared/references/word_count_conventions.md`), tables and inline math included, with markdown markup tokens excluded (table pipes, separator rows, bare `$$`). Statements and references not counted.

| Section | Target | Actual | Deviation |
|---|---|---|---|
| Abstract (not counted) | 250 | 254 | +2% |
| 1 Introduction | 650 | 575 | −12% |
| 2 Answer in brief | 750 | 759 | +1% |
| 3 System, data, notation | 750 | 650 | −13% |
| 4 Literature constraints | 1,000 | 1,142 | +14% |
| 5 Front/back re-derived | 850 | 889 | +5% |
| 6 (a) Embedding decoder | 550 | 478 | −13% |
| 7 (b) Any layer | 1,050 | 1,151 | +10% |
| 8 (c) Across layers | 900 | 766 | −15% |
| 9 Identifiability, coverage | 850 | 726 | −15% |
| 10 Protocol, first experiment | 950 | 983 | +3% |
| 11 Discussion | 600 | 609 | +2% |
| 12 Limitations | 350 | 306 | −13% |
| 13 Conclusion | 150 | 131 | −13% |
| **Main text** | **9,400** | **9,165** | **−2.5%** |
| A Method cards | 3,000 | 2,560 | −15% |
| B Controls | 200 | 198 | −1% |
| C Taxonomy | 300 | 336 | +12% |
| D Worked example | 500 | 494 | −1% |
| E Survey method | 400 | 459 | +15% |
| **Appendices** | **4,400** | **4,047** | **−8%** |

Sections 8, 9 and Appendices A and E sit at the ±15% edge.

Files produced:
- `paper.md`: the paper.
- `fig_decoder.py` → `fig_decoder.png` (Fig. 2, from `results_square_decoder.json`; validated colorblind-safe palette, markers as secondary encoding; checked visually).
- `fig_compositions.py` → `fig_compositions.png` (Fig. 3, schematic of the four compositions; checked visually).
- Fig. 1 (outline) replaced by the display equation of Section 3.2 plus Table A in Section 2. Fig. 4 (optional flowchart) omitted, since the decision rules in Section 10.5 are a four-item list.
- Keyed source (`[@key]` citations) and the IEEE numbering script are in the session scratchpad (`scratchpad/paper/part1–4.md`, `refs.py`, `build.py`). `paper.md` is generated from them.

Deviations from skill defaults, declared:
- The `<!--ref:slug--><!--anchor:…-->` markers are not emitted. The bibliographies carry no slugs, and the caller asked for a clean paper. This matches `07_report_final.md`.
- Figure numbering keeps the outline's IDs (Figs. 2 and 3). No figure is numbered 1.

## Dimension Scores

### D1: section_completeness
**pass.** All 13 main sections, Appendices A–E and the six mandatory statements in `11_outline.md` are written. Every outline subsection is present except two planned figures: Fig. 1 is replaced by the Section 3.2 equation plus Table A, and optional Fig. 4 is omitted, both declared above. No `[MATERIAL GAP]` placeholders. Table A, Table 3 and Tables 4–7 exist as the outline specifies. Appendix tables were renumbered B1, C1, D1, D2. Two subsections were merged: the M10 summary sits inside Section 7.2, and the 07 Table 3 controls sit in Appendix B with a main-text pointer in Section 2.

### D2: citation_density
**pass.** 89 sources, all from `07_report_final.md`'s list (itself drawn from `02_biblio_T*.md`), each cited at least once; no new sources. Every surveyed claim carries an IEEE number. Own measurements point to Tables 1, 2, 6 or Fig. 2. Own analytic results are marked "follows from" or "by construction", with the derivations in Appendix A. Hypotheses are marked "predicts" or "has not checked". Three facts rest on project files rather than the literature (ONNX graph reads, `lc0_planes.py`, commit 473b3ce), and the text names each file. The tier marks for Tier 2–4 sources appear in the reference list only (07_response #38).

### D3: argument_blueprint_fidelity
**pass.** SA1 is covered in §5.2 and §6 (C1.1–C1.7, with C1.6 stated as an unverified prediction). SA2 is in §4.1–4.4 and §7 (C2.1–C2.12, with M0 as floor and the policy split in §7.4). SA3 is in §4.5 and §8 (C3.1–C3.8). SA4 is in §9.1 and SA5 in §9.2–9.4, with the error = residual-share derivation in Appendix A. SA6 is in §10.5 with decision rules. The counter-arguments are each met. CA1 and CA2 are answered in §7.2, §9.1 and §11.2, CA3 in §6.3 and §11.1, CA4 in §7.3 (M3), CA5 in §4.6, §11.3 and C4, CA6 in §7.2 (γ, C8), and CA7 in §8.4 and §11.3. CA8 and CA10 are in §1 and §12, CA9 in C7 and H6, CA11 in Table 7, and CA12 in §4.1 and §11.2. All fixes in `07_response.md` are kept:
- oriented ρ_L;
- M7 linear only between nodes;
- smolgen frozen with patterns;
- ln1 taps;
- castling on the king;
- the working note quoted verbatim and not corrected;
- Kulkarni only in M12;
- Jenner "use head ablation in their evidence";
- AxBench follow-up unverified;
- Lin's EV exception;
- 89 distinct works.

The drafting notes are also followed. Coverage is never called faithfulness, and "pin credit" is defined as the credit of the changed families.

### D4: total_word_count
**pass.** Main text has 9,165 words. That is −2.5% against the outline's 9,400, inside the ±10% band (8,460–10,340) and inside the config's 8,000–10,000. The appendices have 4,047 words, −8% against 4,400 and inside 3,960–4,840.

### D5: per_section_word_count
**pass.** All 13 main sections and all five appendices are within ±15% of their `11_outline.md` allocations (table above). Section 8 (−15%), Section 9 (−15%), Appendix A (−15%) and Appendix E (+15%) are at the edge. Appendix A is below target because the card content the main text already defines (M2, M6, M8∘M2 regression) is referenced, not repeated. The counting convention is fixed above and applied uniformly. No section was offset against another: each was adjusted against its own band.

### D6: paragraph_structure
**pass.** Discursive paragraphs in §§1–11 open with their claim, give a citation or a table/figure reference, explain it, and link onward. This is about 85% of body paragraphs, counted by hand on a pass through the draft. The non-TEEL blocks are procedural and listed in the outline: the contributions list, the five consequences in §5.2, the seven causes in §5.4, the hypotheses, the decision rules, the method cards and the Appendix D steps. Every section ends with a transition naming the next section (§1→2 through §12→13). Style checks:
- 0 em dashes;
- 26 prose semicolons in about 13,200 words (≤2/1000), one of them inside the verbatim working-note quote;
- none of the 25 flagged terms;
- no throat-clearing openers;
- "X, not Y" contrasts kept only where they state a technical distinction (for example, sensitivity map vs complete attribution).

### D7: register_consistency
**pass.** Terms are defined once and then used consistently:
- Section 3.2 defines decoder, probe, encoder direction, attribution, tap, named part, residual and oriented pair.
- Section 7.2 defines coverage ρ_L, and Section 9.3 generalises it to Cov.

Chess terms are glossed for ML readers where they carry weight: absolute pin in §10.5, interposition and pinner-off-line edits in §10.6, FEN and fen-only history in §3.1. The prose uses full forms, the third person ("this paper") and present tense for analytic claims. Measured numbers are stated in the past tense or as table values. No colloquialisms appear. The working note's "->" and "deep jungle of attention" occur only inside the verbatim quote.

## Failure Condition Checks

### F1
did not fire. No mandatory dimension (D1, D2, D3) scores block.

### F4
did not fire. No mandatory dimension scores warn.

### F2
did not fire. No high-priority dimension (D4, D5) scores block.

### F3
did not fire. No high-priority dimension scores warn. D5 has four sections at the ±15% edge, all inside the band.

### F0
fired. Every mandatory dimension (D1, D2, D3) scores pass.

## Writer Decision

writer_decision=accept
