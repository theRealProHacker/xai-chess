# Paper Outline (academic-paper Phase 2, structure_architect_agent)

Inputs: `10_config.md` (binding), `07_report_final.md` (core), `07_response.md`, `03_synthesis.md`, `01_scoping.md`, `02_biblio_T1–T8.md`. Phase 1 skipped (research done in deep-research). Approval gate skipped by directive (no user interaction).

## Structure pattern

Hybrid: Pattern 3 (Theoretical/methods analysis) + Pattern 2 (thematic survey), answer-first. Not IMRaD: nothing beyond the existing fits has been run. The user's three-part question is answered in Section 2, before background; Sections 6–8 expand one part each; per-method definitions go to Appendix A so the main text reads in one pass.

Working title: *Attributing a Chess Transformer's Outputs to a Fixed Human Feature Vocabulary: Decoders, Probes, and What the Vocabulary Misses* (kept from `07_report_final.md`).

## Overview

The paper opens with the dilemma in the project's own words (front: fit good, path long; back: fit poor, path short) and the measured fits. Section 2 gives the whole answer on one page: (a) at the embedding, decoder attribution is input saliency with feature names; (b) at any layer, decoder substitution gives per-edit attribution and probe-side erasure/interchange test use; (c) four compositions combine layers; first experiment M2 + M6 on pin edits. Sections 3–5 supply the setting, the literature constraints and the re-derivation of the front/back question that justifies (a). Sections 6–8 develop (a), (b), (c), each with method summaries pointing to Appendix A. Section 9 states what is identifiable and why the residual is reported as coverage. Section 10 is the preregistered protocol and the first experiment with decision rules and the worked example. Discussion answers the strongest objection; Limitations state that no method has been run.

## Detailed outline

### Abstract (~250, not counted)
Structured: Background (vocabulary, maps, FVU 0.002 → 0.680) / Purpose / Method (survey of 89 works, taxonomy, ONNX checks) / Answer (a), (b), (c) in one sentence each / Recommendation (M2 + M6 on pin edits, four taps, C1 + C3) / "Nothing beyond existing fits has been run; method outcomes are hypotheses." English only (zh-TW omitted, stated in config).

### 1. Introduction (~650)
**Purpose**: state the problem, the existing maps, the user's dilemma, the question and the contributions.
- 1.1 Problem and existing maps: extractor (30,837 bits/square, ~36.5 on), 20k puzzle FENs, probes and decoders at e, L0–L9.
- 1.2 The dilemma: quote the working note verbatim (back "near-perfect reconstruction … mostly square identity", "deep jungle of attention"; front clearer; hybrid?). State measured square-only and features+square FVU at e and L9 without comment (07_response #29: do not correct the user).
- 1.3 Question (primary RQ from 07 §1) and the user's three sub-questions (a)/(b)/(c) + "what to try first".
- 1.4 Contributions, one line each: locality reduction at e; one decomposition read by five removal operators; four cross-layer compositions incl. M8∘M2; identifiability statement; residual-as-coverage; preregistered first experiment with decision rules. Last line: "The paper surveys and proposes; it tests nothing."
**Sources**: none beyond project; Table 2 numbers.
**Transition**: "The answer first; the evidence after."

### 2. The answer in brief (~750)
**Purpose**: the reader gets the full three-part answer and the first experiment here.
- 2.1 (a) Through the embedding decoder: the embedding is per-square local and nonlinear; M1 = input saliency re-expressed in the vocabulary; use local vocabulary C2; named share ≈1 by construction; C1 tests what FVU 0.002 says about learning. → §6.
- 2.2 (b) Through decoders and encoders at any layer: decoder side M2 (substitution), M6 (named/residual), M9 (family Shapley + error player); encoder side M3 (LEACE erasure vs readability), M4 (concept gradient), M5 (DAS/IIA). M0 surrogate is the floor. Policy: exact four-player split at `/encoder9/ln2`. → §7.
- 2.3 (c) Combining layers: M6 sweep (descriptive), M8 erasure cascade (recomputation), M8∘M2 feature paths (attention live), M7 supervised replacement graph (attention frozen); M12 residual budget, M11 disagreement map. → §8.
- 2.4 What is identifiable: joint effect of an edit, γ_L, ρ_L yes; split among collinear families no. → §9.
- 2.5 First experiment: M2 + M6 on ~1,000 pin edits, taps e(C2)/L2/L5/L9, C1 + C3, M0 floor, ~1 CPU hour under ONNX; decision rules promote M8 or M5. → §10.5.
**Figure 1** (new schematic): token split h_L = q + Σ_G n^G + ε at each tap; three entry routes (a) e, (b) any L via decoder or probe, (c) arrows across taps; output y. **Table A** (new, compact): sub-question → methods → what each returns → identifiable? → ONNX or port → cost.
**Sources**: pointers only.
**Transition**: "Sections 3–5 give the setting and the constraints these answers rest on."

### 3. System, data and notation (~750)
**Purpose**: fix the model, inputs, targets and the canonical split.
- 3.1 Model and taps: t1-256x10, 64 tokens, d=256, post-LN, smolgen; taps `/attn_body/rehape`, `/encoder{i}/ln1` (post-attention, pre-FFN), heads read `/encoder9/ln2`; ONNX on CPU, no backward pass.
- 3.2 Inputs: fen-only history fill, broadcast planes; castling on king square (473b3ce); targets y = ℓ_W − ℓ_L at `/value/dense2`, policy pre-softmax logit.
- 3.3 Notation: x, x′, x_0, B; f, f̃; ~30 families; D_L = q + W f̃; n^G, ε; Δ^true; oriented pairs; q cancels pairwise, dropped in point form.
**Table 1** existing results (probes, MLP decoder, board-level decoder FVU 0.52–0.62, value-head-on-decoded-L9 weak, pin probe 0.945 vs 0.839). Causal operators MaskPinner/LineBlocker named here.
**Sources**: project only (own measurement).
**Transition**: notation in hand, what the literature allows.

### 4. What the literature constrains (~1,000)
**Purpose**: eight constraints that shape every method; one paragraph each, no method catalogue.
- 4.1 Removal rule decides meaning (three removal rules here: bit flip, decoded substitution, board edit).
- 4.2 Readable is not used.
- 4.3 Residual is structured; belongs in the report.
- 4.4 Fixed vocabulary → supervised maps by default; SAEs on the residual.
- 4.5 Cross-layer maps exist only unsupervised; all assume additive stream.
- 4.6 Identifiability limits every map; prescription: ~30 families, linear primary, MLP beside C1, report α-stability.
- 4.7 Attention breaks gradients; smolgen uncovered.
- 4.8 Depth profiles in game networks; per-layer reading must be chained to y.
- Survey method in two sentences (structured survey, eight themes, 89 works, tiers) → Appendix E.
**Sources**: see Evidence Map rows 4.x.
**Transition**: constraints 4.2 and 4.6 bear directly on the front/back numbers; re-derive them.

### 5. The front/back question re-derived (~850)
**Purpose**: justify answer (a) and the choice of middle taps.
- 5.1 Two normalisations (features add vs partial R²); partial R² 0.99 / 0.68 / 0.76 / falls every tap L2→L9. **Table 2** all eleven taps. **Figure 2**: square-only, features-only, features+square FVU and partial R² by tap (regenerate from `lc0-probe/results_square_decoder.json`; existing `lc0-probe/square_decoder_ridge_mlp.png` as reference).
- 5.2 Embedding is local and nonlinear: ONNX formula; five consequences (occupancy; broadcast into ε_e; non-local families spurious with cancelling ε; ρ_e ≈ 1 under C2; random init equally local → C1 prediction).
- 5.3 Coverage against path length (one paragraph).
- 5.4 Seven candidate causes of decay (a)–(g), as a list; only output-unit methods settle (d)–(f).
**Sources**: project ONNX graph, `lc0_planes.py`; Hewitt & Liang; Adebayo; Sutter.
**Transition**: (5.2) fixes what can be done at e → §6.

### 6. (a) Attribution through the embedding decoder (~550)
**Purpose**: what M1 computes, what it means, how to make it admissible.
- 6.1 Point form (sensitivity map, q unassigned) and board-path IG (complete); both reduce to input saliency on the square's planes.
- 6.2 Admissibility: C2 local vocabulary or zero decoded deltas on unchanged-plane squares; conservation breaks through ten post-LN blocks (Ali); needs PyTorch port.
- 6.3 What it can and cannot say; H1.
**Detail** → Appendix A (M1).
**Sources**: Sundararajan 2017; Lundstrom 2022; Ali 2022; Adebayo 2018.
**Transition**: real attribution needs taps where locality no longer holds → §7.

### 7. (b) Attribution through decoders and encoders at any layer (~1,050)
**Purpose**: two directions, three questions each, one baseline.
- 7.1 Baseline M0 (surrogate + family Shapley + residual; floor; C7 mandatory).
- 7.2 Decoder direction: M2 decoded substitution (γ_L consistency), M6 named/residual (two-player Shapley, ρ_L as tap guide, oriented pairs), M9 family Shapley with error player (pairwise vs distributional). M10 one line (relevance chain, port, untested rules).
- 7.3 Encoder direction: M3 LEACE erasure with use-vs-readability curve, M4 concept gradient (family level only, whitened), M5 DAS/IIA with Isolate; C1 decisive.
- 7.4 Policy: bilinear logit from from/to tokens at ln2; exact four-player split; needs ln2 decoder (not fitted); value-vs-policy ρ_L tests cause (f).
**Table 3** (main text, compact): method → question answered → residual reported → runs under ONNX. Full nine-axis taxonomy (07 Table 4) → Appendix C.
**Detail** → Appendix A (M0, M2–M6, M9, M10).
**Transition**: each of these reads one tap; combining taps → §8.

### 8. (c) Combining decoders and encoders across layers (~900)
**Purpose**: the four compositions, what each buys, what each costs.
- 8.1 Sweep: M6 at every tap; κ_L descriptive, not a flow; tied-decoder variant.
- 8.2 Intervene-and-read: M8 erasure/substitution cascade, transfer tensor T^{L→L′}, recomputation diagonal, Δφ_ε.
- 8.3 Feature paths with attention live: M8∘M2 (path patching in decoder coordinates; non-additive; residual branch π_{G→r}).
- 8.4 Replacement graph with attention frozen: M7 (supervised skip transcoders, error nodes, linear edges between nodes at frozen activations; smolgen frozen with patterns; conflict with Jenner head ablation).
- 8.5 Residual and comparison across layers: M12 budget (non-local / nonlinear / carried / remainder) and hybrid dictionary; M11 disagreement map (no learned ensemble: unidentified).
**Figure 3** (new): one diagram of the four compositions on a tap axis e…L9→y, frozen vs live attention marked.
**Detail** → Appendix A (M7, M8, M8∘M2, M11, M12).
**Transition**: all composite outputs inherit the decoder's collinearity → §9.

### 9. Identifiability and the residual as coverage (~850)
**Purpose**: state what numbers are meaningful and how to grade complete methods.
- 9.1 Identifiable / not identifiable list; Table 7 evidence (331 bits, 180 route for one interposition); remedy (attribute to edit; merge unstable families).
- 9.2 For complete pairwise methods, faithfulness error = residual share (derivation).
- 9.3 One coverage definition Cov over oriented pairs, instantiated for M6, M2, pairwise M9, M0; distributional M9 kept off the axis.
- 9.4 Three residual measures, three questions (how much / why / nameable?); "nothing nameable" an admissible outcome.
**Sources**: Abraham 2022 (ICaCE); Aas 2021; Havasi 2022; Schut 2025; Sutter 2025; Beckers & Halpern 2019.
**Transition**: protocol that grades all this → §10.

### 10. Evaluation protocol and first experiment (~950)
**Purpose**: preregistrable plan; the recommended first run.
- 10.1 Ground truth in three tiers (GT-M, GT-C, GT-K) **Table 4**; fen-only edit consequence; noise-floor note for line edits.
- 10.2 Metrics (coverage never ranks methods; error/localisation only for non-pairwise; consistency; IIA/Isolate; stability; control gaps; fixed ablation type).
- 10.3 Hypotheses H1–H6, "none tested".
- 10.4 Staging **Table 5** (A0, A, B, C, cost).
- 10.5 What to try first: pairs, taps, quantities, controls, cost; decision rules (drop tap / promote M8 + M8∘M2 / promote M5 / run M12).
- 10.6 Worked example, short: base FEN after 3.Bb5 d6, edits Bc4 and …Bd7; **Table 6** bits changed by group (07 Table 7); output templates → Appendix D.
**Figure 4** (optional): decision-rule flowchart for 10.5.
**Transition**: what this does and does not answer → §11.

### 11. Discussion (~600)
**Purpose**: answer the user's three questions in the light of §§5–10; meet the strongest objection.
- 11.1 Front, back, hybrid: front is input saliency; choice of taps guided by ρ_L; per-family attribution at L1–L5 where stable.
- 11.2 Strongest objection (coverage not attribution; front trivial; ridge split) and the reply.
- 11.3 Open tensions: nonlinear vs linear probes (Pálsson vs Sutter; Nanda side-to-move); frozen attention vs head-ablation evidence.
- 11.4 Gaps addressed **Table 7** (07 Table 10).
**Transition**: limits.

### 12. Limitations (~350)
Nothing run; fen-only synthetic history; no pin bit; ln2 decoder not fitted; survey not systematic, preprint-heavy, Tier-2 dependence for cross-layer; NLP-only benchmarks; GT-C ≠ faithfulness; engine-derived families leak value (C7); annotations AI-written from abstracts.

### 13. Conclusion (~150)
Three-part answer in three sentences; first experiment in one.

### Statements (not counted)
Data availability (code paths, result JSONs); ethics (none: public games, no humans); CRediT; COI none; funding none; AI-use disclosure (Claude, Anthropic, via Claude Code; text from 07 disclosure).

### Appendices (~4,500, not counted in main)
- A. Method cards M0–M12 (+M8∘M2): definition with equations, question answered, failure modes, controls, cost, residual (from 07 §6, verbatim content, edited for length). ~3,000.
- B. Shared controls C1–C9 (07 Table 3). ~200.
- C. Nine-axis taxonomy (07 Table 4) + note on ONNX vs port. ~300.
- D. Worked example full procedure (steps 1–6), output templates (07 Tables 8, 9), policy variant Bxc6 / d2–d4. ~500.
- E. Survey method: themes, procedure, inclusion, verification, tiers, review rounds. ~400.

## Evidence map

Tags: S = surveyed evidence, M = own measurement or own check (fits, ONNX reads, extractor runs), H = hypothesis/estimate. Stance toward the paper's claims: + supports, − opposes/limits, 0 neutral/context.

| Section | Sources (author-year as in 07 reference list) | Type | Stance | Tables/figures |
|---|---|---|---|---|
| Abstract, §1 | project fits (results_square_decoder.json), extractor | M | + | — |
| §2 | pointers to §§6–10 | M/H | + | Fig 1, Table A |
| §3 | project: ONNX graph, `lc0_planes.py`, commit 473b3ce, pin probe, board-level decoder | M | 0 | Table 1 |
| §4.1 | Covert 2021; Sundararajan & Najmi 2020; Janzing 2020; Frye 2021; Chen 2020; I. E. Kumar 2020; Aas 2021; Lundstrom 2022; Bilodeau 2024 | S | + / − (Kumar, Bilodeau limit Shapley) | — |
| §4.2 | Ravichander 2021; Hewitt & Liang 2019; Belinkov 2022; Elazar 2021; Geiger 2020; Margeloiu 2021; Pálsson & Björnsson 2024; Jenner 2024; Abraham 2022; Huang 2024; Mueller 2025 | S | + | — |
| §4.3 | Zhou 2018; Yuksekgonul 2023; Havasi 2022; Marks 2025; Ameisen 2025; Engels 2025; Yeh 2020; Shang 2024; Leask 2025; Schut 2025 | S | + | — |
| §4.4 | Makelov 2025; Kantamneni 2025; Wu 2025 (AxBench, follow-up unverified); Mueller 2025; Karvonen 2024 | S | + | — |
| §4.5 | Dunefsky 2024; Paulo 2025; Lindsey 2024; Ameisen 2025; Lawson 2025; Farnik 2025; He 2025; Lin 2026 | S | 0 (gap) | — |
| §4.6 | Sutter 2025; Hewitt & Liang 2019; Beckers & Halpern 2019; Park 2024; Nicolson 2025; Mahinpei 2021; Espinosa Zarlenga 2023; Parisini 2026; Huang 2024; Ramaswamy 2023 | S | − (limits) | — |
| §4.7 | Ali 2022; Achtibat 2024, 2023; Kramár 2024; Ameisen 2025; Jenner 2024; Ferrando & Voita 2024; ONNX smolgen | S + M | − | — |
| §4.8 | McGrath 2022; Hammersborg & Strümke 2024; Lomasov 2025; Tomlin 2022; Sandmann 2025 | S | 0 | — |
| §5.1 | project fits | M | + | Table 2, Fig 2 |
| §5.2 | ONNX embedding; `lc0_planes.py`; Adebayo 2018; Sutter 2025; Hewitt & Liang 2019 | M + S | + | — |
| §5.4 | project fits; MLP decoder trend; Engels 2025 | M + H | 0 | — |
| §6 | Sundararajan 2017; Lundstrom 2022; Ali 2022; Adebayo 2018 | S | + / − | — |
| §7.1 | Yuksekgonul 2023; Aas 2021; Chen 2020; A. Kumar 2022; Parisini 2026; board-level decoder | S + M | + | Table 3 |
| §7.2 | Makelov 2025, 2024; Koh 2020; Shin 2023; Zhang & Nanda 2024; Rubenstein 2017; Heimersheim & Nanda 2024; Sundararajan & Najmi 2020; Marks 2025; Dhamdhere 2019; Yeh 2020; Ramaswamy 2023; Shrikumar 2017; Chen 2022; Achtibat 2024 | S | + | — |
| §7.3 | Belrose 2023; Elazar 2021; Ravfogel 2020, 2022; Pálsson & Björnsson 2024; Bai 2023; Kim 2018; Park 2024; Tucker 2021; Nicolson 2025; Geiger 2024, 2021, 2025; Wu 2023, 2024; Gat 2024; Huang 2024; Mueller 2025; Makelov 2024; Sutter 2025; Chan 2022 | S | + / − | — |
| §7.4 | ONNX policy head read | M | + | — |
| §8.1 | Lindsey 2024; Lawson 2025; Dhamdhere 2019 | S | 0 | Fig 3 |
| §8.2 | Elazar 2021; Belrose 2023; Engels 2025; Heimersheim & Nanda 2024; Farnik 2025 | S | + | Fig 3 |
| §8.3 | Goldowsky-Dill 2023 | S | + | Fig 3 |
| §8.4 | Ameisen 2025; Ali 2022; Dunefsky 2024; Paulo 2025; Marks 2025; Lin 2026; Mahinpei 2021; Jenner 2024; Kramár 2024 | S | + / − | Fig 3 |
| §8.5 | Engels 2025; Kulkarni 2026; Karvonen 2024; Poupart 2024; Leask 2025; Shang 2024; He 2025 | S | 0 | — |
| §9 | Table 7 bit counts; Abraham 2022; Aas 2021; Havasi 2022; Schut 2025; Sutter 2025; Beckers & Halpern 2019 | M + S | + | — |
| §10 | Chen 2020; Ramaswamy 2023; Zhang & Nanda 2024; Miller 2024; Heimersheim & Nanda 2024; Tomlin 2022; project MaskPinner/LineBlocker, obstruct noise-floor finding, pin probe 0.945 | S + M + H | + | Tables 4–6, Fig 4 |
| §11 | Pálsson & Björnsson 2024; Sutter 2025; Nanda 2023; Jenner 2024; Lin 2026; Sandmann 2025; Abraham 2022; Huang 2024; Mueller 2025; Frye 2021 | S | 0 | Table 7 |
| §12 | all; tier marks | — | − | — |
| App. A | all method sources above | S | + | — |
| App. B | Adebayo 2018; Sutter 2025; Hewitt & Liang 2019; Elazar 2021; A. Kumar 2022; Parisini 2026; Makelov 2024; Wu 2024; Heimersheim & Nanda 2024; Miller 2024 | S | + | App. Table B1 |

Coverage check: the 89 works cited in `07_report_final.md` all appear above or in Appendix A (which carries every method source). Sources in `02_biblio_T*.md` not cited in 07 are not used (config: cite only 07's list or the bibliographies; no new citations introduced).

## Figures and tables (main text)

| ID | Content | Origin | Status |
|---|---|---|---|
| Fig 1 | Decomposition at each tap + routes (a)/(b)/(c) | new schematic | to draw |
| Fig 2 | FVU (three rows) and partial R² by tap | `results_square_decoder.json` | to plot |
| Fig 3 | Four cross-layer compositions on tap axis | new schematic | to draw |
| Fig 4 | First-experiment decision rules | new flowchart | optional |
| Table A | Sub-question → method → output → identifiable → ONNX/port → cost | new, condensed from 07 Table 4 | to write |
| Table 1 | Existing results | 07 Table 1 | ready |
| Table 2 | Per-tap fits | 07 Table 2 | ready |
| Table 3 | Methods: question, residual, ONNX | condensed 07 Table 4 | to write |
| Table 4 | Ground-truth tiers | 07 Table 5 | ready |
| Table 5 | Staging and cost | 07 Table 6 | ready |
| Table 6 | Bits changed by two pin edits | 07 Table 7 | ready |
| Table 7 | Gaps addressed | 07 Table 10 | ready |

## Transitions (summary)

1→2 dilemma stated, answer given at once · 2→3 answers rest on a setting and constraints · 3→4 notation fixed, now what the literature allows · 4→5 readable≠used and identifiability bear on the fits; re-derive · 5→6 locality fixes what e can give · 6→7 attribution needs taps past locality · 7→8 single taps, then combinations · 8→9 every composite inherits collinearity and a residual · 9→10 grading rules set, protocol and first run · 10→11 what the protocol answers and what it does not · 11→12 limits · 12→13 close.

## Word count summary

| Section | Target |
|---|---|
| Abstract | 250 (not counted) |
| 1 Introduction | 650 |
| 2 Answer in brief | 750 |
| 3 System, data, notation | 750 |
| 4 Literature constraints | 1,000 |
| 5 Front/back re-derived | 850 |
| 6 (a) Embedding decoder | 550 |
| 7 (b) Any layer | 1,050 |
| 8 (c) Across layers | 900 |
| 9 Identifiability and coverage | 850 |
| 10 Protocol and first experiment | 950 |
| 11 Discussion | 600 |
| 12 Limitations | 350 |
| 13 Conclusion | 150 |
| **Main text** | **9,400** (config 8,000–10,000) |
| Appendices A–E | ~4,400 |

Largest section 1,050 (11%), none under 150 words per lowest heading.

## Notes for downstream phases
- IEEE numeric citations; author-year in this outline is for traceability only.
- Do not re-introduce issues fixed in `07_response.md` (oriented ρ_L; M7 linearity only between nodes; smolgen frozen with patterns; taps are ln1; castling on king square; user's note quoted, not corrected; Kulkarni only in M12; Jenner "use head ablation in their evidence"; AxBench follow-up unverified; Lin EV exception; 89 distinct works).
- Section 2 duplicates conclusions on purpose; keep it to pointers and one sentence per method.
