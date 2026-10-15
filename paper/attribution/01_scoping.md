# Phase 1 — Scoping

Mode: full. Paradigm: methods survey + taxonomy + proposed experimental protocol for one system
(lc0 `t1-256x10-distilled-swa-2432500`, 30,837-bit per-square feature vocabulary). No human subjects.
Decisions taken without asking the user are marked **[decision]**.

---

## 1. Research Question Brief

### Topic area
Attributing a chess transformer's outputs to a fixed, human-defined feature vocabulary, using
learned maps between that vocabulary and the residual stream (decoders: features -> token;
encoders/probes: token -> features), at one or several layers, while accounting for what the
vocabulary does not explain.

### Primary research question
Which families of methods can attribute lc0's value and policy outputs to a fixed per-square human
feature vocabulary through learned feature<->activation maps, and how do they trade off
faithfulness under intervention, identifiability, residual (error) coverage, and compute on this
system?

### FINER assessment
| Criterion | Score | Justification |
|---|---|---|
| Feasible | 4/5 | Literature-based; decoders/probes already fitted at every layer; ONNX taps exist; CPU-bound (~180 pos/s) limits gradient-heavy methods and crosscoder training, hence 4. |
| Interesting | 5/5 | Real tension in the user's data: decoder coverage peaks at L1-L2 and falls to 0.32 at L9, so "where to attribute" has no obvious answer. |
| Novel | 3/5 | Supervised dictionaries (Makelov et al.), error nodes (Marks et al.), concept gradients (Bai et al.) exist separately. Novel part: combining them across layers for a *fixed external* vocabulary on a game model, with the residual reported as a first-class result. |
| Ethical | 5/5 | No human data; public model and puzzles. |
| Relevant | 4/5 | Directly decides the project's next experiment; transfers to any domain with a symbolic feature extractor (code, math, molecules). |
| **Average** | **4.2/5** | Passes (>=3.0, none <2). |

### Scope
**In scope**
- Post-hoc attribution to a *given* vocabulary: decoder-based (features -> activation), encoder/probe-based (activation -> features), and multi-layer combinations (several decoders/encoders, crosscoders, transcoders/CLTs, attribution graphs with error nodes, path patching, causal scrubbing, DAS/interchange interventions, concept bottlenecks, completeness/leakage measures).
- Targets: value head (WDL/scalar), policy logits for a chosen move; per-square and board-level.
- Gradient, path-integral, Shapley/occlusion, and interventional attribution through those maps.
- Faithfulness and completeness evaluation methods, incl. measuring the residual the vocabulary misses.
- Chess/game-model interpretability as application precedent.

**Out of scope**
- Unsupervised concept *discovery* as an end in itself (SAE features without a mapping to the vocabulary). Included only as a tool for naming the residual. [decision]
- Pure input-saliency on board planes without a concept layer (cited as baseline only).
- LLM-generated natural-language explanations, commentary, CoT faithfulness (covered elsewhere in the project).
- Retraining lc0 (e.g. true concept-bottleneck training). Discussed, not proposed as the main protocol. [decision]
- Search (MCTS) attribution; single network forward pass only.

**Key assumptions**
- The feature extractor is correct (it is tested in `features/`); attribution error is about the model, not the labels.
- Activation-level interventions on lc0 are meaningful even if off-distribution; we must measure how off-distribution.
- "Faithful" = predicts the effect of interventions (project value: interventional > correlational).

### Sub-questions (map to report sections)
1. **Decoder route.** Given D: (f, square) -> h_L, what attribution operators exist (grad·Jacobian, IG over bits, occlusion/Shapley over bits, decoded-substitution patching), and when are they identifiable given 30k redundant, logically dependent binary features against 256 dims?
2. **Encoder route.** Given a probe P: h_L -> f, how does one turn readability into *use* (gradient along probe directions/TCAV, concept erasure via LEACE/INLP/amnesic probing, interchange along probe subspaces/DAS), and how does one avoid probe-illusion?
3. **Multi-layer combinations.** Which methods chain or stack maps across layers (per-layer decoders with error nodes, crosscoders, transcoders/CLTs, attribution graphs, path patching, causal scrubbing) and how do they resolve the user's coverage-vs-causal-distance trade-off?
4. **Residual.** How do these methods quantify and attribute the part of the output the vocabulary does not explain (error nodes, PCBM-h residual, completeness scores, SAE dark matter), and can the residual itself be named?
5. **Evaluation.** Which protocol, using this project's interventional operators (MaskPinner, LineBlocker) as ground truth, can rank the candidate methods on lc0 within CPU budget?

### Candidate questions considered
| # | Candidate | FINER avg | Why not selected |
|---|---|---|---|
| 1 | Primary RQ above | 4.2 | Selected |
| 2 | "What is the single best way to get feature attribution from the decoder?" | 3.4 | User explicitly wants several approaches; presupposes decoder is the right object. |
| 3 | "At which layer is attribution to human features most faithful?" | 3.8 | Becomes sub-question 3; too narrow alone and assumes single-layer answer. |
| 4 | "How much of lc0's value computation is not expressible in the vocabulary?" | 4.0 | Central to the user but needs a method first; becomes sub-question 4. |

---

## 2. Methodology Blueprint

### Paradigm
Pragmatist. The question is applied (which method to run next) and spans formal properties
(identifiability, completeness axioms) and empirical ones (reported faithfulness).

### Method
Structured methods survey (not a PRISMA systematic review — the literature is mostly
arXiv/blog, terminology unstable, so exhaustive counting would be false precision) [decision], ending in:
(a) a taxonomy with a fixed axis set, (b) a per-method fit assessment for lc0, (c) a proposed,
pre-specified experimental protocol (not executed in this paper).

### Search strategy
- **Sources**: arXiv (cs.LG, cs.AI, cs.CL), OpenReview (ICLR, NeurIPS, TMLR), proceedings of NeurIPS/ICML/ICLR/ACL/EMNLP/AAAI/FAccT, transformer-circuits.pub, Distill. LessWrong/Alignment Forum as gray literature only. Google Scholar for forward-citation chasing. Project files `PAPERS.md`, `paper/references.bib`, `RESEARCH_SUMMARIES.md` for already-verified entries.
- **Procedure**: per theme (Section 3) keyword search -> seed papers -> one round backward + forward citation chase on each seed -> stop when two consecutive chases add nothing in-scope.
- **Window**: 2016-2026, except foundational axiomatic work (Shapley, IG, probing) of any date.

### Inclusion / exclusion
Include if the work (i) defines or evaluates a method mapping activations to/from a concept
vocabulary or sparse dictionary and attributing outputs through it, or (ii) defines a faithfulness/
completeness/identifiability criterion for such attributions, or (iii) applies such methods to a
game-playing network. Exclude: pure input-saliency without concepts (except baselines), text-only
rationale methods, works without a method description (position pieces, news), duplicates
(keep the archival version).

### Source tiers
| Tier | Type | Use |
|---|---|---|
| 1 | Peer-reviewed venue (NeurIPS/ICML/ICLR/ACL/EMNLP/TMLR/JMLR, CVPR for CBMs) | Claims of fact |
| 2 | transformer-circuits.pub, Distill, DeepMind/Anthropic/OpenAI technical reports, arXiv with >=1 independent replication | Method definitions; claims flagged "not peer-reviewed" |
| 3 | Plain arXiv preprints | Method existence; empirical claims hedged |
| 4 | LessWrong/AF posts, blogs (e.g. logit lens, causal scrubbing writeup) | Gray lit: only for methods with no Tier 1-3 source; labelled |

### Analytic framework — taxonomy axes
Every method gets one row with these columns:
1. **Vocabulary entry point**: where human features enter — input side (decoder target = features), readout side (probe), bottleneck (replaces activations), dictionary-matched (post-hoc naming of SAE latents), or none.
2. **Direction**: decode (f -> h), encode (h -> f), both (autoencoder/bottleneck), or neither (pure intervention).
3. **Layer span**: single layer / per-layer independent / cross-layer (crosscoder, CLT) / path-level (path patching, graphs).
4. **Map class**: linear, sparse-linear, nonlinear (MLP), or identity (intervention on activations directly).
5. **Attribution operator**: gradient/Jacobian, path integral (IG), Shapley/occlusion over features, activation patching/interchange, ablation/erasure.
6. **Residual handling**: ignored, error node kept in graph, residual fit (PCBM-h), completeness score reported, residual decomposed further.
7. **Identifiability**: is the attribution unique given redundant/correlated features (unique / unique up to regularizer / non-unique)?
8. **Faithfulness test available**: none / deletion-insertion / interventional against ground truth / causal-abstraction accuracy (IIA).
9. **Cost on this system**: forward passes and backward passes per position, training cost, and whether ONNX-on-CPU suffices or a PyTorch port of lc0 is needed (gradients need one).

### Analysis steps
1. Build the per-method table (axes 1-9) from the bibliography.
2. Group into families; per family write: definition, what it assumes, failure modes, lc0 fit.
3. Map families onto the user's front/back trade-off (coverage vs causal distance) and show which combinations cover the gap.
4. Derive the protocol: methods x layers x targets, ground-truth interventions, metrics, budget.

### Validity criteria
| Criterion | Strategy |
|---|---|
| Coverage | Theme-wise citation chasing with explicit stopping rule; DA checkpoint 2 asks what is missing. |
| Citation accuracy | Every arXiv ID / venue verified against the source before Phase 4; unverified IDs carried as `[verify]`. |
| Representation accuracy | Method definitions quoted or paraphrased from the paper's own equations, not from secondary summaries. |
| Applicability claims | Every "works on lc0" claim tied to a stated cost estimate or to an existing project result. |
| No over-claiming | Survey reports what papers show; protocol predictions marked as hypotheses. |

### Limitations by design
- Not a systematic review; coverage judged, not counted.
- Field is fast and preprint-heavy; Tier 3/4 share will be high in themes 4-5.
- Protocol is proposed, not run; its cost estimates are approximations.
- Gradient methods need a differentiable lc0 (PyTorch port or ONNX->torch); feasibility noted per method.

### Ethics / IRB / reporting / preregistration
No human subjects; no IRB. Reporting: no EQUATOR guideline fits; follow a PRISMA-style flow note
for transparency only. Preregistration: not for the survey; recommended (OSF) for the proposed protocol.

---

## 3. Bibliography themes for Phase 2 (parallel search)

Seed IDs from memory are marked `[verify]` where not already in `PAPERS.md`.

**T1. Attribution operators and their axioms (gradient, path, Shapley), incl. on discrete/dependent features**
Keywords: integrated gradients, completeness axiom, DeepLIFT, Shapley value attribution, on-manifold Shapley, baseline choice, binary feature occlusion, LRP transformers, AttnLRP, sanity checks saliency.
Seeds: Sundararajan IG 1703.01365; Shrikumar DeepLIFT 1704.02685; Lundberg SHAP 1705.07874; Frye on-manifold Shapley 2006.01272; Janzing feature relevance 1910.13413; Adebayo sanity checks 1810.03292; Ali conservative LRP 2202.07304; Achtibat AttnLRP 2402.05602; Chefer 2012.09838 [all verify].

**T2. Concept attribution through learned concept maps (the decoder/encoder bridge)**
Keywords: concept activation vectors, concept gradients, nonlinear concept attribution, causal concept effect, concept-to-representation map, TCAV statistical testing, concept sensitivity.
Seeds: Kim TCAV 1711.11279 (PAPERS [31]); Bai Concept Gradient 2208.14966; Goyal CaCE 1907.07165; Feder CausaLM 2005.13407; Abraham CEBaB 2205.14140; Ghorbani ACE 1902.03129 [verify].

**T3. Probing validity, layerwise readout, and concept erasure (encoder route)**
Keywords: probing classifier, control task, selectivity, amnesic probing, INLP, LEACE, tuned lens, logit lens, probe vs use, representation reading.
Seeds: Alain & Bengio 1610.01644; Hewitt & Liang 1909.03368; Belinkov survey 2102.12452; Ravichander 2005.00719; Elazar amnesic 2006.00995; Ravfogel INLP 2004.07667; Belrose LEACE 2306.03819; Belrose tuned lens 2303.08112; Park linear representation hypothesis 2311.03658 [verify].

**T4. Concept bottlenecks, supervised dictionaries, completeness and leakage**
Keywords: concept bottleneck model, post-hoc CBM residual, leakage, concept embedding model, completeness score, ConceptSHAP, supervised dictionary, hybrid bottleneck, side channel.
Seeds: Koh CBM 2007.04612; Yuksekgonul PCBM(-h) 2205.15480; Mahinpei leakage 2106.13314; Margeloiu 2105.04289; Espinosa Zarlenga CEM 2209.09056; Yeh completeness 1910.07969; Makelov et al. principled SAE evals w/ supervised dictionaries 2405.08366 [verify].

**T5. Sparse dictionaries across layers: SAEs, transcoders, crosscoders, CLTs, and the residual**
Keywords: sparse autoencoder, transcoder, cross-layer transcoder, sparse crosscoder, Jacobian SAE, SAE dark matter, reconstruction error, SAEBench, feature splitting.
Seeds: Cunningham 2309.08600; Bricken "Towards Monosemanticity" (transformer-circuits 2023); Dunefsky transcoders 2406.11944; Lindsey "Sparse Crosscoders" (transformer-circuits 2024); Engels dark matter 2410.14670; Farnik Jacobian SAEs 2502.18147; Karvonen SAEBench 2503.09532; Poupart contrastive SAE chess 2406.04028 (PAPERS [39]) [verify].

**T6. Circuit discovery and attribution graphs with error nodes**
Keywords: path patching, activation patching best practice, attribution patching, AtP*, ACDC, sparse feature circuits, error nodes, attribution graph, replacement model, circuit tracing.
Seeds: Wang IOI 2211.00593; Goldowsky-Dill path patching 2304.05969; Conmy ACDC 2304.14997; Syed attribution patching 2310.10348; Kramár AtP* 2403.00745; Zhang & Nanda 2309.16042; Heimersheim & Nanda 2404.15255; Marks sparse feature circuits 2403.19647; Ameisen et al. "Circuit Tracing" (transformer-circuits 2025) [verify].

**T7. Causal abstraction and interchange interventions**
Keywords: causal abstraction, interchange intervention accuracy, distributed alignment search, boundless DAS, causal scrubbing, interchange intervention training, interpretability illusion subspace, RAVEL, AxBench.
Seeds: Geiger causal abstraction 2301.04709; Geiger DAS 2303.02536; Geiger IIT 2112.00826; Chan et al. causal scrubbing (AF 2022, Tier 4); Makelov interpretability illusion 2311.17030; Huang RAVEL 2402.17700; Wu AxBench 2501.17148 [verify].

**T8. Interpretability of game-playing networks (application precedent)**
Keywords: AlphaZero concepts, Leela Chess Zero interpretability, chess transformer probing, look-ahead, concept discovery chess, Go network interpretability, SAE chess.
Seeds (all in PAPERS.md): McGrath 2111.09259; Schut 2310.16410; Karvonen 2403.15498; Jenner 2406.00877; Sandmann 2508.21380; Lin 2604.10158; Poupart 2406.04028; Jiao SAE skill (OpenReview Wxl0JMgDoU); Hammersborg 2309.09702; Mészáros 2510.20783.

---

## 4. Devil's Advocate Report — Checkpoint 1

### Verdict: PASS (conditional) — no critical issues; Major 1-4 must be built into Phase 2-3, not left to the discussion section.

### Critical issues
No critical issues identified.

### Major issues
1. **Decoder attribution is not identifiable.**
   - Type: Method. Location: SQ1, the context's `grad_E · ∂D/∂f`.
   - Problem: D maps ~thousands of kept bits to 256 dims; its nullspace is huge. Ridge picks the min-norm solution, so per-feature attribution `W_iᵀ ∇_E y` depends on the regularizer and on which correlated bits are present (e.g. "occupant=knight" vs the 40 knight-derived route bits). Two decoders with equal FVU can give opposite rankings. Also D is non-invertible: there is no f for a given E, so "attribute E to f" means "attribute D(f) to f", and E − D(f) (FVU 0.002 at the embedding, 0.68 at L9) gets nothing.
   - Recommendation: make identifiability a taxonomy axis (done, axis 7); in Phase 2 search for group-level attribution (attribute to feature *families*, not bits), regularization-path stability, and Shapley over families. Protocol must report attribution stability across ridge alphas and across decoder seeds.

2. **Gradients w.r.t. binary, logically dependent features are off-manifold.**
   - Type: Method. Location: SQ1, SQ2.
   - Problem: ∂/∂f at f∈{0,1} measures response to "0.3 of a pin". IG from a zero baseline walks through feature vectors no board produces. Flipping a single bit (occlusion) yields inconsistent feature sets ("pinned" with no pinner), then D(f') is an embedding no position produces. These are the same objections raised against off-manifold Shapley (Frye; Janzing).
   - Recommendation: Phase 2 must cover on-manifold/conditional Shapley and board-level counterfactuals. Protocol: the gold standard is the project's real board interventions (MaskPinner, LineBlocker, piece removal) giving a consistent f'; feature-space interventions are compared against it, not trusted alone.

3. **The front/back framing may be inverted.**
   - Type: Framing. Location: user's journey hypothesis, SQ3.
   - Problem: an embedding-level attribution must still pass the gradient through all 10 blocks — that is the "deep jungle of attention". An L9 attribution passes only through the value/policy head. So front = high coverage, long causal path; back = low coverage, short path. The trade-off is coverage vs path length, not reconstruction vs clarity. Also the embedding is (in lc0) an affine function of the input planes; FVU 0.002 there mostly says the vocabulary contains occupancy, so embedding-level attribution reduces to input saliency re-expressed in feature terms.
   - Recommendation: state the trade-off in these terms in the report; add "path length from map to output" to axis 3 discussion; include an occupancy-only vocabulary as a control to show what the 30k other bits add.

4. **Readable ≠ used (encoder route).**
   - Type: Method/Evidence. Location: SQ2; context result 1 ("near-perfectly readable at the embedding").
   - Problem: probes at the embedding read features computable from occupancy; high probe accuracy says nothing about use downstream (Ravichander; Elazar). Probe-direction interventions can also produce interpretability illusions (Makelov 2311.17030).
   - Recommendation: encoder-route methods are graded only by erasure/interchange effect on the output, with control probes and random-subspace baselines.

### Minor issues
- **Label leakage from engine-derived features.** `mate`, `material_win` and threat-line bits come from an engine; they partially encode the value target. Attribution to them will be high for trivial reasons. Report results with and without engine-derived families.
- **Square identity confound.** "Features add" is the right quantity; attributions must be taken net of the square-only decoder, or square identity absorbs credit at deep layers.
- **Per-square vs board-level mismatch.** Vocabulary is per-square, value is global; board-level decoders overfit (test FVU 0.52-0.62). Protocol should prefer per-square decoders at intermediate layers and gradient/patching to the global output over direct global decoders.
- **Breadth risk.** Nine method families in one paper invites shallow treatment. Cap: each family gets definition, lc0 fit, one-paragraph failure modes; depth goes to the 3-4 the protocol selects.
- **Gradient access.** ONNX on CPU gives no backward pass; any gradient method needs a torch port. Cost axis must say so.
- **Tier-2 dependence.** Crosscoders, CLTs and circuit tracing are transformer-circuits.pub only; flag as not peer-reviewed.

### Observations
- Error nodes (Marks; Ameisen) are the closest existing answer to the user's "measure what the vocabulary misses": a per-layer decoder residual becomes a node whose attribution is reported alongside the named features.
- A per-layer supervised decoder with error node at each of L0-L9 is essentially a supervised CLT/replacement model; this is the natural "several decoders on several layers" design and should be the anchor of SQ3.
- DAS with the vocabulary as the high-level causal model gives an interventional faithfulness number (IIA) per feature family per layer — a clean fit for the project's preference for interventions.

### Strongest counter-argument
"Attribution through a ridge decoder fitted on 30k correlated bits is an artifact of the regularizer; it explains the decoder, not lc0. The only defensible numbers are interventional: real board edits or interchange interventions, with the residual reported. Everything gradient-based over binary features is decoration."

### What's missing
- Methods that attribute to feature *groups* with dependence structure (grouped Shapley, hierarchical attribution).
- Evidence on whether supervised (fixed-vocabulary) dictionaries match unsupervised SAEs in faithfulness — Makelov et al. is the main seed; need more.
- Any prior work attributing *through a fitted decoder* rather than a probe — may not exist under that name; search "concept gradient", "decoder-based concept attribution", "generative concept attribution".

### Stress test
| Test | Result |
|---|---|
| Remove strongest source — does argument hold? | Yes: the taxonomy does not hinge on one paper. |
| Flip the RQ — is "no decoder/encoder method is faithful here" credible? | Yes, plausibly at deep layers; the survey must allow that conclusion. |
| Different context — does it generalize? | Yes, to any model with an external symbolic feature extractor. |
| "So what?" | Yes: it chooses the project's next experiment and quantifies the vocabulary gap. |

`[DA-DECISION: no rebuttals this checkpoint | ACTION: n/a]`
