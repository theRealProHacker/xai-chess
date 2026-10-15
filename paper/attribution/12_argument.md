# Argument Blueprint (academic-paper Phase 3, argument_builder_agent, full mode)

Built on `11_outline.md`; content from `07_report_final.md` and `07_response.md`. Section numbers refer to the outline.

**Evidence tags.** [S] surveyed evidence (cited source; annotations from abstracts and landing pages). [M] own measurement or own check: fits, extractor runs, ONNX graph reads, `lc0_planes.py`. [M-a] own analytic result: follows by algebra from the definitions, no data. [H] hypothesis or estimate: nothing run.

## Central thesis

Attribution of lc0's value and policy outputs to a fixed per-square feature vocabulary is best organised as removal operators on one decomposition of each token, $h_L=q_L+\sum_G n^G_L+\varepsilon_L$, read at chosen taps: at the embedding it reduces to input saliency in feature names; at middle taps decoded substitution on real board edits gives the per-edit attribution, with probe-side erasure and interchange testing whether the named content is used; across layers four compositions trace where named content is built and where it leaves the vocabulary. Only the joint effect of an edit is identifiable, and the residual is reported as coverage, not as error.

Arguable because: the user's working hypothesis (front attribution clearer; a hybrid) and the common practice of ranking attributions by faithfulness error both point elsewhere.

## Sub-arguments

### SA1 (a). At the embedding, decoder attribution is input saliency with feature names. → §5.2, §6

| Claim | Evidence | Tag | Reasoning |
|---|---|---|---|
| C1.1 The embedding is per-square local and elementwise nonlinear | $h_{e,s}=g_s\odot\mathrm{mish}(W[\text{planes}_s;\text{posenc}_s]+b)+a_s$ from the ONNX graph | M | Nothing from other squares enters $h_{e,s}$ |
| C1.2 The 0.002 FVU says the vocabulary contains occupancy | Table 2; fen-only fill copies the board into all history frames | M | A local map of occupancy is fit by an occupancy-containing vocabulary |
| C1.3 Non-local families get spurious credit at $e$, cancelled by $\varepsilon_e$ | Follows from C1.1 and the decoder definition | M-a | Edit changes non-local bits on squares whose planes did not change; $W^G\Delta\tilde f\neq0$ there, $\varepsilon_e$ absorbs it |
| C1.4 Hence use local vocabulary C2, or zero decoded deltas on unchanged-plane squares | C1.3; control tasks [Hewitt & Liang 2019] | M-a, S | Removes the spurious term |
| C1.5 Under C2, $\rho_e\approx1$ by construction | C1.1, C2 | M-a | $\varepsilon'_e-\varepsilon_e\approx0$ for edits leaving broadcast planes alone |
| C1.6 A random-initialised embedding fits near 0.002 | Locality is architectural; sanity checks [Adebayo 2018]; random-net IIA [Sutter 2025] | H | Prediction, tested by C1 (H1) |
| C1.7 Gradients through ten post-LN blocks violate conservation | [Ali 2022] | S | Point-form M1 is a sensitivity map; board-path IG is complete but path-long |

**Conclusion**: the front answers "which of this square's inputs, named"; it does not answer what lc0 computed. Attribution worth doing starts where locality no longer holds.

### SA2 (b). At any layer, decoder substitution attributes; encoder-side erasure and interchange test use; readability alone does neither. → §4.2, §4.4, §7

| Claim | Evidence | Tag | Reasoning |
|---|---|---|---|
| C2.1 Removal rule fixes meaning; three rules exist here (bit flip, decoded substitution, board edit) | [Covert 2021; Sundararajan & Najmi 2020; Chen 2020; Janzing 2020; Frye 2021] | S | Choose the rule before the method |
| C2.2 Real board edits tie attribution to an end-task | [Bilodeau 2024] proves near-random for complete linear methods on rich model classes and argues for end-task tying | S | The edit is the end-task |
| C2.3 Readable ≠ used | [Ravichander 2021; Elazar 2021; Geiger 2020; Margeloiu 2021; Pálsson & Björnsson 2024]; benchmarks CEBaB, RAVEL, MIB | S | Probe accuracy at a tap is not attribution |
| C2.4 Families are readable at $e$ and stay readable with depth; some gain 0.04–0.14 by L5 | Table 1 | M | Readability does not discriminate taps; use must be measured |
| C2.5 With a fixed vocabulary, supervised maps are the default | [Makelov 2025; Kantamneni 2025; Wu 2025; Mueller 2025] | S | $D_L$ has the supervised-dictionary form; SAEs go on $\varepsilon$ |
| C2.6 M2's joint patch and $\gamma_L$ measure whether the decoded edit commutes with the real one | [Rubenstein 2017; Makelov 2025; Koh 2020] | S | Causal consistency as a check |
| C2.7 M6 gives an exact two-player split; on oriented pairs $\rho_L$ is well defined | Completeness since $g_L(q+n'+\varepsilon')=y(x')$; antisymmetry | M-a | Coverage per tap, not per family |
| C2.8 LEACE erasure (M3) measures use of $G$ with its linear correlates | [Belrose 2023; Elazar 2021] | S | Use-vs-readability curve from M3 + probes |
| C2.9 DAS (M5) tests mediation; random networks give near-perfect IIA | [Geiger 2024; Wu 2023; Sutter 2025; Huang 2024] | S | C1 is decisive for M5 |
| C2.10 M0 surrogate is the correlational floor | Board-level decoder FVU 0.52–0.62, still falling [M]; PCBM-h [Yuksekgonul 2023] | M, S | An interventional method that localises no better adds nothing |
| C2.11 Policy logit for a→b is bilinear in the from- and to-square tokens at `/encoder9/ln2`; exact four-player split | ONNX graph | M, M-a | Natural two-square localisation; needs an ln2 decoder (not fitted) |
| C2.12 Per-family credit is reportable at L1–L5 where $\rho_L$ is high, above C1 and C3-stable | Table 2 partial $R^2$ 0.76 at L1–L2, falling after | M + H | Prediction (H2) |

**Conclusion**: decoder direction answers "how much and through which families"; encoder direction answers "is it used". Report both at the same tap.

### SA3 (c). Combining layers has four distinct forms with different costs; none needs a supervised cross-layer dictionary that does not exist. → §4.5, §8

| Claim | Evidence | Tag | Reasoning |
|---|---|---|---|
| C3.1 No source fits a supervised cross-layer map; existing ones assume an additive stream | [Dunefsky 2024; Paulo 2025; Lindsey 2024; Ameisen 2025; Lawson 2025; Lin 2026]; lc0 taps are post-LN [M] | S, M | Gap; designs must tolerate post-LN |
| C3.2 M6 sweep compares independent decoders; $\kappa_L$ is descriptive, not a flow | Attention mixes squares, LN renormalises, decoders fitted separately | M-a | Do not read $\kappa_L$ as block contribution |
| C3.3 M8 measures recomputation and transfer into the residual | [Elazar 2021; Belrose 2023; Engels 2025; Heimersheim & Nanda 2024] | S | Intervene at $L$, read at $L'>L$ |
| C3.4 M8∘M2 gives feature paths with attention live | Path patching [Goldowsky-Dill 2023] in decoder coordinates | S | Cheap under ONNX; non-additive, reported against $\Delta^{(L)}_G$ |
| C3.5 M7 gives a graph with linear edges only between nodes at frozen attention, LN and activations | [Ameisen 2025; Ali 2022]; smolgen biases logits before softmax [M] | S, M | Freezing hides QK and smolgen routing |
| C3.6 Freezing attention may misplace credit on lc0 | Head ablation in look-ahead evidence [Jenner 2024]; AtP* recomputation [Kramár 2024] | S | M8∘M2 is the live-attention alternative |
| C3.7 M12 splits the residual by cause before anything in it is called unnamed | [Engels 2025] error decomposition; seven causes (§5.4) | S, M-a | Missing vocabulary vs weak map [Havasi 2022] |
| C3.8 A learned ensemble across methods is unidentified | With $\varepsilon$ included every complete method has zero loss | M-a | Hence M11 compares, does not fuse |

**Conclusion**: the user's "hybrid" is four concrete objects; the cheapest two (M8, M8∘M2) run under ONNX.

### SA4. Only the joint effect of an edit is identifiable. → §9.1

| Claim | Evidence | Tag | Reasoning |
|---|---|---|---|
| C4.1 A pin edit changes many collinear families | Interposition …Bd7: 331 bits on 21 squares, 180 route bits; Bc4: 356 bits (Table 6) | M | Pin appears only via x-ray and mobility bits |
| C4.2 Ridge splits $W_L$ among collinear families by its penalty | Linear algebra of ridge | M-a | Single-family credit partly an artefact of $\alpha$ |
| C4.3 Identifiable: $\Delta^{\rm true}$; given $D_L$: joint named patch, $\gamma_L$, $\phi_n$, $\phi_\varepsilon$, $\rho_L$; LEACE effect of $G$ with correlates | Definitions; prediction $W_L\Delta\tilde f$ far less α-sensitive than coefficients | M-a | Still moves with fit; C3, C4 report how much |
| C4.4 Unrestricted maps make any network abstract any algorithm; constructive abstraction conditions unmet | [Sutter 2025; Beckers & Halpern 2019; Park 2024; Nicolson 2025] | S | Supports linear-primary and per-edit reporting |
| C4.5 Remedy: attribute to the edit, describe it by changed families, merge unstable families until C3 ranks stabilise | [Aas 2021] grouping | S | Stated merge |

### SA5. The residual is coverage, not error. → §9.2–9.4

| Claim | Evidence | Tag | Reasoning |
|---|---|---|---|
| C5.1 For complete pairwise methods, ICaCE-style error equals residual share | CEBaB error [Abraham 2022]; dummy axiom / zero delta | M-a | Ranking by error rewards pushing credit to named families |
| C5.2 One definition Cov over oriented pairs instantiates for M6, M2 joint, pairwise M9, M0 | Definitions | M-a | Comparable on one axis; distributional M9 excluded |
| C5.3 The user's priority is what the vocabulary misses | Project values (`00_context.md`) | — | Coverage is a finding about lc0 + vocabulary |
| C5.4 Residuals are structured and reportable | [Zhou 2018; Yuksekgonul 2023; Marks 2025; Ameisen 2025; Engels 2025] | S | Precedent for error nodes |
| C5.5 Naming residual content depends on dictionary and candidate bank | [Leask 2025; Shang 2024; Schut 2025] | S | "Nothing nameable" must be an admissible outcome |

### SA6. First experiment: M2 + M6 on pin edits. → §10.5

| Claim | Evidence | Tag | Reasoning |
|---|---|---|---|
| C6.1 Pins give clean edits and chess-side ground truth | MaskPinner (GT-C); pin probe AUROC 0.945 at block 8 | M | Connects to existing project assets |
| C6.2 Cost ≈ one CPU hour under ONNX, no port | $2k+2\approx26$ suffix passes/tap/setting, $k\approx12$; 4 taps, 1,000 pairs, 5 α + C1; ~180 pos/s | H (estimate) | Cheapest run that decides the next |
| C6.3 Decision rules fixed in advance: drop taps inside C1 range; high $\rho_L$ + small $\gamma_L$ → M8 + M8∘M2; large $\gamma_L$ where probes read the pin → M5; sharp $\rho$ drop L5→L9 → M12 first | Follows from SA2–SA5 | M-a / H | Each outcome names the next method |

## Synthesis

SA1 removes the front as an attribution site; SA2 places per-edit attribution at middle taps and pairs it with use tests; SA3 answers the hybrid question with four compositions; SA4 limits what any of them can report per family; SA5 says how to report what is left; SA6 turns SA1–SA5 into one run whose outcome selects the next method.

## Counter-arguments and rebuttals

| # | Counter-argument | Target | Strategy | Rebuttal |
|---|---|---|---|---|
| CA1 | **Coverage ≠ attribution.** $\rho_L$ has two players; it never says which feature moved the value. | SA2, SA5 | Concede and limit | $\rho_L$ is a tap-selection guide and a residual measure. Per-family attribution comes from M2 (and M9) at taps where $\rho_L$ is high, above C1, C3-stable. Stated in Abstract, §2, §7.2, §11. |
| CA2 | **Identifiability.** "Pin got 0.12" is a ridge artefact; collinear families cannot be separated. | SA2, SA4 | Concede and limit | Accepted as a limit of every decoder-split method. Report joint effect always; per-family only where C3 ranks are stable; otherwise merge and say so. M3 reports $G$ with its correlates by design. Table 6 shows the scale (331 bits). |
| CA3 | **Embedding locality makes the front trivial.** FVU 0.002 and $\rho_e\approx1$ say nothing about lc0. | SA1 | Reframe | That is the finding: the front is input saliency in feature names, admissible only under C2. It moves attribution to L1–L5. C1 measures whatever learning the 0.002 does reflect (broadcast-plane weighting). |
| CA4 | **Readable ≠ used.** Probes read every family at $e$; encoder attribution is just accuracy. | SA2 | Concede and redirect | Encoder methods here are erasure (M3), concept gradient (M4) and interchange (M5), not accuracy. M3 plots use against readability; H5 predicts use lags readability. |
| CA5 | **Non-linear maps can fit anything.** MLP decoders, probes or DAS rotations find structure in random networks. | SA2, SA3 | Concede and control | Linear maps primary; MLP only beside C1 (C4); DAS judged against C1 [Sutter 2025]; M7 reads features through $P_L$, not extractor bits, to avoid leakage [Mahinpei 2021]. Opposite view noted: Pálsson & Björnsson recommend nonlinear probes; left open (§11.3). |
| CA6 | Decoded patches are off-distribution and can open dormant pathways. | SA2 | Acknowledge + diagnose | Joint patch endpoints are real; $\gamma_L$ measures the gap; C8 row/nullspace split as diagnostic, with Wu 2024's caution that it flags valid solutions too. |
| CA7 | Frozen attention in M7 contradicts lc0 head-ablation evidence. | SA3 | Acknowledge, offer alternative | M8∘M2 keeps attention live; AtP*-style recomputation bounds what freezing hides. Tension stated as open. |
| CA8 | Nothing has been run; the paper is speculation. | all | Acknowledge as scope | Position paper with preregistered protocol; every outcome tagged hypothesis; measured inputs limited to Tables 1, 2, 6; first run costs one CPU hour and its decision rules are fixed before running. |
| CA9 | Engine-derived families (mate, material lines) encode value; attribution to them is circular. | SA2, SA5 | Control | C7 with and without them is mandatory; H6 predicts most credit vanishes under C7. |
| CA10 | FEN inputs have synthetic history; results may not hold on games. | all | Limit scope | Edits re-encoded with the same fill, so no artefact; claims scoped to synthetic history; replication on real-history positions required. |
| CA11 | Lin et al. (2026) already built a replacement model of lc0. | SA3 | Distinguish | Unsupervised, hand-labelled, reconstruction per module; no fixed human vocabulary, no output-level residual share. |
| CA12 | Complete Shapley-type attributions are no better than random [Bilodeau 2024; I. E. Kumar 2020]. | SA2 | Reframe | Bilodeau's remedy is end-task tying; the end-task here is a real board edit with GT-M. |

## Logical flow

```
§1  Dilemma (front fits, back short path) + measured FVUs → RQ (a)/(b)/(c) + first run
     ↓
§2  Answer stated: (a) saliency, (b) substitution + use tests, (c) four compositions; identifiable = joint; first run
     ↓
§3  Setting, taps, targets, decomposition h = q + Σn + ε
     ↓
§4  Constraints: removal rule; readable≠used; residual reportable; supervised default;
    cross-layer only unsupervised; identifiability; attention breaks gradients; depth profiles
     ↓
§5  Fits re-derived → embedding local (SA1 premise); partial R² falls after L2; seven causes
     ↓
§6  (a) SA1 → front = input saliency; C2 required
     ↓
§7  (b) SA2 → M0 floor; M2/M6/M9 decoder; M3/M4/M5 encoder; policy at ln2
     ↓
§8  (c) SA3 → M6 sweep, M8, M8∘M2, M7; M12, M11
     ↓
§9  SA4 identifiable = joint; SA5 residual = coverage
     ↓
§10 SA6 protocol, hypotheses, first run, decision rules, worked pin example
     ↓
§11 CA1–CA3 met in one paragraph; open tensions (CA5, CA7); gaps table
     ↓
§12 CA8, CA10, tiers, GT-C ≠ faithfulness
     ↓
§13 Three-part answer + first run
```

## Argument strength assessment

| Sub-argument | Evidence strength | Logic validity | Counter-arg risk | Score |
|---|---|---|---|---|
| SA1 embedding = input saliency | Strong (ONNX read + algebra; C1.6 is a prediction) | Valid | Low | Strong (85) |
| SA2 substitution attributes, encoder tests use | Strong for literature claims; outcomes on lc0 hypothetical | Qualified (C2.12) | Medium (CA1, CA5) | Strong (75) |
| SA3 four compositions | Moderate (Tier-2 heavy; no run) | Valid as taxonomy | Medium (CA7) | Adequate (65): phrase as options with costs, not results |
| SA4 identifiability | Strong (measured bit counts + algebra + theory) | Valid | Low | Strong (85) |
| SA5 residual = coverage | Strong (analytic) | Valid | Low | Compelling (90) for the error = residual-share identity |
| SA6 first experiment | Moderate (cost estimate; outcomes unknown) | Valid conditional rules | Medium (CA8) | Adequate (60): present as a plan, never as a result |

Weak-argument indicator check: none of the eight indicators present in SA1, SA4, SA5. SA3 and SA6 rely on stated-but-untested assumptions; acceptable for non-core arguments, flagged as plans.

## Notes for the draft writer

- Terse: every sentence a claim; no throat-clearing, no hype, minimal em dashes. User calls verbose prose "AI slop".
- Hedging by tag: [S] "X report/show"; [M] plain past tense with numbers; [M-a] "follows from", "by construction"; [H] "predicts", "would", "is expected to", and never in Results-style tense.
- Do not correct the user's note ("near-perfect reconstruction" at the back); quote it and give the numbers.
- Say "coverage" for $\rho_L$ and Cov; never "faithfulness" for them.
- "Pin credit" only as credit of the families an edit changes (no pin bit).
- Keep every fix in `07_response.md`; see the list at the end of `11_outline.md`.
- IEEE numeric citations; mark sub-Tier-1 sources in the reference list only.
- Section 2 must stand alone: a reader who stops there has (a), (b), (c), identifiability, and the first run.
