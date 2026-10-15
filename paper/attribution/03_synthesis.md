# Phase 3 — Synthesis: attributing lc0 outputs to a per-square human vocabulary

Inputs: `00_context.md`, `01_scoping.md` (RQ, SQ1-5, 9-axis taxonomy, DA Checkpoint 1), `02_biblio_T1..T8.md`.
Only sources in those files are cited. Citation key `[Tk.n]` = entry n of `02_biblio_Tk.md`
(T3 entries `T3-n` are written `[T3.n]`). Tiers as in `01_scoping.md` §2 (1 peer-reviewed,
2 lab report / transformer-circuits, 3 preprint, 4 gray). Claims about lc0 that no source makes are
marked **[inference]** or **[hypothesis]**. Nothing here has been run.

Revised after DA Checkpoint 2 (`04_da_checkpoint2.md`, one loop).

Process note: the deep-research skill's hidden `<!--ref-->`/`<!--anchor-->` markers and the claim-intent
manifest were not emitted; the biblio files carry no citation slugs, and `[Tk.n]` keys serve the same
tracing purpose. Flagged so a downstream finalizer does not treat their absence as an error.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $x$, $x'$, $x_0$ | board; counterfactual board (real edit); baseline board, $x_0\sim\mathcal B$ (real positions) |
| $f=\phi(x)\in\{0,1\}^{64\times K}$, $f_s$ | Rust extractor output; features of square $s$; $\tilde f_s=f_s-\bar f_s$ centred per square |
| $\mathcal G=\{G_1,\dots,G_m\}$, $m\approx 30$ | feature families (occupant, pawn structure, x-ray, threat, route, role, mate, material, ...) |
| $h_L\in\mathbb R^{64\times256}$, $L\in\{e,0,\dots,9\}$ | token at tap $L$ ($e$ = embedding `/attn_body/rehape`) |
| $g_L$ | suffix network $h_L\mapsto y$ |
| $y$ | target: value scalar (W−L logit or Q) or policy logit $z_m$ of move $m$ |
| $D_L(f_s,s)=q_{L,s}+W_L\tilde f_s$ | ridge decoder; $q_{L,s}$ = per-square mean (square-only part) |
| $n_{L,s}=W_L\tilde f_s$, $n^G_{L,s}=W^G_L\tilde f^G_s$ | named part; its family-$G$ share |
| $\varepsilon_{L,s}=h_{L,s}-D_L(f_s,s)$ | error node (what the vocabulary misses at $L$, square $s$) |
| $P_L: h_{L,s}\mapsto\hat f_s$ | probe (encoder) |
| $\Delta^{\rm true}=y(x')-y(x)$ | lc0's own effect of a real board edit |

Centring $f$ per square makes the split $h=q+n+\varepsilon$ canonical given $W_L$: $q$ is the square
mean, $n$ and $\varepsilon$ are zero-mean. Because $q_{L,s}$ depends only on $s$, it is identical for $x$
and $x'$, so **in any two-board comparison the square-identity part drops out exactly**. This disposes of
the DA minor issue "square identity absorbs credit" for every pairwise method below; it persists only for
gradient-at-a-point methods (M1, M4, M10 without baseline).

---

## 1. Thematic synthesis

### Literature matrix (theme × sources)

| Theme | Sources (key) | Tiers | Strength |
|---|---|---|---|
| Th1 Removal rule decides what is explained | T1.1,5,6,7,8,9,10,11,12; T1.3 | 1 (T1.11: 3) | Strong |
| Th2 Readable ≠ used; interventional grading | T3.2,4,5,6,10; T4.8; T7.4,5; T8.3,5; T2.7,9; T1.16,17 | 1 (T2.7, T4.8: 3) | Strong |
| Th3 The residual is structured and must be reported | T2.12; T4.2,9,12,13; T5.6,9,10,13; T6.12,13; T7.9; T8.2 | 1-2, T7.9: 4 | Moderate |
| Th4 Supervised maps ≥ unsupervised dictionaries when a vocabulary exists | T4.15/T5.11; T5.12,15; T6.15; T7.13,15 | 1 | Moderate-strong |
| Th5 Cross-layer maps | T5.3,4,5,6,7,8; T2.6; T3.13; T8.7 | 1-3 (T5.5,6: 2, not peer-reviewed) | Emerging |
| Th6 Identifiability, map capacity, leakage | T7.2,10,11,12; T3.2,16; T2.5,15; T4.7,10,11; T7.13 | 1 (T4.7, T7.11: 3) | Strong (as critique) |
| Th7 Attention as the obstacle for gradients | T1.14,15; T2.13; T6.6,7,8,9; T5.6/T6.13; T8.5 | 1-3 | Moderate |
| Th8 Depth profile of human concepts in game nets | T8.1,4,6,7,13,14; T3.12; T8.15 | 1 (T8.6,7: 3; T8.14 non-main track; T8.15: 4) | Weak-moderate; own probe result contrary |

### Th1. The removal rule, not the operator, decides what an attribution means — Strong

Gradient, IG, DeepLIFT, SHAP and occlusion differ less than their removal rules do. Covert et al. (2021)
[T1.12] classify 26 methods by *how features are removed*; Sundararajan & Najmi (2020) [T1.6] show
conditional, interventional and baseline Shapley give different answers under the same axioms; Janzing
et al. (2020) [T1.8] argue for interventional removal because one explains a function, while Frye et al.
(2021) [T1.10] show off-manifold removal yields wrong explanations. Chen et al. (2020) [T1.11, Tier 3]
dissolve the dispute: interventional = true to the model, conditional = true to the data. Applied here,
the vocabulary admits three removal rules, each with a different explanandum:

| Removal rule | Explains | Status |
|---|---|---|
| flip bits of $f$, re-decode | $\text{lc0}\circ D_L$ (true to the decoder-composed model) | off-manifold in $f$ (DA Major 2) |
| decoded substitution from a real $x'$: $n^G(x)\to n^G(x')$ | lc0 at layer $L$, given the decoder is right | endpoints real, mix not |
| real board edit $x\to x'$ | lc0 (true to model and to chess) | on-manifold, but changes many families at once |

Aas et al. (2021) [T1.9] add the fix the DA asked for: sum Shapley values over groups of dependent
features. Lundstrom et al. (2022) [T1.3] supply distributional baselines (average over real positions)
and internal-neuron IG, which licenses attributing at a hidden layer. Kumar et al. (2020) [T1.7] show both
Shapley variants fail as importance measures (off-manifold vs credit to correlates); Bilodeau et al.
(2024) [T1.17] prove complete, linear attributions can be uninformative about model behaviour unless tied
to a concrete query, which in this project means a board edit. **Consequence**: every method in §2 states its removal rule and is graded against real edits.

### Th2. Readability is not use; interventions are the convergent fix — Strong

Five independent lines agree. NLP probing: Ravichander et al. (2021) [T3.5] (irrelevant properties stay
probe-readable), Hewitt & Liang (2019) [T3.2] (control tasks), Belinkov (2022) [T3.4]. Erasure: Elazar et
al. (2021) [T3.6] find high probe accuracy with small behavioural effect. Causal abstraction: Geiger et
al. (2020) [T7.4] show readable ≠ causally used with interchange interventions. Bottlenecks: Margeloiu et
al. (2021) [T4.8, Tier 3]. Chess: Pálsson & Björnsson (2024) [T8.3] find linear-probe accuracy an
unreliable proxy for importance in a chess agent (they recommend nonlinear probes plus erasure; Th6 keeps
linear maps primary for identifiability, a tension this synthesis does not resolve); Jenner et al. (2024) [T8.5] validate an lc0 probe by
causal effect. Benchmarks with ground-truth counterfactuals (CEBaB [T2.9]; RAVEL [T7.13]; MIB [T6.15/T7.14])
find many correlational methods fail. The context's result 1 ("most families near-perfectly readable at
the embedding") is therefore a statement about the extractor's dependence on occupancy, not about lc0's
use **[inference from T3.2, T3.5]**.

### Th3. The residual is structured, not noise, and should be a reported term — Moderate

Explicit residual terms appear in four separate traditions: IBD's residual (65.9% unexplained share in
ResNet50, Zhou et al., 2018 [T2.12]); PCBM-h's residual predictor (Yuksekgonul et al., 2023 [T4.2]);
Havasi et al.'s side channel (2022) [T4.9]; error nodes in feature circuits (Marks et al., 2025 [T6.12])
and replacement models (Ameisen et al., 2025 [T5.6/T6.13], Tier 2). Engels et al. (2025) [T5.9] show SAE
error is half linearly predictable and that maps from earlier layers reduce it most; Yeh et al. (2020)
[T4.12] normalise "what fraction of the head's accuracy concept reconstructions retain". Disagreement is
about *naming*: Res-CBM (Shang et al., 2024 [T4.13]) and concept-bottleneck SAEs (Kulkarni et al., 2026
[T5.13]) assume residual directions can be matched to candidate concepts; Leask et al. (2025) [T5.10]
show unsupervised dictionaries have no canonical size, so "the residual contains concept X" is
dictionary-dependent. Schut et al. (2025) [T8.2] are the only game-network attempt to name what the
human vocabulary misses, validated behaviourally, not interventionally. Havasi et al. [T4.9] add the
distinction this project needs: residual from **missing vocabulary** vs residual from **too weak a map**.

### Th4. With a fixed vocabulary, supervised maps are the stronger default — Moderate-strong

Makelov et al. (2025) [T4.15/T5.11] use supervised attribute dictionaries (activation = sum of
per-attribute-value vectors, i.e. the structure of $D_L$) as the yardstick for SAEs, judged by
sufficiency, necessity and edit control. Kantamneni et al. (2025) [T5.15] and AxBench (Wu et al., 2025
[T7.15]) find supervised directions match or beat SAE latents; MIB [T7.14] finds DAS best and SAE features
no better than neurons. Karvonen et al. (2024) [T5.12/T8.10] score dictionaries in chess/Othello LMs against
rule-computed board features. The only counter-signal is degree: Makelov et al. report newer SAEs approach
the supervised dictionary. Net: SAEs belong on $\varepsilon_L$ (M12), not as a replacement for $D_L$.

### Th5. Cross-layer maps exist only unsupervised, and mostly unreviewed — Emerging

Transcoders (Dunefsky et al., 2024 [T5.3]) and skip transcoders (Paulo et al., 2025 [T5.4], Tier 3) map
a block's input to its output; crosscoders (Lindsey et al., 2024 [T5.5], Tier 2) share one latent set
across layers; CLTs (Ameisen et al., 2025 [T5.6], Tier 2) write from layer $\ell$ to all later layers with
error nodes; multi-layer SAEs (Lawson et al., 2025 [T5.7]) and Jacobian SAEs (Farnik et al., 2025 [T5.8])
make layer-stability and block Jacobians measurable. GCAV (He et al., 2025 [T2.6]) is the only concept-map
method fusing several layers. On lc0, Lin et al. (2026) [T8.7, Tier 3] replace every MLP and attention
module of BT4 with sparse layers (explained variance > 0.6 per module), labelled by hand and checked
against rule-based detectors. No source fits a *supervised* cross-layer map. One structural caveat no source
addresses: CLT/crosscoder reasoning assumes an additive residual stream; lc0 is post-LN, so a layer's
token is a normalised, not summed, function of earlier writes **[inference]**.

### Th6. Identifiability: capacity, inner product and correlated concepts — Strong as critique

Sutter et al. (2025) [T7.12] prove that with unrestricted alignment maps any network abstracts any
algorithm and report near-perfect IIA on random networks; Hewitt & Liang [T3.2] make the same point for
probes. Beckers & Halpern (2019) [T7.2]: our vocabulary is not a constructive abstraction (bits are
functions of overlapping square sets), so per-bit attribution cannot be unique. Park et al. (2024)
[T3.16]: probe directions and decoder columns coincide only under a causal inner product, so chaining them
needs whitening. Nicolson et al. (2025) [T2.5]: CAVs are layer-inconsistent and entangled. Leakage
(Mahinpei et al. [T4.7]; Espinosa Zarlenga et al. impurity [T4.10]; Parisini et al. [T4.11]) and RAVEL's
Isolate score [T7.13] give measurable diagnostics. Ramaswamy et al. (2023) [T2.15] cap useful
explanations at ~32 concepts. **Convergent prescription**: attribute to ~30 families, linear maps
primary, MLP maps only with a random-init control, report stability across regularisation.

### Th7. Attention is where gradient attributions break — Moderate

Ali et al. (2022) [T1.14] show Gradient×Input violates conservation through attention and LayerNorm and fix
it by detaching both; AttnLRP [T1.15] gives softmax rules; CRP [T2.13] conditions relevance on a latent
subspace. AtP* (Kramár et al., 2024 [T6.7], Tier 2) identifies the false negatives of linearised patching:
attention saturation and direct/indirect cancellation. Ameisen et al. [T6.13] avoid the problem by
*freezing* attention — which on lc0 is costly, because Jenner et al. [T8.5] place look-ahead in specific
attention heads. Ferrando & Voita (2024) [T6.9] give a forward-only route decomposition that fits ONNX. lc0's
smolgen is covered by none of the rule sets.

### Th8. Human concepts peak early and fade in game networks — Moderate

McGrath et al. (2022) [T8.1] and Hammersborg & Strümke (2024) [T8.4] track concepts per layer in
AlphaZero and lc0; Lomasov et al. (2025) [T8.14; Creative AI track, 6 concepts, a different 270M chess
transformer] report human-concept probe accuracy falling from 85% in early layers to 50-65% late. That is
probe decay on another model, not our decoder-FVU decay. The project's own probe result points the other
way: families stay readable with depth and some gain to L5. So on lc0 the features do not vanish; they
shrink as a share of token variance (§2). Tomlin et al. (2022) [T8.13] find later layers better in Go —
possibly a label-type difference (below, hypothesis). Sandmann et al. (2025) [T8.6, Tier 3] show lc0's late layers can override correct
mid-layer solutions: what a layer encodes need not be what the output uses, so per-layer readings must be
chained to $y$.

### Contradictions & resolutions

| Claim A | Claim B | Resolution |
|---|---|---|
| Janzing [T1.8]: interventional removal | Frye [T1.10]: on-manifold removal | Reconciled by Chen [T1.11]: different explananda. Bit flips through $D$ explain lc0∘D; board edits explain lc0 and chess. Report both, label which. |
| Makelov [T7.10]: subspace patching yields illusions | Wu et al. reply [T7.11, Tier 3]: definition also flags valid distributed solutions | Reconcilable: run the row-space/nullspace decomposition as a diagnostic, not a gate. |
| Tomlin [T8.13]: Go terms best read late | Lomasov [T8.14], context table: human concepts peak early | **[hypothesis]** Tomlin's labels are commentary mentions (noisy, partly outcome-adjacent, though atari is board-computable); ours and Lomasov's are board-computed. Also: our decay is in FVU, Lomasov's in probe accuracy, and our own probes do not decay. Engine-derived families (`mate`, `material`) should behave like Tomlin's — a testable prediction. |
| Li et al. [T3.14/T8.11]: nonlinear probes needed | Sutter [T7.12]: nonlinear maps vacuous | Reconciled by Nanda et al. [T3.15/T8.12]: with the right (side-to-move-relative) encoding the representation is linear. Use linear maps on a relative vocabulary; MLP maps only with random-init control. |
| Sundararajan IG [T1.1]: complete, axiomatic | Bilodeau [T1.17]: complete methods can be uninformative | Not contradictory: axioms are necessary, not sufficient. Grade against edits. |
| Engels [T5.9]: error is ~half linearly predictable | Marks [T6.12]: error nodes carry large uninterpretable effect | Conditional: predictable ≠ nameable. Decompose $\varepsilon_L$ before calling it unnamed (M12). |
| Ameisen [T6.13]: freeze attention, exact linear graph | Jenner [T8.5]: lc0's key computation is in attention heads | **Unresolved for lc0.** Frozen-attention graphs (M7) will attribute routing to whoever wrote the values, not to QK. Mitigate with AtP*-style QK recomputation [T6.7]; report as a limitation. |
| Makelov 2025 [T4.15]: new SAEs approach supervised | Kantamneni [T5.15], AxBench [T7.15], MIB [T7.14]: SAEs don't beat baselines | Degree difference, same direction: supervised ≥ SAE. |

#### Cross-Paper Tension Inventory

```yaml
cross_paper_tensions:
  - pair_id: CP-001
    paper_a: "T1.8 Janzing 2020"
    paper_b: "T1.10 Frye 2021"
    candidate_basis: "opposite finding direction"
    overlap_topic: "which removal distribution feature attribution should use"
    a_finding: "interventional (marginal) removal; explain the function"
    a_evidence_pointer: "02_biblio_T1.md entry 8 annotation"
    b_finding: "on-manifold removal; off-manifold Shapley misleads"
    b_evidence_pointer: "02_biblio_T1.md entry 10 annotation"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "03_synthesis.md > Contradictions & resolutions, row 1"
    scholar_confirmation: "pending"
  - pair_id: CP-002
    paper_a: "T7.10 Makelov 2024"
    paper_b: "T7.11 Wu 2024 reply"
    candidate_basis: "opposite finding direction"
    overlap_topic: "validity of subspace interchange interventions"
    a_finding: "high IIA can come from dormant pathways"
    a_evidence_pointer: "02_biblio_T7.md entry 10"
    b_finding: "illusion criterion also flags valid distributed solutions"
    b_evidence_pointer: "02_biblio_T7.md entry 11"
    pair_assessment: "contradiction"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "03_synthesis.md > Contradictions & resolutions, row 2"
    scholar_confirmation: "pending"
  - pair_id: CP-003
    paper_a: "T8.13 Tomlin 2022"
    paper_b: "T8.14 Lomasov 2025"
    candidate_basis: "opposite finding direction"
    overlap_topic: "layer at which human concepts are most readable in game nets"
    a_finding: "Go terms best predicted from later layers"
    a_evidence_pointer: "02_biblio_T8.md entry 13"
    b_finding: "chess concepts peak early, fall to 50-65% late"
    b_evidence_pointer: "02_biblio_T8.md entry 14"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "03_synthesis.md > Contradictions & resolutions, row 3"
    scholar_confirmation: "pending"
  - pair_id: CP-004
    paper_a: "T3.14 Li 2023"
    paper_b: "T7.12 Sutter 2025"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "nonlinear maps between representation and concepts"
    a_finding: "MLP probes recover Othello board; edits along them work"
    a_evidence_pointer: "02_biblio_T3.md T3-14"
    b_finding: "unrestricted maps make abstraction vacuous; high IIA on random nets"
    b_evidence_pointer: "02_biblio_T7.md entry 12"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "03_synthesis.md > Contradictions & resolutions, row 4"
    scholar_confirmation: "pending"
  - pair_id: CP-005
    paper_a: "T1.1 Sundararajan 2017"
    paper_b: "T1.17 Bilodeau 2024"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "whether completeness makes attributions trustworthy"
    a_finding: "IG satisfies completeness and sensitivity"
    a_evidence_pointer: "02_biblio_T1.md entry 1"
    b_finding: "complete linear methods can be no better than random for model inference"
    b_evidence_pointer: "02_biblio_T1.md entry 17"
    pair_assessment: "no_material_conflict"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
  - pair_id: CP-006
    paper_a: "T5.9 Engels 2025"
    paper_b: "T6.12 Marks 2025"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "nature of dictionary reconstruction error"
    a_finding: "error is ~half linearly predictable, structured"
    a_evidence_pointer: "02_biblio_T5.md entry 9"
    b_finding: "error nodes carry large, uninterpretable effect"
    b_evidence_pointer: "02_biblio_T6.md entry 12"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "03_synthesis.md > Contradictions & resolutions, row 6"
    scholar_confirmation: "pending"
  - pair_id: CP-007
    paper_a: "T6.13 Ameisen 2025"
    paper_b: "T8.5 Jenner 2024"
    candidate_basis: "agent-noted cross-cluster"
    overlap_topic: "whether attention can be frozen when attributing"
    a_finding: "freezing attention gives exact linear graphs"
    a_evidence_pointer: "02_biblio_T6.md entry 13"
    b_finding: "lc0 look-ahead lives in specific attention heads"
    b_evidence_pointer: "02_biblio_T8.md entry 5"
    pair_assessment: "contradiction"
    resolution_status: "flagged_unresolved"
    scholar_confirmation: "pending"
  - pair_id: CP-008
    paper_a: "T4.15 Makelov 2025"
    paper_b: "T5.15 Kantamneni 2025"
    candidate_basis: "shared RQ subtopic"
    overlap_topic: "supervised vs unsupervised dictionaries"
    a_finding: "newer SAEs approach supervised dictionaries"
    a_evidence_pointer: "02_biblio_T4.md entry 15"
    b_finding: "SAE probes do not consistently beat activation probes"
    b_evidence_pointer: "02_biblio_T5.md entry 15"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "03_synthesis.md > Contradictions & resolutions, row 8"
    scholar_confirmation: "pending"
  - pair_id: CP-009
    paper_a: "T8.6 Sandmann 2025"
    paper_b: "T3.12 nostalgebraist 2020 logit lens"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "reading intermediate layers through the output head"
    a_finding: "mid-layer lc0 solutions are overridden late"
    a_evidence_pointer: "02_biblio_T8.md entry 6"
    b_finding: "intermediate states read through the head show the running prediction"
    b_evidence_pointer: "02_biblio_T3.md T3-12"
    pair_assessment: "no_material_conflict"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
  - pair_id: CP-010
    paper_a: "T5.6 Ameisen 2025 (CLT)"
    paper_b: "T8.7 Lin 2026"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "replacement models and unexplained variance"
    a_finding: "completeness 0.80 / replacement 0.61 with error nodes (LLM)"
    a_evidence_pointer: "02_biblio_T5.md entry 6"
    b_finding: "per-module EV > 0.6 on lc0 BT4; residual handling not stated"
    b_evidence_pointer: "02_biblio_T8.md entry 7"
    pair_assessment: "insufficient_overlap"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
```

**Coverage note.** ~122 entries across 8 bibliographies; 10 candidate pairs assessed (shared construct,
opposite direction, cross-cluster). This is a scoped advisory scan, not complete pairwise contradiction
detection; cross-theme pairs not listed may conflict. Bibliographic coupling was used only for inclusion.

---

## 2. The front/back framing, re-derived from the numbers (DA Major 3; DA-2 M-1, M-2, M-4)

The context's table already contradicts the journey's wording. Square-only FVU is 0.164 at the embedding
and 0.897 at L9: square identity explains ~84% of embedding variance and ~10% of L9 variance. So "near-
perfect reconstruction, but mostly square identity" describes the **front**, not the back.

Two normalisations of the same table give different shapes, and both must be reported:

| tap | $e$ | L0 | L1 | L2 | L5 | L9 |
|---|---|---|---|---|---|---|
| features-add (share of total variance) | 0.16 | 0.39 | 0.62 | 0.63 | 0.45 | 0.22 |
| partial $R^2$ (add / square-only FVU) | 0.99 | 0.68 | 0.76 | 0.76 | 0.53 | 0.24 |

Features-add is capped by the square-only FVU, so its rise from $e$ to L2 is mostly the shrinking
square-identity share. Of the variance not explained by square identity, the vocabulary explains most at
$e$, has a local bump at L1-L2 relative to L0, and falls monotonically after.

Three consequences shape the method catalogue:

1. **Embedding attribution = attribution to the square's own input planes.** From the ONNX graph,
   $h_{e,s}=g_s\odot\mathrm{mish}(W[\text{planes}_s;\text{posenc}_s]+b)+a_s$: elementwise-nonlinear and strictly
   local (planes$_s$ = the square's 112 plane values: pieces over 8 history frames plus broadcast castling /
   rule-50 planes; posenc and gates $g_s,a_s$ learned per square). The 0.002 FVU therefore says the vocabulary
   contains each square's occupant; history, castling and rule-50 content land in $\varepsilon_e$. A
   non-local family (x-ray, threat, route, mate line) can enter $h_{e,s}$ only through correlation with the
   square's own planes. Worse, in any pairwise method at $e$ a board edit changes non-local bits on squares
   whose planes did not change; the decoder assigns $W^G\Delta\tilde f^G_s\neq0$ there and $\varepsilon_e$
   cancels it exactly — spurious named credit and an equal, opposite residual, by construction. At $e$,
   attribution is therefore restricted to the local vocabulary (C2), or decoded deltas are zeroed on squares
   with unchanged planes.
2. **The trade-off is coverage vs path length, not reconstruction vs clarity.** At $e$ a gradient crosses
   all 10 blocks ("the jungle"); at L9 only the head. No single site is preferred by the table alone: raw
   features-add favours L1-L2, partial $R^2$ favours $e$, where §2.1 makes it uninformative.
3. **The decay is a result, not a nuisance — but not yet a result about $y$.** The fall of partial $R^2$ from
   L2 to L9 has at least seven candidate causes, to separate before calling anything an "unnamed concept":
   (a) *non-locality* — $D_L$ sees only $f_s$, but attention moves information between squares;
   (b) *map class* — linear map too weak (the MLP decoder's similar trend weakens this but doesn't exclude it);
   (c) *genuinely unnamed* content;
   (d) *dilution* — the named content stays (the project's probes stay accurate with depth) while other
   content grows;
   (e) *post-LN renormalisation* — taps are LN outputs, a per-token rescaling a linear map of $f_s$ cannot fit;
   (f) *output-specific content* — e.g. per-square policy move-scores, which a value target never uses;
   (g) *fit capacity* — thousands of kept bits at 16k training positions.
   (d)-(f) mean FVU decay need not be decay in what explains $y$. M6 measures the decay in output units;
   M12 splits the residual into (a)/(b)/carry-over/remainder. This replaces "hybrid of front and back" with
   **one curve across layers whose endpoints are the user's front and back**.

---

## 3. Method catalogue

Twelve methods in six approach families; the families, not the count, are the claim:
(i) **decoder split** — one decomposition $h_L=q+\sum_G n^G+\varepsilon$ read with different removal
operators: gradient/IG (M1), substitution (M2), two-player Shapley (M6), family Shapley (M9), relevance
(M10); (ii) **erasure** (M3, cascaded across layers in M8); (iii) **probe gradient** (M4);
(iv) **causal alignment** (M5); (v) **supervised replacement graph** (M7); (vi) **residual analysis** (M12).
M11 compares the others. Genuinely cross-layer: M6 (sweep), M7, M8, M12; M9 only with site players.
Each method lists the residual it reports. Axis codes: see §4 table.

Global controls (referenced as **C1-C9**):

| Code | Control | From |
|---|---|---|
| C1 | random-init lc0: refit maps, rerun method | [T1.16], [T7.12] |
| C2 | local vocabulary: occupant + square (+ history/castling/rule-50 bits if added); at $e$ the only admissible vocabulary, since $h_{e,s}$ is a function of the square's own planes (§2.1) | DA Major 3; ONNX graph; [T3.2] |
| C3 | ridge-α sweep ($10^{-2}$-$10^{2}$) and decoder seed / data-split resampling; report rank stability | DA Major 1 |
| C4 | linear vs MLP map; MLP only reported next to C1 | [T7.12], [T3.2] |
| C5 | shuffled-label refit of $D_L$/$P_L$ | [T1.16] |
| C6 | random equal-rank / equal-norm direction | [T3.6] |
| C7 | with vs without engine-derived families (`mate`, `material`, threat lines) | [T3.10], [T4.11] |
| C8 | row-space vs nullspace split of a patch direction (diagnostic) | [T7.10], [T7.11] |
| C9 | noising and denoising directions; fixed ablation type | [T6.4], [T6.14] |

Throughput assumption: 180 pos/s ONNX CPU full forward; a suffix from tap $L$ costs about
$(10-L)/10$ of that **[inference]**. Torch-port backward ≈ 2-3× a forward.

---

### M1. Decoder–gradient at the embedding (DG-e), point and board-path forms

**Definition.** Point form (gradient × decoded input):
$$A^{(e)}_{G}(x)=\sum_s\big\langle\nabla_{h_{e,s}}y,\;n^G_{e,s}\big\rangle,\quad A^{(e)}_\varepsilon=\sum_s\langle\nabla_{h_{e,s}}y,\varepsilon_{e,s}\rangle,\quad A^{(e)}_q=\sum_s\langle\nabla_{h_{e,s}}y,q_{e,s}\rangle .$$
This is the user's $\nabla_E y\cdot\partial D/\partial f$ times $\tilde f$. Board-path form (IG between two real
boards, in activation space): with $\gamma(\alpha)=h_e(x_0)+\alpha\,(h_e(x)-h_e(x_0))$ and, by linearity of $D_e$,
$h_e(x)-h_e(x_0)=\sum_G W^G_e(\tilde f^G-\tilde f^G_0)+(\varepsilon_e-\varepsilon_{e,0})$:
$$\mathrm{IG}_G=\int_0^1\!\big\langle\nabla y(\gamma(\alpha)),\,W^G_e(\tilde f^G-\tilde f^G_0)\big\rangle d\alpha,\qquad \sum_G\mathrm{IG}_G+\mathrm{IG}_\varepsilon=y(x)-y(x_0).$$
$q$ cancels. Baseline: $x_0$ a real edit of $x$, or $\mathbb E_{x_0\sim\mathcal B}$ (distributional IG, [T1.3]).

**Sources.** IG [T1.1], distributional baselines [T1.3]; CG as the encoder-side twin [T2.2]; no source
attributes through a fitted decoder [T2 gaps].

**Answers.** Which families carry the input-side sensitivity of $y$, re-expressed in the vocabulary.

**Failure modes.** $h_{e,s}$ is local in the square's planes (§2.1), so M1 is input saliency on those
planes re-expressed in the vocabulary; non-local families get credit only via correlation with local
content, and in the board-path form get spurious credit cancelled by $\mathrm{IG}_\varepsilon$ on squares whose
planes did not change — zero those deltas or use C2. Split among occupancy-derived families is set by
ridge, not lc0 (DA Major 1). The point form is not complete (first-order, binary $f$): a sensitivity map,
not an attribution with a residual. Gradients through 10 post-LN blocks violate conservation [T1.14];
interior of the path off-manifold [T6.8].

**Controls.** C2 decisive; C3; C1 (a random-init embedding is also local in the planes, so the decoder fit
at $e$ should stay near-perfect — a direct demonstration that $e$-level FVU says nothing about learning).

**Compute.** Torch port. Point: 1 backward/position. IG: ~32 steps. ~1k positions: minutes.

**Residual report.** $\mathrm{IG}_\varepsilon$ (history/rule-50 content and the 0.2% misfit).

**Role.** Baseline that operationalises the DA inversion; expected to be dominated by occupancy.

---

### M2. Decoded-substitution patching at layer $L$ (DSP-$L$), board-sourced

**Definition.** For a real edit $x\to x'$ and family $G$, patch on squares $S$ (all, or those where $f^G$
changed):
$$h^{\rm patch}_{L,s}=h_{L,s}(x)+W^G_L\big(\tilde f^G_s(x')-\tilde f^G_s(x)\big),\quad s\in S;\qquad \Delta^{(L)}_G=g_L(h^{\rm patch})-y(x).$$
Also patch all changed families jointly ($\Delta^{(L)}_{\rm named}$) and leave-one-family-out.
Consistency gap (exact-transformation check, [T7.1]):
$$\gamma_L=\Delta^{\rm true}-\Delta^{(L)}_{\rm named}.$$

**Sources.** Supervised dictionary edits [T4.15/T5.11]; CBM test-time intervention [T4.1], [T4.14];
in-distribution corruption [T6.3]; consistency [T7.1]; matching when no exact edit exists [T2.11].

**Answers.** If lc0's layer-$L$ representation of $G$ moved as the decoder says this edit moves it, how
much would $y$ move — and how much of the true effect is left over.

**Failure modes.** Decoded edit is off-distribution; dormant pathways [T7.10]; correlated families make
single-family patches non-additive (OR-gates, [T6.4]) — hence joint and leave-one-out.

**Controls.** C6 (equal-norm random direction), C8, C3, C1, C9.

**Compute.** ONNX with graph split at the tap: prefix once, suffix per patch. ~30 families × 11 taps ≈
330 suffix passes ≈ 1-2 s/position. 1k edits ≈ 30 min.

**Residual report.** $\gamma_L$ per layer; plus the error-swap patch $\varepsilon_L(x)\to\varepsilon_L(x')$ (feeds M6).

---

### M3. Family erasure per layer (LEACE-Δ), with use-vs-readability curve

**Definition.** Per family $G$, layer $L$, with $W=(\Sigma_{hh}^{1/2})^{+}$ and $P$ the orthogonal
projector onto $\mathrm{col}(W\Sigma_{hz_G})$ ($z_G$ = family bits):
$$r^G_L(h)=h-W^{+}PW\,(h-\mu),\qquad \Delta^{\rm erase}_{G,L}(x)=g_L\big(r^G_L(h_L(x))\big)-y(x),$$
applied on all 64 squares. Report $\mathbb E|\Delta^{\rm erase}|$ and signed means, next to probe AUROC of
$G$ at $L$ (readability). Joint erasure of all families gives $\Delta^{\rm erase}_{\rm all,L}$.

**Sources.** LEACE [T3.9]; amnesic probing [T3.6]; INLP/R-LACE lineage [T3.7], [T3.8]; chess evidence for
erasure over accuracy [T8.3].

**Answers.** Is $G$'s linearly available information at $L$ used downstream.

**Failure modes.** Linear-only guarantee (an MLP can still recover $G$); large families have rank near
256, so erasure becomes "erase everything" — report erased rank and variance removed; concept-target
correlation inflates engine-derived families [T3.10]; recomputation from other squares/layers hides use
(backup behaviour [T6.4]; measured in M8).

**Controls.** C6 (random equal-rank in whitened coordinates), C7, C1, C2.

**Compute.** ONNX-compatible affine patch; one covariance per family per layer (11 × 30, cheap). Same
pass count as M2.

**Residual report.** $y$ after $\Delta^{\rm erase}_{\rm all,L}$: the output that survives removal of all
linearly named information at $L$ is carried by unnamed (or nonlinearly named) content.

---

### M4. Concept gradient through the probe at layer $L$ (CG-$L$), family-level

**Definition.** With family-score probe $P^{\mathcal G}_L:h_{L,s}\mapsto c_s\in\mathbb R^m$ and Jacobian
$J_s=\partial c_s/\partial h_{L,s}\in\mathbb R^{m\times256}$:
$$\mathrm{CG}_{L,s}=(J_s^{\top})^{+}\,\nabla_{h_{L,s}}y,\qquad \rho^{\perp}_{L,s}=\nabla_{h_{L,s}}y-J_s^{\top}\mathrm{CG}_{L,s}.$$
Must be done at family level ($m\approx30<256$). At bit level $K\gg256$, $J_s^\top$ is surjective and
$\rho^\perp\equiv0$ trivially: the "residual" vanishes by construction and the min-norm split is
arbitrary **[inference from T2.2's definition]**. Use the whitened inner product for $J$ [T3.16].

**Sources.** Concept Gradient [T2.2]; TCAV as linear special case [T2.1]; layer-inconsistency of CAVs
[T2.5]; probe-gradient edits [T3.11].

**Answers.** Local sensitivity of $y$ to each family's probe coordinates at $L$, and the share of the
gradient no family direction spans.

**Failure modes.** Probe direction ≠ model direction [T3.16], [T3.11]; readable ≠ used (but the gradient
is taken through lc0, so it is a local-use measure); pseudo-inverse unstable when family directions are
collinear [T2.5].

**Controls.** C1, C3 (probe regularisation), C2, control-task probes [T3.2]; sign agreement with M3.

**Compute.** Torch port; 1 backward per layer-set (all taps in one pass) per position.

**Residual report.** $\|\rho^\perp_L\|^2/\|\nabla_{h_L}y\|^2$ per layer — the gradient-side analogue of FVU.

---

### M5. Interchange alignment per family per layer (DAS/IIA, boundless, multi-task)

**Definition.** Learn orthogonal $R_L$ and subspace size $k_G$ (boundless DAS). For base $x$, source $x'$:
$$h^{\rm int}_{L,s}=h_{L,s}(x)+R_L^{\top}\Pi_{k_G}R_L\big(h_{L,s}(x')-h_{L,s}(x)\big),\ s\in S.$$
High-level counterfactual: for pairs where $x'$ is a minimal real edit whose feature delta is confined to
$G$ (up to the extractor's own downstream dependencies), target $y(x')$; elsewhere a matched board
[T2.11]. Graded IIA $=1-\mathbb E(g_L(h^{\rm int})-y^{*})^2/\mathbb E(y(x)-y^{*})^2$. Multi-task DAS adds
Isolate: pairs that differ only in a correlated $G'$ must leave the $G$-interchange output unchanged [T7.13].
Initialise $R_L$ from $W^G_L$ (decoder columns) and from probe directions; compare.

**Sources.** DAS [T7.7], boundless DAS [T7.8], causal abstraction and IIA [T7.3], [T7.5], RAVEL/MDAS [T7.13],
MIB causal-variable track [T7.14].

**Answers.** Does a subspace at $L$ mediate $G$'s effect on $y$, how many dimensions it takes, and is it
separable from its correlates.

**Failure modes.** Dormant-pathway illusion [T7.10]; requires clean pairs, which exist for some families
(pins, blockers, piece removal) and not others (e.g. "strong square"); multi-square families need joint
interventions on several tokens [T7.8].

**Controls.** C1 (Sutter's random-net IIA is the key baseline [T7.12]), linear only, C8, $k$ sweep.

**Compute.** Torch; per family × layer a small optimisation over pairs (d=256): CPU minutes each;
30 × 11 ≈ hours.

**Residual report.** $1-\mathrm{IIA}$; plus interchange of the orthogonal complement of all family
subspaces jointly (causal-scrubbing-style [T7.9]): output change it causes = share mediated outside the
vocabulary's subspaces.

---

### M6. Named/error split of the same effect at every layer (layer sweep, telescoped) — multi-layer

**Definition.** For a pair $(x,x')$ and each tap $L$, two players: named $n_L$ and error $\varepsilon_L$
($q$ cancels, §0). Exact two-player Shapley with four suffix passes:
$$\phi^{(L)}_{n}=\tfrac12\Big[g_L(q+n'+\varepsilon)-g_L(q+n+\varepsilon)+g_L(q+n'+\varepsilon')-g_L(q+n+\varepsilon')\Big],\quad \phi^{(L)}_{\varepsilon}=\Delta^{\rm true}-\phi^{(L)}_n.$$
Completeness is exact because $g_L(q+n'+\varepsilon')=y(x')$. Named share, aggregated over pairs so it
stays defined when $\Delta^{\rm true}\approx0$, and its layer-to-layer difference:
$$\rho_L=\textstyle\sum_{\rm pairs}\phi^{(L)}_n\big/\sum_{\rm pairs}|\Delta^{\rm true}|,\qquad \kappa_L=\rho_L-\rho_{L+1}.$$
$\kappa_L$ is descriptive, not a flow through block $L$: $\rho_L$ and $\rho_{L+1}$ use independently fitted
decoders on different tokens, attention mixes squares and post-LN renormalises between taps, so $\kappa_L$
also moves with decoder quality. Read block-level structure only where $\kappa_L$ is stable under C3. Family resolution:
replace $n$ by $\{n^G\}$ (→ M9). Gradient variant: conductance through $\mathrm{span}(W_L)$ vs its
complement [T1.4]. Tied-decoder variant: fit $W_L=W+\Delta_L$ with a penalty on $\Delta_L$ (supervised
crosscoder, [T5.5], [T5.7]) to test whether a family's direction is layer-stable.

**Sources.** Two-player split = baseline Shapley [T1.6] with error node as a player [T6.12]; conductance
[T1.4]; completeness score [T4.12]; crosscoders [T5.5]; multi-layer SAE [T5.7].

**Answers.** The user's question directly: at each layer, how much of an actual output change is carried
by what the vocabulary names. Front and back are the two ends of $\rho_L$; no ad-hoc hybrid is needed.

**Failure modes.** Off-manifold mixes ($n'$ with $\varepsilon$); non-locality of $D_L$ makes cross-square
moved information count as error (§2.3a); $\rho_L$ can exceed 1 or go negative when $n$ and $\varepsilon$
cancel — report both $\phi$'s, not only the ratio.

**Controls.** C1, C2 (at $e$, $\rho_e$ under the local vocabulary vs full), C3, C9, neighbour-augmented decoder
$D_L(f_s,\sum_t\bar a^{L}_{st}f_t)$ using lc0's mean attention as a locality control **[proposal]**.

**Compute.** ONNX; 2 new suffix passes × 11 taps per pair (the other two are $y(x)$, $y(x')$) ≈ 0.1 s.
10k pairs ≈ 20 min.

**Residual report.** It *is* the residual report: $\phi^{(L)}_\varepsilon$ and $\kappa_L$.

---

### M7. Supervised local replacement graph with error nodes (supervised CLT analogue) — multi-layer

**Definition.** Per position, freeze attention patterns $A_L(x)$ and LN scales [T1.14], [T6.13]. Replace
each FFN by a supervised skip transcoder that *reads* features from the model, not the extractor:
$$\mathrm{FFN}_L(u)\approx S_Lu+V_L\,\sigma\big(P_L(u)\big)+b_L,\qquad r_{L,s}=\mathrm{FFN}_L(u_s)-[\cdots]\ \ (\text{error node}).$$
With attention and LN frozen, the network is linear in: embedding, transcoder feature writes
$V^G_L\sigma(P^G_L(u_s))$ at each block and square, and error nodes. Edge weight from node $(L,s,G)$ to
$y$ = activation × virtual weight through frozen attention (square-to-square routes). Completeness =
share of $|$attribution$|$ through non-error nodes; replacement score as in [T6.13].
Reading $f$ via $P_L$ (not feeding extractor bits) matters: an oracle-fed transcoder can fit well even
if lc0 never computes $G$ (leakage, [T4.7]).

**Sources.** Transcoders [T5.3], skip transcoders [T5.4], CLT + error nodes + frozen attention
[T5.6/T6.13], sparse feature circuits [T6.12], Jacobian SAEs [T5.8]; lc0 unsupervised precedent [T8.7].

**Answers.** Which family, written at which block on which square, reaches $y$ along which square route;
where in depth the error nodes take over.

**Failure modes.** Frozen QK hides attention routing, which is where lc0 look-ahead lives [T8.5] (CP-007,
unresolved); post-LN: freezing LN scale is a local linearisation, not exact **[inference]**; smolgen
unmodelled; one graph per position.

**Controls.** C1, C3, C4, AtP*-style QK recomputation to bound what freezing hides [T6.7]; validate
pruned graphs by ablation against M2/M6.

**Compute.** Torch port; transcoders are ridge fits (minutes); graph per position = forward + one
backward per output (linearity gives all edges via Jacobian-vector products) — seconds on CPU at d=256.

**Residual report.** Error-node share per block and per square; completeness/replacement per position.

---

### M8. Cross-layer erasure/interchange cascade (family→family transfer tensor) — multi-layer

**Definition.** Erase (M3) or substitute (M2) family $G$ at $L$; read every later tap $L'>L$:
$$T^{L\to L'}_{G\to G'}=\mathbb E_x\big[\,\mathrm{score}_{G'}\big(P_{L'}(h_{L'})\big)-\mathrm{score}_{G'}\big(P_{L'}(h^{\,\text{erase }G@L}_{L'})\big)\big],$$
$\mathrm{Rec}^{L\to L'}_G=T^{L\to L'}_{G\to G}$ (does $G$ reappear later = recomputation), and
$\Delta\|\varepsilon_{L'}\|$ / change in $\phi^{(L')}_\varepsilon$ (does $G$ feed unnamed computation).
Gradient variant: block Jacobian $\partial c^{G'}_{L+1}/\partial n^G_L$ [T5.8]. Decoder-read variant (the
user's "several decoders on several layers" in the strict sense): substitute $n^G_L(x)\to n^G_L(x')$ with
$D_L$ (M2) and read the change at $L'$ through $D_{L'}$'s own split, $\Delta n^{G'}_{L'}$ by least squares on
$W_{L'}$, and $\Delta\varepsilon_{L'}$ — a family→family transfer tensor in which every reading is a decoder.

**Sources.** Amnesic probing [T3.6], LEACE scrubbing at every layer [T3.9], Jacobian SAEs [T5.8],
earlier-layer maps explain later error [T5.9], backup behaviour [T6.4].

**Answers.** Which named features lc0 builds from which, where it recomputes an erased feature (so
single-site erasure underestimates use), and which named features are converted into error.

**Failure modes.** Probes applied to patched (off-distribution) activations; recomputation from
unerased squares if $S\ne$ all squares; transfer confounded by extractor-level dependencies between $G$
and $G'$ (report the extractor's own dependency graph alongside).

**Controls.** C6, C1, C7.

**Compute.** ONNX; one patched forward per $(G,L)$ yields all later taps: 30 × 10 passes/position.

**Residual report.** $\Delta\phi_\varepsilon$ per $(G,L)$: named → unnamed flow.

---

### M9. Family Shapley with an error player, per layer, resampled-board baseline — multi-approach

**Definition.** Players $\mathcal P=\{G_1,\dots,G_m,\varepsilon\}$. For coalition $S$ and baseline $x_0\sim\mathcal B$:
$$h^{(S)}_{L,s}=q_{L,s}+\sum_{G\in S}n^G_{L,s}(x)+\sum_{G\notin S}n^G_{L,s}(x_0)+\mathbb 1[\varepsilon\in S]\,\varepsilon_{L,s}(x)+\mathbb 1[\varepsilon\notin S]\,\varepsilon_{L,s}(x_0),$$
$v(S)=\mathbb E_{x_0}[g_L(h^{(S)})]$; $\varphi_p$ = Shapley value; $\sum_p\varphi_p=y(x)-\mathbb E\,y(x_0)$.
Baseline pool matched on material/phase to limit off-manifold mixing. Global variant: ConceptSHAP with
the completeness score as value [T4.12]. Site extension: players $(G,L)$, to ask at which depth $G$'s
representation is used **[proposal]**.

**Sources.** Grouped Shapley under dependence [T1.9]; BShap [T1.6]; distributional baselines [T1.3];
ConceptSHAP [T4.12]; Ramaswamy's ~32-concept cap [T2.15].

**Answers.** A fair split of one position's output among ~30 families **and** the vocabulary's miss, at
each layer; $\varphi_\varepsilon$ is the residual on equal footing with named families.

**Failure modes.** Mixed coalitions combine families from different boards (true to lc0∘$D_L$, not to
chess [T1.11]) and leave the manifold [T1.7]; grouping removes within-family non-identifiability only —
families are collinear with each other (occupant vs every occupancy-derived family), so the split of
$W_L$ into $W^G_L$ is still ridge-dependent at family level (C3 must report family-rank stability); with
$x_0=x'$, unchanged families get zero by the dummy axiom, so localisation is trivial in pairwise form;
cost grows with players.

**Controls.** C1, C3, C7, C2; compare sign/rank with M2 on edited families.

**Compute.** ONNX; permutation sampling, $(m{+}1)\times M\times|\mathcal B|$ suffix passes, e.g. 31 × 20 × 8
≈ 5k ≈ 30 s/position/layer. 200 positions × 3 layers ≈ 5 h.

**Residual report.** $\varphi_\varepsilon$ per layer and its share $\varphi_\varepsilon/\sum|\varphi|$.

---

### M10. Conservative relevance chain through $D_L$ (AttnLRP / G-DeepSHAP) — multi-approach

**Definition.** One modified backward pass gives relevance $R_{L,s}\in\mathbb R^{256}$ at every tap
(AttnLRP rules [T1.15]; LN and attention detached [T1.14]). Split each dimension's relevance over
components in proportion to their (baseline-differenced, G-DeepSHAP rescale [T1.13]) contributions,
with the relevance itself propagated against the same baseline (DeepLIFT/G-DeepSHAP rules, not
zero-referenced LRP, or the split mixes two references):
$$R^{c}_{L,s}=\sum_j R_{L,s,j}\,\frac{\Delta c_{j}}{\Delta h_{L,s,j}},\quad c\in\{n^{G_1},\dots,n^{G_m},\varepsilon\},\qquad \sum_c R^c_{L,s}=\sum_j R_{L,s,j}.$$
All layers from one pass: the cheapest full layer profile.

**Sources.** AttnLRP [T1.15], conservative propagation [T1.14], CRP [T2.13], G-DeepSHAP chains [T1.13],
DeepLIFT rescale [T1.2].

**Answers.** A conserved per-family relevance profile across depth, comparable to M6/M9 at a fraction of
the cost.

**Failure modes.** Conservation is a design property, not faithfulness [T1.15]; ratio unstable where
$\Delta h_j\approx0$; smolgen and post-LN rules unprincipled; DeepLIFT rules through attention untested [T1.13].

**Controls.** C1 (sanity check [T1.16]), agreement with M6 (exact) on the same pairs.

**Compute.** Torch port with custom backward; 1 pass/position for all layers.

**Residual report.** $R^\varepsilon_L$ per layer.

---

### M11. Cross-method, cross-layer disagreement map — multi-approach

**Definition.** For family attributions $A^{(k,L)}_G$ from methods $k$ at taps $L$, on a shared pair set:
rank agreement $\tau_{(k,L),(k',L')}$ (Kendall over families), sign agreement, and residual shares
$A^{(k,L)}_\varepsilon/\Delta^{\rm true}$ side by side. No weights are fitted. A learned ensemble that matches
$\sum_G A_G$ to $\Delta^{\rm true}$ was considered and rejected: it rewards layers and methods with small
$\varepsilon$ (selecting against the residual), and with $\varepsilon$ included every complete method has zero
loss, so the weights are unidentified.

**Sources.** Cross-layer fusion precedent GCAV [T2.6] (its learned fusion is the identifiability problem
avoided here); end-task grounding [T1.17]; CEBaB-style scoring [T2.9].

**Answers.** Where methods and layers agree on which families matter, and where an attribution is a
property of the operator rather than of lc0.

**Failure modes.** Agreement is not faithfulness: methods sharing the decoder split (family (i)) share its
ridge artefacts and will agree for that reason; compare within and across approach families separately.

**Controls.** C1 (agreement on a random net is the floor), C3.

**Compute.** Negligible beyond the component methods.

**Residual report.** Spread of $A_\varepsilon/\Delta$ across methods at the same layer.

---

### M12. Residual budget and hybrid dictionary on $\varepsilon_L$ — multi-approach

**Definition.** (i) Budget: regress $\varepsilon_{L,s}$ on nested predictor sets and record explained
variance increments:
$$\varepsilon_{L,s}\approx\underbrace{B^{\rm nl}_L\,\psi(f_s)}_{\text{(b) nonlinear in }f_s}+\underbrace{B^{\rm nb}_L\textstyle\sum_t\bar a^L_{st}f_t}_{\text{(a) non-local}}+\underbrace{B^{\rm cf}_L\,\varepsilon_{L-1,s}}_{\text{carried forward}}+\eta_{L,s},$$
order permuted (Shapley over the three blocks) since increments depend on order. (ii) Hybrid dictionary:
size-swept sparse code $z=\mathrm{SAE}(\eta_L)$; add $z$ as extra players in M9 / nodes in M7. (iii)
Naming: match latents to held-back candidate features or new extractor bits (coverage as in [T5.12]);
contrastive variant on board pairs [T8.8].

**Sources.** Error decomposition [T5.9]; hybrid concept-bottleneck SAE [T5.13]; residual naming
[T4.13]; non-canonical dictionaries [T5.10]; concept discovery for the miss [T8.2], [T2.14]; missing-
vocabulary vs weak-map [T4.9].

**Answers.** How much of the decay (§2.3) is non-locality, map class, carry-over, or remainder; whether
the remainder contains stable, causally used, nameable units.

**Failure modes.** Unsupervised latents are size-dependent [T5.10]; naming works only if the candidate
bank contains the concept [T4.13]; predictable ≠ nameable.

**Controls.** C1 (latents that also appear on a random net are not concepts), size sweep, seed stability.

**Compute.** Regression: minutes. SAEs on $\eta_L$ (d=256): CPU hours per layer.

**Residual report.** The budget itself; then share of $\varphi_\varepsilon$ recovered by named latents.

---

## 4. Comparison table

Axes (from `01_scoping.md`): 1 entry point · 2 direction · 3 layer span · 4 map class · 5 operator · 6
residual · 7 identifiability · 8 faithfulness test · 9 cost. "Multi" = combines layers/maps/operators.

| # | Method | 1 Entry | 2 Dir | 3 Span | 4 Map | 5 Operator | 6 Residual | 7 Identif. | 8 Faith test | 9 Cost | Multi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | DG-e | input | decode | single ($e$), path = 10 blocks | linear | gradient / board-path IG | IG$_\varepsilon$ | non-unique within occupancy-derived bits | vs edits | torch, cheap | no |
| M2 | DSP-$L$ | input | decode | per-layer | linear | substitution patching | consistency gap $\gamma_L$ | unique per joint patch; split by leave-one-out | interventional (edits) | ONNX, ~1-2 s/pos | no |
| M3 | LEACE-Δ | readout | encode | per-layer | linear | erasure | output after erasing all named | unique given $G$ (closed form) | interventional + random-rank | ONNX, cheap | no |
| M4 | CG-$L$ | readout | encode | per-layer | linear/MLP | Jacobian pseudo-inverse | $\|\rho^\perp\|$ (family level only) | min-norm | vs edits; sign vs M3 | torch, 1 bwd | no |
| M5 | DAS/IIA | readout | both | per-layer | linear rotation | interchange | $1-$IIA; complement interchange | up to rotation in subspace | IIA, Isolate | torch, hours | no |
| M6 | Named/error sweep | input | decode | all layers, differenced | linear | 2-player Shapley patching | $\phi_\varepsilon^{(L)}$, $\kappa_L$ | exact (2 players) | completeness exact; C3 stability | ONNX, 0.1 s/pair | yes |
| M7 | Supervised replacement graph | both | both | cross-layer, path-level | sparse-linear | linear edges, frozen attn | error nodes per block/square | regulariser-dependent | ablation vs M2/M6 | torch, s/pos | yes |
| M8 | Erasure cascade | readout | encode→encode | cross-layer | linear | erasure/substitution + re-probe | $\Delta\phi_\varepsilon$ | per $(G,L)$ unique | interventional | ONNX, 300 pass/pos | yes |
| M9 | Family Shapley + ε | input | decode | per-layer (+ sites) | linear | Shapley (BShap, distributional) | $\varphi_\varepsilon$ | unique given value fn, baseline and $W^G$ split (family-level ridge dependence) | localisation (distributional form) | ONNX, ~30 s/pos/layer | only with site players |
| M10 | Relevance chain | input | decode | all layers, 1 pass | linear | DeepLIFT/G-DeepSHAP rescale | $R^\varepsilon_L$ | rule-dependent | agreement with M6 | torch custom bwd | no (sweep) |
| M11 | Disagreement map | mixed | mixed | all | — | comparison of above | spread of residual shares | none fitted | random-net floor | negligible | yes |
| M12 | Residual budget + hybrid dict | dictionary-matched | both | per-layer, uses $L-1$ | linear + sparse | regression + SAE + M9 | budget (a)/(b)/carry/remainder | non-unique (SAE size) | random-init, size sweep | CPU hours | yes |

Five methods combine layers or approaches (M6, M7, M8, M11, M12); M9 joins them only with site players.
Five of the twelve (M1, M2, M6, M9, M10) are operators on one decoder split (§3, family (i)). Five need no
torch port (M2, M3, M6, M8, M9).

---

## 5. Gap analysis

| # | Gap | Type | Evidence of absence | Honest novelty claim |
|---|---|---|---|---|
| G1 | Chain-rule or patching attribution *through a fitted decoder* $f\to h$ | Methodological | T2 gaps (CG uses an encoder pseudo-inverse; CaCE a generative decoder), T7 gaps (all alignment in encode direction), T8 gaps | First use of decoder-route attribution on a game net. Mathematically modest: with linear $D$, M1 is gradient×input in a reparametrised basis and at $e$ reduces to input saliency. The contribution is the interventional variants (M2, M6, M9), not M1. |
| G2 | Supervised cross-layer replacement with error nodes | Methodological | T5 gaps ("no paper trains a supervised cross-layer dictionary"), T4 gaps | M7 combines [T5.3], [T6.13], [T4.15]; novel as a combination, Tier-2 dependence for the CLT half. |
| G3 | Residual/error share reported for a fixed external vocabulary | Empirical | T5 gaps; T8 gaps (Lin et al. report reconstruction, not graph residual) | $\rho_L$, $\kappa_L$ curves (M6) and the residual budget (M12) are new measurements, not new theory. |
| G4 | Interventional validation of named feature families on lc0 | Empirical | T8 gaps: lc0 interventions exist only at square level [T8.5-T8.7] | The protocol in §6. |
| G5 | Attribution operators for binary, logically constrained features | Theoretical | T1 coverage notes: closest are grouping [T1.9] and on-manifold [T1.10] | Not solved here; board edits and matched pairs sidestep it for families with exact edits only. |
| G6 | Post-LN residual stream | Theoretical | No source treats it; CLT/crosscoder designs assume additive writes [T5.5], [T5.6] **[inference]** | Flag only; frozen-LN linearisation (M7) is local. |
| G7 | Separating "unnamed" from "non-local" in decoder residuals | Methodological | No source decomposes residual by locality; Engels [T5.9] decomposes by linear predictability only | M12(i) budget; neighbour-augmented decoder control in M6. |
| G8 | Interventional concept benchmarks outside NLP | Empirical | CEBaB, RAVEL, MIB, CPM all NLP [T2 gaps], [T6], [T7] | A chess benchmark with exact do() operators (MaskPinner, LineBlocker, board edits). |
| G9 | QK/attention routing in feature-level graphs | Methodological | Freezing is standard [T6.13]; lc0's look-ahead is in attention [T8.5] | Unresolved; M7 must report it as a limit. |

**Bibliography gaps (not cited; for Phase 2 follow-up if wanted).** No source retrieved on: hierarchical /
nested-group attribution (e.g. Owen-value style coalitions over family → bit); quantifying how far a
patched activation is off-distribution (beyond [T7.10]); attribution rules for smolgen-type dynamic
attention biases; decoders that condition on other squares' features.

---

## 6. Evaluation protocol sketch

**Ground truth, three tiers, kept separate** (Th1):

| Tier | Source | Measures | Used for |
|---|---|---|---|
| GT-M (true to lc0) | real board edits on lc0: piece removal; moving the pinner off the line (pin removal); interposition on an empty line square (LineBlocker realised on the board); single pawn moves | $\Delta^{\rm true}=y(x')-y(x)$, $\Delta\phi$ per family | grading every method |
| GT-C (true to chess) | MaskPinner / LineBlocker in patched Stockfish | engine's rule-level effect $\Delta v^{\rm SF}$; label "pin/line decisive" | does lc0 attribution to `pin`/`line` families track the engine's causal effect — a separate claim about lc0's alignment with chess, not about attribution faithfulness |
| GT-K (consistency) | decoder edit vs board edit at the same layer | $\gamma_L$ (M2) | how far feature-space edits are from realisable ones [T7.1] |

History: lc0 inputs carry 8 history frames; apply each edit to every frame (or use history-free inputs
for both boards), else $\Delta^{\rm true}$ includes a history artefact. Pair construction: per family, minimal edits where the extractor's feature delta is concentrated in that
family (count changed bits outside it and report). Controls for LineBlocker-type edits need a non-zero
noise floor: control squares chosen by geometry alone previously gave a near-zero floor and skewed
verdicts causal (project note, not a cited source). Position pool: puzzle FENs plus non-puzzle positions
(probe set bias, [T2.15]).

**Metrics.**
- Coverage, not error: for complete pairwise methods (M1 board-path, M2 joint, M6, M9 with $x_0=x'$),
  unchanged families get zero by the dummy axiom, so the ICaCE-style quantity
  $\mathbb E\big|\sum_{G\in\delta}A_G-\Delta^{\rm true}\big|/\mathbb E|\Delta^{\rm true}|$ [T2.9] equals the residual share
  exactly. It is reported as coverage and never used to rank methods; ranking on it would select against
  the residual. These methods are graded by $\gamma_L$, control gaps and stability.
- ICaCE error and localisation (rank of the edited family among $|A_G|$, single-family edits) only for
  non-pairwise methods: M1 point form, M4, M9 distributional, M10.
- IIA and Isolate (M5) [T7.13].
- Residual: $\phi^{(L)}_\varepsilon$, $\varphi_\varepsilon$, $\gamma_L$, completeness [T4.12], [T6.13].
- Stability: Spearman of family rankings across C3 settings and C4 map class.
- Control gaps: metric(trained) − metric(C1 random-init), − metric(C2 local vocabulary), − metric(C5 shuffled).
- Fixed and reported: ablation type, noising/denoising, value metric = logit not probability [T6.3], [T6.14].

**Pre-specified hypotheses [hypothesis].**
H1. At $e$, with decoded deltas zeroed on squares whose planes are unchanged, M1/M6/M9 attributions under
the full vocabulary match the local-vocabulary control (C2) within stability noise; without the zeroing,
non-local families receive credit cancelled by $\varepsilon_e$ (§2.1).
H2. $\rho_L$ (M6) falls with depth after L2; whether it peaks at L1-L2 or at $e$ is open (§2 table).
H3. In the M12 budget, non-locality explains a non-trivial share of $\varepsilon_L$ at L5-L9.
H4. Among non-pairwise methods, interventionally grounded ones (M9 distributional) localise single-family
edits better than gradient/relevance methods (M1 point, M4, M10); pairwise methods are excluded because
localisation is trivial for them.
H5. LEACE use-curves (M3) lag readability curves; recomputation (M8 Rec) is high for occupancy-derived families.
H6. Engine-derived families gain attribution with depth (Tomlin-like profile, CP-003) and lose most under C7.

**Staging and budget (order-of-magnitude, CPU).**

| Stage | Methods | Needs | Est. cost |
|---|---|---|---|
| A | M6, M2, M3, M8, M9 | ONNX graph split at taps; edit pairs | ~1 day CPU for 1-10k pairs |
| B | M1, M4, M5, M7, M10 | torch port of t1-256x10 (verified against ONNX outputs) | port effort + ~1 day CPU |
| C | M11, M12 | outputs of A/B; SAE training on $\eta_L$ | hours-days |

Stage A answers the user's core question (named vs missed share per layer, per family) without a port.
Preregister H1-H6, edit construction and metrics before Stage A (OSF, per `01_scoping.md`).

---

## 7. Evidence convergence map

```
Strong:   [==========] Th1 removal rule (11 sources, Tier 1)
Strong:   [==========] Th2 readable ≠ used (15 sources, Tier 1-3)
Strong:   [========= ] Th6 identifiability/capacity critiques (13 sources, Tier 1-3)
Moderate: [=======   ] Th4 supervised ≥ unsupervised (6 sources, Tier 1)
Moderate: [======    ] Th3 residual structured (13 sources, Tier 1-4; naming contested)
Moderate: [======    ] Th7 attention breaks gradients (9 sources, Tier 1-3)
Weak-mod: [====      ] Th8 depth profile in game nets (8 sources, Tier 1-4; Go and own probes contrary)
Emerging: [===       ] Th5 cross-layer maps (9 sources; key ones Tier 2, none supervised)
Gap:      [          ] decoder-route attribution; supervised CLT; named-family interventions on lc0
```

## 8. Theoretical integration

Causal abstraction [T7.3] is the common frame: $D_L$ and $P_L$ are alignment maps between lc0's layer-$L$
state and a high-level model whose variables are feature families; $\varepsilon_L$ is the unaligned part.
Each method is an intervention on the aligned part (M2, M3, M5, M9), the unaligned part (M5 complement,
M6, M12), or both across layers (M6-M8). Covert's removal taxonomy [T1.12] classifies the operators; Chen's
model/data distinction [T1.11] fixes what each explains. The user's front/back question becomes: where
along depth does the alignment stop holding, measured in output units ($\rho_L$), and why (M12 budget).

## 9. Synthesis limitations

- Every claim about lc0 is a hypothesis; nothing was run. Cost estimates assume suffix cost ∝ remaining depth.
- Key cross-layer sources (crosscoders, CLT, circuit tracing) are Tier 2; T7 is 73% one author lineage.
- Interventional benchmarks are NLP; transfer to chess rests on the project's operators.
- The embedding was checked against the ONNX graph: local and mish-nonlinear in the square's planes, not
  affine as DA Checkpoint 1 assumed. The other lc0 structural claims (post-LN taps, smolgen) are from the
  context brief.
- Pair inventory is a scoped scan, not exhaustive.
- GT-C (MaskPinner/LineBlocker) grades alignment of lc0 with Stockfish's rule-level effect, not attribution
  faithfulness to lc0; conflating the two would overclaim.
