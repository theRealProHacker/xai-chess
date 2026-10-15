<!-- Phase 6 (revision) of the deep-research pipeline. Revises 05_report.md against 06_review_editor.md,
06_review_ethics.md and 06_review_da3.md; every change is listed in 07_response.md. Citations are APA 7
author-year; every cited work is an entry in a 02_biblio_Tk.md file. The skill's hidden ref/anchor markers are
not emitted: the bibliographies carry no slugs. -->

# Attributing a Chess Transformer's Outputs to a Fixed Human Feature Vocabulary: Decoders, Probes, and What the Vocabulary Misses

xai-chess project, Phase 6 revised report, 9 October 2026

## Abstract

**Background.** A per-square vocabulary of 30,837 rule-computed chess features has been mapped to the square tokens of the Leela Chess Zero (lc0) transformer `t1-256x10-distilled-swa-2432500` in both directions: ridge decoders (features to token) and linear probes (token to features), at the embedding and at every encoder block. Decoder fraction of variance unexplained (FVU) rises from 0.002 at the embedding to 0.680 at block 9. **Purpose.** To identify methods that turn these maps into attributions of lc0's value and policy outputs, including methods that combine maps or layers, and to measure the part of the output the vocabulary does not name. **Method.** A structured methods survey (89 cited works from eight theme bibliographies), a nine-axis taxonomy, five adversarial reviews, and checks of the embedding, heads and input encoding against the ONNX graph and the project's code. **Findings.** Twelve methods and one correlational baseline fall into six approach families. The embedding is a per-square nonlinear function of that square's input planes, so embedding-level attribution is input saliency in feature terms. The share of a real output change carried by named features at each tap says where attribution is worth doing. It is a coverage curve, not a per-feature attribution. Joint named effects, consistency gaps and the named share are identifiable; credit split among collinear families is not. For complete pairwise methods the usual faithfulness error equals the residual share, so it is reported as coverage. **Recommendation.** Run decoded substitution and the named/residual split first, on pin edits at four taps, with random-initialisation and ridge-sweep controls. Nothing beyond the existing fits has been run. Method outcomes are hypotheses.

*Keywords:* feature attribution, concept-based explanation, causal abstraction, residual coverage, identifiability, chess, Leela Chess Zero

## 1. Introduction

A chess engine's evaluation can be explained in chess terms only if some map connects its internal state to those terms. The xai-chess project has such a map. A Rust extractor (`features/square/`) emits, for every square, a binary vector of 30,837 bits: occupant, pawn structure, x-rays, threats, routes, piece roles, castling rights on the king, and mate and material lines computed with an engine. About 36.5 bits are on per square. On 20,000 Lichess puzzle positions, split 80/20 by position, two families of maps have been fitted against lc0's square tokens ($d=256$) at the embedding and at each of the ten encoder blocks. Probes read the features from the token. Decoders predict the token from the features plus a one-hot square identity.

Attribution of the output to features is easiest to define where the decoder fits and where the path to the output is short, and these two places are at opposite ends of the network. The project's working note states the dilemma: start at the back, where the reconstruction is "mostly square identity" and attribution must fight through "the deep jungle of attention"; start at the front, where attribution is clearer; perhaps a hybrid combines both. The measured fits at the two ends: square identity alone leaves FVU 0.164 at the embedding and 0.897 at block 9; features plus square identity leave 0.002 and 0.680. Table 2 gives every tap.

This report answers one primary question. *Which families of methods can attribute lc0's value and policy outputs to a fixed per-square feature vocabulary through learned feature–activation maps, and how do they trade off faithfulness under intervention, identifiability, residual coverage, and compute on this system?* The user asked for several approaches rather than one winner, with weight on methods that combine maps across layers, and for a clear first experiment.

Four findings organise the answer. First, the embedding is a local, elementwise-nonlinear function of each square's own input planes, so embedding-level attribution is input saliency with feature names (Section 5.2). Second, five methods are removal operators on one decomposition of the token into square mean, named part and residual. Read at every tap, it gives a curve of the named share of real output changes. The curve says where attribution is worth doing, not which feature did it. Third, every real edit changes several collinear families, and the credit split among them is set by the decoder's regulariser, not by lc0 (Section 7.1). Fourth, a first experiment runs under ONNX in about an hour of CPU time and decides what to run next (Section 9.5).

The report surveys and proposes; it tests nothing.

## 2. Method of the Survey

The design is a structured methods survey, not a PRISMA systematic review: the literature is mostly arXiv and lab reports with unstable terminology.

**Themes and sources.** Eight themes were searched in parallel: attribution operators and their axioms, concept attribution through learned concept maps, probing and erasure, concept bottlenecks and completeness, sparse dictionaries across layers, circuit discovery with error nodes, causal abstraction, and interpretability of game-playing networks. Forum posts were admitted only as gray literature for methods without another source.

**Procedure and inclusion.** Per theme: keyword search, seed papers, one round of citation chasing per seed, and a stop when two consecutive chases added nothing in scope; window 2016 to 2026, plus foundational axiomatic work. Included were works that attribute outputs through a map to or from a concept vocabulary or dictionary, define a faithfulness, completeness or identifiability criterion for such attributions, or apply such methods to a game-playing network.

**Verification and tiers.** Every entry's title, authors, year, venue and identifier were checked against the arXiv record and the publisher or proceedings page. Sources were tiered: 1 peer-reviewed, 2 lab technical reports, 3 preprints and workshop papers, 4 gray literature. The eight theme bibliographies hold about 122 entries, with overlaps between themes; 89 distinct works are cited here. The reference list marks every source below Tier 1.

**Analysis.** Each method received a row on nine axes: vocabulary entry point, map direction, layer span, map class, attribution operator, residual handling, identifiability, faithfulness test, and cost. Two devil's-advocate reviews challenged the scoping and the synthesis; an editorial, an ethics and a third devil's-advocate review challenged the draft report. `07_response.md` lists every resulting change.

## 3. System, Data, and Notation

### 3.1 The model, inputs and targets

The network is lc0 `t1-256x10-distilled-swa-2432500`: 64 square tokens of width 256, ten post-LayerNorm encoder blocks with eight heads and smolgen attention biases, and policy, value and moves-left heads. It runs as ONNX on CPU at about 180 positions per second. The embedding tap is `/attn_body/rehape`. Block taps L0–L9 are each block's first LayerNorm output (`/encoder{i}/ln1`: after attention and its skip connection, before the feed-forward network). The heads read block 9's second LayerNorm output (`/encoder9/ln2`). ONNX on CPU has no backward pass, so gradient methods need a PyTorch port.

Positions are Lichess puzzle FENs, which carry no move history. The input planes therefore use lc0's fen-only history fill (`lc0-probe/lc0_planes.py`): all eight history frames hold the current board, an en-passant pawn is moved back in the older frames, and the repetition planes are zero. Castling rights (four planes), side to move, the rule-50 count and two constant planes are broadcast to all 64 squares. The extractor puts castling rights on the king's square (`king.castle`, commit 473b3ce), and the decoder fits postdate that commit.

The value target is $y=\ell_W-\ell_L$, the difference of the win and loss logits entering the WDL softmax (`/value/dense2`), from the side to move's view. The policy target is the pre-softmax logit of one move (Section 6.8).

**Table 1**
*Existing results other than the per-tap decoder fits (held-out positions; nothing else has been run)*

| Result | Value |
|---|---|
| Linear probe, token at layer $L$ to its own features | most families near-perfectly readable at the embedding; `pawn_cover`, strong squares, mate, passed pawn gain 0.04–0.14 by L5 |
| MLP decoder, features + square to token | trend similar to ridge (Table 2) |
| Board-level decoder, all 64 squares' features to value-head activations and value output | test FVU 0.52–0.62 at 151k positions, still falling with data |
| Value head applied to decoded L9 tokens | weak |
| MLP probe for "pin is relevant" | peak AUROC 0.945 at block 8, vs 0.839 for Stockfish NNUE |

*Note.* Decoders keep 14,565 bits; bits with fewer than 20 positive training examples were dropped.

The project also has exact causal operators in a patched Stockfish: MaskPinner removes the rule-level effect of a pin and LineBlocker places a ray-selective phantom blocker on a line. They supply interventional ground truth about chess, used separately from ground truth about lc0 (Section 9.1).

### 3.2 Notation and the canonical split

Let $x$ be a board, $x'$ a real edit of it, and $x_0$ a baseline board from a pool $\mathcal B$ of real positions. The extractor gives $f=\phi(x)\in\{0,1\}^{64\times K}$, with $f_s$ the bits of square $s$ and $\tilde f_s=f_s-\bar f_s$ their per-square centring. Bits are grouped into about 30 families $\mathcal G=\{G_1,\dots,G_m\}$. The token at tap $L\in\{e,0,\dots,9\}$ is $h_L\in\mathbb R^{64\times256}$, and the suffix network from $L$ to the output is $g_L$. The decoder is

$$D_L(f_s,s)=q_{L,s}+W_L\tilde f_s,$$

with $q_{L,s}$ the per-square mean. Writing $n_{L,s}=W_L\tilde f_s$ for the named part, $n^G_{L,s}=W^G_L\tilde f^G_s$ for family $G$'s share, and $\varepsilon_{L,s}=h_{L,s}-D_L(f_s,s)$ for the residual, every token splits as $h_L=q_L+\sum_G n^G_L+\varepsilon_L$. The residual is the error node: what the vocabulary misses at $L$ on $s$.

Because $q_{L,s}$ depends only on $s$, it is identical for $x$ and $x'$ and cancels in every two-board comparison. In the single-point gradient form (M1) it is not cancelled but dropped: $\langle\nabla y,q\rangle$ is assigned to no player.

Write $\Delta^{\rm true}=y(x')-y(x)$ for lc0's response to a real edit, and $P_L:h_{L,s}\mapsto\hat f_s$ for a probe. Pairs are oriented so that $\Delta^{\rm true}\ge0$: if an edit lowers $y$, $x$ and $x'$ are swapped.

## 4. What the Literature Establishes

### 4.1 The removal rule decides what an attribution means

Attribution methods differ less than their removal rules do: Covert et al. (2021) classify 26 explanation methods by how they remove features. Sundararajan and Najmi (2020) show that conditional, interventional and baseline Shapley values give different answers under the same axioms. Janzing et al. (2020) argue for interventional removal, Frye et al. (2021) show that off-manifold removal misleads, and Chen et al. (2020) reconcile the two: interventional removal is true to the model, conditional removal true to the data. I. E. Kumar et al. (2020) show that both Shapley variants fail as importance measures.

The vocabulary admits three removal rules. Flipping bits of $f$ and re-decoding explains lc0 composed with the decoder, usually on a vector no board produces. Substituting the decoded named part from a real edited board explains lc0 at layer $L$, assuming the decoder is right; its endpoints are real, its mixtures are not. A real board edit explains lc0 itself and is on-manifold, but changes many families at once. Aas et al. (2021) add the remedy for dependent features: Shapley values over groups. Lundstrom et al. (2022) provide distributional baselines and internal-neuron IG. Bilodeau et al. (2024) prove that complete, linear attribution methods can do no better than random at inferring model behaviour for rich model classes, and argue that attributions must be tied to a concrete end-task. Here the end-task is a board edit.

### 4.2 Readable is not used

Irrelevant properties stay probe-readable (Ravichander et al., 2021), probe accuracy must be read against control tasks (Hewitt & Liang, 2019; Belinkov, 2022), and erasure finds high probe accuracy with small behavioural effect (Elazar et al., 2021). Interchange interventions find readable variables that are not causally used (Geiger et al., 2020), and concept bottlenecks show the same gap (Margeloiu et al., 2021). On a chess agent, Pálsson and Björnsson (2024) find linear-probe accuracy an unreliable proxy for concept importance and recommend nonlinear probes with erasure. Jenner et al. (2024) validate an lc0 probe by its causal effect. Benchmarks with counterfactual ground truth, CEBaB (Abraham et al., 2022), RAVEL (Huang et al., 2024) and MIB (Mueller et al., 2025), find that many correlational methods fail. The project's readable-at-the-embedding result therefore says how much the extractor's bits depend on occupancy, not whether lc0 uses them.

### 4.3 The residual is structured and belongs in the report

Interpretable basis decomposition leaves 65.9% of a ResNet-50 prediction unexplained (Zhou et al., 2018). Post-hoc concept bottlenecks add a residual predictor (Yuksekgonul et al., 2023), and Havasi et al. (2022) add a side channel. Feature circuits keep one error node per site (Marks et al., 2025), and replacement models keep error nodes per layer and position (Ameisen et al., 2025). Engels et al. (2025) find that about half of a sparse autoencoder's (SAE) error is linearly predictable from the input activation, and that maps from earlier layers reduce it most. Yeh et al. (2020) define completeness as the accuracy of the model's head on concept-reconstructed activations, normalised between random and full accuracy.

The disagreement concerns naming. Residual concept bottlenecks match residual directions to candidate concepts from a larger bank (Shang et al., 2024). Leask et al. (2025) show that unsupervised dictionaries have no canonical size, so "the residual contains concept X" depends on the dictionary. Schut et al. (2025) are the only game-network attempt to name what a human vocabulary misses, validated behaviourally rather than by intervention. Havasi et al. (2022) separate residual from missing vocabulary and residual from a map too weak to use the vocabulary.

### 4.4 With a fixed vocabulary, supervised maps are the stronger default

Makelov et al. (2025) use supervised attribute dictionaries, in which an activation is a sum of per-attribute-value vectors (the structure of $D_L$), as the yardstick for SAEs. Kantamneni et al. (2025) and AxBench (Wu et al., 2025) find supervised directions match or beat SAE latents for detection and steering; a 2026 follow-up, not verified here, contests the AxBench SAE result. MIB finds distributed alignment search (DAS) strongest and SAE features no better than neurons (Mueller et al., 2025). Karvonen et al. (2024) score dictionaries in chess and Othello models against rule-computed board properties. Newer SAEs approach the supervised dictionary (Makelov et al., 2025), but SAEs belong on the residual, not in place of $D_L$.

### 4.5 Cross-layer maps exist only unsupervised

Transcoders map a block's input to its output (Dunefsky et al., 2024), with a skip variant (Paulo et al., 2025). Crosscoders share one latent set across layers (Lindsey et al., 2024). Cross-layer transcoders write from one layer to all later ones and keep error nodes (Ameisen et al., 2025). Multi-layer SAEs (Lawson et al., 2025) and Jacobian SAEs (Farnik et al., 2025) make layer stability and block Jacobians measurable. GCAV (He et al., 2025) is the only concept-map method found that fuses layers. On lc0, Lin et al. (2026) replace every MLP and attention module of a larger network with sparse layers, reach explained variance above 0.6 per module (lower for attention in middle layers), label features by hand and check them against rule-based detectors. No source fits a supervised cross-layer map. These designs also assume an additive residual stream; lc0's taps are post-LN.

### 4.6 Identifiability limits every map

Sutter et al. (2025) prove that with unrestricted alignment maps any network can be made to abstract any algorithm, and report near-perfect interchange intervention accuracy (IIA) on random networks. Hewitt and Liang (2019) make the same point for probes. Beckers and Halpern (2019) give conditions for constructive abstraction that this vocabulary does not meet, since its bits are functions of overlapping sets of squares. Park et al. (2024) show that probe directions and decoder columns coincide only under a causal inner product. Concept activation vectors are inconsistent across layers and entangled (Nicolson et al., 2025). Leakage has measurable diagnostics (Mahinpei et al., 2021; Espinosa Zarlenga et al., 2023; Parisini et al., 2026; Huang et al., 2024). Ramaswamy et al. (2023) find human usefulness capped near 32 concepts. The convergent prescription: attribute to about 30 families, keep linear maps primary, report MLP maps only beside a random-initialisation control, and report stability across regularisation.

### 4.7 Attention is where gradient attributions break

Gradient×Input violates conservation through attention and LayerNorm; Ali et al. (2022) restore it by detaching both. AttnLRP gives softmax rules (Achtibat et al., 2024), and concept relevance propagation conditions relevance on a latent subspace (Achtibat et al., 2023). AtP* identifies the false negatives of linearised patching: attention saturation and cancellation between direct and indirect paths (Kramár et al., 2024). Ameisen et al. (2025) avoid the problem by freezing attention. On lc0 that is costly: Jenner et al. (2024) use attention-head ablation in their evidence for look-ahead. Ferrando and Voita (2024) give a forward-only route decomposition that would run under ONNX. No relevance rule set covers smolgen, which adds a position-dependent bias to the attention logits before the softmax (ONNX graph).

### 4.8 Depth profiles of human concepts in game networks

McGrath et al. (2022) and Hammersborg and Strümke (2024) track concepts per layer in AlphaZero and lc0. Lomasov et al. (2025) report human-concept probe accuracy falling from 85% early to 50–65% late in a different, 270M-parameter chess transformer, on six concepts: probe decay on another model, not this project's decoder-FVU decay. The project's probes point the other way: families stay readable with depth. Tomlin et al. (2022) find Go terms best read late, possibly because their labels are commentary mentions. Sandmann et al. (2025) show lc0's late layers overriding correct mid-layer solutions, so what a layer encodes need not be what the output uses. Per-layer readings must be chained to $y$.

## 5. The Front/Back Question, Re-derived

### 5.1 Two normalisations of one table

Table 2 gives the decoder fits at every tap. "Features add" is a share of total token variance, capped by the square-only FVU. Dividing by that cap gives the share of non-square variance the vocabulary explains, a partial $R^2$.

**Table 2**
*Ridge decoder fits at every tap (test FVU) and the variance the vocabulary explains*

| Tap | $e$ | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Square only | 0.164 | 0.575 | 0.818 | 0.828 | 0.832 | 0.825 | 0.841 | 0.849 | 0.871 | 0.879 | 0.897 |
| Features only | 0.156 | 0.239 | 0.212 | 0.205 | 0.282 | 0.321 | 0.403 | 0.436 | 0.454 | 0.555 | 0.685 |
| Features + square | 0.002 | 0.183 | 0.198 | 0.195 | 0.272 | 0.311 | 0.392 | 0.427 | 0.447 | 0.549 | 0.680 |
| Features add | 0.16 | 0.39 | 0.62 | 0.63 | 0.56 | 0.51 | 0.45 | 0.42 | 0.42 | 0.33 | 0.22 |
| Partial $R^2$ | 0.99 | 0.68 | 0.76 | 0.76 | 0.67 | 0.62 | 0.53 | 0.50 | 0.49 | 0.38 | 0.24 |

*Note.* From `lc0-probe/results_square_decoder.json` (ridge $\alpha=1$, 14,565 bits). Features add = square only − features + square; partial $R^2$ = features add ÷ square only. L$i$ = `/encoder{i}/ln1`.

The rise of features-add from $e$ to L2 mostly reflects the shrinking square-identity share. Partial $R^2$ is highest at $e$, 0.68 at L0, 0.76 at L1–L2, and falls at every tap from L2 to L9. From L1 on, features alone fit nearly as well as features plus square. Neither normalisation alone picks a best layer.

### 5.2 The embedding is local and nonlinear

The exported ONNX graph gives the embedding per square as

$$h_{e,s}=g_s\odot\mathrm{mish}\big(W[\text{planes}_s;\ \text{posenc}_s]+b\big)+a_s,$$

where $\text{planes}_s$ are the square's 112 input values (13 planes for each of eight history frames, plus the broadcast planes), $\text{posenc}_s$ is a learned positional code, and $g_s,a_s$ are learned per-square gates. The map is elementwise-nonlinear and strictly local: nothing from other squares reaches $h_{e,s}$.

Five consequences follow.

1. The 0.002 FVU says the vocabulary contains each square's occupant. Under fen-only fill the history frames copy the current board, so on this data they carry nothing beyond occupancy except the en-passant detail.
2. Broadcast content the vocabulary does not place on the square falls into $\varepsilon_e$: the rule-50 count everywhere, and castling rights on every square except the king's.
3. A non-local family (x-ray, threat, route, mate line) enters $h_{e,s}$ only through correlation with the square's own planes. In a pairwise method at $e$, an edit changes non-local bits on squares whose planes did not change; the decoder assigns $W^G\Delta\tilde f^G_s\neq0$ there and $\varepsilon_e$ cancels it exactly. The result is spurious named credit with an equal and opposite residual. At $e$, attribution must use the local vocabulary (C2) or zero decoded deltas on squares whose planes are unchanged.
4. Under C2, $\varepsilon'_e-\varepsilon_e\approx0$ for edits that leave the broadcast planes alone, so the named share at $e$ is close to 1 by construction. The front end of the curve in Section 6.1 carries no information about what lc0 learned.
5. A random-initialised embedding is equally local. Control C1 tests what the 0.002 says about learning. If a random embedding also fits near 0.002, the embedding FVU says little beyond locality and occupancy. A clearly larger random-init FVU would measure how far lc0 learned to give the broadcast planes low weight at $e$. On fen-only input the history frames cannot separate the two cases, since they duplicate occupancy.

### 5.3 Coverage against path length

At $e$ a gradient crosses all ten blocks; at L9 it crosses the last feed-forward block and the head. Coverage is high at the front and the causal path long. At the back, coverage is lower and the path short.

### 5.4 Seven candidate causes of the decay

The fall in partial $R^2$ from L2 to L9 has at least seven candidate causes, to be separated before any part of $\varepsilon_L$ is called an unnamed concept:

- (a) *non-locality*: $D_L$ sees only $f_s$, while attention moves information between squares.
- (b) *map class*: a linear map is too weak (the MLP decoder's similar trend weakens this but does not exclude it).
- (c) *genuinely unnamed content*.
- (d) *dilution*: named content stays, as the probes show, while other content grows.
- (e) *post-LN renormalisation*: a per-token rescaling that a linear map of $f_s$ cannot fit.
- (f) *output-specific content*, such as policy content a value target never uses.
- (g) *fit capacity*: 14,565 kept bits against about one million training tokens (16,000 positions × 64 squares). Capacity is the same at every tap, so (g) adds to the decay only if the bits that matter at depth are rare.

Causes (d) to (f) mean that FVU decay need not be decay in what explains $y$. Only methods that measure in output units can settle that; M6 does so at every tap, and M12 splits the residual into causes.

## 6. Methods

Twelve methods and one baseline fall into six approach families. Five methods (M1, M2, M6, M9, M10) read one decomposition, $h_L=q_L+\sum_Gn^G_L+\varepsilon_L$, with different removal operators. The other families are erasure (M3, cascaded in M8), probe gradients (M4), causal alignment (M5), a supervised replacement graph (M7), and residual analysis (M12). M0 is a correlational baseline and M11 compares the rest. In the strict sense of several decoders on several layers composed into one object, M7, M8's decoder-read variant and its composition with M2 (M8∘M2) qualify. M6 and M10 sweep independent decoders across taps.

Each method gives definition, question answered, failure modes, controls, cost and residual. Controls C1 to C9 are shared (Table 3). Costs assume 180 forward passes per second under ONNX, a suffix from tap $L$ costing about $(10-L)/10$ of a pass, and a backward pass in a PyTorch port costing two to three forward passes. They are estimates.

**Table 3**
*Shared controls*

| Code | Control | Basis |
|---|---|---|
| C1 | Random-initialised lc0: refit maps, rerun the method | Adebayo et al. (2018); Sutter et al. (2025) |
| C2 | Local vocabulary (occupant + square, plus rule-50 and per-square castling bits if added); the only admissible vocabulary at $e$ | Section 5.2; Hewitt & Liang (2019) |
| C3 | Ridge-$\alpha$ sweep ($10^{-2}$ to $10^{2}$), decoder seeds, data resampling; report family-rank stability | Section 4.6 |
| C4 | Linear vs MLP map; MLP reported only next to C1 | Sutter et al. (2025) |
| C5 | Shuffled-label refit of $D_L$ or $P_L$ | Adebayo et al. (2018) |
| C6 | Random direction of equal rank or norm | Elazar et al. (2021) |
| C7 | With and without engine-derived families (`mate`, `material_win`, threat lines) | A. Kumar et al. (2022); Parisini et al. (2026) |
| C8 | Row-space vs nullspace split of a patch direction, as a diagnostic | Makelov et al. (2024); Wu et al. (2024) |
| C9 | Noising and denoising directions; fixed ablation type | Heimersheim & Nanda (2024); Miller et al. (2024) |

### 6.0 Baseline

**M0. Feature surrogate with family Shapley and a residual.** Fit a surrogate $s(\phi(x))\approx y(x)$ on all 64 squares' features, or $s(P_L(h_L))$ on probe outputs at tap $L$; in the second case add a residual predictor $r(h_L)$ fitted to $y-s$, as in post-hoc concept bottlenecks with a residual (PCBM-h; Yuksekgonul et al., 2023). Attribute $s$ with Shapley values over families (Aas et al., 2021); the residual is $y-s$. On a pair, report $A^{\rm M0}_G(x')-A^{\rm M0}_G(x)$ beside $\Delta^{\rm true}$. M0 answers which families *predict* $y$, which is true to the data rather than to lc0 (Chen et al., 2020). The project already has its first stage: the board-level decoder of Table 1 reaches test FVU 0.52–0.62 and is still improving with data. Engine-derived families predict value for trivial reasons, so C7 is mandatory, and collinear families share credit by the surrogate's regulariser. M0 is the floor: an interventional method that localises edits no better than M0 adds nothing. Cost: fits in minutes, no lc0 passes beyond $y$. Residual: $\mathbb E(y-s)^2/\mathrm{Var}(y)$, and on pairs the coverage of Section 8 with $A_{\rm named}=s(x')-s(x)$.

### 6.1 Family (i): the decoder split

**M1. Decoder gradient at the embedding.** The point form multiplies the output gradient by each component:

$$A^{(e)}_G(x)=\sum_s\big\langle\nabla_{h_{e,s}}y,\ n^G_{e,s}\big\rangle,\qquad A^{(e)}_\varepsilon=\sum_s\big\langle\nabla_{h_{e,s}}y,\ \varepsilon_{e,s}\big\rangle.$$

This is the user's $\nabla_E y\cdot\partial D/\partial f$ multiplied by $\tilde f$. It is first-order on binary features and leaves $\langle\nabla y,q\rangle$ unassigned, so it is a sensitivity map, not a complete attribution. The board-path form runs IG (Sundararajan et al., 2017) in activation space between two real boards. With $\gamma(\alpha)=h_e(x_0)+\alpha\,(h_e(x)-h_e(x_0))$ and $D_e$ linear,

$$\mathrm{IG}_G=\int_0^1\sum_s\big\langle\nabla_{h_{e,s}}y(\gamma(\alpha)),\ W^G_e(\tilde f^G_s-\tilde f^G_{0,s})\big\rangle\,d\alpha,\qquad \sum_G\mathrm{IG}_G+\mathrm{IG}_\varepsilon=y(x)-y(x_0).$$

The baseline is a real edit of $x$ or an average over real positions (Lundstrom et al., 2022). By Section 5.2, M1 is input saliency on each square's planes re-expressed in the vocabulary. Gradients through ten post-LN blocks violate conservation (Ali et al., 2022). Controls: C2 decides, C3, C1. Cost: one backward pass per position for the point form, about 32 for IG, after a PyTorch port. Residual: $\mathrm{IG}_\varepsilon$, which collects broadcast-plane content and the 0.2% misfit.

**M2. Decoded-substitution patching at layer $L$.** For a real edit and family $G$, patch the named part on the squares $S$ where $f^G$ changed:

$$h^{\rm patch}_{L,s}=h_{L,s}(x)+W^G_L\big(\tilde f^G_s(x')-\tilde f^G_s(x)\big),\ \ s\in S;\qquad \Delta^{(L)}_G=g_L(h^{\rm patch})-y(x).$$

Patching all changed families jointly gives $\Delta^{(L)}_{\rm named}=g_L(q+n'+\varepsilon)-y(x)$; leave-one-out patches give $\Delta^{(L)}_{-G}$. The consistency gap

$$\gamma_L=\Delta^{\rm true}-\Delta^{(L)}_{\rm named}$$

checks whether the decoder edit commutes with the real one (causal consistency; Rubenstein et al., 2017). The design follows supervised dictionary edits (Makelov et al., 2025), test-time concept interventions (Koh et al., 2020; Shin et al., 2023), and in-distribution corruption (Zhang & Nanda, 2024). M2 answers: if lc0's layer-$L$ representation moved as the decoder says this edit moves it, how far would $y$ move, and how much of the true effect is left over? The decoded edit is off-distribution and can activate dormant pathways (Makelov et al., 2024), and single-family patches are not additive when families interact like OR gates (Heimersheim & Nanda, 2024). The joint patch is identifiable; single-family and leave-one-out patches are not (Section 7.1). Controls: C6, C8, C3, C1, C9. Cost: with $k$ changed families, $2k+1$ suffix passes per tap; the pin edits of Section 9.6 change about a dozen families. Residual: $\gamma_L$.

**M6. Named/residual split of the same effect at every layer.** For an oriented pair and each tap, treat the named part $n_L$ and the residual $\varepsilon_L$ as two players ($q$ cancels). The exact two-player Shapley value needs four suffix evaluations, two of which are $y(x)$ and $y(x')$:

$$\phi^{(L)}_n=\tfrac12\Big[g_L(q+n'+\varepsilon)-g_L(q+n+\varepsilon)+g_L(q+n'+\varepsilon')-g_L(q+n+\varepsilon')\Big],\qquad \phi^{(L)}_\varepsilon=\Delta^{\rm true}-\phi^{(L)}_n.$$

The first difference is M2's joint patch, so M6 is M2 symmetrised over the two residual contexts. Completeness is exact because $g_L(q+n'+\varepsilon')=y(x')$. Swapping $x$ and $x'$ negates $\phi_n$, so on oriented pairs the named share and its difference between taps are

$$\rho_L=\frac{\sum_{\rm pairs}\phi^{(L)}_n}{\sum_{\rm pairs}\Delta^{\rm true}}=\frac{\sum_{\rm pairs}\mathrm{sign}(\Delta^{\rm true})\,\phi^{(L)}_n}{\sum_{\rm pairs}|\Delta^{\rm true}|},\qquad \kappa_L=\rho_L-\rho_{L+1}.$$

Without orientation, a complete vocabulary on edits of mixed sign would give $\rho_L\approx0$. This is baseline Shapley (Sundararajan & Najmi, 2020) with the error node as a player (Marks et al., 2025). M6 answers how much of a real output change is carried by what the vocabulary names at each tap. It does not say which family: it has two players. $\rho_L$ is therefore the guide for where per-family attribution (M2, M9) is worth running, at taps where it is high, above C1 and stable under C3. At $e$ under C2 it is near 1 by construction (Section 5.2). $\kappa_L$ is descriptive, not a flow through block $L$: the decoders are fitted independently, attention mixes squares, LayerNorm renormalises, and $\kappa_L$ moves with decoder quality. Mixed coalitions such as $n'$ with $\varepsilon$ are off-manifold, and information moved between squares counts as residual. $\rho_L$ can exceed 1 or turn negative when the players cancel, so both $\phi$ values are reported. Variants: conductance through the span of $W_L$ against its complement (Dhamdhere et al., 2019), and a tied decoder $W_L=W+\Delta_L$ with a penalty on $\Delta_L$, a supervised analogue of crosscoders (Lindsey et al., 2024; Lawson et al., 2025). Controls: C1, C2 at $e$, C3, C9, and a neighbour-augmented decoder $D_L(f_s,\sum_t\bar a^L_{st}f_t)$ with lc0's mean attention $\bar a^L$ as a locality control. Cost: one suffix pass per tap per pair beyond M2's joint patch. Residual: $\phi^{(L)}_\varepsilon$ and $1-\rho_L$.

**M9. Family Shapley with an error player.** Players are $\mathcal P=\{G_1,\dots,G_m,\varepsilon\}$. For coalition $S$ and baseline $x_0$,

$$h^{(S)}_{L,s}=q_{L,s}+\sum_{G\in S}n^G_{L,s}(x)+\sum_{G\notin S}n^G_{L,s}(x_0)+\mathbb 1[\varepsilon\in S]\,\varepsilon_{L,s}(x)+\mathbb 1[\varepsilon\notin S]\,\varepsilon_{L,s}(x_0),$$

with value $v(S)=\mathbb E_{x_0}[g_L(h^{(S)})]$ and Shapley values $\varphi_p$ summing to $y(x)-\mathbb E\,y(x_0)$. Two forms answer different questions. *Pairwise* M9 explains $x'$ against the single baseline $x$ of an oriented pair, so $\sum_p\varphi_p=\Delta^{\rm true}$ and unchanged families receive zero by the dummy axiom; its coverage is defined as M6's (Section 8). *Distributional* M9 uses a pool matched on material and phase and explains $y(x)-\mathbb E\,y(x_0)$, not an edit. Grouping follows Aas et al. (2021), the value function baseline Shapley (Sundararajan & Najmi, 2020) with distributional baselines (Lundstrom et al., 2022), a global variant ConceptSHAP's completeness (Yeh et al., 2020), and about 30 players respects Ramaswamy et al.'s (2023) cap. Mixed coalitions combine families from different boards, so the result is true to lc0 composed with $D_L$, not to chess (Chen et al., 2020), and leaves the data manifold (I. E. Kumar et al., 2020). Family values among collinear families depend on the ridge split (Section 7.1). Making players $(G,L)$ asks at which depth $G$ is used. Controls: C1, C3, C7, C2, and sign and rank agreement with M2. Cost: $(m+1)\times M\times|\mathcal B|$ suffix passes by permutation sampling, for example $31\times20\times8\approx5{,}000$, about 30 s per position per layer. Residual: $\varphi_\varepsilon$.

**M10. Conservative relevance chain through $D_L$.** One modified backward pass, in a PyTorch port, propagates relevance against a baseline with DeepLIFT rescale rules (Shrikumar et al., 2017), chained across layers as in G-DeepSHAP (Chen et al., 2022), with attention and LayerNorm detached (Ali et al., 2022); AttnLRP (Achtibat et al., 2024) is used only for its softmax handling. This gives relevance $R_{L,s}\in\mathbb R^{256}$ at every tap, split over components by baseline-differenced ratios:

$$R^c_{L,s}=\sum_j R_{L,s,j}\,\frac{\Delta c_j}{\Delta h_{L,s,j}},\quad c\in\{n^{G_1},\dots,n^{G_m},\varepsilon\},\qquad \sum_c R^c_{L,s}=\sum_j R_{L,s,j}.$$

All relevance is relative to one baseline, so propagation and split use the same reference. M10 gives the cheapest full layer profile. Conservation is a design property, not faithfulness; the ratio is unstable where $\Delta h_j\approx0$; rules through smolgen, post-LN and attention are untested. Controls: C1 (Adebayo et al., 2018) and agreement with M6. Residual: $R^\varepsilon_L$ per layer.

### 6.2 Family (ii): erasure

**M3. Family erasure per layer with a use-versus-readability curve.** For family $G$ at layer $L$, LEACE (Belrose et al., 2023) gives the least-squares-optimal linear erasure in closed form. With whitening $W=(\Sigma_{hh}^{1/2})^{+}$ and $P$ the orthogonal projector onto $\mathrm{col}(W\Sigma_{hz_G})$,

$$r^G_L(h)=h-W^{+}PW(h-\mu),\qquad \Delta^{\rm erase}_{G,L}(x)=g_L\big(r^G_L(h_L(x))\big)-y(x),$$

applied on all 64 squares. Report $\mathbb E|\Delta^{\rm erase}|$ and signed means beside the probe AUROC of $G$ at $L$. The lineage is amnesic probing (Elazar et al., 2021), INLP (Ravfogel et al., 2020) and R-LACE (Ravfogel et al., 2022); the chess case for erasure over accuracy is Pálsson and Björnsson (2024). M3 answers whether $G$'s linearly available information at $L$ is used downstream. The guarantee is linear only. Large families approach rank 256, where erasure removes everything, so erased rank and variance removed are reported. Concept–target correlation inflates engine-derived families (A. Kumar et al., 2022), and recomputation hides use, which M8 measures. Controls: C6 in whitened coordinates, C7, C1, C2. Cost: one affine patch under ONNX. Residual: the output that survives erasing all families jointly.

**M8. Cross-layer erasure and substitution cascade.** Erase (M3) or substitute (M2) family $G$ at $L$ and read every later tap $L'>L$:

$$T^{L\to L'}_{G\to G'}=\mathbb E_x\Big[\mathrm{score}_{G'}\big(P_{L'}(h_{L'})\big)-\mathrm{score}_{G'}\big(P_{L'}(h^{\,\text{erase }G@L}_{L'})\big)\Big].$$

The diagonal measures recomputation, and the change in $\|\varepsilon_{L'}\|$ or $\phi^{(L')}_\varepsilon$ measures whether $G$ feeds unnamed computation. In the decoder-read variant, $n^G_L(x)$ is replaced by $n^G_L(x')$ and the change at $L'$ is read through $D_{L'}$'s split. A gradient variant uses the block Jacobian (Farnik et al., 2025). Sources: amnesic probing (Elazar et al., 2021), LEACE at every layer (Belrose et al., 2023), earlier-layer maps explaining later error (Engels et al., 2025), and backup behaviour (Heimersheim & Nanda, 2024). M8 answers which named features lc0 builds from which, where it recomputes an erased feature, and which named features turn into residual. Probes read off-distribution activations, recomputation from unerased squares appears unless all squares are erased, and transfer is confounded by the extractor's own dependency graph, which is reported alongside. Controls: C6, C1, C7. Cost: one patched forward pass per $(G,L)$ gives all later taps. Residual: $\Delta\phi_\varepsilon$ per $(G,L)$.

**M8∘M2. Feature-path composition.** For a real edit, apply M2's patch of family $G$ at $L$, run the unfrozen network to $L'$, and regress the change at $L'$ on $D_{L'}$'s columns per square by least squares: $h^{\rm patch}_{L'}-h_{L'}(x)\approx\sum_{G'}W^{G'}_{L'}c_{G'}+r_{L'}$. The path scores are

$$\pi_{G\to G'}=g_{L'}\big(h_{L'}(x)+W^{G'}_{L'}c_{G'}\big)-y(x),\qquad \pi_{G\to r}=g_{L'}\big(h_{L'}(x)+r_{L'}\big)-y(x).$$

This is path patching (Goldowsky-Dill et al., 2023) with decoder coordinates at both ends and a residual branch. It answers through which named families at $L'$ the effect of $G$ at $L$ reaches $y$, and how much of it leaves the vocabulary between $L$ and $L'$. Unlike M7 it keeps attention live. Path scores are not additive, so $\sum_{G'}\pi_{G\to G'}+\pi_{G\to r}$ is reported against $\Delta^{(L)}_G$. The regression on $W_{L'}$ inherits the collinearity of Section 7.1, so $G'$ is resolved per family only where C3 ranks are stable. Controls: C1, C3, C6. Cost: one patched pass to $L'$ per $(G,L)$, then one suffix pass per family with nonzero $c_{G'}$; a few hundred passes per pin pair for $L\in\{2,5\}$, $L'\in\{5,9\}$, under ONNX. Residual: $\pi_{G\to r}$.

### 6.3 Family (iii): probe gradient

**M4. Concept gradient through the probe at layer $L$.** With a family-score probe $P^{\mathcal G}_L:h_{L,s}\mapsto c_s\in\mathbb R^m$ and Jacobian $J_s=\partial c_s/\partial h_{L,s}$, the concept gradient (Bai et al., 2023) and its unexplained part are

$$\mathrm{CG}_{L,s}=(J_s^\top)^{+}\,\nabla_{h_{L,s}}y,\qquad \rho^\perp_{L,s}=\nabla_{h_{L,s}}y-J_s^\top\mathrm{CG}_{L,s}.$$

TCAV is the linear special case (Kim et al., 2018). M4 must work at family level, $m\approx30<256$; at bit level $J_s^\top$ is surjective, $\rho^\perp\equiv0$ by construction, and the minimum-norm split is arbitrary. The inner product should be whitened (Park et al., 2024). M4 answers how sensitive $y$ is to each family's probe coordinates and what share of the gradient no family direction spans. Probe directions need not be the model's (Park et al., 2024; Tucker et al., 2021), and the pseudo-inverse is unstable for collinear families (Nicolson et al., 2025). Controls: C1, C3 on probe regularisation, C2, control tasks (Hewitt & Liang, 2019), sign agreement with M3. Cost: one backward pass per position, in a PyTorch port. Residual: $\|\rho^\perp_L\|^2/\|\nabla_{h_L}y\|^2$.

### 6.4 Family (iv): causal alignment

**M5. Interchange alignment per family per layer.** DAS (Geiger et al., 2024), in its boundless form (Wu et al., 2023), learns an orthogonal rotation $R_L$ and a subspace size $k_G$. For base $x$ and source $x'$,

$$h^{\rm int}_{L,s}=h_{L,s}(x)+R_L^\top\Pi_{k_G}R_L\big(h_{L,s}(x')-h_{L,s}(x)\big),\quad s\in S.$$

The high-level counterfactual is $y(x')$ for minimal real edits whose feature change is concentrated in $G$, otherwise a matched board (Gat et al., 2024). Graded IIA is $1-\mathbb E\big(g_L(h^{\rm int})-y^*\big)^2/\mathbb E\big(y(x)-y^*\big)^2$. RAVEL's Isolate score requires pairs that differ only in a correlated family $G'$ to leave the $G$-interchange output unchanged (Huang et al., 2024). $R_L$ is initialised from $W^G_L$ and, separately, from probe directions. The theory is causal abstraction (Geiger et al., 2021, 2025), and MIB's causal-variable track is the closest benchmark (Mueller et al., 2025). M5 answers whether a subspace at $L$ mediates $G$'s effect on $y$, how many dimensions it needs, and whether it separates from its correlates. Subspace patching can produce dormant-pathway illusions (Makelov et al., 2024), though Wu et al. (2024) argue the criterion also flags valid solutions, so C8 is a diagnostic. The decisive control is C1: Sutter et al. (2025) report near-perfect IIA on random networks. Cost: minutes per family and layer on CPU, in a PyTorch port. Residual: $1-\mathrm{IIA}$, plus interchange of the orthogonal complement of all family subspaces jointly, in the manner of causal scrubbing (Chan et al., 2022).

### 6.5 Family (v): supervised replacement graph

**M7. Supervised local replacement graph with error nodes.** Per position, freeze attention patterns $A_L(x)$ and LayerNorm denominators (Ali et al., 2022; Ameisen et al., 2025). Replace each feed-forward block (squared ReLU in this network) with a supervised skip transcoder that reads features from the model:

$$\mathrm{FFN}_L(u)\approx S_Lu+V_L\,\sigma\big(P_L(u)\big)+b_L,\qquad r_{L,s}=\mathrm{FFN}_L(u_s)-\big[S_Lu_s+V_L\sigma(P_L(u_s))+b_L\big].$$

Nodes are the transcoder feature writes $V^G_L\sigma(P^G_L(u_s))$ per block and square, and the error nodes $r_{L,s}$. Given frozen attention patterns, frozen LayerNorm denominators and frozen feature activations, edges between nodes are linear (Ameisen et al., 2025): the edge weight from node $(L,s,G)$ to $y$ is activation times virtual weight through frozen attention, which traces square-to-square routes. The network itself stays nonlinear in the embedding through $\sigma$; linearity holds only between nodes at fixed activations. Completeness is the share of absolute attribution through non-error nodes; Ameisen et al. (2025) report completeness 0.80 and replacement 0.61 on a language model. Reading $f$ through $P_L$ matters: a transcoder fed extractor bits can fit even if lc0 never computes $G$ (leakage; Mahinpei et al., 2021). The parts come from transcoders (Dunefsky et al., 2024; Paulo et al., 2025) and sparse feature circuits (Marks et al., 2025); Lin et al. (2026) is the unsupervised precedent on lc0. M7 answers which family, written at which block on which square, reaches $y$ along which route. Its central weakness: frozen query-key products hide attention routing, and Jenner et al. (2024) use head ablation in their look-ahead evidence, so a frozen graph may credit routing to whoever wrote the values. Smolgen biases the attention logits before the softmax, so freezing patterns freezes it too; its routing is hidden in the same way as query-key routing. AtP*-style recomputation bounds what freezing hides (Kramár et al., 2024). Controls: C1, C3, C4, and ablation of pruned graphs checked against M2 and M6. Cost: ridge fits in minutes; one graph per position in seconds, in a PyTorch port. Residual: error-node share per block and square.

### 6.6 Family (vi): residual analysis

**M12. Residual budget and hybrid dictionary.** First, regress $\varepsilon_{L,s}$ on nested predictor sets:

$$\varepsilon_{L,s}\approx\underbrace{B^{\rm nl}_L\,\psi(f_s)}_{\text{(b) nonlinear in }f_s}+\underbrace{B^{\rm nb}_L\textstyle\sum_t\bar a^L_{st}f_t}_{\text{(a) non-local}}+\underbrace{B^{\rm cf}_L\,\varepsilon_{L-1,s}}_{\text{carried forward}}+\eta_{L,s},$$

with increments averaged over orders (Shapley over the three blocks), following Engels et al.'s (2025) decomposition of SAE error. Second, fit a size-swept sparse code on $\eta_L$ and add its latents as players in M9 or nodes in M7, as in concept-bottleneck SAEs (Kulkarni et al., 2026). Third, try to name latents by matching them to held-back candidate features or new extractor bits, scored as in Karvonen et al. (2024), with a contrastive variant on board pairs (Poupart, 2024). M12 answers how much of the decay in Section 5.4 is non-locality, map class, carry-over or remainder, and whether the remainder holds stable, used, nameable units. Latents depend on dictionary size (Leask et al., 2025), naming works only if the candidate bank contains the concept (Shang et al., 2024), and predictable is not nameable. Controls: C1, size sweep, seed stability. Cost: regressions in minutes, SAEs in CPU hours per layer. Residual: the budget, then the share of $\varphi_\varepsilon$ recovered by named latents.

### 6.7 Comparison across methods

**M11. Cross-method, cross-layer disagreement map.** For family attributions $A^{(k,L)}_G$ from methods $k$ at taps $L$ on one set of oriented pairs, report Kendall rank agreement over families, sign agreement, and coverage as defined in Section 8, aggregated over pairs rather than per pair. A learned ensemble matching $\sum_GA_G$ to $\Delta^{\rm true}$ was rejected: it rewards small $\varepsilon$, and with $\varepsilon$ included every complete method has zero loss and the weights are unidentified. GCAV's learned fusion (He et al., 2025) has the same problem. Agreement is not faithfulness: the decoder-split methods share ridge artefacts, so agreement is computed within and across families separately, against the C1 floor. Cost: negligible.

### 6.8 Policy

The policy head reads `/encoder9/ln2`. For a non-promotion move $a\to b$ the ONNX graph computes

$$\ell_{a\to b}=c\,\big\langle Q\,m(h_a)+b_Q,\ K\,m(h_b)+b_K\big\rangle,\qquad m(h)=\mathrm{mish}(W_1h+b_1),$$

with $m$ applied per token; promotions add a to-square term. The logit depends on two tokens only. At this tap a decoder split gives an exact four-player attribution with players $n_a,\varepsilon_a,n_b,\varepsilon_b$, computed from 16 head evaluations, which separates families on the move's own squares from residual content. That needs a decoder at `/encoder9/ln2`; the existing fits are at ln1. At earlier taps a policy logit is a target like value for M2, M6, M8∘M2 and M9, with $a$ and $b$ marked: credit on families at $a$ or $b$ versus elsewhere says whether the relevant content has been routed onto the move's squares. Moves must be legal on both boards, since the head emits logits for every move in its fixed list and only legal moves are trained. Cause (f) in Section 5.4 implies that late residual may be policy content a value target ignores; $\rho_L$ for value and policy on the same pairs tests it.

## 7. Comparison

**Table 4**
*Methods on the nine taxonomy axes*

| # | Method | Entry | Direction | Layer span | Map | Operator | Residual reported | Identifiability | Faithfulness test | Cost | Combines layers or maps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | Feature surrogate | input | features→$y$ | none or one tap | linear/GBM | Shapley on surrogate | $y-s$ | per-family regulariser-dependent | localisation vs GT-M | minutes | no |
| M1 | Decoder gradient at $e$ | input | decode | $e$; path crosses 10 blocks | linear | gradient, board-path IG | $\mathrm{IG}_\varepsilon$ | per-family ridge-dependent | vs edits | torch; cheap | no |
| M2 | Decoded substitution | input | decode | per layer | linear | substitution patching | $\gamma_L$ | joint patch unique; per-family ridge-dependent | interventional | ONNX; $2k{+}1$ passes/tap | no |
| M3 | Family erasure | readout | encode | per layer | linear | erasure | output after erasing all | unique for $G$ with its linear correlates | interventional + random rank | ONNX; cheap | no |
| M4 | Concept gradient | readout | encode | per layer | linear/MLP | Jacobian pseudo-inverse | $\|\rho^\perp\|$ | min-norm; unstable if collinear | vs edits; sign vs M3 | torch; 1 bwd | no |
| M5 | DAS / IIA | readout | both | per layer | linear rotation | interchange | $1-$IIA; complement swap | up to rotation | IIA, Isolate | torch; hours | no |
| M6 | Named/residual sweep | input | decode | all taps | linear | 2-player Shapley | $\phi^{(L)}_\varepsilon$, $1-\rho_L$ | exact given $D_L$; depends on fit (C3) | completeness exact; C3 | ONNX; 1 pass/tap beyond M2 | sweep |
| M7 | Supervised replacement graph | both | both | cross-layer, path | sparse-linear | linear edges at frozen activations | error nodes | regulariser-dependent | ablation vs M2/M6 | torch; s/pos | yes |
| M8 | Erasure cascade | readout (or decoder) | encode→encode | cross-layer | linear | erasure/substitution, re-read | $\Delta\phi_\varepsilon$ | as M3 (erasure) or M2 (substitution) | interventional | ONNX; 1 pass per $(G,L)$ | yes |
| M8∘M2 | Feature-path composition | input | decode→decode | $L\to L'\to y$ | linear | path patching | $\pi_{G\to r}$ | per-family ridge-dependent at both ends | vs $\Delta^{(L)}_G$ | ONNX; ~hundreds of passes/pair | yes |
| M9 | Family Shapley + $\varepsilon$ | input | decode | per layer (+ site players) | linear | Shapley, pairwise or distributional | $\varphi_\varepsilon$ | per-family ridge-dependent | localisation | ONNX; ~30 s/pos/layer | only with site players |
| M10 | Relevance chain | input | decode | all taps, one pass | linear | DeepLIFT rescale | $R^\varepsilon_L$ | rule-dependent | agreement with M6 | torch, custom bwd | sweep |
| M11 | Disagreement map | mixed | mixed | all | none | comparison | coverage spread | nothing fitted | random-net floor | negligible | yes |
| M12 | Residual budget + hybrid dictionary | dictionary-matched | both | per layer, uses $L-1$ | linear + sparse | regression, SAE | budget (a)/(b)/carry/remainder | non-unique (SAE size) | random init, size sweep | CPU hours | yes |

*Note.* Five methods are operators on one decoder split (M1, M2, M6, M9, M10). M7, M8, M8∘M2, M11 and M12 combine layers or approaches; M6 and M10 sweep. M0, M2, M3, M6, M8, M8∘M2, M9 and M11 run without a PyTorch port.

### 7.1 What is identifiable

Every real edit that changes a pin also changes occupancy-derived families: interposing a bishop in Section 9.6 changes 331 bits, 180 of them route bits. Ridge splits $W_L$ among collinear families by its penalty, so single-family credit from M1, M2, M4, M9 and M8∘M2 is partly an artefact of $\alpha$. C3 detects this but cannot remove it.

*Identifiable* given lc0 and a fixed edit: $\Delta^{\rm true}$. Given $D_L$ as well: the joint named patch $\Delta^{(L)}_{\rm named}$, $\gamma_L$, $\phi^{(L)}_n$, $\phi^{(L)}_\varepsilon$ and $\rho_L$. These depend on the decoder's prediction $W_L\Delta\tilde f$, which is far less sensitive to $\alpha$ than its per-family coefficients, though it still moves with decoder fit; C3 and C4 report how much. Also identifiable: the LEACE effect of erasing $G$ at $L$ (M3), read as the effect of $G$ together with its linear correlates.

*Not identifiable*: the split of a joint effect among collinear families. That covers single-family and leave-one-out patches, per-family Shapley values, concept-gradient coordinates and per-family path scores.

The remedy is to attribute to the edit and describe the edit by the families it changes (Section 9.6). Per-family credit is reported only where C3 ranks are stable; otherwise unstable families are merged into one player (Aas et al., 2021) until ranks stabilise, and the merge is stated.

## 8. The Residual as Coverage

The user's priority is to measure what the vocabulary misses. That has a direct consequence for evaluation.

A standard faithfulness score compares the attribution summed over changed concepts with the true effect, in the style of CEBaB's ICaCE error (Abraham et al., 2022):

$$\mathrm{err}=\frac{\mathbb E\big|\sum_{G\in\delta}A_G-\Delta^{\rm true}\big|}{\mathbb E|\Delta^{\rm true}|}.$$

For a complete pairwise method, unchanged families get nothing (zero delta in M2's joint patch and M1's board-path IG; the dummy axiom in M6 and pairwise M9), and attributions plus residual sum to $\Delta^{\rm true}$. The error is then $\mathbb E|A_\varepsilon|/\mathbb E|\Delta^{\rm true}|$: the residual share. Ranking methods or layers by it would favour whichever pushes most credit into named families, the opposite of measuring what the vocabulary misses.

**Coverage, one definition.** For any method that splits a pair's effect into a named part $A_{\rm named}$ and a residual, over oriented pairs,

$$\mathrm{Cov}=\frac{\sum_{\rm pairs}A_{\rm named}}{\sum_{\rm pairs}\Delta^{\rm true}}.$$

M6 gives $A_{\rm named}=\phi^{(L)}_n$ and $\mathrm{Cov}=\rho_L$; M2's joint patch gives $\Delta^{(L)}_{\rm named}$ and $\mathrm{Cov}=1-\sum\gamma_L/\sum\Delta^{\rm true}$; pairwise M9 gives $\sum_G\varphi_G$; M0 gives $s(x')-s(x)$. M11 compares these on one axis. Distributional M9 explains $y(x)-\mathbb E\,y(x_0)$, not an edit; its absolute residual share $|\varphi_\varepsilon|/\sum_p|\varphi_p|$ answers a different question and is never plotted with Cov.

Coverage is a finding about lc0 and the vocabulary, reported per layer, per family where identifiable, and with and without engine-derived families. Error applies only where an attribution can be wrong without the residual absorbing the difference: M0, the M1 point form, M4, distributional M9 and M10. Complete pairwise methods are graded by $\gamma_L$, by control gaps against C1, C2 and C5, and by stability under C3.

Three residual measures answer three questions. $\phi^{(L)}_\varepsilon$ and $\varphi_\varepsilon$ say how much of a real output change runs through unnamed content at $L$. The M12 budget says why: non-locality, map class, carry-over, or remainder. The M12 dictionary step asks whether the remainder holds stable, used units that a new extractor bit could name. Havasi et al.'s (2022) distinction between missing vocabulary and weak maps is why the budget comes first. Concepts lc0 has that humans have no word for would sit in $\eta_L$ at layers where coverage is low and stable. Schut et al. (2025) are the closest precedent for naming such content. The protocol must allow the answer "nothing nameable" and coverage near zero at depth.

## 9. Evaluation Protocol

### 9.1 Ground truth in three tiers

Ground truth is split by what it is true to (Chen et al., 2020).

**Table 5**
*Ground-truth tiers*

| Tier | Source | Measures | Used for |
|---|---|---|---|
| GT-M (true to lc0) | Real board edits run through lc0: piece removal, pinner off the line, interposition on an empty line square, single pawn moves | $\Delta^{\rm true}$; per-family feature change | Grading every method |
| GT-C (true to chess) | MaskPinner and LineBlocker in patched Stockfish | Engine's rule-level effect; "pin/line decisive" label | Whether lc0's attribution to pin and line content tracks the engine's causal effect: lc0's alignment with chess, not attribution faithfulness |
| GT-K (consistency) | Decoder edit vs board edit at the same layer | $\gamma_L$ (M2) | How far feature-space edits are from realisable ones |

Edits are made on the FEN and re-encoded with the same fen-only fill, so every history frame carries the edit and $\Delta^{\rm true}$ has no history artefact. The cost is scope: lc0 sees synthetic history in which all frames agree, so the protocol says nothing about computation that uses real move history. Generalising needs a replication on game positions with real history. Pairs are minimal edits whose feature change is concentrated in one family where possible, and the number of changed bits outside it is reported. Controls for line-blocking edits need a nonzero noise floor: in earlier project work, control squares chosen by geometry alone gave a near-zero floor that skewed verdicts towards "causal". The pool mixes puzzle and non-puzzle positions, since probe-set choice changes concept explanations (Ramaswamy et al., 2023).

### 9.2 Metrics

- Coverage (Section 8). Never used to rank methods.
- Error and localisation, for non-pairwise methods only (M0, M1 point form, M4, distributional M9, M10). Localisation is computed on the change $A_G(x')-A_G(x)$, with a shared baseline or baseline pool for both boards: the rank of the edited families among $|A_G(x')-A_G(x)|$.
- Consistency: $\gamma_L$ (M2).
- Alignment: IIA and Isolate (M5).
- Stability: Spearman correlation of family rankings across C3 settings and between linear and MLP maps (C4).
- Control gaps: metric on trained lc0 minus metric under C1, C2 and C5.
- Fixed in advance: ablation type, noising or denoising direction, and the target as a logit (Zhang & Nanda, 2024; Miller et al., 2024).

### 9.3 Pre-specified hypotheses

None has been tested.

- **H1.** At $e$, with decoded deltas zeroed on squares whose planes are unchanged, M1, M6 and M9 attributions under the full vocabulary match the local-vocabulary control C2 within stability noise; without the zeroing, non-local families receive credit that $\varepsilon_e$ cancels. Under C1, a random-initialised embedding is fitted by the same vocabulary with FVU close to 0.002, differing mainly by its weight on broadcast planes.
- **H2.** $\rho_L$ falls with depth after L2. Whether it peaks at L0, L1 or L2 is open. $\rho_e$ under C2 is near 1 by construction and excluded.
- **H3.** In the M12 budget, non-locality explains a non-trivial share of $\varepsilon_L$ at L5–L9.
- **H4.** Among non-pairwise methods, distributional M9 localises edits concentrated in one family better than M1 point form, M4 and M10, measured on $A_G(x')-A_G(x)$ with a shared baseline pool, and all four beat M0.
- **H5.** LEACE use curves (M3) lag readability curves, and recomputation (M8) is high for occupancy-derived families.
- **H6.** Engine-derived families gain attribution with depth, resembling Tomlin et al.'s (2022) Go profile, and lose most of it under C7.

### 9.4 Staging and budget

**Table 6**
*Stages and order-of-magnitude CPU cost*

| Stage | Methods | Needs | Estimated cost |
|---|---|---|---|
| A0 | M2 + M6 on pin edits, M0 on the same pairs | ONNX graph split at four taps; pin pairs | about one CPU hour (Section 9.5) |
| A | M3, M8, M8∘M2, M9, M2/M6 on all edit types | edit pairs of every type | about one CPU day |
| B | M1, M4, M5, M7, M10 | PyTorch port verified against ONNX | port effort plus about one CPU day |
| C | M11, M12 | outputs of A and B; SAEs on $\eta_L$ | hours to days |

H1–H6, edit construction and metrics are preregistered before Stage A0.

### 9.5 What to try first

**Experiment.** M2 and M6 on pin edits.

- *Pairs.* About 1,000 from the puzzle pool. Find absolute pins and build two edit types: the pinner moved to a square off the pin line, and a piece of the pinned side placed on an empty square between pinner and pinned piece. Keep material and side to move; reject edits that give check or make the position illegal. Record changed bits per family (as in Table 7) and the MaskPinner label (GT-C). Pins are chosen because the edits are clean, MaskPinner gives chess-side ground truth, and the project's pin probe peaks at AUROC 0.945 at block 8.
- *Taps.* $e$ with the C2 vocabulary only, L2, L5, L9.
- *Quantities.* Per pair and tap: $\Delta^{(L)}_{\rm named}$, $\gamma_L$, $\phi^{(L)}_n$, $\phi^{(L)}_\varepsilon$, and $\Delta^{(L)}_G$, $\Delta^{(L)}_{-G}$ for each changed family. Aggregated: $\rho_L$, $\sum|\gamma_L|/\sum\Delta^{\rm true}$, family-rank stability.
- *Controls.* C1 (random-initialised lc0, decoders refitted at the same taps) and C3 (five $\alpha$ values from $10^{-2}$ to $10^2$, two seeds, one resample). M0 on the same pairs sets the correlational floor.
- *Cost.* About $2k+2\approx26$ suffix passes per tap per setting for $k\approx12$ changed families; four taps, 1,000 pairs, five $\alpha$ values plus C1: about an hour on CPU under ONNX. No PyTorch port.

**Decision rules, fixed before running.**

- At any tap where trained-lc0 $\rho_L$ and family ranks fall inside the C1 range, the decoder split carries no lc0-specific information; drop that tap.
- If $\rho_L$ is high and above C1 at L2 or L5 and $\gamma_L$ is small, the decoded named content carries the effect. The open question is where it is built and where it goes: **promote M8** (erasure cascade with recomputation) and M8∘M2 from those taps to L9. If family ranks are also stable under C3, per-family M2 credit is reportable; if not, report it jointly (Section 7.1).
- If $\gamma_L$ is large at taps where probes read the pin well, the decoder's direction for the changed families does not reproduce what the real edit does: **promote M5** (DAS initialised from $W^G_L$ and from probe directions, with C1).
- If $\rho_L$ drops sharply between L5 and L9, run the M12 budget on that interval before naming anything in $\varepsilon$.

### 9.6 Worked example

The position below illustrates the procedure. It is from an opening, not the puzzle pool, and no lc0 quantity has been computed for it.

Base $x$, after 1.e4 e5 2.Nf3 Nc6 3.Bb5 d6, White to move: `r1bqkbnr/ppp2ppp/2np4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4`. The bishop on b5 pins the knight on c6 to the king on e8. Two edits, both material-preserving and legal with White to move:

- $x'_a$, pinner off the line (bishop b5 to c4): `r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4`.
- $x'_b$, interposition (bishop c8 to d7): `r2qkbnr/pppb1ppp/2np4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4`.

**What the extractor sees.** Running the current extractor on both pairs gives Table 7. The vocabulary has no pin bit. The pin shows as an x-ray bit on the pinner's square (`xray.knight.king.ne.1_2` on b5, which becomes `xray.knight.bishop.ne.1_1` under $x'_b$) and as absent mobility bits on the pinned knight: ten `mobility.abs`/`mobility.rel` bits on c6 switch on under both edits. Everything else that changes is geometry the edit drags along.

**Table 7**
*Bits changed by the two edits, by extractor group (current extractor, engine lines off)*

| | $x\to x'_a$ (Bc4) | $x\to x'_b$ (…Bd7) |
|---|---|---|
| Squares changed | 27 | 21 |
| Bits changed | 356 | 331 |
| `route` | 203 | 180 |
| `mobility` | 40 | 36 |
| `control` | 34 | 36 |
| `role` | 14 | 24 |
| `square.control` | 16 | 20 |
| `pawn_shield` | 15 | 4 |
| `stability` | 15 | 3 |
| `blocked_by` | 3 | 10 |
| `blocks` | 8 | 5 |
| `xray` | 2 | 8 |
| `pawn` | 4 | 2 |
| `occupant` | 2 | 2 |
| `rook` | 0 | 1 |

**Procedure for $(x,x'_b)$, value target, at each tap $L\in\{e\ (\text{C2}),\ \text{L2},\ \text{L5},\ \text{L9}\}$.**

1. Run lc0 on both boards: $y(x)$, $y(x'_b)$, $\Delta^{\rm true}$; swap the boards if $\Delta^{\rm true}<0$.
2. Tap both boards and decode: $n_L$, $\varepsilon_L$, $n'_L$, $\varepsilon'_L$.
3. Patch $n'-n$ on the 21 changed squares into $h_L(x)$ and run the suffix: $\Delta^{(L)}_{\rm named}$ and $\gamma_L$.
4. Patch $n-n'$ into $h_L(x'_b)$: $g_L(q+n+\varepsilon')$, hence $\phi^{(L)}_n$ and $\phi^{(L)}_\varepsilon$.
5. For each of the 13 changed groups, a single patch and a leave-one-out patch: $\Delta^{(L)}_G$, $\Delta^{(L)}_{-G}$.
6. Repeat steps 3–5 for each $\alpha$ in C3 and on the random-initialised network (C1).

**Table 8**
*Output for one pair, value target (template; nothing computed)*

| Tap | $\Delta^{\rm true}$ | $\Delta^{(L)}_{\rm named}$ | $\gamma_L$ | $\phi^{(L)}_n$ | $\phi^{(L)}_\varepsilon$ | $\phi^{(L)}_n$ under C1 | $\phi^{(L)}_n$ range under C3 |
|---|---|---|---|---|---|---|---|
| $e$ (C2) | — | — | — | — | — | — | — |
| L2 | — | — | — | — | — | — | — |
| L5 | — | — | — | — | — | — | — |
| L9 | — | — | — | — | — | — | — |

*Note.* $\Delta^{\rm true}$ is the same in every row. Over all pairs, column $\phi^{(L)}_n$ sums to the numerator of $\rho_L$.

**Table 9**
*Per-family readout for the same pair at L5 (template; nothing computed)*

| Group | $\Delta^{(L)}_G$ | $\Delta^{(L)}_{-G}$ | Rank, $\alpha=10^{-2}$ | Rank, $\alpha=1$ | Rank, $\alpha=10^{2}$ | Reported as |
|---|---|---|---|---|---|---|
| `xray` | — | — | — | — | — | own row if ranks stable, else merged |
| `mobility` | — | — | — | — | — | as above |
| `route` | — | — | — | — | — | as above |
| `control` | — | — | — | — | — | as above |
| … (9 more) | — | — | — | — | — | as above |
| All changed, jointly | $\Delta^{(L)}_{\rm named}$ | — | — | — | — | always |

For the policy, the same pair supports the move Bxc6 (b5 to c6), legal on both boards. At `/encoder9/ln2` its logit splits exactly over $n_{b5},\varepsilon_{b5},n_{c6},\varepsilon_{c6}$, four Shapley values from 16 head evaluations summing to $\ell(x'_b)-\ell(x)$ (Section 6.8). The pair $(x,x'_a)$ needs a move legal on both boards, such as d2–d4.

## 10. Discussion

The user asked how to get attribution from the decoders, whether to start at the front or the back, and whether a hybrid helps.

At the front, the near-perfect fit follows from locality (Section 5.2). Attribution there is input saliency with feature names, the named share is near 1 by construction under the local vocabulary, and whether the embedding FVU reflects learning is a C1 check.

The front/back choice becomes a choice of taps, guided by $\rho_L$. $\rho_L$ is coverage, not attribution: it says where named content carries a real output change, not which family does. Per-family attribution comes from M2, and M9 later, at taps where $\rho_L$ is high and above C1 and where the embedding's locality no longer holds. Table 2 points to L1–L5. Where families are collinear, the per-family answer exists only jointly (Section 7.1). Where $\rho_L$ drops, the M12 budget says whether the drop is non-locality, a weak map, carry-over or content without a name.

Maps combine across layers in four ways. M6 compares independent per-layer decoders on one effect. M8 intervenes at one layer and reads later layers, which measures recomputation. M8∘M2 composes substitution at one layer with decoder reads at a later one into feature paths, keeping attention live. M7 composes block-level maps into one graph at the price of freezing attention. No source fits a supervised cross-layer map, and no game-network paper reports a residual share in output attribution (Table 10).

**Table 10**
*Gaps the protocol addresses*

| Gap | Evidence of absence | Claim |
|---|---|---|
| Attribution through a fitted decoder $f\to h$ | Concept gradients use an encoder pseudo-inverse (Bai et al., 2023); alignment methods work in the encode direction | First use on a game network. M1 reduces to input saliency at $e$; the contribution is the interventional variants M2, M6, M9 |
| Supervised cross-layer replacement with error nodes | All cross-layer dictionaries are unsupervised (Section 4.5) | M7 combines existing parts; it depends on Tier-2 sources |
| Residual share for a fixed external vocabulary | Lin et al. (2026) report reconstruction per module, not a graph residual | $\rho_L$ and the M12 budget are new measurements, not new theory |
| Interventional validation of named families on lc0 | lc0 interventions target squares and attention heads (Jenner et al., 2024), steering directions (Sandmann et al., 2025) or unsupervised sparse features (Lin et al., 2026); none targets a fixed human feature family | The protocol in Section 9 |
| Residual split by locality | Engels et al. (2025) split error by linear predictability only | M12 budget; neighbour-augmented decoder |
| Interventional concept benchmarks outside NLP | CEBaB, RAVEL and MIB are NLP | A chess benchmark with exact board-level and engine-level operators |
| Attribution for binary, logically constrained features | Closest: grouping (Aas et al., 2021) and on-manifold Shapley (Frye et al., 2021) | Not solved; board edits sidestep it only for families with exact edits |
| Post-LN stream; attention routing in feature graphs | Crosscoder and CLT designs assume additive writes and freeze attention | Flagged as limits |

The strongest objection: the report answers how much of lc0 is nameable per layer, not which human feature caused this value; its curve is fixed by construction at the front; and its per-family numbers split credit among collinear families by a ridge penalty. This report accepts most of it. The per-feature answer exists only jointly for collinear families, which is why Section 7.1 states it and why every decoder-split method carries C1 to C3. Where ranks are stable, M2 per changed family is the per-feature answer. Where they are not, M3, M5 and M8 carry the claims that do not follow from linear algebra, and M12 carries the residual.

Two tensions stay open. Pálsson and Björnsson (2024) recommend nonlinear probes for chess agents, while Sutter et al. (2025) argue for linear maps. Nanda et al. (2023) suggest a side-to-move-relative encoding can make a representation linear, which this vocabulary should test. And freezing attention, which makes M7's edges linear, conflicts with Jenner et al.'s (2024) head-ablation evidence for look-ahead.

## 11. Limitations

Nothing in the method catalogue has been run. The empirical inputs are the fits in Tables 1 and 2, from 20,000 positions for the per-square maps, and the extractor bit counts in Table 7. Every statement about how a method behaves on lc0, including H1–H6, the decision rules and the costs, is a hypothesis or an estimate.

All inputs are FEN-only with fen-only history fill. Conclusions hold for lc0 on synthetic history, not for lc0 on games with real history. The vocabulary has no pin bit: pins appear through x-ray and mobility bits, which are collinear with route and control bits, so "pin credit" can be reported only as the credit of the families an edit changes. The policy analysis needs a decoder at `/encoder9/ln2` that has not been fitted.

The survey is not a systematic review, and the field is preprint-heavy. The key cross-layer sources are Tier 2, and much of the causal-abstraction literature comes from one lineage. Interventional benchmarks are all from NLP. The embedding, head structure and input encoding were checked against the ONNX graph and the project's encoder; other structural claims come from the project's description of the model. GT-C grades lc0's agreement with Stockfish's rule-level effects, not the faithfulness of an attribution. Engine-derived features partly encode the value target; C7 reports results with and without them.

## AI-Use Disclosure

This report was produced with Claude (Anthropic), run through Claude Code as a multi-agent pipeline: literature search, verification of bibliographic records against arXiv and publisher pages, evidence synthesis, three devil's-advocate reviews, editorial and ethics review, and drafting. The bibliographic annotations were written by AI agents from abstracts and landing pages. Cited claims were checked against those annotations, not against full texts read by a human. The numbers in Tables 1 and 2 come from the project's own code and result files. The embedding, head and input structure were read from the exported ONNX graph and `lc0-probe/lc0_planes.py`. The bit counts in Table 7 come from running the project's extractor. The project author directs the research and decides which conclusions to adopt.

## References

Sources below Tier 1 are marked: Tier 2 lab technical report, Tier 3 preprint or workshop paper, Tier 4 gray literature.

Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Artificial Intelligence, 298*, Article 103502. https://doi.org/10.1016/j.artint.2021.103502

Abraham, E. D., D'Oosterlinck, K., Feder, A., Gat, Y., Geiger, A., Potts, C., Reichart, R., & Wu, Z. (2022). CEBaB: Estimating the causal effects of real-world concepts on NLP model behavior. In *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2205.14140

Achtibat, R., Dreyer, M., Eisenbraun, I., Bosse, S., Wiegand, T., Samek, W., & Lapuschkin, S. (2023). From attribution maps to human-understandable explanations through concept relevance propagation. *Nature Machine Intelligence, 5*(9), 1006–1019. https://doi.org/10.1038/s42256-023-00711-8

Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., & Samek, W. (2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR 235, pp. 135–168). https://proceedings.mlr.press/v235/achtibat24a.html

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. In *Advances in Neural Information Processing Systems 31* (pp. 9505–9515). https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps

Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R., & Wolf, L. (2022). XAI for transformers: Better explanations through conservative propagation. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 435–451). https://proceedings.mlr.press/v162/ali22a.html

Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., . . . Batson, J. (2025, March 27). Circuit tracing: Revealing computational graphs in language models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html [Tier 2]

Bai, A., Yeh, C.-K., Lin, N. Y. C., Ravikumar, P., & Hsieh, C.-J. (2023). Concept gradient: Concept-based interpretation without linear assumption. In *International Conference on Learning Representations*. https://arxiv.org/abs/2208.14966

Beckers, S., & Halpern, J. Y. (2019). Abstracting causal models. In *Proceedings of the AAAI Conference on Artificial Intelligence, 33*. https://arxiv.org/abs/1812.03789

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics, 48*(1), 207–219. https://doi.org/10.1162/coli_a_00422

Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect linear concept erasure in closed form. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2306.03819

Bilodeau, B., Jaques, N., Koh, P. W., & Kim, B. (2024). Impossibility theorems for feature attribution. *Proceedings of the National Academy of Sciences, 121*(2), e2304406120. https://doi.org/10.1073/pnas.2304406120

Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., Greenblatt, R., Nitishinskaya, J., Radhakrishnan, A., Shlegeris, B., & Thomas, N. (2022, December 3). *Causal scrubbing: A method for rigorously testing interpretability hypotheses* [Online forum post]. AI Alignment Forum. https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing [Tier 4]

Chen, H., Janizek, J. D., Lundberg, S., & Lee, S.-I. (2020). *True to the model or true to the data?* [Preprint]. arXiv. https://arxiv.org/abs/2006.16234 [Tier 3]

Chen, H., Lundberg, S. M., & Lee, S.-I. (2022). Explaining a series of models by propagating Shapley values. *Nature Communications, 13*, Article 4512. https://doi.org/10.1038/s41467-022-31384-3

Covert, I., Lundberg, S., & Lee, S.-I. (2021). Explaining by removing: A unified framework for model explanation. *Journal of Machine Learning Research, 22*(209), 1–90. https://jmlr.org/papers/v22/20-1316.html

Dhamdhere, K., Sundararajan, M., & Yan, Q. (2019). How important is a neuron? In *International Conference on Learning Representations*. https://arxiv.org/abs/1805.12233

Dunefsky, J., Chlenski, P., & Nanda, N. (2024). Transcoders find interpretable LLM feature circuits. In *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2406.11944

Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *Transactions of the Association for Computational Linguistics, 9*, 160–175. https://doi.org/10.1162/tacl_a_00359

Engels, J., Riggs, L., & Tegmark, M. (2025). Decomposing the dark matter of sparse autoencoders. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2410.14670

Espinosa Zarlenga, M., Barbiero, P., Shams, Z., Kazhdan, D., Bhatt, U., Weller, A., & Jamnik, M. (2023). Towards robust metrics for concept representation evaluation. In *Proceedings of the AAAI Conference on Artificial Intelligence, 37*. https://arxiv.org/abs/2301.10367

Farnik, L., Lawson, T., Houghton, C., & Aitchison, L. (2025). Jacobian sparse autoencoders: Sparsify computations, not just activations. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 16178–16236). https://arxiv.org/abs/2502.18147

Ferrando, J., & Voita, E. (2024). Information flow routes: Automatically interpreting language models at scale. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 17432–17445). Association for Computational Linguistics. https://arxiv.org/abs/2403.00824

Frye, C., de Mijolla, D., Begley, T., Cowton, L., Stanley, M., & Feige, I. (2021). Shapley explainability on the data manifold. In *International Conference on Learning Representations*. https://arxiv.org/abs/2006.01272

Gat, Y., Calderon, N., Feder, A., Chapanin, A., Sharma, A., & Reichart, R. (2024). Faithful explanations of black-box NLP models using LLM-generated counterfactuals. In *International Conference on Learning Representations*. https://arxiv.org/abs/2310.00603

Geiger, A., Ibeling, D., Zur, A., Chaudhary, M., Chauhan, S., Huang, J., Arora, A., Wu, Z., Goodman, N., Potts, C., & Icard, T. (2025). Causal abstraction: A theoretical foundation for mechanistic interpretability. *Journal of Machine Learning Research, 26*(83), 1–64. https://jmlr.org/papers/v26/23-0058.html

Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal abstractions of neural networks. In *Advances in Neural Information Processing Systems 34*. https://arxiv.org/abs/2106.02997

Geiger, A., Richardson, K., & Potts, C. (2020). Neural natural language inference models partially embed theories of lexical entailment and negation. In *Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP*. Association for Computational Linguistics. https://arxiv.org/abs/2004.14623

Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024). Finding alignments between interpretable causal variables and distributed neural representations. In *Proceedings of the Third Conference on Causal Learning and Reasoning* (PMLR 236, pp. 160–187). https://proceedings.mlr.press/v236/geiger24a.html

Goldowsky-Dill, N., MacLeod, C., Sato, L., & Arora, A. (2023). *Localizing model behavior with path patching* (arXiv:2304.05969) [Preprint]. arXiv. https://arxiv.org/abs/2304.05969 [Tier 2]

Hammersborg, P., & Strümke, I. (2024). Information based explanation methods for deep learning agents—with applications on large open-source chess models. *Scientific Reports, 14*, 20174. https://doi.org/10.1038/s41598-024-70701-2

Havasi, M., Parbhoo, S., & Doshi-Velez, F. (2022). Addressing leakage in concept bottleneck models. In *Advances in Neural Information Processing Systems 35*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html

He, Z., Sinha, S., Xiong, G., & Zhang, A. (2025). GCAV: A global concept activation vector framework for cross-layer consistency in interpretability. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*. https://arxiv.org/abs/2508.21197

Heimersheim, S., & Nanda, N. (2024). *How to use and interpret activation patching* (arXiv:2404.15255) [Preprint]. arXiv. https://arxiv.org/abs/2404.15255 [Tier 3]

Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 2733–2743). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1275

Huang, J., Wu, Z., Potts, C., Geva, M., & Geiger, A. (2024). RAVEL: Evaluating interpretability methods on disentangling language model representations. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 8669–8687). https://doi.org/10.18653/v1/2024.acl-long.470

Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance quantification in explainable AI: A causal problem. In *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics* (PMLR 108, pp. 2907–2916). https://proceedings.mlr.press/v108/janzing20a.html

Jenner, E., Kapur, S., Georgiev, V., Allen, C., Emmons, S., & Russell, S. (2024). Evidence of learned look-ahead in a chess-playing neural network. In *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2406.00877

Kantamneni, S., Engels, J., Rajamanoharan, S., Tegmark, M., & Nanda, N. (2025). Are sparse autoencoders useful? A case study in sparse probing. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 29018–29049). https://arxiv.org/abs/2502.16681

Karvonen, A., Wright, B., Rager, C., Angell, R., Brinkmann, J., Smith, L., Mayrink Verdun, C., Bau, D., & Marks, S. (2024). Measuring progress in dictionary learning for language model interpretability with board game models. In *Advances in Neural Information Processing Systems 37*. https://doi.org/10.52202/079017-2644

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In *Proceedings of the 35th International Conference on Machine Learning* (PMLR 80). https://arxiv.org/abs/1711.11279

Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). Concept bottleneck models. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 5338–5348). https://proceedings.mlr.press/v119/koh20a.html

Kramár, J., Lieberum, T., Shah, R., & Nanda, N. (2024). *AtP\*: An efficient and scalable method for localizing LLM behaviour to components* (arXiv:2403.00745) [Preprint]. arXiv. https://arxiv.org/abs/2403.00745 [Tier 2]

Kulkarni, A., Weng, T.-W., Narayanaswamy, V., Liu, S., Sakla, W. A., & Thopalli, K. (2026). Interpretable and steerable concept bottleneck sparse autoencoders. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://arxiv.org/abs/2512.10805

Kumar, A., Tan, C., & Sharma, A. (2022). Probing classifiers are unreliable for concept removal and detection. In *Advances in Neural Information Processing Systems 35* (pp. 17994–18008). https://arxiv.org/abs/2207.04153

Kumar, I. E., Venkatasubramanian, S., Scheidegger, C., & Friedler, S. (2020). Problems with Shapley-value-based explanations as feature importance measures. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 5491–5500). https://proceedings.mlr.press/v119/kumar20e.html

Lawson, T., Farnik, L., Houghton, C., & Aitchison, L. (2025). Residual stream analysis with multi-layer SAEs. In *International Conference on Learning Representations*. https://arxiv.org/abs/2409.04185

Leask, P., Bussmann, B., Pearce, M., Bloom, J., Tigges, C., Al Moubayed, N., Sharkey, L., & Nanda, N. (2025). Sparse autoencoders do not find canonical units of analysis. In *International Conference on Learning Representations*. https://arxiv.org/abs/2502.04878

Lin, R., Jin, Z., Zhou, G., Ge, X., Shu, W., Wu, J., Wang, J., He, Z., Zhang, J., & Qiu, X. (2026). *Tracing the thought of a grandmaster-level chess-playing transformer* (arXiv:2604.10158) [Preprint]. arXiv. https://arxiv.org/abs/2604.10158 [Tier 3]

Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., & Olah, C. (2024, October 25). Sparse crosscoders for cross-layer features and model diffing. *Transformer Circuits Thread*. https://transformer-circuits.pub/2024/crosscoders/index.html [Tier 2]

Lomasov, S., Goldfeder, J., Erol, M. H., So, M., Yan, Y., Howard, A., Kutz, N., & Shwartz-Ziv, R. (2025). Exploring human-AI conceptual alignment through the prism of chess. In *Advances in Neural Information Processing Systems 38* (Creative AI Track). https://arxiv.org/abs/2510.26025

Lundstrom, D. D., Huang, T., & Razaviyayn, M. (2022). A rigorous study of integrated gradients method and extensions to internal neuron attributions. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 14485–14508). https://proceedings.mlr.press/v162/lundstrom22a.html

Mahinpei, A., Clark, J., Lage, I., Doshi-Velez, F., & Pan, W. (2021). *Promises and pitfalls of black-box concept learning models* [Workshop paper]. ICML 2021 Workshop on Theoretic Foundation, Criticism, and Application Trend of Explainable AI. https://arxiv.org/abs/2106.13314 [Tier 3]

Makelov, A., Lange, G., Geiger, A., & Nanda, N. (2024). Is this the subspace you are looking for? An interpretability illusion for subspace activation patching. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Ebt7JgMHv1

Makelov, A., Lange, G., & Nanda, N. (2025). Towards principled evaluations of sparse autoencoders for interpretability and control. In *International Conference on Learning Representations*. https://arxiv.org/abs/2405.08366

Margeloiu, A., Ashman, M., Bhatt, U., Chen, Y., Jamnik, M., & Weller, A. (2021). *Do concept bottleneck models learn as intended?* [Workshop paper]. ICLR 2021 Workshop on Responsible AI. https://arxiv.org/abs/2105.04289 [Tier 3]

Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2025). Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In *International Conference on Learning Representations*. https://arxiv.org/abs/2403.19647

McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Wattenberg, M., Hassabis, D., Kim, B., Paquet, U., & Kramnik, V. (2022). Acquisition of chess knowledge in AlphaZero. *Proceedings of the National Academy of Sciences, 119*(47), e2206625119. https://doi.org/10.1073/pnas.2206625119

Miller, J., Chughtai, B., & Saunders, W. (2024). Transformer circuit faithfulness metrics are not robust. In *Conference on Language Modeling*. https://arxiv.org/abs/2407.08734

Mueller, A., Geiger, A., Wiegreffe, S., Arad, D., Arcuschin, I., Belfki, A., Chan, Y. S., Fiotto-Kaufman, J., Haklay, T., Hanna, M., Huang, J., Gupta, R., Nikankin, Y., Orgad, H., Prakash, N., Reusch, A., Sankaranarayanan, A., Shao, S., Stolfo, A., . . . Belinkov, Y. (2025). MIB: A mechanistic interpretability benchmark. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 45069–45108). https://proceedings.mlr.press/v267/mueller25a.html

Nanda, N., Lee, A., & Wattenberg, M. (2023). Emergent linear representations in world models of self-supervised sequence models. In *Proceedings of the 6th BlackboxNLP Workshop* (pp. 16–30). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.blackboxnlp-1.2

Nicolson, A., Schut, L., Noble, J. A., & Gal, Y. (2025). Explaining explainability: Recommendations for effective use of concept activation vectors. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2404.03713

Pálsson, A., & Björnsson, Y. (2024). Empirical evaluation of concept probing for game-playing agents. In *ECAI 2024* (Frontiers in Artificial Intelligence and Applications, Vol. 392, pp. 874–881). IOS Press. https://doi.org/10.3233/FAIA240574

Parisini, E., Chakraborti, T., Harbron, C., MacArthur, B. D., & Banerji, C. R. S. (2026). Leakage and interpretability in concept-based models. *Journal of Machine Learning Research, 27*(174), 1–40. https://www.jmlr.org/beta/papers/v27/25-1121.html

Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR 235). https://arxiv.org/abs/2311.03658

Paulo, G., Shabalin, S., & Belrose, N. (2025). *Transcoders beat sparse autoencoders for interpretability* (arXiv:2501.18823) [Preprint]. arXiv. https://arxiv.org/abs/2501.18823 [Tier 3]

Poupart, Y. (2024). *Contrastive sparse autoencoders for interpreting planning of chess-playing agents* (arXiv:2406.04028) [Preprint]. arXiv. https://arxiv.org/abs/2406.04028 [Tier 3]

Ramaswamy, V. V., Kim, S. S. Y., Fong, R., & Russakovsky, O. (2023). Overlooked factors in concept-based explanations: Dataset choice, concept learnability, and human capability. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://arxiv.org/abs/2207.09615

Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 7237–7256). https://doi.org/10.18653/v1/2020.acl-main.647

Ravfogel, S., Twiton, M., Goldberg, Y., & Cotterell, R. (2022). Linear adversarial concept erasure. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162). https://proceedings.mlr.press/v162/ravfogel22a.html

Ravichander, A., Belinkov, Y., & Hovy, E. (2021). Probing the probing paradigm: Does probing accuracy entail task relevance? In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics* (pp. 3363–3377). https://doi.org/10.18653/v1/2021.eacl-main.295

Rubenstein, P. K., Weichwald, S., Bongers, S., Mooij, J. M., Janzing, D., Grosse-Wentrup, M., & Schölkopf, B. (2017). Causal consistency of structural equation models. In *Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence*. https://arxiv.org/abs/1707.00819

Sandmann, E., Lapuschkin, S., & Samek, W. (2025). *The algorithm is not the behavior: Learned priors override look-ahead in a chess-playing neural network* (arXiv:2508.21380) [Preprint]. arXiv. https://arxiv.org/abs/2508.21380 [Tier 3]

Schut, L., Tomašev, N., McGrath, T., Hassabis, D., Paquet, U., & Kim, B. (2025). Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero. *Proceedings of the National Academy of Sciences, 122*(13), e2406675122. https://doi.org/10.1073/pnas.2406675122

Shang, C., Zhou, S., Zhang, H., Ni, X., Yang, Y., & Wang, Y. (2024). Incremental residual concept bottleneck models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 11030–11040). https://doi.org/10.1109/CVPR52733.2024.01049

Shin, S., Jo, Y., Ahn, S., & Lee, N. (2023). A closer look at the intervention procedure of concept bottleneck models. In *Proceedings of the 40th International Conference on Machine Learning*. https://arxiv.org/abs/2302.14260

Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning important features through propagating activation differences. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 3145–3153). https://proceedings.mlr.press/v70/shrikumar17a.html

Sundararajan, M., & Najmi, A. (2020). The many Shapley values for model explanation. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 9269–9278). https://proceedings.mlr.press/v119/sundararajan20b.html

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 3319–3328). https://proceedings.mlr.press/v70/sundararajan17a.html

Sutter, D., Minder, J., Hofmann, T., & Pimentel, T. (2025). The non-linear representation dilemma: Is causal abstraction enough for mechanistic interpretability? In *Advances in Neural Information Processing Systems 38*. https://arxiv.org/abs/2507.08802

Tomlin, N., He, A., & Klein, D. (2022). Understanding game-playing agents with natural language annotations. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 797–807). https://doi.org/10.18653/v1/2022.acl-short.90

Tucker, M., Qian, P., & Levy, R. (2021). What if this modified that? Syntactic interventions with counterfactual embeddings. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021* (pp. 862–875). https://doi.org/10.18653/v1/2021.findings-acl.76

Wu, Z., Arora, A., Geiger, A., Wang, Z., Huang, J., Jurafsky, D., Manning, C. D., & Potts, C. (2025). AxBench: Steering LLMs? Even simple baselines outperform sparse autoencoders. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 67035–67080). https://proceedings.mlr.press/v267/wu25a.html

Wu, Z., Geiger, A., Huang, J., Arora, A., Icard, T., Potts, C., & Goodman, N. (2024). *A reply to Makelov et al. (2023)'s "interpretability illusion" arguments* [Preprint]. arXiv. https://arxiv.org/abs/2401.12631 [Tier 3]

Wu, Z., Geiger, A., Icard, T., Potts, C., & Goodman, N. (2023). Interpretability at scale: Identifying causal mechanisms in Alpaca. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2305.08809

Yeh, C.-K., Kim, B., Arik, S. O., Li, C.-L., Pfister, T., & Ravikumar, P. (2020). On completeness-aware concept-based explanations in deep neural networks. In *Advances in Neural Information Processing Systems 33*. https://arxiv.org/abs/1910.07969

Yuksekgonul, M., Wang, M., & Zou, J. (2023). Post-hoc concept bottleneck models. In *International Conference on Learning Representations*. https://arxiv.org/abs/2205.15480

Zhang, F., & Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. In *International Conference on Learning Representations*. https://arxiv.org/abs/2309.16042

Zhou, B., Sun, Y., Bau, D., & Torralba, A. (2018). Interpretable basis decomposition for visual explanation. In *Computer Vision – ECCV 2018* (pp. 122–138). Springer. https://doi.org/10.1007/978-3-030-01237-3_8
