<!-- Phase 4 (composition) of the deep-research pipeline. Inputs: 00_context.md, 01_scoping.md,
02_biblio_T1..T8.md, 03_synthesis.md (revised after DA checkpoint 2), 04_da_checkpoint2.md.
Citations are APA 7 author-year; every cited work is an entry in a 02_biblio_Tk.md file. The skill's hidden
ref/anchor markers and claim-intent manifest are not emitted: the bibliographies carry no slugs. -->

# Attributing a Chess Transformer's Outputs to a Fixed Human Feature Vocabulary: Decoders, Probes, and What the Vocabulary Misses

xai-chess project, Phase 4 research report, 9 October 2026

## Abstract

**Background.** A per-square vocabulary of 30,837 rule-computed chess features has been mapped to the token activations of the Leela Chess Zero (lc0) transformer `t1-256x10-distilled-swa-2432500` in both directions: ridge decoders (features to token) and linear probes (token to features), at the embedding and at every encoder block. The decoders explain the embedding almost completely (fraction of variance unexplained, FVU, 0.002) and the last block poorly (0.68). **Purpose.** To identify which methods can turn these maps into attributions of lc0's value and policy outputs, including methods that combine several maps or layers, and to report the part of the output the vocabulary does not name. **Method.** A structured methods survey of about 122 verified sources in eight themes, organised by a nine-axis taxonomy, two devil's-advocate reviews, and a check of the embedding against the ONNX graph. **Findings.** Twelve methods fall into six approach families: a decoder split read with five removal operators, erasure, probe gradients, causal alignment, a supervised replacement graph, and residual analysis. The embedding is a per-square, mish-nonlinear function of that square's own input planes, so embedding-level attribution is input saliency in feature terms. The front/back choice becomes one curve across layers: the share of a real output change carried by named features at each tap. **Implications.** For complete pairwise methods the usual faithfulness error equals the residual share, so it is reported as coverage and never used to rank methods. Nothing beyond the existing decoder and probe fits has been run. Every method outcome stated here is a hypothesis for a pre-specified protocol.

*Keywords:* feature attribution, concept-based explanation, causal abstraction, residual coverage, chess, Leela Chess Zero, mechanistic interpretability

## 1. Introduction

A chess engine's evaluation can be explained in chess terms only if some map connects its internal state to those terms. The xai-chess project has such a map. A Rust extractor (`features/square/`) emits, for every square of a position, a binary vector of 30,837 bits: occupant, pawn structure, x-rays, threats, routes, piece roles, and mate and material lines computed with an engine. About 36.5 bits are on per square. On 20,000 Lichess puzzle positions, split 80/20 by position, two families of maps have been fitted against lc0's square tokens ($d=256$) at the embedding and at each of the ten post-LN encoder blocks. Probes read the features from the token. Decoders predict the token from the features plus a one-hot square identity.

The results so far pose a concrete problem. Decoders reconstruct the embedding almost perfectly and the deep layers poorly. Attribution of the output to features is easiest to define where the decoder fits and where the path to the output is short, and these two places are at opposite ends of the network. The project's working note states the dilemma directly: start at the back and the reconstruction is "mostly square identity", so attribution must fight through "the deep jungle of attention"; start at the front and the reconstruction is worse deeper but attribution is clearer; perhaps a hybrid combines both.

This report answers one primary question. *Which families of methods can attribute lc0's value and policy outputs to a fixed per-square feature vocabulary through learned feature–activation maps, and how do they trade off faithfulness under intervention, identifiability, residual coverage, and compute on this system?* The user asked for several approaches rather than one winner, with weight on methods that combine maps across layers.

Three findings organise the answer. First, square identity explains 84% of embedding variance and 10% of layer-9 variance, so "mostly square identity" describes the front, not the back. Second, the embedding is a local, elementwise-nonlinear function of each square's own input planes, which makes embedding-level attribution input saliency with feature names. Third, five of the twelve methods are removal operators on one decomposition of the token into square mean, named part and residual. Read at every tap, that decomposition gives one curve whose endpoints are the user's front and back, and no ad hoc hybrid is needed.

The report is scoped honestly. It surveys and proposes, and tests nothing. The only empirical numbers are the existing decoder, probe, and value-head fits listed in Table 1. Statements about how a method will behave on lc0 are hypotheses.

Section 2 describes the survey method. Section 3 fixes the system and notation. Section 4 summarises what the literature establishes. Section 5 re-derives the front/back question. Section 6 presents the twelve methods with their mathematics, and Section 7 compares them. Section 8 treats the residual as a coverage measure. Section 9 gives the evaluation protocol. Sections 10 and 11 discuss and state limitations.

## 2. Method of the Survey

The design is a structured methods survey, not a PRISMA systematic review. The literature is mostly arXiv and lab reports with unstable terminology, so counting studies exhaustively would give false precision. The survey ends in a taxonomy with a fixed axis set, a per-method assessment of fit to lc0, and a pre-specified experimental protocol that this report does not execute.

**Themes and sources.** Eight themes were searched in parallel: attribution operators and their axioms, concept attribution through learned concept maps, probing and erasure, concept bottlenecks and completeness, sparse dictionaries across layers, circuit discovery with error nodes, causal abstraction, and interpretability of game-playing networks. Sources were arXiv, OpenReview, major ML, NLP and vision proceedings, the Transformer Circuits Thread, and the project's existing bibliography. Forum posts were admitted only as gray literature for methods without another source.

**Procedure.** For each theme: keyword search, seed papers, one round of backward and forward citation chasing per seed, and a stop when two consecutive chases added nothing in scope. The window was 2016 to 2026, with foundational axiomatic work of any date.

**Inclusion.** A work was included if it defines or evaluates a method that maps activations to or from a concept vocabulary or dictionary and attributes outputs through it, defines a faithfulness, completeness, or identifiability criterion for such attributions, or applies such methods to a game-playing network. Pure input saliency without concepts was admitted only as a baseline.

**Verification and tiers.** Every entry's title, authors, year, venue and identifier were checked against the arXiv record and the publisher or proceedings page. One wrong arXiv identifier was corrected, and unverifiable venues were downgraded to preprint. Sources were tiered: 1 peer-reviewed, 2 lab technical reports (Transformer Circuits Thread), 3 preprints, 4 gray literature. About 122 entries survived screening.

**Analysis.** Each method received a row on nine axes: vocabulary entry point, map direction, layer span, map class, attribution operator, residual handling, identifiability, faithfulness test, and cost on this system. Two devil's-advocate reviews challenged the scoping and the synthesis. The second found that the draft metric rewarded hiding the residual and that the embedding had been wrongly assumed affine; the ONNX graph settled the second point. Both corrections are built in.

## 3. System, Data, and Notation

### 3.1 The model and the maps

The network is lc0 `t1-256x10-distilled-swa-2432500`: 64 square tokens of width 256, ten post-LayerNorm encoder blocks with eight heads and smolgen attention biases, and policy and value heads. It runs as ONNX on CPU at about 180 positions per second. Activations are tapped at the embedding (`/attn_body/rehape`), at both LayerNorm outputs of each block, inside the feed-forward networks, and in the value head. ONNX on CPU provides no backward pass, so any gradient method needs a PyTorch port.

Table 1 lists every empirical number this report relies on.

**Table 1**
*Existing results on held-out positions (from the project; nothing else has been run)*

| Result | Value |
|---|---|
| Linear probe, token at layer $L$ to its own features | most families near-perfectly readable at the embedding; `pawn_cover`, strong squares, mate, passed pawn gain 0.04–0.14 by layer 5 |
| Ridge decoder FVU, square one-hot only | $e$ 0.164 · L0 0.575 · L1 0.818 · L2 0.828 · L5 0.841 · L9 0.897 |
| Ridge decoder FVU, features + square | $e$ 0.002 · L0 0.183 · L1 0.198 · L2 0.195 · L5 0.392 · L9 0.680 |
| Variance added by features | $e$ 0.16 · L0 0.39 · L1 0.62 · L2 0.63 · L5 0.45 · L9 0.22 |
| MLP decoder | similar trend |
| Board-level decoder, all 64 squares' features to value-head activations | test FVU 0.52–0.62 at 151k positions, still falling with data |
| Value head applied to decoded layer-9 tokens | weak |
| MLP probe for "pin is relevant" | peak AUROC 0.945 at block 8, vs 0.839 for Stockfish NNUE |

*Note.* $e$ = embedding; L$i$ = encoder block $i$. Bits with fewer than 20 positive training examples were dropped.

The project also has exact causal operators in a patched Stockfish: MaskPinner removes the rule-level effect of a pin and LineBlocker places a ray-selective phantom blocker on a line. They supply interventional ground truth about chess, which the protocol in Section 9 uses separately from ground truth about lc0.

### 3.2 Notation and the canonical split

Let $x$ be a board, $x'$ a real edit of it, and $x_0$ a baseline board drawn from a pool $\mathcal B$ of real positions. The extractor gives $f=\phi(x)\in\{0,1\}^{64\times K}$, with $f_s$ the bits of square $s$ and $\tilde f_s=f_s-\bar f_s$ their per-square centring. Bits are grouped into about 30 families $\mathcal G=\{G_1,\dots,G_m\}$. The token at tap $L\in\{e,0,\dots,9\}$ is $h_L\in\mathbb R^{64\times256}$, the suffix network from tap $L$ to the output is $g_L$, and the target $y$ is either the value (win-minus-loss logit) or the policy logit of one move. The decoder is

$$D_L(f_s,s)=q_{L,s}+W_L\tilde f_s,$$

with $q_{L,s}$ the per-square mean. Writing $n_{L,s}=W_L\tilde f_s$ for the named part, $n^G_{L,s}=W^G_L\tilde f^G_s$ for family $G$'s share of it, and

$$\varepsilon_{L,s}=h_{L,s}-D_L(f_s,s)$$

for the residual, every token splits as $h_L=q_L+\sum_G n^G_L+\varepsilon_L$. The residual $\varepsilon_L$ is the error node: what the vocabulary misses at layer $L$ on square $s$.

Centring makes the split canonical given $W_L$, and it has one useful consequence. Because $q_{L,s}$ depends only on $s$, it is identical for $x$ and $x'$. In every two-board comparison the square-identity term cancels exactly. Square identity can absorb credit only in methods that evaluate a gradient at a single point.

Write $\Delta^{\rm true}=y(x')-y(x)$ for lc0's own response to a real edit, and $P_L:h_{L,s}\mapsto\hat f_s$ for a probe.

## 4. What the Literature Establishes

Eight themes emerged. They differ in strength, and two of them argue against the user's starting intuition.

### 4.1 The removal rule decides what an attribution means

Gradient, Integrated Gradients (IG), DeepLIFT, SHAP and occlusion differ less than their removal rules do. Covert et al. (2021) classify 26 explanation methods by how they remove features. Sundararajan and Najmi (2020) show that conditional, interventional and baseline Shapley values give different answers under the same axioms. Janzing et al. (2020) argue for interventional removal because the object explained is a function, while Frye et al. (2021) show that off-manifold removal produces misleading explanations. Chen et al. (2020) reconcile the two: interventional removal is true to the model, conditional removal is true to the data. I. E. Kumar et al. (2020) show that both Shapley variants fail as importance measures, through off-manifold evaluation on one side and credit to correlated features on the other.

Here the vocabulary admits three removal rules, each with its own explanandum. Flipping bits of $f$ and re-decoding explains lc0 composed with the decoder, and the flipped vector is usually one no board produces. Substituting the decoded named part from a real edited board explains lc0 at layer $L$, assuming the decoder is right. Its endpoints are real, its mixture is not. A real board edit explains lc0 itself and is on-manifold, but changes many families at once. Aas et al. (2021) add the remedy for dependent features: sum Shapley values over groups. Lundstrom et al. (2022) provide distributional baselines and internal-neuron IG, which licenses attributing at a hidden layer. Bilodeau et al. (2024) prove that complete, linear attribution methods can be uninformative about model behaviour unless tied to a concrete query. In this project the concrete query is a board edit.

### 4.2 Readable is not used

Five independent lines agree. In NLP probing, irrelevant properties stay probe-readable (Ravichander et al., 2021) and probe accuracy must be read against control tasks (Hewitt & Liang, 2019; Belinkov, 2022). Erasure finds high probe accuracy with small behavioural effect (Elazar et al., 2021). Interchange interventions show readable variables that are not causally used (Geiger et al., 2020). Concept bottlenecks show the same gap (Margeloiu et al., 2021). On a chess agent, Pálsson and Björnsson (2024) find linear-probe accuracy an unreliable proxy for concept importance and recommend nonlinear probes with erasure. Jenner et al. (2024) validate an lc0 probe by its causal effect. Benchmarks with counterfactual ground truth, CEBaB (Abraham et al., 2022), RAVEL (Huang et al., 2024) and MIB (Mueller et al., 2025), find that many correlational methods fail.

The project's finding that most families are readable at the embedding therefore says how much the extractor's bits depend on occupancy, not whether lc0 uses them.

### 4.3 The residual is structured and belongs in the report

Explicit residual terms appear in four traditions. Interpretable basis decomposition leaves 65.9% of a ResNet-50 prediction unexplained (Zhou et al., 2018). Post-hoc concept bottlenecks add a residual predictor (Yuksekgonul et al., 2023), and Havasi et al. (2022) add a side channel. Feature circuits keep one error node per site (Marks et al., 2025), and replacement models keep error nodes per layer and position (Ameisen et al., 2025, Tier 2). Engels et al. (2025) find that about half of a sparse autoencoder's (SAE) error vector is linearly predictable from the input activation and that maps from earlier layers reduce it most. Yeh et al. (2020) define completeness as the fraction of the head's accuracy that concept reconstructions retain.

The disagreement concerns naming. Residual concept bottlenecks (Shang et al., 2024) and concept-bottleneck SAEs (Kulkarni et al., 2026) assume residual directions can be matched to candidate concepts. Leask et al. (2025) show that unsupervised dictionaries have no canonical size, so "the residual contains concept X" depends on the dictionary. Schut et al. (2025) are the only game-network attempt to name what a human vocabulary misses, and they validate behaviourally rather than by intervention. Havasi et al. (2022) supply a distinction this project needs: residual from missing vocabulary versus residual from a map too weak to use the vocabulary.

### 4.4 With a fixed vocabulary, supervised maps are the stronger default

Makelov et al. (2025) use supervised attribute dictionaries, in which an activation is a sum of per-attribute-value vectors (the structure of $D_L$), as the yardstick against which SAEs are judged by sufficiency, necessity and edit control. Kantamneni et al. (2025) and AxBench (Wu et al., 2025) find supervised directions match or beat SAE latents, and MIB finds distributed alignment search (DAS) strongest and SAE features no better than neurons (Mueller et al., 2025). Karvonen et al. (2024) score dictionaries in chess and Othello language models against rule-computed board properties. The one counter-signal is a matter of degree: Makelov et al. (2025) report that newer SAEs approach the supervised dictionary. SAEs belong on the residual, not in place of $D_L$.

### 4.5 Cross-layer maps exist only unsupervised and mostly unreviewed

Transcoders map a block's input to its output (Dunefsky et al., 2024), with a skip variant (Paulo et al., 2025, Tier 3). Crosscoders share one latent set across layers (Lindsey et al., 2024, Tier 2). Cross-layer transcoders (CLTs) write from one layer to all later ones and keep error nodes (Ameisen et al., 2025, Tier 2). Multi-layer SAEs (Lawson et al., 2025) and Jacobian SAEs (Farnik et al., 2025) make layer stability and block Jacobians measurable. GCAV (He et al., 2025) is the only concept-map method found that fuses several layers. On lc0, Lin et al. (2026, Tier 3) replace every MLP and attention module of a larger lc0 network with sparse layers, reach explained variance above 0.6 per module, label features by hand and check them against rule-based detectors. No source fits a supervised cross-layer map. These designs also assume an additive residual stream. lc0's taps are post-LN, so a layer's token is a normalised, not summed, function of earlier writes. No source addresses this.

### 4.6 Identifiability limits every map

Sutter et al. (2025) prove that with unrestricted alignment maps any network can be made to abstract any algorithm, and report near-perfect interchange intervention accuracy (IIA) on random networks. Hewitt and Liang (2019) make the same point for probes. Beckers and Halpern (2019) give conditions for constructive abstraction that this vocabulary does not meet, since its bits are functions of overlapping sets of squares, so per-bit attribution cannot be unique. Park et al. (2024) show that probe directions and decoder columns coincide only under a causal inner product, so chaining a probe with a decoder needs whitening. Concept activation vectors are inconsistent across layers and entangled (Nicolson et al., 2025). Leakage has measurable diagnostics (Mahinpei et al., 2021; Espinosa Zarlenga et al., 2023; Parisini et al., 2026; Huang et al., 2024). Ramaswamy et al. (2023) find human usefulness capped near 32 concepts. The convergent prescription is to attribute to about 30 families, keep linear maps primary, report MLP maps only beside a random-initialisation control, and report stability across regularisation.

### 4.7 Attention is where gradient attributions break

Gradient×Input violates conservation through attention and LayerNorm, and Ali et al. (2022) restore it by detaching both. AttnLRP gives softmax rules (Achtibat et al., 2024), and concept relevance propagation conditions relevance on a latent subspace (Achtibat et al., 2023). AtP* identifies the false negatives of linearised patching, attention saturation and cancellation between direct and indirect paths (Kramár et al., 2024). Ameisen et al. (2025) avoid the problem by freezing attention. On lc0 that is costly: Jenner et al. (2024) locate look-ahead in specific attention heads. Ferrando and Voita (2024) give a forward-only route decomposition that would run under ONNX. No rule set covers smolgen.

### 4.8 Depth profiles of human concepts in game networks

McGrath et al. (2022) and Hammersborg and Strümke (2024) track concepts per layer in AlphaZero and lc0. Lomasov et al. (2025) report human-concept probe accuracy falling from 85% in early layers to 50–65% late in a different, 270M-parameter chess transformer, on six concepts. That is probe decay on another model, not this project's decoder-FVU decay. The project's own probes point the other way: families stay readable with depth and some gain by layer 5. On lc0 the features do not vanish; they shrink as a share of token variance. Tomlin et al. (2022) find Go terms best read late, possibly because their labels are commentary mentions. Sandmann et al. (2025, Tier 3) show lc0's late layers overriding correct mid-layer solutions, so what a layer encodes need not be what the output uses. Per-layer readings must be chained to $y$.

## 5. The Front/Back Question, Re-derived

### 5.1 Two normalisations of one table

Table 1's "features add" row is a share of total token variance, and it is capped by the square-only FVU. Dividing by that cap gives the share of non-square variance the vocabulary explains, a partial $R^2$. Table 2 shows that the two quantities peak in different places.

**Table 2**
*Variance explained by the vocabulary, two normalisations*

| Tap | $e$ | L0 | L1 | L2 | L5 | L9 |
|---|---|---|---|---|---|---|
| Features add (share of total variance) | 0.16 | 0.39 | 0.62 | 0.63 | 0.45 | 0.22 |
| Partial $R^2$ (add ÷ square-only FVU) | 0.99 | 0.68 | 0.76 | 0.76 | 0.53 | 0.24 |

*Note.* Computed from the decoder FVUs in Table 1.

The rise of features-add from $e$ to L2 mostly reflects the shrinking square-identity share. Of the variance square identity does not explain, the vocabulary explains most at $e$, shows a local bump at L1–L2 relative to L0, and declines monotonically after. Neither row alone picks a best single layer.

### 5.2 The embedding is local and nonlinear

The exported ONNX graph gives the embedding tap per square as

$$h_{e,s}=g_s\odot\mathrm{mish}\big(W[\text{planes}_s;\ \text{posenc}_s]+b\big)+a_s,$$

where $\text{planes}_s$ are the square's 112 input-plane values (pieces over eight history frames plus broadcast castling and rule-50 planes), $\text{posenc}_s$ is a learned positional code, and $g_s,a_s$ are learned per-square gates. The map is elementwise-nonlinear and strictly local: no information from other squares reaches $h_{e,s}$.

Three consequences follow. The 0.002 FVU says the vocabulary contains each square's occupant. History, castling and rule-50 content fall into $\varepsilon_e$. A non-local family (x-ray, threat, route, mate line) can enter $h_{e,s}$ only through correlation with the square's own planes. And in any pairwise method at $e$, a board edit changes non-local bits on squares whose planes did not change. The decoder assigns $W^G\Delta\tilde f^G_s\neq0$ there, and $\varepsilon_e$ cancels it exactly. The result is spurious named credit with an equal and opposite residual, produced by construction. At $e$, attribution must either use a local vocabulary (occupant and square, plus history, castling and rule-50 bits if added) or zero decoded deltas on squares whose planes are unchanged. A random-initialised embedding is equally local, so a decoder would fit it almost as well, which shows directly that the embedding FVU says nothing about learning.

### 5.3 Coverage against path length

At $e$ a gradient crosses all ten blocks, at L9 only the head. Coverage is high at the front and the causal path long. At the back, coverage is lower and the path short. Square identity is the dominant term at the front, not the back.

### 5.4 Seven candidate causes of the decay

The fall in partial $R^2$ from L2 to L9 has at least seven candidate causes, and they must be separated before any part of $\varepsilon_L$ is called an unnamed concept:

- (a) *non-locality*: $D_L$ sees only $f_s$, while attention moves information between squares.
- (b) *map class*: a linear map is too weak (the MLP decoder's similar trend weakens this but does not exclude it).
- (c) *genuinely unnamed content*.
- (d) *dilution*: named content stays, as the probes show, while other content grows.
- (e) *post-LN renormalisation*: taps are LayerNorm outputs, a per-token rescaling that a linear map of $f_s$ cannot fit.
- (f) *output-specific content*, such as per-square policy move scores that a value target never uses.
- (g) *fit capacity*: thousands of kept bits at 16,000 training positions.

Causes (d) to (f) mean that FVU decay need not be decay in what explains $y$. Only methods that measure in output units can settle that. Method M6 below does so at every tap, and M12 splits the residual into causes. Together they replace the "hybrid of front and back" with one curve across layers whose endpoints are the user's front and back.

## 6. Methods

Twelve methods fall into six approach families. The families, not the count, carry the claim. Five methods (M1, M2, M6, M9, M10) read one decomposition, $h_L=q_L+\sum_Gn^G_L+\varepsilon_L$, with different removal operators. Erasure (M3, cascaded in M8), probe gradients (M4), causal alignment (M5), a supervised replacement graph (M7), and residual analysis (M12) are the other five families. M11 compares the rest. In the strict sense of "several decoders on several layers" composed into one object, only M7 and the decoder-read variant of M8 qualify; M6 applies independent decoders at every tap.

Each method below gives its definition, question answered, failure modes, controls, cost and residual. Controls C1 to C9 are shared (Table 3). Costs assume 180 full forward passes per second under ONNX, a suffix from tap $L$ costing about $(10-L)/10$ of a forward pass, and a backward pass in a PyTorch port costing two to three forward passes. These are estimates.

**Table 3**
*Shared controls*

| Code | Control | Basis |
|---|---|---|
| C1 | Random-initialised lc0: refit maps, rerun the method | Adebayo et al. (2018); Sutter et al. (2025) |
| C2 | Local vocabulary (occupant + square, plus history/castling/rule-50 bits if added); the only admissible vocabulary at $e$ | Section 5.2; Hewitt & Liang (2019) |
| C3 | Ridge-$\alpha$ sweep ($10^{-2}$ to $10^{2}$), decoder seeds, data resampling; report family-rank stability | Section 4.6 |
| C4 | Linear vs MLP map; MLP reported only next to C1 | Sutter et al. (2025) |
| C5 | Shuffled-label refit of $D_L$ or $P_L$ | Adebayo et al. (2018) |
| C6 | Random direction of equal rank or norm | Elazar et al. (2021) |
| C7 | With and without engine-derived families (`mate`, `material`, threat lines) | A. Kumar et al. (2022); Parisini et al. (2026) |
| C8 | Row-space vs nullspace split of a patch direction, as a diagnostic | Makelov et al. (2024); Wu et al. (2024) |
| C9 | Noising and denoising directions; fixed ablation type | Heimersheim & Nanda (2024); Miller et al. (2024) |

### 6.1 Family (i): the decoder split

**M1. Decoder gradient at the embedding.** The point form multiplies the output gradient by each component of the decomposition:

$$A^{(e)}_G(x)=\sum_s\big\langle\nabla_{h_{e,s}}y,\ n^G_{e,s}\big\rangle,\qquad A^{(e)}_\varepsilon=\sum_s\big\langle\nabla_{h_{e,s}}y,\ \varepsilon_{e,s}\big\rangle.$$

This is the user's $\nabla_E y\cdot\partial D/\partial f$ multiplied by $\tilde f$. The board-path form runs IG (Sundararajan et al., 2017) in activation space between two real boards. With $\gamma(\alpha)=h_e(x_0)+\alpha\,(h_e(x)-h_e(x_0))$ and the linearity of $D_e$,

$$\mathrm{IG}_G=\int_0^1\big\langle\nabla y(\gamma(\alpha)),\ W^G_e(\tilde f^G-\tilde f^G_0)\big\rangle\,d\alpha,\qquad \sum_G\mathrm{IG}_G+\mathrm{IG}_\varepsilon=y(x)-y(x_0).$$

The baseline is a real edit of $x$ or an average over real positions (Lundstrom et al., 2022). M1 answers which families carry the input-side sensitivity of $y$. By Section 5.2 that is input saliency on each square's planes, re-expressed in the vocabulary. The point form is first-order on binary features and therefore not complete; it is a sensitivity map, not an attribution with a residual. The split among occupancy-derived families is set by the ridge penalty, not by lc0. Gradients through ten post-LN blocks violate conservation (Ali et al., 2022). Controls: C2 decides, C3, and C1. Cost: one backward pass per position for the point form and about 32 for IG; minutes for 1,000 positions, after a PyTorch port. Residual: $\mathrm{IG}_\varepsilon$, which collects history and rule-50 content and the 0.2% misfit. M1 is the baseline that makes the locality argument concrete; occupancy should dominate it.

**M2. Decoded-substitution patching at layer $L$.** For a real edit $x\to x'$ and family $G$, patch the named part on the squares $S$ where $f^G$ changed:

$$h^{\rm patch}_{L,s}=h_{L,s}(x)+W^G_L\big(\tilde f^G_s(x')-\tilde f^G_s(x)\big),\ \ s\in S;\qquad \Delta^{(L)}_G=g_L(h^{\rm patch})-y(x).$$

Patch all changed families jointly to get $\Delta^{(L)}_{\rm named}$, and leave one family out at a time. The consistency gap

$$\gamma_L=\Delta^{\rm true}-\Delta^{(L)}_{\rm named}$$

checks whether the decoder edit commutes with the real one, in the sense of causal consistency (Rubenstein et al., 2017). The design follows supervised dictionary edits (Makelov et al., 2025), test-time interventions in concept bottlenecks (Koh et al., 2020; Shin et al., 2023), and in-distribution corruption (Zhang & Nanda, 2024). M2 answers: if lc0's layer-$L$ representation of $G$ moved as the decoder says this edit moves it, how far would $y$ move, and how much of the true effect is left over? The decoded edit is off-distribution, can activate dormant pathways (Makelov et al., 2024), and single-family patches are not additive when families interact like OR gates (Heimersheim & Nanda, 2024); hence the joint and leave-one-out variants. Controls: C6, C8, C3, C1, C9. Cost: ONNX with the graph split at the tap, about 330 suffix passes per position for 30 families at 11 taps, so roughly 30 minutes for 1,000 edits. Residual: $\gamma_L$ per layer, plus the error-swap patch $\varepsilon_L(x)\to\varepsilon_L(x')$ that feeds M6.

**M6. Named/error split of the same effect at every layer.** For a pair $(x,x')$ and each tap, treat the named part $n_L$ and the residual $\varepsilon_L$ as two players ($q$ cancels). The exact two-player Shapley value needs four suffix evaluations, two of which are $y(x)$ and $y(x')$:

$$\phi^{(L)}_n=\tfrac12\Big[g_L(q+n'+\varepsilon)-g_L(q+n+\varepsilon)+g_L(q+n'+\varepsilon')-g_L(q+n+\varepsilon')\Big],\qquad \phi^{(L)}_\varepsilon=\Delta^{\rm true}-\phi^{(L)}_n.$$

Completeness is exact because $g_L(q+n'+\varepsilon')=y(x')$. The named share, aggregated over pairs so that it stays defined when $\Delta^{\rm true}\approx0$, and its difference between taps are

$$\rho_L=\frac{\sum_{\rm pairs}\phi^{(L)}_n}{\sum_{\rm pairs}|\Delta^{\rm true}|},\qquad \kappa_L=\rho_L-\rho_{L+1}.$$

This is baseline Shapley (Sundararajan & Najmi, 2020) with the error node as a player (Marks et al., 2025). M6 answers the user's question directly: at each layer, how much of an actual output change is carried by what the vocabulary names. Front and back are the two ends of $\rho_L$. The difference $\kappa_L$ is descriptive and not a flow through block $L$: the two decoders are fitted independently on different tokens, attention mixes squares and LayerNorm renormalises between taps, and $\kappa_L$ moves with decoder quality. Block-level structure is read only where $\kappa_L$ is stable under C3. Mixed coalitions such as $n'$ with $\varepsilon$ are off-manifold, and non-locality makes information moved between squares count as residual. $\rho_L$ can exceed 1 or turn negative when the players cancel, so both $\phi$ values are reported. Variants: conductance through the span of $W_L$ against its complement (Dhamdhere et al., 2019), and a tied decoder $W_L=W+\Delta_L$ with a penalty on $\Delta_L$, a supervised analogue of crosscoders (Lindsey et al., 2024; Lawson et al., 2025), to test whether a family's direction is stable across layers. Controls: C1, C2 at $e$, C3, C9, and a neighbour-augmented decoder $D_L(f_s,\sum_t\bar a^L_{st}f_t)$ that uses lc0's mean attention $\bar a^L$ as a locality control (a proposal). Cost: two new suffix passes per tap per pair, about 0.1 s; 10,000 pairs in about 20 minutes. M6 is itself the residual report: $\phi^{(L)}_\varepsilon$ and $\rho_L$.

**M9. Family Shapley with an error player.** Players are $\mathcal P=\{G_1,\dots,G_m,\varepsilon\}$. For coalition $S$ and baseline $x_0\sim\mathcal B$,

$$h^{(S)}_{L,s}=q_{L,s}+\sum_{G\in S}n^G_{L,s}(x)+\sum_{G\notin S}n^G_{L,s}(x_0)+\mathbb 1[\varepsilon\in S]\,\varepsilon_{L,s}(x)+\mathbb 1[\varepsilon\notin S]\,\varepsilon_{L,s}(x_0),$$

with value $v(S)=\mathbb E_{x_0}[g_L(h^{(S)})]$ and Shapley values $\varphi_p$ satisfying $\sum_p\varphi_p=y(x)-\mathbb E\,y(x_0)$. The baseline pool is matched on material and phase to limit off-manifold mixing. Grouping follows Aas et al. (2021), the value function follows baseline Shapley (Sundararajan & Najmi, 2020) with distributional baselines (Lundstrom et al., 2022), a global variant uses ConceptSHAP's completeness score (Yeh et al., 2020), and about 30 players respects Ramaswamy et al.'s (2023) cap. M9 splits one position's output among about 30 families and the vocabulary's miss at each layer, with $\varphi_\varepsilon$ on equal footing with named families. Mixed coalitions combine families from different boards, so the result is true to lc0 composed with $D_L$, not to chess (Chen et al., 2020), and leaves the data manifold (I. E. Kumar et al., 2020). Grouping removes non-identifiability within a family only. Families are collinear with each other (occupant against every occupancy-derived family), so the split of $W_L$ into $W^G_L$ still depends on the ridge penalty, and C3 must report family-rank stability. With $x_0=x'$, unchanged families receive zero by the dummy axiom, so localisation is trivial in the pairwise form. Making players $(G,L)$ asks at which depth $G$ is used (a proposal). Controls: C1, C3, C7, C2, and sign and rank agreement with M2 on edited families. Cost: permutation sampling with $(m+1)\times M\times|\mathcal B|$ suffix passes, for example $31\times20\times8\approx5{,}000$, about 30 s per position per layer; 200 positions at three layers in about five hours. Residual: $\varphi_\varepsilon$ and its share $\varphi_\varepsilon/\sum|\varphi|$.

**M10. Conservative relevance chain through $D_L$.** One modified backward pass with AttnLRP rules (Achtibat et al., 2024), attention and LayerNorm detached (Ali et al., 2022), gives relevance $R_{L,s}\in\mathbb R^{256}$ at every tap. Each dimension's relevance is split over components in proportion to their baseline-differenced contributions, with relevance propagated against the same baseline using DeepLIFT rescale rules (Shrikumar et al., 2017) as chained in G-DeepSHAP (Chen et al., 2022):

$$R^c_{L,s}=\sum_j R_{L,s,j}\,\frac{\Delta c_j}{\Delta h_{L,s,j}},\quad c\in\{n^{G_1},\dots,n^{G_m},\varepsilon\},\qquad \sum_c R^c_{L,s}=\sum_j R_{L,s,j}.$$

Using one reference throughout matters: zero-referenced relevance split by baseline-differenced ratios would mix two references. M10 gives the cheapest full layer profile, comparable to M6 and M9. Conservation is a design property, not faithfulness, and the ratio is unstable where $\Delta h_j\approx0$. Rules for smolgen and post-LN are not principled, and DeepLIFT rules through attention are untested. Controls: C1 as a sanity check (Adebayo et al., 2018) and agreement with M6 on the same pairs. Cost: one custom backward pass per position for all layers, in a PyTorch port. Residual: $R^\varepsilon_L$ per layer.

### 6.2 Family (ii): erasure

**M3. Family erasure per layer with a use-versus-readability curve.** For family $G$ at layer $L$, LEACE (Belrose et al., 2023) gives the least-squares-optimal linear erasure in closed form. With whitening $W=(\Sigma_{hh}^{1/2})^{+}$ and $P$ the orthogonal projector onto $\mathrm{col}(W\Sigma_{hz_G})$,

$$r^G_L(h)=h-W^{+}PW(h-\mu),\qquad \Delta^{\rm erase}_{G,L}(x)=g_L\big(r^G_L(h_L(x))\big)-y(x),$$

applied on all 64 squares. Report $\mathbb E|\Delta^{\rm erase}|$ and signed means beside the probe AUROC of $G$ at $L$. The lineage is amnesic probing (Elazar et al., 2021), INLP (Ravfogel et al., 2020) and R-LACE (Ravfogel et al., 2022). The chess case for erasure over accuracy is Pálsson and Björnsson (2024). M3 answers whether $G$'s linearly available information at $L$ is used downstream. The guarantee is linear only, since a nonlinear reader can still recover $G$. Large families approach rank 256, where erasure becomes "erase everything", so erased rank and variance removed must be reported. Correlation between concept and target inflates engine-derived families (A. Kumar et al., 2022), and recomputation from other squares or later layers hides use, which M8 measures. Controls: C6 in whitened coordinates, C7, C1, C2. Cost: one affine patch under ONNX, covariances cheap, pass count as M2. Residual: the output that survives erasing all families jointly, carried by unnamed or nonlinearly named content.

**M8. Cross-layer erasure and substitution cascade.** Erase (M3) or substitute (M2) family $G$ at $L$ and read every later tap $L'>L$:

$$T^{L\to L'}_{G\to G'}=\mathbb E_x\Big[\mathrm{score}_{G'}\big(P_{L'}(h_{L'})\big)-\mathrm{score}_{G'}\big(P_{L'}(h^{\,\text{erase }G@L}_{L'})\big)\Big].$$

The diagonal $T^{L\to L'}_{G\to G}$ measures recomputation, and the change in $\|\varepsilon_{L'}\|$ or in $\phi^{(L')}_\varepsilon$ measures whether $G$ feeds unnamed computation. In the decoder-read variant, which is the user's "several decoders on several layers" in the strict sense, $n^G_L(x)$ is replaced by $n^G_L(x')$ and the change at $L'$ is read through $D_{L'}$'s own split, giving $\Delta n^{G'}_{L'}$ by least squares on $W_{L'}$ and $\Delta\varepsilon_{L'}$. A gradient variant uses the block Jacobian $\partial c^{G'}_{L+1}/\partial n^G_L$ (Farnik et al., 2025). Sources: amnesic probing (Elazar et al., 2021), LEACE at every layer (Belrose et al., 2023), Engels et al.'s (2025) finding that earlier-layer maps explain later error, and backup behaviour under ablation (Heimersheim & Nanda, 2024). M8 answers which named features lc0 builds from which, where it recomputes an erased feature (so single-site erasure underestimates use), and which named features turn into residual. Probes read patched, off-distribution activations; recomputation from unerased squares appears unless all squares are erased; and transfer is confounded by dependencies inside the extractor, whose own dependency graph should be reported alongside. Controls: C6, C1, C7. Cost: one patched forward pass per $(G,L)$ gives all later taps, about 300 passes per position. Residual: $\Delta\phi_\varepsilon$ per $(G,L)$, the flow from named to unnamed.

### 6.3 Family (iii): probe gradient

**M4. Concept gradient through the probe at layer $L$.** With a family-score probe $P^{\mathcal G}_L:h_{L,s}\mapsto c_s\in\mathbb R^m$ and Jacobian $J_s=\partial c_s/\partial h_{L,s}\in\mathbb R^{m\times256}$, the concept gradient (Bai et al., 2023) and its unexplained part are

$$\mathrm{CG}_{L,s}=(J_s^\top)^{+}\,\nabla_{h_{L,s}}y,\qquad \rho^\perp_{L,s}=\nabla_{h_{L,s}}y-J_s^\top\mathrm{CG}_{L,s}.$$

TCAV is the linear special case (Kim et al., 2018). M4 must work at family level, $m\approx30<256$. At bit level $K\gg256$, $J_s^\top$ is surjective, $\rho^\perp\equiv0$ by construction, and the minimum-norm split is arbitrary. The inner product for $J$ should be whitened (Park et al., 2024). M4 answers how sensitive $y$ is locally to each family's probe coordinates and what share of the gradient no family direction spans. Probe directions need not be the model's directions (Park et al., 2024; Tucker et al., 2021). Because the gradient is taken through lc0, M4 measures local use rather than readability alone. The pseudo-inverse is unstable when family directions are collinear (Nicolson et al., 2025). Controls: C1, C3 on probe regularisation, C2, control-task probes (Hewitt & Liang, 2019), and sign agreement with M3. Cost: one backward pass per position for all taps, in a PyTorch port. Residual: $\|\rho^\perp_L\|^2/\|\nabla_{h_L}y\|^2$, the gradient-side analogue of FVU.

### 6.4 Family (iv): causal alignment

**M5. Interchange alignment per family per layer.** DAS (Geiger et al., 2024), in its boundless form (Wu et al., 2023), learns an orthogonal rotation $R_L$ and a subspace size $k_G$. For base $x$ and source $x'$,

$$h^{\rm int}_{L,s}=h_{L,s}(x)+R_L^\top\Pi_{k_G}R_L\big(h_{L,s}(x')-h_{L,s}(x)\big),\quad s\in S.$$

The high-level counterfactual is $y(x')$ for pairs in which $x'$ is a minimal real edit whose feature change is confined to $G$, up to the extractor's own downstream dependencies, and otherwise a matched board (Gat et al., 2024). Graded IIA is $1-\mathbb E\big(g_L(h^{\rm int})-y^*\big)^2/\mathbb E\big(y(x)-y^*\big)^2$. Multi-task DAS adds RAVEL's Isolate score: pairs that differ only in a correlated family $G'$ must leave the $G$-interchange output unchanged (Huang et al., 2024). $R_L$ is initialised from the decoder columns $W^G_L$ and, separately, from probe directions. The theory is causal abstraction (Geiger et al., 2021, 2025), and MIB's causal-variable track is the closest benchmark (Mueller et al., 2025). M5 answers whether a subspace at $L$ mediates $G$'s effect on $y$, how many dimensions it needs, and whether it separates from its correlates. Subspace patching can produce dormant-pathway illusions (Makelov et al., 2024), though Wu et al. (2024) argue the criterion also flags valid distributed solutions, so C8 is a diagnostic rather than a gate. Clean pairs exist for some families (pins, blockers, piece removal) and not for others, such as "strong square". Multi-square families need joint interventions. The decisive control is C1: Sutter et al. (2025) report near-perfect IIA on random networks. Maps stay linear, and $k$ is swept. Cost: a small optimisation per family and layer at $d=256$, minutes on CPU each, hours for 30 families at 11 taps. Residual: $1-\mathrm{IIA}$, plus interchange of the orthogonal complement of all family subspaces jointly, in the manner of causal scrubbing (Chan et al., 2022, Tier 4). The output change it causes is the share mediated outside the vocabulary's subspaces.

### 6.5 Family (v): supervised replacement graph

**M7. Supervised local replacement graph with error nodes.** Per position, freeze attention patterns $A_L(x)$ and LayerNorm scales (Ali et al., 2022; Ameisen et al., 2025). Replace each feed-forward block with a supervised skip transcoder that reads features from the model rather than from the extractor:

$$\mathrm{FFN}_L(u)\approx S_Lu+V_L\,\sigma\big(P_L(u)\big)+b_L,\qquad r_{L,s}=\mathrm{FFN}_L(u_s)-\big[S_Lu_s+V_L\sigma(P_L(u_s))+b_L\big].$$

With attention and LayerNorm frozen, the network is linear in the embedding, the transcoder feature writes $V^G_L\sigma(P^G_L(u_s))$ at each block and square, and the error nodes $r_{L,s}$. The edge weight from node $(L,s,G)$ to $y$ is activation times virtual weight through frozen attention, which traces square-to-square routes. Completeness is the share of absolute attribution passing through non-error nodes, and a replacement score follows Ameisen et al. (2025), who report completeness 0.80 and replacement 0.61 on a language model. Reading $f$ through $P_L$ matters: a transcoder fed extractor bits can fit well even if lc0 never computes $G$, which is leakage (Mahinpei et al., 2021). The parts come from transcoders (Dunefsky et al., 2024; Paulo et al., 2025) and sparse feature circuits (Marks et al., 2025); Lin et al. (2026) is the unsupervised precedent on lc0. M7 answers which family, written at which block on which square, reaches $y$ along which route, and where in depth error nodes take over. Its central weakness is unresolved: frozen query-key products hide attention routing, and Jenner et al. (2024) place lc0's look-ahead in attention heads. A frozen graph will credit routing to whoever wrote the values. AtP*-style recomputation of the query-key path bounds what freezing hides (Kramár et al., 2024). Freezing LayerNorm scales is a local linearisation, not exact, and smolgen is unmodelled. Controls: C1, C3, C4, and ablation of pruned graphs checked against M2 and M6. Cost: transcoders are ridge fits taking minutes; one graph per position needs a forward pass and one backward pass per output, seconds on CPU. Residual: error-node share per block and square, and completeness and replacement per position.

### 6.6 Family (vi): residual analysis

**M12. Residual budget and hybrid dictionary.** First, regress $\varepsilon_{L,s}$ on nested predictor sets and record the explained-variance increments:

$$\varepsilon_{L,s}\approx\underbrace{B^{\rm nl}_L\,\psi(f_s)}_{\text{(b) nonlinear in }f_s}+\underbrace{B^{\rm nb}_L\textstyle\sum_t\bar a^L_{st}f_t}_{\text{(a) non-local}}+\underbrace{B^{\rm cf}_L\,\varepsilon_{L-1,s}}_{\text{carried forward}}+\eta_{L,s}.$$

Increments depend on order, so they are averaged over orders (Shapley over the three blocks). This follows Engels et al.'s (2025) decomposition of SAE error. Second, fit a size-swept sparse code $z=\mathrm{SAE}(\eta_L)$ on the remainder and add $z$ as extra players in M9 or nodes in M7, in the manner of concept-bottleneck SAEs (Kulkarni et al., 2026). Third, try to name latents by matching them to held-back candidate features or new extractor bits, scored for coverage as in Karvonen et al. (2024), with a contrastive variant on board pairs (Poupart, 2024). M12 answers how much of the decay in Section 5.4 is non-locality, map class, carry-over or remainder, and whether the remainder contains stable, causally used, nameable units. Unsupervised latents depend on dictionary size (Leask et al., 2025), naming works only if the candidate bank contains the concept (Shang et al., 2024), and predictable is not the same as nameable. Controls: C1 (latents that also appear on a random network are not concepts), size sweep, seed stability. Cost: regressions in minutes, SAEs on $\eta_L$ in CPU hours per layer. Residual: the budget itself, then the share of $\varphi_\varepsilon$ recovered by named latents.

### 6.7 Comparison across methods

**M11. Cross-method, cross-layer disagreement map.** For family attributions $A^{(k,L)}_G$ from methods $k$ at taps $L$ on a shared set of pairs, report Kendall rank agreement over families, sign agreement, and residual shares $A^{(k,L)}_\varepsilon/\Delta^{\rm true}$ side by side. No weights are fitted. A learned ensemble that matches $\sum_GA_G$ to $\Delta^{\rm true}$ was considered and rejected. It rewards layers and methods with small $\varepsilon$, which selects against the residual. With $\varepsilon$ included, every complete method has zero loss and the weights are unidentified. GCAV's learned cross-layer fusion (He et al., 2025) has the same identifiability problem. M11 shows where methods and layers agree on which families matter and where an attribution belongs to the operator rather than to lc0. Agreement is not faithfulness: the five decoder-split methods share ridge artefacts and will agree for that reason, so agreement is computed within and across families separately. The random-network floor (C1) bounds chance agreement. Cost is negligible beyond the component methods.

## 7. Comparison

Table 4 places the twelve methods on the nine taxonomy axes.

**Table 4**
*Methods on the nine taxonomy axes*

| # | Method | Entry | Direction | Layer span | Map | Operator | Residual reported | Identifiability | Faithfulness test | Cost | Combines layers or maps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | Decoder gradient at $e$ | input | decode | $e$; path crosses 10 blocks | linear | gradient, board-path IG | $\mathrm{IG}_\varepsilon$ | non-unique within occupancy-derived bits | vs edits | torch; cheap | no |
| M2 | Decoded substitution | input | decode | per layer | linear | substitution patching | $\gamma_L$ | unique per joint patch | interventional | ONNX; 1–2 s/pos | no |
| M3 | Family erasure | readout | encode | per layer | linear | erasure | output after erasing all | unique given $G$ | interventional + random rank | ONNX; cheap | no |
| M4 | Concept gradient | readout | encode | per layer | linear/MLP | Jacobian pseudo-inverse | $\|\rho^\perp\|$ (family level) | min-norm | vs edits; sign vs M3 | torch; 1 bwd | no |
| M5 | DAS / IIA | readout | both | per layer | linear rotation | interchange | $1-$IIA; complement swap | up to rotation | IIA, Isolate | torch; hours | no |
| M6 | Named/error sweep | input | decode | all taps, differenced | linear | 2-player Shapley | $\phi^{(L)}_\varepsilon$, $\rho_L$ | exact | completeness exact; C3 | ONNX; 0.1 s/pair | yes (sweep) |
| M7 | Supervised replacement graph | both | both | cross-layer, path | sparse-linear | linear edges, frozen attention | error nodes per block/square | regulariser-dependent | ablation vs M2/M6 | torch; s/pos | yes |
| M8 | Erasure cascade | readout (or decoder) | encode→encode | cross-layer | linear | erasure/substitution, re-read | $\Delta\phi_\varepsilon$ | unique per $(G,L)$ | interventional | ONNX; ~300 passes/pos | yes |
| M9 | Family Shapley + $\varepsilon$ | input | decode | per layer (+ site players) | linear | Shapley, distributional | $\varphi_\varepsilon$ | ridge-dependent at family level | localisation | ONNX; ~30 s/pos/layer | only with site players |
| M10 | Relevance chain | input | decode | all taps, one pass | linear | DeepLIFT rescale | $R^\varepsilon_L$ | rule-dependent | agreement with M6 | torch, custom bwd | no |
| M11 | Disagreement map | mixed | mixed | all | none | comparison | spread of residual shares | nothing fitted | random-net floor | negligible | yes |
| M12 | Residual budget + hybrid dictionary | dictionary-matched | both | per layer, uses $L-1$ | linear + sparse | regression, SAE, M9 | budget (a)/(b)/carry/remainder | non-unique (SAE size) | random init, size sweep | CPU hours | yes |

*Note.* Five methods combine layers or approaches (M6, M7, M8, M11, M12), M9 only with site players. Five are operators on one decoder split (M1, M2, M6, M9, M10). Five run without a PyTorch port (M2, M3, M6, M8, M9).

The methods that run on ONNX today are the interventional ones, which is also the project's preferred kind of evidence. The gradient and graph methods wait on a PyTorch port.

## 8. The Residual as Coverage

The user's priority is to measure what the vocabulary misses, not only whether named features are faithful. That priority has a direct consequence for evaluation, found in the second devil's-advocate review.

A standard faithfulness score for concept attribution compares the attribution summed over changed concepts with the true effect, in the style of CEBaB's ICaCE error (Abraham et al., 2022):

$$\mathrm{err}=\frac{\mathbb E\big|\sum_{G\in\delta}A_G-\Delta^{\rm true}\big|}{\mathbb E|\Delta^{\rm true}|}.$$

For any complete pairwise method (M1 board-path, M2 joint, M6, M9 with $x_0=x'$), unchanged families receive zero by the dummy axiom and the attributions plus the residual sum to $\Delta^{\rm true}$. The error is then exactly $|A_\varepsilon|/|\Delta^{\rm true}|$: the residual share. Ranking methods, layers or decoders by it would favour whichever pushes most credit into named families. That is the opposite of measuring what the vocabulary misses. The same holds for any ensemble fitted to match $\Delta^{\rm true}$ with named terms only.

The protocol therefore separates two quantities. *Coverage* is the named share, $\rho_L$ from M6 or $1-\varphi_\varepsilon/\sum|\varphi|$ from M9. It is a finding about lc0 and the vocabulary, reported per layer, per family, and with and without engine-derived families. *Error* applies only where an attribution can be wrong without the residual absorbing the difference: the M1 point form, M4, M9 with distributional baselines, and M10. Complete pairwise methods are graded instead by the consistency gap $\gamma_L$, by control gaps against C1, C2 and C5, and by stability under C3.

Three residual measures then answer three different questions. $\phi^{(L)}_\varepsilon$ (M6) and $\varphi_\varepsilon$ (M9) say how much of a real output change runs through content the vocabulary does not name at layer $L$. The M12 budget says why: non-locality, map class, carry-over from the previous layer, or a remainder. The M12 dictionary and naming step asks whether the remainder contains stable units that lc0 uses and that a new extractor bit could name. Havasi et al.'s (2022) distinction between missing vocabulary and weak maps is the reason the budget comes before the dictionary. A residual explained by a neighbour-augmented or nonlinear decoder is a property of the map, not of the vocabulary.

Concepts lc0 has that humans have no word for would sit in the remainder $\eta_L$ after the budget, at layers where $\rho_L$ is low and stable. Schut et al. (2025) are the closest precedent for naming such content in a game network, validated behaviourally. Whether $\eta_L$ holds anything nameable is open, and the protocol must allow the answer "no" and a $\rho_L$ near zero at depth.

## 9. Evaluation Protocol

### 9.1 Ground truth in three tiers

Ground truth is split by what it is true to, following Chen et al. (2020).

**Table 5**
*Ground-truth tiers*

| Tier | Source | Measures | Used for |
|---|---|---|---|
| GT-M (true to lc0) | Real board edits run through lc0: piece removal, moving a pinner off the line, interposition on an empty line square, single pawn moves | $\Delta^{\rm true}=y(x')-y(x)$; per-family feature change | Grading every method |
| GT-C (true to chess) | MaskPinner and LineBlocker in patched Stockfish | Engine's rule-level effect; "pin/line decisive" label | Whether lc0's attribution to pin and line families tracks the engine's causal effect: a claim about lc0's alignment with chess, not about attribution faithfulness |
| GT-K (consistency) | Decoder edit vs board edit at the same layer | $\gamma_L$ (M2) | How far feature-space edits are from realisable ones |

lc0's input carries eight history frames. Each edit is applied to every frame, or both boards use history-free input. Otherwise $\Delta^{\rm true}$ contains a history artefact. For each family, pairs are minimal edits whose feature change is concentrated in that family, and the number of changed bits outside it is reported. Controls for line-blocking edits need a nonzero noise floor: in earlier project work, control squares chosen by geometry alone gave a near-zero floor that skewed verdicts towards "causal". The position pool mixes puzzle and non-puzzle positions, since probe-set choice changes concept explanations (Ramaswamy et al., 2023).

### 9.2 Metrics

- Coverage: $\rho_L$, $\phi^{(L)}_\varepsilon$, $\varphi_\varepsilon$, completeness (Yeh et al., 2020; Ameisen et al., 2025). Never used to rank methods.
- Error and localisation (rank of the edited family among $|A_G|$ on single-family edits): only for non-pairwise methods, M1 point form, M4, M9 distributional, M10.
- Consistency: $\gamma_L$ (M2).
- Alignment: IIA and Isolate (M5).
- Stability: Spearman correlation of family rankings across C3 settings and between linear and MLP maps (C4).
- Control gaps: metric on trained lc0 minus metric under C1, C2 and C5.
- Fixed in advance and reported: ablation type, noising or denoising direction, and the value metric as a logit rather than a probability (Zhang & Nanda, 2024; Miller et al., 2024).

### 9.3 Pre-specified hypotheses

None of the six has been tested.

- **H1.** At $e$, with decoded deltas zeroed on squares whose planes are unchanged, M1, M6 and M9 attributions under the full vocabulary match the local-vocabulary control C2 within stability noise. Without the zeroing, non-local families receive credit that $\varepsilon_e$ cancels.
- **H2.** $\rho_L$ falls with depth after L2. Whether it peaks at L1–L2 or at $e$ is open, given Table 2.
- **H3.** In the M12 budget, non-locality explains a non-trivial share of $\varepsilon_L$ at L5–L9.
- **H4.** Among non-pairwise methods, interventionally grounded ones (M9 distributional) localise single-family edits better than gradient and relevance methods (M1 point form, M4, M10).
- **H5.** LEACE use curves (M3) lag readability curves, and recomputation (M8) is high for occupancy-derived families.
- **H6.** Engine-derived families gain attribution with depth, resembling Tomlin et al.'s (2022) Go profile, and lose most of it under C7.

### 9.4 Staging and budget

**Table 6**
*Stages and order-of-magnitude CPU cost*

| Stage | Methods | Needs | Estimated cost |
|---|---|---|---|
| A | M6, M2, M3, M8, M9 | ONNX graph split at taps; edit pairs | about one CPU day for 1,000–10,000 pairs |
| B | M1, M4, M5, M7, M10 | PyTorch port of the network, verified against ONNX outputs | port effort plus about one CPU day |
| C | M11, M12 | Outputs of A and B; SAE training on $\eta_L$ | hours to days |

Stage A answers the core question, the named and missed share per layer and per family, without a port. H1–H6, edit construction and metrics should be preregistered before Stage A.

## 10. Discussion

The user asked how to get attribution from the decoders, whether to start at the front or the back, and whether a hybrid helps. The survey gives three answers.

First, the front is less informative than it looks. Its near-perfect fit follows from locality (Section 5.2), so attribution there is input saliency with feature names attached, and the embedding FVU measures the extractor rather than lc0's learning.

Second, the choice between front and back is not a choice. M6 evaluates the same real output change at every tap and reports how much of it the named part carries. That curve, $\rho_L$, is the hybrid, and nothing has to be blended. Where $\rho_L$ drops is where the vocabulary stops describing what lc0 uses, and the M12 budget then says whether the drop is non-locality, a weak map, carry-over, or content without a name.

Third, maps can be combined across layers in three distinct ways. M6 compares independent per-layer decoders on one effect, cheaply and under ONNX. M8 intervenes at one layer and reads later layers through decoders or probes, which yields a family-to-family transfer tensor and measures recomputation. M7 composes block-level maps into one linear graph, at the price of freezing the attention that Jenner et al. (2024) find does the look-ahead. No source fits a supervised cross-layer map, and no game-network paper reports a residual share in output attribution (Table 7).

**Table 7**
*Gaps the protocol addresses, with honest novelty claims*

| Gap | Evidence of absence | Claim |
|---|---|---|
| Attribution through a fitted decoder $f\to h$ | Concept gradients use an encoder pseudo-inverse (Bai et al., 2023); alignment methods work in the encode direction | First use on a game network. With linear $D$, M1 is gradient×input in a new basis and reduces to input saliency at $e$; the contribution is the interventional variants M2, M6, M9 |
| Supervised cross-layer replacement with error nodes | All cross-layer dictionaries are unsupervised (Section 4.5) | M7 combines existing parts; novel as a combination, with Tier-2 dependence |
| Residual share for a fixed external vocabulary | Lin et al. (2026) report reconstruction per module, not a graph residual | $\rho_L$ and the M12 budget are new measurements, not new theory |
| Interventional validation of named families on lc0 | lc0 interventions exist at square level only (Jenner et al., 2024; Sandmann et al., 2025; Lin et al., 2026) | The protocol in Section 9 |
| Residual split by locality | Engels et al. (2025) split error by linear predictability only | M12 budget; neighbour-augmented decoder |
| Interventional concept benchmarks outside NLP | CEBaB, RAVEL and MIB are NLP | A chess benchmark with exact board-level and engine-level operators |
| Attribution for binary, logically constrained features | Closest: grouping (Aas et al., 2021) and on-manifold Shapley (Frye et al., 2021) | Not solved; board edits sidestep it only for families with exact edits |
| Post-LN residual stream; attention routing in feature graphs | Crosscoder and CLT designs assume additive writes and freeze attention | Flagged as limits, not solved |

The strongest objection deserves a direct reply. Five of the twelve methods are one ridge decomposition read through different operators. Where the decomposition is complete, faithfulness to lc0 composed with the decoder holds by construction, and what remains informative is the residual and the regulariser-dependent family split. On this view, the empirical content reduces to M6's curve and stability checks, and at the embedding even that curve is fixed by locality. This report accepts most of the objection. It is the reason the methods are counted as six families, the reason coverage is reported rather than optimised, and the reason C1 to C3 are attached to every decoder-split method. The interventional methods outside the split (M3, M5, M8) and the residual budget (M12) carry the claims that do not follow from linear algebra.

Two tensions are left open. Pálsson and Björnsson (2024) recommend nonlinear probes for chess agents, while the identifiability results of Sutter et al. (2025) argue for linear maps. Nanda et al. (2023) suggest a side-to-move-relative encoding can make the representation linear, which this vocabulary should test. And freezing attention, which makes M7 exact, conflicts with where lc0 does its look-ahead (Jenner et al., 2024).

## 11. Limitations

Nothing in the method catalogue has been run. The only empirical inputs are the probe, decoder and value-head fits in Table 1, from 20,000 positions for the per-square maps. Every statement about how a method behaves on lc0, including H1–H6 and the cost estimates, is a hypothesis or an estimate. Costs assume that suffix cost scales with remaining depth.

The survey is not a systematic review. Coverage was judged by a citation-chasing stopping rule, not counted. The field is preprint-heavy. The key cross-layer sources (crosscoders, CLTs, circuit tracing) are Tier 2 and not peer-reviewed, and much of the causal-abstraction literature comes from one research lineage. Interventional benchmarks are all from NLP, so their transfer to chess rests on the project's own operators.

The embedding's locality and nonlinearity were checked against the ONNX graph. Other structural claims, such as post-LN taps and smolgen, come from the project's description of the model. GT-C grades lc0's agreement with Stockfish's rule-level effects, not the faithfulness of an attribution to lc0; treating one as the other would overclaim. Engine-derived features partly encode the value target, so attribution to them can be high for trivial reasons; C7 reports results with and without them. The pairwise contradiction scan across sources covered ten candidate pairs and is not exhaustive.

## AI-Use Disclosure

This report was produced with AI-assisted research tools. The pipeline used AI agents for literature search, source verification against arXiv and publisher records, evidence synthesis, two devil's-advocate reviews, and drafting. The empirical numbers in Table 1 come from the project's own code and were not generated by the pipeline. The embedding's functional form was read from the exported ONNX graph. Cited claims were checked against the annotated bibliographies produced in Phase 2. The project author directs the research and decides which conclusions to adopt.

## References

Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Artificial Intelligence, 298*, Article 103502. https://doi.org/10.1016/j.artint.2021.103502

Abraham, E. D., D'Oosterlinck, K., Feder, A., Gat, Y., Geiger, A., Potts, C., Reichart, R., & Wu, Z. (2022). CEBaB: Estimating the causal effects of real-world concepts on NLP model behavior. In *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2205.14140

Achtibat, R., Dreyer, M., Eisenbraun, I., Bosse, S., Wiegand, T., Samek, W., & Lapuschkin, S. (2023). From attribution maps to human-understandable explanations through concept relevance propagation. *Nature Machine Intelligence, 5*(9), 1006–1019. https://doi.org/10.1038/s42256-023-00711-8

Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., & Samek, W. (2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR 235, pp. 135–168). https://proceedings.mlr.press/v235/achtibat24a.html

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. In *Advances in Neural Information Processing Systems 31* (pp. 9505–9515). https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps

Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R., & Wolf, L. (2022). XAI for transformers: Better explanations through conservative propagation. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 435–451). https://proceedings.mlr.press/v162/ali22a.html

Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., . . . Batson, J. (2025, March 27). Circuit tracing: Revealing computational graphs in language models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Bai, A., Yeh, C.-K., Lin, N. Y. C., Ravikumar, P., & Hsieh, C.-J. (2023). Concept gradient: Concept-based interpretation without linear assumption. In *International Conference on Learning Representations*. https://arxiv.org/abs/2208.14966

Beckers, S., & Halpern, J. Y. (2019). Abstracting causal models. In *Proceedings of the AAAI Conference on Artificial Intelligence, 33*. https://arxiv.org/abs/1812.03789

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics, 48*(1), 207–219. https://doi.org/10.1162/coli_a_00422

Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect linear concept erasure in closed form. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2306.03819

Bilodeau, B., Jaques, N., Koh, P. W., & Kim, B. (2024). Impossibility theorems for feature attribution. *Proceedings of the National Academy of Sciences, 121*(2), e2304406120. https://doi.org/10.1073/pnas.2304406120

Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., Greenblatt, R., Nitishinskaya, J., Radhakrishnan, A., Shlegeris, B., & Thomas, N. (2022, December 3). *Causal scrubbing: A method for rigorously testing interpretability hypotheses* [Online forum post]. AI Alignment Forum. https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing

Chen, H., Janizek, J. D., Lundberg, S., & Lee, S.-I. (2020). *True to the model or true to the data?* [Preprint]. arXiv. https://arxiv.org/abs/2006.16234

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

Hammersborg, P., & Strümke, I. (2024). Information based explanation methods for deep learning agents—with applications on large open-source chess models. *Scientific Reports, 14*, 20174. https://doi.org/10.1038/s41598-024-70701-2

Havasi, M., Parbhoo, S., & Doshi-Velez, F. (2022). Addressing leakage in concept bottleneck models. In *Advances in Neural Information Processing Systems 35*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html

He, Z., Sinha, S., Xiong, G., & Zhang, A. (2025). GCAV: A global concept activation vector framework for cross-layer consistency in interpretability. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*. https://arxiv.org/abs/2508.21197

Heimersheim, S., & Nanda, N. (2024). *How to use and interpret activation patching* (arXiv:2404.15255) [Preprint]. arXiv. https://arxiv.org/abs/2404.15255

Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 2733–2743). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1275

Huang, J., Wu, Z., Potts, C., Geva, M., & Geiger, A. (2024). RAVEL: Evaluating interpretability methods on disentangling language model representations. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 8669–8687). https://doi.org/10.18653/v1/2024.acl-long.470

Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance quantification in explainable AI: A causal problem. In *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics* (PMLR 108, pp. 2907–2916). https://proceedings.mlr.press/v108/janzing20a.html

Jenner, E., Kapur, S., Georgiev, V., Allen, C., Emmons, S., & Russell, S. (2024). Evidence of learned look-ahead in a chess-playing neural network. In *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2406.00877

Kantamneni, S., Engels, J., Rajamanoharan, S., Tegmark, M., & Nanda, N. (2025). Are sparse autoencoders useful? A case study in sparse probing. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 29018–29049). https://arxiv.org/abs/2502.16681

Karvonen, A., Wright, B., Rager, C., Angell, R., Brinkmann, J., Smith, L., Mayrink Verdun, C., Bau, D., & Marks, S. (2024). Measuring progress in dictionary learning for language model interpretability with board game models. In *Advances in Neural Information Processing Systems 37*. https://doi.org/10.52202/079017-2644

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In *Proceedings of the 35th International Conference on Machine Learning* (PMLR 80). https://arxiv.org/abs/1711.11279

Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). Concept bottleneck models. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 5338–5348). https://proceedings.mlr.press/v119/koh20a.html

Kramár, J., Lieberum, T., Shah, R., & Nanda, N. (2024). *AtP\*: An efficient and scalable method for localizing LLM behaviour to components* (arXiv:2403.00745) [Preprint]. arXiv. https://arxiv.org/abs/2403.00745

Kulkarni, A., Weng, T.-W., Narayanaswamy, V., Liu, S., Sakla, W. A., & Thopalli, K. (2026). Interpretable and steerable concept bottleneck sparse autoencoders. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://arxiv.org/abs/2512.10805

Kumar, A., Tan, C., & Sharma, A. (2022). Probing classifiers are unreliable for concept removal and detection. In *Advances in Neural Information Processing Systems 35* (pp. 17994–18008). https://arxiv.org/abs/2207.04153

Kumar, I. E., Venkatasubramanian, S., Scheidegger, C., & Friedler, S. (2020). Problems with Shapley-value-based explanations as feature importance measures. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 5491–5500). https://proceedings.mlr.press/v119/kumar20e.html

Lawson, T., Farnik, L., Houghton, C., & Aitchison, L. (2025). Residual stream analysis with multi-layer SAEs. In *International Conference on Learning Representations*. https://arxiv.org/abs/2409.04185

Leask, P., Bussmann, B., Pearce, M., Bloom, J., Tigges, C., Al Moubayed, N., Sharkey, L., & Nanda, N. (2025). Sparse autoencoders do not find canonical units of analysis. In *International Conference on Learning Representations*. https://arxiv.org/abs/2502.04878

Lin, R., Jin, Z., Zhou, G., Ge, X., Shu, W., Wu, J., Wang, J., He, Z., Zhang, J., & Qiu, X. (2026). *Tracing the thought of a grandmaster-level chess-playing transformer* (arXiv:2604.10158) [Preprint]. arXiv. https://arxiv.org/abs/2604.10158

Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., & Olah, C. (2024, October 25). Sparse crosscoders for cross-layer features and model diffing. *Transformer Circuits Thread*. https://transformer-circuits.pub/2024/crosscoders/index.html

Lomasov, S., Goldfeder, J., Erol, M. H., So, M., Yan, Y., Howard, A., Kutz, N., & Shwartz-Ziv, R. (2025). Exploring human-AI conceptual alignment through the prism of chess. In *Advances in Neural Information Processing Systems 38* (Creative AI Track). https://arxiv.org/abs/2510.26025

Lundstrom, D. D., Huang, T., & Razaviyayn, M. (2022). A rigorous study of integrated gradients method and extensions to internal neuron attributions. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 14485–14508). https://proceedings.mlr.press/v162/lundstrom22a.html

Mahinpei, A., Clark, J., Lage, I., Doshi-Velez, F., & Pan, W. (2021). *Promises and pitfalls of black-box concept learning models* [Workshop paper]. ICML 2021 Workshop on Theoretic Foundation, Criticism, and Application Trend of Explainable AI. https://arxiv.org/abs/2106.13314

Makelov, A., Lange, G., Geiger, A., & Nanda, N. (2024). Is this the subspace you are looking for? An interpretability illusion for subspace activation patching. In *International Conference on Learning Representations*. https://openreview.net/forum?id=Ebt7JgMHv1

Makelov, A., Lange, G., & Nanda, N. (2025). Towards principled evaluations of sparse autoencoders for interpretability and control. In *International Conference on Learning Representations*. https://arxiv.org/abs/2405.08366

Margeloiu, A., Ashman, M., Bhatt, U., Chen, Y., Jamnik, M., & Weller, A. (2021). *Do concept bottleneck models learn as intended?* [Workshop paper]. ICLR 2021 Workshop on Responsible AI. https://arxiv.org/abs/2105.04289

Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2025). Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In *International Conference on Learning Representations*. https://arxiv.org/abs/2403.19647

McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Wattenberg, M., Hassabis, D., Kim, B., Paquet, U., & Kramnik, V. (2022). Acquisition of chess knowledge in AlphaZero. *Proceedings of the National Academy of Sciences, 119*(47), e2206625119. https://doi.org/10.1073/pnas.2206625119

Miller, J., Chughtai, B., & Saunders, W. (2024). Transformer circuit faithfulness metrics are not robust. In *Conference on Language Modeling*. https://arxiv.org/abs/2407.08734

Mueller, A., Geiger, A., Wiegreffe, S., Arad, D., Arcuschin, I., Belfki, A., Chan, Y. S., Fiotto-Kaufman, J., Haklay, T., Hanna, M., Huang, J., Gupta, R., Nikankin, Y., Orgad, H., Prakash, N., Reusch, A., Sankaranarayanan, A., Shao, S., Stolfo, A., . . . Belinkov, Y. (2025). MIB: A mechanistic interpretability benchmark. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 45069–45108). https://proceedings.mlr.press/v267/mueller25a.html

Nanda, N., Lee, A., & Wattenberg, M. (2023). Emergent linear representations in world models of self-supervised sequence models. In *Proceedings of the 6th BlackboxNLP Workshop* (pp. 16–30). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.blackboxnlp-1.2

Nicolson, A., Schut, L., Noble, J. A., & Gal, Y. (2025). Explaining explainability: Recommendations for effective use of concept activation vectors. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2404.03713

Pálsson, A., & Björnsson, Y. (2024). Empirical evaluation of concept probing for game-playing agents. In *ECAI 2024* (Frontiers in Artificial Intelligence and Applications, Vol. 392, pp. 874–881). IOS Press. https://doi.org/10.3233/FAIA240574

Parisini, E., Chakraborti, T., Harbron, C., MacArthur, B. D., & Banerji, C. R. S. (2026). Leakage and interpretability in concept-based models. *Journal of Machine Learning Research, 27*(174), 1–40. https://www.jmlr.org/beta/papers/v27/25-1121.html

Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR 235). https://arxiv.org/abs/2311.03658

Paulo, G., Shabalin, S., & Belrose, N. (2025). *Transcoders beat sparse autoencoders for interpretability* (arXiv:2501.18823) [Preprint]. arXiv. https://arxiv.org/abs/2501.18823

Poupart, Y. (2024). *Contrastive sparse autoencoders for interpreting planning of chess-playing agents* (arXiv:2406.04028) [Preprint]. arXiv. https://arxiv.org/abs/2406.04028

Ramaswamy, V. V., Kim, S. S. Y., Fong, R., & Russakovsky, O. (2023). Overlooked factors in concept-based explanations: Dataset choice, concept learnability, and human capability. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://arxiv.org/abs/2207.09615

Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 7237–7256). https://doi.org/10.18653/v1/2020.acl-main.647

Ravfogel, S., Twiton, M., Goldberg, Y., & Cotterell, R. (2022). Linear adversarial concept erasure. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162). https://proceedings.mlr.press/v162/ravfogel22a.html

Ravichander, A., Belinkov, Y., & Hovy, E. (2021). Probing the probing paradigm: Does probing accuracy entail task relevance? In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics* (pp. 3363–3377). https://doi.org/10.18653/v1/2021.eacl-main.295

Rubenstein, P. K., Weichwald, S., Bongers, S., Mooij, J. M., Janzing, D., Grosse-Wentrup, M., & Schölkopf, B. (2017). Causal consistency of structural equation models. In *Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence*. https://arxiv.org/abs/1707.00819

Sandmann, E., Lapuschkin, S., & Samek, W. (2025). *The algorithm is not the behavior: Learned priors override look-ahead in a chess-playing neural network* (arXiv:2508.21380) [Preprint]. arXiv. https://arxiv.org/abs/2508.21380

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

Wu, Z., Geiger, A., Huang, J., Arora, A., Icard, T., Potts, C., & Goodman, N. (2024). *A reply to Makelov et al. (2023)'s "interpretability illusion" arguments* [Preprint]. arXiv. https://arxiv.org/abs/2401.12631

Wu, Z., Geiger, A., Icard, T., Potts, C., & Goodman, N. (2023). Interpretability at scale: Identifying causal mechanisms in Alpaca. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2305.08809

Yeh, C.-K., Kim, B., Arik, S. O., Li, C.-L., Pfister, T., & Ravikumar, P. (2020). On completeness-aware concept-based explanations in deep neural networks. In *Advances in Neural Information Processing Systems 33*. https://arxiv.org/abs/1910.07969

Yuksekgonul, M., Wang, M., & Zou, J. (2023). Post-hoc concept bottleneck models. In *International Conference on Learning Representations*. https://arxiv.org/abs/2205.15480

Zhang, F., & Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. In *International Conference on Learning Representations*. https://arxiv.org/abs/2309.16042

Zhou, B., Sun, Y., Bau, D., & Torralba, A. (2018). Interpretable basis decomposition for visual explanation. In *Computer Vision – ECCV 2018* (pp. 122–138). Springer. https://doi.org/10.1007/978-3-030-01237-3_8
