# Attributing a Chess Transformer's Outputs to a Fixed Human Feature Vocabulary: Decoders, Probes, and What the Vocabulary Misses

xai-chess project

## Abstract

Explaining a network in chess terms needs a map from concepts to activations and a removal rule. For the Leela Chess Zero (lc0) transformer, ridge decoders from 30,837 rule-computed per-square features to each square token leave 0.002 of token variance unexplained at the embedding and 0.680 at the last block. This paper asks which methods turn such maps into attributions of lc0's value and policy outputs, and what the vocabulary leaves unnamed. Only the fits, bit counts and the exported graph were measured. Methods and outcomes are proposals from a survey of 89 works. The embedding is local to each square, so decoder gradients, integrated gradients and feature occlusion there all reduce to input saliency in feature names. At later layers, patching the decoded change of a real board edit attributes that edit, a named/residual split measures coverage, and four compositions combine layers. Only an edit's joint effect is identifiable, not its split among collinear feature families. The recommended first experiment runs both methods on about 1,000 pin-edit pairs at five layers with random-initialisation and ridge controls, costs an estimated two CPU hours, and has numeric decision rules fixed in advance.

*Keywords:* concept-based explanation, activation patching, causal abstraction, Shapley values, identifiability, mechanistic interpretability, Leela Chess Zero

## 1. Introduction

A chess network's evaluation can be explained in chess terms only through a map from its internal state to those terms. The xai-chess project has two such maps. A Rust extractor (`features/square/`) emits for every square a binary vector of 30,837 bits: occupant, pawn structure, x-rays, threats, routes, piece roles, castling rights on the king, and mate and material lines computed with an engine. About 36.5 bits are on per square. On 20,000 Lichess puzzle positions, split 80/20 by position, probes read these features from lc0's square tokens ($d=256$), and decoders predict the token from the features plus a one-hot square identity. Both were fitted at the embedding and at each of the ten encoder blocks.

Attribution of an output to features is easiest to define where the decoder fits well and where the path to the output is short. On lc0 these places sit at opposite ends of the network. The project's working note frames this as a dilemma. At the back, reconstruction is near-perfect but mostly square identity, and attribution must pass through "the deep jungle of attention". At the front, attribution is clearer. The note asks whether a hybrid helps. The measured fits at the two ends are as follows. Square identity alone leaves FVU 0.164 at the embedding and 0.897 at block 9. Features plus square identity leave 0.002 and 0.680. Table 2 gives every tap.

This paper answers one question. *Which methods can attribute lc0's value and policy outputs to a fixed per-square feature vocabulary through learned feature–activation maps, and how do they trade off faithfulness under intervention, identifiability, residual coverage and compute on this system?* The question has three parts, each posed in the working note: (a) attribution through the decoder at the embedding, (b) attribution through decoders and encoders at any layer, and (c) methods that combine several maps across layers. A fourth request is a first experiment.

The maps alone do not give an attribution. A probe shows that information is present at a tap, not that lc0 uses it [@ravichander2021], [@elazar2021], and a decoder fit says how much token variance the vocabulary explains, not how much of an output change runs through it. Attribution needs a removal rule and an output [@covert2021]. The project gives priority to what the vocabulary misses, so every method below reports a residual beside its named part.

The contribution is an assembly of existing pieces for this system, not new theory.

1. A reduction: at the embedding, decoder gradients, integrated gradients and feature occlusion are all input saliency in feature names (Sections 5.2 and 6).
2. One token decomposition, $h_L=q_L+\sum_G n^G_L+\varepsilon_L$, read by five removal operators at any tap, with one coverage definition. For complete pairwise methods the usual faithfulness error coincides with the residual share, so coverage replaces it (Sections 7 and 9.2).
3. Four cross-layer compositions with their costs, including a feature-path composition that keeps attention live (Section 8).
4. A statement of what is identifiable on this vocabulary (Section 9.1).
5. A protocol for preregistration, with a first experiment of about two CPU hours and numeric decision rules (Section 10).

The measured inputs are the existing fits and extractor bit counts (Tables 1, 2 and 5). Everything else is analysis or hypothesis. Section 2 gives the answer, and Sections 3 to 10 the evidence.

## 2. The Answer in Brief

### 2.1 (a) Through the embedding decoder

The embedding computes each square's token from that square's 112 input values alone, through an elementwise nonlinearity (Section 5.2, read from the ONNX graph). A decoder that fits it at FVU 0.002 shows that the vocabulary contains occupancy. Three forms of attribution run through it: the decoder gradient (M1a), integrated gradients between real boards (M1b) and occlusion of feature families or bits (M1c). All three are input saliency on a square's planes, renamed into features. Non-local families such as x-rays and routes receive spurious credit at the embedding that the residual cancels exactly, so the admissible vocabulary there is local (control C2). Under C2 the named share is predicted to be near 1 by construction (H1). Whether the 0.002 reflects anything learned is a random-initialisation check (C1). See Section 6.

### 2.2 (b) Through decoders and encoders at any layer

Decoded substitution (M2) patches the decoded change of a real board edit into the token at tap $L$ and runs the rest of the network. Its consistency gap $\gamma_L$ measures how far the decoded edit is from the real one. The named/residual split (M6) symmetrises M2 into an exact two-player Shapley value. Its aggregate $\rho_L$ is a two-player attribution and a coverage measure: it does not say which family moved the output, and it guides where per-family attribution (M2, M9) is worth running. The encoder direction tests whether readable content is used, by LEACE erasure (M3), a family-level concept gradient (M4) and interchange alignment (M5). A feature surrogate (M0) is the correlational floor. A policy logit depends on two tokens at the heads' input, which permits an exact four-player split. See Section 7.

### 2.3 (c) Combining layers

The M6 sweep compares independent decoders on one output change. The erasure cascade (M8) intervenes at one layer and reads later layers. Feature paths (M8∘M2) substitute at one layer, read the change at a later layer through that layer's probe and decoder, and push each part to the output with attention live. A supervised replacement graph (M7) chains block-level transcoders with error nodes but freezes attention. A residual budget (M12) splits what remains by cause, and M11 compares methods without fusing them. See Section 8.

### 2.4 Identifiability

The joint effect of a real edit, the consistency gap and the named share are identifiable given the decoder. Its split among collinear families is not. A pin edit changes over 300 bits in 12 or 13 of the 28 families (Table 5). Section 9 gives the remedy and the coverage definition.

### 2.5 First experiment

Run M2 and M6 on about 1,000 pin-edit pairs at the embedding (local vocabulary), L2, L5, L8 and L9, with C1 on three random networks, a ridge sweep (C3) and M0 on the same pairs. It costs an estimated two CPU hours under ONNX, refits included. Numeric decision rules fixed in advance promote M8 and M8∘M2, or M5, or M12 (Section 10.5).

**Table A**
*Methods by sub-question: the question each answers, its residual, and its cost*

| Part | Method | Question answered | Residual reported | Identifiable | Cost (ONNX unless "port") |
|---|---|---|---|---|---|
| (a) | M1a decoder gradient at $e$ | which square inputs, named, move $y$ | $\langle\nabla y,\varepsilon_e\rangle$ | joint only | port; 1 backward pass per position |
| (a) | M1b board-path IG at $e$ | the same, complete over a real pair | $\mathrm{IG}_\varepsilon$ | joint only | port; about 32 backward passes |
| (a) | M1c feature occlusion at $e$ | effect of replacing a family's decoded share | $g_e(q_e)-g_e(q_e+\varepsilon_e)$ | joint only | $m{+}1$ passes per position |
| (b) | M0 feature surrogate | which families predict $y$ | $y-s$ | no (regulariser) | minutes |
| (b) | M2 decoded substitution | effect of the decoded edit; consistency | $\gamma_L$ | joint yes | $2k{+}1$ suffix passes per tap |
| (b) | M6 named/residual split | named share of a real change | $\phi^{(L)}_\varepsilon$, $1-\rho_L$ | yes, given $D_L$ | 1 pass per tap beyond M2 |
| (b) | M9 family Shapley + error | per-family share of a change | $\varphi_\varepsilon$ | joint only | about 30 s per position per tap |
| (b) | M10 relevance chain | the M9 split from one relevance pass | $R^\varepsilon_L$ | rule-dependent | port; 1 modified backward pass |
| (b) | M3 LEACE erasure | is $G$'s linear content used downstream | output after erasing all families | yes | 1 affine patch |
| (b) | M4 concept gradient | sensitivity of $y$ to family probe coordinates | $\|\rho^\perp\|^2/\|\nabla y\|^2$ | min-norm only | port; 1 backward pass |
| (b) | M5 interchange alignment | does a subspace mediate $G$ | $1-$IIA | up to rotation | port; hours |
| (c) | M8 erasure cascade | where named content is built or lost | $\Delta\phi_\varepsilon$ | as M3 or M2 | 1 pass per $(G,L)$ |
| (c) | M8∘M2 feature paths | through which families at $L'$ the effect reaches $y$ | $\pi_{G\to r}$ | joint only | hundreds of passes per pair |
| (c) | M7 replacement graph | along which routes, with attention frozen | error nodes | regulariser-dependent | port; seconds per position |
| (c) | M12 residual budget | why the residual exists; is it nameable | $\eta_L$ | no (SAE size) | CPU hours |
| (c) | M11 disagreement map | do methods and taps agree | coverage spread | nothing fitted | negligible |

*Note.* $k$ is the number of changed families, $m=28$ the number of families. "Joint only" means the joint effect of an edit is identifiable and per-family splits are not. Full taxonomy in Appendix C, method definitions in Appendix A.

Controls C1 to C9 are listed in Appendix B. Sections 3 to 5 give the setting and the constraints these answers rest on.

## 3. System, Data and Notation

### 3.1 Model, taps, inputs and targets

The network is lc0 `t1-256x10-distilled-swa-2432500`: 64 square tokens of width 256, ten post-LayerNorm encoder blocks with eight heads and smolgen attention biases, and policy, value and moves-left heads. It runs as ONNX on CPU at about 180 positions per second. The embedding tap is `/attn_body/rehape`. Block taps L0 to L9 are each block's first LayerNorm output, `/encoder{i}/ln1`, taken after attention and its skip connection and before the feed-forward network. The heads read `/encoder9/ln2`. L0 is the first tap at which a token can carry information from other squares. The ONNX Runtime inference build used here has no backward pass, so gradient methods need a PyTorch port.

Positions are Lichess puzzle FENs, which carry no move history. The input planes therefore use lc0's fen-only history fill (`lc0-probe/lc0_planes.py`). All eight history frames hold the current board, an en-passant pawn is moved back in the older frames, and the repetition planes are zero. Castling rights (four planes), side to move, the rule-50 count and two constant planes are broadcast to all 64 squares. lc0 sees the board from the side to move, and the labels follow it: black-to-move positions are colour-mirrored before extraction (`lc0-probe/square_probe.py`), so "white" in a bit name means the side to move. The plane that tells lc0 that Black is to move (`lc0_planes.py`, plane 108) is carried by no feature. The extractor puts castling rights on the king's square (`king.castle`, commit 473b3ce), and the decoder fits postdate that commit.

The value target is $y=\ell_W-\ell_L$, the difference of the win and loss logits entering the WDL softmax at `/value/dense2`, from the side to move's view. The policy target is the pre-softmax logit of one move (Section 7.4).

**Table 1**
*Existing results other than the per-tap decoder fits (held-out positions, own measurements)*

| Result | Value |
|---|---|
| Linear probe, token at layer $L$ to its own features | most families near-perfectly readable at the embedding; `pawn_cover`, strong squares, mate and passed pawn gain 0.04–0.14 by L5 |
| MLP decoder, features + square to token | trend similar to ridge (Table 2) |
| Board-level decoder, all 64 squares' features to value-head activations and value output | test FVU 0.52–0.62 at 151k positions, still falling with data |
| Value head applied to decoded L9 tokens | test FVU 0.81 at `/value/dense2` (0.88 at `/value/dense1`), against 1.54 (1.34) from square identity alone; 3,960 held-out of 20,000 positions |
| MLP probe for "pin is relevant" | peak AUROC 0.945 at block 8, against 0.839 for Stockfish NNUE |

*Note.* Per-square decoders keep the 14,565 bits with at least 20 positive training tokens. The board-level decoder keeps 17,831 bits.

The project also has exact causal operators in a patched Stockfish. MaskPinner removes the rule-level effect of a pin, and LineBlocker places a ray-selective phantom blocker on a line. They supply interventional ground truth about chess, kept separate from ground truth about lc0 (Section 10.1).

### 3.2 Notation and the canonical split

Let $x$ be a board, $x'$ a real edit of it, and $x_0$ a baseline board from a pool $\mathcal B$ of real positions. The extractor gives $f=\phi(x)\in\{0,1\}^{64\times K}$, with $f_s$ the bits of square $s$ and $\tilde f_s=f_s-\bar f_s$ their per-square centring. Bits are grouped into $m=28$ families $\mathcal G=\{G_1,\dots,G_m\}$, the extractor's top-level name prefixes (`occupant`, `route`, `xray`, …). "Family" means this grouping throughout. The token at tap $L\in\{e,0,\dots,9\}$ is $h_L\in\mathbb R^{64\times256}$, and the suffix network from $L$ to the output is $g_L$. A *decoder* maps features to a token:

$$D_L(f_s,s)=q_{L,s}+W_L\tilde f_s,$$

with $q_{L,s}$ the per-square mean. A *probe* $P_L:h_{L,s}\mapsto\hat f_s$ maps a token to features. Probe-side methods form the *encoder direction*. In that phrase "encoder" means a probe, never an lc0 encoder block. Writing $n_{L,s}=W_L\tilde f_s$ for the named part, $n^G_{L,s}=W^G_L\tilde f^G_s$ for family $G$'s share, and $\varepsilon_{L,s}=h_{L,s}-D_L(f_s,s)$ for the residual, every token splits as

$$h_L=q_L+\sum_G n^G_L+\varepsilon_L.$$

The residual is the error node: what the vocabulary misses at $L$ on $s$. An *attribution* assigns a share of an output quantity to the players $n^G$ and $\varepsilon$ under a stated removal rule.

Because $q_{L,s}$ depends only on $s$, it is identical for $x$ and $x'$ and cancels in every two-board comparison. In the point gradient form (M1a) it does not cancel and is left unassigned. Write $\Delta^{\rm true}=y(x')-y(x)$ for lc0's response to a real edit. Pairs are *oriented* so that $\Delta^{\rm true}\ge0$: if an edit lowers $y$, $x$ and $x'$ are swapped.

With the notation fixed, Section 4 states what the literature allows these maps to support.

## 4. What the Literature Constrains

### 4.1 The removal rule decides what an attribution means

Covert et al. [@covert2021] classify 26 explanation methods by how they remove features. Conditional, interventional and baseline Shapley values give different answers under the same axioms [@sundararajan2020]. Janzing et al. [@janzing2020] argue for interventional removal, Frye et al. [@frye2021] show that off-manifold removal misleads, and Chen et al. [@chen2020] reconcile the two: interventional removal is true to the model, conditional removal true to the data. I. E. Kumar et al. [@kumarIE2020] show that both Shapley variants fail as importance measures.

This vocabulary admits three removal rules. Flipping bits of $f$ and re-decoding (M1c at the embedding) explains lc0 composed with the decoder, usually on a vector no board produces. Substituting the decoded named part of a real edited board explains lc0 at layer $L$ if the decoder is right. Its endpoints are real and its mixtures are not. A real board edit explains lc0 itself and stays on the data manifold, but it changes many families at once. Aas et al. [@aas2021] give Shapley values over groups as the remedy for dependent features, and Lundstrom et al. [@lundstrom2022] give distributional baselines and internal-neuron integrated gradients (IG). Bilodeau et al. [@bilodeau2024] prove that complete, linear attribution methods can do no better than random at inferring model behaviour for rich model classes, and argue that attributions must be tied to a concrete end-task. Here the end-task is a board edit.

### 4.2 Readable is not used

Irrelevant properties stay probe-readable [@ravichander2021], probe accuracy must be read against control tasks [@hewitt2019], [@belinkov2022], and erasure finds high probe accuracy with small behavioural effect [@elazar2021]. Interchange interventions [@geiger2020] and concept bottlenecks [@margeloiu2021] show the same gap. On chess agents, Pálsson and Björnsson [@palsson2024] find linear-probe accuracy an unreliable proxy for concept importance and recommend nonlinear probes with erasure. Jenner et al. [@jenner2024] validate an lc0 probe by its causal effect. Counterfactual benchmarks, CEBaB [@abraham2022], RAVEL [@huang2024] and MIB [@mueller2025], find that many correlational methods fail. Readability at the embedding therefore says how strongly the bits depend on occupancy, not whether lc0 uses them.

### 4.3 The residual is structured and reportable

Interpretable basis decomposition leaves 65.9% of a ResNet-50 prediction unexplained [@zhou2018]. Concept bottlenecks add a residual predictor [@yuksekgonul2023] or side channel [@havasi2022], and feature circuits and replacement models keep error nodes [@marks2025], [@ameisen2025]. Engels et al. [@engels2025] find that about half of a sparse autoencoder's (SAE) error is linearly predictable from the input activation, and that linear maps from earlier-layer SAE outputs reduce the nonlinear remainder most.

Naming the residual is contested. Residual concept bottlenecks match residual directions to candidates from a larger concept bank [@shang2024]. Leask et al. [@leask2025] show that unsupervised dictionaries have no canonical size, so "the residual contains concept X" depends on the dictionary. Schut et al. [@schut2025] are the only game-network attempt found to name what a human vocabulary misses, validated behaviourally rather than by intervention. Havasi et al. [@havasi2022] separate residual due to missing vocabulary from residual due to a map too weak to use the vocabulary.

### 4.4 With a fixed vocabulary, supervised maps are the default

Makelov et al. [@makelov2025] use supervised attribute dictionaries, in which an activation is a sum of per-attribute-value vectors, as the yardstick for SAEs. $D_L$ has this form. Kantamneni et al. [@kantamneni2025] and AxBench [@wu2025] find that supervised directions match or beat SAE latents for detection and steering. MIB finds distributed alignment search (DAS) strongest and SAE features no better than neurons [@mueller2025]. This paper therefore places SAEs on the residual of $D_L$.

### 4.5 Cross-layer maps exist only unsupervised

Transcoders map a block's input to its output [@dunefsky2024], [@paulo2025], crosscoders share latents across layers [@lindsey2024], and cross-layer transcoders write to all later layers with error nodes [@ameisen2025]. Multi-layer [@lawson2025] and Jacobian SAEs [@farnik2025] make layer stability and block Jacobians measurable. GCAV [@he2025] is the only concept-map method found that fuses layers. On lc0, Lin et al. [@lin2026] replace every MLP and attention module of a larger network with sparse layers, reach explained variance above 0.6 per module (lower for attention in middle layers), label features by hand and check them against rule-based detectors. No source fits a supervised cross-layer map, and all these designs assume an additive residual stream. lc0's taps are post-LayerNorm.

### 4.6 Identifiability limits every map

Sutter et al. [@sutter2025] prove that with unrestricted alignment maps any network abstracts any algorithm, and report near-perfect interchange intervention accuracy (IIA) on random networks. Beckers and Halpern [@beckers2019] give conditions for constructive abstraction that this vocabulary does not meet, since its bits are functions of overlapping sets of squares. Park et al. [@park2024] show that probe directions and decoder columns coincide only under a causal inner product. Concept activation vectors are inconsistent across layers and entangled [@nicolson2025]. Leakage has measurable diagnostics [@mahinpei2021], [@espinosa2023], [@parisini2026], [@huang2024]. Ramaswamy et al. [@ramaswamy2023] find human usefulness capped near 32 concepts. The sources converge on one prescription: attribute to about 30 families, keep linear maps primary, report MLP maps only beside a random-initialisation control, and report stability across regularisation.

### 4.7 Attention breaks gradient attributions

Gradient×Input violates conservation through attention and LayerNorm, which Ali et al. [@ali2022] restore by detaching both. AttnLRP gives softmax rules [@achtibat2024], and concept relevance propagation conditions on a latent subspace [@achtibat2023]. AtP* identifies the false negatives of linearised patching, attention saturation and cancellation between direct and indirect paths [@kramar2024]. Ameisen et al. [@ameisen2025] avoid the problem by freezing attention. On lc0 that is costly, because Jenner et al. [@jenner2024] use attention-head ablation in their evidence for look-ahead. Ferrando and Voita [@ferrando2024] give a forward-only route decomposition that would run under ONNX. No relevance rule set covers smolgen, which adds a position-dependent bias to the attention logits before the softmax (ONNX graph).

### 4.8 Depth profiles in game networks

McGrath et al. [@mcgrath2022] and Hammersborg and Strümke [@hammersborg2024] track concepts per layer in AlphaZero and lc0. Lomasov et al. [@lomasov2025] report human-concept probe accuracy falling from 85% early to 50–65% late in a different, 270M-parameter chess transformer. That is probe decay on another model, and the project's probes point the other way. Sandmann et al. [@sandmann2025] show lc0's late layers overriding correct mid-layer solutions, so per-layer readings must be chained to $y$.

The survey behind this section is described in Appendix E. Constraints 4.2 and 4.6 bear directly on the front/back numbers, which Section 5 re-derives.

## 5. The Front/Back Question Re-derived

### 5.1 Two normalisations of one table

Table 2 gives the decoder fits at every tap. "Features add" is a share of total token variance and is capped by the square-only FVU. Dividing by that cap gives the share of non-square variance the vocabulary explains, a partial $R^2$. Figure 1 plots both.

**Table 2**
*Ridge decoder fits at every tap (test FVU) and the variance the vocabulary explains (own measurement)*

| Tap | $e$ | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Square only | 0.164 | 0.575 | 0.818 | 0.828 | 0.832 | 0.825 | 0.841 | 0.849 | 0.871 | 0.879 | 0.897 |
| Features only | 0.156 | 0.239 | 0.212 | 0.205 | 0.282 | 0.321 | 0.403 | 0.436 | 0.454 | 0.555 | 0.685 |
| Features + square | 0.002 | 0.183 | 0.198 | 0.195 | 0.272 | 0.311 | 0.392 | 0.427 | 0.447 | 0.549 | 0.680 |
| Features add | 0.16 | 0.39 | 0.62 | 0.63 | 0.56 | 0.51 | 0.45 | 0.42 | 0.42 | 0.33 | 0.22 |
| Partial $R^2$ | 0.99 | 0.68 | 0.76 | 0.76 | 0.67 | 0.62 | 0.53 | 0.50 | 0.49 | 0.38 | 0.24 |

*Note.* From `lc0-probe/results_square_decoder.json` (ridge $\alpha=1$, 14,565 bits, 20,000 positions). Features add = square only − features + square; partial $R^2$ = features add ÷ square only. L$i$ = `/encoder{i}/ln1`.

![Figure 1](fig_decoder.png)

**Figure 1.** Ridge decoder fits by tap. (a) Test FVU for square identity only, features only, and features plus square identity. (b) Partial $R^2$, the share of non-square token variance the features explain. It peaks at L2 (0.765, marginally above L1's 0.758). Data as in Table 2, script `fig_decoder.py`.

The rise of features-add from $e$ to L2 mostly reflects the shrinking square-identity share. Partial $R^2$ is highest at $e$, 0.68 at L0 and 0.76 at L1 and L2, and it falls at every tap from L2 to L9. From L1 on, features alone fit nearly as well as features plus square. Neither normalisation picks a best layer.

### 5.2 The embedding is local and nonlinear

The exported ONNX graph gives the embedding per square as

$$h_{e,s}=g_s\odot\mathrm{mish}\big(W[\text{planes}_s;\ \text{posenc}_s]+b\big)+a_s,$$

where $\text{planes}_s$ are the square's 112 input values (13 planes for each of eight history frames, plus the broadcast planes), $\text{posenc}_s$ is a learned positional code, and $g_s,a_s$ are learned per-square gates. The map is elementwise-nonlinear and strictly local: nothing from other squares reaches $h_{e,s}$. Five consequences follow.

1. The 0.002 FVU says the vocabulary contains each square's occupant. Under fen-only fill the history frames copy the current board, so on this data they carry nothing beyond occupancy except the en-passant detail.
2. Broadcast content the vocabulary does not place on the square falls into $\varepsilon_e$: the rule-50 count and the side-to-move plane everywhere, and castling rights on every square except the king's.
3. A non-local family (x-ray, threat, route, mate line) enters $h_{e,s}$ only through correlation with the square's own planes. In a pairwise method at $e$, an edit changes non-local bits on squares whose planes did not change. The decoder assigns $W^G\Delta\tilde f^G_s\neq0$ there, and $\varepsilon_e$ cancels it exactly. The result is spurious named credit with an equal and opposite residual. At $e$, attribution must therefore use the local vocabulary (C2) or zero the decoded deltas on squares whose planes are unchanged, as in control-task practice [@hewitt2019].
4. If the C2 decoder also fits near 0.002, then $\varepsilon'_e-\varepsilon_e\approx0$ for edits that leave the broadcast planes alone, and the named share at $e$ is close to 1 by construction. The C2 decoder has not been fitted, so this is a prediction (H1), checked by a fit of seconds in Stage A0. If it holds, the front end of the coverage curve carries no information about what lc0 learned.
5. A random-initialised embedding is equally local. This paper predicts, but has not checked, that it also fits near 0.002. Control C1 tests it [@adebayo2018], [@sutter2025]. A fit near 0.002 would mean the embedding FVU says little beyond locality and occupancy. A clearly larger one would measure how far lc0 learned to down-weight the broadcast planes at $e$. On fen-only input the history frames cannot separate the two cases, since they duplicate occupancy.

### 5.3 Coverage against path length

At $e$ coverage is high and the causal path crosses all ten blocks. At L9 coverage is lower and the path crosses one feed-forward network and the head. The working note's dilemma is this trade-off, and Section 5.2 shows that the high front coverage is architectural.

### 5.4 Seven candidate causes of the decay

The fall in partial $R^2$ from L2 to L9 has at least seven candidate causes, to be separated before any part of $\varepsilon_L$ is called an unnamed concept.

- (a) *Non-locality*: $D_L$ sees only $f_s$, while attention moves information between squares.
- (b) *Map class*: a linear map is too weak. The MLP decoder's similar trend weakens this cause but does not exclude it.
- (c) *Genuinely unnamed content.*
- (d) *Dilution*: named content stays, as the probes show, while other content grows.
- (e) *Post-LN renormalisation*: a per-token rescaling that a linear map of $f_s$ cannot fit.
- (f) *Output-specific content*, such as policy content a value target never uses.
- (g) *Fit capacity*: 14,565 bits against about one million training tokens, the same at every tap, so (g) adds to the decay only if the bits that matter at depth are rare.

Causes (d) to (f) mean that FVU decay need not be decay in what explains $y$. Only methods in output units settle this: M6 at every tap, and M12, which splits the residual by cause.

Section 6 draws the consequences for the embedding.
