# Phase 2 — Bibliography, T5: sparse dictionaries across layers and the residual

Theme: SAEs, transcoders, crosscoders, cross-layer transcoders (CLTs), SAE dark matter /
reconstruction error, supervised and hybrid dictionaries.

## Search strategy

- Sources: arXiv API (metadata for every ID), transformer-circuits.pub (fetched), ICLR/ICML/NeurIPS/PMLR
  proceedings and virtual pages (via web search), CVF open access. Project `PAPERS.md` /
  `paper/references.bib` reused for Dunefsky [101] and Marks [102].
- Keywords: sparse autoencoder; transcoder; skip transcoder; sparse crosscoder; cross-layer transcoder;
  multi-layer SAE; Jacobian SAE; SAE dark matter; reconstruction error; supervised sparse dictionary;
  concept bottleneck SAE; supervised SAE; SAE vs probe baselines; canonical units.
- Seeds from 01_scoping.md, then one backward/forward pass from Makelov, Engels and Karvonen (board games).
- Verification rule: title, authors, year, ID checked on arXiv API or the publisher page; venue checked
  on proceedings/virtual page or, for TMLR/ICML-SAEBench, on the authors' arXiv comment (flagged).

## Flow

```
Records identified: 22 (arXiv API 19, transformer-circuits.pub 3)
Duplicates removed: 0
Screened: 22
Excluded on scope: 7
  - Gao et al. 2406.04093 (TopK scaling; training recipe, no attribution/residual content beyond others)
  - Karvonen, Rager et al. SAEBench 2503.09532 (ICML 2025; LM benchmark, superseded here by Karvonen board-game paper)
  - Paulo & Belrose 2501.16615 (preprint; seed instability — overlaps Leask et al., which is peer-reviewed)
  - Bussmann et al. Matryoshka SAEs 2503.17547 (architecture variant; no bearing on fixed vocabulary)
  - Klotz et al. 2606.24716 (ECCV 2026; latent-to-concept matching for vision SAEs; candidate for T4/naming residual)
  - Marks et al. 2403.19647 (verified, ICLR 2025; error nodes belong to T6)
  - Poupart 2406.04028 (verified, PAPERS [39]; belongs to T8)
Excluded on verification failure: 0
Included: 15
```

Tier key (01_scoping.md): 1 peer-reviewed; 2 transformer-circuits / lab report; 3 arXiv preprint.

---

## A. Unsupervised dictionaries at one layer (baseline objects)

**1. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2024). Sparse autoencoders find highly interpretable features in language models. In *International Conference on Learning Representations (ICLR 2024)*. arXiv:2309.08600. https://arxiv.org/abs/2309.08600** — Tier 1
Note: ICLR proceedings list the authors as Huben, Cunningham, Smith (= Riggs), Ewart, Sharkey; arXiv order used here.
Defines the residual-stream SAE: encode h -> sparse z (ReLU, L1), decode z -> ĥ, train on MSE. Shows SAE
latents localise the IOI counterfactual behaviour better than PCA/ICA directions via activation patching.
For lc0: the unsupervised mirror of our setup. Our decoder D(f) is an SAE whose code f is fixed by the
Rust extractor, not learned. Axes: entry point none (dictionary-matched only after naming); direction both;
single layer; sparse-linear; attribution by patching. Weakness: the reconstruction error is discarded,
and the paper gives no criterion for whether latents match an external vocabulary.

**2. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N. L., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Tamkin, A., Nguyen, K., McLean, B., . . . Olah, C. (2023, October 4). Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features/index.html** — Tier 2, not peer-reviewed
(24 authors; APA 7 truncation after 19, then last author.)
One-layer transformer, MLP activations, SAE with up to ~131k latents; introduces feature splitting
(latents subdivide as dictionary grows) and loss-recovered as a fidelity metric (splice ĥ back in,
measure downstream loss). For lc0: "splice the decoded token back in and measure value/policy change" is the
right fidelity test for our D(f), stronger than FVU. Feature splitting predicts our granularity problem:
the 30k bits are a fixed, very fine split, so per-bit attribution inherits the instability of fine
splits. Weakness: single layer, toy-scale model, evaluation mostly qualitative.

## B. Maps between layers: transcoders, crosscoders, CLTs, multi-layer SAEs

**3. Dunefsky, J., Chlenski, P., & Nanda, N. (2024). Transcoders find interpretable LLM feature circuits. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. arXiv:2406.11944. https://arxiv.org/abs/2406.11944** — Tier 1 (PAPERS [101])
A transcoder replaces an MLP sublayer: encode its input sparsely, decode to its output. Because the map is
input -> output, attributions factor into an input-invariant term (weights between transcoder features)
and an input-dependent term (activation), giving weight-based circuits through MLPs. For lc0: the template
for a *supervised transcoder* — encode h_L to features f (our probe), decode f to h_{L+1} (a decoder targeting the
next layer, not the same one). That separates "what is present" from "what the block computes from it".
Axes: direction both; span one block; sparse-linear; attribution = weight × activation. Weakness: does
not cover attention; error term per sublayer still has to be carried as a node.

**4. Paulo, G., Shabalin, S., & Belrose, N. (2025). *Transcoders beat sparse autoencoders for interpretability* (arXiv:2501.18823) [Preprint]. arXiv. https://arxiv.org/abs/2501.18823** — Tier 3
Same model and data: transcoder latents score higher on automated interpretability than SAE latents.
Adds the *skip transcoder* (affine skip connection), lowering reconstruction loss at no interpretability
cost. For lc0: the skip term is a principled place for the part of h_{L+1} that is a linear function of
h_L and needs no feature — separating "carried forward" from "computed from features", which matters given
our FVU decay with depth. Weakness: preprint; interpretability judged by LLM-scored explanations, not
by interventions.

**5. Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., & Olah, C. (2024, October 25). Sparse crosscoders for cross-layer features and model diffing. *Transformer Circuits Thread*. https://transformer-circuits.pub/2024/crosscoders/index.html** — Tier 2, not peer-reviewed
A crosscoder reads from and writes to several layers with one shared latent set; argues for
"cross-layer superposition" — a feature smeared over many layers' residual streams — which per-layer
dictionaries count repeatedly. Also used to diff models. For lc0: directly answers "several decoders on
several layers": one shared f decoded to all of L0-L9 via per-layer decoder matrices, so attribution
to a feature is one number, not ten that must be reconciled. Our ridge decoders are a supervised,
per-layer-independent crosscoder; fitting them jointly is the cross-layer version. Weakness: acausal
variant ignores layer order, so it does not say where a feature is computed; not peer-reviewed.

**6. Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., . . . Batson, J. (2025, March 27). Circuit tracing: Revealing computational graphs in language models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html** — Tier 2, not peer-reviewed
(27 authors; APA 7 truncation.)
Cross-layer transcoder: a feature at layer ℓ reads the residual at ℓ (JumpReLU) and has separate
decoders into the MLP outputs of ℓ and every later layer. Error nodes absorb what the CLT misses, so the
local replacement model reproduces the original exactly; attribution graph edges are linear given frozen
attention. Reports completeness 0.80 / replacement 0.61 and next-token agreement ~50% without error
correction for the 10M-feature CLT. For lc0: the closest existing design to "per-layer supervised
decoder + error node at each layer", and its completeness/replacement scores are directly the
"what the vocabulary misses" number the user wants. Axes: cross-layer; sparse-linear; gradient/linear
edges with frozen attention; residual = error node in graph. Weakness: attention frozen, so the "deep
jungle of attention" is held fixed rather than attributed; needs a differentiable lc0 port.

**7. Lawson, T., Farnik, L., Houghton, C., & Aitchison, L. (2025). Residual stream analysis with multi-layer SAEs. In *International Conference on Learning Representations (ICLR 2025)*. arXiv:2409.04185. https://arxiv.org/abs/2409.04185** — Tier 1
One SAE trained on residual vectors from every layer. A given latent is usually active at a single layer
per token, but which layer varies across tokens; multi-layer activity grows with model size; tuned-lens
alignment of bases changes little. For lc0: a shared dictionary across L0-L9 (or a shared supervised D
with a layer one-hot, mirroring our square one-hot) tests whether a feature's representation is
layer-stable; our FVU table suggests it is not past L2. Axes: cross-layer shared dictionary; residual
ignored. Weakness: descriptive; no output attribution.

**8. Farnik, L., Lawson, T., Houghton, C., & Aitchison, L. (2025). Jacobian sparse autoencoders: Sparsify computations, not just activations. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 16178–16236). arXiv:2502.18147. https://arxiv.org/abs/2502.18147** — Tier 1
Paired SAEs on an MLP's input and output, trained so the Jacobian between their latents is sparse; uses
k-sparse encoders to make the Jacobian tractable. Jacobians are sparser on trained than random LLMs.
For lc0: makes "∂(output feature)/∂(input feature)" a trained-for object. With fixed f on both sides
(probe at L, decoder at L+1), the Jacobian through a block is a feature-to-feature attribution matrix, and
JSAE shows how to estimate it cheaply and what sparsity to expect. Axes: direction both; one block;
attribution operator = Jacobian; identifiability improved by the sparsity prior. Weakness: MLP only;
gradient-based, so binary-feature off-manifold objection (DA Major 2) applies.

## C. The residual: dark matter and its structure

**9. Engels, J., Riggs, L., & Tegmark, M. (2025). Decomposing the dark matter of sparse autoencoders. *Transactions on Machine Learning Research*. arXiv:2410.14670. https://arxiv.org/abs/2410.14670** — Tier 1 (venue per authors' arXiv record)
SAE error ("dark matter") is not noise: about half the error vector and >90% of its norm is linearly
predictable from the input activation; per-token error scales predictably with SAE size. The remaining
"nonlinear" error holds few learnable features yet costs a proportional share of downstream loss; linear
maps from earlier-layer SAE outputs reduce it most. For lc0: gives the decomposition we should apply to
E − D(f) at each layer: (i) linear-in-h part (re-fit as a linear readout), (ii) nonlinear part
(attribute as an error node). The "earlier-layer SAE output" result supports our multi-layer design:
decoders at L-1 can explain part of layer L's residual. Weakness: LM-only; "linearly predictable from h"
is not "nameable", so it does not by itself name the missing concepts.

**10. Leask, P., Bussmann, B., Pearce, M., Bloom, J., Tigges, C., Al Moubayed, N., Sharkey, L., & Nanda, N. (2025). Sparse autoencoders do not find canonical units of analysis. In *International Conference on Learning Representations (ICLR 2025)*. arXiv:2502.04878. https://arxiv.org/abs/2502.04878** — Tier 1
SAE stitching shows smaller SAEs are incomplete (larger ones contain novel latents); meta-SAEs show larger
latents decompose into smaller ones, so neither size is atomic. For lc0: the unsupervised route to naming
the residual has no unique answer — dictionary size is a free choice — which weakens "the residual
contains concept X" claims. It also argues for our fixed vocabulary as a reference frame, with SAEs on
E − D(f) as an exploratory, size-swept add-on. Axis: identifiability (non-unique). Weakness: LM-only;
no intervention-based ground truth.

## D. Supervised and hybrid dictionaries, and SAEs against supervised baselines

**11. Makelov, A., Lange, G., & Nanda, N. (2025). Towards principled evaluations of sparse autoencoders for interpretability and control. In *International Conference on Learning Representations (ICLR 2025)*. arXiv:2405.08366. https://arxiv.org/abs/2405.08366** — Tier 1
Builds supervised sparse dictionaries from known task attributes (IOI: names, positions) — each attribute
value gets a learned vector, activations are reconstructed as their sum — and uses them as the yardstick
for SAEs on sufficiency, necessity and edit control. Newer SAEs (gated, TopK) approach the supervised
dictionary. For lc0: this is the published form of our decoder D(f) (attribute-value dictionary with
additive reconstruction) and its evaluation protocol (patch reconstructions, edit one attribute, check
output). Axes: entry point = input side; direction decode; single layer; linear; patching/editing;
residual reported as reconstruction quality. Weakness: small attribute set on one LM task; does not
address 30k correlated bits or cross-layer chaining.

**12. Karvonen, A., Wright, B., Rager, C., Angell, R., Brinkmann, J., Smith, L., Mayrink Verdun, C., Bau, D., & Marks, S. (2024). Measuring progress in dictionary learning for language model interpretability with board game models. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://doi.org/10.52202/079017-2644 (arXiv:2408.00113)** — Tier 1
Uses chess and Othello LMs, where board-state properties ("knight on f3") are known, to score SAEs with
supervised metrics: coverage of the board-state feature set and board reconstruction from latents.
Introduces p-annealing. For lc0: the nearest precedent to our setting — per-square human features as
ground truth for a dictionary in a chess model. Its coverage metric is our probe-route score; our
decoder-route FVU is the converse measure. Gap it leaves: it scores SAE latents against features, not
output attribution, and its feature set is small (board state), unlike our 30k-bit vocabulary.

**13. Kulkarni, A., Weng, T.-W., Narayanaswamy, V., Liu, S., Sakla, W. A., & Thopalli, K. (2026). Interpretable and steerable concept bottleneck sparse autoencoders. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026)*. arXiv:2512.10805. https://arxiv.org/abs/2512.10805** — Tier 1
Hybrid dictionary: prune low-utility SAE latents, then add a concept bottleneck tied to a user-defined
concept set, so the latent space = supervised concepts + retained unsupervised latents. Motivation: user
concepts are often absent from SAEs. For lc0: the exact "partly supervised" design — our f as fixed latents,
plus a small learned sparse code for E − D(f). Attribution then splits into named features and unnamed
latents, each interveneable, which turns the residual from one error node into inspectable units.
Axes: entry point bottleneck + dictionary-matched; both directions; residual decomposed further.
Weakness: vision/LVLM only; metrics are its own; no cross-layer version.

**14. El Harzli, O., Wallner, H., Nam, Y., & Tao, H. X. (2026). *Supervised sparse auto-encoders for interpretable and compositional representations* (arXiv:2602.00924) [Preprint]. arXiv. https://arxiv.org/abs/2602.00924** — Tier 3
Decoder-only supervised SAE: jointly learn sparse concept embeddings and decoder weights to reconstruct
feature vectors, borrowing unconstrained-feature models from neural-collapse theory; no L1 penalty.
Shows compositional generalisation and feature-level edits on Stable Diffusion 3.5 prompt embeddings.
For lc0: same structure as our D(f) (decoder only, code fixed by supervision) and a test we have not run:
decode unseen feature *combinations* and check the token. Weakness: preprint, diffusion-embedding domain,
no attribution to model outputs.

**15. Kantamneni, S., Engels, J., Rajamanoharan, S., Tegmark, M., & Nanda, N. (2025). Are sparse autoencoders useful? A case study in sparse probing. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 29018–29049). arXiv:2502.16681. https://arxiv.org/abs/2502.16681** — Tier 1
Under data scarcity, class imbalance, label noise and covariate shift, SAE-latent probes do not beat
plain activation probes consistently; SAE-plus-baseline ensembles do not beat baseline ensembles.
For lc0: when a vocabulary exists, supervised maps are the stronger default; SAEs earn a place only on
what the vocabulary misses. Supports the scoping decision (SAEs as residual-naming tool only).
Weakness: probing tasks, not attribution; LM-only.

---

## Coverage against taxonomy axes

| Axis | Filled by |
|---|---|
| Entry point: input side (decoder) | 11, 14 |
| Entry point: bottleneck / hybrid | 13 |
| Entry point: dictionary-matched | 1, 2, 12 |
| Layer span: cross-layer | 5, 6, 7 |
| Layer span: one block (input -> output) | 3, 4, 8 |
| Attribution: Jacobian / linear edges | 6, 8, 3 |
| Residual: error node | 6 |
| Residual: decomposed further | 9, 13 |
| Identifiability | 10 (non-unique), 8 (sparsity prior) |
| Faithfulness: patching / loss recovered | 1, 2, 11 |
| Chess precedent | 12 |

Gaps: no paper trains a supervised *cross-layer* dictionary (a supervised CLT/crosscoder) or reports an
error-node share for a fixed external vocabulary. That is the open slot for this project.
