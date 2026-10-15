# Phase 2 — Bibliography T4: concept bottlenecks, supervised dictionaries, completeness, leakage

Search date 2026-10-09. Sources: arXiv API (metadata for every ID), PMLR, NeurIPS/ICLR/CVPR proceedings,
JMLR, ICML virtual site. No T4 entry was already in `PAPERS.md` / `paper/references.bib`.
Verification rule: title, authors, year, venue and ID confirmed on the arXiv or publisher page; anything
unconfirmed dropped.

Flow: 7 seeds + 11 candidates from search and citation chasing = 18 screened. 18 confirmed to exist.
3 excluded on scope (Kim et al. ProbCBM 2306.01574, Oikarinen et al. Label-free CBM 2304.06129: image-
specific concept sourcing; Kantamneni et al. 2502.16681: SAE-vs-probe comparison, handed to T5).
0 dropped for failed verification. 15 included.

Axis numbers refer to the taxonomy in `01_scoping.md` (1 entry point, 2 direction, 3 layer span,
4 map class, 5 attribution operator, 6 residual handling, 7 identifiability, 8 faithfulness test, 9 cost).

---

## A. Bottlenecks: the vocabulary replaces the activation

**1. Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). Concept bottleneck models. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 5338–5348).** https://proceedings.mlr.press/v119/koh20a.html (arXiv:2007.04612)
Tier 1. Input -> concepts -> label, trained independently, sequentially or jointly; test-time intervention =
overwrite a predicted concept with its true value and read the output change.
For lc0: the encoder route turned into an architecture (axis 1 bottleneck, axis 2 encode). Needs retraining
lc0, out of scope; its intervention protocol is the template for "set bit i, measure value change" (axis 8).
Weakness: assumes the concept set is complete; joint training trades concept fidelity for accuracy (see 3-5).

**2. Yuksekgonul, M., Wang, M., & Zou, J. (2023). Post-hoc concept bottleneck models. In *International Conference on Learning Representations (ICLR 2023)*.** https://arxiv.org/abs/2205.15480
Tier 1. Projects a frozen backbone's embedding onto concept vectors (CAVs or CLIP text), fits a sparse linear
predictor on concept scores; PCBM-h adds a residual predictor on the raw embedding to recover lost accuracy.
For lc0: closest published design to "probe at layer L, then explain the head through the probe outputs"
without retraining (axes 1, 4 sparse-linear, 3 single layer). PCBM-h is the residual-as-first-class-term the
user wants (axis 6: residual fit), and the share of output carried by the residual measures what the
vocabulary misses. Weakness: the residual is fit after the concepts, so credit split depends on fitting order
and regularisation (axis 7 non-unique); concept-to-output map is linear.

**3. Espinosa Zarlenga, M., Barbiero, P., Ciravegna, G., Marra, G., Giannini, F., Diligenti, M., Shams, Z., Precioso, F., Melacci, S., Weller, A., Lio, P., & Jamnik, M. (2022). Concept embedding models: Beyond the accuracy-explainability trade-off. In *Advances in Neural Information Processing Systems 35* (NeurIPS 2022).** https://proceedings.neurips.cc/paper_files/paper/2022/hash/867c06823281e506e8059f5c13a57f75-Abstract-Conference.html (arXiv:2209.09056)
Tier 1. Each concept becomes a learned embedding (mixture of a "true" and "false" vector weighted by the
concept probability), keeping accuracy when the concept set is incomplete while still allowing interventions.
For lc0: a per-bit embedding is structurally what a decoder D(f) already does (feature -> vector in d=256);
CEM shows high-dimensional concept channels buy accuracy by carrying extra information, i.e. leakage
(axis 6 hidden inside concept vectors). Weakness: attribution per concept is no longer a scalar; the extra
capacity is exactly what items 6-8 warn about.

**4. Chen, Z., Bei, Y., & Rudin, C. (2020). Concept whitening for interpretable image recognition. *Nature Machine Intelligence, 2*, 772–782.** https://arxiv.org/abs/2002.01650
Tier 1. Replaces a batch-norm layer with whitening plus a rotation that aligns latent axes with given
concepts; the layer can be placed at any depth, and the paper reports how concept separability changes by depth.
For lc0: an in-model, per-layer aligned basis (axis 3 per-layer, axis 4 linear orthogonal); its depth sweep
is a precedent for the user's "which layer" question. Weakness: requires retraining; orthogonality forces
correlated concepts (knight occupancy vs knight-route bits) apart, which a 30k-bit dependent vocabulary
cannot satisfy in 256 dims.

**5. Sun, C.-E., Oikarinen, T., Ustun, B., & Weng, T.-W. (2025). Concept bottleneck large language models. In *International Conference on Learning Representations (ICLR 2025)*.** https://arxiv.org/abs/2412.07992
Tier 1. Inserts a concept bottleneck layer into LLMs for classification and generation, with an unsupervised
side channel for generation and concept-level steering.
For lc0: transformer precedent for a bottleneck with an explicit side channel (axis 6), i.e. a trained
version of "named features + residual". Weakness: retraining; concept labels come from an LLM, not a
verified extractor; text domain.

**6. Delfosse, Q., Sztwiertnia, S., Rothermel, M., Stammer, W., & Kersting, K. (2024). Interpretable concept bottlenecks to align reinforcement learning agents. In *Advances in Neural Information Processing Systems 37* (NeurIPS 2024).** https://proceedings.neurips.cc/paper_files/paper/2024/hash/7b76eea0c3683e440c3d362620f578cd-Abstract.html (arXiv:2401.05821)
Tier 1. SCoBots: successive bottlenecks (object properties, then relations between objects) before an
interpretable policy; used to find and fix a shortcut in Pong.
For lc0: the only game-agent bottleneck found; relational concepts are the analogue of the per-square
x-ray/threat/route bits. Stacked bottlenecks are a supervised multi-layer map (axis 3). Weakness: Atari,
object extractor given, policy deliberately small; no transfer evidence to a 10-layer transformer.

## B. Leakage: concept channels carry more than the concept

**7. Mahinpei, A., Clark, J., Lage, I., Doshi-Velez, F., & Pan, W. (2021). Promises and pitfalls of black-box concept learning models [Workshop paper]. ICML 2021 Workshop on Theoretic Foundation, Criticism, and Application Trend of Explainable AI.** https://arxiv.org/abs/2106.13314
Tier 3 (workshop). Shows soft concept outputs encode information beyond the concepts, so a downstream
predictor can use it even when concepts are irrelevant; natural fixes do not remove it.
For lc0: a ridge/MLP decoder from 30k bits, or a probe whose soft outputs feed the head, can route
non-concept information through "concept" coordinates. Attribution to a bit is then partly attribution to
leaked signal (axis 7). Weakness: small synthetic setups.

**8. Margeloiu, A., Ashman, M., Bhatt, U., Chen, Y., Jamnik, M., & Weller, A. (2021). Do concept bottleneck models learn as intended? [Workshop paper]. ICLR 2021 Workshop on Responsible AI.** https://arxiv.org/abs/2105.04289
Tier 3 (workshop). Saliency on CBM concept units does not localise to the input regions that define the
concept: high concept accuracy does not imply the unit encodes the concept.
For lc0: the bottleneck version of "readable is not used" (DA Major 4); a high-AUROC probe direction
is not thereby the concept. Weakness: saliency-based evidence, itself weak (T1).

**9. Havasi, M., Parbhoo, S., & Doshi-Velez, F. (2022). Addressing leakage in concept bottleneck models. In *Advances in Neural Information Processing Systems 35* (NeurIPS 2022).** https://proceedings.neurips.cc/paper_files/paper/2022/hash/944ecf65a46feb578a43abfd5cddd960-Abstract.html
Tier 1. Traces leakage to an insufficient concept set and an inexpressive concept predictor; fixes with hard
concepts, an autoregressive concept predictor and an explicit side channel; evaluates by intervention accuracy.
For lc0: separates the two failure causes the project must also separate: missing vocabulary (the residual,
axis 6) vs. a too-weak map (decoder FVU from model class, axis 4). The side channel is a named error node.
Weakness: requires training; side channel reduces intervenability by design.

**10. Espinosa Zarlenga, M., Barbiero, P., Shams, Z., Kazhdan, D., Bhatt, U., Weller, A., & Jamnik, M. (2023). Towards robust metrics for concept representation evaluation. In *Proceedings of the AAAI Conference on Artificial Intelligence, 37*.** https://arxiv.org/abs/2301.10367
Tier 1. Defines oracle impurity score and niche impurity score: how much a learned concept representation
predicts *other* concepts than its own.
For lc0: directly computable on the existing probes/decoders with the ground-truth bits; quantifies how
much a "pin" direction also encodes "attacked" or "occupant" (axis 7 diagnostic). Cheap (axis 9: no
gradients). Weakness: measures impurity with respect to the given concepts only, blind to unnamed ones.

**11. Parisini, E., Chakraborti, T., Harbron, C., MacArthur, B. D., & Banerji, C. R. S. (2026). Leakage and interpretability in concept-based models. *Journal of Machine Learning Research, 27*(174), 1–40.** https://www.jmlr.org/beta/papers/v27/25-1121.html (arXiv:2504.14094)
Tier 1. Information-theoretic concepts-task leakage and interconcept leakage scores; reported to predict
behaviour under interventions better than prior measures.
For lc0: concepts-task leakage asks how much of the value target a decoder/probe's concept channel carries
beyond the true bits; relevant to the engine-derived `mate`/`material` bits (DA minor issue). Interconcept
leakage is the dependence problem of axis 7. Weakness: mutual-information estimation on 30k binary bits is
hard; validated on bottleneck models, not post-hoc maps.

## C. Completeness: how much of the output the vocabulary explains

**12. Yeh, C.-K., Kim, B., Arik, S. O., Li, C.-L., Pfister, T., & Ravikumar, P. (2020). On completeness-aware concept-based explanations in deep neural networks. In *Advances in Neural Information Processing Systems 33* (NeurIPS 2020).** https://proceedings.neurips.cc/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html (arXiv:1910.07969)
Tier 1. Completeness score: accuracy of the model's head applied to activations reconstructed from concept
scores, normalised between random and full accuracy. ConceptSHAP: Shapley values over concepts with
completeness as the value function.
For lc0: the exact quantity for "value head on decoded layer-L tokens" (context result 3), now as a normalised
score per layer (axis 6: completeness reported). ConceptSHAP over feature *families* answers DA Major 1
(group-level attribution, axis 5 Shapley, axis 7 unique given the value function). Weakness: completeness
uses the reconstruction (off-manifold activations); Shapley over 2^k family subsets is exponential, feasible
only for a few dozen families.

**13. Shang, C., Zhou, S., Zhang, H., Ni, X., Yang, Y., & Wang, Y. (2024). Incremental residual concept bottleneck models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)* (pp. 11030–11040).** https://doi.org/10.1109/CVPR52733.2024.01049 (arXiv:2404.08978)
Tier 1. Res-CBM: optimisable residual vectors fill the gap left by an incomplete concept bank, then an
incremental module maps each residual vector to candidate concepts from a larger bank; adds a concept-
utilisation-efficiency metric.
For lc0: a recipe for *naming the residual*: fit residual directions on decoder error, then match them to
held-back bit families or new candidate features (axis 6: residual decomposed further). Matches the user's
"measure what the vocabulary misses". Weakness: naming only works if the bank contains the missing
concept; image-domain, CLIP-based.

**14. Shin, S., Jo, Y., Ahn, S., & Lee, N. (2023). A closer look at the intervention procedure of concept bottleneck models. In *Proceedings of the 40th International Conference on Machine Learning* (ICML 2023).** https://arxiv.org/abs/2302.14260
Tier 1. Compares concept-selection criteria and intervention levels for test-time concept interventions;
intervention gains depend strongly on training scheme and on concept dependence.
For lc0: guidance for "decoded-substitution patching" (flip bit, re-decode, run upstream): which bits to
intervene on first and why gains can be negative when bits are dependent (axis 8). Weakness: trained-CBM
setting; assumes interventions are on-distribution.

## D. Supervised dictionaries in transformers

**15. Makelov, A., Lange, G., & Nanda, N. (2025). Towards principled evaluations of sparse autoencoders for interpretability and control. In *International Conference on Learning Representations (ICLR 2025)*.** https://proceedings.iclr.cc/paper_files/paper/2025/hash/53356aebeea8ffd40a8ac3bb66243162-Abstract-Conference.html (arXiv:2405.08366)
Tier 1. On GPT-2 Small IOI, builds supervised feature dictionaries from known task attributes (activation
reconstructed as a sum of per-attribute-value vectors), evaluates them by whether patched reconstructions
preserve the output (sufficiency/necessity) and whether editing one attribute's vector controls behaviour;
uses them as the reference to grade SAEs.
For lc0: the closest precedent to the user's decoder D(f): a fixed external vocabulary mapped additively into
the residual stream at one site, judged interventionally (axes 1 decode, 4 linear, 8 interventional). Edit-
by-swapping feature vectors is the transformer analogue of item 1's concept intervention. Weakness: few,
mutually exclusive attributes vs. 30k dependent bits; single task, single model; no cross-layer chaining.

---

## Gaps for Phase 3
- No work found that fits a supervised dictionary at *several* layers and chains them (the user's
  "several decoders" design); items 4 and 6 are the nearest, both retrained. Search T5/T6 for this.
- Leakage metrics (10, 11) and completeness (12) are defined for bottleneck/post-hoc concept models, not for
  decoders f -> h; adapting them is a contribution, not a citation.
- No chess/Go concept-bottleneck model found (search 2026-10-09).
