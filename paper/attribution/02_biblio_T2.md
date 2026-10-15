# T2 bibliography: concept attribution through learned concept maps

Theme (01_scoping.md §3): methods that learn a map between activations and a concept vocabulary
(h -> c probe/CAV, or c -> h generative/decoder) and attribute model outputs through it.

## Search strategy

- **Sources**: arXiv API (metadata + abstracts), NeurIPS/ICML/ICLR proceedings pages, Crossref DOIs, OpenReview where reachable. Project `PAPERS.md` [31] (TCAV) reused.
- **Keywords**: concept activation vector, TCAV significance, concept gradient, nonlinear concept attribution, causal concept effect, counterfactual concept explanation, concept importance estimation, CAV layer inconsistency, cross-layer CAV.
- **Procedure**: 6 seeds -> one backward/forward chase per seed (Kim/Feder/Geiger/Fel lines) -> stop.
- **Window**: 2017-2025.
- **Inclusion/exclusion**: per 01_scoping.md. Unsupervised concept discovery kept only where it supplies the attribution operator (ACE, Fel).
- **Verification**: every entry's title, authors, year and arXiv ID checked against the arXiv API record on 2026-10-09; venues checked against the proceedings page, Crossref DOI record, or the arXiv journal-ref / comment field (marked "(arXiv comment)" where that is the only venue evidence).

## Flow

Candidates screened: 18. Included: 15. Dropped: 3 (see end).

## Sources (N = 15)

Axis shorthand (01_scoping.md): 1 entry point, 2 direction, 3 layer span, 4 map class,
5 attribution operator, 6 residual, 7 identifiability, 8 faithfulness test, 9 cost on lc0.

### A. Linear and nonlinear concept maps with gradient attribution

1. **Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018).** Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In *Proceedings of the 35th International Conference on Machine Learning* (PMLR 80). https://arxiv.org/abs/1711.11279 — Tier 1. (PAPERS.md [31])
   - CAV = normal of a linear classifier separating concept vs. random activations at one layer; sensitivity = directional derivative of the logit along the CAV; TCAV score = fraction of inputs with positive sensitivity, tested against random-CAV runs.
   - Axes: readout side; encode (h -> c); per-layer independent; linear; gradient; residual ignored; one direction per concept, non-unique under correlated concepts; statistical test only, no interventional check.
   - lc0: a per-square linear probe (our encoder result 1) already gives one CAV per bit per layer; TCAV is the cheapest gradient readout of it. Needs a torch port for the logit gradient.
   - Weakness: readable ≠ used; with ~30k correlated bits the CAVs are nearly collinear, so per-bit scores are unstable (Nicolson, #13).

2. **Bai, A., Yeh, C.-K., Lin, N. Y. C., Ravikumar, P., & Hsieh, C.-J. (2023).** Concept gradient: Concept-based interpretation without linear assumption. In *International Conference on Learning Representations (ICLR 2023)* (arXiv comment). https://arxiv.org/abs/2208.14966 — Tier 1.
   - Defines CG(x) = ∇g(x)^† ∇f(x): model gradient mapped into concept space through the Moore–Penrose pseudo-inverse of the concept model g's Jacobian; CAV is the special case g linear. g may be nonlinear (finetuned copy of f).
   - Axes: readout side; encode; single layer (layer chosen by unfreezing from the top until concept accuracy saturates; best at the penultimate layer in their CUB study); nonlinear; Jacobian chain rule; residual = the part of ∇f outside the row space of ∇g, silently dropped; unique up to the pseudo-inverse's min-norm choice; evaluated by concept recall vs. labels, not intervention.
   - lc0: the closest published formula to the user's "gradient chained through a fitted map". With an encoder g: token_L -> features, CG gives per-feature attribution at layer L; the dropped orthogonal component is exactly the "what the vocabulary misses" quantity and should be reported, not discarded. Authors show results depend on gradient normalization in special cases.

3. **Crabbé, J., & van der Schaar, M. (2022).** Concept activation regions: A generalized framework for concept-based explanations. In *Advances in Neural Information Processing Systems 35* (arXiv comment). https://arxiv.org/abs/2209.11222 — Tier 1.
   - Replaces the CAV direction by a kernel-SVM region; gives global explanations and local concept-based feature importance; radial-kernel CARs are invariant to latent isometries.
   - Axes: readout; encode; per-layer; nonlinear (kernel); membership/gradient; residual ignored; identifiability improved only w.r.t. isometries.
   - lc0: fits bits that are not linearly readable at early layers (pawn_cover, mate). Kernel cost scales with sample count; fine at 20k positions per square.

4. **Fel, T., Boutin, V., Moayeri, M., Cadène, R., Bethune, L., Andéol, L., Chalvidal, M., & Serre, T. (2023).** A holistic approach to unifying automatic concept extraction and concept importance estimation. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2306.07304 — Tier 1.
   - Frames concept methods as (1) dictionary learning A ≈ UV on activations, (2) importance = an attribution method (gradient, IG, occlusion, Sobol) applied in concept space U; derives optimality guarantees for importance estimators and metrics for extraction.
   - Axes: dictionary (unsupervised here, but the factorization is vocabulary-agnostic); both directions; single layer; linear; any T1 operator; reconstruction error measured; non-unique (factorization choice).
   - lc0: supplies the formal "attribution operator applied in concept coordinates" template the protocol needs; swap their learned U for our fixed feature vocabulary and keep their evaluation metrics.

5. **Nicolson, A., Schut, L., Noble, J. A., & Gal, Y. (2025).** Explaining explainability: Recommendations for effective use of concept activation vectors. *Transactions on Machine Learning Research* (arXiv comment: accepted 02/2025). https://arxiv.org/abs/2404.03713 — Tier 1.
   - Shows CAVs are inconsistent across layers, entangled with other concepts, and spatially dependent; gives detection tools; synthetic "Elements" dataset with known concept→class ground truth; negative-set choice changes CAV meaning.
   - Axes: faithfulness critique (axis 8) and identifiability (axis 7) evidence.
   - lc0: all three properties apply directly — per-layer probes, entangled correlated bits, and 64 square positions. Supports group-level (family) attribution and a synthetic ground-truth check (our board interventions play the role of Elements).

6. **He, Z., Sinha, S., Xiong, G., & Zhang, A. (2025).** GCAV: A global concept activation vector framework for cross-layer consistency in interpretability. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2025)* (arXiv comment). https://arxiv.org/abs/2508.21197 — Tier 1 (venue from comment only).
   - Aligns per-layer CAVs by contrastive learning and fuses them with attention into one global CAV; reports lower TCAV-score variance.
   - Axes: readout; encode; cross-layer; linear + learned fusion; gradient; residual ignored.
   - lc0: the only verified concept-map method that combines maps from several layers, i.e. the user's "several decoders on several layers". The fusion weights are learned, so they add their own identifiability problem; treat as a design precedent, not a validated answer.

### B. Causal concept effect and counterfactual concept maps

7. **Goyal, Y., Feder, A., Shalit, U., & Kim, B. (2019).** Explaining classifiers with causal concept effect (CaCE). *arXiv*. https://arxiv.org/abs/1907.07165 — Tier 3.
   - Defines CaCE = E[f | do(concept=1)] − E[f | do(concept=0)]; shows correlational scores (incl. TCAV-style) mislead under confounding; estimates CaCE with a conditional VAE as generator when do() cannot be simulated.
   - Axes: input side; decode (c -> x via VAE); none (input-level); nonlinear; intervention; residual n/a; unique as a causal estimand; tested on datasets with known effect.
   - lc0: the right estimand for the protocol. Our board edits (MaskPinner, LineBlocker, piece removal) are exact do() operators, so no VAE is needed; feature-space decoder attributions are graded against CaCE computed this way.

8. **Feder, A., Oved, N., Shalit, U., & Reichart, R. (2021).** CausaLM: Causal model explanation through counterfactual language models. *Computational Linguistics, 47*(2), 333–386. https://doi.org/10.1162/coli_a_00404 — Tier 1.
   - Fine-tunes the representation model with adversarial auxiliary tasks so it forgets one concept while keeping confounders; effect = output difference between original and concept-blind representation.
   - Axes: representation; neither (retrained encoder); all layers; nonlinear; ablation by retraining; residual n/a; depends on adversarial success.
   - lc0: retraining lc0 per concept is too costly; relevant as the "erase, then compare" baseline that T3's LEACE/INLP do cheaply. Issue/page numbers from Crossref-indexed citation; Crossref record confirms title, authors, journal, 2021.

9. **Abraham, E. D., D'Oosterlinck, K., Feder, A., Gat, Y., Geiger, A., Potts, C., Reichart, R., & Wu, Z. (2022).** CEBaB: Estimating the causal effects of real-world concepts on NLP model behavior. In *Advances in Neural Information Processing Systems 35*. https://arxiv.org/abs/2205.14140 — Tier 1.
   - Benchmark with human-written counterfactuals per concept; scores explanation methods (CaCE estimators, TCAV, CausaLM, INLP, ...) by error against the true individual causal effect (ICaCE).
   - Axes: faithfulness test = interventional against ground truth (axis 8).
   - lc0: the evaluation design to copy. Real board interventions give per-position counterfactual pairs; each attribution method in the paper gets an ICaCE-style error. Their finding that many methods fail this test is the strongest prior that probe/CAV scores need interventional grading.

10. **Wu, Z., D'Oosterlinck, K., Geiger, A., Zur, A., & Potts, C. (2023).** Causal proxy models for concept-based model explanations. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR 202, pp. 37313–37334). https://arxiv.org/abs/2209.14279 — Tier 1.
    - Trains a proxy that matches the black box factually and whose internal representations can be intervened on (interchange-intervention training) to simulate counterfactual behaviour; uses approximate counterfactuals.
    - Axes: bottleneck/aligned subspace; both; single layer; nonlinear; interchange intervention; residual = proxy–model gap; strong state on CEBaB.
    - lc0: bridges T2 and T7: a per-layer decoder could be trained with an interchange objective instead of ridge FVU, so attributions are causal by construction. Costs training a proxy; needs counterfactual board pairs (we have them).

11. **Gat, Y., Calderon, N., Feder, A., Chapanin, A., Sharma, A., & Reichart, R. (2024).** Faithful explanations of black-box NLP models using LLM-generated counterfactuals. In *International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2310.00603 — Tier 1.
    - Argues approximating counterfactuals is required for faithful concept explanations; compares generated counterfactuals vs. matching in a learned embedding space.
    - Axes: input side; intervention vs. matching; faithfulness argument (axis 8).
    - lc0: matching (nearest position differing in one feature family) is the cheap fallback where no exact board edit exists for a feature family (e.g. "strong square").

### C. Decompositions with explicit residuals and concept-conditional propagation

12. **Zhou, B., Sun, Y., Bau, D., & Torralba, A. (2018).** Interpretable basis decomposition for visual explanation. In *Computer Vision – ECCV 2018* (LNCS, pp. 122–138). Springer. https://doi.org/10.1007/978-3-030-01237-3_8 — Tier 1.
    - Learns concept vectors in feature space from a labelled concept corpus, then decomposes a class's weight vector into a sparse nonnegative sum of concept vectors plus a residual; contribution of each concept to the score is read off the decomposition. Reports the unexplained share: mean residual 65.9% for ResNet50 scene classification with 660 concepts, and notes it need not approach zero.
    - Axes: readout; encode then project; single (final) layer; sparse-linear; linear decomposition; residual explicitly kept; non-unique (greedy sparse selection).
    - lc0: the earliest verified precedent for "named concepts + explicit unexplained term" in a prediction decomposition; maps onto value-head decomposition at `/value/dense1` with our features.

13. **Achtibat, R., Dreyer, M., Eisenbraun, I., Bosse, S., Wiegand, T., Samek, W., & Lapuschkin, S. (2023).** From attribution maps to human-understandable explanations through concept relevance propagation. *Nature Machine Intelligence, 5*(9), 1006–1019. https://doi.org/10.1038/s42256-023-00711-8 — Tier 1.
    - Conditions LRP on a chosen latent channel/subspace to get "where" and "what" per prediction; concept atlases and composition analyses.
    - Axes: none (concept = channel or subspace, identity map); per-layer; identity; conservative relevance propagation; relevance conserved, so no residual by design; deterministic.
    - lc0: an alternative to gradients for pushing relevance from the output back to a layer, then into our decoder's feature coordinates. Needs AttnLRP-style rules for attention (T1); no ONNX path.

14. **Ghorbani, A., Wexler, J., Zou, J., & Kim, B. (2019).** Towards automatic concept-based explanations. In *Advances in Neural Information Processing Systems 32*. https://arxiv.org/abs/1902.03129 — Tier 1.
    - ACE: segments inputs, clusters segment activations into concepts, scores importance with TCAV; states desiderata (meaningfulness, coherency, importance).
    - Axes: none (unsupervised); encode; single layer; linear; TCAV gradient.
    - lc0: relevant as the vocabulary-free counterpart: ACE-style clusters of square tokens that no feature bit predicts are a direct measurement of what the vocabulary misses.

15. **Ramaswamy, V. V., Kim, S. S. Y., Fong, R., & Russakovsky, O. (2023).** Overlooked factors in concept-based explanations: Dataset choice, concept learnability, and human capability. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023)* (arXiv comment). https://arxiv.org/abs/2207.09615 — Tier 1.
    - Explanations change with the probe dataset; concepts are often harder to learn than the classes they explain; human studies cap useful explanations at ~32 concepts.
    - Axes: faithfulness/validity critique.
    - lc0: three direct consequences: probe set = Lichess puzzle FENs biases explanations toward tactics; low-positive bits are unreliable (we already drop <20); report attributions at feature-family level (~32 families), not 30k bits.

## Source verification

| Source | Level | Venue | Author | Method | Currency | Overall |
|---|---|---|---|---|---|---|
| Kim 2018 TCAV | conf. | pass | pass | pass | seminal | A |
| Bai 2023 CG | conf. | pass (comment) | pass | pass | pass | A |
| Crabbé 2022 CAR | conf. | pass (comment) | pass | pass | pass | A |
| Fel 2023 | conf. | pass (journal-ref) | pass | pass | pass | A |
| Nicolson 2025 | journal | pass (comment) | pass | pass | pass | A |
| He 2025 GCAV | conf. | warn (comment only) | pass | warn (single group, new) | pass | B |
| Goyal 2019 CaCE | preprint | warn (no venue) | pass | pass | pass | B |
| Feder 2021 CausaLM | journal | pass (Crossref) | pass | pass | pass | A |
| Abraham 2022 CEBaB | conf. | pass (proceedings PDF) | pass | pass | pass | A |
| Wu 2023 CPM | conf. | pass (PMLR) | pass | pass | pass | A |
| Gat 2024 | conf. | pass (ICLR proceedings) | pass | pass | pass | A |
| Zhou 2018 IBD | conf. | pass (Crossref) | pass | pass | seminal | A |
| Achtibat 2023 CRP | journal | pass (Crossref) | pass | pass | pass | A |
| Ghorbani 2019 ACE | conf. | pass (NeurIPS page) | pass | pass | seminal | A |
| Ramaswamy 2023 | conf. | pass (comment) | pass | pass | pass | A |

No predatory venues. COI: Kim (Google) co-authors TCAV, ACE, CaCE — one group's framing dominates the CAV line; CEBaB/CPM/Gat share the Feder/Reichart/Potts groups. Peer-reviewed share: 14/15.

## Dropped

- Schrouff et al. (2021), "Best of both worlds" (arXiv 2106.08641): exists, but no venue found and method details not checked against the paper — gray zone, dropped.
- Pfau et al. (2021), "Robust semantic interpretability: revisiting CAVs" (arXiv 2104.02768): verified, ICML WHI workshop only; dropped to cap the list (its activation-perturbation test duplicates #7/#9 at lower tier).
- Graziani et al. (2018), "Regression concept vectors" (arXiv 1904.04520): verified, out of scope (continuous-concept CAV variant, nothing new on attribution through the map).

## Gaps for synthesis

- No verified paper attributes through a fitted **decoder** (c -> h) by the chain rule. CaCE (#7) uses a decoder to *generate* interventions; CG (#2) uses the pseudo-inverse of an *encoder*. The user's `∇_E y · ∂D/∂f` route has no direct precedent under these search terms; CG's pseudo-inverse is its encoder-side twin, and the two coincide only when D and g are inverse on the feature manifold.
- Residual handling is rare: only IBD (#12) keeps an explicit residual; CG discards it in the pseudo-inverse.
- Cross-layer combination: GCAV (#6) only, single group, comment-only venue.
- All interventional benchmarks (CEBaB, CPM, Gat) are NLP; transfer to lc0 relies on our board-edit operators.
