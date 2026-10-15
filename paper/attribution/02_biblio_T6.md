# Phase 2 — Bibliography T6: circuit discovery and attribution graphs with error nodes

Search date: 2026-10-09. Sources: arXiv API (metadata for every ID below), ACL Anthology, ICLR virtual site,
transformer-circuits.pub, PAPERS.md [99]-[104] (already verified 2026-09-03, re-checked here).
Keywords: path patching, activation patching best practice, attribution patching, AtP*, ACDC, edge attribution
patching, sparse feature circuits, error nodes, attribution graph, replacement model, circuit faithfulness metric.
Procedure: 9 seeds from 01_scoping.md -> backward/forward chase (Hanna, Miller, MIB, Haklay, Ferrando, Bhaskar,
Ge) -> stop after the MIB citation list added nothing new in-scope.

Records screened: 21. Included: 15. Dropped: 6 (see end).
Tiers per 01_scoping.md. Axis numbers refer to the taxonomy (1 entry point, 2 direction, 3 layer span, 4 map
class, 5 operator, 6 residual, 7 identifiability, 8 faithfulness test, 9 cost).

Common lc0 caveat for this whole family: all gradient-based variants (AtP, AtP*, EAP, EAP-IG, attribution graphs,
sparse feature circuits) need a backward pass, i.e. a PyTorch port of lc0; the ONNX/CPU setup supports only the
pure patching variants (path patching, ACDC, activation patching), at ~180 forward passes/s.

---

## Patching foundations

**1. Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2211.00593**
Tier 1. Hand-built circuit (26-28 heads) found by path patching and mean-ablation; scored on faithfulness,
completeness and minimality. Axes: 1 none, 3 path-level, 5 patching, 8 the three criteria. For lc0: the
completeness criterion (does the circuit's complement still compute the behaviour?) is the template for asking
whether the named-feature subgraph is complete. Weakness: manual, one task, mean-ablation baseline choice
drives results; completeness numbers showed gaps the authors acknowledge.

**2. Goldowsky-Dill, N., MacLeod, C., Sato, L., & Arora, A. (2023). *Localizing model behavior with path patching* (arXiv:2304.05969). arXiv. https://arxiv.org/abs/2304.05969**
Tier 2 (preprint; method replicated as standard by 1, 3, 9). Formalizes path patching: intervene on a specific
path (sender -> receiver through the residual) rather than a node; frames hypotheses as which paths matter.
Axes: 3 path-level, 4 identity, 5 patching. For lc0: the way to test "feature f at square s, layer L reaches the
value head only via attention to square t" with forward passes only (ONNX-compatible). Weakness: cost grows with
paths; needs a counterfactual input per test, which here should be a real board edit (MaskPinner/LineBlocker),
not a feature-space flip.

**3. Zhang, F., & Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. *International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2309.16042**
Tier 1. Shows localization conclusions flip with the corruption method (Gaussian noise vs symmetric token
replacement) and metric (logit difference vs probability). Axes: 8, plus a methods-validity check on 5. For lc0:
the corruption must be an in-distribution board (STR analogue = minimally different legal position), and the
metric should be the value logit / move logit, not WDL probability. Directly backs DA Major 2.

**4. Heimersheim, S., & Nanda, N. (2024). *How to use and interpret activation patching* (arXiv:2404.15255). arXiv. https://arxiv.org/abs/2404.15255**
Tier 3 (tutorial). Denoising vs noising patching measure different things (sufficiency vs necessity); backup
behaviour and OR-gates break single-node ablation stories. For lc0: report both directions per feature family;
redundant feature bits (DA Major 1) will show as OR-gates under noising. Weakness: no new experiments; guidance.

## Automated discovery and gradient approximations

**5. Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards automated circuit discovery for mechanistic interpretability. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. https://arxiv.org/abs/2304.14997**
Tier 1. ACDC: greedy pruning of computational-graph edges by KL change under patching. Axes: 3 path-level,
4 identity, 5 patching, 7 order-dependent (greedy). For lc0: runs on ONNX with forward passes only, but the graph
over 10 layers x 8 heads x 64 positions is large; per-position edges are expensive. Weakness: over components,
not features; misses edges whose effects cancel; threshold-sensitive.

**6. Syed, A., Rager, C., & Conmy, A. (2023). *Attribution patching outperforms automated circuit discovery* (arXiv:2310.10348). NeurIPS 2023 ATTRIB Workshop. https://arxiv.org/abs/2310.10348**
Tier 3 (workshop). Edge Attribution Patching (EAP): first-order Taylor estimate of every edge's patching
effect from two forward + one backward pass; beats ACDC on recovering known circuits. Axes: 5 gradient x
activation-difference, 9 cheap but needs gradients. For lc0: the obvious estimator for "decoded feature
contribution": (D(f_clean) - D(f_corrupt)) · grad_h y is EAP with the decoder as the node. Weakness: linearization
fails where attention saturates.

**7. Kramár, J., Lieberum, T., Shah, R., & Nanda, N. (2024). *AtP\*: An efficient and scalable method for localizing LLM behaviour to components* (arXiv:2403.00745). Google DeepMind. https://arxiv.org/abs/2403.00745**
Tier 2 (lab technical report). Systematic study of attribution patching (AtP); identifies two false-negative
modes (attention-softmax saturation; cancellation between direct and indirect effects) and fixes them
(recompute attention patterns for queries/keys; GradDrop), with a bound on remaining false negatives.
Axes: 5, 8 (compared against exhaustive patching). For lc0: the error analysis to run before trusting any
gradient-times-delta attribution through 10 blocks of attention — exactly the "deep jungle" path. Weakness:
evaluated on LLM components, not on learned feature maps.

**8. Hanna, M., Pezzelle, S., & Belinkov, Y. (2024). Have faith in faithfulness: Going beyond circuit overlap when finding model mechanisms. *Conference on Language Modeling (COLM 2024)*. https://arxiv.org/abs/2403.17806**
Tier 1. EAP-IG: integrated gradients along the clean->corrupt path for edge scores; argues circuits must be
judged by faithfulness, not overlap with a reference circuit. Axes: 5 path integral, 8. For lc0: IG between two
real boards (not from a zero baseline) is the on-manifold-ish variant that addresses DA Major 2 partially —
the endpoints are real, the path still is not. Weakness: interpolated activations are off-distribution.

**9. Ferrando, J., & Voita, E. (2024). Information flow routes: Automatically interpreting language models at scale. *Proceedings of EMNLP 2024* (pp. 17432-17445). Association for Computational Linguistics. https://arxiv.org/abs/2403.00824**
Tier 1. Builds per-prediction graphs by attributing each residual update to its inputs (ALTI-style, top-down
pruning) with one forward pass; no counterfactual needed. Axes: 3 path-level, 5 decomposition, 9 forward-only.
For lc0: the only graph method here that fits ONNX/CPU and needs no contrast board; gives square-to-square routes
over 64 tokens. Weakness: contribution = proximity-based decomposition, not interventional; no feature vocabulary.

**10. Haklay, T., Orgad, H., Bau, D., Mueller, A., & Belinkov, Y. (2025). Position-aware automatic circuit discovery. *Proceedings of ACL 2025 (Volume 1: Long Papers)* (pp. 2792-2817). Association for Computational Linguistics. https://arxiv.org/abs/2502.04577**
Tier 1. Extends edge attribution to position-specific edges and uses schemas to align positions across
variable-length inputs. For lc0: positions are squares, fixed and meaningful, so position-aware circuits come for
free; the schema idea maps to aligning squares by role (pinned piece, pinner, king) across positions. Weakness:
schema construction partly LLM-assisted; text tasks only.

**11. Bhaskar, A., Wettig, A., Friedman, D., & Chen, D. (2024). Finding transformer circuits with edge pruning. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. https://arxiv.org/abs/2406.16778**
Tier 1. Learns a sparse edge mask by gradient descent instead of greedy search or linearization; more faithful
circuits at scale. Axes: 5 optimization over masks, 7 regularizer-dependent. For lc0: alternative to EAP when
linearization fails; needs a differentiable model and training per task. Weakness: mask depends on sparsity
penalty, so not unique.

## Feature-level graphs with error nodes (the anchor for SQ3/SQ4)

**12. Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2025). Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2403.19647**
Tier 1 (PAPERS [102]). Nodes are SAE features plus one SAE error node per site; edges by attribution patching
/ IG; circuits ablated to test faithfulness and edited (SHIFT) to remove spurious cues. Axes: 1 dictionary,
3 path-level across layers, 5 gradient/IG, 6 error node kept in graph, 8 ablation faithfulness. For lc0: replace
the SAE with the per-layer supervised decoder D_L; E_L - D_L(f) becomes the error node, and its attribution share
is the "what the vocabulary misses" number the user wants. Weakness: error nodes often carry large effect and are
uninterpretable; per-layer dictionaries are independent, so the graph mixes reconstructions with residuals.

**13. Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., et al. (2025, March 27). Circuit tracing: Revealing computational graphs in language models. *Transformer Circuits Thread*. Anthropic. https://transformer-circuits.pub/2025/attribution-graphs/methods.html**
Tier 2 (not peer-reviewed; PAPERS [99]). Cross-layer transcoder replaces the MLPs; a local replacement model per
prompt adds error nodes (true MLP output minus CLT output) and freezes attention patterns and norm denominators,
so edges are exact linear attributions (activation x virtual weight); graphs pruned; completeness and replacement
scores reported. Axes: 3 cross-layer, 4 sparse-linear, 5 exact linear given frozen attention, 6 error node +
completeness score. For lc0: the closest design to "several decoders on several layers" — a supervised CLT whose
latents are the 30k bits, with error nodes per layer. Weakness: freezing attention hides QK effects (in lc0,
where routing between squares is attention, that may be most of the computation); graphs explain one input.

**14. Miller, J., Chughtai, B., & Saunders, W. (2024). Transformer circuit faithfulness metrics are not robust. *Conference on Language Modeling (COLM 2024)*. https://arxiv.org/abs/2407.08734**
Tier 1. Same circuit's faithfulness score varies widely with ablation type, which edges/positions are ablated,
and metric normalization. Axis 8. For lc0: the protocol must fix and report ablation type (mean / resample from
real boards / zero) and position granularity before comparing decoder-based against patching-based attributions.
Weakness: diagnostic only; no fix proposed beyond reporting.

**15. Mueller, A., Geiger, A., Wiegreffe, S., Arad, D., Arcuschin, I., Belfki, A., et al. (2025). MIB: A mechanistic interpretability benchmark. *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. https://arxiv.org/abs/2504.13151**
Tier 1. Two tracks: circuit localization (compares ACDC, EAP, EAP-IG, edge pruning etc. on faithfulness
curves) and causal-variable localization (DAS, SAEs, probes judged by interchange accuracy). Finds attribution
+ IG-based edge methods strong, and SAE features not better than neurons for causal-variable localization.
Axes: 8 for both families. For lc0: ready-made metric definitions (integrated faithfulness over circuit sizes;
IIA) for the protocol; the causal-variable track is the bridge to T7. Weakness: LLM tasks only.

---

## Dropped / deferred (6)

- Ge, X., et al. (2024), linear computation graphs, arXiv:2405.13868 — verified, but transcoder-based and
  overlaps 12-13; deferred to T5.
- Dunefsky et al. (2024), transcoders, arXiv:2406.11944 — verified (NeurIPS 2024); belongs to T5.
- He, Z., et al. (2024), patch-free circuits in Othello-GPT, arXiv:2402.12201 — verified, but author comment
  "Not final version"; deferred to T5/T8.
- Lindsey et al. (2025), "On the Biology of a Large Language Model" — application of 13, no method; peripheral.
- Nanda, N. (2023), "Attribution patching" blog post — Tier 4; superseded by 6 and 7.
- Bayat Makou et al. (2026), arXiv:2606.06267 (PAPERS [104]) — in PAPERS.md as TMLR 2026; could not re-confirm
  the TMLR acceptance in this pass, so not cited here (gray zone = fail).

## Distributional note
Venue: 10/15 peer-reviewed; time: 14/15 in 2023-2025 (field age, not a search gap); all LLM/text — no T6 method
has been applied to a game network with a fixed external vocabulary. That gap is T8's to confirm.
