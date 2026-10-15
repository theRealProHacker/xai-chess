# Phase 2 — Bibliography T3: probing validity, layerwise readout, erasure (encoder route)

Searched and verified 2026-10-09. Every entry was checked against its arXiv abstract record (export API:
title, authors, date) and, where a venue is claimed, against the publisher page (ACL Anthology `.bib`,
NeurIPS proceedings, PMLR, LessWrong). Already verified in `PAPERS.md`: Elazar et al. [92], Belinkov [107].

Axis numbers refer to the taxonomy in `01_scoping.md` §2 (1 entry point, 2 direction, 3 layer span,
4 map class, 5 attribution operator, 6 residual, 7 identifiability, 8 faithfulness test, 9 cost).

## Search log

- Sources: arXiv export API (id lookup), ACL Anthology, NeurIPS/PMLR proceedings, LessWrong, web search for venue checks.
- Seeds: the nine T3 seeds in `01_scoping.md`, all verified. Backward/forward chase added 9 candidates
  (MDL probing, information-theoretic probing, RLACE, Kumar et al., grammatical-number usage,
  counterfactual embeddings, Othello world model ×2, logit lens). Second chase round added nothing in scope.
- Identified 18, verified 18, included 16, excluded 2 (in scope but redundant; see end), dropped as unverifiable 0.

## Annotated bibliography

### A. Probe validity: what probe accuracy does and does not show

**[T3-1] Alain, G., & Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes* (arXiv:1610.01644). arXiv. https://arxiv.org/abs/1610.01644**
Tier 3 (presented at the ICLR 2017 workshop track per MLAnthology; arXiv is the archival copy).
Founding paper for per-layer linear probes: separability of the target rises with depth. This is the
template of the project's encoder result 1. Fills axes 1 (readout), 2 (encode), 3 (per-layer independent),
4 (linear). Attribution operator: none; faithfulness test: none. Weakness: it is about the output label,
not about intermediate concepts, and says nothing on use.

**[T3-2] Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 2733–2743). ACL. https://doi.org/10.18653/v1/D19-1275**
Tier 1. Control task = same inputs, random but consistent labels per word type; selectivity = task accuracy
minus control accuracy. Expressive (MLP) probes reach high control accuracy, so accuracy measures the probe.
For lc0: a control task that assigns random labels per (square, occupant) type tests whether the
"near-perfect at the embedding" readout is just occupancy memorisation. Directly supports the occupancy-only
control in DA Major 3. Axis 8 (selectivity, correlational only).

**[T3-3] Voita, E., & Titov, I. (2020). Information-theoretic probing with minimum description length. In *Proceedings of EMNLP 2020* (pp. 183–196). ACL. https://doi.org/10.18653/v1/2020.emnlp-main.14**
Tier 1. Replaces accuracy with codelength (online coding): how much data the probe needs to reach its accuracy.
Stable across seeds and probe sizes where accuracy is not. Useful for the per-layer families table: features
that are "near-perfectly readable" at every layer may still differ sharply in how easily they are extracted
(e.g. pawn_cover, mate). Correlational; no attribution operator.

**[T3-4] Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics, 48*(1), 207–219. https://doi.org/10.1162/coli_a_00422**
Tier 1. Survey of probe pitfalls: baseline choice, probe complexity, correlation vs causation, dataset
confounds; points to amnesic/counterfactual methods as the fix. Use as the framing citation for SQ2 and DA
Major 4. Already in `PAPERS.md` [107].

**[T3-5] Ravichander, A., Belinkov, Y., & Hovy, E. (2021). Probing the probing paradigm: Does probing accuracy entail task relevance? In *Proceedings of EACL 2021* (pp. 3363–3377). ACL. https://doi.org/10.18653/v1/2021.eacl-main.295**
Tier 1. Models trained on data where a property is irrelevant still encode it to probe-detectable levels.
Readability is therefore not evidence of use. The lc0 analogue: many of the 30k bits are cheap functions of
occupancy and will be readable whether or not the value head uses them. Core citation for "readable ≠ used".

### B. From readability to use: erasure and counterfactual interventions

**[T3-6] Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *Transactions of the ACL, 9*, 160–175. https://doi.org/10.1162/tacl_a_00359**
Tier 1. Remove a property from layer L with INLP, measure the change in the model's output; compare with
removing an equal-rank random subspace. High probe accuracy often coincides with small behavioural effect.
Closest existing template for the encoder route: axes 1 readout, 2 encode, 3 per-layer, 5 ablation/erasure,
8 interventional (with random-subspace control). Weaknesses: erasure is off-distribution; INLP removes more
than the concept (collateral damage); per-family erasure needs rank budgets that may exceed 256 dims when
families are large. Already in `PAPERS.md` [92].

**[T3-7] Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. In *Proceedings of ACL 2020* (pp. 7237–7256). ACL. https://doi.org/10.18653/v1/2020.acl-main.647**
Tier 1. INLP: repeatedly train a linear classifier, project onto its nullspace. Linear only; greedy; number of
iterations sets how much is removed. For lc0, erasure rank per family is a free parameter that the protocol
must fix and report. Axis 4 linear, axis 7 non-unique (depends on iteration count and classifier).

**[T3-8] Ravfogel, S., Twiton, M., Goldberg, Y., & Cotterell, R. (2022). Linear adversarial concept erasure. In *Proceedings of ICML 2022* (PMLR Vol. 162). https://proceedings.mlr.press/v162/ravfogel22a.html**
Tier 1. R-LACE: minimax over a rank-k orthogonal projection, removing a concept with fewer dimensions than INLP.
Lower collateral damage, but iterative and not closed-form. Mostly superseded for this project by LEACE;
cited to show the erasure family's progression.

**[T3-9] Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect linear concept erasure in closed form. In *Advances in Neural Information Processing Systems 36* (NeurIPS 2023). https://arxiv.org/abs/2306.03819**
Tier 1. Closed-form least-squares edit that makes the concept linearly unpredictable (equal class-conditional
means) with minimal change to the representation; "concept scrubbing" applies it at every layer. Best fit
for lc0 encoder-route attribution: cheap (one covariance per family per layer, ONNX-compatible as an affine
patch, no gradients; axis 9 favourable), and the per-family output change is an interventional attribution
(axis 5 erasure, axis 8 interventional). Weaknesses: only linear guarantees (an MLP probe may still recover
the concept); multi-label families need one erasure per bit or a joint covariance; still off-distribution.

**[T3-10] Kumar, A., Tan, C., & Sharma, A. (2022). Probing classifiers are unreliable for concept removal and detection. In *Advances in Neural Information Processing Systems 35* (pp. 17994–18008). https://arxiv.org/abs/2207.04153**
Tier 1. When the concept correlates with the task label, probe-based removal also removes task-relevant
signal, and probes latch onto spurious correlates. Directly relevant to the engine-derived families
(`mate`, `material_win`), which correlate with the value target: erasing them will damage value prediction
partly for that reason, inflating their attribution. Supports the DA minor issue on label leakage.

**[T3-11] Tucker, M., Qian, P., & Levy, R. (2021). What if this modified that? Syntactic interventions with counterfactual embeddings. In *Findings of ACL-IJCNLP 2021* (pp. 862–875). ACL. https://doi.org/10.18653/v1/2021.findings-acl.76**
Tier 1. Gradient-step the representation against a probe until the probe reads a different value, then
observe the model's behaviour. A test of use that flips the concept rather than deleting it. For lc0:
the natural encoder-route counterpart to the decoder route's "substitute D(f')": push the token along the
probe gradient for "pin present" and compare with MaskPinner's board-level effect. Needs gradients through
the probe only (cheap) but a forward pass through lc0 from L onward. Weakness: the probe's gradient direction
may not be the model's direction (see T3-15 and Makelov et al., T7).

### C. Layerwise readout to the output: lenses

**[T3-12] nostalgebraist. (2020, August 31). *Interpreting GPT: The logit lens* [Online forum post]. LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens**
Tier 4 (gray literature; included because the method has no peer-reviewed origin). Apply the final unembedding
to intermediate residual states to read the model's running prediction. lc0 analogue: apply the value/policy
heads to intermediate tokens — context result 3 ("value head applied to decoded layer-9 tokens: weak") is
already a logit-lens experiment. Brittle when intermediate layers are not in the head's basis.

**[T3-13] Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., Smith, L., Halawi, D., Biderman, S., & Steinhardt, J. (2023). *Eliciting latent predictions from transformers with the tuned lens* (arXiv:2303.08112). arXiv. https://arxiv.org/abs/2303.08112**
Tier 3 (no peer-reviewed venue found). Learn one affine translator per layer into the final layer's basis, then
apply the output head; includes causal checks that features important to the lens are important to the model.
For lc0: a per-layer tuned lens gives a layerwise value/policy estimate; composing it with a per-layer probe or
decoder gives a two-map path f → h_L → output with a short causal distance, which is one concrete answer to the
user's front/back hybrid. Axis 3 per-layer, axis 4 affine. Weakness: the translator is fit to predict the
output, so it may route around the model's actual computation.

### D. Linearity and world-model probes in board games

**[T3-14] Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Emergent world representations: Exploring a sequence model trained on a synthetic task. In *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2210.13382**
Tier 1. Othello-GPT: nonlinear (MLP) probes recover the board; editing activations by gradient against the
probe changes predicted legal moves as the edited board implies. Board-game precedent for validating a probe by
intervention with a ground-truth board, which is exactly what MaskPinner/LineBlocker provide. Weakness: needed
nonlinear probes; interventions are per-layer gradient edits with tuning.

**[T3-15] Nanda, N., Lee, A., & Wattenberg, M. (2023). Emergent linear representations in world models of self-supervised sequence models. In *Proceedings of the 6th BlackboxNLP Workshop* (pp. 16–30). ACL. https://doi.org/10.18653/v1/2023.blackboxnlp-1.2**
Tier 1 (workshop, archival). The Othello board is linear once encoded relative to the player to move
("mine/theirs"); linear probe directions then support vector-addition interventions. Two lessons for lc0:
the feature basis matters (lc0 also uses side-to-move-relative planes, so the vocabulary should be too), and
linear interventions along probe directions are a cheap encoder-route attribution operator.

**[T3-16] Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. In *Proceedings of ICML 2024* (PMLR Vol. 235). https://arxiv.org/abs/2311.03658**
Tier 1. Formalises "linear representation" in three senses (subspace, probe, steering vector) and shows they
coincide under a causal inner product, not the Euclidean one. For lc0: probe directions and decoder columns
(the encode and decode routes) are only comparable under the right inner product; a whitened/causal metric
should be used when comparing or chaining them. Axis 7: identifiability depends on the inner product.

## Excluded after screening

- Pimentel, T., et al. (2020). Information-theoretic probing for linguistic structure. ACL 2020, 4609–4622 (verified). Redundant with T3-2/T3-3 for this paper's purpose.
- Lasri, K., et al. (2022). Probing for the usage of grammatical number. ACL 2022, 8818–8831 (verified). Same design as T3-6 with a usage-specific probe; cite only if Phase 3 needs a second amnesic example.

## Coverage notes

- Venue skew: 13/16 ACL-family or ML-conference; 1 gray (logit lens), 2 arXiv-only. No distributional skew
  advisory beyond the expected NLP concentration: nearly all probing-validity work comes from NLP.
- Gap for downstream phases: no work found that applies control tasks, MDL or LEACE to a chess network.
  The closest game precedent is Othello (T3-14, T3-15); chess probing validity in `PAPERS.md` [66]
  (Pálsson & Björnsson, ECAI 2024) belongs to T8.
- Verification: 18/18 candidates confirmed (title, authors, year, ID; venue where claimed). Alain & Bengio
  venue is workshop-only and carried as arXiv. Tuned lens has no venue; carried as Tier 3.
