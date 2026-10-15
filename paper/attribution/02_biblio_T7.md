# Phase 2 — Bibliography T7: causal abstraction and interchange interventions

Search date: 2026-10-09. Agent: bibliography + source verification.

## Search strategy

- Sources: arXiv API (metadata, comments, journal_ref), PMLR, JMLR, ACL Anthology, OpenReview API, NeurIPS virtual site, LessWrong GraphQL API (for the causal scrubbing post).
- Keywords: causal abstraction; interchange intervention (accuracy, training); distributed alignment search; boundless DAS; causal scrubbing; subspace activation patching illusion; RAVEL; AxBench; causal variable localization; non-linear alignment maps.
- Seeds from 01_scoping.md T7, then one round of backward/forward chasing (Geiger group lineage; theory roots: Rubenstein, Beckers & Halpern; critiques: Makelov, Sutter; benchmarks: RAVEL, MIB, AxBench).
- Repo reuse: none. `PAPERS.md` and `paper/references.bib` contain no T7 entries.
- Verification rule: title, authors, year, venue and ID confirmed on the arXiv page **and** on the publisher/venue page (PMLR, JMLR, ACL Anthology, OpenReview, neurips.cc, AAAI/UAI per arXiv journal_ref). If something couldn't be confirmed, it was dropped.

Flow: 19 candidates screened → 4 dropped (below) → **15 verified and included**.

Dropped:
| Candidate | Reason |
|---|---|
| Wu et al., pyvene (arXiv 2403.07809) | Venue not confirmed on publisher page; tooling, not method. Mention as software only. |
| Arora et al., CausalGym (arXiv 2402.12560) | Venue not confirmed; linguistic-task benchmark, redundant with RAVEL/MIB. |
| Wu et al., ReFT (arXiv 2404.03592) | Venue not confirmed; steering/finetuning, outside attribution scope. |
| Mueller et al., "The quest for the right mediator" (arXiv 2408.01416) | arXiv metadata fetch returned nothing; not confirmed. |

Tiers per 01_scoping.md: 1 = peer-reviewed venue, 2 = lab report / replicated preprint, 3 = plain preprint, 4 = blog/forum.

---

## Annotated bibliography

### A. Theory of abstraction between causal models

**1. Rubenstein, P. K., Weichwald, S., Bongers, S., Mooij, J. M., Janzing, D., Grosse-Wentrup, M., & Schölkopf, B. (2017).** Causal consistency of structural equation models. In *Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence (UAI 2017)*. https://arxiv.org/abs/1707.00819 — Tier 1.
Defines exact transformations: a macro model is consistent with a micro model when interventions on macro variables agree with the micro interventions that implement them. Formal statement of the scoping Major issue 2: a feature-space edit (flip "pinned") is only meaningful if some board-level intervention realises it with the same downstream effect. Gives the consistency check we want between decoder-space edits D(f') and real board edits (MaskPinner/LineBlocker). Weakness: no algorithm for finding the map; assumes the macro variables are given (here they are, which helps).

**2. Beckers, S., & Halpern, J. Y. (2019).** Abstracting causal models. In *Proceedings of the AAAI Conference on Artificial Intelligence, 33* (AAAI-19). https://arxiv.org/abs/1812.03789 — Tier 1.
Refines Rubenstein into a hierarchy (exact, uniform, τ-, strong and constructive abstraction). Constructive abstraction (each high-level variable is a function of a disjoint set of low-level variables) is the clean case. Our vocabulary breaks it: 30k bits are functions of overlapping square sets, and a token mixes many bits. So for lc0 we should expect at most non-constructive abstraction, which is why identifiability is weak (axis 7). Theory only; no neural networks.

**3. Geiger, A., Ibeling, D., Zur, A., Chaudhary, M., Chauhan, S., Huang, J., Arora, A., Wu, Z., Goodman, N., Potts, C., & Icard, T. (2025).** Causal abstraction: A theoretical foundation for mechanistic interpretability. *Journal of Machine Learning Research, 26*(83), 1–64. https://jmlr.org/papers/v26/23-0058.html (arXiv 2301.04709) — Tier 1.
The reference text for T7. Generalises abstraction to arbitrary mechanism transformations. Recasts activation/path patching, causal scrubbing, SAEs, DAS and steering as causal abstraction with different alignment maps. Defines graded faithfulness (interchange intervention accuracy, IIA). For us it supplies the single vocabulary for every axis: the decoder D and probe P are alignment maps; the decoder residual is the part of the low-level state the high-level model doesn't account for. Weakness: the framework is permissive. It says how to score an alignment, not which map class is admissible (see 15).

### B. Interchange interventions and alignment search (methods)

**4. Geiger, A., Richardson, K., & Potts, C. (2020).** Neural natural language inference models partially embed theories of lexical entailment and negation. In *Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP* (EMNLP 2020 workshop). Association for Computational Linguistics. https://arxiv.org/abs/2004.14623 — Tier 1 (workshop).
First use of interchange interventions as a probe alternative: they swap a hypothesised representation between two inputs and check that the output follows the symbolic model. They show probe-readable ≠ causally used, which is the scoping Major issue 4 in its original form. The method is neuron-aligned and brute-force.

**5. Geiger, A., Lu, H., Icard, T., & Potts, C. (2021).** Causal abstractions of neural networks. In *Advances in Neural Information Processing Systems, 34* (NeurIPS 2021). https://arxiv.org/abs/2106.02997 — Tier 1.
Formalises the neural network as a low-level causal model and the interpretable algorithm as the high-level one. Measures IIA over all pairs of base/source inputs. The direct template for "is feature family F at layer L used": pair two boards, swap the aligned representation of F on square s, and compare lc0's output to the high-level counterfactual. On lc0 the counterfactual labels can come from engine evaluation of the real edited board. Weakness: it requires a high-level model that predicts the output, and our vocabulary is a feature set, not a value function. So the high-level model must be the engine or a fitted head (axis 8).

**6. Geiger, A., Wu, Z., Lu, H., Rozner, J., Kreiss, E., Icard, T., Goodman, N., & Potts, C. (2022).** Inducing causal structure for interpretable neural networks. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 7324–7338). https://proceedings.mlr.press/v162/geiger22a.html (arXiv 2112.00826) — Tier 1.
Interchange intervention training (IIT) trains a network so that zero IIT loss guarantees the causal model is an abstraction of it. Retraining lc0 is out of scope. The usable part: IIT loss can train a *replacement* or an adapter (e.g. a per-layer feature-bottlenecked module) whose feature slots are guaranteed to be causally used. That gives a supervised-bottleneck baseline (links T4). Cost: needs a differentiable lc0 (torch port).

**7. Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024).** Finding alignments between interpretable causal variables and distributed neural representations. In *Proceedings of the Third Conference on Causal Learning and Reasoning* (PMLR 236, pp. 160–187). https://proceedings.mlr.press/v236/geiger24a.html (arXiv 2303.02536) — Tier 1.
DAS learns an orthogonal rotation by gradient descent so that a high-level variable lives in a low-dimensional subspace, scored by IIA. This is the encoder route made interventional: probe P gives readability, while a DAS subspace for family F gives a per-layer IIA. Ridge decoder columns for F are a natural initialisation or comparison. It fills axes 2 (both), 4 (linear), 5 (interchange) and 8 (IIA). Weaknesses: one subspace per variable and pairwise counterfactual data; with 30k bits it has to work per family, not per bit. Susceptible to the illusion in 10.

**8. Wu, Z., Geiger, A., Icard, T., Potts, C., & Goodman, N. (2023).** Interpretability at scale: Identifying causal mechanisms in Alpaca. In *Advances in Neural Information Processing Systems, 36* (NeurIPS 2023). https://neurips.cc/virtual/2023/poster/70504 (arXiv 2305.08809) — Tier 1.
Boundless DAS learns the subspace size (soft boundary masks) instead of brute-forcing it, and scales to 7B parameters. For lc0 (d=256, 64 tokens) scale isn't the issue. The learned boundary is: it answers "how many of 256 dims does family F occupy at layer L", a capacity figure to set against the decoder's FVU. Code is in the pyvene library. Weakness: per-token, per-layer search; multi-square features need joint interventions on several tokens.

**9. Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., Greenblatt, R., Nitishinskaya, J., Radhakrishnan, A., Shlegeris, B., & Thomas, N. (2022, December 3).** Causal scrubbing: A method for rigorously testing interpretability hypotheses [Redwood Research]. *AI Alignment Forum / LessWrong*. https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing — Tier 4 (not peer-reviewed; formally subsumed by 3).
Hypothesis = correspondence from the computational graph to an interpretation. Activations that the hypothesis says are equivalent get resampled from other inputs, and the share of performance kept ("loss recovered") is the score. It is the closest existing completeness measure for a multi-layer hypothesis: resample everything the vocabulary claims is irrelevant (the decoder residual, or the orthogonal complement of DAS subspaces). The loss not recovered measures what the vocabulary misses. Weaknesses: the score can be inflated by cancelling errors and doesn't reliably reject incomplete hypotheses; it requires a resampling distribution over boards that share the feature values.

### C. Critiques and failure modes

**10. Makelov, A., Lange, G., Geiger, A., & Nanda, N. (2024).** Is this the subspace you are looking for? An interpretability illusion for subspace activation patching. In *International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=Ebt7JgMHv1 (arXiv 2311.17030; the arXiv version lists three authors) — Tier 1.
Subspace patching can reach high IIA by activating a dormant pathway: the found direction combines a component correlated with the variable and one in a causally inert subspace that becomes active off-distribution. For us this is the main risk for both DAS on probe/decoder directions and decoded-substitution patching D(f'). Proposed control: decompose the patch direction into its parts in the row space versus the nullspace of downstream weights, and check each part separately. Fits axis 8 as a required control.

**11. Wu, Z., Geiger, A., Huang, J., Arora, A., Icard, T., Potts, C., & Goodman, N. (2024).** A reply to Makelov et al. (2023)'s "interpretability illusion" arguments [Preprint]. arXiv. https://arxiv.org/abs/2401.12631 — Tier 3.
Argues that the illusion definition also flags intended, valid distributed explanations, so "illusion" isn't a decisive verdict. Use it to hedge: report the nullspace decomposition from 10 as a diagnostic, not a pass/fail gate.

**12. Sutter, D., Minder, J., Hofmann, T., & Pimentel, T. (2025).** The non-linear representation dilemma: Is causal abstraction enough for mechanistic interpretability? In *Advances in Neural Information Processing Systems, 38* (NeurIPS 2025, spotlight). https://neurips.cc/virtual/2025/poster/117348 (arXiv 2507.08802) — Tier 1.
With unrestricted (non-linear) alignment maps, any network abstracts any algorithm, and they report near-perfect IIA on randomly initialised models. This bears directly on the MLP decoder and MLP probes: a high-capacity map can create the alignment it reports. Required controls follow: (a) the same map class fitted on a randomly initialised lc0, (b) report IIA against map complexity (linear vs MLP width). It supports keeping ridge or linear maps as the primary attribution maps (axis 4, axis 7).

### D. Benchmarks for causal variable localisation

**13. Huang, J., Wu, Z., Potts, C., Geva, M., & Geiger, A. (2024).** RAVEL: Evaluating interpretability methods on disentangling language model representations. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 8669–8687). https://doi.org/10.18653/v1/2024.acl-long.470 — Tier 1.
Scores a method by Cause (an intervention on attribute A changes A) and Isolate (it leaves the correlated attribute B unchanged). Introduces multi-task DAS (MDAS), which finds subspaces satisfying both. This maps directly onto our redundancy problem ("occupant=knight" vs knight-route bits; pin vs x-ray): Isolate is the metric for whether a family-level attribution is separable from its correlates. They report that SAEs, PCA and probes lag MDAS.

**14. Mueller, A., Geiger, A., Wiegreffe, S., Arad, D., Arcuschin, I., Belfki, A., Chan, Y. S., Fiotto-Kaufman, J., Haklay, T., Hanna, M., Huang, J., Gupta, R., Nikankin, Y., Orgad, H., Prakash, N., Reusch, A., Sankaranarayanan, A., Shao, S., Stolfo, A., Tutek, M., Zur, A., Bau, D., & Belinkov, Y. (2025).** MIB: A mechanistic interpretability benchmark. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 45069–45108). https://proceedings.mlr.press/v267/mueller25a.html (arXiv 2504.13151) — Tier 1.
Two tracks: circuit localisation (bridges to T6) and causal variable localisation, which ranks featurisers (neurons, PCA, SAE, DAS) by IIA. Finding: supervised DAS is best and SAE features are no better than neurons. It is a ready-made protocol shape for our evaluation section, with feature families as the causal variables and lc0 layers as the localisation sites.

**15. Wu, Z., Arora, A., Geiger, A., Wang, Z., Huang, J., Jurafsky, D., Manning, C. D., & Potts, C. (2025).** AxBench: Steering LLMs? Even simple baselines outperform sparse autoencoders. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR 267, pp. 67035–67080). https://proceedings.mlr.press/v267/wu25a.html (arXiv 2501.17148) — Tier 1.
For concept detection and steering, supervised directions (DiffMean, linear probes) beat SAE latents. This is evidence for the project's fixed-vocabulary route over unsupervised dictionaries: per-family DiffMean directions are a cheap baseline for both decoder columns and DAS. Weakness: the domain is LLM text concepts, and steering is not attribution. A 2026 follow-up (arXiv 2605.31183, not verified here) contests the SAE result.

---

## Taxonomy coverage (axes from 01_scoping.md)

| # | Entry point | Direction | Layer span | Map | Operator | Residual | Identif. | Faithfulness test |
|---|---|---|---|---|---|---|---|---|
| 1, 2, 3 | theory | — | any | any | intervention | 3: graded faithfulness | 2: constructive vs not | IIA (3) |
| 4, 5 | readout | neither | single | identity (neurons) | interchange | ignored | non-unique | IIA |
| 6 | bottleneck | both | per-layer | learned | interchange (training) | ignored | unique if loss=0 | IIA |
| 7, 8 | readout | encode (rotation) | single per run | linear | interchange | orthogonal complement implicit | up to rotation within subspace | IIA |
| 9 | hypothesis | neither | path-level | identity | resampling ablation | loss not recovered = completeness | — | loss recovered |
| 10, 11, 12 | critiques | — | — | linear (10) / non-linear (12) | interchange | — | 12: vacuous for non-linear maps | controls |
| 13, 14, 15 | benchmarks | encode | single | linear / SAE | interchange / steering | — | 13: Isolate score | Cause/Isolate, IIA |

Gaps for synthesis: no T7 source attributes *through a fitted decoder f → h*; all align in the encode direction (rotation/projection of h). No source uses a fixed vocabulary of thousands of correlated binary variables. Cost: all methods except 9 need gradients, which means a torch port of lc0 (axis 9).

## Verification audit

| # | arXiv ID checked | Venue page checked | Status |
|---|---|---|---|
| 1 | 1707.00819 (journal_ref UAI 2017) | arXiv journal_ref | PASS |
| 2 | 1812.03789 ("Appears in AAAI-19") | arXiv comment | PASS (pages omitted, unverified) |
| 3 | 2301.04709 | jmlr.org v26/23-0058 | PASS |
| 4 | 2004.14623 (BlackboxNLP 2020) | arXiv comment | PASS (pages omitted, unverified) |
| 5 | 2106.02997 (NeurIPS 2021) | arXiv comment | PASS |
| 6 | 2112.00826 | PMLR v162 geiger22a | PASS |
| 7 | 2303.02536 | PMLR v236 geiger24a | PASS |
| 8 | 2305.08809 | neurips.cc poster 70504 | PASS |
| 9 | — | LessWrong API: title, 8 authors, 2022-12-03 | PASS (Tier 4) |
| 10 | 2311.17030 | OpenReview Ebt7JgMHv1 (ICLR 2024 poster, 4 authors) | PASS |
| 11 | 2401.12631 | arXiv only | PASS (preprint) |
| 12 | 2507.08802 | neurips.cc poster 117348 | PASS |
| 13 | 2402.17700 | ACL Anthology 2024.acl-long.470 | PASS |
| 14 | 2504.13151 | PMLR v267 mueller25a | PASS |
| 15 | 2501.17148 | PMLR v267 wu25a | PASS |

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: venue tier / author group. Concentration: Geiger or Wu as author = 11/15 (73%).
- Advisory: T7 is dominated by one Stanford lineage; critiques (10, 12) and theory (1, 2, 9) are the independent voices. No expansion: the field is this concentrated.
