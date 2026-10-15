# Phase 2 — Bibliography T1: attribution operators and their axioms

Theme: gradient, path-integral, Shapley/removal and LRP operators; their axioms; behaviour on
discrete and dependent features. Annotations say what each source means for attributing lc0
outputs to per-square feature bits through decoders D (f -> h_L), probes P (h_L -> f), or chains
of maps across layers. Axis numbers refer to the taxonomy in `01_scoping.md` §2.

## Search strategy

- Sources: arXiv API (metadata for every ID), PMLR / NeurIPS / JMLR / PNAS / Nature Comms / Elsevier
  landing pages, ICLR virtual pages and mlanthology for ICLR venues. Repo `PAPERS.md` and
  `paper/references.bib` contain no T1 entries, so nothing was reused.
- Procedure: 9 seeds from `01_scoping.md` -> one backward/forward chase, targeted at three gaps the
  DA report named: dependent/grouped features (DA Major 1, 2), attribution through a composition of
  models (the decoder chain), and attribution to hidden units (multi-layer).
- Verification: title, authors, year, venue and ID checked on the arXiv abstract/API record and on the
  publisher or proceedings page. A source whose venue could not be confirmed is cited as a preprint
  and tiered 3.

```
Records identified: 23  (9 seeds, 14 from chasing)
Wrong ID found and corrected: 1  (Lundstrom et al.: 2202.08380 is a quantum-channel paper; correct ID 2202.11912)
Excluded after screening (verified, out of scope or redundant): 6
  Chefer et al. 2012.09838 (superseded for transformers by Ali et al. and AttnLRP)
  Ancona et al. 1711.06104 (gradient-method equivalences; covered by IG/DeepLIFT/SHAP)
  Kindermans et al. 1711.00867 (input-invariance of saliency; image-specific)
  Jullum et al. groupShapley 2106.12228 (preprint; grouping covered by Aas et al.)
  Bach et al. 2015 LRP, PLoS ONE (cited inside the Ali et al. entry, not separately)
  Smilkov SmoothGrad (pure input saliency)
Failed verification (dropped): 0
Included: 17  (Tier 1: 16, Tier 3: 1)
```

---

## A. Path and gradient operators (axis 5: gradient/Jacobian, IG)

**1. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 3319–3328).** https://proceedings.mlr.press/v70/sundararajan17a.html (arXiv:1703.01365) — Tier 1.
Defines Integrated Gradients and the axioms sensitivity, implementation invariance and completeness
(attributions sum to F(x) − F(baseline)). For lc0, completeness is what makes a decoder-chained IG
useful: attributions over feature bits plus an error term sum exactly to the value change. Weakness
here: the straight path from a zero baseline in {0,1}^30837 passes through fractional, logically
inconsistent feature vectors (DA Major 2), and the result depends on the baseline. Needs a
differentiable lc0 (torch port; ONNX on CPU gives no backward pass).

**2. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning important features through propagating activation differences. In *Proceedings of the 34th International Conference on Machine Learning* (PMLR 70, pp. 3145–3153).** https://proceedings.mlr.press/v70/shrikumar17a.html (arXiv:1704.02685) — Tier 1.
DeepLIFT: backpropagates differences from a reference activation with rescale/reveal-cancel rules;
satisfies summation-to-delta, one-pass cost. Relevant as the cheap, reference-based alternative to
IG when chaining D into lc0, and as the base rule for G-DeepSHAP (entry 12). Weakness: rule-based
propagation through attention and LayerNorm is not principled, and the reference choice carries
the same off-manifold problem as the IG baseline.

**3. Lundstrom, D. D., Huang, T., & Razaviyayn, M. (2022). A rigorous study of integrated gradients method and extensions to internal neuron attributions. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 14485–14508).** https://proceedings.mlr.press/v162/lundstrom22a.html (arXiv:2202.11912) — Tier 1.
Shows IG's uniqueness claim needs an extra axiom (non-decreasing positivity), gives Lipschitz
conditions, extends single-baseline axioms to distributional baselines, and gives internal-neuron
IG. For lc0: distributional baselines (average over real positions' feature vectors) are the
principled fix for the zero-bit baseline; the internal-neuron result supports attributing through
a layer-L decoder rather than only at the input. Axis 7: uniqueness is conditional on the axiom set.

**4. Dhamdhere, K., Sundararajan, M., & Yan, Q. (2019). How important is a neuron? In *International Conference on Learning Representations*.** https://iclr.cc/virtual/2019/poster/927 (arXiv:1805.12233) — Tier 1.
Conductance: IG decomposed by the chain rule into the flow through each hidden unit. This is the
closest formal template for the user's multi-layer question: conductance through the subspace
spanned by a layer-L decoder's columns, versus through its orthogonal complement, splits the
output into "named" and "residual" flow at every layer with completeness preserved (axis 3, 6).
Weakness: defined for units, not learned non-orthogonal directions; correlated decoder columns
make the split non-unique (axis 7).

## B. Shapley and removal operators, and the dependence problem (axis 5: Shapley/occlusion; axis 7)

**5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774).** https://papers.nips.cc/paper/7062-a (arXiv:1705.07874) — Tier 1.
SHAP: additive feature attribution with local accuracy, missingness and consistency; unifies LIME,
DeepLIFT and Shapley sampling; KernelSHAP assumes feature independence. Defines the operator class
for "occlusion/Shapley over feature bits or families" through D. Weakness for lc0: independence
assumption is false for derived bits (occupant vs route bits), and 30k players is infeasible
without grouping by family.

**6. Sundararajan, M., & Najmi, A. (2020). The many Shapley values for model explanation. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 9269–9278).** https://proceedings.mlr.press/v119/sundararajan20b.html (arXiv:1908.08474) — Tier 1.
Shows the Shapley uniqueness theorem does not pin down a method: conditional, interventional and
baseline Shapley give different outputs, and data-conditional variants credit features the model
never reads. Proposes Baseline Shapley (BShap) with a proper uniqueness result. For lc0: BShap over
feature families with a real-position baseline is the cleanest identifiable variant; the "credit
to unused features" result is a direct warning for correlated engine-derived bits.

**7. Kumar, I. E., Venkatasubramanian, S., Scheidegger, C., & Friedler, S. (2020). Problems with Shapley-value-based explanations as feature importance measures. In *Proceedings of the 37th International Conference on Machine Learning* (PMLR 119, pp. 5491–5500).** https://proceedings.mlr.press/v119/kumar20e.html (arXiv:2002.11097) — Tier 1.
Argues both interventional and conditional Shapley fail as importance measures: the former leaves
the data manifold, the latter credits non-causal correlates; additive scores do not match human
use. Supports the DA counter-argument that only interventional ground truth (MaskPinner,
LineBlocker) can settle attributions. No fix offered; a critique source.

**8. Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance quantification in explainable AI: A causal problem. In *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics* (PMLR 108, pp. 2907–2916).** https://proceedings.mlr.press/v108/janzing20a.html (arXiv:1910.13413) — Tier 1.
Argues feature removal should use interventional (marginal) expectations: one explains the
model's input-output function, so do(x_S) on model inputs, not observational conditioning. For lc0
this cuts the other way from Frye: if the "model" is lc0∘D, flipping a bit in f is an intervention
on D's input and is legitimate as a statement about lc0∘D, but says nothing about chess positions.
Use to state which object an attribution explains.

**9. Aas, K., Jullum, M., & Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Artificial Intelligence, 298*, Article 103502.** https://doi.org/10.1016/j.artint.2021.103502 (arXiv:1903.10464) — Tier 1.
Extends KernelSHAP to dependent features via conditional distributions (Gaussian, copula,
empirical) and shows independence-based SHAP can be badly wrong under dependence; proposes summing
Shapley values over groups of dependent variables. Directly answers DA "what's missing": grouped
attribution over feature families. Weakness: conditional estimators are for continuous tabular data;
binary, logically constrained bits need an empirical (nearest-real-position) sampler.

**10. Frye, C., de Mijolla, D., Begley, T., Cowton, L., Stanley, M., & Feige, I. (2021). Shapley explainability on the data manifold. In *International Conference on Learning Representations*.** https://iclr.cc/virtual/2021/poster/2856 (arXiv:2006.01272) — Tier 1.
On-manifold Shapley via a generative imputer or by learning the value function directly; shows
off-manifold Shapley can give wrong and high-dimensionally unintelligible explanations. For lc0:
the feature extractor itself is a perfect on-manifold sampler if one edits boards, not bits —
the board-level counterfactual the DA recommends. Weakness: inherits Kumar's "credit to
correlates" problem.

**11. Chen, H., Janizek, J. D., Lundberg, S., & Lee, S.-I. (2020). *True to the model or true to the data?* arXiv.** https://arxiv.org/abs/2006.16234 — Tier 3 (preprint; no venue confirmed).
Frames the interventional vs conditional choice as a goal choice: interventional = true to the
model, conditional = true to the data; both are correct for different questions. Gives the paper a
vocabulary for the decoder route: bit-flip attribution through D is true to lc0∘D, board-edit
attribution is true to chess. Hedge empirical claims.

**12. Covert, I., Lundberg, S., & Lee, S.-I. (2021). Explaining by removing: A unified framework for model explanation. *Journal of Machine Learning Research, 22*(209), 1–90.** https://jmlr.org/papers/v22/20-1316.html (arXiv:2011.14878) — Tier 1.
Classifies 26 methods by (how features are removed, what behaviour is explained, how influence is
summarised). Use as the formal backbone of axis 5: every decoder/probe operator in this paper can
be placed by its removal rule (zero bit, baseline position, conditional sample, board edit). Survey;
no new operator.

## C. Attribution through compositions of models (axis 3: chained maps)

**13. Chen, H., Lundberg, S. M., & Lee, S.-I. (2022). Explaining a series of models by propagating Shapley values. *Nature Communications, 13*, Article 4512.** https://doi.org/10.1038/s41467-022-31384-3 (arXiv:2105.00108) — Tier 1.
G-DeepSHAP: propagates Shapley-based attributions through a chain of models (feature extractor ->
model -> score) using DeepLIFT-style rules with a baseline distribution. The nearest published
precedent for "f -> D -> h_L -> lc0 blocks L..9 -> value", treating D as the first model of the
series. Weakness: approximation quality through attention untested; the residual h_L − D(f) has no
explicit slot and must be added as its own input.

## D. Relevance propagation for transformers (axis 5: conservative decomposition)

**14. Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R., & Wolf, L. (2022). XAI for transformers: Better explanations through conservative propagation. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 435–451).** https://proceedings.mlr.press/v162/ali22a.html (arXiv:2202.07304) — Tier 1.
Shows Gradient×Input breaks conservation through attention and LayerNorm; fixes it by detaching
the attention weights and LN scale in the backward pass (LRP-style, building on Bach et al., 2015,
PLoS ONE, https://doi.org/10.1371/journal.pone.0130140). lc0 is post-LN with smolgen attention, so
naive gradients from value to a layer-L decoder are expected to violate conservation; this is the
cheap fix. Needs a torch port; smolgen not covered.

**15. Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., & Samek, W. (2024). AttnLRP: Attention-aware layer-wise relevance propagation for transformers. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR 235, pp. 135–168).** https://proceedings.mlr.press/v235/achtibat24a.html (arXiv:2402.05602) — Tier 1.
LRP rules for softmax attention and matrix multiplication, one backward pass, attributes to both
inputs and latent representations. Relevance at a hidden layer can be projected onto decoder
directions, giving a conservative per-layer feature attribution plus residual. Weakness: LRP
conservation is a design property, not a faithfulness guarantee; still needs interventional checks.

## E. Validity limits of attribution (axis 8)

**16. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. In *Advances in Neural Information Processing Systems 31* (pp. 9505–9515).** https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps (arXiv:1810.03292) — Tier 1.
Model- and data-randomisation tests: several saliency methods are invariant to both. Transfers as
two cheap controls for the protocol: re-randomise lc0's top blocks, and re-fit D on shuffled
feature labels; a decoder-route attribution that survives either is decoder structure, not lc0.

**17. Bilodeau, B., Jaques, N., Koh, P. W., & Kim, B. (2024). Impossibility theorems for feature attribution. *Proceedings of the National Academy of Sciences, 121*(2), e2304406120.** https://doi.org/10.1073/pnas.2304406120 (arXiv:2212.11870) — Tier 1.
Proves complete and linear methods (IG, SHAP) can be no better than random at inferring model
behaviour (e.g. algorithmic recourse, spurious-feature detection) for rich model classes; argues
attributions must be tied to a concrete end-task and tested by repeated model queries. Strongest
formal support for the project's preference for interventional ground truth over axiom-satisfying
scores.

---

## Coverage notes for Phase 3 (not synthesis)

- Axes filled: 5 (all operator families), 7 (6, 7, 8, 9, 11), 8 (16, 17), 3 (4, 13), 6 (4, 13 only
  implicitly). No T1 source treats the decoder residual as a first-class attribution target; that
  comes from T4–T6 (error nodes, PCBM-h).
- No source handles binary, logically constrained features directly; closest are 9 (grouping) and
  10 (on-manifold). Board-edit counterfactuals remain a project-specific design.
- Skew advisory: venue concentration in PMLR (8 of 17); author concentration in Lundberg/Lee
  (4) and Sundararajan (3). Justified by the topic; noted for Phase 3.
