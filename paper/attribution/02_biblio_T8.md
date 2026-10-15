# Phase 2 — Bibliography T8: interpretability of game-playing networks

Search date 2026-10-09. Theme definition, inclusion criteria and tiers: `01_scoping.md` §2-3.

## Search strategy

- Sources: arXiv API (metadata for every arXiv ID), Crossref (every DOI), NeurIPS / ACL Anthology / IJCAI / IOS Press / PNAS landing pages, OpenReview, Alignment Forum (Tier 4 only). Seeds from `PAPERS.md`.
- Keywords: AlphaZero concepts; Leela Chess Zero interpretability; lc0 sparse autoencoder / transcoder; chess transformer probing; look-ahead; Othello world model; Go concept probing; KataGo / Leela Zero interpretability; concept probing reliability.
- Chase: backward from Jenner, Lin, Karvonen et al.; forward via search results. Stopped after two rounds without new in-scope hits.

Flow: 26 records identified (10 scoping seeds, 16 from search/chase) → 3 out of scope after screening → 2 failed verification → 6 verified but not selected (redundant or input-saliency only) → **15 included**.

## Verification

Every entry below was checked against the arXiv API record and, where a DOI exists, Crossref. Venues were confirmed on the venue's own page (proceedings listing, ACL Anthology, Crossref container title, or the arXiv comment field naming the venue).

| # | Short ref | Venue confirmed via | Tier | Outcome |
|---|---|---|---|---|
| 1 | McGrath et al. 2022 | Crossref (PNAS 119(47)) | 1 | VERIFIED |
| 2 | Schut et al. 2025 | Crossref (PNAS 122(13)) | 1 | VERIFIED |
| 3 | Pálsson & Björnsson 2024 | Crossref + IOS Press (ECAI, FAIA 392) | 1 | VERIFIED |
| 4 | Hammersborg & Strümke 2024 | Crossref (Sci. Rep. 14) | 1 | VERIFIED |
| 5 | Jenner et al. 2024 | NeurIPS 2024 proceedings listing | 1 | VERIFIED |
| 6 | Sandmann et al. 2025 | arXiv (no venue) | 3 | PLAUSIBLE |
| 7 | Lin et al. 2026 | arXiv (no venue) | 3 | PLAUSIBLE |
| 8 | Poupart 2024 | arXiv comment (InterpPol @ RLC workshop) | 3 | PLAUSIBLE |
| 9 | Karvonen 2024 | arXiv comment (COLM 2024) | 1 | PLAUSIBLE (venue from author comment) |
| 10 | Karvonen et al. 2024 | NeurIPS 2024 proceedings listing | 1 | VERIFIED |
| 11 | Li et al. 2023 | arXiv comment + OpenReview ID (ICLR 2023 oral) | 1 | PLAUSIBLE (venue from author comment) |
| 12 | Nanda et al. 2023 | Crossref (BlackboxNLP 2023) | 1 | VERIFIED |
| 13 | Tomlin et al. 2022 | ACL Anthology (ACL 2022 short) | 1 | VERIFIED |
| 14 | Lomasov et al. 2025 | NeurIPS 2025 proceedings page (Creative AI track) | 1 (non-main track) | VERIFIED |
| 15 | Du 2023 | LessWrong/AF post header | 4 | PLAUSIBLE (gray lit, labelled) |

No predatory venues. COI: Schut and McGrath are DeepMind authors evaluating DeepMind's own model (intellectual, low). Lin et al. and He et al. share a lab (Fudan, Qiu), which matters for the transcoder/Lorsa method lineage.

**Dropped (failed verification)**
- Jiao, Eilender, Tang & Anderson, "Understanding Skill Adaptation in Transformers Using Sparse Autoencoders: Chess as a Model System" (OpenReview Wxl0JMgDoU, listed in `PAPERS.md` [53]). OpenReview blocked automated access; no search hit for the exact title. The only related artifact is a GitHub repository (CSSLab/maia2-skill-adaptation) with a different title and no citation. Gray zone → FAIL. `PAPERS.md` [53] should be re-checked by hand.
- "Go concept probing with amnesic probes", as one search summary described it: the cited paper (FAIA240574) is about chess, not Go. The Go claim is false, and the paper itself is kept as #3.

**Screened out (out of scope)**: Mészáros et al. 2025 (2510.20783; behavioural OOD tests only, no internal maps); Spinnato 2025 (2510.25775; SHAP over pieces on input, no concept layer); Wu 2019 KataGo (1902.10565; training paper).

**Verified, not selected (redundant)**: Pálsson & Björnsson IJCAI 2023 (doi:10.24963/ijcai.2023/541; Stockfish NNUE probing, superseded for our purposes by #3); Cruz 2025 (2505.21552; extends #5); He et al. 2024 (2402.12201; Othello-GPT dictionary circuits, same lineage as #7); Puri et al. ICLR 2020 (1912.12191; SARFA input saliency for chess agents, baseline only); Monroe et al. 2026 Chessformer (2605.19091, ICLR 2026 per arXiv comment; architecture paper whose interpretability part is secondary); Hammersborg & Strümke 2022 (2211.05500).

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: domain. Concentration: chess = 11/15 (73%).
- Advisory: this is a coverage signal, not a defect. Go has one Tier 1 probing paper (#13) and one Tier 4 causal study (#15). No peer-reviewed KataGo/Leela Zero intervention or dictionary paper was found. The skew matches the system under study (lc0).

## Annotated bibliography (APA 7)

Axis numbers refer to the taxonomy in `01_scoping.md` §2: 1 entry point, 2 direction, 3 layer span, 4 map class, 5 attribution operator, 6 residual handling, 7 identifiability, 8 faithfulness test, 9 cost.

### 1. McGrath et al. (2022) — Tier 1
McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Wattenberg, M., Hassabis, D., Kim, B., Paquet, U., & Kramnik, V. (2022). Acquisition of chess knowledge in AlphaZero. *Proceedings of the National Academy of Sciences, 119*(47), e2206625119. https://doi.org/10.1073/pnas.2206625119 (arXiv:2111.09259)

Fits sparse linear regression probes from each AlphaZero residual block to Stockfish evaluation-function concepts, tracked over layers and training steps ("what-when-where"). Fills axes 1-4 as readout-side, encode, per-layer independent, sparse-linear. It is the direct ancestor of our encoder route. Weaknesses: correlational only (axis 8 = none). Concepts are scalar board-level terms, not per-square. There is no attribution to the output and no residual measure. The arXiv author list omits Wattenberg; the PNAS version includes him.

### 2. Schut et al. (2025) — Tier 1
Schut, L., Tomašev, N., McGrath, T., Hassabis, D., Paquet, U., & Kim, B. (2025). Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero. *Proceedings of the National Academy of Sciences, 122*(13), e2406675122. https://doi.org/10.1073/pnas.2406675122 (arXiv:2310.16410)

Finds sparse concept vectors by convex optimization over contrasting MCTS rollouts, then filters them for teachability (transfer to a student network) and novelty (not present in human games). It is the closest published attempt to name what a human vocabulary misses, the project's residual question (axis 6, "residual decomposed further"). Validation is behavioural: a student network and four grandmasters. There is no interventional faithfulness test on AlphaZero itself. Concepts are discovered, not mapped from a fixed vocabulary, so the paper serves our residual-naming step rather than attribution.

### 3. Pálsson & Björnsson (2024) — Tier 1
Pálsson, A., & Björnsson, Y. (2024). Empirical evaluation of concept probing for game-playing agents. In *ECAI 2024* (Frontiers in Artificial Intelligence and Applications, Vol. 392, pp. 874–881). IOS Press. https://doi.org/10.3233/FAIA240574

Tests probe architectures on a world-class chess agent and measures concept importance through self-play. Linear-probe accuracy is an unreliable proxy for importance; the authors recommend nonlinear probes and amnesic-style erasure. This is chess-specific evidence for DA Major 4 (readable ≠ used) and supports grading the encoder route by erasure effect (axes 5 and 8). Weakness: the agent and concept set differ from lc0 and our 30k-bit vocabulary, so the size of the gap does not transfer.

### 4. Hammersborg & Strümke (2024) — Tier 1
Hammersborg, P., & Strümke, I. (2024). Information based explanation methods for deep learning agents—with applications on large open-source chess models. *Scientific Reports, 14*, 20174. https://doi.org/10.1038/s41598-024-70701-2 (arXiv:2309.09702)

Replicates the McGrath concept detection on open-source lc0 networks with similar results, which shows the probe findings transfer to lc0. It adds an input-masking method that, by construction, controls exactly which input information reaches the model. This is a principled occlusion operator for discrete inputs (axis 5), but it works on board planes, not on a concept layer. Useful as the input-side baseline and as a template for consistent, on-manifold deletion (DA Major 2).

### 5. Jenner et al. (2024) — Tier 1
Jenner, E., Kapur, S., Georgiev, V., Allen, C., Emmons, S., & Russell, S. (2024). Evidence of learned look-ahead in a chess-playing neural network. In *Advances in Neural Information Processing Systems 37* (NeurIPS 2024). https://arxiv.org/abs/2406.00877

Uses per-square activation patching on the lc0 transformer, attention-head ablation, and a bilinear probe that predicts the move two plies ahead (92% in single-best-line positions). It establishes square tokens as the natural patching unit (axis 5 = patching on identity maps). The probe is validated by causal effect, not accuracy alone. This is the strongest precedent for interventional ground truth on lc0. Limits: the analysis covers puzzle positions with a unique best line, and the attributed concept is "future-move square", not a feature vocabulary.

### 6. Sandmann, Lapuschkin & Samek (2025) — Tier 3
Sandmann, E., Lapuschkin, S., & Samek, W. (2025). *The algorithm is not the behavior: Learned priors override look-ahead in a chess-playing neural network* (arXiv:2508.21380) [Preprint]. arXiv. https://arxiv.org/abs/2508.21380

Applies a logit lens to lc0's policy head and finds "forgotten puzzles": correct solutions are top-ranked in middle layers and overridden late. Steering against a safety prior recovers 61.7% of them. Relevance to the decoder route: what a layer encodes can be the opposite of what the output uses, so per-layer attributions must be chained to the output, not read per layer (axis 3, path length). Not peer-reviewed. The steering direction is found post hoc.

### 7. Lin et al. (2026) — Tier 3
Lin, R., Jin, Z., Zhou, G., Ge, X., Shu, W., Wu, J., Wang, J., He, Z., Zhang, J., & Qiu, X. (2026). *Tracing the thought of a grandmaster-level chess-playing transformer* (arXiv:2604.10158) [Preprint]. arXiv. https://arxiv.org/abs/2604.10158

Replaces every MLP (transcoder) and attention module (Lorsa) of lc0 BT4-1024x15x32h with sparse layers: 16,384 features per layer, top-k 30 per square. Explained variance is >0.6 per module, lower for attention in middle layers. Features are labelled by hand and then checked against rule-based chess-concept detectors (precision and recall), and pathways are validated by ablation and copy-patching. This is the closest prior work to "several maps on several layers" for lc0 (axes 3-4 = per-layer sparse replacement). It also points to our alternative: rule-based detectors used as the vocabulary from the start rather than to label features afterwards. Weaknesses: residual handling in the graphs is not stated; 30-40% of variance is unexplained per module, the same scale as our L5 decoder.

### 8. Poupart (2024) — Tier 3
Poupart, Y. (2024). *Contrastive sparse autoencoders for interpreting planning of chess-playing agents* (arXiv:2406.04028) [Preprint; Workshop on Interpretable Policies in RL @ RLC 2024]. arXiv. https://arxiv.org/abs/2406.04028

Trains SAEs on differences between pairs of lc0 trajectories, so features encode plan-relevant contrasts. Includes sanity checks against spurious correlation. Relevant as an unsupervised tool for naming the residual (scoping: SAEs only in that role) and as a contrastive framing that matches our real-board interventions (pinned vs MaskPinner). Weaknesses: mostly qualitative, workshop-level, no output attribution.

### 9. Karvonen (2024) — Tier 1
Karvonen, A. (2024). Emergent world models and latent variable estimation in chess-playing language models. In *Proceedings of the Conference on Language Modeling (COLM 2024)*. https://arxiv.org/abs/2403.15498

Linear probes on a PGN-trained GPT recover board state and a player-skill latent, and adding probe directions edits both board state and play strength. This is an encoder route validated by intervention along probe directions (axes 2 and 5). It is a chess precedent for the probe-direction interchange that DAS formalizes. Weakness: the probe-direction edit can create interpretability illusions (Makelov et al., T7). The model is a sequence model, not a board-token network.

### 10. Karvonen et al. (2024) — Tier 1
Karvonen, A., Wright, B., Rager, C., Angell, R., Brinkmann, J., Smith, L., Verdun, C. M., Bau, D., & Marks, S. (2024). Measuring progress in dictionary learning for language model interpretability with board game models. In *Advances in Neural Information Processing Systems 37* (NeurIPS 2024). https://arxiv.org/abs/2408.00113

Scores SAEs on chess and Othello LMs by how well latents match a predefined set of board-state properties (coverage, board reconstruction). This is the only Tier 1 work that uses a fixed, rule-computed game vocabulary as ground truth for dictionaries. It provides the supervised-dictionary benchmark logic for our "SAE latents vs 30k bits" comparison (axis 1 = dictionary-matched). Limits: the vocabulary is small and occupancy-centred, and it evaluates representation only, not attribution to outputs.

### 11. Li et al. (2023) — Tier 1
Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Emergent world representations: Exploring a sequence model trained on a synthetic task. In *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2210.13382

Nonlinear (MLP) probes recover the Othello board from Othello-GPT. Board edits made by gradient steps against the probe change the predicted legal moves, and "attribution via intervention" maps which tiles drive a prediction. This is the template for the nonlinear encoder route with probe-guided intervention (axes 4-5) and for latent saliency over board squares. Weakness: nonlinear probes raise the probe-expressivity concern (Hewitt & Liang, T3), which #12 later resolved for this model.

### 12. Nanda, Lee & Wattenberg (2023) — Tier 1
Nanda, N., Lee, A., & Wattenberg, M. (2023). Emergent linear representations in world models of self-supervised sequence models. In *Proceedings of the 6th BlackboxNLP Workshop* (pp. 16–30). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.blackboxnlp-1.2

Shows that the Othello board becomes linearly decodable once it is encoded as "mine/theirs" rather than "black/white", and steers the model with vector arithmetic. Lesson for our vocabulary: a linear map's apparent coverage depends on the feature encoding chosen, and a side-to-move-relative encoding can turn "nonlinear" into "linear". It bears on identifiability (axis 7) and on how decoder FVU should be read. Workshop paper, but widely replicated.

### 13. Tomlin, He & Klein (2022) — Tier 1
Tomlin, N., He, A., & Klein, D. (2022). Understanding game-playing agents with natural language annotations. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 797–807). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.acl-short.90

Builds linear probes from Go policy-network layers (imitation-learned and AlphaGo-Zero-style RL) to mentions of terms such as ko and atari in 10k annotated games. The terms are best predicted from later layers. This is the Go precedent for probing a fixed human vocabulary per layer (axes 1-3). Its layer profile runs opposite to ours, where named features peak early. Weaknesses: noisy labels from comment mentions; correlational only.

### 14. Lomasov et al. (2025) — Tier 1 (NeurIPS Creative AI track)
Lomasov, S., Goldfeder, J., Erol, M. H., So, M., Yan, Y., Howard, A., Kutz, N., & Shwartz-Ziv, R. (2025). Exploring human-AI conceptual alignment through the prism of chess. In *Advances in Neural Information Processing Systems 38* (NeurIPS 2025, Creative AI Track). https://arxiv.org/abs/2510.26025

Probes a 270M grandmaster-level chess transformer: human concepts reach up to 85% probe accuracy in early layers and fall to 50-65% in deep layers. Concept recognition drops 10-20% on Chess960. This is independent evidence for the same decay our decoder shows (features add 0.63 FVU at L2 vs 0.22 at L9), and it supports treating late layers as vocabulary-poor (axis 6). Weaknesses: six concepts, 240 annotated Chess960 positions, probe accuracy only, non-main track.

### 15. Du (2023) — Tier 4 (gray literature)
Du, H. (2023, March 1). *Inside the mind of a superhuman Go model: How does Leela Zero read ladders?* LessWrong / AI Alignment Forum. https://www.lesswrong.com/posts/FF8i6SLfKb4g7C4EL/inside-the-mind-of-a-superhuman-go-model-how-does-leela-zero

Rewrites Leela Zero (Go, conv-resnet) as an explicit residual network, applies a logit lens through the policy head, and localizes ladder-breaker detection with path patching and causal scrubbing. This is the only Go source found with path-level interventions (axis 3 = path-level, axis 8 = causal scrubbing). It shows a residual-stream preferred basis that makes per-channel reading feasible. Not peer-reviewed (Redwood internship write-up); it is used only as method precedent for Go.

## Gaps for Phase 3

- No game-network paper attributes outputs through a fitted feature→activation decoder; all prior work is encoder-side (probes) or unsupervised dictionaries. This supports the novelty claim in `01_scoping.md`.
- No game-network paper reports an error node or residual share in output attribution. Lin et al. report reconstruction quality per module but not in graphs.
- Interventional validation on lc0 exists for square-level patching (#5, #6, #7), not for named feature families.
