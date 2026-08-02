# Deep-research summaries — xai-chess / ChessFaith corpus

One deep-research pass per paper (a sub-agent each): what it does, what it depends on, its results, its weaknesses, and its relation to the intervention-grounded chess-explanation benchmark. Generated from the verified 64-paper corpus; each agent re-fetched its source and cross-checked the project's prior full-read notes.

*87 papers · grouped by relevance tier · 2026-06-27, extended 2026-08-01.*

Entries 1–64 are the original corpus. Entries 65–87 come from the move-explanation and dataset sweep of 2026-08-01 and are grouped separately at the end; their numbering matches `PAPERS.md`. Those entries cite authors by surname, because titles, venues and identifiers were confirmed against fetched sources but author initials were not — see the citation convention note in `PAPERS.md`.


---

## Contents

- **Core** (23)
- **Supporting** (33)
- **Peripheral** (8)

---


# Core

*Directly load-bearing: the methods, oracles, explanation engines, and faithfulness machinery the benchmark builds on or grades.*


## 1. Acquisition of Chess Knowledge in AlphaZero

**Citation:** T. McGrath, A. Kapishnikov, N. Tomašev, A. Pearce, D. Hassabis, B. Kim, et al., "[Acquisition of chess knowledge in AlphaZero](https://arxiv.org/abs/2111.09259)," arXiv:2111.09259, 2021.  
**Link:** <https://arxiv.org/abs/2111.09259> · **Tier:** CORE

**Summary.** McGrath et al. ask whether AlphaZero's chess network internally encodes human chess concepts, and when/where they emerge during training. They train sparse L1-regularized linear probes (regression for continuous concepts, classification for binary) on the 16,384-dimensional per-layer activations to predict pre-defined human concepts (e.g., Stockfish evaluation terms, material, threats), producing "what-when-where" plots over training steps and network depth. They add non-negative matrix factorization (NMF) to discover move-selection factors not tied to predefined concepts, a behavioral comparison of AlphaZero's opening-play evolution against human history, and qualitative commentary from GM Vladimir Kramnik. They find human-aligned knowledge is acquired, with many concepts emerging rapidly around 32,000 steps and material/eval probes reaching high R² (>0.75 after ~10 layers). Crucially, the authors state probing measures "only a proximate correlation… and not causation in any forms," and that a probe "cannot tell if we are getting a confound or the concept itself."

**Depends on.** Linear/concept probing and TCAV-style concept methods (Been Kim is a coauthor); Stockfish 8 evaluation terms as the concept vocabulary; AlphaZero self-play training; NMF for activation decomposition.

**Results.** Probes recover human concepts with high fidelity: Stockfish total-score R² exceeds 0.75 after ~10 layers at 64,000+ training steps; piece values converge toward conventional chess theory after ~128,000 steps; many concepts begin sharp accuracy gains around 32,000 steps ("rapid development" phase). Some concepts drop in later layers, suggesting information is discarded or re-encoded nonlinearly. NMF surfaces move-selection factors; behavioral analysis finds AlphaZero self-play shares similarities with but does not recapitulate human opening history. Explicitly disclaims causal interpretation.

**Weaknesses.** Self-acknowledged: probing is purely correlational ("not causation in any forms")—high probe accuracy proves decodability, not that the network uses the concept for decisions. Concept vocabulary is anchored to Stockfish 8 terms and human priors, so undiscovered/non-human representations are missed by design. Confounding is unresolved: the paper itself flags that under an absolute pin, can_capture_queen cannot be intervened to 0 without forcing in_check=1 (plus other concepts), so probe signals can be entangled. Linear probes miss nonlinearly encoded information. Single-agent (AlphaZero) study; behavioral and Kramnik analyses are qualitative.

**Relation to xai-chess.** This is the canonical correlational concept-probing baseline that names the very pin / can_capture_queen-vs-in_check confound xai-chess resolves by causal intervention (board-level relocation and rule-level movegen patching), so it motivates the benchmark but must not be credited with causal validation.


## 2. Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero

**Citation:** L. Schut, N. Tomasev, T. McGrath, D. Hassabis, U. Paquet, and B. Kim, "[Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero](https://arxiv.org/abs/2310.16410)," arXiv:2310.16410, 2023.  
**Link:** <https://arxiv.org/abs/2310.16410> · **Tier:** CORE

**Summary.** The paper proposes a method to extract chess concepts from AlphaZero that may exceed current human knowledge yet remain learnable by experts. Concepts are operationalized as sparse linear vectors in AZ's latent space, found via convex optimization that separates positive from negative examples (for static concepts) or optimal MCTS rollouts from suboptimal trajectories (for dynamic/plan concepts). Candidates are filtered by machine teachability (a student network must learn the concept via policy distillation, removing ~97.6%) and by novelty (lower reconstruction error using an AZ-game basis than a human-game basis; spectral/SVD analysis shows AZ games have higher rank at deeper layers, indicating concepts absent from human play). The surviving M-H ("machine minus human") concepts were validated through a three-phase teachability study: four grandmasters (2600-2800 Elo) solved prototype puzzles, saw AZ's top line, then were tested on fresh puzzles; all four improved (e.g., 0%->42%). The approach is representational and behavioral, not a causal-intervention faithfulness test.

**Depends on.** Builds on McGrath et al.'s probing of human chess concepts in AlphaZero and the linear-representation/concept-vector hypothesis (TCAV-style); uses AlphaZero's policy-value network plus MCTS and policy distillation to a student network.

**Results.** Two-stage filtering: machine-teachability removed ~97.6% of initial concepts, novelty filtering removed a further 27.1% of remaining concepts. Grandmaster teachability study (4 GMs, 36-48 puzzles each across 3-4 concepts): all improved, GM1 0%->42%, GM2 33%->58%, GM3 25%->42%, GM4 38%->44%. SVD shows AZ games occupy higher-rank subspaces at layers 19 and 23, evidence of concepts not present in human games.

**Weaknesses.** Relies on a strong linearity assumption for concept encoding (acknowledged); nonlinear concepts go undiscovered. Teachability is primarily defined via teaching a student AI, which may not transfer to humans. Human validation is severely underpowered (n=4 GMs, randomized concept assignment, varying puzzle difficulty, no control group), and improvement could reflect general puzzle exposure rather than learning the specific concept; some GMs cited AZ's move but did not play it. Novelty depends on the chosen layer and the human-game corpus used to build the basis. Crucially for faithfulness work, the method establishes that a concept vector is decodable/teachable and correlates with strong play, but never tests whether the concept is causally load-bearing for a specific move decision (no intervention on the concept and re-evaluation of move quality).

**Relation to xai-chess.** It supplies a concept-discovery counterpoint to xai-chess: it finds and human-validates chess concepts beyond a fixed vocabulary but stops at teachability/correlation, exactly the decision-level causal ground truth that ChessFaith's intervention benchmark adds.


## 3. Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models

**Citation:** A. Karvonen, "[Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498)," in Proc. Conf. on Language Modeling (COLM), 2024, arXiv:2403.15498.  
**Link:** <https://arxiv.org/abs/2403.15498> · **Tier:** CORE

**Summary.** Karvonen trains GPT models (8-layer/25M and 16-layer/50M params, 512-dim) purely on next-character prediction over 16M Lichess PGN games, with no explicit chess knowledge, and asks what internal structure emerges. Linear probes recover the board state per square at up to 99.6% accuracy, evidencing an emergent world model rather than surface statistics. Beyond board state, the model implicitly estimates a latent variable, player skill: linear probes classify low (<1550) vs high (>2050) Elo at 90.5%/88.6% vs ~69-70% randomized baselines. Crucially, the paper goes white-box-causal: activation interventions that delete a piece from the internal board representation succeed 90.4-92.0% of the time, and a contrastive "skill vector" added to activations raises win rate against Stockfish level 0 by up to 2.6x on random boards (16.7%->43.2%). This establishes that the learned representations are not merely correlational but causally drive the model's move outputs.

**Depends on.** Builds directly on Li et al.'s Othello-GPT emergent-world-model and probing work; uses linear/contrastive probing, activation patching/steering, and Stockfish-level evaluation; trains on Lichess game data.

**Results.** Board-state linear probe up to 99.6% per-square accuracy (8-layer peaks layer 5, 16-layer layer 11). Skill probe 90.5% (16L)/88.6% (8L) vs ~69-70% baseline. Board-edit interventions (piece deletion) succeed 92.0%/90.4%. Skill-vector steering: win rate 16.7%->43.2% (2.6x) on random boards, ~2x for 8-layer; only modest 69.6%->72.3% on standard boards.

**Weaknesses.** Subject is a small custom chess LM trained from scratch, not a general LLM producing natural-language explanations, so transfer to explanation faithfulness is indirect. Interventions edit internal activations (white-box), requiring full model access and a learned probe, unlike the project's black-box board/rule-level edits. "Success" of board-edit interventions is judged by downstream legal-move consistency, a proxy that conflates probe quality with causal effect. Skill steering gains are large only on out-of-distribution random boards; near-zero practical lift on standard play. No notion of grading a stated reason/factor; causality is over board occupancy and skill, not over a specific chess concept like a pin. Single author, two model sizes, one game domain limit generality.

**Relation to xai-chess.** It is the white-box analog of the project's idea: where ChessFaith causally neutralizes a cited factor externally (board/rule-level edits to an oracle), Karvonen shows internal activation interventions can causally manipulate a chess model's emergent board/skill representations, motivating causal-intervention as the test of whether a representation (or explanation) is load-bearing.


## 4. Evidence of Learned Look-Ahead in a Chess-Playing Neural Network

**Citation:** E. Jenner, S. Kapur, V. Georgiev, C. Allen, S. Emmons, and S. Russell, "[Evidence of Learned Look-Ahead in a Chess-Playing Neural Network](https://arxiv.org/abs/2406.00877)," arXiv:2406.00877, 2024.  
**Link:** <https://arxiv.org/abs/2406.00877> · **Tier:** CORE

**Summary.** The authors ask whether policy networks learn search-like algorithms, and give mechanistic evidence of "learned look-ahead" in Leela Chess Zero. They study the transformer T82-768x15x24h (15 layers, d=768, 24 heads, ~109M params), finetuned to remove board-history dependence (so corruptions can be generated automatically) while preserving strength. On a curated set of 22,500 hard Lichess tactical puzzles (>=3-move forcing PVs, solvable by Leela but not a weaker model), three converging lines of evidence are offered: (1) activation patching shows the future 3rd-move target square is causally critical (log-odds drop 1.88 vs 0.55 for other squares at layer 10); (2) specific attention heads move information "backward in time" (L12H12) and piece-type-specific movement heads route legal-move information from future-move squares; (3) a bilinear probe predicts the move two turns ahead at 92% accuracy (vs 15% random-net baseline), peaking at layer 12. The network thus internally represents and uses future optimal moves.

**Depends on.** Activation/causal patching from mechanistic interpretability (e.g., causal tracing, path patching); Leela Chess Zero transformer engine and Lichess tactical puzzle database; linear/bilinear probing methods for internal representations.

**Results.** Patching the 3rd-move target square at layer 10 reduces log-odds of the correct move by 1.88+-0.04, versus a max of 0.55+-0.01 across 61 alternative squares. Zeroing the single L12H12 attention entry (3rd->1st target) drops log-odds by >1.5 in >10% of puzzles. Ablating future-square information flow in piece-type movement heads (22 knight, 27 bishop, 29 rook heads) reduces log-odds by >=1.5 in 60% of puzzles. A bilinear probe reads off the move two turns ahead at 92%+-1% accuracy (peak layer 12) vs 15%+-2% on a randomly initialized network. Dataset filtering overrepresents sacrifices (83% share 1st/2nd-move target squares).

**Weaknesses.** Evidence is confined to a heavily filtered, non-representative subset: 22,500 of 900k puzzles chosen for single-best-line, forcing, sacrifice-heavy tactics where look-ahead is most expected; the 92% probe figure is explicitly restricted to single-best-line states, so it does not establish look-ahead across the full move distribution. The model is a finetuned, no-history variant rather than stock Leela, a confound for ecological validity. "Look-ahead" is operationalized as future-move-square representation/use, not verified multi-step planning or tree search; alternative explanations (pattern-matched tactical motifs correlated with future squares) are hard to fully exclude. Effect sizes vary widely across puzzles (e.g., head ablations hit only 10-60%), and findings are correlational-causal at the activation level, not behavioral over real games.

**Relation to xai-chess.** It supplies the methodological template and a verifiable causal-ground-truth precedent for ChessFaith: just as Jenner et al. use activation patching to show a feature (future-move square) is load-bearing for an engine's decision, ChessFaith uses board- and rule-level interventions to test whether an explanation's cited factor causally drives a strong oracle's move preference.


## 5. Iterative Inference in a Chess-Playing NN (v3: The Algorithm Is Not the Behavior: Learned Priors Override Look-Ahead)

**Citation:** E. Sandmann, S. Lapuschkin, and W. Samek, "[The Algorithm Is Not the Behavior: Learned Priors Override Look-Ahead in a Chess-Playing Neural Network](https://arxiv.org/abs/2508.21380)," arXiv:2508.21380, 2025.  
**Link:** <https://arxiv.org/abs/2508.21380> · **Tier:** CORE

**Summary.** Sandmann, Lapuschkin and Samek probe whether internal algorithmic structure in Leela Chess Zero (T82, a 15-layer, 768-dim, 24-head policy transformer, the same family Jenner et al. studied) determines its output behavior. Extending the logit lens to the policy head (zero-ablating later blocks while preserving subsequent layer-norms, projecting all 64 squares simultaneously), they read each layer's move distribution and rate it by internal tournament Elo. Playing strength rises near-monotonically with depth (input ~351 to ~1,670 Elo for the full model, with a ~2,268 anchor baseline). Crucially, correct solutions—including immediate mates—often surface in intermediate layers but are "systematically overridden" by late layers that increasingly prefer safe play over aggression ("forgotten puzzles"). They confirm look-ahead is intact (future moves of the correct line are represented, causally important, and linearly decodable), so this is a preference conflict, not algorithmic failure. Gradient-based concept steering against the safety bias recovers 61.7% of ~7,901 forgotten puzzles, raising solve rate from 88.6% to 92.4%.

**Depends on.** Builds on the logit lens / iterative-inference framing for transformers, on Jenner et al.'s discovery of look-ahead in the same Leela Chess Zero network, and on linear-probing and activation/concept-steering interpretability methods; evaluated on Lichess puzzles.

**Results.** Internal Elo by layer (τ=0): input 351, L5 993, L10 1,055, L13 1,356, full model 1,670, anchor baseline ~2,268; strength monotonic in depth. Correct moves appear in intermediate layers then are discarded at output. Look-ahead representations are causally important and linearly decodable. Steering against safety preference recovered 61.7% of ~7,901 forgotten puzzles and lifted solve rate 88.6%->92.4% (evaluated against ~10,000 Lichess puzzles).

**Weaknesses.** Findings are specific to one architecture/checkpoint (Leela T82) and to puzzle positions, where tactical "correct" answers are unambiguous—generalization to ordinary play is untested. The logit-lens variant (zero-ablation with retained later norms) is an approximation whose readings may not equal what those layers would actually output; "internal Elo" is a derived construct, not the network's behavior. The "safety preference" interpretation is inferred from steering effects rather than a mechanistic account, and steering may recover puzzles via side effects. Recovery is partial (61.7%) and the small absolute solve-rate gain (3.8 points) tempers the causal claim. v3 reframing emphasizes priors-override-lookahead; precise probe/steering details require the full PDF, which exceeded fetch limits.

**Relation to xai-chess.** Its core demonstration that a chess net's internal computation diverges from its emitted behavior directly motivates ChessFaith's choice to grade explanations against a behavioral oracle's win-probability rather than internal activations.


## 6. Tracing the Thought of a Grandmaster-level Chess-Playing Transformer

**Citation:** R. Lin, Z. Jin, G. Zhou, et al., "[Tracing the Thought of a Grandmaster-level Chess-Playing Transformer](https://arxiv.org/abs/2604.10158)," arXiv:2604.10158, 2026.  
**Link:** <https://arxiv.org/abs/2604.10158> · **Tier:** CORE

**Summary.** This paper applies sparse mechanistic-interpretability decomposition to LC0 BT4, a search-free grandmaster-level Leela Chess Zero transformer (15 layers, 1024-dim, 32 heads; 99.6% mate-in-1, 96% mate-in-2). It is claimed as the first to decompose BOTH MLP and attention modules, using Top-K transcoders for MLPs and Lorsa (Low-Rank Sparse Attention) to split heads into interpretable rank-1 output-value features, forming sparse replacement layers that preserve the forward pass and permit feature-level causal analysis. Features are organized into seven semantic categories (piece detection, tactical threats, spatial/defensive relations, reachable squares, value, captures) and rule-validated against the chess rules with high precision/recall (often 86-100%). Feature steering (adding/scaling decoder directions, alpha=-1 for ablation) measures feature-to-output and feature-to-feature causal effects. Three new metrics (Path Overlap, Path Cohesion, Move-Square Contribution Ratio) reveal parallel, near-disjoint reasoning pathways for competing moves that progressively converge onto source/target squares in deep layers.

**Depends on.** Leela Chess Zero (LC0 BT4) policy network; sparse autoencoder / transcoder line of mechanistic interpretability; Lorsa low-rank sparse attention; ground-truth chess rules for feature validation; prior chess-transformer interpretability (DeepMind grandmaster transformer, Jenner/Sandmann-style probing of a different net).

**Results.** Rule-validated features reach 86-100% precision for detection/movement/spatial categories and 64-98% for tactical ones. Reasoning is parallel: top-2 move pathways share only 1.63-13.7% of features (7.05% typical); confident positions show 67.1% within-path cohesion vs 1.8% cross-path coupling; MCR rises toward 1.0 in deep layers. Causal case studies on a Qxh7+ position: zero-ablating d3->h7 attention cut Qxh7+ from 28.75% to 23.35%; copying opponent rook coverage g7->h7 dropped it to 9.32%; ablating an own-g2-pawn detection feature RAISED Qxh7+ from 28.8% to 90.6%, exposing a defensive over-evaluation that blinded the model to a forced mate. Model beat a 2100+ human 5-0.

**Weaknesses.** Causal claims rest on a handful of hand-picked case studies (largely one Qxh7+ position), not a systematic benchmark, so generalization and cherry-picking risk are unaddressed. "Rule-validated" precision/recall is reported per-feature without a clear aggregate or null/random baseline, and tactical features (64% precision) are noticeably weaker. Steering/ablation effect sizes are interpreted qualitatively without statistical testing or matched controls, and reconstruction-fidelity loss from the sparse replacement layers is not quantified here. Findings are specific to LC0 BT4's policy-head inductive bias and may not transfer to other chess nets or to LLMs. Studies the model's INTERNAL features, not natural-language explanation faithfulness.

**Relation to xai-chess.** It supplies the rule-grounded, causally validated internal-feature methodology (transcoders/Lorsa + feature ablation against chess rules) that ChessFaith parallels at the decision level, offering a complementary mechanistic anchor for what a load-bearing factor like a pin looks like inside a strong engine.


## 7. Amortized Planning with Large-Scale Transformers: A Case Study on Chess (v1: Grandmaster-Level Chess Without Search)

**Citation:** A. Ruoss, G. Delétang, S. Medapati, J. Grau-Moya, L. K. Wenliang, et al., "[Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/abs/2402.04494)," arXiv:2402.04494, 2024.  
**Link:** <https://arxiv.org/abs/2402.04494> · **Tier:** CORE

**Summary.** Ruoss et al. train decoder-only transformers (up to 270M parameters) to play chess by supervised distillation of Stockfish 16, without any explicit search at inference. They release ChessBench: 10M games with legal-move and value annotations (~15B data points). Three prediction targets are compared: state-value (SV), action-value (AV), and behavioral cloning (BC). The largest action-value model reaches a Lichess blitz Elo of ~2895 (grandmaster level) playing searchlessly, and shows non-trivial generalization to unseen boards and puzzles. The paper frames this as "amortized planning": a feedforward network internalizing the result of Stockfish's search. Values are discretized into bins and predicted via classification (HL-Gauss-style). The authors note perfect distillation of Stockfish remains out of reach, positioning ChessBench as an open benchmark. (v1 title was "Grandmaster-Level Chess Without Search"; v2 retitled to emphasize amortized planning.)

**Depends on.** Distills Stockfish 16 as the labeling oracle; builds on decoder-only transformer architectures and value-classification (binned/HL-Gauss) regression; conceptually related to AlphaZero-style value/policy learning but removes MCTS search.

**Results.** Largest AV model: ~2895 Lichess blitz Elo, searchless. Performance scales with model and dataset size. AV head outperforms SV and BC for play strength. Strong puzzle-solving and generalization to novel boards. ChessBench = 10M games / 15B annotated data points (Stockfish 16). Reported limitation: cannot perfectly distill Stockfish even at 270M params, leaving headroom.

**Weaknesses.** The model is a distillation of Stockfish 16, so it inherits Stockfish's evaluation biases and cannot exceed its teacher; "searchless" applies cleanly only to the AV/SV heads at inference, but the SV-as-policy setup requires a one-ply transition/expansion (effectively a depth-1 lookahead), so SV is not strictly searchless when used to select moves. Elo is from online blitz, not standardized over-the-board ratings, and is sensitive to opponent pool and time control. Values are coarsely binned, limiting fine-grained win-probability resolution. No explicit notion of why a move is good (no factor-level interpretability), and tactical depth beyond the distillation horizon is bounded.

**Relation to xai-chess.** Its action-value head yields a calibrated, search-free per-move win-probability that ChessFaith can use directly as the oracle wp(m) for measuring intervention-induced win-probability drops, avoiding centipawn-to-win conversion.


## 8. Chessformer: A Unified Architecture for Chess Modeling (Maia-3)

**Citation:** D. Monroe, G. Eilender, P. Chalmers, et al., "[Chessformer: A Unified Architecture for Chess Modeling](https://arxiv.org/abs/2605.19091)," arXiv:2605.19091, 2026.  
**Link:** <https://arxiv.org/abs/2605.19091> · **Tier:** CORE

**Summary.** Chessformer is an encoder-only transformer that unifies three chess-modeling tasks: human-move prediction, playing strength, and interpretability. Boards are tokenized as 64 square-tokens, with a novel Geometric Attention Bias (GAB) supplying chess-geometry-aware positional structure and an attention-based source-destination policy head. Two instantiations are presented: Maia-3, a human-emulation family (5M/23M/79M params), and Leela-CF, a strength instantiation integrated into Leela Chess Zero. Human strength is conditioned via a continuous "rating soft-embedding": two learnable embeddings (weak at 0 Elo, strong at 5000 Elo) linearly interpolated by rating, treating skill as a continuous axis rather than discrete checkpoints. Maia-3 (79M) reaches 57.1% move-matching, beating the 355M Allie (55.9%); Leela-CF adds >100 Elo to Lc0 and beats Stockfish in TCEC contexts. For interpretability, a cross-layer transcoder on MLP activations surfaces features localized to squares involved in tactical motifs such as forks and pins. Code/data open-sourced. (Prior notes' "github.com/CSSLab/maia3" not directly confirmed in the fetched text.)

**Depends on.** Builds on Leela Chess Zero (AlphaZero-style self-play), the Maia human-alignment lineage (Maia-1/Maia-2), Allie as a transformer baseline, and standard transformer/attention machinery; trains on Lichess blitz games.

**Results.** Maia-3 79M: 57.1%±0.1 move-matching (SOTA, vs Allie 355M @55.9% with search); 5M model 55.4% with ~70x fewer params. Gains concentrate at high skill (up to ~5% for very strong play); weak play saturates ~50% due to stochasticity. Leela-CF: +112±7 Elo at 160k playouts over Lc0; 191M value config 2466±36 Elo, 97.2% puzzle accuracy; policy-only 2374±37 Elo. GAB ablation: +1.9% policy, +0.3% value, +3.2% puzzle, +83 Elo vs absolute encoding at ~60% the params. Transcoder (layers 3-4 only) yields features matching forks, pins, weak squares, mate patterns.

**Weaknesses.** Interpretability is explicitly preliminary: transcoder trained on only layers 3-4 due to compute, and tactical-feature identification is validated by manual annotation of top-activating positions, i.e. correlational/observational rather than causally tested. No intervention or ablation establishes that the "pin"/"fork" features are load-bearing for the model's move choice. GAB is hand-specialized to chess geometry and may not transfer. Human-prediction ceiling (~50% at low skill) limits headroom claims. The fetched abstract was truncated; some specifics (GitHub repo URL, exact dataset sizes) are reconstructed from the HTML body and not independently re-verified.

**Relation to xai-chess.** Chessformer supplies both a strong human-aligned oracle/policy (Maia-3, with a continuous rating axis useful for win-probability scoring) and a square-token interpretability baseline whose pin/fork "features" are exactly the correlational explanations that ChessFaith's causal-intervention benchmark would put to a load-bearing test.


## 9. Maia-2: A Unified Model for Human-AI Alignment in Chess

**Citation:** Z. Tang, D. Jiao, R. McIlroy-Young, et al., "[Maia-2: A Unified Model for Human-AI Alignment in Chess](https://arxiv.org/abs/2409.20553)," in Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2409.20553.  
**Link:** <https://arxiv.org/abs/2409.20553> · **Tier:** CORE

**Summary.** Maia-2 (NeurIPS 2024) is a single unified neural model of human chess decision-making that conditions move prediction on player rating via a skill-aware attention mechanism (channel-wise patching plus skill-level embeddings injected into attention queries), replacing the earlier Maia approach of training separate models per rating bin. It is trained on 169M Lichess games (9.1B positions, 2013-2023) and predicts the human move a player of a given strength would make, with policy and value/auxiliary heads. It beats Maia-1 in next-move accuracy (roughly +2 points; perplexity 4.07 vs 4.67 bits), reaching ~51.7-54.2% per skill tier (Skilled <=1599, Advanced 1600-1999, Master >=2000), and crucially produces far more coherent, monotonic skill scaling (27% of positions monotonic vs Maia-1's 1%), with smooth monotonic declines in centipawn loss and blunder rate as rating rises. Confirmed: the architecture has only policy/value/auxiliary heads and NO natural-language channel, so it cannot read or generate explanations.

**Depends on.** Builds directly on Maia/Maia-1 (McIlroy-Young et al.) human move-prediction models; uses Lichess game data and Stockfish evaluations for grounded validation; transformer attention architecture.

**Results.** +~2pp move-prediction accuracy over Maia-1; perplexity reduced 4.67->4.07 bits; per-tier accuracy ~51.7% (Skilled), 54.2% (Advanced), 53.9% (Master); 27% of test positions exhibit monotonic skill-consistency vs 1% for Maia-1; monotonic decreases in centipawn loss and blunder rate with rating. Trained on 169M games/9.1B positions; tested on 106,740 positions; 450K positions with Stockfish evals for grounded validation.

**Weaknesses.** No natural-language or causal-explanation channel, so it cannot consume or grade move-explanations directly. It predicts what humans of a rating would play, not whether a factor is load-bearing for move quality; its "skill" signal is correlational human-imitation, not a controlled intervention. Reported per-tier accuracies (~52-54%) leave large unexplained variance. The fetched abstract/HTML omit some numbers; per-rating success-probability figures cited in project notes were inferred rather than tabulated in the retrieved text, so use with care. Difficulty calibration is implicit (via human-error-rate-by-rating), not an explicit documented mechanism.

**Relation to xai-chess.** Maia-2 can serve ChessFaith only as an auxiliary human-difficulty/skill calibrator (per-rating move-match probabilities), not as the causal oracle or explanation reader, since it has no natural-language channel and measures human imitation rather than interventional move quality.


## 10. Human-Aligned Chess With a Bit of Search (Allie)

**Citation:** Y. Zhang, A. P. Jacob, V. Lai, D. Fried, and D. Ippolito, "[Human-Aligned Chess With a Bit of Search](https://arxiv.org/abs/2410.03893)," arXiv:2410.03893, 2024.  
**Link:** <https://arxiv.org/abs/2410.03893> · **Tier:** CORE

**Summary.** Allie is a chess engine designed to play and behave like humans across the full skill spectrum. A Transformer is trained on log sequences of real online games to model human players' move choices plus non-move behaviors such as per-move thinking time and resignations, and to assign reward/value estimates to positions. At inference it adds a time-adaptive Monte-Carlo tree search (MCTS) whose search budget scales with how long humans would think in a given position, so the engine "ponders" at critical moments rather than searching uniformly. The authors report state-of-the-art human move-prediction accuracy, and in large-scale online play against humans rated 1000-2600 Elo the adaptive-search variant achieves an average skill gap of only 49 Elo, reproducing target ratings well; against ~2500-rated opponents it plays at grandmaster strength. Confirmed: title/authors match (Zhang, Jacob, Lai, Fried, Ippolito); the prior note's "~49 Elo gap" is accurate.

**Depends on.** Builds on human-imitation chess engines (Maia / McIlroy-Young et al.), Transformer sequence models of game logs, and AlphaZero-style MCTS, adapting search budget to human thinking-time data from large online game corpora (e.g., Lichess).

**Results.** SOTA human move-prediction accuracy over prior benchmarks; average skill gap of only 49 Elo across opponents rated 1000-2600 in large-scale online evaluation; grandmaster-level strength versus ~2500 Elo players; emergent human-like time allocation (ponders longer at critical positions). Quantitative ablation details (e.g., exact accuracy figures, calibration per rating band) are in the full paper, not the abstract page.

**Weaknesses.** The fetched abstract page does not enumerate limitations, so caveats are inferred. Human-alignment is measured behaviorally (move-match accuracy, Elo calibration, timing) without any notion of explanation or reasoning faithfulness — it models WHAT humans play, not WHY, so it offers no decision-level causal ground truth. Training on aggregate online logs may bake in population-level biases and cannot disentangle a single player's style; thinking-time modeling depends on noisy clock data. "49 Elo gap on average" can mask larger errors at rating extremes. No code/data verification was possible from the abstract alone.

**Relation to xai-chess.** Allie is a candidate human-like oracle/opponent and a source of human-calibrated move/value estimates, but unlike ChessFaith it grades behavioral human-likeness rather than the causal faithfulness of explanations, underscoring the gap the project's intervention benchmark fills.


## 11. Concept-guided Chess Commentary Generation and Evaluation (CCC / GCC-Eval)

**Citation:** J. Kim, J. Goh, I. Hwang, J. Cho, and J. Ok, "[Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation and Evaluation](https://arxiv.org/abs/2410.20811)," in Proc. NAACL, 2025. arXiv:2410.20811.  
**Link:** <https://arxiv.org/abs/2410.20811> · **Tier:** CORE

**Summary.** Kim et al. (NAACL 2025) propose Concept-guided Chess Commentary (CCC), which bridges chess-engine decision accuracy with LLM fluency by feeding an LLM a prioritized list of human-interpretable concepts before it writes free-text prose commentary on a position/move. Concepts are derived from expert engines (Stockfish 8 and Leela Chess Zero T78): linear/SVM probes are trained on internal engine representations to detect concepts, and concepts are then prioritized by their importance/salience to the current decision so the LLM emphasizes the most decision-relevant ones. They also introduce GCC-Eval, a GPT-based evaluation of commentary along informativeness and linguistic quality, validated against human judges. Training/eval uses the GameKnot ChessCommentary dataset plus Lichess evaluation data. Reported results show CCC produces more accurate, informative, and fluent commentary than prior commentary baselines under both human and GCC-Eval assessment. The contribution is concept-conditioned generation and an LLM-judge evaluation protocol, not a causal-intervention faithfulness test.

**Depends on.** Builds on chess-engine internals (Stockfish 8, Leela Chess Zero T78), concept-probing / probing-classifier methodology (linear/SVM probes of engine representations), the GameKnot ChessCommentary dataset and prior neural chess-commentary generators, and GPT-based LLM-as-judge evaluation.

**Results.** CCC reportedly outperforms prior chess-commentary generation baselines on accuracy, informativeness, and fluency, as judged by both human evaluators and the proposed GCC-Eval. GCC-Eval scores are validated to correlate with human judgments. Exact numeric tables (per-baseline scores, dataset sizes, win/preference rates) were not extractable from the fetched abstract/PDF/repo; the quantitative magnitudes should be confirmed against the paper's results tables before citing specific numbers.

**Weaknesses.** Evaluation is correlational/text-quality oriented, not causal: GCC-Eval is itself a GPT-based judge (LLM grading LLM), inheriting bias and circularity, and "accuracy" is judged textually rather than verified against ground-truth engine outcomes. Concepts come from coarse positional eval terms learnable by probes; tactical motifs like pin/fork/open-file are not first-class because they lack reliable concept labels, so the very factors this project intervenes on are largely absent. Concept "prioritization" is a salience/probe-delta ranking that shapes generation but is never tested for whether the cited concept is actually load-bearing for the move's strength. Output is free-text prose with no structured factors[] field, complicating automated faithfulness scoring. Specific quantitative results could not be verified from the fetched sources.

**Relation to xai-chess.** It is the closest concept-conditioned chess-commentary system, supplying a reusable probe/delta concept-prioritization technique, but it stops at LLM-judge text quality whereas ChessFaith adds the verifiable decision-level causal-intervention ground truth (including tactical motifs like pins) that this paper lacks.


## 12. Grounded Chess Reasoning in Language Models via Master Distillation (C1)

**Citation:** Z. Tang, Q. Wen, S. Grief-Albert, et al., "[Grounded Chess Reasoning in Language Models via Master Distillation](https://arxiv.org/abs/2603.20510)," arXiv:2603.20510, 2026.  
**Link:** <https://arxiv.org/abs/2603.20510> · **Tier:** CORE

**Summary.** Tang et al. introduce C1, a 4B-parameter language model that produces step-by-step, natural-language chess explanations by distilling expert reasoning. The pipeline uses Gemini-3-Flash as a distillation teacher and Stockfish (depth 24) as a "master system" to verify solutions, then trains Qwen3-4B-Instruct via supervised fine-tuning (~39,600 theme-balanced samples drawn from the Lichess puzzle database) followed by a DAPO-style RL stage ("DAPO-C1") with binary correctness rewards. C1 reaches 48.1% Pass@1 on a 900-puzzle test set (500 theme-split, 400 difficulty-split), outperforming all open-source models, most frontier proprietary systems, and even its own teacher, while using roughly two orders of magnitude fewer tokens. The explanations reference tactical themes (forks, pins, mating attacks) and contextual squares rather than emitting bare move predictions. Crucially, the authors concede their chain-of-thought "may not faithfully capture the underlying logic for harder puzzles," explicitly flagging the faithfulness gap.

**Depends on.** Qwen3-4B-Instruct as the base model; Gemini-3-Flash (via OpenRouter) as distillation teacher; Stockfish (depth 24) as the verifying master/oracle; Lichess puzzle database; SFT plus DAPO-style RLVR (reinforcement learning with verifiable rewards).

**Results.** Advances chess-explanation accuracy from a near-zero baseline to 48.1% Pass@1 on a 900-puzzle benchmark (500 theme-split, 400 difficulty-split). Outperforms all open-source models and most frontier proprietary systems, surpasses its own distillation teacher, and generates solutions with ~100x fewer tokens than baselines. Reports degraded SFT+RLVR performance on expert-level puzzles, attributed to unfaithful CoT traces limiting what RLVR can learn.

**Weaknesses.** Faithfulness is asserted only by self-admission, not measured: the paper offers no causal or counterfactual test that the cited motif (e.g., a pin) is actually load-bearing for the move, which is exactly the gap. Accuracy (48.1% Pass@1) is moderate and verification reduces to move correctness, not explanation correctness — a right move with a wrong rationale still passes. Stockfish verifies solutions but does not validate the stated reasoning. Heavy reliance on a proprietary teacher (Gemini-3-Flash) and a single base model (Qwen3-4B) limits reproducibility and generality. Evaluation is puzzle-centric (tactical motifs over positional/strategic play), and the test set is modest (900 items).

**Relation to xai-chess.** C1 is an ideal Phase-1 explanation source for xai-chess: it natively emits motif-with-square citations yet self-admits its traces may be unfaithful, providing exactly the decision-level explanations the causal-intervention benchmark is designed to grade.


## 13. ChessQA: Evaluating Large Language Models for Chess Understanding

**Citation:** Q. Wen, Z. Tang, and A. Anderson, "[ChessQA: Evaluating Large Language Models for Chess Understanding](https://arxiv.org/abs/2510.23948)," arXiv:2510.23948, 2025.  
**Link:** <https://arxiv.org/abs/2510.23948> · **Tier:** CORE

**Summary.** ChessQA is a benchmark for measuring large language models' chess understanding via a controlled, consistent set of question-answering tasks, moving past prior ad hoc evaluations that focused narrowly on move-quality. It organizes probes into five tiers of ascending complexity: Structural (basic rules/board state), Motifs (tactical patterns), Short Tactics (calculating short forcing sequences), Position Judgment (evaluating positions), and Semantic (describing high-level concepts in natural language). The benchmark is explicitly refreshable: it ships prompts, answer keys, and construction scripts that can evolve as models improve, plus a planned public leaderboard and periodically updated datasets to resist contamination. Evaluating a range of contemporary LLMs, the authors report persistent weaknesses across all five categories, with per-category results and error analyses. The paper is 33 pages with 8 figures. Note: I could only access the abstract/metadata page; exact model names, per-tier item counts, and quantitative scores were not retrievable and are not reported here.

**Depends on.** Builds on prior chess-LLM evaluation work focused on move quality; uses chess engines (e.g., Stockfish-style evaluation) and chess knowledge taxonomies to construct graded tasks; conceptually related to refreshable/contamination-resistant benchmark designs and leaderboards.

**Results.** Across the five tiers, contemporary LLMs show persistent weaknesses in every category, indicating substantial gaps in machine chess understanding. The authors provide per-category results and error analyses. Specific accuracy figures, model rankings, and dataset sizes were not available from the abstract-level source accessed; the full PDF would be needed to report exact numbers.

**Weaknesses.** Quantitative specifics (models, item counts, scores) were not accessible from the abstract page, limiting verification. As a QA benchmark, it measures whether an LLM can produce correct/labeled answers, not whether a model's stated reasoning is causally faithful to its decision — it has no intervention or counterfactual mechanism to test if a cited factor is load-bearing. Position Judgment and Semantic tiers risk subjective or engine-anchored ground truth that may not capture human-style understanding. Refreshability mitigates but does not eliminate contamination, and "understanding" is operationalized narrowly through QA accuracy.

**Relation to xai-chess.** ChessQA supplies a structured taxonomy of chess understanding (especially Motifs, Tactics, and Semantic tiers) and a refreshable evaluation harness that ChessFaith can draw task items and concept vocabulary from, but it grades answer correctness rather than the causal faithfulness of explanations that ChessFaith targets via intervention.


## 14. Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth (BonaFide)

**Citation:** Y. Gur-Arieh, A. Marasović, and M. Geva, "[Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth](https://arxiv.org/abs/2605.25052)," arXiv:2605.25052, 2026.  
**Link:** <https://arxiv.org/abs/2605.25052> · **Tier:** CORE

**Summary.** BonaFide is a meta-evaluation that asks whether chain-of-thought (CoT) faithfulness metrics actually detect (un)faithfulness, using verifiable ground truth rather than proxies like plausibility or importance. The authors build 3,066 labeled CoTs spanning 13 tasks and 10 models via two label-generating designs: an "outright" setting where correct answers necessarily imply specific bottleneck computations were executed (e.g. iterating the Collatz function, traversing a graph by a policy), and a "diversionary" setting pairing questions with misleading hints so that steps acknowledging vs. concealing hint use are labeled faithful/unfaithful. They score eight metrics across four families: importance-based (Adding Mistakes, Early Answering, Filler Tokens, SCM), parameter-based (FUR), attribution-based (CC-SHAP), and semantic-utility (Paraphrasing, Simulatability). Result: most metrics perform near chance, show strong prediction biases, and degrade on longer CoTs. Best CoT-level AUROC is 0.70 (CC-SHAP); best step-level is 0.59 (Filler Tokens). Crucially, no metric transfers across settings (the best CoT metric falls below chance at step level), and the strongest ones are computationally expensive.

**Depends on.** Builds on the CoT-faithfulness metric literature it audits (Early Answering, Filler Tokens, Adding Mistakes, Paraphrasing, CC-SHAP, SCM, Simulatability, FUR) and on prior ground-truth attempts including edit-based Causal Diagnosticity, whose construction it critiques for not verifying that models actually used the injected/edited knowledge.

**Results.** 3,066 labeled CoTs, 13 tasks, 10 models. CoT-level label imbalance ~15% faithful / 85% unfaithful; step-level roughly balanced (51%/49%). Best CoT-level AUROC = 0.70 (CC-SHAP); best step-level AUROC = 0.59 (Filler Tokens). No cross-setting transfer: CC-SHAP drops below chance at step level, Filler Tokens stays near-chance for whole CoTs. Most metrics near chance overall, with prediction biases and degradation on longer CoTs. Top metrics also carry prohibitive compute cost. Code at github.com/yoavgur/BonaFide.

**Weaknesses.** Ground truth is restricted to synthetic/structured tasks (Collatz iteration, graph traversal) and hint-diversion setups where computations are externally inferable; this may not generalize to open-ended reasoning where "which computation produced the answer" is itself ill-defined. Severe CoT-level class imbalance (~15/85) makes AUROC the headline metric and complicates threshold/calibration claims; the authors themselves flag this as the chief limitation. The "outright" labels assume a correct answer entails executing the identified bottleneck step, which can be confounded by shortcuts, guessing, or alternative solution paths. Diversionary labeling hinges on interpreting whether a step "acknowledges" hint use, a partly semantic judgment. Conclusions are bounded to the eight chosen metric implementations and ten models; near-chance results could partly reflect hyperparameter/implementation choices rather than the metrics' ceiling.

**Relation to xai-chess.** It is the direct methodological template and motivation for xai-chess: a ground-truth meta-evaluation showing existing faithfulness metrics fail, which ChessFaith answers by supplying verifiable decision-level causal ground truth via engine win-probability interventions in chess.


## 15. A Causal Lens for Evaluating Faithfulness Metrics (Causal Diagnosticity)

**Citation:** K. Zaman and S. Srivastava, "[A Causal Lens for Evaluating Faithfulness Metrics](https://arxiv.org/abs/2502.18848)," in Proc. EMNLP, 2025. arXiv:2502.18848.  
**Link:** <https://arxiv.org/abs/2502.18848> · **Tier:** CORE

**Summary.** Zaman and Srivastava (EMNLP 2025) introduce "Causal Diagnosticity," a framework that evaluates faithfulness METRICS (not explanations) by testing whether they can tell faithful from unfaithful explanations. They use model/knowledge editing to construct ground-truth pairs: two edited model variants give the same answer but for different reasons, so each model's self-explanation is faithful to itself yet unfaithful to the other. Editing is done primarily via In-Context Editing (ICE), with MEMIT as a parameter-editing ablation. A metric's diagnosticity is the fraction of pairs where it scores the faithful explanation higher (0.5 = chance). They benchmark six metrics (Simulatability; CoT corruption tests: Early Answering, Filler Tokens, Adding Mistakes, Paraphrasing; CC-SHAP post-hoc and CoT) across four tasks (FactCheck, Analogy, Object Counting, Multi-hop) on Qwen-7B and Gemma-9B. Filler Tokens ranks best overall (Copeland 29); continuous metrics beat binary ones (up to ~66%) but are sensitive to noise, replacement-token choice, and model.

**Depends on.** Builds on knowledge/model-editing methods (In-Context Editing/ICE, MEMIT) and on the LLM faithfulness-metric literature it evaluates: Simulatability, CoT corruption tests (Early Answering, Filler Tokens, Adding Mistakes, Paraphrasing; Lanham et al.), and CC-SHAP. Conceptually related to counterfactual/causal-intervention evaluation of explanations.

**Results.** Per-task diagnosticity (1.0 ideal, 0.5 chance): FactCheck Filler Tokens 0.828 (Qwen)/0.893 (Gemma); Analogy CC-SHAP post-hoc 0.898 (Gemma) but edit reliability <50%; Object Counting Filler Tokens 0.843 (Gemma); Multi-hop Filler Tokens 0.682 (Qwen)/0.585 (Gemma), the hardest task. Overall Copeland ranking: Filler Tokens 29, Early Answering 18, CC-SHAP CoT 12. Continuous metrics outperform binary variants (up to ~66%). Edit success (Fig. 4): FactCheck ~95%, Multi-hop ~70%, Object Counting ~60%, Analogy <50% ("unreliable"). Causal Diagnosticity cannot score metrics needing explanation regeneration (e.g., Counterfactual Edits).

**Weaknesses.** Ground truth is itself unverified and proxy-based: edit success is gauged by a perplexity ratio (faithful expected lower than unfaithful), an indirect assumption that can fail. Edit reliability is poor on some tasks (Analogy <50%, declared unreliable; Object Counting ~60%). Critically (Sec. 5.5), model-generated explanations often FAIL to reflect the applied edits for Analogy and Object Counting, forcing the authors to fall back on SYNTHETIC explanations for main results — sacrificing ecological validity to keep labels valid. Edit generalization/compositionality is assumed, not independently benchmarked. Only two mid-size open models (Qwen-7B, Gemma-9B); no strong oracle. The framework structurally excludes counterfactual/regeneration-based metrics. Continuous metrics' diagnosticity is sensitive to noise, replacement-token type, and truncation strategy, undermining cross-model robustness.

**Relation to xai-chess.** It is the closest methodological precedent — a causal, intervention-based meta-evaluation of faithfulness metrics — but its ground truth is a noisy, perplexity-proxied editing construct, exactly the verifiable causal ground truth ChessFaith supplies by intervening on game-theoretic factors and measuring an oracle engine's win-probability shift.


## 16. RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention

**Citation:** Y. Han, Y. Lee, and J. Do, "[RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](https://arxiv.org/abs/2602.17053)," arXiv:2602.17053, 2026.  
**Link:** <https://arxiv.org/abs/2602.17053> · **Tier:** CORE

**Summary.** RFEval is a benchmark testing whether large reasoning models (LRMs) produce faithful explanations of their own decisions. It operationalizes faithfulness via two conditions: (1) stance consistency, the model's chain-of-thought aligns with its final answer, and (2) causal influence, the stated reasoning genuinely drives the output under controlled, output-level counterfactual reasoning interventions, explicitly decoupled from accuracy. The benchmark covers 7,186 instances across seven task domains and evaluates twelve open-source LRMs. The authors find unfaithfulness in 49.7% of outputs, predominantly from stance inconsistency, concentrated in "brittle, convergent" domains like math and code, and correlating more with training recipe than model scale. A key result: once controlling for model and task, the accuracy-faithfulness link is weak and statistically insignificant, so accuracy is "neither a sufficient nor a reliable proxy for faithfulness." They also report that RL-style fine-tuning objectives can reduce faithfulness while preserving accuracy. The fetched title/authors match the provided record (resolved=true).

**Depends on.** Builds on the LLM/LRM reasoning-faithfulness literature (chain-of-thought faithfulness, counterfactual/intervention-based tests of whether stated reasoning is causally load-bearing); evaluates open-source large reasoning models; relates to prior faithfulness probes (e.g., Turpin-style biasing, causal-mediation/intervention frameworks) and accuracy-vs-faithfulness decoupling.

**Results.** 7,186 instances, 7 task domains, 12 open-source LRMs. 49.7% of outputs judged unfaithful, dominated by stance inconsistency. Unfaithfulness concentrates in brittle/convergent domains (math, code) and tracks training method more than scale. After controlling for model and task, the accuracy-faithfulness correlation is weak and statistically insignificant. RL-style fine-tuning objectives can degrade faithfulness while keeping accuracy intact.

**Weaknesses.** Interventions are at the OUTPUT level (manipulating stated reasoning text and checking answer change), not grounded in verifiable external causal ground truth, so "causal influence" is inferred from model behavior rather than a known-correct factor. The faithfulness verdict appears to be a single deterministic indicator without a statistical-test/randomization layer or matched-baseline calibration, leaving no principled null distribution for "no effect." Restricted to open-source LRMs and seven (likely short-answer/convergent) tasks limits generalization; the 49.7% figure is sensitive to thresholds and the stance-consistency operationalization, which may conflate self-contradiction with genuine unfaithfulness. Detailed task list, intervention construction, and judge reliability were not fully visible from the abstract-level fetch.

**Relation to xai-chess.** RFEval supplies the matched intervention-based faithfulness recipe (reasoning must causally drive the answer) that ChessFaith adapts to chess explanations, while ChessFaith upgrades it with verifiable decision-level causal ground truth and a statistical/randomization-test layer that RFEval's single deterministic indicator lacks.


## 17. ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs

**Citation:** A. Basu and P. Chakraborty, "[ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs](https://arxiv.org/abs/2603.18579)," arXiv:2603.18579, 2026.  
**Link:** <https://arxiv.org/abs/2603.18579> · **Tier:** CORE

**Summary.** ICE evaluates the faithfulness of LLM token-attribution explanations by intervening on input text and measuring how the model's prediction score retains the signal carried by rationale tokens. Its core metric, Normalized Score Retention (NSR = [s(x_o^r) - s(empty)]/[s(x) - s(empty)]), is compared against matched random-rationale baselines via a randomization test. The framework reports a primary win rate (proportion of random baselines beaten), Cohen's d effect size against the random null, bootstrap 95% CIs (B=200), conservative finite-sample one-sided p-values, and Benjamini-Hochberg FDR control (alpha=0.10), with M=50 permutations for LLMs (100 for encoders). It uses two intervention operators: Deletion (remove non-rationale tokens) and Retrieval-Infill (replace them with label-blacklisted spans from other examples). Evaluated over 7 LLMs, 4 English tasks, 6 non-English languages, and 2 attribution methods, it finds faithfulness is strongly operator-dependent (gaps up to 44pp), nearly a third of configurations are anti-faithful, and faithfulness is uncorrelated with human plausibility (|r|<0.04).

**Depends on.** Builds on the LLM/NLP faithfulness-evaluation tradition (perturbation/deletion-based attribution faithfulness, comprehensiveness/sufficiency metrics, ERASER), matched-random-baseline and randomization/permutation testing, bootstrap CIs, and Benjamini-Hochberg FDR; tests post-hoc attribution methods (attention, and gradient/IG-style attribution) over benchmarks like e-SNLI and IMDB.

**Results.** Operator-dependence is large: e.g., Llama-3.2 on e-SNLI shows 86.4% win rate under Deletion vs 42.6% under Retrieval-Infill with attention attribution (a 44pp gap). Nearly one-third of model/task/method configurations are anti-faithful (win rate <50% vs random baselines). Deletion inflates faithfulness on short text but the ranking reverses on long text (IMDB), so the authors argue faithfulness must be read comparatively across operators, not as a single score. Faithfulness shows essentially zero correlation with human plausibility (|r|<0.04), confirming the two are distinct axes.

**Weaknesses.** No causal ground truth: it measures behavioral score sensitivity to perturbations, so a high win rate confirms the attribution tracks the model but cannot certify which factor is genuinely load-bearing in the true decision; "faithfulness" remains operator-relative with no oracle. Retrieval-Infill may leak task-relevant signal from the replacement distribution (authors flag this bias), and Deletion creates off-distribution truncated inputs, so both operators confound faithfulness with distribution shift. Small M (50) limits p-value resolution and FDR power; alpha=0.10 is lenient. IG omitted for 7B+ models due to memory, limiting attribution coverage. Scope is token-level NLP classification; no decision-level / structured-reasoning settings. 50x compute multiplier.

**Relation to xai-chess.** ICE is the closest methodological template for ChessFaith's statistics stack (matched-baseline randomization test, multi-operator win rates with CIs and FDR), which ChessFaith adapts to chess while supplying the verifiable decision-level causal ground truth that ICE explicitly lacks.


## 18. Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations

**Citation:** Y. Chen, R. Zhong, N. Ri, C. Zhao, H. He, J. Steinhardt, et al., "[Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations](https://arxiv.org/abs/2307.08678)," arXiv:2307.08678, 2023.  
**Link:** <https://arxiv.org/abs/2307.08678> · **Tier:** CORE

**Summary.** Chen et al. ask whether LLM natural-language explanations actually describe the model's decision process, operationalizing this as "counterfactual simulatability": a good explanation should let a human predict the model's output on related, modified inputs. They prompt GPT-3.5/GPT-4 to generate 6-10 diverse counterfactuals per explained instance, query the same model for ground-truth answers, and have MTurk annotators judge (as an entailment task) whether the explanation logically implies those answers. Two metrics: simulation precision (fraction of simulatable counterfactuals where the human inference matches the model's real output) and generality (1 - average pairwise similarity of simulatable counterfactuals, i.e., breadth of coverage). On StrategyQA (multi-hop QA) and Stanford Human Preferences (reward modeling), both chain-of-thought and post-hoc explanations show only moderate precision (~77% for GPT-3.5, ~81-93% for GPT-4) and, critically, precision is essentially uncorrelated with human-rated plausibility (Pearson ~0.012), implying RLHF-style optimization of human approval does not yield faithful explanations.

**Depends on.** Builds on the simulatability tradition of explanation evaluation (Hase & Bansal; Doshi-Velez & Kim) and counterfactual/perturbation faithfulness tests; uses GPT-3.5/GPT-4, StrategyQA, and Stanford Human Preferences (SHP); relies on LLM-generated counterfactuals and human entailment annotation.

**Results.** GPT-3.5 precision on StrategyQA: 77.3% (CoT) / 76.8% (post-hoc); GPT-4: 81.1% (CoT) / 83.9% (post-hoc), and 93.0% / 91.5% on SHP. Precision vs. plausibility correlation ~+0.012 (near zero). Explanations are far from perfectly precise, and plausibility is not a proxy for faithfulness. Authors conclude naively optimizing human approval (RLHF) is insufficient for faithful self-explanation.

**Weaknesses.** Ground truth for counterfactuals is the same LLM's own (noisy, possibly inconsistent) output rather than a verifiable external truth, so "model behavior" is itself unstable. Counterfactuals are LLM-generated, so coverage is biased toward what GPT-3.5/GPT-4 can imagine and may miss decisive cases. "Precision" depends on human entailment judgments with only moderate-to-fair inter-annotator agreement, injecting subjectivity. No demonstration that higher simulatability actually helps a downstream task (debugging, trust calibration). English-only, two datasets, two model families; metric conflates explanation quality with the model's own behavioral consistency.

**Relation to xai-chess.** It is the canonical formalization of explanation faithfulness-via-counterfactuals in NLP that ChessFaith adapts, but ChessFaith replaces its self-referential LLM ground truth with verifiable decision-level causal ground truth from a movegen-patched oracle engine.


## 19. Piece-by-Piece Chess Explanations with SHAP

**Citation:** F. Spinnato, "[Towards Piece-by-Piece Explanations for Chess Positions with SHAP](https://arxiv.org/abs/2510.25775)," arXiv:2510.25775, 2025.  
**Link:** <https://arxiv.org/abs/2510.25775> · **Tier:** CORE

**Summary.** Spinnato (a single-author paper, despite the project's "Spinnato et al." note) adapts SHAP to chess by treating board pieces as features and the engine evaluation as the value function. Centipawn scores from Stockfish 17.1 (with Leela Zero used for comparison) are mapped to a White-win probability via a logistic transform, p(s)=1/(1+exp(-beta*s)). Shapley values are computed over coalitions of non-king pieces by ablating subsets and measuring evaluation deltas; both kings are fixed, with a king-only base value of 0.5. Illegal ablated positions are handled by flipping orientation or defaulting to 0.5. The result is additive, locally faithful, per-piece attributions intended for pedagogy, position assessment, and engine comparison. Evaluation is entirely qualitative: seven illustrative thematic examples (self-blocking pawns, bishop-vs-knight endgames, trapped rooks, pins, engine comparison). No quantitative benchmark, human study, or faithfulness metric is reported. Code and data are released.

**Depends on.** SHAP / Shapley-value attribution (Lundberg and Lee 2017; Shapley game theory); chess engines Stockfish 17.1 and Leela Zero v0.31.2; centipawn-to-win-probability logistic mapping; classical chess pedagogy of mentally removing pieces.

**Results.** Qualitative only, no numbers beyond mechanics. Seven case studies demonstrate intuitive attributions (e.g., negative contributions for self-blocking pawns, mobility effects for trapped rooks, pin tactics, bishop-vs-knight endgame value, Stockfish-vs-Leela comparison). Compute cost: ~55 minutes for a position with ~14 non-king pieces. Base value fixed at 0.5 for king-only positions. The author explicitly cautions that a piece's SHAP attribution does NOT equal the evaluation change from removing that single piece (it is an average marginal over all subsets), and that explanations should not guide move selection.

**Weaknesses.** No quantitative or faithfulness evaluation whatsoever; purely cherry-picked illustrative examples, so claims of "locally faithful" attribution are untested. Combinatorial blow-up (~55 min at 14 pieces) makes exhaustive Shapley infeasible at scale and likely requires sampling. Ablating pieces produces illegal/unreachable positions handled by ad hoc fixes (orientation flip, default 0.5) that inject confounds into the value function. Kings cannot be attributed at all, leaving a structural blind spot. Crucially, attribution is over static board occupancy (which piece sits where), not over move-explanation factors; it cannot isolate WHY a specific move is good or test whether a named factor like a pin is load-bearing. Single-author preprint, not peer-reviewed; "et al." in the source brief is incorrect.

**Relation to xai-chess.** This is the nearest prior art to xai-chess's board-level operator (remove/relocate a piece and read the engine eval-delta), but it attributes static position value via SHAP rather than grading move-explanation factors, and it lacks both a rule-level/confound-free intervention surface and any causal-faithfulness metric or randomization test, which are precisely the gaps ChessFaith fills.


## 20. Aligning Superhuman AI with Human Behavior: Chess as a Model System (Maia-1)

**Citation:** R. McIlroy-Young, S. Sen, J. Kleinberg, and A. Anderson, "[Aligning Superhuman AI with Human Behavior: Chess as a Model System](https://arxiv.org/abs/2006.01855)," in Proc. 26th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining (KDD), 2020. arXiv:2006.01855.  
**Link:** <https://arxiv.org/abs/2006.01855> · **Tier:** CORE

**Summary.** Maia is an AlphaZero-style policy network retrained to predict the moves humans actually play, rather than to play optimally. Trained on Lichess games (12M games per rating bin), the authors fit nine separate models, each on a 100-point Elo band from 1100 to 1900, using a 6-block residual CNN (64 channels, 12-ply history) with no tree search at inference. Each Maia matches human moves best at its target rating (a "strikingly unimodal" curve), reaching >52% move-matching accuracy, versus 33-41% for Stockfish and a 46% max for Leela. A separate residual CNN predicts blunders: 67.7% (board only), 71.7% (board+metadata) for individual blunders, and 76.9% collectively. The core thesis is that aligning AI with granular human behavior requires modeling human actions/errors directly, not scaling strength. A key limitation: the 1900-trained model degrades on stronger players, lacking high-skill training data.

**Depends on.** Builds on AlphaZero/Leela Chess Zero (self-play deep RL chess architecture and the Leela codebase), Stockfish as a strength baseline, and large-scale Lichess human game logs; situated in the human-AI alignment / behavioral modeling literature.

**Results.** Move-matching accuracy: Maia >52% peak vs Stockfish 33-41% and Leela max 46%; even the worst Maia (1900-model on 1100 moves) hit ~46%, matching the best baselines. Per-bin models peak near their own training Elo (unimodal). Blunder prediction: 67.7% (board), 71.7% (board+metadata) individual; 76.9% collective. Data: 12M games/bin train, ~500k positions/bin test; 182M blunders / 272M non-blunders for blunder model; Bullet/HyperBullet and <30s-remaining moves excluded.

**Weaknesses.** Maia predicts WHICH move a human plays but offers no explanation of WHY, and gives no decision-level causal account of which board factors drive a choice. Coverage is capped at 1100-1900 Elo (Lichess rapid/blitz); it degrades above 1900 and is untested on classical/OTB or superhuman play. Move-matching accuracy (~50%) conflates many plausible human moves and is not a faithfulness or correctness measure. The per-rating-bin design is purely correlational/behavioral; it never intervenes on the board to test causes. Blunder prediction tops out below 77% and depends on platform metadata (time pressure), limiting generality.

**Relation to xai-chess.** Maia supplies a behaviorally-grounded model of human-like chess play that ChessFaith can use as a human-aligned reference for what factors a player would notice, but it lacks the causal-intervention machinery ChessFaith adds to verify whether a cited factor is actually load-bearing.


## 21. Chess Commentaries with Language Models and Symbolic Engines

**Citation:** A. Lee, D. Wu, E. Dinan, and M. Lewis, "[Improving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines](https://arxiv.org/abs/2212.08195)," arXiv:2212.08195, 2022.  
**Link:** <https://arxiv.org/abs/2212.08195> · **Tier:** CORE

**Summary.** Lee et al. generate natural-language chess commentary by combining a controllable BART language model with a symbolic chess engine (Leela, an AlphaZero variant) rather than Stockfish. The engine supplies grounded signals per move: win probability after the move, the optimal move, and the win-probability gap, which is bucketed into move-quality labels (excellent/good/inaccurate/mistake/blunder). Training extracts "control tags" (commentary type, move quality, suggested moves, pronouns, length) from data and trains P(C|G,M,T) where G is game-state, M move, T tags; at inference Leela produces the tags that steer generation. Game-state is encoded as move history, piece locations, and attack relationships. They train on ~374k game-state/commentary pairs from gameknot.com plus ~56k QA pairs for tag pre-training, targeting three commentary categories (description, quality, comparison). Tag-extraction F1 ranges 0.667-0.827; human raters prefer their outputs over Jhamtani et al. (2018) in 72-75% of cases. Notably, crowdworkers sometimes disagree with the super-human engine's assessments.

**Depends on.** Builds on Jhamtani et al. (2018) hand-crafted-feature LSTM chess commentary (the main baseline); BART (Lewis et al.); the Leela Chess Zero / AlphaZero engine line for move evaluation; controllable text generation via control tags.

**Results.** Tag-extraction F1 of 0.667-0.827 across tag types; human evaluators preferred their commentary over Jhamtani et al. (2018) in 72-75% of cases across the three commentary types. Ablation baselines (unconditioned BART, game-state-only BART) underperform the full tag-conditioned model on perplexity. No BLEU reported; evaluation rests on perplexity and crowdworker preference.

**Weaknesses.** Engine grounding is coarse: it injects only win-probability/optimal-move/quality bucket, not the tactical motifs (pins, forks, skewers) that explanations actually cite, so factor-level grounding is shallow. Reported logical-reasoning errors and grounding failures (commentary contradicting the board) when needed deductions aren't in the input representation. Multi-faceted analyses are weak. Evaluation is preference-based human judgment with no faithfulness or causal test of whether stated reasons are load-bearing; authors themselves note crowdworkers disagree with the super-human engine, undermining human eval as ground truth. Data is noisy amateur web commentary (gameknot/forums). No public causal or counterfactual validation of explanation correctness.

**Relation to xai-chess.** It is the canonical engine-grounded chess-commentary generator that ChessFaith complements: where Lee et al. produce explanations grounded in coarse engine win-probabilities and evaluate by human preference, ChessFaith instead grades factor-level explanations by causal intervention against an oracle engine, supplying the faithfulness test their preference-based evaluation lacks.


## 22. Automated Chess Commentator powered by a Neural Chess Engine

**Citation:** H. Zang, Z. Yu, and X. Wan, "[Automated Chess Commentator Powered by Neural Chess Engine](https://arxiv.org/abs/1909.10413)," arXiv:1909.10413, 2019.  
**Link:** <https://arxiv.org/abs/1909.10413> · **Tier:** CORE

**Summary.** Zang, Yu, and Wan (2019) build an automated chess commentator that couples a learned neural chess engine with category-specific text decoders. The engine encodes a board as 20 feature planes (piece positions, repetition history, castling rights) through CNN layers into a state embedding that jointly predicts a move-policy distribution and a scalar winning-rate value (an AlphaZero-style policy/value net trained on ~36M FICS positions plus self-play). These internal representations and self-play rollouts feed LSTM decoders with attention to generate five commentary types: description, move quality (from win-rate change), comparison (actual vs. engine-preferred move), planning (via self-play simulation), and broader contexts. They train and evaluate on the Jhamtani et al. (2018) Chess Commentary corpus (~298K move-comment pairs, 11,578 GAMEKNOT games), reporting BLEU/METEOR and human ratings (fluency, accuracy, insight, overall) against GAC, KWG, template, and retrieval baselines. The multi-task variant (SCC-mult) wins human evaluation with significance, showing the internal engine adds analytical depth over external-feature pipelines.

**Depends on.** Builds on the Jhamtani et al. (2018) Chess Commentary dataset and the GAC commentator baseline; the engine is an AlphaZero-style CNN policy/value network trained on FICS games plus self-play; generation uses seq2seq LSTM decoders with Luong attention.

**Results.** Multi-task SCC-mult leads human evaluation with statistical significance (p<0.01) over GAC and retrieval baselines, scoring highest on accuracy (3.91/5) and insight (3.51/5). Automatic scores are modest and category-dependent (e.g., Quality BLEU-4 ~20.06%, METEOR ~25.37%; Comparison/Planning/Contexts BLEU-4 under ~4%). Ablations show a stronger engine improves accuracy/overall, while multi-task learning slightly lowers BLEU but improves overall human-judged quality.

**Weaknesses.** The "insight"/"accuracy" of commentary is judged by surface text-overlap metrics (BLEU/METEOR are very low for most categories) and subjective human Likert scores, with no causal or counterfactual test of whether a cited factor (e.g., a pin or a win-rate change) actually drives the move's value — exactly the verifiable decision-level ground truth this is missing. The engine's win-rate signal is generated by the same model, so quality/comparison claims are self-referential rather than validated against a strong independent oracle. Generated explanations are templated to fixed categories and not grounded in tactically named concepts, and the GAMEKNOT corpus is noisy crowd commentary, limiting reliability of the supervision signal.

**Relation to xai-chess.** It is a canonical chess-commentary generation system that produces factor-citing move explanations (pins, comparisons, plans) but evaluates them only by text overlap and human ratings, motivating xai-chess's causal-intervention benchmark that tests whether such cited factors are actually load-bearing.


## 23. Understanding Learned Look-Ahead in Chess Neural Networks

**Citation:** D. Cruz, "[Understanding the Learned Look-Ahead Behavior of Chess Neural Networks](https://arxiv.org/abs/2505.21552)," arXiv:2505.21552, 2025.  
**Link:** <https://arxiv.org/abs/2505.21552> · **Tier:** CORE

**Summary.** This mechanistic-interpretability study (resolved; note arXiv lists a single author, Diogo Cruz, not "Cruz et al.") investigates how Leela Chess Zero's policy network (a 15-layer, ~109M-parameter transformer treating the 64 squares as sequence positions, finetuned to take only the current board) performs implicit look-ahead. Building on Jenner et al. (2024), it applies three causal/correlational interpretability tools: activation patching (swapping clean vs. corrupted board-state activations, scored by log-odds reduction on the correct move), linear probing of hidden states for future moves, and attention-head ablation. Findings: look-ahead is strongly position-dependent rather than universal; the network encodes information about moves up to ~7 plies ahead (probe accuracy decays but stays above baseline); patching the squares of specific future moves changes the current move choice; in ~50/50 branching positions, corrupting one branch's 3rd-move square shifts choice to the alternative branch, evidencing simultaneous multi-line consideration; and specialized heads (L12H12 checkmate-sensitive, L12H17 otherwise) handle pattern-specific cases.

**Depends on.** Extends Jenner et al. (2024) "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network"; uses Leela Chess Zero (AlphaZero-style policy transformer); standard mechanistic-interpretability methods (activation/causal patching, linear probing, attention-head ablation).

**Results.** Probing recovers future-move information up to the 7th ply (accuracy "considerably low but still non-negligible" vs. random). Activation patching of future-move squares produces large log-odds shifts in some puzzles (e.g., 5th-move square strong for puzzle set 11223) but negligible in others (11233), showing context-dependence. Branch-corruption experiments flip the chosen line in near-equiprobable positions, supporting parallel multi-branch evaluation. Head ablation isolates checkmate-tuned (L12H12) vs. non-checkmate (L12H17) attention heads.

**Weaknesses.** Authors concede they cannot distinguish "true planning" from sophisticated pattern matching. Evidence is largely qualitative/per-puzzle with small, hand-selected positions; no systematic dataset-wide effect sizes or significance testing, and 7-move analysis is data-limited. Patching may miss effects distributed across components. Scope is one finetuned LCZero variant in chess only, limiting generality. Probing accuracy can reflect decodability rather than causal use; the paper conflates correlational (probing) and causal (patching) evidence without a unified metric.

**Relation to xai-chess.** It supplies the internal-representation analogue of ChessFaith's external causal intervention: where Cruz patches/ablates a network's activations to test whether future-move information is load-bearing for the move, ChessFaith intervenes on the board/rules and measures an oracle's win-probability drop to test whether a cited factor is load-bearing for the explanation.


# Supporting

*Methodologically adjacent: interpretability, simulatability, concept-eval, and search-explanation work that informs the design.*


## 24. Towards A Rigorous Science of Interpretable Machine Learning

**Citation:** F. Doshi-Velez and B. Kim, "[Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/abs/1702.08608)," arXiv:1702.08608, 2017.  
**Link:** <https://arxiv.org/abs/1702.08608> · **Tier:** SUPPORTING

**Summary.** A position paper arguing that interpretable ML lacks rigorous definitions and measurement standards, and proposing scaffolding to fix this. The authors offer a working definition of interpretability (the ability to explain or present in understandable terms to a human) and argue it is a means to satisfy auxiliary desiderata (safety, non-discrimination, trust, transferability, causality) that escape standard performance metrics; where those needs are absent or fully formalizable, interpretability may be unnecessary. Their central contribution is a taxonomy of evaluation along a cost/validity spectrum: application-grounded (real humans, real tasks), human-grounded (real humans, simplified proxy tasks), and functionally-grounded (no humans; a formal proxy metric validated against human studies). They call for matching the evaluation tier to claims, defining latent task/method factors (e.g., global vs local, time constraints, cognitive chunks) so results generalize, and treating interpretability claims as hypotheses requiring evidence rather than intuition. Prior project notes accurately reflect the source.

**Depends on.** Frames a broad research agenda rather than building on a specific method; synthesizes prior interpretability/explanation work and HCI evaluation practice into a unifying taxonomy.

**Results.** Conceptual, not empirical: no datasets or quantitative results. Deliverables are (1) a definition of interpretability and an account of when it is/ isn't needed, (2) the three-tier evaluation taxonomy (application-, human-, functionally-grounded) with explicit cost/specificity tradeoffs, and (3) a proposed open research agenda including a data-driven taxonomy of task- and method-level latent factors so functionally-grounded proxies can be validated against human-grounded studies.

**Weaknesses.** It is a manifesto: it diagnoses the problem and names tiers but supplies no concrete metrics, benchmarks, or operational protocols, so "rigor" remains aspirational. The interpretability definition is admittedly loose and not measurable as stated. The proposed latent factors are speculative and never empirically derived. Crucially for causal-faithfulness work, the framework centers human understanding and says little about verifying that an explanation reflects the model's actual decision mechanism; functionally-grounded evaluation is described abstractly without telling you how to build a faithful proxy. Now nearly a decade old, it predates the LLM-explanation faithfulness literature.

**Relation to xai-chess.** It provides the foundational vocabulary that situates ChessFaith as a functionally-grounded evaluation, while the project supplies the concrete, causal, decision-level proxy metric the paper calls for but never specifies.


## 25. Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?

**Citation:** P. Hase and M. Bansal, "[Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?](https://arxiv.org/abs/2005.01831)," in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, arXiv:2005.01831.  
**Link:** <https://arxiv.org/abs/2005.01831> · **Tier:** SUPPORTING

**Summary.** Hase and Bansal evaluate explainable-AI methods by whether they improve human "simulatability" — a user's ability to anticipate model behavior on new inputs. They run controlled human-subject studies across tabular and text classification, using two protocols: forward simulation (predict the model's output on unseen examples after seeing explained examples) and counterfactual simulation (predict how the model responds to perturbed inputs). They compare five explanation conditions: LIME, Anchor, Decision Boundary, a Prototype model, and a Composite of all four, each against a no-explanation control. The central finding is that explanation effectiveness is rare and method-specific: LIME helped in tabular classification and the Prototype method helped in counterfactual tests, but clear evidence of benefit appeared in very few cases overall. Critically, users' subjective ratings of explanation quality did NOT predict actual helpfulness for simulation, exposing a gap between perceived and functional value of explanations and motivating rigorous behavioral evaluation over self-report.

**Depends on.** Builds on prior XAI methods it evaluates (LIME, Anchors, prototype/case-based networks, decision-boundary explanations) and on the simulatability framing from interpretability desiderata literature (e.g., Lipton; Doshi-Velez and Kim).

**Results.** Across forward and counterfactual human-subject tests on tabular and text data with five explanation conditions: LIME significantly improved simulatability in tabular classification; the Prototype method helped in counterfactual simulation; but clear evidence of effectiveness was found in very few cases overall. Subjective user ratings of explanations were not predictive of how much those explanations actually helped users predict model behavior.

**Weaknesses.** Effectiveness is measured via human simulation, which is noisy, costs-intensive, and sensitive to interface, instructions, and user expertise; high variance with limited participants yields wide confidence intervals and weak statistical power, so many null results may reflect underpowering rather than true ineffectiveness. Findings are limited to a few datasets/models and the specific explanation implementations chosen; the "Composite" condition may overload users rather than test additivity. Simulatability conflates whether an explanation reveals the true model reasoning with whether users can exploit it — a faithful explanation could still fail this human-centric test, and vice versa.

**Relation to xai-chess.** It supplies the human-centric counterpoint and cautionary precedent — that subjective ratings do not equal functional helpfulness — motivating ChessFaith's shift to verifiable, model-side causal-intervention ground truth rather than human or self-reported judgments of explanation quality.


## 26. ALMANACS: A Simulatability Benchmark for Language Model Explainability

**Citation:** E. Mills, S. Su, S. Russell, and S. Emmons, "[ALMANACS: A Simulatability Benchmark for Language Model Explainability](https://arxiv.org/abs/2312.12747)," arXiv:2312.12747, 2023.  
**Link:** <https://arxiv.org/abs/2312.12747> · **Tier:** SUPPORTING

**Summary.** ALMANACS is a fully automated simulatability benchmark for LLM explainability. It scores explanation methods by whether they help a separate predictor (GPT-4 doing in-context prediction over 10 nearest-neighbor training examples, retrieved via Sentence-BERT cosine similarity) better predict a target model's output probabilities on unseen inputs. Targets are flan-alpaca-gpt4-xl (3B) and vicuna-7b-v1.3 (7B). The dataset spans 12 safety-relevant topics, each with 15 adversarially selected templates (5 placeholders x 15 values), 500 train + 50 test questions per template (180k train / 18k test total). A deliberate train/test distributional shift (test questions use unseen value combinations and entirely new values) is meant to defeat spurious in-context matching and reward genuinely explanatory content. Four methods are evaluated (Counterfactual, Rationalization, Attention, Integrated Gradients) against a no-explanation control, scored by KLDiv, total-variation distance, and Spearman correlation. The central, cautionary finding: averaged across topics, no explanation method beats the explanation-free control.

**Depends on.** Builds on the simulatability/forward-prediction tradition in interpretability (Doshi-Velez & Kim; Hase & Bansal) and on counterfactual, rationalization, attention, and Integrated Gradients explanation methods; uses GPT-4 as an automated in-context predictor and Sentence-BERT (all-mpnet-base-v2) retrieval.

**Results.** Mean KLDiv (lower better) shows explanations at or above the control. flan-alpaca-gpt4-xl: NoExpl 0.10, Counterfactual 0.09, Rationalization 0.10, Attention 0.10, IntegratedGradients 0.10. vicuna-7b-v1.3: NoExpl 0.08, Counterfactual/Attention 0.08, IntegratedGradients 0.09, Rationalization 0.10. Naive learned baselines (e.g., LogisticRegression 0.11 flan / 0.09 vicuna) were worse than NoExpl. The only isolated win was Counterfactuals on the Sycophancy topic for flan-alpaca (KLDiv 0.19 to 0.15). Conclusion: explanations did not improve simulatability over no explanation across the board, and the field remains substantially unsolved.

**Weaknesses.** The null result may reflect the evaluation design as much as the explanations: the GPT-4 predictor with only 10 retrieved examples may be a weak/insensitive student, and there is no validation that any known-good explanation can move the metric (no positive control / ceiling), so "explanations don't help" is confounded with "the student can't exploit help." Only two small open models (3B, 7B) and a single predictor (GPT-4) are tested, limiting generality. Probability outputs are scalar yes/no, narrowing what an explanation can convey. Attention/IntegratedGradients are token-attribution methods awkwardly forced into a textual predictor prompt, possibly unfairly. Authors concede it is not a replacement for human evaluation. Adversarial template selection may bias toward hard, atypical behaviors.

**Relation to xai-chess.** It is the cautionary blueprint for any simulatability track in ChessFaith: before claiming explanations help, the benchmark must include a positive control proving a known-good explanation moves the predictor, and must construct the train/test shift so in-context example matching cannot substitute for the explanation's causal content.


## 27. ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability

**Citation:** A. Poché, A. Jacovi, A. M. Picard, et al., "[ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](https://arxiv.org/abs/2501.05855)," arXiv:2501.05855, 2025.  
**Link:** <https://arxiv.org/abs/2501.05855> · **Tier:** SUPPORTING

**Summary.** ConSim evaluates concept-based explanations end-to-end via automated simulatability: how well a meta-predictor (simulator) can reproduce an explained classifier's outputs given the concepts and their explanations. It replaces costly human simulator studies with LLMs (primarily GPT-4o-mini; also Gemini-1.5 Flash/Pro). The protocol runs three phases: an Initial Phase (task description plus global concept-importance scores), a Learning Phase (sample-explanation-prediction examples), and an Evaluation Phase (predict labels on held-out samples). It benchmarks concept-extraction methods (NMF, SAE, ICA, PCA, SVD) and interpretation methods (top-5 maximally-activating words vs. o1-generated concept labels) across text datasets (BIOS, IMDB, Rotten Tomatoes, TweetEval emotion) and base models (DistilBERT, T5, Llama3-8B). Scores are aggregated across non-comparable settings using Copeland ranked-choice voting, with t-tests for significance. Findings: NMF ranks best (then SAE, ICA, PCA, SVD, no-explanation baseline); maximally-activating-words beats LLM labels; rankings stay consistent across the three LLM simulators, supporting scalable automated evaluation.

**Depends on.** Builds on simulatability as a faithfulness/usefulness criterion (Hase & Bansal; Doshi-Velez & Kim), LLM-as-judge/simulator paradigms, concept-based explanation methods (TCAV, NMF/SAE/ICA concept extraction, sparse autoencoders), and ranked-aggregation (Copeland voting).

**Results.** NMF is the best concept-extraction method, followed by SAE, ICA, PCA, SVD, and a no-explanation baseline last; concept-maximally-activating-words (CMAW) interpretation outperforms o1-generated concept-alignment labels across datasets. Rankings are stable across three distinct LLM simulators (GPT-4o-mini, Gemini-1.5 Flash and Pro), with most pairwise differences statistically significant (t-test, p<0.05), demonstrating that automated simulatability gives consistent, scalable rankings of explanation methods.

**Weaknesses.** No validation that LLM-simulator rankings match human-simulator rankings (authors explicitly concede "no actual proof"), so the central scalability claim rests on an unverified proxy. Scope is limited to text classification; the assumption that vision models behave like NLP models for penultimate-layer concept extraction is untested. The SAE implementation omits recent improvements, likely understating SAE. Simulatability measures whether explanations help predict outputs (plausibility/usefulness), not whether concepts are genuinely causally responsible for the model's decision—it is correlational, not interventional. Risk of LLM-simulator shortcut-learning or shared-bias inflation when judge and explanations are both LLM-mediated.

**Relation to xai-chess.** ConSim exemplifies the LLM-as-automated-evaluator-for-explanations paradigm that ChessFaith parallels, but its simulatability proxy is correlational, underscoring the project's contribution of verifiable decision-level causal ground truth via direct interventions rather than predict-the-output simulation.


## 28. Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting

**Citation:** M. Turpin, J. Michael, E. Perez, and S. R. Bowman, "[Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388)," in Advances in Neural Information Processing Systems (NeurIPS), arXiv:2305.04388, 2023.  
**Link:** <https://arxiv.org/abs/2305.04388> · **Tier:** SUPPORTING

**Summary.** Turpin et al. show that chain-of-thought (CoT) explanations from LLMs can systematically misrepresent the true causes of a model's predictions. Their core method is a causal-style intervention on the input: they inject biasing features that should not affect the answer, then check whether the model's stated reasoning ever mentions them. Two biases are used: a "suggested answer" / answer-reordering bias that always makes "(A)" correct in few-shot examples, and a stereotype-aligned bias on a social-bias task. On a suite of 13 BIG-Bench Hard tasks with GPT-3.5 and Claude 1.0, the planted bias drops accuracy by as much as 36%, yet the generated CoT rarely acknowledges the bias and instead constructs plausible-sounding rationalizations for the biased answer. The takeaway: a faithful-looking explanation is not evidence that the cited reason was load-bearing, so faithfulness must be tested by intervention rather than read off the explanation text. They argue improving faithfulness needs targeted methods.

**Depends on.** Builds on chain-of-thought prompting (Wei et al.) and the broader explanation-faithfulness literature (Jacovi & Goldberg); uses BIG-Bench Hard tasks and a BBQ-style social-bias task; evaluates GPT-3.5 and Claude 1.0.

**Results.** Planted biasing features (e.g., always-(A) answer reordering) reduce model accuracy by up to 36% across the 13 BIG-Bench Hard tasks; models almost never cite the bias in their CoT and instead rationalize the biased answer. On the social-bias task, models produce stereotype-consistent answers while giving explanations that omit the stereotype as the cause. Demonstrates that CoT explanations are frequently unfaithful and that plausibility does not imply faithfulness.

**Weaknesses.** Limited to two proprietary, now-dated models (GPT-3.5, Claude 1.0) and multiple-choice BBH/social-bias tasks, so generality to other architectures, scales, and open-ended generation is untested. "Unfaithfulness" is operationalized as failure to verbalize a known planted bias — a necessary but coarse proxy that detects omission, not the full causal mechanism, and cannot quantify how much each cited factor actually drove the output. The interventions perturb the input distribution (few-shot ordering), which may shift behavior in ways beyond the intended single biasing feature. No ground-truth decision-level causal labels exist; faithfulness is inferred behaviorally rather than measured against a verifiable causal graph.

**Relation to xai-chess.** It motivates ChessFaith's central premise — that explanations must be validated by causal intervention rather than trusted on their face — and supplies the input-perturbation/planted-bias template that the chess benchmark replaces with verifiable decision-level causal ground truth via piece-relocation and rule-suspension operators.


## 29. Measuring Faithfulness in Chain-of-Thought Reasoning

**Citation:** T. Lanham, A. Chen, A. Radhakrishnan, et al., "[Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702)," arXiv:2307.13702, 2023.  
**Link:** <https://arxiv.org/abs/2307.13702> · **Tier:** SUPPORTING

**Summary.** Lanham et al. (Anthropic) ask whether a language model's chain-of-thought (CoT) actually drives its final answer, or is post-hoc rationalization. They define faithfulness operationally via four interventions on the reasoning trace and measure resulting answer changes: (1) early answering—truncating CoT at each sentence to see if the answer is already fixed; (2) adding mistakes—injecting a corrupted step and checking whether the model follows the error; (3) paraphrasing—rewording reasoning to rule out steganographic phrase-encoding; (4) filler tokens—replacing CoT with "…" to rule out mere extra compute. Tested on a 175B RLHF assistant (plus 13B/70B) across eight multiple-choice benchmarks (AQuA, ARC Easy/Challenge, HellaSwag, LogiQA, MMLU, OpenBookQA, TruthfulQA). Faithfulness varies enormously by task (AQuA highly faithful; ARC Easy near post-hoc) and, strikingly, decreases with scale: for 7/8 tasks the 13B model conditions on CoT more than the 175B. Paraphrasing and filler experiments show gains come from reasoning content, not phrasing or extra tokens.

**Depends on.** Builds on chain-of-thought prompting (Wei et al.) and prior critiques that explanations may be unfaithful/post-hoc (Turpin et al. on biasing features; general interpretability-faithfulness literature). Uses standard QA benchmarks (AQuA, ARC, HellaSwag, LogiQA, MMLU, OpenBookQA, TruthfulQA) and an RLHF-tuned assistant.

**Results.** Early answering: answer-change rates span 44% (AQuA, faithful) to ~2% (ARC Easy, post-hoc). Adding mistakes: AQuA ~28% answer changes vs near-zero on easy tasks. Paraphrasing: paraphrased-CoT accuracy closely matches original on almost all tasks (no steganographic encoding). Filler tokens: replacing CoT with "…" gives no accuracy gain (benefit is not extra compute). Inverse scaling: for 7/8 tasks the 13B model changes its answer post-CoT more than the 175B; on synthetic addition, faithfulness drops as model size rises and as task difficulty falls. Conclusion: CoT faithfulness is task- and scale-dependent, not guaranteed.

**Weaknesses.** No ground-truth faithfulness: "faithfulness" is inferred only from behavioral sensitivity to interventions, so a model could be both insensitive yet genuinely using the reasoning, or sensitive for spurious reasons—the metrics are proxies, not causal proof of the internal mechanism. Restricted to multiple-choice tasks with discrete answers, easing answer-change measurement but limiting generality to open-ended generation. Single model family (one Anthropic RLHF assistant); results may not transfer across architectures/training regimes. Interventions like "adding mistakes" rely on a separate model to generate plausible errors, introducing confounds. The inverse-scaling claim is correlational and per-task noisy. No external causal oracle anchors whether the cited factor is truly decision-relevant—exactly the verifiable ground truth the chess benchmark supplies.

**Relation to xai-chess.** It provides the core multi-operator intervention recipe (truncation, corruption/mistake injection, paraphrase, filler controls) and the answer-change metric that ChessFaith adapts into causal win-probability drops, while ChessFaith supplies the verifiable decision-level ground truth this behavioral approach lacks.


## 30. Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?

**Citation:** A. Jacovi and Y. Goldberg, "[Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?](https://arxiv.org/abs/2004.03685)," in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 4198-4205. arXiv:2004.03685.  
**Link:** <https://arxiv.org/abs/2004.03685> · **Tier:** SUPPORTING

**Summary.** This widely-cited position paper (ACL 2020) argues that NLP interpretability research conflates two distinct criteria: faithfulness (how accurately an explanation reflects the model's true reasoning process) and plausibility (how convincing it is to humans). The authors warn that conflation is dangerous in high-stakes settings and lay out methodological guidelines. They identify three implicit assumptions underlying existing faithfulness evaluations: the Model Assumption (two models make the same predictions iff they reason the same way), the Prediction Assumption (similar inputs yield similar decisions iff reasoning is similar), and the Linearity Assumption (input parts contribute importance independently, justifying heatmaps/erasure). Key guidelines: be explicit about which property you evaluate; do not use human judgment or gold labels to assess faithfulness (those measure plausibility); distrust "inherently interpretable" claims; reject end-task/utility-based proxies. They argue the binary faithful/unfaithful bar is unrealistic and advocate a graded, conditional notion: measure the degree and likelihood of faithfulness per task/model and across input subregions rather than as a global all-or-nothing property.

**Depends on.** Builds on and critiques the broader interpretability/explainability literature (e.g., LIME, attention-as-explanation debates, erasure and probing methods) and Lipton's "Mythos of Model Interpretability"; it is a conceptual/position paper rather than an empirical method, synthesizing prior evaluation practices.

**Results.** No experiments; the output is a conceptual framework. Deliverables: the faithfulness/plausibility distinction; the three-assumption taxonomy used to organize and disprove faithfulness claims (assumptions chiefly support disproof via counterexamples, not proof); a set of prescriptive guidelines (separate faithfulness from plausibility, avoid human judgment/gold labels/utility metrics, validate inherent-interpretability claims); and the proposal to treat faithfulness as graded ("grayscale") and conditional on task, model, and input subregion rather than binary.

**Weaknesses.** Purely conceptual: it offers definitions and prescriptions but no concrete metric, benchmark, or operationalization for measuring graded faithfulness, leaving "how much is enough" unanswered. The three assumptions are framed mainly for disproving faithfulness, giving little constructive guidance for positive verification. Its strict ban on human judgment is contestable when ground-truth reasoning is unobservable in real models. NLP/deep-learning focused; transfer to other domains is implicit. Provides no causal-intervention recipe, so the "load-bearing" test must be borrowed from elsewhere.

**Relation to xai-chess.** It supplies the foundational faithfulness-vs-plausibility distinction and the graded, "load-bearing" framing that ChessFaith operationalizes by causally neutralizing a cited factor and measuring an oracle engine's win-probability drop, replacing unobservable NLP reasoning with verifiable decision-level causal ground truth.


## 31. Interpretability Beyond Feature Attribution: Testing with Concept Activation Vectors (TCAV)

**Citation:** B. Kim, M. Wattenberg, J. Gilmer, et al., "[Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://arxiv.org/abs/1711.11279)," in Proc. Int. Conf. Machine Learning (ICML), 2018; arXiv:1711.11279.  
**Link:** <https://arxiv.org/abs/1711.11279> · **Tier:** SUPPORTING

**Summary.** TCAV makes neural-network explanations human-concept-centric rather than per-feature. A user defines a concept (e.g., "stripes") via example images; a linear classifier is trained to separate that concept's layer activations from random counterexamples, and the classifier's normal vector becomes the Concept Activation Vector (CAV). Conceptual sensitivity is the directional derivative of a class logit along the CAV; the TCAV score is the fraction of a class's inputs with positive sensitivity, i.e., the share whose prediction is pushed toward the class by the concept. Crucially, TCAV trains many CAVs against many random counterexample sets (~500) and runs a two-sided t-test (with Bonferroni correction) so that scores are only reported if significantly different from random concepts. Experiments on GoogleNet/Inception-v3 recover expected concepts (red->fire engine, stripes->zebra), expose gender/race biases, pass a controlled noise ground-truth test, and apply to diabetic-retinopathy fundus images. A human study found saliency maps conveyed importance only 52% of the time.

**Depends on.** Builds on deep CNN classifiers (GoogleNet, Inception-v3) and their internal activation spaces; contrasts with gradient/saliency feature-attribution methods (e.g., integrated gradients, SmoothGrad); uses linear probing of activations and standard significance testing (t-test, Bonferroni).

**Results.** Recovered intuitive concept-class associations and uncovered unintended gender/race sensitivity not in training labels. A controlled caption-vs-image noise experiment showed TCAV correctly tracked which modality the model relied on, matching approximated ground truth across noise levels. Sanity checks: CAV-based image sorting and Deep Dream visualizations aligned with concepts. In diabetic retinopathy, TCAV flagged that the model over-predicted level-1 as level-2 using features a clinician deemed inappropriate. Human evaluation: saliency maps communicated concept importance correctly only ~52% of the time (chance 50%), motivating TCAV's quantitative alternative.

**Weaknesses.** CAVs assume concept directions are linearly separable in activation space (local-linearity assumption) and are entirely dependent on the user-curated example set, which can encode spurious cues; even random image sets yield non-trivial vectors, hence the heavy reliance on statistical filtering. The method explains correlational sensitivity of internal representations to a concept, not a verified causal effect on decisions, and requires labeled concept examples plus white-box gradient access. The significance test guards against random concepts but not against confounded or entangled concepts. Layer choice affects results; the medical case shows TCAV can surface features experts reject, complicating ground-truth validation. The adversarial-detection result is acknowledged as insufficient for a defense.

**Relation to xai-chess.** TCAV is a key precursor for measuring whether a named, human-defined concept actually drives a model's output, but it stays correlational and white-box, whereas ChessFaith supplies the verifiable decision-level causal ground truth TCAV lacks by intervening on the concept (board-level relocation or rule-level movegen patch) and measuring an oracle engine's win-probability change.


## 32. Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep RL

**Citation:** A. Atrey, K. Clary, and D. Jensen, "[Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep Reinforcement Learning](https://arxiv.org/abs/1912.05743)," in Proc. Int. Conf. Learning Representations (ICLR), 2020. arXiv:1912.05743.  
**Link:** <https://arxiv.org/abs/1912.05743> · **Tier:** SUPPORTING

**Summary.** Atrey, Clary, and Jensen (ICLR 2020) argue that saliency maps used to explain deep RL agents are typically unfalsifiable and subjective, with a survey finding only ~7% of saliency claims experimentally validated. They propose a counterfactual methodology that intervenes on the latent game state (not pixels), using ToyBox, a fully parameterized reimplementation of Atari games, so the pixel-generating process is preserved. They evaluate three saliency methods (Jacobian/gradient, perturbation/Gaussian-blur, and object/template-matching) on Breakout and Amidar across three case studies: brick/tunnel translation, score manipulation, and enemy-distance manipulation. They distinguish distortion, semantics-preserving, and fat-hand interventions. In all three studies, intuitive human-generated hypotheses about agent behavior (tunnel-aiming, using displayed score, reacting to nearby enemies) were falsified: saliency patterns persisted or were confounded by location even as behavior changed, with near-zero correlation/causal effect for enemy distance. They conclude saliency maps should be treated as an exploratory, hypothesis-generating tool, not an explanatory one reflecting causal structure.

**Depends on.** Builds on saliency/attribution methods for deep RL (Jacobian gradients; Greydanus et al. perturbation saliency; object/template-matching saliency), the ToyBox parameterized Atari environment for state intervention, and the broader counterfactual/interventional reasoning tradition (Pearl-style do-interventions) applied to evaluating explanations.

**Results.** Across three Atari case studies all initial hypotheses were contradicted. Breakout tunnel saliency was confounded by board location rather than reflecting a causal tunnel-aiming strategy. Amidar score interventions (reset, randomized, fixed, decremented) changed accumulated reward/behavior while saliency over the score region stayed similar. Amidar enemy-distance showed near-zero Pearson correlation and near-zero causal effect of proximity on saliency. A literature survey found only ~7% of saliency-based claims included experimental validation. Net conclusion: saliency maps cannot be trusted to reflect causal relationships between semantic concepts and agent behavior.

**Weaknesses.** Scope is narrow: only two games (Breakout, Amidar) and feed-forward agents; recurrent agents are excluded because repeated interventions interact with memorization. The method requires a fully intervenable, parameterized simulator (ToyBox), so it does not transfer to observation-only or real-world settings. It evaluates only three saliency techniques and a handful of hand-picked hypotheses, so falsification of these does not establish that saliency is uninformative in general (absence-of-evidence risk). The agent's learned semantic space may differ from human concepts, making "falsification" partly a mismatch of ontologies. Results are largely qualitative/descriptive with limited statistical hypothesis testing or confidence intervals, and no ground-truth notion of what the agent actually computes.

**Relation to xai-chess.** It is the closest precedent for ChessFaith's core move: using state-level (rather than pixel-level) causal interventions in a parameterized game environment to falsify explanation claims, which xai-chess extends from saliency-map plausibility to verifiable decision-level causal ground truth for factor-based chess explanations.


## 33. Not All Explanations Are Created Equal: Investigating the Pitfalls of Current XAI Evaluation

**Citation:** J. Shymanski, J. Brue, and S. Sen, "[Not All Explanations Are Created Equal: Investigating the Pitfalls of Current XAI Evaluation](https://arxiv.org/abs/2511.03730)," arXiv:2511.03730, 2025 (also Ch. 9, Bi-directionality in Human-AI Collaborative Systems, Springer, 2025).  
**Link:** <https://arxiv.org/abs/2511.03730> · **Tier:** SUPPORTING

**Summary.** Shymanski, Brue, and Sen argue that prevailing XAI evaluation, especially subjective user-satisfaction surveys, is too weak to distinguish good explanations from bad ones, because almost any explanation raises satisfaction over none. They test this with a chess-teaching agent on Amazon Mechanical Turk (108 recruited; ~102 practice / 70 test users), a between-subjects design with three conditions: no explanation, "placebic" explanations ("this move is most advantageous"), and "actionable" explanations naming the tactic ("this forks the king and rook"). Participants solved up to 10 training and 5 testing fork/pin puzzles. Subjective measures showed no significant differences across conditions (practice satisfaction p=.504, agent satisfaction p=.851, perceived explanatory power p=.213). Objective transfer differed sharply: actionable beat placebic on testing scores (p=.0236); the actionable group improved 2.2% from practice to test while the placebic group dropped 21.3%. The authors conclude evaluation must measure objective, actionable utility rather than satisfaction.

**Depends on.** Builds on the placebic-vs-actionable explanation distinction (Langer's "placebic information" lineage) and the broader human-grounded XAI evaluation literature critiquing satisfaction/trust surveys; uses a chess-tactics (fork/pin) tutoring setting.

**Results.** Satisfaction and perceived explanatory power did not differ across no-explanation, placebic, and actionable conditions (p=.504, .851, .213). Objective learning transfer did: actionable > placebic on test scores (p=.0236); actionable group +2.2% practice-to-test vs placebic -21.3%; placebic helped practice over none (p=.0238) but not durable test performance. Demonstrates satisfaction is blind to explanation quality while actionability predicts transfer.

**Weaknesses.** Small, MTurk crowd sample with uneven cell sizes (29/32/35 survey respondents) and heavy attrition (108 to 70 testers) limits power and generalizability. Confounds the authors themselves flag: screen-blur during explanations may have frustrated users, and the no-explanation group could answer faster without interruption, so condition effects conflate content with UX friction. Many participants lacked chess expertise, muddying who can benefit. Outcome is human learning/transfer, not faithfulness to a model's actual reasoning, so it does not establish whether explanations reflect the agent's true decision basis. p=.0236 is modest and not corrected for multiple comparisons.

**Relation to xai-chess.** Empirically motivates this project's rejection of subjective satisfaction in favor of an objective, causal-intervention metric for grading chess move-explanations, and does so in the same chess-tactics (fork/pin) domain.


## 34. Towards Explaining Monte-Carlo Tree Search by Using Its Enhancements

**Citation:** J. Kowalski, M. H. M. Winands, M. Wiśniewski, S. Reda, and A. Wilbik, "[Towards Explaining Monte-Carlo Tree Search by Using Its Enhancements](https://arxiv.org/abs/2506.13223)," arXiv:2506.13223, 2025.  
**Link:** <https://arxiv.org/abs/2506.13223> · **Tier:** SUPPORTING

**Summary.** This work proposes using Monte-Carlo Tree Search (MCTS) enhancements as a source of additional, structured data for generating post-hoc, model-agnostic explanations of search-based agents, while remaining knowledge-free (domain-independent, no encoded game rules). The authors survey several standard enhancements and repurpose their internal statistics for explanation: MAST (move-average sampling) and its n-gram extension NST for context-free move value and move/reply exchanges; GRAVE/AMAF for cross-phase move estimates; Score-Bounded MCTS for pessimistic/optimistic terminal bounds; and PN-MCTS (proof-number) for proof/disproof workload on subtrees. They argue these statistics yield richer explanations than vanilla visit counts. The proof-of-concept is built atop the Ludii general game-playing system (1,400+ games, e.g., Breakthrough, Connect Four, Ultimate Tic-Tac-Toe), producing mixed quantitative (visit counts, win probabilities, AMAF scores, rankings) and qualitative natural-language narratives across six example positions (Table I). It is explicitly a work-in-progress: no user study or quantitative evaluation; LLM-polished narration is left to future work.

**Depends on.** Builds on classic MCTS and its enhancements (MAST, NST, GRAVE/AMAF, Score-Bounded MCTS, PN-MCTS/proof-number search) and the Ludii general game-playing framework; positioned within the "Explainable Search" subfield of XAI.

**Results.** No empirical metrics. Deliverable is a qualitative proof-of-concept: six annotated game positions where enhancement statistics are surfaced as explanations combining numeric scores with interpretive text. Claims that enhancement-derived data (e.g., AMAF move estimates, score bounds, proof-number workloads) provide higher-quality, more contextual explanations than visit-count-only baselines, all while staying domain-independent.

**Weaknesses.** Purely illustrative: no user study, no quantitative explanation-quality metric, and crucially no faithfulness/causal validation that the surfaced statistics actually drive the agent's chosen move. Explanations are descriptive readouts of internal counters, not tested interventions, so there is no ground truth for whether a cited factor is load-bearing. Knowledge-free framing limits explanations to opaque numbers/n-grams rather than human-meaningful tactical concepts (pins, hanging pieces). Generality across 1,400 Ludii games is asserted, not measured. Natural-language layer is hand-written/aspirational. Self-described work-in-progress.

**Relation to xai-chess.** It is a precedent for explaining a search agent by mining its internal search structure (motivating a deferred MCTS-subtree intervention) rather than a faithfulness competitor, since it offers descriptive, knowledge-free statistics with no causal-intervention test of whether cited factors are load-bearing—the exact gap ChessFaith fills.


## 35. Toward Template-Free Explainability for Monte Carlo Tree Search

**Citation:** S. Lu, M. Bahavarnia, H. Baroud, Y. Zhang, H. Purohit, and A. Mukhopadhyay, "[Toward Template-Free Explainability for Monte Carlo Tree Search](https://arxiv.org/abs/2605.16524)," arXiv:2605.16524, 2026.  
**Link:** <https://arxiv.org/abs/2605.16524> · **Tier:** SUPPORTING

**Summary.** The paper proposes an LLM-based, template-free framework for explaining Monte Carlo Tree Search (MCTS) decisions, motivated by the brittleness of prior approaches that require hand-crafted formal-logic constraints requiring updates when the problem changes. The pipeline maps a natural-language user question to a structured intent category, checks whether the existing search tree contains sufficient evidence to answer it, triggers targeted subtree expansion when evidence is lacking, then generates an explanation grounded in tree statistics (visit counts, value estimates, risk information). It handles node-level, path-level, and general-behavior questions including contrastive, counterfactual, and multi-action-trajectory queries. Evaluation is limited to the FrozenLake grid-world on a handcrafted set of 21 annotated question-tree pairs. Reported metrics cover intent-extraction accuracy (e.g., 85.7% question-type, 95.2% target-state, 71.4% target-action), 100% answerability detection, and keyword-based grounding checks (71.4% passing all checks). The authors argue LLMs can serve as end-to-end explainers for probabilistic search without intermediate formal representations.

**Depends on.** Builds on Monte Carlo Tree Search and prior template/formal-logic (e.g., CTL/LogiEx-style) MCTS explainability; relies on LLMs as natural-language explanation generators; evaluated on the FrozenLake RL benchmark.

**Results.** On FrozenLake with 21 question-tree pairs: intent extraction accuracy of 85.7% (question type), 95.2% (target state), 71.4% (target action), 90.5% (target path); answerability detection 100% (21/21); grounding checks: core decision mentioned 76.2%, risk-calculation evidence 90.5%, asked state-action pair 100%, all checks passed 71.4% (15/21). No baseline comparisons; targeted expansion not separately ablated.

**Weaknesses.** Evaluation is very narrow: a single toy domain (FrozenLake) and only 21 hand-authored question-tree pairs, with no baselines (no LogiEx/CTL comparison) and no ablation isolating the targeted-expansion mechanism. "Faithfulness" is assessed only by a keyword/grounding heuristic the authors themselves call an initial check, not a real causal or intervention-based test — so it cannot establish that explanations reflect the search's actual decision drivers. Metrics conflate intent-parsing accuracy with explanation quality; reported numbers are on tiny n (e.g., 71.4% = 15/21), giving wide uncertainty. Known issues with verbosity and incomplete action coverage. Generalization to large asymmetric trees or richer domains (e.g., chess) is unvalidated, and LLM hallucination over tree statistics is not bounded.

**Relation to xai-chess.** It is a tree-search explanation method whose self-acknowledged keyword "grounding" check exemplifies exactly the non-causal, surface-level faithfulness proxy that ChessFaith's intervention-based causal benchmark is designed to replace with verifiable decision-level ground truth.


## 36. MCTS Explainability via Computation Tree Logic

**Citation:** Z. An, H. Baier, A. Dubey, A. Mukhopadhyay, and M. Ma, "[Enabling MCTS Explainability for Sequential Planning Through Computation Tree Logic](https://arxiv.org/abs/2407.10820)," in Proc. 27th European Conf. on Artificial Intelligence (ECAI), 2024. arXiv:2407.10820.  
**Link:** <https://arxiv.org/abs/2407.10820> · **Tier:** SUPPORTING

**Summary.** An et al. (ECAI 2024) propose a framework for explaining Monte Carlo Tree Search (MCTS) decisions in sequential planning, demonstrated on transportation routing. The method translates user-defined requirements into formal Computation Tree Logic (CTL) specifications via natural-language templates, quantitatively verifies the MCTS search tree's states and actions against those specifications, and renders the verification results back into human-readable explanations through templates. The pipeline thus bridges natural language and rigorous temporal logic, letting non-expert users probe whether the planner's behavior satisfies stated requirements. The authors evaluate via a user study (82 participants) and report that their explanations significantly outperform baselines in user preference/satisfaction. The contribution is a verifiable, formal-logic-grounded explainer for a search-based decision process, rather than a post-hoc attribution method. The title given in the prompt is a short form; the full title is "Enabling MCTS Explainability for Sequential Planning Through Computation Tree Logic." Authors and arXiv ID match the prompt.

**Depends on.** Builds on Monte Carlo Tree Search for sequential planning, Computation Tree Logic / formal model checking, and template-based natural-language generation; situated in explainable AI (XAI) for sequential decision-making and planning agents.

**Results.** Explanations generated from CTL verification were preferred by users over baseline explanation methods in a survey of 82 participants, reported as a significant improvement in user preference/satisfaction. Evaluation is human-preference-based in a transportation routing domain; the paper does not report quantitative fidelity/causal-faithfulness metrics for the explanations.

**Weaknesses.** Evaluation is limited to subjective user preference (n=82) in a single transportation-routing domain, so generalization to other MCTS applications (e.g., game-playing engines like chess) is untested. It measures whether users like or understand explanations, not whether explanations are causally faithful to the planner's actual decision — preference is not faithfulness. The approach requires user-specified requirements and hand-built language templates, limiting scalability and risking that explanations reflect the template designer's framing. CTL verification over large search trees may face scalability/complexity issues, which the paper underplays. No comparison to intervention- or perturbation-based faithfulness tests; no decision-level causal ground truth.

**Relation to xai-chess.** It is a search-tree (MCTS) explainability method that, unlike ChessFaith's causal-intervention faithfulness test, grounds explanations in formal CTL specifications and evaluates them by user preference rather than by measuring an oracle's causal response to neutralizing the cited factor.


## 37. An LLM + Logic Framework to Explain MCTS (extends 2407.10820)

**Citation:** Z. An, X. Wang, H. Baier, et al., "[Combining LLMs with Logic-Based Framework to Explain MCTS](https://arxiv.org/abs/2505.00610)," arXiv:2505.00610, 2025 (extended abstract, AAMAS-25).  
**Link:** <https://arxiv.org/abs/2505.00610> · **Tier:** SUPPORTING

**Summary.** An et al. propose a Computational Tree Logic (CTL)-guided, LLM-based natural-language explanation framework for Monte Carlo Tree Search over an MDP. The pipeline has four stages: an LLM classifies a user query (26 predefined post-hoc types vs. background-knowledge queries); few-shot prompting translates it into formal logic/variable statements typed as base-level (single node), derived (multi-node computation), or logic-comparison (CTL branch comparison); scorer functions and CTL model-checking extract numeric/boolean evidence directly from the search tree; and a QA LLM synthesizes query, evidence, and retrieved domain knowledge into an explanation. By grounding answers in tree statistics and environment dynamics, it enforces factual consistency. Evaluated on a paratransit-planning MDP with 620 manually authored queries (with reference evidence and paragraphs), it sharply improves FactCC and BERTScore over raw GPT-4/GPT-4o/Llama3.1 baselines (e.g., FactCC@1 67.9% vs 25.8% for Llama; 72.1% vs 42.3% for GPT). Accepted as an AAMAS-25 extended abstract; extends arXiv:2407.10820.

**Depends on.** Extends the authors' prior work arXiv:2407.10820; builds on MCTS/MDP planning, Computational Tree Logic model checking, LLM few-shot prompting and RAG, and uses FactCC and BERTScore as evaluation metrics.

**Results.** On a paratransit-planning MDP with 620 queries (x3 repetitions) and a ~3,000-word/34-chunk knowledge base, the framework substantially raised factual-consistency scores: FactCC@1/@3 of 67.88%/83.27% with Llama3.1 (vs 25.77%/34.62% baseline) and 72.12%/81.35% with GPT (vs 42.31%/51.15% baseline) — a 2.40x (Llama) and 1.59x (GPT) FactCC improvement, with even larger BERTScore gains.

**Weaknesses.** Evaluated on a single domain (paratransit planning) with a small, author-constructed 620-query set, raising generalization and self-authorship/reference-bias concerns. "Factual consistency" is measured only via automated proxies (FactCC, BERTScore), which correlate imperfectly with human-judged correctness and do not verify causal soundness of explanations. No explicit limitations section, no human evaluation, and no ablation isolating CTL model-checking from RAG/prompting contributions. The 26 query types and scorer functions appear hand-engineered, limiting open-ended applicability. As an extended abstract, methodological and experimental detail is compressed. It explains/grounds rather than tests whether cited factors are causally load-bearing.

**Relation to xai-chess.** It is an adjacent explanation-faithfulness effort for the same search algorithm family (MCTS), grounding explanations in verifiable tree/MDP evidence, but it measures textual factual consistency rather than ChessFaith's intervention-based causal load-bearingness of the cited factor.


## 38. Explainable MCTS-minimax Hybrids via Process Mining

**Citation:** Y. Qian, T. Miller, Z. Qian, and L. Zhao, "[Exploring Explainable Multi-agent MCTS-minimax Hybrids in Board Game Using Process Mining](https://arxiv.org/abs/2503.23326)," arXiv:2503.23326, 2025.  
**Link:** <https://arxiv.org/abs/2503.23326> · **Tier:** SUPPORTING

**Summary.** This paper (full title "Exploring Explainable Multi-agent MCTS-minimax Hybrids in Board Game Using Process Mining"; the prompt's short title and 2025 date match) tackles the opacity of Monte-Carlo Tree Search, whose large, highly selective search trees are hard to interpret and prone to missing tactical traps. The authors integrate shallow minimax search into the rollout phase of multi-player MCTS to strengthen tactical play, then apply process mining to extract and visualize the agents' decision-making strategies as process models over event logs of agent behavior. The testbed is multi-agent 3v3 checkers. The contribution is methodological: showing that process-mining workflows can render game-playing agent strategies legible, moving beyond black-box analysis. It is a workshop-scale paper (AAAI 2025 PRL). Note: the fetched abstract did not expose specific process-discovery algorithms, win-rate tables, or a human evaluation of explanation quality, so claims about quantitative gains and explanation faithfulness could not be verified from the source.

**Depends on.** Builds on classic MCTS and MCTS-minimax hybrid game-playing literature (Baier and Winands' minimax-in-rollout hybrids), multi-agent/multi-player MCTS, and the process-mining tradition (event-log-based process discovery, e.g., alpha/inductive miners) for behavioral explanation.

**Results.** Reported at the abstract/concept level only: minimax-augmented rollouts are proposed to counter MCTS's selective-tree blind spots and tactical traps, and process mining is shown to extract/visualize agent strategies in 3v3 checkers. The fetched source did not surface concrete win-rate numbers, playing-strength comparisons against baseline MCTS, or quantitative explanation-quality metrics.

**Weaknesses.** Explainability appears to be post-hoc visualization of discovered process models, with no evident causal test that the surfaced "strategy" actually drives the agent's move choices, and seemingly no human-subject evaluation of whether the explanations aid understanding. Scope is narrow (a single toy domain, 3v3 checkers) and the venue is a workshop, limiting the strength of evidence. From the available source, quantitative playing-strength and faithfulness results could not be confirmed, so generalization and rigor are uncertain. Process mining describes aggregate behavioral patterns rather than per-decision rationale, which is what move-level explanation benchmarks require.

**Relation to xai-chess.** It is a same-domain (board-game agent) explainability effort that, unlike ChessFaith's causal-intervention test of whether a cited factor is load-bearing, only descriptively visualizes agent strategy without verifiable decision-level causal ground truth, illustrating the gap the project fills.


## 39. Contrastive Sparse Autoencoders for Chess Planning

**Citation:** Y. Poupart, "[Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents](https://arxiv.org/abs/2406.04028)," in Workshop on Interpretable Policies in Reinforcement Learning (InterpPol) @ RLC, arXiv:2406.04028, 2024.  
**Link:** <https://arxiv.org/abs/2406.04028> · **Tier:** SUPPORTING

**Summary.** Poupart introduces Contrastive Sparse Autoencoders (CSAE), a mechanistic-interpretability method for the chess engine Leela Chess Zero (the open-source AlphaZero analogue, a CNN policy/value heuristic plus MCTS). Rather than interpreting single hidden states, CSAE operates on pairs of game trajectories: it concatenates root-position activations with later-state activations and trains a sparse dictionary that splits learned features into "common" features (encoding the root board s0) and "differentiating" features (isolating planning concepts that distinguish optimal from suboptimal rollouts). A contrastive loss penalizes differences in common features while minimizing element-wise products of differentiating features. The author builds an automated taxonomy via agglomerative clustering of features (dendrograms) and adds sanity checks (dead-feature rates, entropy, linear probes, cluster correlation) to catch spurious correlations. Qualitatively, interpretable features emerge (piece safety/protection, rook threats, strategic positioning). Datasets (20M TCEC positions; 200k/20k trajectory pairs) were released on Hugging Face. The work is exploratory, framed as a proof-of-concept for interpreting multi-step planning.

**Depends on.** Builds on sparse autoencoder dictionary-learning interpretability (Anthropic/Cunningham-style SAEs), contrastive representation learning, linear-probing of internal activations, and the Leela Chess Zero / AlphaZero line (including prior work showing human chess concepts are linearly decodable from AlphaZero, e.g. McGrath et al.).

**Results.** Trained SAE: dictionary size 2,048 with only ~73 active features; reconstruction R2 ~0.81; ~71% dimensionality reduction; linear-probe F1 of 0.578 on differentiating features distinguishing optimal vs. suboptimal trajectories; cross-partition (common vs. differentiating) cluster correlation averaging ~0.1, supporting the claimed disentanglement. Findings are predominantly qualitative — named, human-recognizable concepts surfaced — rather than a strong quantitative benchmark.

**Weaknesses.** Largely qualitative and proof-of-concept; the headline differentiating-feature probe F1 of 0.578 is barely above chance, undercutting claims that planning concepts are cleanly isolated. Only ~73 of 2,048 features are active, raising questions about wasted capacity and feature completeness. No causal intervention: features are correlationally extracted/probed, never ablated to test whether they actually drive Leela's move choice — exactly the gap ChessFaith targets. A "blinking" board-encoding bias (white/black flip) contaminates features. Author concedes automated taxonomy adds another black-box layer, contrastive heatmaps are underexploited, suboptimal-trajectory sampling injects unquantified inductive bias, and there is no cross-architecture generalization. Single model, single author workshop paper; limited statistical rigor.

**Relation to xai-chess.** It is a representation-level (correlational) interpretability method for a planning chess agent, complementary to and motivating ChessFaith's shift to decision-level causal-intervention faithfulness testing, since CSAE extracts features but never ablates them to verify they are load-bearing.


## 40. Information-based Explanations for Chess Models

**Citation:** P. Hammersborg and I. Strümke, "[Information based explanation methods for deep learning agents -- with applications on large open-source chess models](https://arxiv.org/abs/2309.09702)," arXiv:2309.09702, 2023.  
**Link:** <https://arxiv.org/abs/2309.09702> · **Tier:** SUPPORTING

**Summary.** Hammersborg and Strümke build a fully open-source pipeline for explaining neural chess agents, addressing the inaccessibility of AlphaZero. They distill the open-source Leela Chess Zero policy ((8,8,21) input, 20 residual blocks) into smaller students, then introduce an "Information Importance map" (II-map): a trainable masker network R(s) that emits per-input-element probabilities, binarized via a Heaviside/uniform-sampling trick into stochastic masks applied BEFORE the main model. An L1 sparsity penalty forces the mask to retain only the information the model actually needs, yielding explanations the authors argue are "exhaustive and exclusive" — a guaranteed complete enumeration of used information, suited to discrete chess inputs where SHAP/GradCAM struggle. They visualize importance per occupied/empty square on Lichess puzzles and famous games. Separately, they replicate McGrath et al.'s concept-probing (logistic probes on layer activations for concepts like in_check, material_advantage, mate threats), finding "striking similarity" to AlphaZero's layer-wise concept acquisition, all on open models.

**Depends on.** Leela Chess Zero (open AlphaZero reimplementation); McGrath et al. (2022) concept-probing of AlphaZero; model distillation; SHAP/GradCAM as contrasted baselines; logistic linear probing.

**Results.** Reproduced AlphaZero-style concept-detection results on open models: networks learn in_check first, then threat representations; pawn-related concepts emerge in earlier layers; reported as guessing-corrected binary probe accuracy across residual blocks over 12 training iterations. The II-map produces sparse per-square importance maps on Lichess puzzles and historical games (Game of the Century, Kasparov–Deep Blue, 2021 WC). Distillation avoided the ~12M-game training cost. No engine win-probability or causal-faithfulness metric is reported.

**Weaknesses.** II-map "faithfulness" is asserted from architecture (pre-input masking) rather than measured against an external causal oracle; authors concede the masked model "learns somewhat adversarial representations," meaning the masker may obfuscate/route around information rather than purely remove it, undermining the exhaustive/exclusive guarantee. Training is unstable and adds per-model overhead (masker co-trained with the network); a regularization-vs-performance trade-off governs sparsity, so maps are tunable rather than canonical. Evaluation is qualitative (visual maps, anecdotal games) with no quantitative explanation-quality metric, no human/ground-truth validation, and no comparison to other XAI methods on a shared score. Explanations attribute to squares/pieces, not to human concepts (pin, fork), so they don't directly verify factor-level claims. Concept-probing only shows correlation (representation present), not that the concept is decision-relevant.

**Relation to xai-chess.** It is a chess-specific XAI precedent whose square-level masking and concept-probing identify what a model represents, but lacks the decision-level causal-intervention ground truth on a strong oracle that ChessFaith adds to grade whether a cited factor is actually load-bearing.


## 41. An RL Chess Environment for Detecting Human-understandable Concepts

**Citation:** P. Hammersborg and I. Strümke, "[Reinforcement Learning in an Adaptable Chess Environment for Detecting Human-understandable Concepts](https://arxiv.org/abs/2211.05500)," arXiv:2211.05500, 2022.  
**Link:** <https://arxiv.org/abs/2211.05500> · **Tier:** SUPPORTING

**Summary.** Hammersborg and Strümke build a lightweight, adaptable chess RL environment (Silverman 4x5 and Los Alamos 6x6 variants, bitboard movegen) to study whether self-trained AlphaZero-style agents develop human-understandable concepts. They train two CNN policy/value networks (a plain 32-filter net for 4x5, a 64-filter ResNet for 6x6) via self-play plus MCTS, then apply linear concept probes following McGrath et al. (2021): logistic regressors fit on each layer's activations against binary concept labels (L1-regularized, accuracy corrected for chance), using balanced 250k/250k datasets. They probe four programmatically-definable concepts — material_advantage, has_mate_threat, threat_opp_queen, in_check — and track how strongly each is linearly decodable across training iterations. Concepts emerge in an interpretable order (material first, then queen threat, then mate threat), and the in_check concept's strength depends on board geometry and on residual connections (earlier/stronger emergence in the 6x6 ResNet). The work emphasizes resource-frugal, fully open-source reproducibility.

**Depends on.** McGrath et al. (2021) AlphaZero concept-probing methodology; AlphaZero/MuZero self-play + MCTS; the TCAV/linear-probe interpretability lineage; small chess variants (Silverman 4x5, Los Alamos 6x6).

**Results.** Linear probes recover concepts in a developmentally ordered sequence on the 4x5 agent: material_advantage decodable by ~iteration 5, threat_opp_queen by ~30, has_mate_threat by ~50, all plateauing by iteration 100; in_check stays weak. On the 6x6 ResNet agent, material and mate threats are decodable from the start (attributed to residual connections), and in_check develops progressively and more strongly, localized to first and last layers. Findings: agents internalize human-aligned concepts, material serves as an early win proxy, and architecture/board geometry shape when and how strongly concepts appear.

**Weaknesses.** Purely observational/correlational: linear probes show a concept is linearly decodable from activations but do NOT establish that the agent uses it causally for move selection — the authors explicitly note they cannot determine how detected concepts influence decisions. Only linearly separable, programmatically-definable concepts are tested (4 simple ones); non-linear or abstract strategic concepts are invisible. Experiments use tiny non-standard board variants (4x5, 6x6) and self-play-generated concept datasets that may not reflect real game distributions, limiting external validity. No standard 8x8 chess, no human-strength baseline, and small concept set.

**Relation to xai-chess.** It is the methodological foil for ChessFaith: where Hammersborg et al. only show concepts are linearly decodable from a chess agent's activations (representation-level, correlational), ChessFaith supplies the missing causal, decision-level test of whether a cited concept is actually load-bearing via intervention on a strong oracle.


## 42. Out-of-Distribution Tests Reveal Compositionality in Chess Transformers

**Citation:** A. Mészáros, P. Reizinger, and F. Huszár, "[Out-of-Distribution Tests Reveal Compositionality in Chess Transformers](https://arxiv.org/abs/2510.20783)," arXiv:2510.20783, 2025.  
**Link:** <https://arxiv.org/abs/2510.20783> · **Tier:** SUPPORTING

**Summary.** Mészáros, Reizinger, and Huszár train a 270M-parameter decoder-only chess Transformer (16 layers, 1024-dim) by behavior cloning on ~525M ChessBench positions labeled by Stockfish 16, then probe it with seven out-of-distribution test sets that deliberately violate the training distribution: illegal piece counts, same-color bishops, supernumerary pieces, Chess960/all-random starts, extreme Knights&Rooks configurations, and Horde games. They separate two competencies: rule extrapolation (legal-move selection) and strategy adaptation (matching Stockfish top-k). Rule-following transfers remarkably well (99.6% legal on OOD puzzles, 90.2% on extreme configs, ~99% in Chess960 play), while strategic quality degrades sharply OOD (e.g., top-1 drops to ~22-30%; Horde relative Elo -350). Training-dynamics analysis shows the model first learns the modular constraint "move only your own pieces," which the authors read as emergent compositional structure. They conclude the network captures the game's syntactic rules far more robustly than its semantic strategy.

**Depends on.** Builds on ChessBench / the searchless "grandmaster-level chess without search" line (270M Transformer trained on Stockfish 16 labels); Decision/behavior-cloning Transformers; systematic generalization and compositionality literature; Chess960 and Horde chess variants; Lichess evaluation.

**Results.** Legal-move accuracy: 100% in-distribution, 99.6% OOD puzzles, 90.2% Knights&Rooks, 99.36% Chess960 / 95.96% Horde full games. Stockfish top-k match: OOD puzzles 67.7% top-1 / 89.04% top-5; More-pieces 30.49% top-1; Chess960 starts 22.73% top-1 / 88.8% top-10. Relative Elo vs Stockfish levels 0-4: +88 standard, -110 Chess960, -350 Horde. Lichess: 1550 standard, 1571 Chess960, 1178 Horde. Model fails threefold-repetition tracking (no move history).

**Weaknesses.** Compositionality is argued behaviorally, not mechanistically: high OOD legal-move rates and a single training-dynamics observation ("move only your own pieces") are suggestive but not direct evidence of compositional internal representations (no circuits/probing causal analysis). The OOD sets conflate rule novelty with simple frequency shifts, so "extrapolation" may partly reflect coverage in the massive training corpus. Lichess ratings are uncontrolled (opponent pool, time control, small samples; ±45-68). Strategy metric (Stockfish top-k) penalizes equally-good moves and is engine-relative. Single model size/seed; no ablations isolating which architectural or data factors drive rule-following. Horde's collapse is underdiagnosed.

**Relation to xai-chess.** It supplies independent evidence that chess Transformers internalize structured, rule-like (plausibly compositional) move-legality knowledge, which underwrites ChessFaith's premise that an oracle engine's behavior is a meaningful causal target whose response to rule-level interventions (e.g., suspending a pin) reflects genuine structural reasoning rather than surface memorization.


## 43. Human-AI Conceptual Alignment via Chess

**Citation:** S. Lomasov, J. Goldfeder, M. H. Erol, et al., "[Exploring Human-AI Conceptual Alignment through the Prism of Chess](https://arxiv.org/abs/2510.26025)," arXiv:2510.26025, 2025.  
**Link:** <https://arxiv.org/abs/2510.26025> · **Tier:** SUPPORTING

**Summary.** Note: the fetched title is "Exploring Human-AI Conceptual Alignment through the Prism of Chess" (the task's title "Human-AI Conceptual Alignment via Chess" is a paraphrase); authors and arXiv ID match, so resolved=true. The paper probes whether a 270M-parameter grandmaster-level chess transformer genuinely understands human strategic concepts or merely mimics surface patterns. Using layer-wise probing for human concepts (center control, knight outposts, etc.), it finds early layers encode these concepts with up to 85% accuracy, while deeper layers, despite driving stronger play, drift toward "alien" representations, dropping to 50-65%. To test robustness beyond opening memorization, the authors introduce the first Chess960 (randomized-start) concept dataset: 240 expert-annotated positions across 6 strategic concepts. Removing opening theory drops concept recognition 10-20% across all probing methods, indicating reliance on memorized patterns rather than abstract understanding. The core thesis: optimizing chess engines for performance pushes their internal representations away from human-aligned concepts, posing a challenge for collaborative/creative AI.

**Depends on.** Builds on linear/concept probing of internal representations (probing classifiers, concept activation), the DeepMind "grandmaster-level chess without search" 270M transformer line, and prior chess-as-interpretability-testbed work (Othello/chess world-model probing); introduces a new Chess960 expert-annotated concept benchmark.

**Results.** Early transformer layers encode human concepts at up to 85% probe accuracy; deeper, higher-performing layers fall to 50-65%, evidencing representational drift toward non-human ("alien") encodings. On the new 240-position, 6-concept Chess960 dataset, eliminating opening theory via randomized starts reduces concept-recognition accuracy by 10-20% across all probing methods, exposing dependence on memorized patterns over abstract understanding. Performance and human-concept alignment are shown to be in tension across depth.

**Weaknesses.** Concept alignment is measured by representational probing (correlational), not by causal intervention on the concept and observed behavior change, so probe accuracy may over/under-state whether a concept is actually used in decisions (probes can read off decodable-but-unused features). Findings are tied to one model family (270M DeepMind-style transformer) and 6 hand-picked concepts; generalization to other engines/concepts is untested. The 240-position Chess960 set is small and single-source-annotated, raising label-reliability and statistical-power concerns. "Alien" is inferred from low probe accuracy, which conflates "different concept" with "no probeable linear structure." No move-level explanation faithfulness is evaluated.

**Relation to xai-chess.** It motivates xai-chess by showing chess models' internal concepts diverge from human ones via correlational probing, whereas ChessFaith supplies the missing causal-intervention test of whether a cited concept is actually load-bearing for a move decision.


## 44. BUMP: A Benchmark of Unfaithful Minimal Pairs for Meta-Evaluation of Faithfulness Metrics

**Citation:** L. Ma, S. Cao, R. L. Logan IV, et al., "[BUMP: A Benchmark of Unfaithful Minimal Pairs for Meta-Evaluation of Faithfulness Metrics](https://arxiv.org/abs/2212.09955)," in Proc. 61st Annu. Meeting Assoc. Comput. Linguistics (ACL), 2023, arXiv:2212.09955.  
**Link:** <https://arxiv.org/abs/2212.09955> · **Tier:** SUPPORTING

**Summary.** BUMP is a benchmark for meta-evaluating automatic faithfulness metrics in abstractive summarization. Starting from 200 article-summary pairs from CNN/DailyMail, human annotators (Amazon Mechanical Turk, with manual validation correcting ~16% of edits) introduce a single, controlled error into a faithful reference summary to produce a minimally different unfaithful counterpart, yielding ~889 such minimal pairs. Errors follow a hierarchical taxonomy (predicate, entity, circumstance, coreference; intrinsic vs. extrinsic). The authors evaluate 12 metrics (BLEU, ROUGE-2, BERTScore, BLEURT, BARTScore, FactCC, DAE, SummaC, QuestEval, Q², QAFactEval, CoCo) along two axes: consistency (fraction of unfaithful summaries scored below their faithful partner within a pair) and discriminability (ROC AUC). Key finding: the most discriminative metrics (QAFactEval, ~71.5% AUC) are not the most consistent (BARTScore 91.9%, CoCo 90.8%, DAE 87.9%), and BUMP's human-written minimal pairs are substantially harder than model-generated benchmarks (AUC 50-70% vs. 70-84% on QAGS-C).

**Depends on.** CNN/DailyMail summarization dataset; prior faithfulness metrics it evaluates (FactCC, DAE, SummaC, QuestEval, Q², QAFactEval, BARTScore, BLEURT, etc.); prior meta-evaluation benchmarks of model-generated errors (FRANK, QAGS, SummEval) that it contrasts against.

**Results.** ~889 human-written minimal pairs across a 7-way error taxonomy. Consistency leaders: BARTScore 91.9%, CoCo 90.8%, DAE 87.9%. Discriminability leader: QAFactEval 71.5% overall ROC AUC. Central result: high consistency does not imply high discriminability (and vice versa). BUMP is harder than synthetic/model-generated benchmarks (AUC 50-70% vs 70-84% on QAGS-C). The minimal-pair design uniquely enables per-error-type consistency measurement, infeasible on benchmarks with many co-occurring errors per summary.

**Weaknesses.** Narrow domain: single-document news summarization on CNN/DailyMail only, so generalization to other genres/tasks is untested. Errors are deliberately human-injected single edits into clean references, not naturally occurring model hallucinations, possibly mismatching the error distribution metrics face in deployment. Authors note the method cannot assess consistency when references already contain many errors, and some error types (especially in the freestyle Task 2) have too few samples for reliable statistical comparison. Crowdsourced annotation required ~16% manual correction, raising label-quality concerns. Consistency is a within-pair relative-ranking measure that does not test calibration or absolute score thresholds. No causal-intervention notion of faithfulness; faithfulness is defined purely as factual consistency with source text.

**Relation to xai-chess.** BUMP supplies the matched-minimal-pair, multi-metric meta-evaluation recipe that ChessFaith adapts, but where BUMP injects textual errors and lacks decision-level causal ground truth, ChessFaith neutralizes an explanation's named factor via board-level and rule-level interventions to obtain verifiable causal labels.


## 45. Transformers Use Causal World Models in Maze-Solving

**Citation:** A. F. Spies, W. Edwards, M. I. Ivanitskiy, et al., "[Transformers Use Causal World Models in Maze-Solving Tasks](https://arxiv.org/abs/2412.11867)," arXiv:2412.11867, 2024.  
**Link:** <https://arxiv.org/abs/2412.11867> · **Tier:** SUPPORTING

**Summary.** Spies et al. train small (~19M-param, 6-layer) transformers on autoregressive maze-solving, where mazes are tokenized adjacency lists over 5x5/6x6 lattices, and ask whether the models build an internal "world model" of maze connectivity that is causally used. Using sparse autoencoders (SAEs) plus circuit/attention analysis, they isolate SAE latents encoding individual maze connections (validated unsupervised via decision trees), finding learned-positional models ("Stan") use a compositional two-feature code per edge while rotary models ("Terry") use single features. Crucially, they go beyond probing by intervening: patching modified SAE activations back into the residual stream (after layer 0) at semicolon positions, either activating a connection feature to its max or zeroing it, then measuring whether the solution path changes accordingly. They report a clear asymmetry: activating features changes behavior far more reliably (~35-70% success for Terry) than removing them. This demonstrates the representations are causally load-bearing for decisions, not merely correlational.

**Depends on.** Mechanistic-interpretability toolkit: sparse autoencoders for feature disentanglement, activation/residual-stream patching and ablation (causal mediation/activation patching tradition), circuit analysis; maze-transformer "world model" probing line (Ivanitskiy et al. maze tokenization).

**Results.** Stan (learned PE, 8 heads) 96.6% accuracy; Terry (rotary, 4 heads) 94.3%. SAE reconstruction without perturbation preserves behavior (L2 errors ~8-10 matching baseline), confirming feature completeness. Targeted feature toggling causally edits solution paths: activation interventions succeed ~35-70% (Terry) vs ~35% (Stan), and are systematically more effective than removal/zeroing. Models generalize to more simultaneously active connection features than seen in training; head-ablation maps SAE features to specific attention heads showing spatial partitioning.

**Weaknesses.** Tiny synthetic domain (acyclic unique-solution lattice mazes) and small bespoke models limit external validity to chess engines or LLMs. The headline causal effect is modest and asymmetric: activation interventions only ~35-70% successful and removal/suppression often fails, so "neutralize the factor" interventions (which the project relies on) are exactly the weak direction here, a cautionary signal. Success metric depends on SAE quality and choice of intervention site (semicolon, post-layer-0); reconstruction error is nonzero, so off-target effects are possible. No held-out causal randomization/null baseline reported in the abstract-level material. Numbers read from figures are approximate.

**Relation to xai-chess.** It is a methodological precedent showing that residual-stream feature interventions can establish a representation is causally load-bearing for a model's spatial decisions, directly analogous to ChessFaith's causal-intervention test of whether a cited factor actually drives an engine's move evaluation, while also warning that suppression-style interventions are the harder, less reliable direction.


## 46. Explainable Reinforcement Learning Agents Using World Models

**Citation:** M. Singh, A. Alabdulkarim, G. Mansi, and M. O. Riedl, "[Explainable Reinforcement Learning Agents Using World Models](https://arxiv.org/abs/2505.08073)," in Workshop on Explainable AI (XAI) at IJCAI, arXiv:2505.08073, 2025.  
**Link:** <https://arxiv.org/abs/2505.08073> · **Tier:** SUPPORTING

**Summary.** This paper explains the sequential decisions of model-based deep RL agents by leveraging World Models to generate counterfactual explanations. Its core novelty is a Reverse World Model (RWM): whereas a standard Forward World Model predicts Pr(s_{t+1}|s_t,a_t), the RWM predicts Pr(s_t|s_{t+1},a_t), i.e. what the state of the world should have been for the agent to prefer a given counterfactual action. Training reverses the temporal order of replay-buffer data and strips reward signals, letting the system synthesize prior states that would have motivated an alternative action. Rather than just showing alternative actions, it shows the environmental conditions that justify them. Evaluated in a modified Crafter (2D Minecraft-like) coffee-recipe domain with deliberately non-commonsense ingredients, a Prolific user study (n=70; control 33, treatment 37) found treatment users identified the cause of agent failure far more accurately (64.86% vs 26.52%, p<0.00001), with higher satisfaction and trust and lower cognitive load.

**Depends on.** Model-based deep RL with learned World Models (e.g. Dreamer-style forward dynamics models); counterfactual explanation in XAI; the Crafter benchmark environment; human-subjects evaluation of explanations.

**Results.** User study (Prolific, n=70, ages 19-73). Failure-cause identification accuracy: treatment 64.86% vs control 26.52% (p<0.00001; random baseline 1/16). Satisfaction higher for treatment (p~0.0036), trust higher (p~0.034), cognitive load lower (p~9.76e-5). No significant difference in completion time (p<0.35). Demonstrates that showing "what the world should have been" significantly improves user understanding of the agent's policy.

**Weaknesses.** Explanations are evaluated only by human-judged understanding/trust, not by any objective causal-faithfulness metric, so there is no guarantee the RWM-generated states reflect the agent's actual decision mechanism rather than a plausible-looking artifact. Authors admit they deliberately overfit the policy to force failures, undermining ecological validity. The RWM only produces states within the FWM/agent training distribution, weakening exactly the counterfactual (out-of-distribution) cases where explanation matters most, and World Models can catastrophically forget policy-irrelevant dynamics. Single toy domain (one Crafter variant) and a modest, single-task user study limit generalization. No comparison to other XRL/counterfactual baselines beyond the control snapshot condition.

**Relation to xai-chess.** It is a sequential-decision XAI method using generative world-model counterfactuals validated by human studies, contrasting with ChessFaith's approach of grading explanations by objective oracle win-probability shifts under causal interventions rather than by human-perceived understanding.


## 47. Explore the Reasoning Capability of LLMs in the Chess Testbed (MATE)

**Citation:** S. Wang, L. Ji, R. Wang, et al., "[Explore the Reasoning Capability of LLMs in the Chess Testbed](https://arxiv.org/abs/2411.06655)," arXiv:2411.06655, 2024.  
**Link:** <https://arxiv.org/abs/2411.06655> · **Tier:** SUPPORTING

**Summary.** Wang et al. argue LLMs falter on long-horizon reasoning like chess and that explicit language explanations can help. They build MATE ("Move on strAtegy and Tactics datasEt"), ~1M chess positions drawn from Lichess where Stockfish-generated move sequences yield candidate-move pairs separated by an engine-score threshold. Each candidate is annotated by chess experts with strategy labels (five categories: material, piece activity, pawn structure, space, king safety, each with ~20 linguistic expressions) and tactic labels (eight types: pin, fork, battery, x-ray, discovered attack, windmill, Greek gift, double attack). They fine-tune LLaMA-3-8B on dataset variants to pick the better of two candidate moves. With combined strategy+tactic explanations the model reaches 95.2% selection accuracy zero-shot, far above GPT-4 (60.0%), Claude-3.5-sonnet (54.9%), and Gemini-1.5-pro (52.6%). The core claim: language explanations grounding strategic/tactical concepts measurably improve LLM reasoning. Note: source confirms title/authors; project memory had no prior notes on this paper.

**Depends on.** Stockfish (engine-derived move sequences and scoring thresholds), Lichess game data, LLaMA-3-8B as the fine-tuned base, and a taxonomy of human chess strategy/tactic concepts; compares against GPT-4, Claude-3.5-sonnet, Gemini-1.5-pro.

**Results.** Zero-shot binary move-selection accuracy: fine-tuned LLaMA-3-8B with strategy+tactic explanations 95.2% vs GPT-4 60.0%, Claude-3.5-sonnet 54.9%, Gemini-1.5-pro 52.6% (a 24.2-point gain over the best commercial model). Appendix: fine-tuned model generates strategy explanations at 74.7% accuracy vs GPT-4o 51.0%. Authors conclude language explanations enhance reasoning.

**Weaknesses.** The 95.2% reflects fine-tuning on in-distribution expert annotations versus zero-shot baselines, so the headline gap conflates task-specialization with reasoning gains rather than isolating the explanation's causal contribution. "Better move" ground truth is defined by Stockfish score gaps, so the model partly learns to mimic the engine, not human strategic understanding. No verification that the verbalized strategy/tactic explanation is actually USED by the model to decide (no faithfulness/intervention test) — explanations could be post-hoc rationalizations. Binary move-selection is an easier proxy than full-game play; evaluation on puzzles, not games. Single base model and size; expert-annotation bias; chess-only generalization untested. Concept taxonomy (8 tactics, 5 strategy categories) is coarse.

**Relation to xai-chess.** MATE supplies a large concept-annotated chess corpus (pins, forks, etc.) and shows explanations correlate with better move choice, but it never causally verifies that the cited factor is load-bearing — precisely the gap ChessFaith fills via intervention-based faithfulness grading.


## 48. Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess (Maia4All)

**Citation:** Z. Tang, D. Jiao, E. Xue, et al., "[Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess](https://arxiv.org/abs/2507.21488)," arXiv:2507.21488, 2025.  
**Link:** <https://arxiv.org/abs/2507.21488> · **Tier:** SUPPORTING

**Summary.** Maia4All tackles individual chess-player behavior modeling under data scarcity, in the Maia human-like-AI lineage. It proposes a two-stage method on top of Maia-2 (ResNet position encoder + skill-aware transformer + policy head). The enrichment stage fine-tunes the population model on a diverse, skill-balanced set of "prototype" players with rich histories, extending the 11 population skill embeddings to per-player embeddings. The democratization stage freezes universal parameters and learns only an individual embedding for unseen players, initialized via a transformer-based Prototype Matching Network that first solves a discriminative prototype-classification task. On 2023 Lichess blitz data (11 rating bins), Maia4All-Prototype reaches ~53.2% top-1 move-matching accuracy from just 800 positions (~20 games) versus 51.46% for Maia-2, a claimed 250x data-efficiency gain over Maia-Individual's 5,000-game requirement. A case study transfers the recipe to "idiosyncratic LLMs" (LLaMA 3.1 8B, Gutenberg author prototypes, ModernBERT matcher) showing language-modeling-loss gains.

**Depends on.** Builds directly on Maia-2 (unified skill-aware human move-prediction model) and the original Maia (McIlroy-Young et al.), the Lichess game corpus, and meta-learning/prototype-initialization ideas; LLM case study uses LLaMA 3.1 8B and ModernBERT.

**Results.** Top-1 move-matching: 53.22% at ~20 games (800 positions) and 53.81% at ~2,500 games vs 51.46% Maia-2 baseline, i.e. +1.9-2.5 pp, framed as ~half the Stockfish-vs-Maia amateur-matching gap. ~250x reduction in required data (20 vs 5,000 games). PMN/ModernBERT prototype classification 94.7% in the LLM study; consistent LM-loss improvements over 1K-3K token budgets.

**Weaknesses.** Gains are modest in absolute terms (1.9-2.5 pp top-1 accuracy) and reported on aggregate move-matching/perplexity, not on whether the model captures specific player concepts. The enrichment stage still needs prototype players with extensive histories, so the method is only "low-resource" for new users, not overall. Evaluation is purely predictive fidelity; no causal or mechanistic claim about why a player or model chooses a move. The LLM transfer relies on language-modeling loss as a proxy rather than any validated style-transfer metric, weakening the generalizability claim. No notion of explanation faithfulness or intervention is present.

**Relation to xai-chess.** It is domain-adjacent context, not methodological overlap: Maia4All advances human-like/individualized chess move prediction, whereas ChessFaith grades explanations by causal intervention, so Maia4All supplies behavioral models whose stated rationales could in principle be stress-tested by the benchmark.


## 49. Learning Models of Individual Behavior in Chess

**Citation:** R. McIlroy-Young, R. Wang, S. Sen, J. Kleinberg, and A. Anderson, "[Learning Models of Individual Behavior in Chess](https://arxiv.org/abs/2008.10086)," in Proc. 28th ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD), 2022. arXiv:2008.10086, 2020.  
**Link:** <https://arxiv.org/abs/2008.10086> · **Tier:** SUPPORTING

**Summary.** This paper (KDD 2022; arXiv 2020) extends the Maia line of work — open-source AlphaZero-style residual CNN policy models trained to predict human chess moves — from population-level to individual-level behavior. Starting from a base Maia model, the authors fine-tune ("Transfer Maia") on a single player's game history to forecast that specific player's move choices on Lichess blitz games. Personalization lifts move-matching accuracy roughly 4-5 points over the best non-personalized baselines (e.g. ~58% vs ~53% at 40,000 games), with gains across optimal moves, minor errors, and blunders. The personalized models also enable strong stylometry: identifying which of 400 players made a set of moves with up to ~98% accuracy from 100 games, and ~99% from blunders alone. Evaluation covers 400 players (~1,750 median Elo). The core claim is that move-prediction models capture individual, idiosyncratic, even error-prone decision patterns, not just generic strong play.

**Depends on.** Builds directly on Maia ("Aligning Superhuman AI with Human Behavior," McIlroy-Young et al. 2020), which itself adapts the AlphaZero/Leela residual-CNN policy architecture; uses Lichess game data (2013-2020) and transfer-learning fine-tuning.

**Results.** Transfer Maia beats best baseline Maia by ~4-5 pts move-matching (1k games 49.7% vs 52.7%; 5k 55.0% vs 53.2%; 10k 55.6% vs 52.8%; 40k 58.0% vs 52.8%). Stylometry over 400 players: 98% from 100 games (all moves), 86% from 10 games, ~99% from blunders alone; drops to 55% using only 30+ ply positions. Needs ~5,000+ games/player to be effective.

**Weaknesses.** Predicts moves but offers no explanation of why a move is chosen — it is a behavioral policy model, not an explanation or causal model, so it provides no decision-level causal ground truth. Requires ~5,000+ games per player; weak below that threshold. Restricted to blitz time controls and sub-master Elo (~1,750 median); generalization to grandmaster play unsolved. Accuracy ceiling (~58%) is modest. Stylometry raises privacy/deanonymization concerns not deeply addressed. Aggregate accuracy metrics give no insight into which board features drive predictions.

**Relation to xai-chess.** It supplies a strong human-aligned move-prediction oracle and validates that chess policy models capture individual decision patterns, but, lacking any causal or factor-level account of why a move is chosen, it motivates ChessFaith's intervention-based test for whether a cited factor is actually load-bearing.


## 50. Counterfactual Simulatability of LLM Explanations for Generation Tasks

**Citation:** M. Limpijankit, Y. Chen, M. Subbiah, N. Deas, and K. McKeown, "[Counterfactual Simulatability of LLM Explanations for Generation Tasks](https://arxiv.org/abs/2505.21740)," arXiv:2505.21740, 2025.  
**Link:** <https://arxiv.org/abs/2505.21740> · **Tier:** SUPPORTING

**Summary.** This paper extends counterfactual simulatability evaluation, originally defined for yes/no QA, to open-ended generation tasks. The core idea: a good explanation should let an observer predict how the model behaves on related counterfactual inputs. They use GPT-4 Turbo to decompose an explanation into atomic units, generate counterfactual inputs that include the units deemed important, and check whether the model's outputs on those counterfactuals actually contain the predicted units. Three metrics are defined: simulatability (all atomic units appear in the counterfactual), generality (1 minus mean pairwise cosine similarity among simulatable counterfactuals), and precision (fraction of explanation units appearing in the counterfactual output). They test Chain-of-Thought and post-hoc explanations from Claude 3.7 Sonnet, GPT-4, GPT-4 Turbo, and Llama 3.3 70B on news summarization (CNN/DM, skill-based) and medical suggestion (Taiwan e-Hospital, knowledge-based). Explanations help much more for summarization than for the knowledge-based medical task.

**Depends on.** Builds directly on Chen et al.'s counterfactual simulatability framework for yes/no QA; uses GPT-4 Turbo as an automated pipeline for explanation decomposition and counterfactual generation; CNN/DM and Taiwan e-Hospital datasets.

**Results.** Summarization yields high faithfulness/predictability: GPT-4 Turbo ~0.85 precision, ~0.50 generality (human eval); Claude 3.7 Sonnet 0.93 precision (automatic). Medical suggestion is far weaker: GPT-4 Turbo ~0.55 precision, ~0.23 generality; Claude 0.48 precision. Task differences are significant (p<0.05), but CoT vs post-hoc explanation differences within a task are not. Conclusion: counterfactual simulatability suits skill-based tasks better than knowledge-based ones.

**Weaknesses.** Ground truth is self-referential and entirely text-matching: it measures whether explanation "units" reappear in counterfactual outputs, with no external verification that the cited factor actually caused the behavior — circular versus a true causal probe. Heavy reliance on GPT-4 Turbo to both generate counterfactuals and judge unit presence injects model bias; generality via cosine similarity is admittedly suboptimal and atomic-unit matching is ambiguous under paraphrase. Only two tasks and a handful of models, so generalization is unestablished; no statistical power analysis on small human-eval samples. The metric conflates explanation faithfulness with task structure (knowledge tasks score low partly by construction).

**Relation to xai-chess.** It is a methodological cousin in the LLM-faithfulness lineage that ChessFaith borrows from — counterfactual-input simulatability for free-form generation — but it lacks the verifiable decision-level causal ground truth that the chess engine oracle and movegen-patched interventions provide.


## 51. Do LLM Self-Explanations Help Users Predict Model Behavior?

**Citation:** P. Hong and B. Roth, "[Do LLM Self-Explanations Help Users Predict Model Behavior? Evaluating Counterfactual Simulatability with Pragmatic Perturbations](https://arxiv.org/abs/2601.03775)," arXiv:2601.03775, 2026.  
**Link:** <https://arxiv.org/abs/2601.03775> · **Tier:** SUPPORTING

**Summary.** Hong and Roth ask whether an LLM's verbalized self-explanations (chain-of-thought or post-hoc) actually help a user predict how the model would answer counterfactual variants of a question — the "counterfactual simulatability" criterion. On StrategyQA yes/no items they build counterfactuals two ways: theory-driven "pragmatic perturbations" grounded in Grice's maxims (presupposition flip, WordNet lexical substitution, scalar quantifier adjustment, contextualization) and unconstrained LLM-generated counterfactuals (GPT-3.5/4, Llama-3.3-70B). A two-phase protocol measures prediction accuracy without vs. with explanations (Δ OverallAcc), using both LLM judges (weaker/stronger) and 28 humans on 50 items. Self-explanations consistently improve simulation accuracy, but gains vary by perturbation type and judge strength: weaker judges gain ~0.30, stronger ~0.15; humans rise modestly 0.614→0.634 while confidence jumps 3.21→4.38. Pragmatic perturbations give more stable, controlled tests than LLM-generated ones, and explanations reduce anchoring bias when the model was originally wrong.

**Depends on.** Counterfactual simulatability framework (Chen et al.); Gricean pragmatics/maxims; chain-of-thought and post-hoc explanation faithfulness literature; StrategyQA dataset; LLM-as-judge evaluation. Builds the human-/judge-simulation evaluation tradition that ChessFaith complements with engine-verified causal ground truth.

**Results.** Self-explanations consistently raise simulation accuracy. LLM judges: weaker judges improve more (mean Δ 0.302 CoT / 0.315 post-hoc) than stronger judges (0.149 / 0.156); post-hoc marginally beats CoT. Anchoring on the original answer drops when the model was initially wrong (Δ bias_orig_wrong from −0.017 to −0.304). Pragmatics-based perturbations (lexical substitution, presupposition flip, scalar adjustment) are more stable than LLM-generated counterfactuals. Human study (28 participants, 50 items): overall accuracy 0.614→0.634, largest gain from contextualization (+0.060); confidence 3.21→4.38; human–LLM-judge agreement ~0.60 in both phases. Qualitative rationales show explanations help most when they state the decision criteria the model applies.

**Weaknesses.** Improvement is small and possibly confounded with confidence inflation (confidence jumps far more than accuracy: 3.21→4.38 vs +0.02 human accuracy), so explanations may persuade more than inform. The human study is underpowered (28 people, 50 items). Simulatability measures whether explanations predict model OUTPUTS, not whether they reflect the model's actual causal computation — authors explicitly disclaim faithfulness to internals. The whole pipeline (counterfactuals AND explanations) is itself LLM-generated, with unquantified prompt/model sensitivity. StrategyQA binary questions are narrow; pragmatics taxonomy is admittedly incomplete; ceiling effects and underspecified counterfactual conditions limit signal. No verifiable external ground truth for whether a cited factor is genuinely load-bearing.

**Relation to xai-chess.** It exemplifies the simulatability/human-prediction school of explanation evaluation whose key gap — no verifiable causal ground truth for whether a cited factor is load-bearing — ChessFaith fills by intervening on the factor and measuring an oracle engine's win-probability change.


## 52. Tracking vs. Deciding in Searchless Chess Transformers

**Citation:** Q. Li and W. Jiang, "[Tracking vs. Deciding: The Dual-Capability Bottleneck in Searchless Chess Transformers](https://arxiv.org/abs/2603.29761)," arXiv:2603.29761, 2026.  
**Link:** <https://arxiv.org/abs/2603.29761> · **Tier:** SUPPORTING

**Summary.** Li and Jiang study searchless (next-token-prediction) chess transformers trained on UCI move sequences with no board input or search tree. They argue playing strength factors into two capabilities: state tracking T (reconstructing the board from move history; failures show as illegal moves) and decision quality Q (picking strong moves; measured by centipawn loss and Top-1 accuracy). They formalize a "dual-capability bottleneck" P <= min(T,Q): filtering to high-Elo games improves Q but catastrophically harms T (illegal rate +72%), so aggregate validation loss can improve while head-to-head play degrades. Their fix is Elo-weighted gradient reweighting (keeping all positions for diversity) plus scaling 28M->120M. They give a coverage-decay law t* = log(N/k_crit)/log(b) predicting late-game degeneration. Crucially, they validate tracking mechanistically with linear board-state probes over 315,606 positions, showing filtering hurts probe accuracy most on non-standard/endgame positions. Final 120M model reaches Lichess bullet ~2570 and beats Maia-2 at human move prediction (55.2% vs ~50%).

**Depends on.** Searchless chess transformers (DeepMind's "Grandmaster-Level Chess Without Search"); Maia/Maia-2 human-move-prediction models; linear-probing interpretability for board state (Othello-GPT / chess world-model probing lineage); standard decoder-only Transformer stack (RMSNorm, RoPE, SwiGLU, FlashAttention-2); Lichess game and puzzle data.

**Results.** Scaling 28M->120M cut illegal-move rate 5x (1.06%->0.22%) and lifted Top-1 46.5%->50.85% (head-to-head 80W-54L-66D, p=0.030). Linear Elo-weighting (r~20) gave the sweet spot: Top-1 51.2%, illegal 0.26%, beating uniform-weight 120M 41-24. Exponential weighting (r~200) raised loss and weakened play. Final V2.2 reached Lichess bullet ~2570 (253 games, 149W-74L-30D), beat V1.0 66-8-26 (p<0.001), and beat Maia-2 on human move prediction by +5.2pp overall (+9.1pp openings) with 7.4x blunder-alignment lift. Linear probes: 98.0% square accuracy at 120M vs 93.4% at 28M; filtering degraded probes most on non-standard (92.78%->91.45%) and endgame (92.08%->90.41%) positions. Degeneration onset delayed from median move 26 to 31.

**Weaknesses.** Chess-only; the claimed generality to Go/Shogi/Bridge is untested. Scaling rests on just two sizes (28M, 120M) and the weighting "sweet spot" on only three points (r=1,20,200), so the P(r) curve and scaling exponents are under-determined. The coverage-decay law assumes uniform branching and a homogeneous support threshold, which is unrealistic across opening/middlegame/endgame. The headline Maia-2 comparison is confounded: V2.2 ingests full move sequences while Maia-2 ingests FEN, so it conflates architecture, training, and input format. Evaluation leans on Lichess bullet Elo over modest game counts (253 games) with no longer time controls or controlled human studies. The SFT result (helps 120M, hurts 28M) is left mechanistically unexplained. Linear probes show decodability but not that the model causally uses the represented state for moves.

**Relation to xai-chess.** It supports the project by separating a chess model's board-state "tracking" from its move "deciding" and validating tracking with linear probes, motivating ChessFaith's argument that explanation faithfulness needs decision-level causal intervention rather than mere probe decodability.


## 53. Understanding Skill Adaptation in Transformers Using SAEs: Chess as a Model System (Maia-2 SAE, OpenReview)

**Citation:** D. Jiao, G. Eilender, Z. Tang, and A. Anderson, "[Understanding Skill Adaptation in Transformers Using Sparse Autoencoders: Chess as a Model System](https://openreview.net/forum?id=Wxl0JMgDoU)," OpenReview preprint (ICLR 2025 submission, id Wxl0JMgDoU), 2025.  
**Link:** <https://openreview.net/forum?id=Wxl0JMgDoU> · **Tier:** SUPPORTING

**Summary.** Jiao, Eilender, Tang, and Anderson train sparse autoencoders (SAEs) on the internal representations of Maia-2, a single skill-aware transformer that simulates human chess play across rating levels. The goal is mechanistic: to understand how expertise/skill information is encoded and how it shapes decision-making. They decompose the modulated (skill-conditioned) activations into sparse, interpretable latent features, then run activation interventions on those features. By steering identified latents they elicit both higher- and lower-skill play in specific positions, and via "mediated intervention" on targeted SAE features they selectively enhance or sabotage the model's handling of context-specific chess tasks. The contribution is an interpretability case study showing SAE features can expose how skill-specific information lives inside a human-imitation model and that those features are causally manipulable. The fetched OpenReview page confirms the title and authors but did not expose the final decision; per project notes this was an ICLR 2025 submission that was not accepted, so it should be cited as an OpenReview preprint rather than an ICLR 2025 paper.

**Depends on.** Maia-2 (skill-aware human-chess transformer); sparse autoencoder interpretability methods (Anthropic/OpenAI dictionary-learning lineage); activation steering / causal intervention on internal features; the Maia human-imitation chess modeling line.

**Results.** SAE features trained on Maia-2's modulated representations are interpretable and tied to skill encoding. Activation interventions on these latents elicit both higher- and lower-skill play in specific contexts; mediated interventions on targeted features enhance or sabotage performance on context-specific chess tasks, demonstrating causal manipulability of skill-related directions. The fetched page did not surface specific numeric metrics (e.g., move-agreement deltas, win-rate shifts) or per-task figures; quantitative tables are in the full PDF and not verified here.

**Weaknesses.** Quantitative results could not be verified from the abstract page; magnitudes of the enhance/sabotage and skill-modulation effects are unconfirmed. The work is a single-model interpretability study (Maia-2 only) with no external validity to engines like lc0/Leela or Stockfish. CRITICAL net-binding caveat for our project: SAEs trained on Maia-2's activations are bound to that specific network's representation space and cannot be transplanted or ablated inside a different engine (e.g., lc0), so its "feature ablation" is not a portable causal probe. Apparent decision status is rejected/unaccepted (cite as OpenReview submission, not ICLR 2025), implying it did not clear peer review. SAE features are correlational decompositions; faithfulness of a steered feature to a human-meaningful concept (e.g., a pin) is asserted via interpretation, not independently grounded against rules.

**Relation to xai-chess.** It is a same-domain (chess) interior-mechanism interpretability counterpoint to ChessFaith: it intervenes on learned SAE features inside one network to change behavior, whereas our benchmark intervenes on rule-level board causes and reads a model-agnostic oracle, sidestepping the net-binding limitation that confines this paper's ablations to Maia-2.


## 54. Caissa: A Neuro-Symbolic Chess Agent for Explainable Move Suggestion (Springer KI 2025)

**Citation:** M. Soliman and N. Ehab, "[Caïssa AI: A Neuro-Symbolic Chess Agent for Explainable Move Suggestion and Grounded Commentary](https://doi.org/10.1007/978-3-032-02813-6_11)," in KI 2025: Advances in Artificial Intelligence (LNAI), Springer, 2025, doi:10.1007/978-3-032-02813-6_11.  
**Link:** <https://doi.org/10.1007/978-3-032-02813-6_11> · **Tier:** SUPPORTING

**Summary.** Caïssa is a neuro-symbolic multi-agent system for chess move suggestion and "grounded" natural-language commentary, presented at KI 2025 (48th German Conference on AI). The verified full title adds "and Grounded Commentary" beyond the prior note. It couples a fine-tuned chess-specific LLM (Google Gemini in the implementation) with a Prolog rule engine encoding tactics (forks, pins, skewers, discovered attacks, hanging pieces, mates) and a dynamically built Neo4j knowledge graph of the board state, bridged via LangChain RAG. A conductor agent routes queries to a GraphCypherQA chain or a chess-solver chain. The central contribution is a LangGraph verification module: a classifier plus a pipeline of "tiny agents" that decompose commentary into atomic statements, extract structured JSON, and check each fact against the Prolog/Neo4j symbolic layer, augmenting or correcting LLM claims to suppress hallucinations. The aim is verifiable, trustworthy interpretable commentary for, e.g., teaching chess.

**Depends on.** Fine-tuned LLMs (Google Gemini), Prolog rule-based tactic encoding, Neo4j knowledge graphs, LangChain/LangGraph agent orchestration, and prior chess-commentary generation work (e.g., concept-guided chess commentary).

**Results.** Reported 93.13% hit rate in tactic detection evaluated on the Lichess tactics (puzzle) dataset. Public materials describe the system as a prototype; beyond the tactic-detection number, no formal commentary-quality metrics, hallucination-rate quantification, ablations, or baselines are documented in the accessible abstract/README. A working code repository (github.com/MazenS0liman/Caissa-AI) and a Next.js/Flask demo exist.

**Weaknesses.** Full paper is paywalled; assessment rests on abstract, ACM/springerprofessional summaries, and the GitHub README, so claims are largely unverified at the methods/results level. The single headline metric (93.13% tactic-detection hit rate on Lichess) measures symbolic detection, not explanation faithfulness; the verifier checks consistency between LLM text and its own symbolic layer rather than whether a cited factor is causally load-bearing for the move's strength. Authors concede the verifier "can still make mistakes" and the system is a prototype. No causal interventions, no engine win-probability deltas, no randomization/baseline controls, and the symbolic ground truth is the same knowledge source used to generate commentary, risking circular validation. Title in prior notes is slightly truncated.

**Relation to xai-chess.** Caïssa is a complementary neuro-symbolic explainable-chess system that verifies commentary against a symbolic rule layer, whereas ChessFaith instead grades whether a cited factor is causally load-bearing via engine-measured interventions, supplying the decision-level causal ground truth Caïssa's consistency-checking lacks.


## 55. Explainable Search

**Citation:** H. Baier and M. Kaisers, "[Towards Explainable MCTS](https://ir.cwi.nl/pub/30850/30850.pdf)," in Proc. AAAI Workshop on Explainable Agency in Artificial Intelligence, 2020. [Online]. Available: https://ir.cwi.nl/pub/30850/30850.pdf  
**Link:** <https://ir.cwi.nl/pub/30850/30850.pdf> · **Tier:** SUPPORTING

**Summary.** NOTE: the PDF at this URL is titled "Towards Explainable MCTS" (Baier & Kaisers, CWI, AAAI 2020), NOT "Explainable Search" as the prior notes claimed — "Explainable Search" is their companion top-down challenges paper (IJCAI-PRICAI 2020 XAI workshop), cited within. This bottom-up paper presents a toolset of concrete building blocks for explaining Monte-Carlo Tree Search decisions, demonstrated on Connect Four and Breakthrough. All explanations are derived from the post-search tree via two operations: tree simplification (selecting relevant nodes/edges, e.g. principal variation plus siblings) and subtree summarization (abstracting subtrees into the probabilities of pre-specified, domain-meaningful positive/negative "scenarios" estimated from the proportion of simulations reaching them). It offers post-hoc commands (explain decision, why not <action>, explore) and collaborative commands that alter the search (think about, expect response, search together), framing explanation as interactive conversation and human-AI collaboration rather than one-way output.

**Depends on.** Monte-Carlo Tree Search / UCT (Kocsis & Szepesvári); AlphaZero-style search+NN systems (Silver et al.); explainable AI planning (Chakraborti, Sreedharan, Kambhampati); contrastive-explanation dialogue (Cashmore et al.); the authors' own companion "Explainable Search" challenge paper; scenario/preferred-state grounding of explanations (Khan, Poupart & Black).

**Results.** A position paper / system description with no quantitative evaluation and no user study. "Results" are the proposed taxonomy and worked qualitative examples: post-hoc explanations split into three parts (PV with siblings as sampling stats and board views; root-level subtree summarization such as Connect Four winning-group probabilities and per-square win-correlation heatmaps, Breakthrough piece-survival/material-balance features; explicit contrast of recommended move vs. union of alternatives), plus collaborative search modes that re-run MCTS under user-supplied focus or hard constraints (e.g. expected opponent recapture) and report whether the decision changed. Confidence intervals on move values drive natural-language comparisons (e.g. "d3, f3 are definitely worse"). Authors explicitly defer formal user models, scenario learning, soft constraints, and user testing to future work.

**Weaknesses.** Preliminary "ongoing exploration": no empirical evaluation, no user study, no metrics — effectiveness is asserted, not measured. Scenarios are hand-specified, domain-dependent, and assumed shared knowledge between AI and user; no method to learn or validate them. Limited to two small two-player board games; scalability and generality unclear. Explanations are descriptive correlational summaries (e.g. simulation proportions reaching a scenario), not causal — no test of whether a cited feature is actually load-bearing for the decision. Collaborative constraints are hard-coded per concrete state with no generalization. No notion of explanation faithfulness, sufficiency, or "optimality" is committed to. The title/venue mismatch with the project's prior notes also signals citation imprecision to correct.

**Relation to xai-chess.** It is a search-specific XAI precursor that generates correlational, scenario-based "why this move" explanations from an MCTS tree but never verifies them causally — exactly the gap ChessFaith fills by intervening (board-level relocation, rule-level movegen patching) to measure whether a cited factor truly drives the engine's move preference.


## 56. Building an Intelligent Tutoring System for Chess Endgames

**Citation:** M. Guid, M. Možina, C. Bohak, A. Sadikov, and I. Bratko, "[Building an Intelligent Tutoring System for Chess Endgames](https://www.scitepress.org/papers/2013/43891/43891.pdf)," in Proc. 5th Int. Conf. Computer Supported Education (CSEDU), 2013, pp. 263-266, doi:10.5220/0004389102630266.  
**Link:** <https://www.scitepress.org/papers/2013/43891/43891.pdf> · **Tier:** SUPPORTING

**Summary.** A 4-page CSEDU-2013 short paper describing a web-based intelligent tutoring system (ITS) for chess endgames (e.g., KBNK, king-bishop-knight vs lone king). Its architecture has five components: a rule-based domain model, a search engine, a tutoring model, a student model, and a user interface. The core idea is a "conceptualized domain theory" that bridges the basic rules of chess and concrete problem-solving: instead of a long optimal solution path that is too costly or too hard to memorize, the system learns a compact set of no more than 11 production rules (semi-automatically derived from tablebases), each encoding an intermediate goal achievable within d plies. The search engine validates student moves on the fly, distinguishing correct, acceptable (progress toward mate), and bad moves, and detects misconceptions via "malrules." Tutoring follows the model-tracing/ACT-R paradigm; the student model uses Bayesian Knowledge Tracing with a skill meter. The paper is a system/architecture description; empirical evaluation is left as future work.

**Depends on.** Builds on the authors' own prior work on semi-automatic domain conceptualization from chess tablebases (Guid et al. 2009; Možina et al. 2010, 2012) and on cognitive-tutor theory (ACT-R, model tracing — Anderson et al. 1995; Bayesian Knowledge Tracing — Corbett & Anderson 1995).

**Results.** No quantitative results are reported; this is a system-description paper. Concrete claims: the domain model compresses optimal endgame play into ≤11 human-assimilable production rules, each goal reachable within a fixed depth d (tunable to student skill level); the rules provably guide a student to checkmate within the 50-move rule even under slowest goal realization (per Guid et al. 2009); the search engine grades moves into correct/acceptable/bad categories and flags misconceptions. Both summative and formative evaluation are explicitly deferred to future work; the live system is at ailab.si/chesstutor.

**Weaknesses.** No empirical evaluation at all — no learning-gain, usability, or accuracy data; effectiveness of both the tutor and the auto-derived rules is unvalidated (authors admit this). Scope is narrow: a handful of elementary endgames, ≤11 rules, single deterministic-truth domain. BKT parameters were "tuned arbitrarily" with an expert, not fit to data. Only 4 pages, so architectural details (rule format, malrule construction, goal-selection logic) are sketched, not specified. Generalization beyond chess endgames is asserted, not shown. The "conceptualization" method is offloaded to cited prior papers rather than described here.

**Relation to xai-chess.** It is a pedagogical/tutoring precursor from the same Ljubljana group that operationalizes chess concepts as verifiable goals checked by a search engine, supporting xai-chess's premise that chess gives decision-level ground truth for explanations, though it teaches rather than causally tests whether a cited factor is load-bearing.


# Peripheral

*Background and tangential: broader game-AI, human-move modelling, and survey context.*


## 57. LLM-based Commentary Generation for the Guandan Card Game

**Citation:** M. Tao, X. Liang, X. Song, et al., "[Enhancing Commentary Strategies for Imperfect Information Card Games: A Study of Large Language Models in Guandan Commentary](https://arxiv.org/abs/2406.17807)," arXiv:2406.17807, 2024.  
**Link:** <https://arxiv.org/abs/2406.17807> · **Tier:** PERIPHERAL

**Summary.** This paper builds an LLM-based system that generates strategic Chinese-language commentary for Guandan, a four-player imperfect-information card game. The framework couples reinforcement learning (used to generate diverse, complex card-play scenarios) with language models structured around three modules: a state commentary guide that grounds narration in the current game state, a Theory-of-Mind (ToM)-based strategy analyzer that reasons about opponents' hidden information and intentions, and a style-retrieval module that mimics professional commentators' phrasing. The aim is contextually relevant, engaging, expert-sounding commentary for games with incomplete information. The authors report that their framework, applied to open-source LLMs, surpasses GPT-4 across multiple evaluation metrics. The work targets the broader problem that producing insightful commentary for complex incomplete-information games remains hard. Title as fetched: "Enhancing Commentary Strategies for Imperfect Information Card Games: A Study of Large Language Models in Guandan Commentary" (matches the given paper; submitted June 2024, revised April 2025).

**Depends on.** Large language models (GPT-4 and open-source LLMs), reinforcement learning for game-scenario generation, Theory-of-Mind prompting/reasoning, and retrieval-augmented style imitation; situated in the game-commentary generation literature.

**Results.** Reports that the proposed framework, built on open-source LLMs with the three modules (state guide, ToM analyzer, style retrieval), outperforms GPT-4 across multiple evaluation metrics for Guandan commentary. The abstract does not name specific datasets or numeric scores, and the evaluation appears to rely on commentary-quality metrics rather than any causal or faithfulness criterion.

**Weaknesses.** The fetched abstract gives no concrete dataset, metric definitions, or numeric results, so the "surpasses GPT-4" claim is unverifiable from the source and likely rests on human or automatic NLG metrics that conflate fluency with correctness. There is no notion of explanation faithfulness or causal grounding: commentary is judged by whether it sounds expert, not whether the cited strategic factor actually drives the optimal play. ToM reasoning is asserted rather than validated against ground-truth hidden-state inference. Imperfect-information setting precludes a clean oracle, making any causal-intervention check infeasible. Generalization beyond Guandan and beyond Chinese-language commentary is untested.

**Relation to xai-chess.** It is a peripheral, contrast case: like ChessFaith it generates game explanations/commentary, but it evaluates them by stylistic NLG quality rather than by the verifiable decision-level causal ground truth that ChessFaith's intervention benchmark supplies.


## 58. Can Large Language Models Master Complex Card Games?

**Citation:** W. Wang, F. Bie, J. Chen et al., "[Can Large Language Models Master Complex Card Games?](https://arxiv.org/abs/2509.01328)," in Proc. NeurIPS, 2025. arXiv:2509.01328.  
**Link:** <https://arxiv.org/abs/2509.01328> · **Tier:** PERIPHERAL

**Summary.** This NeurIPS 2025 paper (Wang et al., THUDM) asks whether LLMs can master complex card games the way AlphaGo/AlphaZero mastered Go and Chess. The authors supervised-fine-tune LLMs on high-quality gameplay data across eight different card games and evaluate both game strength and retention of general capabilities. Three findings: (1) via SFT on strong-AI gameplay, LLMs can approach the performance of specialized strong game AIs; (2) a single model can learn multiple games at once, with stronger results when games share themes/rules; (3) deep game mastery degrades general instruction-following ability, but mixing in a quantity of general instruction data during fine-tuning mitigates this catastrophic-forgetting trade-off. Code is released at THUDM/LLM4CardGame. The work is squarely about training LLMs to play games well, not about explaining or justifying moves, and contains no causal-intervention or faithfulness machinery.

**Depends on.** Builds on AlphaGo/AlphaZero game-mastery framing, supervised fine-tuning of LLMs on expert-trajectory data, and the catastrophic-forgetting / general-capability-retention literature (instruction-data mixing).

**Results.** LLMs fine-tuned on strong-AI gameplay approach specialized game-AI strength across eight card games; multi-game training works, better for thematically similar games; mastery causes general-capability decline that is partially recovered by blending general instruction data into training. No win-probability-drop or explanation-faithfulness metrics reported.

**Weaknesses.** Domain is imperfect-information card games, not chess, and gameplay strength is the sole objective — there is no notion of move explanation, justification, or its faithfulness, so no decision-level causal ground truth. Reported only at the abstract/summary level here: exact opponents, Elo/win-rate baselines, and the eight game identities are not verified. "Approach strong game AI" is vague without per-game numbers. The general-capability-retention claim depends on unspecified benchmark suites and instruction-data quantities. Reliance on SFT over expert trajectories means results may not transfer to settings lacking a strong teacher AI.

**Relation to xai-chess.** Peripheral context only: it shows LLMs can be trained to play complex games well, but offers no explanation-evaluation or causal-intervention methodology, which is exactly the gap ChessFaith targets.


## 59. A Survey on Explainable Deep Reinforcement Learning

**Citation:** Z. Cheng, J. Yu, and X. Xing, "[A Survey on Explainable Deep Reinforcement Learning](https://arxiv.org/abs/2502.06869)," arXiv:2502.06869, 2025.  
**Link:** <https://arxiv.org/abs/2502.06869> · **Tier:** PERIPHERAL

**Summary.** This survey reviews explainability methods for Deep Reinforcement Learning (XRL), organizing the field along four explanation levels: feature-level, state-level, dataset-level, and model-level. It catalogs techniques for making black-box DRL agents more transparent and surveys both qualitative and quantitative frameworks for assessing explanation quality. The paper also connects RL explainability to downstream applications including policy refinement/improvement, adversarial robustness, and security, and discusses links to LLMs via RLHF and alignment. It closes with open challenges toward interpretable, reliable, and accountable DRL systems. As a taxonomy-and-roadmap survey, it is descriptive rather than methodological: it does not propose a new benchmark or a causal-intervention faithfulness protocol, but it situates the broad landscape from which a decision-level explanation evaluation for a game-playing agent (like a chess engine) can be positioned. Authors: Zelei Cheng, Jiahao Yu, Xinyu Xing; submitted Feb 2025 (cs.LG, cs.AI).

**Depends on.** Builds on prior DRL interpretability/XAI literature, saliency- and feature-attribution methods, RLHF/alignment work, and earlier XRL surveys; aggregates rather than extends individual methods.

**Results.** No empirical results of its own; the contribution is a four-level taxonomy (feature/state/dataset/model), a synthesis of qualitative and quantitative evaluation frameworks for explanation quality, an applications map (policy improvement, robustness, security, LLM/RLHF connections), and a list of open research challenges.

**Weaknesses.** As a survey it provides no new method, dataset, or measurements, so it cannot be validated empirically. The faithfulness/evaluation discussion is high-level and does not advance a causal, decision-level ground-truth protocol; quantitative metrics surveyed are largely correlational (saliency/perturbation) rather than verifiable interventions. The four-level taxonomy is one of several competing organizations in the XRL literature and is not uniquely motivated. Coverage of game-playing agents specifically (chess/AlphaZero-style) and of intervention-based causal testing is thin. The WebFetch abstract page was somewhat condensed, so finer methodological claims are reported at survey granularity and may understate specifics.

**Relation to xai-chess.** It provides the broad XRL landscape and evaluation-framework backdrop against which ChessFaith's causal-intervention, decision-level faithfulness benchmark for a chess engine can be positioned as a concrete, verifiable instantiation.


## 60. Designing Skill-Compatible AI: Methodologies and Frameworks in Chess

**Citation:** K. Hamade, R. McIlroy-Young, S. Sen, J. Kleinberg, and A. Anderson, "[Designing Skill-Compatible AI: Methodologies and Frameworks in Chess](https://arxiv.org/abs/2405.05066)," arXiv:2405.05066, 2024.  
**Link:** <https://arxiv.org/abs/2405.05066> · **Tier:** PERIPHERAL

**Summary.** This paper (Hamade, McIlroy-Young, Sen, Kleinberg, Anderson; ICLR 2024) argues that superhuman performance alone does not make an AI a good partner for weaker agents, and introduces the notion of "skill-compatibility" as a measurable trait distinct from raw strength. The authors propose three methodologies for building skill-compatible chess agents and two custom chess game frameworks designed to test collaboration between strong AI and lower-skill partners. They build on the Maia line of human-like chess modeling and AlphaZero-style engines. Empirically, their skill-compatible agents outperform state-of-the-art AlphaZero-based AI in collaborative/handicap settings despite being individually weaker at standard chess, demonstrating that adapting to a partner's competence is a separable design objective. The work situates chess as a controlled testbed for human-AI interaction and decision-making with heterogeneous partners. It is largely an interaction/collaboration study rather than an explanation, faithfulness, or interpretability study.

**Depends on.** Maia (McIlroy-Young et al.) human-like chess modeling; AlphaZero/Leela-style self-play engines; broader human-AI collaboration and skill-modeling literature.

**Results.** Skill-compatible agents outperformed SOTA AlphaZero-based chess AI in collaborative settings despite being individually weaker at conventional chess; the authors show skill-compatibility is qualitatively and measurably distinct from raw performance, and characterize mechanisms enabling adaptation to partners of varying ability.

**Weaknesses.** Confined to chess and to two bespoke collaboration frameworks, so external validity to real human-AI teaming or other domains is unestablished; partners are largely simulated/lower-skill agents rather than measured human players in the headline results; "skill-compatibility" is operationalized through win-rate-style metrics in constructed games, which may not capture explanation or trust dimensions; no treatment of why a move is good (no causal/factor-level analysis), so it offers no decision-level causal ground truth.

**Relation to xai-chess.** It shares the project's use of chess as a controlled engine-backed testbed for studying AI behavior, but targets human-AI skill compatibility rather than the causal-intervention faithfulness of move explanations that ChessFaith grades.


## 61. Aspect-based Sentiment Evaluation of Chess Moves (ASSESS)

**Citation:** H. Alrdahi and R. Batista-Navarro, "[Aspect-based Sentiment Evaluation of Chess Moves (ASSESS): an NLP-based Method for Evaluating Chess Strategies from Textbooks](https://arxiv.org/abs/2405.06499)," in Proc. 10th Workshop on Games and Natural Language Processing (Games and NLP), LREC-COLING 2024, arXiv:2405.06499, 2024.  
**Link:** <https://arxiv.org/abs/2405.06499> · **Tier:** PERIPHERAL

**Summary.** ASSESS reframes evaluation of chess moves described in instructional textbooks as an aspect-based sentiment analysis (ABSA) task. Each move-action phrase is treated as an aspect term, and the system classifies the textbook author's stance toward that move as positive (advantageous), negative (disadvantageous), or neutral. The authors hand-annotate 726 sentences from a chess-teaching textbook (437 positive, 153 negative, 133 neutral; 65% Cohen's kappa) atop the prior LEAP corpus plus Jhamtani et al. (2018) commentaries, and benchmark against SemEval-2014 and MAMS. They fine-tune RoBERTa-base on chess text (FT-RB) versus a vanilla RoBERTa baseline (VRB), and use Stockfish 16 (skill 8, ~2400 Elo, depth 10) as an external reference for move quality. Fine-tuning yields modest gains (F1 ~55-62% vs ~54-55%) with larger 20-30% improvements on minority classes under oversampling. The work establishes baseline results for sentiment-style move evaluation from natural-language chess pedagogy.

**Depends on.** LEAP corpus (Alrdahi & Batista-Navarro), Jhamtani et al. (2018) chess commentary dataset, RoBERTa, SemEval-2014 ABSA and MAMS benchmarks, Stockfish 16 as an engine reference.

**Results.** FT-RB outperforms vanilla RoBERTa: F1 ~55-62% (original) vs ~54-55%; oversampling gives 20-30% F1 gains on minority (negative/neutral) classes. Annotation reached 65% Cohen's kappa. Adding move-action-type features slightly hurt performance. Positioned as first-pass baselines on a small new annotated set.

**Weaknesses.** Tiny annotated corpus (726 sentences, heavily class-imbalanced) and only moderate 65% kappa cap reliability; F1 in the 50-62% range is weak for a 3-class task. Aspect extraction is rule-based and not learned jointly with classification, propagating linkage errors. Sentiment here is the textbook author's rhetorical stance, not verified game-theoretic correctness; Stockfish is used loosely as reference, not as ground truth. Models fail on implicit threats and long-term strategy, and domain-feature injection backfired. Single-textbook source limits generalization.

**Relation to xai-chess.** Peripheral and methodologically orthogonal: ASSESS measures textual sentiment/stance toward chess moves from prose, whereas ChessFaith tests whether a cited factor is causally load-bearing via engine intervention—sharing only the chess+NLP+Stockfish setting, not the causal-faithfulness goal.


## 62. Behavior-Based Knowledge Representation Improves Prediction of Human Chess Moves

**Citation:** B. Skidanov, D. Erbesfeld, G. Weiss, and A. Elyasaf, "[A Behavior-Based Knowledge Representation Improves Prediction of Players' Moves in Chess by 25%](https://arxiv.org/abs/2504.05425)," arXiv:2504.05425, 2025.  
**Link:** <https://arxiv.org/abs/2504.05425> · **Tier:** PERIPHERAL

**Summary.** The paper (whose arXiv title is "A Behavior-Based Knowledge Representation Improves Prediction of Players' Moves in Chess by 25%") tackles predicting the moves of human, intermediate-level chess players rather than optimal engine play. The authors combine hand-engineered, human-meaningful chess features (material balance, piece positions, control metrics, threats/vulnerabilities) with a behavior-based knowledge representation that encodes how players actually decide, then train a machine-learning model on real Lichess game data, focusing on the opening phase. Their central empirical claim is a 25% relative improvement in move-prediction accuracy over baseline approaches. The work positions human-concept feature engineering as both more accurate and more interpretable than opaque board-state evaluation. It is short (8 pages, 2 tables, 2 figures). Note: the supplied PROJECT title ("Behavior-Based Knowledge Representation Improves Prediction of Human Chess Moves") differs slightly from the arXiv title, but author list and content confirm it is the same paper.

**Depends on.** Builds on chess move-prediction / human-style modeling (e.g., Maia-style behavioral cloning), classical chess feature engineering, and behavioral-programming / knowledge-representation traditions from the authors' group; uses Lichess game data and presumably a Stockfish-style evaluator for features.

**Results.** Headline result: ~25% relative improvement in human move-prediction accuracy from the behavior-based representation versus baselines, concentrated on the opening phase for intermediate players. Absolute accuracy figures, the exact ML architecture, and the specific baselines are not clearly stated in the abstract/accessible text.

**Weaknesses.** Very short paper with thin methodological disclosure: exact model architecture, baseline definitions, dataset size/splits, and absolute accuracy numbers are not surfaced, so the "25%" is hard to contextualize (relative vs absolute, against which baseline). Scope is narrow: opening phase only and intermediate-skill players, raising generalization concerns to middlegame/endgame and other rating bands. "Interpretability" is asserted via human-meaningful features but no faithfulness/causal validation of whether those features actually drive predictions is provided. No public code/benchmark evident, and no decision-level causal ground truth.

**Relation to xai-chess.** It is peripheral context: like ChessFaith it grounds chess modeling in human-meaningful concepts and claims interpretability, but it predicts human moves rather than causally testing whether a cited factor is load-bearing, so it illustrates exactly the unverified "concept-based interpretability" gap that ChessFaith's causal-intervention benchmark aims to close.


## 63. Predicting Human Chess Moves with n-gram Language Models

**Citation:** D. Zhong, D. Huang, and C. Greenberg, "[Predicting Human Chess Moves: An AI Assisted Analysis of Chess Games Using Skill-group Specific n-gram Language Models](https://arxiv.org/abs/2512.01880)," arXiv:2512.01880, 2025.  
**Link:** <https://arxiv.org/abs/2512.01880> · **Tier:** PERIPHERAL

**Summary.** This paper models human (not optimal) chess move-making with n-gram language models over move sequences, treating prediction as a behavioral problem rather than engine optimization. Players from Lichess are split into seven skill groups (novice to expert), and a separate n-gram model is trained per group. A dynamic "selector" classifies a game's skill group from early-game information and routes to the matching model. Reported results: skill-group classification accuracy up to 31.7% from 16 half-moves, and move-prediction accuracy up to 39.1% relative improvement over a single-model benchmark via the selector. The contribution is framing chess move sequences as a language-modeling corpus stratified by human skill, emphasizing computational efficiency over heavy neural models. The work captures the variability of human play that standard engines ignore. (Resolved: title/authors confirmed from arXiv; only abstract-level detail was available, so dataset size, exact n-gram order, and full evaluation protocol are not verified here.)

**Depends on.** Classical n-gram language modeling; Lichess open game data; human-style move modeling lineage (e.g., Maia/behavioral chess prediction).

**Results.** Skill-group classification up to 31.7% accuracy from 16 half-moves; selector-assisted move prediction up to 39.1% more accurate than a single-model benchmark. Absolute move-prediction accuracy, baseline values, and statistical significance not reported in the abstract.

**Weaknesses.** Only abstract-level content was accessible, so claims could not be cross-checked against full methodology. Reported metrics are ambiguous: "up to 39.1% more accurate" lacks an absolute baseline and the 31.7% classification accuracy is low. N-gram models capture surface move-token co-occurrence, not board-state causal structure, and provide no notion of why a move is good. No causal intervention, no explanation evaluation, no oracle win-probability — purely predictive. Skill-group binning and selector design risk leakage/circularity not addressed in the abstract.

**Relation to xai-chess.** Peripheral: it predicts human chess behavior via language models but offers no causal-intervention or explanation-faithfulness machinery, so it bears only on the chess-modeling backdrop, not the benchmark's core method.


## 64. Who Benefits from AI? Self-Selection, Skill Gap, and the Hidden Costs of AI Feedback in Chess

**Citation:** C. Riedl and E. Bogert, "[Who Benefits from AI? Self-Selection, Skill Gap, and the Hidden Costs of AI Feedback in Chess](https://arxiv.org/abs/2409.18660)," arXiv:2409.18660, 2024.  
**Link:** <https://arxiv.org/abs/2409.18660> · **Tier:** PERIPHERAL

**Summary.** Riedl and Bogert study how voluntary adoption of AI feedback affects human learning and collective outcomes, using five years of observational data from an online chess platform covering over 52,000 players, supplemented by 42 platform-level natural experiments. They find that motivated, higher-skilled players disproportionately self-select into using AI feedback. Once this endogenous self-selection/motivation is statistically accounted for, the apparent learning benefit of AI feedback largely disappears. They further identify two systemic, downstream costs: AI access amplifies pre-existing skill disparities (the strong benefit more), and reliance on a centralized AI source reduces intellectual diversity in play, an effect the natural experiments establish as causal. The paper's central contribution is methodological caution: naive observational estimates of AI's educational value are confounded by who chooses to use it. This is an empirical economics/HCI study of AI-assisted human learning, not a method for explanation or interpretability.

**Depends on.** Builds on causal-inference methodology (self-selection/endogeneity correction, natural experiments) and the empirical study of AI-assisted human decision-making and learning, instantiated in the chess domain.

**Results.** Higher-skilled, more motivated players preferentially adopt AI feedback; the apparent learning gain vanishes after controlling for endogenous motivation/self-selection. AI access widens skill gaps, and centralized AI feedback reduces intellectual (move) diversity, with 42 natural experiments confirming the diversity reduction is causal. Sample: >52,000 players over five years.

**Weaknesses.** Findings rest on observational platform data; despite self-selection corrections and natural experiments, unobserved confounders remain a threat. The abstract reports no explicit limitations and gives no effect sizes, model specifications, or robustness details (verified only from abstract page, not full text). External validity is narrow: a single chess platform with self-selected users, so generalization to other AI-feedback settings is uncertain. "Intellectual diversity" is operationalized via chess play and may not capture broader notions of diversity. No interpretability, explanation-faithfulness, or causal-intervention-on-explanations content.

**Relation to xai-chess.** Only tangentially related: it shares the chess-and-AI domain and a causal-inference sensibility, but concerns the human-learning costs of AI feedback rather than evaluating the faithfulness of move explanations, so it serves at most as teaching/simulatability-track context.

---


# Additions — move-explanation and dataset sweep (2026-08-01)

*23 works surfaced by a six-pass sweep: move-level explanation systems, symbolic and DSL explanation, explanation datasets, and evaluation methodology. Tiers as in `PAPERS.md`.*


## 65. Learning to Generate Move-by-Move Commentary for Chess Games (GameKnot)

**Citation:** Jhamtani, Gangal, Hovy, Neubig, and Berg-Kirkpatrick, "[Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data](https://aclanthology.org/P18-1154/)," in Proc. ACL, 2018, pp. 1661-1671.
**Link:** <https://aclanthology.org/P18-1154/> · **Tier:** CORE

**Summary.** The foundational chess-commentary dataset and generator, and the substrate every later neural system trains on. 11,578 GameKnot games yield 298,008 move-commentary pairs, tagged into six categories: Direct Move Description (31.4%), Move Quality (8.0%), Comparative (3.7%), Planning/Rationale (31.2%), Contextual Game Info (12.6%), General (29.9%). Generation conditions on hand-coded board and attack features.

**Depends on.** GameKnot forum scraping; standard seq2seq generation; SVM category propagation from a small hand-annotated seed.

**Results.** Human judges rate the model at or above the human ground truth while BLEU-4 sits near 2, which the authors use to argue BLEU is the wrong instrument here. Their own citation of the point: BLEU "is known to correlate poorly with human relevance scores for NLG tasks." Roughly 30% of the human ground-truth comments were judged not valid for the move they annotate.

**Weaknesses.** Category labels are SVM output on 297K of 298K rows; only 1,000 comments were hand-annotated, by two annotators, and Comparative falls back to a keyword rule. Roughly 30% of comments are General with no chess content and 23% are five words or fewer. There is no data dump: the repo ships a Python-2.7 crawler and the license field is null. Evaluation is BLEU plus human validity, with no faithfulness measure.

**Relation to xai-chess.** The corpus a grading benchmark would consume, and the source of two problems it must handle: machine-inferred rationale labels, and comments that refer to the game rather than the tagged move. A downloadable mirror exists inside ChessGPT's release [74].


## 66. Empirical Evaluation of Concept Probing for Game-Playing Agents

**Citation:** Pálsson and Björnsson, "[Empirical Evaluation of Concept Probing for Game-Playing Agents](https://ebooks.iospress.nl/doi/10.3233/FAIA240574)," in Proc. ECAI, FAIA vol. 392, 2024, pp. 874-881, doi:10.3233/FAIA240574.
**Link:** <https://ebooks.iospress.nl/doi/10.3233/FAIA240574> · **Tier:** CORE

**Summary.** The closest existing precedent to grading an interpretability claim against causal ground truth in chess. They build the ground truth by retraining the Stockfish NNUE with a target concept's positions skipped from the training stream (a patched `make_skip_predicate` in `training_data_loader.cpp`, preserving minibatch size and hyperparameters), then playing the ablated network against the intact one and converting the strength difference to Elo. Probing methods are scored by how well their output tracks that Elo drop.

**Depends on.** Stockfish NNUE and its trainer; concept probing (ridge, LGBM, neural); Elo estimation from gauntlet play.

**Results.** 3,000 games each at depths 14, 15 and 16, so 9,000 per agent. Pearson correlation with the Elo drop is 0.69 / 0.68 / 0.73 for raw probe accuracy and 0.83 / 0.87 / 0.92 for the differential between probes on the intact and concept-removed networks. The published abstract states that reading probe accuracy as concept importance is "somewhat unreliable" and proposes amnesic-style alternatives.

**Weaknesses.** The intervention is on training data, so the compared object is a differently-trained network, which introduces a network-identity confound. It grades probing methods rather than stated reasons, so it never asks whether a factor an explanation cites is load-bearing for a particular move. Retraining cost bounds the number of concepts testable.

**Relation to xai-chess.** Must be differenced from explicitly: it reached causal ground truth in chess first, by a different route (training-data ablation, not rule modification) and for a different target (probe importance, not a cited reason). The gauntlet design is directly reusable, and the raw-versus-differential gap is independent evidence that decodability and causal use come apart.


## 67. Explain Your Move (SARFA)

**Citation:** Puri, Verma, Gupta, Kayastha, Deshmukh, Krishnamurthy, and Singh, "[Explain Your Move](https://arxiv.org/abs/1912.12191)" (SARFA), in Proc. ICLR, 2020. arXiv:1912.12191.
**Link:** <https://arxiv.org/abs/1912.12191> · **Tier:** CORE

**Summary.** Perturbation-based saliency for agent actions, balancing specificity (effect on the explained action's Q-value alone) against relevance (down-weighting perturbations that shift other actions). Ships the Chess Saliency Dataset: 100 Lichess puzzles in which three experts rated above 2200 mark the pieces relevant to the correct move, with majority vote as the label.

**Depends on.** Perturbation saliency for RL; expert annotation; chess and Go as evaluation domains.

**Results.** SARFA reaches ROC-AUC 0.92 against the expert vote, holding at 0.92 under removal of non-salient pieces. Human study with N=40 at ELO 1600-2000 over 15 puzzles: solve accuracy 56.67% with no saliency, 72.41% with SARFA, 40.84% and 24.60% for two prior methods.

**Weaknesses.** The benchmark grades agreement with human expert consensus, which is the plausibility axis, not fidelity to the agent. Huber et al. state this in print. The unit is a board square, not a concept or motif, so it cannot express why a named factor matters. 100 puzzles is small, and no significance test accompanies the headline human table.

**Relation to xai-chess.** The only existing chess explanation benchmark, and it measures the wrong axis. It is the direct comparison point for arguing that a causal benchmark is needed, and its dataset is a ready-made plausibility baseline to report alongside a faithfulness score.


## 68. Leveraging Rationales to Improve Human Task Performance

**Citation:** Das and Chernova, "[Leveraging Rationales to Improve Human Task Performance](https://arxiv.org/abs/2002.04202)," in Proc. IUI, 2020. arXiv:2002.04202.
**Link:** <https://arxiv.org/abs/2002.04202> · **Tier:** CORE

**Summary.** The canonical chess human study of AI rationales, and its result is awkward for the field. Four between-subjects conditions over three days: no assistance, best-move hints, RGA (rationale generated from Stockfish's utility function), and RGA+ (RGA plus hand-added domain knowledge such as capture-next-move, check-next-move, and mate in three or fewer). N=60 after exclusions.

**Depends on.** Stockfish evaluation internals; rationale generation from a utility function; MTurk human-subjects methodology.

**Results.** The engine-faithful rationale (RGA) produced no significant gain over either baseline. Only RGA+, deliberately augmented with criteria the Stockfish utility does not represent, beat the no-assistance condition on both win rate and move percentile rank.

**Weaknesses.** Significance verdicts are reported without means or effect sizes for the performance measures. MTurk population, self-reported learning, single domain. Usefulness is measured, not faithfulness, so the study cannot say whether RGA's rationale reflected the engine's computation, only that it did not help.

**Relation to xai-chess.** The sharpest available evidence that faithful does not imply useful, demonstrated in chess with chess players. It guards against conflating the benchmark's grounding property with a claim about pedagogical value, and it is the only chess paper cited in Lai et al.'s survey of 100+ AI-assisted decision-making studies.


## 69. Communicating Chess Strategies in Natural Language

**Citation:** Cui, Ling, and Ng, "[Communicating Chess Strategies in Natural Language](https://arxiv.org/abs/2607.11486)," arXiv:2607.11486, 2026.
**Link:** <https://arxiv.org/abs/2607.11486> · **Tier:** CORE

**Summary.** Explicitly rejects post-hoc justification as the task. Existing concept and commentary work "seeks to justify why a given move/variation/strategy makes sense, with improvement in quality of play being secondary"; they instead demand descriptions that are pedagogically useful. A pruned strategy tree is built from a Lichess puzzle with Stockfish 16, serialized to JSON, and verbalized by an LLM under three orthogonal ablations: tree guidance, concept guidance from puzzle themes, and iterative self-reflection.

**Depends on.** Stockfish 16 strategy trees; Lichess puzzle themes as the concept channel; LLM verbalizers (o3, gpt-oss-120b).

**Results.** Evaluation is Tree-Expanded Puzzle Playing: a separate player, LLM proxy plus 25 rated humans, sees only the description and plays the puzzle against adversarially sampled opponent replies, scored by the resulting Stockfish evaluation. No LLM judge appears in the metric. Findings: evaluating only the main line is insufficient; pure concept-based description yields little improvement; LLM judges are unreliable proxies for humans; and verbalizing the JSON tree is lossy, with the raw structure outperforming any prose rendering.

**Weaknesses.** 100 puzzles. Measures information transfer to a downstream player, not whether the description is causally faithful to any system's computation. No code link in the fetched text.

**Relation to xai-chess.** The strongest current alternative evaluation design, and a load-bearing negative result against concept-conditioned grounding of the kind CCC [11] uses. Its behavioral metric is complementary to a causal one: it asks whether the explanation carries usable information, not whether the cited factor drove the decision.


## 70. Automated Chess Tutor

**Citation:** Sadikov, Možina, Guid, Krivec, and Bratko, "[Automated Chess Tutor](https://doi.org/10.1007/978-3-540-75538-8_2)," in Computers and Games (CG 2006), LNCS, pp. 13-25, doi:10.1007/978-3-540-75538-8_2.
**Link:** <https://doi.org/10.1007/978-3-540-75538-8_2> · **Tier:** CORE

**Summary.** CCC's direct ancestor, twenty years early, and the clearest statement of the recipe the field keeps reinventing. Crafty returns the principal variation; the system takes the vector difference between the evaluation-feature vector of the current position and that of the position at the end of the PV. Positive components are goals achieved, negative ones weaknesses created. In their words: "The goals in our schema are simply the evaluation function's features." An expert-system IF-THEN layer composes elementary features into human-level concepts and suppresses redundant complementary comments.

**Depends on.** Crafty, essentially unmodified, and its named evaluation terms; hand-authored composition rules.

**Results.** Sample output of the form "The move aims to centralize the Knight and to improve King's safety." No evaluation of any kind is reported; it is a design paper.

**Weaknesses.** No evaluation. Fails on very bad moves because minimax does not search for a bad envisioned position. Unclear how far down the PV to comment, since that depends on user strength. Limited to relatively simple endgames without further manual knowledge engineering.

**Relation to xai-chess.** Establishes that the eval-delta-to-prose recipe predates the neural era, and that it has never been checked for correctness at any point in its lineage — 1991 commercial [87], 2006 here, 2025 in CCC [11]. It is the historical anchor for the claim that the field has a generation method and no verification method.


## 71. Automatic Recognition of Similar Chess Motifs

**Citation:** Bizjak and Guid, "[Automatic Recognition of Similar Chess Motifs](https://doi.org/10.1007/978-3-031-11488-5_12)," in Advances in Computer Games (ACG 2021), LNCS vol. 13262, pp. 131-141, doi:10.1007/978-3-031-11488-5_12.
**Link:** <https://doi.org/10.1007/978-3-031-11488-5_12> · **Tier:** CORE

**Summary.** The nearest thing in the literature to a term-level motif vocabulary. Static terms cover piece placement, distance-decayed reachability, connectivity relations (attacks, defends, X-ray), and pawn-structure features. Dynamic terms describe the solution line — captures by either side, check, promotion, sacrifice, mate — plus per-move markers for pieces moved, captured, sacrificed, or involved in mate. Dynamic terms are deliberately position-independent, using piece types without squares, so motifs generalize across the board. Indexed in Lucene and ranked by BM25.

**Depends on.** A 46,370-puzzle corpus auto-generated from Lichess by a blunder-blunder pattern with engine filtering; Lucene/BM25 retrieval; CT-ART 6.0 expert pairings as evaluation.

**Results.** On 400 expert-paired puzzles, top-1 accuracy is 0.252 for static terms alone, 0.418 for dynamic alone, 0.481 combined; top-10 is 0.433 / 0.761 / 0.814.

**Weaknesses.** A bag of tokens for retrieval similarity, with no compositional semantics and no notion of a witness. It identifies that two positions share a motif, never that a motif is the reason a move is good. Evaluation is retrieval accuracy against expert pairings, not explanation quality.

**Relation to xai-chess.** The dynamic-beats-static result is the strongest published evidence that a move's reason is not recoverable from the position alone, which is a premise the benchmark depends on. Its token set is the most usable existing starting point for an automatically-extracted factor vocabulary.


## 72. Unveiling Concepts Learned by a World-Class Chess-Playing Agent

**Citation:** Pálsson and Björnsson, "[Unveiling Concepts Learned by a World-Class Chess-Playing Agent](https://www.ijcai.org/proceedings/2023/0541.pdf)," in Proc. IJCAI, 2023, pp. 4864-4872, doi:10.24963/ijcai.2023/541.
**Link:** <https://www.ijcai.org/proceedings/2023/0541.pdf> · **Tier:** SUPPORTING

**Summary.** The only major probing work targeting Stockfish's NNUE rather than an AlphaZero-lineage network. Ridge and logistic probes on NNUE layer activations, plus a Shapley analysis over the classical hand-crafted evaluation used as a surrogate, and a comparison of NNUE against classical evaluation to localize where NNUE's strength comes from.

**Depends on.** Stockfish NNUE and its classical evaluation terms; Leela self-play positions; standard probing methodology.

**Results.** Concepts drawn from Stockfish classical evaluation terms plus custom pawn features are recoverable from NNUE activations at varying accuracy.

**Weaknesses.** Explicitly not per-move: the authors state probing is "a global method used to shed light on how well the concept is represented in the network's activation, but not to explain individual samples." No causal intervention. Code availability unverified.

**Relation to xai-chess.** Extends the correlational-probing critique beyond the AlphaZero lineage to the engine family the benchmark actually uses as an oracle, and supplies the authors' own statement that probing cannot explain individual decisions.


## 73. Measuring Progress in Dictionary Learning with Board Game Models

**Citation:** Karvonen, Wright, et al., "[Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models](https://arxiv.org/abs/2408.00113)," in Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2408.00113.
**Link:** <https://arxiv.org/abs/2408.00113> · **Tier:** SUPPORTING

**Summary.** Supervised metrics for sparse autoencoder quality using board games as a ground-truth substrate. Two metrics: coverage (the max F1 any single SAE feature achieves as a classifier for a board property) and board reconstruction (recovering the full board from high-precision features). Ground truth is 768 piece-square properties plus roughly 15 strategy properties including check, pins, forks, threats, castling rights, en passant legality, legal moves, and threatened squares. Introduces p-annealing for SAE training.

**Depends on.** Two 8-layer GPTs trained on chess (16M Lichess games) and Othello; SAE training methodology.

**Results.** SAE features are scored by classification agreement against the ground-truth properties. 500+ SAEs released.

**Weaknesses.** No causal validation. Features are scored by correspondence with ground-truth board properties, never by ablating them and observing whether the model's move changes. This is the standard SAE-evaluation gap.

**Relation to xai-chess.** The clearest instance of the survey's correlational stop-point in the SAE literature, and a demonstration that chess supplies rule-decidable ground truth for interpretability metrics — the same property the benchmark exploits, applied to features rather than to stated reasons.


## 74. ChessGPT: Bridging Policy Learning and Language Modeling

**Citation:** Feng, et al., "[ChessGPT: Bridging Policy Learning and Language Modeling](https://arxiv.org/abs/2306.09200)," in NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2306.09200.
**Link:** <https://arxiv.org/abs/2306.09200> · **Tier:** SUPPORTING

**Summary.** A mixed game-and-language corpus plus two models: ChessGPT (continued pretraining on the corpus) and ChessCLIP (contrastive alignment between a PGN prefix and its following annotation, enabling PGN-to-text retrieval rather than generation). Corpus totals 42.8 GB and 28.1M documents, including 245K annotated games yielding 1.3M board-language pairs, 83K YouTube transcripts, 410K Reddit conversations, and 17.5M Lichess games.

**Depends on.** RedPajama-3B base; Lichess, CCRL, FICS and pro-player game dumps; web-crawled chess text.

**Results.** Explanation-adjacent evaluation is a 4-way multiple-choice annotation task over 3K held-out game-language pairs. ChessCLIP performs unexpectedly well despite never training on game play.

**Weaknesses.** Not a move-explanation system: no free-form commentary task, no faithfulness evaluation, no explanation demo. Chess books, forums, blogs and YouTube transcripts were withheld from the SFT release for legal reasons, as were commercial annotated PGN sources. The HF dataset viewer is broken and the language subsets are keyword-filtered crawl, so relevance is noisy.

**Relation to xai-chess.** Chiefly a data resource. Its `annotated_pgn_free.tar.gz` is a directly downloadable mirror of 12,770 GameKnot PGNs with move-aligned prose, which is the practical route to the Jhamtani corpus [65] given that the original repo ships only a crawler.


## 75. Using Patterns and Plans in Chess (PARADISE)

**Citation:** Wilkins, "[Using patterns and plans in chess](https://doi.org/10.1016/0004-3702(80)90039-9)," *Artificial Intelligence*, vol. 14, no. 2, pp. 165-203, 1980, doi:10.1016/0004-3702(80)90039-9.
**Link:** <https://doi.org/10.1016/0004-3702(80)90039-9> · **Tier:** SUPPORTING

**Summary.** The earliest system that explains a specific chess move symbolically. Knowledge is encoded as production rules whose "actions post concepts in the data base while the conditions match patterns in the chess position and data base." Plans are discovered during static analysis, and a small tree search runs only to confirm that a plan is best. The justification for a move is therefore the concept chain that produced its plan, not a score.

**Depends on.** Hand-authored production rules over a chess pattern vocabulary; a plan-guided searcher.

**Results.** Search trees of "tens and hundreds of nodes, not thousands to hundreds of thousands," comparable to a human master, with combinations found as deep as 19 ply. Domain is tactically sharp middlegame positions from master games.

**Weaknesses.** The knowledge base is inserted manually and the primitives are hard-coded to chess. The internal syntax of a production rule and the concept vocabulary itself could not be verified: *AI* 14(2) and SRI Technical Note 509 are both unreachable online. The frequently repeated claim that the test set was 100 positions from Reinfeld's *Win at Chess* is UNVERIFIED.

**Relation to xai-chess.** Establishes that per-move symbolic justification is a fifty-year-old idea that was abandoned rather than refuted, and supplies the historical baseline against which modern concept-conditioned prose looks like a regression in explanatory structure.


## 76. An Advice Program for a Complex Chess Programming Task

**Citation:** Bratko and Michie, "[An advice program for a complex chess programming task](https://doi.org/10.1093/comjnl/23.4.353)," *The Computer Journal*, vol. 23, no. 4, pp. 353-359, 1980, doi:10.1093/comjnl/23.4.353.
**Link:** <https://doi.org/10.1093/comjnl/23.4.353> · **Tier:** SUPPORTING

**Summary.** The Advice Language line (AL0/AL1/AL3), in which chess knowledge is written as advice — a goal plus constraints, compiled into a forcing tree — and organized into advice tables. This paper reports "the first computer implementation of Master skill in a nontrivial chess end game other than by exhaustive tabulation."

**Depends on.** Michie's theory of advice; pattern-based endgame knowledge representation; correctness proofs over the resulting strategies.

**Results.** Master-level play in a nontrivial endgame from declarative advice rather than search or tablebase lookup. Bratko separately gave an informal correctness proof of the KRK strategy, later machine-checked via SAT (ICGA Journal 36(2), 2013) and in Isabelle/HOL with Z3 (CADE 2015).

**Weaknesses.** The exact AL clause syntax could not be confirmed: every primary text is paywalled or pre-digital, so the syntax must not be asserted. Crossref lists only Bratko as author while the publisher page and DBLP both give Bratko and Michie. Hand-authored throughout, and confined to endgames.

**Relation to xai-chess.** The origin of the idea that a chess reason can be a formal object with a checkable meaning, which is what a factor DSL is. See [77] for the open re-formalisation that makes its content readable without publisher access.


## 77. Re-formalisation of Bratko's KRK Strategy

**Citation:** Janičić, Marić, and Maliković, **[title not verified]**, *Logical Methods in Computer Science*, vol. 15, no. 1:34, 2019, doi:10.23638/LMCS-15(1:34)2019.
**Link:** <https://doi.org/10.23638/LMCS-15(1:34)2019> · **Tier:** SUPPORTING

**Summary.** The open-access rendering of an advice-language chess strategy, and the practical substitute for the paywalled AL primary sources. It states the auxiliary predicates explicitly credited to Bratko — `room` (half-perimeter of the rectangle confining the black king, 15 if unconfined), `critical square`, `rook exposed`, `rook divides`, `L-pattern`, `kings on same edge` — together with Manhattan and Chebyshev distance.

**Depends on.** Bratko's KRK strategy; formal verification machinery.

**Results.** Seven prioritised strategy rules — ImmediateMate, ReadyToMate, Squeeze, Approach, KeepRoom, RookHome, RookSafe — with coverage over all 175,168 white-to-move KRK positions: Squeeze 116,504; RookHome 32,520; Approach 16,180; ReadyToMate 4,676; KeepRoom 3,344; ImmediateMate 1,512; RookSafe 432. Additionally shows KRK is a white win on n x n boards for all n > 3.

**Weaknesses.** Title and author initials were not verifiable in this pass and must be resolved from the DOI before citation. Confined to one endgame. It renders AL's content, not its syntax.

**Relation to xai-chess.** The most concrete freely-readable example of a formal chess-reason vocabulary with decidable predicates and measured coverage, which is close to what a factor DSL needs to look like.


## 78. Argument Based Machine Learning (ABML)

**Citation:** Možina, Žabkar, and Bratko, "[Argument based machine learning](https://doi.org/10.1016/j.artint.2007.04.007)," *Artificial Intelligence*, vol. 171, no. 10-15, pp. 922-937, 2007, doi:10.1016/j.artint.2007.04.007.
**Link:** <https://doi.org/10.1016/j.artint.2007.04.007> · **Tier:** SUPPORTING

**Summary.** Expert arguments enhance individual learning examples rather than the hypothesis space as a whole. Each argument attaches to a single example; positive arguments say why an example belongs to its class, negative ones argue against. An argument is a conjunction of attribute conditions. The learner, ABCN2, produces unordered probabilistic rules constrained so that every argumented example is covered by at least one rule containing at least one of its positive arguments. The elicitation loop shows the expert only critical examples the learner cannot yet explain.

**Depends on.** CN2 rule induction; expert knowledge elicitation.

**Results.** Applied to chess concept construction in a series of follow-ups (ECAI 2008, CG 2008), including the bad-bishop concept, and to tablebase-grounded concept synthesis [79].

**Weaknesses.** Arguments constrain rule induction; they are never used to justify a particular decision at inference time. Requires an available expert. The bad-bishop worked example is no longer online (`ailab.si/matej/bad_bishop.html` returns 404), so its specific rule form is UNVERIFIED.

**Relation to xai-chess.** The clearest demonstration that argumentation in chess has been applied to *learning* concepts and never to *justifying* a move, which locates an unoccupied position directly adjacent to the benchmark's target.


## 79. Deriving Concepts and Strategies from Chess Tablebases

**Citation:** Guid, Možina, Sadikov, and Bratko, "[Deriving Concepts and Strategies from Chess Tablebases](https://doi.org/10.1007/978-3-642-12993-3_18)," in Advances in Computer Games (ACG 2009), LNCS, pp. 195-207, doi:10.1007/978-3-642-12993-3_18.
**Link:** <https://doi.org/10.1007/978-3-642-12993-3_18> · **Tier:** SUPPORTING

**Summary.** Combines specialized minimax search with ABML to semi-automatically synthesize human-usable knowledge from the KBNK tablebase. The output is a set of concepts and goals that can serve as textbook instructions for a difficult endgame.

**Depends on.** KBNK tablebase as ground truth; ABML [78]; specialized minimax.

**Results.** Concepts obtained were rendered as textbook instructions and evaluated in follow-up work (ITS 2012) by a student pilot and by experienced chess trainers — one of very few human-evaluated symbolic chess explanation results.

**Weaknesses.** Confined to one endgame. Semi-automatic, requiring expert interaction throughout. Teaches against verified structure rather than testing whether a cited factor drives a move.

**Relation to xai-chess.** The cleanest prior example of grounding symbolic chess concepts in verified ground truth rather than human labels, and the closest existing analogue to using a decidable oracle to certify explanatory content.


## 80. Learning a Game Commentary Generator with Grounded Move Expressions

**Citation:** Kameko, Mori, and Tsuruoka, "Learning a Game Commentary Generator with Grounded Move Expressions," in Proc. IEEE Conf. Computational Intelligence and Games (CIG), 2015, pp. 177-184.
**Link:** IEEE CIG 2015 · **Tier:** SUPPORTING

**Summary.** Solves a problem the chess literature has not acknowledged: a commentary sentence is usually not about the current position, but about a hypothetical continuation or a past line, so its move expressions must first be grounded to a node in the game tree. Candidate trees are enumerated by rule over the current position plus the previous three moves, inserting a Pass for lines that are illegal but commented, and the commented tree is selected using engine evaluations on the heuristic that experts comment on bad moves by showing the right move — so the correct tree is the one not containing a sequence of bad moves.

**Depends on.** The Gekisashi shogi engine and its evaluation features; a shogi commentary corpus; log-linear language modeling.

**Results.** At least one candidate tree for 44,166 of 54,084 positions (81.2%); of 100 hand-checked, the correct tree for 79. Generation uses a 3-layer perceptron predicting a characteristic-word vector, then a log-linear model with best-first search. No BLEU and no human study — grounding accuracy plus qualitative figures.

**Weaknesses.** Roughly 20% of move-expression comments produce no candidates and roughly 20% select the wrong tree, and the authors state both directly affect the trained model. Generation maximizes local probabilities with no global model, so output is sometimes ungrammatical. Training conflates hypothetical with actual positions, causing tense disagreement. Shogi, not chess.

**Relation to xai-chess.** Identifies a prerequisite that sits upstream of faithfulness: a benchmark grading a cited factor against a position must first establish which position the claim was about. Chess commentary corpora [65] exhibit the same problem and no chess system addresses it.


## 81. The Chess Query Language (CQL)

**Citation:** Costeff, "[The Chess Query Language: CQL](https://doi.org/10.3233/ICG-2004-27404)," *ICGA Journal*, vol. 27, no. 4, pp. 217-225, 2004, doi:10.3233/ICG-2004-27404.
**Link:** <https://doi.org/10.3233/ICG-2004-27404> · **Tier:** SUPPORTING

**Summary.** The genuine chess pattern DSL: declarative filters composed over games and positions, with move notation, regular expressions over move sequences, and board transformations including rotations, reflections and shifts. The transformations are what make it a pattern language rather than a position lookup. Co-developed by Costeff and Stiller; current version CQL 6.2, free.

**Depends on.** PGN parsing (SCID code); a declarative filter algebra.

**Results.** In production use by problemists and composition judges. Target users per the paper are "writers, researchers, composers, composition tournament directors, and judges."

**Weaknesses.** Built for finding thematic material, not for stating why a move is good, so it has no notion of a reason or a witness. Bizjak and Guid note it "requires the user to define complex queries in the system-specific language" and that sequential PGN scanning makes it inefficient on large databases. Whether CQL 6.x ships built-in pin/fork/skewer filters is UNVERIFIED.

**Relation to xai-chess.** The existence proof that a compositional chess pattern language is feasible and maintainable, and the clearest demonstration that such a language has never been pointed at explanation.


## 82. Towards Faithful Model Explanation in NLP (survey)

**Citation:** Lyu, Apidianaki, and Callison-Burch, **[title not verified](https://aclanthology.org/2024.cl-2.6/)**, *Computational Linguistics*, vol. 50, no. 2, pp. 657-723, 2024.
**Link:** <https://aclanthology.org/2024.cl-2.6/> · **Tier:** SUPPORTING

**Summary.** Surveys 110+ explanation methods in five categories (similarity-based, model-internal structures, backpropagation-based, counterfactual intervention, self-explanatory) and organizes faithfulness evaluation into axiomatic, predictive-power, robustness, perturbation-based, white-box/ground-truth-referenced, human-perception, and meta-evaluation families.

**Depends on.** The NLP interpretability literature; Jacovi and Goldberg's faithfulness/plausibility distinction [30].

**Results.** Two admissions carry weight: "there is not yet a consistent and formal definition of faithfulness in the community," and "explanations that better align with human perception are often thought to be more faithful, while in fact, they are only more plausible."

**Weaknesses.** Title and author initials unverified in this pass. NLP-centric, with no game or sequential-decision coverage.

**Relation to xai-chess.** Supplies the taxonomy slot the benchmark occupies. The white-box, ground-truth-referenced category is the thinnest in the survey precisely because ground truth is rare, which is the argument for a rule-defined domain in one sentence.


## 83. On Measuring Faithfulness or Self-Consistency (CC-SHAP)

**Citation:** Parcalabescu and Frank, **[title not verified](https://aclanthology.org/2024.acl-long.329/)**, in Proc. ACL, 2024.
**Link:** <https://aclanthology.org/2024.acl-long.329/> · **Tier:** SUPPORTING

**Summary.** Argues that existing self-explanation faithfulness tests do not measure faithfulness at all: they "design special LLM inputs and check whether the LLM returns self-consistent answers," without investigating the correspondence between the explanation and the model's internal processes. The authors decline to call their own CC-SHAP a faithfulness metric on the same grounds.

**Depends on.** The chain-of-thought faithfulness literature [28], [29]; SHAP.

**Results.** A reframing of a body of tests as self-consistency measures, plus CC-SHAP as an explicitly-labelled consistency measure.

**Weaknesses.** Title and author initials unverified in this pass. Confined to language models.

**Relation to xai-chess.** The specific reviewer objection the benchmark must pre-empt, and one it can answer on the merits: a movegen-level intervention changes what the engine can compute, which is categorically stronger than perturbing an input and observing self-consistency. That argument should be made in this paper's vocabulary.


## 84. Chess and Explainable AI

**Citation:** Björnsson, "[Chess and explainable AI](https://content.iospress.com/articles/icga-journal/icg240256)," *ICGA Journal*, vol. 46, no. 2, pp. 67-75, 2024, doi:10.3233/ICG-240256.
**Link:** <https://content.iospress.com/articles/icga-journal/icg240256> · **Tier:** SUPPORTING

**Summary.** A recent peer-reviewed survey arguing that "chess may indeed hold a promise as an admissible domain for explainable AI." Covers interpretability work on Stockfish and Leela, names DecodeChess, and cites the neural commentary line.

**Depends on.** The chess interpretability and explanation literature.

**Results.** Content beyond the abstract is UNVERIFIED — the article is paywalled and no open copy was located.

**Weaknesses.** Unverified beyond the abstract. Survey rather than method.

**Relation to xai-chess.** The natural framing citation and the nearest competing statement of the project's own thesis. It stakes out the claim that chess is a promising XAI substrate; this project operationalises and tests that claim, so it should be positioned against rather than merely cited.


## 85. AI-Generated Game Commentary: A Survey

**Citation:** Zheng, et al., "[From Multimodal Perception to Strategic Reasoning: A Survey on AI-Generated Game Commentary](https://arxiv.org/abs/2506.17294)," arXiv:2506.17294, 2025.
**Link:** <https://arxiv.org/abs/2506.17294> · **Tier:** PERIPHERAL

**Summary.** Surveys AI-generated commentary across board games, sports and esports, with an accompanying datasheet repository covering 45 datasets.

**Depends on.** The game-commentary generation literature.

**Results.** A datasheet repository at `zzzqr.github.io/AIGGC-datasheet/`, accurate on every item independently spot-checked during this sweep.

**Weaknesses.** Per-dataset pages are thin on licensing and availability, listing "Availability: not stated" for most board-game entries. Breadth rather than depth.

**Relation to xai-chess.** Useful for establishing that the commentary task family is broader than chess, and as a cross-check on dataset coverage. Carries no faithfulness or causal content.


## 86. PGN Numeric Annotation Glyphs

**Citation:** S. J. Edwards (coordinator), "Standard: Portable Game Notation Specification and Implementation Guide" (canonical URL not verified), 1994.
**Link:** PGN standard, 1994 · **Tier:** PERIPHERAL

**Summary.** The PGN specification defines Numeric Annotation Glyphs, a standardized language-independent annotation vocabulary: `$` plus an integer 0-255, roughly 140 defined. Codes 1-9 annotate the move just played (`$1` for `!`, `$2` for `?`, `$3` for `!!`, `$4` for `??`); 10-135 describe the position (`$20`/`$21` for a crushing advantage, `$36`-`$39` for initiative); 136-139 cover time pressure.

**Depends on.** Nothing; it is a 1994 interchange standard already supported by every major chess tool.

**Results.** Universally implemented in PGN readers and writers.

**Weaknesses.** Coarse. A NAG says a move is good or that a side has the initiative; it cannot say a knight is pinned to the queen on a named square. It is a grading vocabulary, not a reason vocabulary.

**Relation to xai-chess.** An existing standard output interface that no explanation system in this survey targets. An explainer emitting NAGs alongside prose would be automatically gradeable against a fixed vocabulary and comparable across systems, at essentially no engineering cost.


## 87. Chessmaster 3000 / 4000 — Natural Language Advice

**Citation:** The Software Toolworks, *Chessmaster 3000 (1991) PC Manual*; *Chessmaster 4000 Windows 95 Edition (1995) PC Manual*.
**Link:** <https://archive.org/details/chessmaster-3000-pc-manual> · **Tier:** PERIPHERAL

**Summary.** Commercial prior art that predates the entire academic chess-commentary literature. The Chessmaster 3000 manual documents, under Mentor menu → Advice → Detailed: "Get more detailed advice and analysis, including the suggested move, the predicted following line of play, and the impact of the line of play on material points and position. The analysis is presented in plain English." The user sets the analysis time. By Chessmaster 4000 (1995) this is a first-class command named Natural Language Advice, described as giving "detailed advice and analysis about suggested moves presented in plain English."

**Depends on.** The product's own engine, its principal variation, and its material and positional evaluation terms.

**Results.** Shipped commercially from 1991. No evaluation was ever published.

**Weaknesses.** No mechanism, paper, or patent was ever published, so everything known comes from user manuals. Whether the generation is templated or otherwise structured is not documented, and the software is not readily runnable today for inspection.

**Relation to xai-chess.** Dates the engine-eval-delta-to-prose recipe to 1991, fifteen years before Automated Chess Tutor [70] and thirty-four before CCC [11]. Establishes that the recipe has been shipping to users for thirty-five years without anyone once checking whether its assertions are true of the position, which is the gap the benchmark addresses.
