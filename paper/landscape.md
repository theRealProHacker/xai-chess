# Move-level chess XAI — what exists

A landscape survey of systems that explain a *specific chess move*, in natural language or in a formal representation, and of the datasets behind them.

Compiled 2026-08-01 from six parallel research passes: natural-language commentary, LLM-era and commercial systems, concept and mechanistic interpretability, symbolic and DSL explanation, datasets, and evaluation.

This is a companion to `paper/survey.md`, not a replacement. That document argues a thesis about causal-intervention evaluation. This one inventories what exists.

Every claim traces to a fetched source. Anything that could not be confirmed that way is marked **UNVERIFIED** inline and must not be cited on this document's authority.

---

## 1. Bottom line

**Four systems in fifty years explain a specific move symbolically**: PARADISE (1980), Automated Chess Tutor (2006), CCC (2025), Caïssa AI (2025).

Everything else classifies whole puzzles, analyses the model globally, or attributes to squares without naming a motif.

**No formal grammar of chess motifs exists.** Three artifacts come close, and none denotes "the reason this move is good": CQL queries positions, Bizjak & Guid tokenise motifs for retrieval, `cook.py` is imperative Python booleans.

**Nobody measures whether an explanation is true.** Every generation system is scored on fluency, human preference, LLM-judge rating, or downstream move accuracy. None checks whether an asserted tactical claim holds on the board.

**No benchmark measures causal faithfulness of chess move explanations.** The one chess explanation benchmark that exists grades agreement with a human expert vote — the plausibility axis.

**Causal work and concept work are disjoint populations.** Strong causal machinery explains via squares and future moves. Human concept vocabularies come with no intervention. One 2026 preprint spans both, and it is unreviewed.

---

## 2. Definitions

Three notions are routinely conflated in this literature. The rest of this document depends on keeping them apart.

**Faithfulness** — the explanation reflects the system's actual decision process. Established by intervention or by ground truth, never by asking a human.

**Plausibility** — the explanation convinces a human, or matches a human-written reference. This is what almost every chess metric measures.

**Simulatability** — a human can predict the agent's behaviour given the explanation. A third axis, and not a proxy for either of the others.

Jacovi & Goldberg (ACL 2020) supply the canonical statement and three guidelines that indict most of the work below: faithfulness evaluation must not use human judgement of interpretation quality, human-provided gold labels, or user performance.

Chen et al. (ICML 2024) settle the empirical question. Counterfactual-simulation precision correlates with human plausibility ratings at **r = +0.012**, against inter-annotator plausibility agreement of +0.388. The axes are orthogonal.

---

## 3. The grounding problem

Before asking whether an explanation is *true*, you must know what it is *about*. Chess work has not addressed this. Shogi work solved it in 2015.

**Kameko, Mori & Tsuruoka**, IEEE CIG 2015, pp. 177–184. Their insight: a commentary sentence is usually not about the current position. It refers to a hypothetical continuation or a past line.

So the comment's move expressions must first be grounded to a *node in the game tree*. They enumerate candidate trees over the current position plus the previous three moves, inserting `Pass` for lines that are illegal but commented.

Selection uses engine evaluations, `Ev_T(t) = Σ_m Ev_M(m) − max_m Ev_M(m) + B_P + B_B`. The heuristic: experts comment on bad moves *by showing the right move*, so the correct tree is the one **not** containing a sequence of bad moves.

Generation is a 3-layer perceptron predicting a characteristic-word vector from the engine's evaluation features, then a log-linear language model conditioned on position plus predicted words.

Results: at least one candidate tree for 44,166 of 54,084 positions (81.2%); of 100 hand-checked, the correct tree for 79. There is **no BLEU and no human study** — grounding accuracy and qualitative figures only.

Their own error analysis is the load-bearing part. About 20% of move-expression comments produce no candidates and about 20% select the wrong tree, and both "directly affect the performance of the trained model."

Every chess system assumes the comment refers to the current position, and several then blame that assumption for hallucination. This sits upstream of faithfulness: you cannot check a claim against a position until you know which position.

Predecessor: **Kaneko**, IPSJ Journal 53(11):2525–2532, 2012 (Japanese), template-based over search output. Attested only as a reference in the Kameko PDF, so bibliography-grade. Kameko's critique — template variety "is much smaller than that of human experts."

**Go commentary generation: nothing found.** The nearest work (Tomlin, He & Klein, ACL 2022, arXiv:2204.07531) probes AlphaGo Zero for domain terms such as *ko* and *atari*. That is interpretation, not generation.

---

## 4. Systems that explain a specific move

### 4a. Natural language

| System | Yr | Engine in loop? | Symbolic intermediate | Eval of the explanation |
|---|---|---|---|---|
| Jhamtani et al., GameKnot commentary | 2018 | No (board features) | Hand-coded board/attack features | BLEU + human validity |
| Butner, ChessCoach | 2021 | Yes (own AZ-style) | No | Author: "often wrong" |
| Lee, Wu, Dinan & Lewis | 2022 | Yes (Leela) | 5 discrete control tags | Human A/B, perplexity |
| Kim et al., **CCC** | 2025 | Yes (Lc0 probes + SF8) | 20 scored concepts | GCC-Eval + 5 raters |
| Tang et al., **C1** | 2026 | Yes (SF depth-24 PV) | PV trace, not logic | **None** — move accuracy only |
| Cui, Ling & Ng | 2026 | Yes (SF16 strategy tree) | Strategy tree (JSON) | Downstream puzzle-playing utility |
| DecodeChess (commercial) | 2019–23 | Yes (SF search verifies) | **Verified concept predicates** | None published |
| ChessBase Fritz "Full analysis" | 1996– | Yes | Tactical pattern rules | 3× ICCA Herschberg Award |

The published field is smaller than it looks. A DBLP sweep plus enumeration of all 46 papers citing Jhamtani puts the complete list of published game-commentary generators in 1990–2022 at six.

Those six: Liao & Chang 1990 (Othello), Sadikov et al. 2006, Guid et al. 2008, Jhamtani et al. 2018, Zang et al. 2019, Lee et al. 2022. The commercial systems in §4b predate all of them and published nothing.

Worth stating if you build on Lee et al.'s tag scheme: it does not appear in DBLP at all, no venue was found, and there is no code or data release. It is arXiv-only.

**Lee et al. (arXiv:2212.08195)** is the only system with an explicit symbolic control interface. BART conditioned on piece-square tokens, PGN, an attack list (`White R_a1$P_a2`), and five control tags supplied by Leela at inference.

Its weak point is upstream: the tag extractors score only 67–73% F1, so the signal linking board facts to text is itself noisy.

**CCC (NAACL 2025)** reads 20 concept vectors from LeelaChessZero T78 layer 40 via linear SVMs, trained on 200K Lichess positions labelled by Stockfish 8 eval terms.

Concepts are ranked by pre/post-move score delta and the top ones handed to GPT-4o. Its own human numbers are the field's clearest diagnosis: **fluency 0.91, correctness 0.60** across 5 raters (mean 1776 rapid).

Concept guidance buys style, not truth. Inter-rater Fleiss κ is weak away from correctness, 0.18–0.24. Code: `github.com/ml-postech/concept-guided-chess-commentary`.

**C1 (arXiv:2603.20510)** names the exact tension and resolves it with a prompt rather than a verifier. From §3.2.2: too faithful and the model parrots the moves; too exploratory and it hallucinates and loses grounding.

Reward and metric are both final-move exact match. The explanations are never evaluated.

**Cui, Ling & Ng (arXiv:2607.11486)** explicitly reject post-hoc justification and evaluate by Tree-Expanded Puzzle Playing. A separate player sees only the description and plays the puzzle against adversarially sampled replies.

Scoring is the resulting Stockfish evaluation. No LLM judge anywhere in the metric. Findings: main-line-only evaluation is insufficient, LLM judges are unreliable proxies for humans, and pure concept-based description yields little improvement.

One further result matters for anyone building a DSL: verbalising the strategy tree is *lossy*. The raw JSON outperforms every prose rendering of it.

### 4b. Commercial annotators

Chessprogramming.org was 503 throughout this survey, so these were resolved from primary sources — engine source, archived manuals, vendor help pages.

| Product | Verbal explanation? | What it emits | Source |
|---|---|---|---|
| **Chessmaster 3000** (1991) | **Yes** | "Natural language advice in plain, clear English" | 1991 manual (archive.org) |
| **Chessmaster 4000** (1995) | **Yes** | Named command "Natural Language Advice" | 1995 manual (archive.org) |
| ChessBase Fritz "Full analysis" | Yes | "variations, textual commentary… opening references" | ChessBase help |
| Crafty `annotate` | **No** | PGN + engine lines + Informant symbols | `annotate.c` source |
| ChessBase "Assisted Analysis" | **No** | Colour-coded board overlay | ChessBase help |
| Shredder Chess Tutor | **No** | Arrows and symbols on the board | Shredder site |
| HIARCS Chess Explorer | **No** (apparently) | Notation, symbols, statistics | HIARCS site |
| Aquarium / IDeA | Unresolved | Output format not publicly documented | — |

**Chessmaster shipped a natural-language move explainer in 1991, using the recipe everyone still uses.** From the Chessmaster 3000 manual, Mentor → Advice → Detailed:

> "Get more detailed advice and analysis, including the suggested move, the predicted following line of play, and the impact of the line of play on material points and position. **The analysis is presented in plain English.**"

That is engine PV plus material and positional deltas, rendered as prose, with user-set think time. Structurally identical to Automated Chess Tutor (2006) and CCC (2025).

By Chessmaster 4000 (1995) it is a first-class named command. No mechanism was ever published for either, so the lineage is invisible to the academic literature that reinvented it.

**Crafty's `annotate` is not a verbal annotator** — a persistent misconception, settled here from source. It emits PGN plus engine variations and scores, gated by a `margin` threshold in pawns.

Output is `.can`, HTML with board bitmaps, or LaTeX. The complete set of English strings in `annotate.c` is the four status messages: which side is being annotated, for which player, and at what margin.

The only qualitative judgements are LaTeX macros from the `chess` package — `\wbetter`, `\bupperhand`, `\wdecisive`, `\equal` — that is, Informant symbols. Zero natural language.

This is why Sadikov et al. wrapped Crafty in 2006: Crafty itself does not do this.

**DecodeChess is the architectural outlier.** Its mechanism, in its own words: "Every suggested chess concept is verified to being relevant and important to the decoded position."

An open file is asserted "only if it can be proved to affect the future course of the game," using "vast searching abilities of the search engine (Stockfish in our case)."

Propose, then verify against search, then assert. Output is slot-filled templates and there is no LLM anywhere. It is the closest thing to a causal-faithfulness design that has ever shipped.

There is **no paper and no patent**. Google Patents returns zero for assignee Decodea and for inventors Ofer Shamai and Zeev Fine. Everything known comes from marketing copy that happens to be technically specific.

The product is dormant; the blog archive ends April 2023.

**ChessBase "Assisted Analysis" is a colour overlay, not text**: green for a very good move through red for a very bad one. Its stated design goal is to withhold evaluations so the user calculates independently.

### 4c. Symbolic, logical, and DSL

| System | Yr | Formal representation | Origin | Available |
|---|---|---|---|---|
| Wilkins, **PARADISE** | 1980 | Production rules; conditions match patterns, actions post concepts into a database; concepts → plans → guided search | Hand | Papers only |
| Michie & Bratko, **Advice Language** AL0/AL1/AL3 | 1976–82 | Advice = goal + constraints, compiled to a forcing tree | Hand | Papers |
| Bratko KRK strategy (re-formalised 2019) | 1978→2019 | 7 named rules over `room`, `critical square`, `rook exposed`, `rook divides`, `L-pattern` | Hand | **Open (LMCS)** |
| Gadwal/Greer/McCalla, UMRAO | 1990/93 | Extension of Michie's advice language → strategy graph | Hand | Paper |
| **Automated Chess Tutor** | 2006 | Eval-feature vector difference across the PV, plus an IF-THEN composition layer | Hand+ | Paper |
| ABML line (Možina/Guid/Sadikov/Bratko) | 2007–13 | Arguments attached to individual examples; ABCN2 unordered probabilistic rules | Learned from expert arguments | Papers |
| Guid et al., KBNK from tablebase | 2009–13 | ABML + specialised minimax → textbook instructions | Semi-auto | Papers |
| **Caïssa AI** | 2025 | Prolog predicates returning **witness pieces** | Hand | **GPL-2.0** |

**PARADISE** (Wilkins, *Artificial Intelligence* 14(2):165–203, 1980) encodes knowledge as production rules whose "actions post concepts in the data base while the conditions match patterns in the chess position and data base."

It discovers plans during static analysis and searches only to *confirm* the plan. Search trees run to tens and hundreds of nodes, with combinations to 19 ply.

The justification is therefore the concept chain that produced the plan, not a post-hoc score. This is the property later systems lost.

**UNVERIFIED:** the internal syntax of a PARADISE rule and its concept vocabulary. *AI* 14(2) and SRI Technical Note 509 are both unreachable online. Also unverified: the widely repeated claim that the test set was 100 positions from Reinfeld's *Win at Chess*.

**Advice Language.** The exact AL clause syntax could not be confirmed — every primary text is paywalled or pre-digital. **Do not assert the syntax.**

What is freely readable is the resulting KRK strategy, re-formalised by Janičić, Marić & Maliković (*Logical Methods in Computer Science* 15(1:34), 2019).

Their auxiliary predicates are credited to Bratko: `room` (half-perimeter of the rectangle confining the black king, 15 if unconfined), `critical square`, `rook exposed`, `rook divides`, `L-pattern`, `kings on same edge`.

The strategy is seven prioritised rules — `ImmediateMate`, `ReadyToMate`, `Squeeze`, `Approach`, `KeepRoom`, `RookHome`, `RookSafe` — with coverage counts over all 175,168 white-to-move KRK positions.

That paper is the right citation for AL's *content*, though not for its syntax.

**Automated Chess Tutor** (Sadikov, Možina, Guid, Krivec & Bratko, CG 2006, pp. 13–25) is CCC's direct ancestor, twenty years early.

Crafty returns the principal variation; the system takes the vector difference between the eval-feature vector now and at the end of that PV. Positive components are goals achieved, negative ones are weaknesses created.

In their words: "The goals in our schema are simply the evaluation function's features." An expert-system IF-THEN layer composes basics into human-level features and suppresses redundant complements.

**No evaluation is reported.** CCC's mechanism is structurally identical with learned probes substituted for hand-written eval terms.

**Caïssa AI** (Soliman & Ehab, KI 2025, LNCS 15956, pp. 148–160) exposes seven per-move predicates in `reason.pl` — `fork_reason/5`, `absolute_pin_reason`, `relative_pin_reason`, `skewed_reason`, and three more.

Each takes a (from, to) move and **returns the witness pieces**. A LangGraph module splits LLM commentary into atomic claims and checks each against the Prolog knowledge base.

**Evaluation UNVERIFIED** — paywalled, no open-access copy, no evaluation artifacts in the repo. Its reference [18] also cites arXiv:2306.12948 for ChessGPT, which is a number-theory paper; the real one is arXiv:2306.09200.

---

## 5. Motif vocabularies and the witness gap

### The witness gap

Lichess's shipped detectors are the de-facto motif DSL, and they are imperative Python booleans that discard their own evidence.

Every detector is a `(puzzle) -> bool` querying `python-chess` primitives: `board.pin`, `board.attacks`, `board.attackers`, `board.checkers`, `SquareSet.between`, plus helpers such as `util.is_hanging` and `material_diff`.

Two structural consequences follow.

**They classify a puzzle line, not a move.** `fork` returns `True` if *any* solver move forks. The output is a puzzle tag, never "this move is good because X."

**They discard the witness.** `fork()` counts `nb > 1` attacked pieces and throws away which ones. Contrast Caïssa's `fork_reason(Piece, Color, Pos, NextPos, ListOfOpponents)`.

Only `zugzwang` is engine-grounded, and it is a genuine counterfactual: push a null move, re-analyse at depth 30, and fire if `win_chances(score) < win_chances(rev_score) - 0.3`.

That null-move test is the only counterfactual construct in the entire shipped Lichess vocabulary.

The natural-language theme definitions live in `lila/translation/source/puzzleTheme.xml` and are maintained separately from the Python predicates, with no formal link. That is a soundness hazard for anyone treating the tags as ground truth.

### Pattern languages that are not explanations

**CQL** (Costeff, *ICGA Journal* 27(4):217–225, 2004; CQL 6.2, free) is the real chess pattern DSL: declarative filters over PGN with move-sequence regexes and board transformations.

The transformations — rotations, reflections, shifts — are what make it a pattern language rather than position lookup. But it was built for *finding* thematic material, not explaining a move.

**UNVERIFIED:** whether CQL 6.x ships built-in `pin`/`fork`/`skewer` filters. The documentation page fetched does not list them.

**Bizjak & Guid**, ACG 2021, LNCS 13262, pp. 131–141, is the closest thing to a term-level motif DSL.

*Static* terms cover placement (`Ra1`), distance-decayed reachability (`Rb1|0.89`), connectivity — attacks `B>pg7`, defends `R<Kh1`, X-ray `R=pa7` — and pawn structure.

*Dynamic* terms describe the solution line: `?px`, `?ox`, `?+`, `?=`, `?S` for sacrifice, `?#`, plus per-move markers. They are deliberately position-independent so motifs generalise across the board.

Retrieval is Lucene with BM25. On 400 expert-paired CT-ART puzzles, top-1 scores are static-only 0.252, dynamic-only 0.418, combined 0.481; top-10 are 0.433, 0.761, 0.814.

> **Dynamic beats static by a wide margin. That is the strongest published evidence that the reason for a move is not recoverable from the position alone.**

### An unused output interface

Chess already has a standardised, language-independent symbolic annotation vocabulary: Numeric Annotation Glyphs, from Edwards' PGN specification (1994).

A NAG is `$` plus an integer 0–255, with roughly 140 defined. Codes 1–9 annotate the move played (`$1` for `!`, `$2` for `?`, `$4` for `??`); 10–135 describe the position; 136–139 cover time pressure.

**No system in this literature targets NAGs as an output interface.** A move explainer emitting `$1`/`$21`/`$36` alongside prose would be gradeable against an existing standard and comparable across systems.

### Engine grounding has regressed

Stockfish 15's `evaluate.cpp` had a `Trace` namespace with named terms — material, imbalance, mobility, threat, passed, space, winnable — printed as an additive MG/EG table by colour.

**That is gone.** Master's `nnue_misc.cpp::trace` prints only a bucket-level `Material (PSQT) | Positional (Layers) | Total` table. The per-square table its code comment still promises is no longer emitted.

Modern Stockfish has no term-level explanation at all. This is why both McGrath (2022) and Kim (2025) reach back to **Stockfish 8** for a named concept vocabulary.

The Stockfish Evaluation Guide (`hxim.github.io/Stockfish-Evaluation-Guide`) preserves a JavaScript reimplementation, last updated August 2020.

---

## 6. Concept-based interpretability

Most of this work is global. It explains the model, not the move.

| Work | Unit | Per-move? | Causal validation |
|---|---|---|---|
| McGrath et al., PNAS 2022 | sparse linear probe / CAV | **No** — global | **None** (explicitly declined) |
| Schut et al., PNAS 2025 | sparse concept vector, L1-min under ranking constraints | Obliquely (prototypes) | No activation intervention |
| Pálsson & Björnsson, IJCAI 2023 | probes on **Stockfish NNUE** | **No** (states this outright) | None |
| Karvonen, COLM 2024 | linear probes on chess-GPT | Yes (behavioural) | **Yes** — 41% → 92% legal |
| Karvonen et al., NeurIPS 2024 | SAE features | No | **None** — F1 vs ground truth |
| Jenner et al., NeurIPS 2024 | residual activations, attention heads | **Yes** | **Yes** — patching + ablation |
| Lin et al., arXiv 2604.10158 | transcoders + Lorsa on LC0 BT4 | **Yes** — per-move pathways | **Yes** — steering, zero-ablation |
| Chessformer, arXiv 2605.19091 | cross-layer transcoder | No | **None** (authors flag it) |

**McGrath et al. §4.5 is the passage to quote.** They argue intervention is the correct validation, then decline it, on two grounds.

First, chess concepts are mutually entangled — you cannot zero `can_capture_queen` without setting `in_check`. Second, internal edits risk pushing the network off-distribution.

So the field's highest-profile "chess concepts in a network" result rests entirely on probe accuracy.

**Jenner et al. is the methodological benchmark for causal work in chess.** Patching the 3rd-move target square at layer 10 costs **1.88 ± 0.04** log-odds, against **0.55 ± 0.01** for the maximum over all 61 other squares.

Zero-ablating a *single* attention entry in L12H12 reduces log-odds by more than 1.5 in over 10% of puzzles — a larger effect than ablating all 4,095 other entries in that head combined.

**Sandmann et al. (arXiv:2508.21380) is the sharpest negative result in the field, and nobody has built on it.** Leela computes the correct solution mid-network, then overrides it at the output.

They call these "forgotten puzzles," and steering against the learned prior recovers 61.7% of them. A probe fires and the behaviour does not follow. That is the empirical case against probe-accuracy-as-explanation.

Two reproducibility cliffs. AlphaZero's weights and games are proprietary in both PNAS papers, so neither the probes nor the concept vectors can be re-derived.

Consequently every causal result in the field runs on Leela or chess-GPT, never on AlphaZero.

**TCAV proper has never been run on chess.** Both chess "CAV" papers borrow the vector construction and drop the directional-derivative sensitivity score. An unclaimed and cheap baseline.

---

## 7. Datasets

| Dataset | Size | Explanation side | Move-aligned | License |
|---|---|---|---|---|
| **GameKnot commentary** (Jhamtani 2018) | 11,578 games / **298,008** pairs | free text + 6 category tags | **yes** | none stated |
| ↳ *mirror*: ChessGPT `annotated_pgn_free.tar.gz` | 12,770 GameKnot PGNs | inline `{}` prose | yes | apache-2.0 declared; upstream grey |
| Waterhorse/chess_data (ChessGPT full) | 37.6 GB / 461 files | mixed | partly | apache-2.0 (mixed upstream) |
| **Lichess puzzles** | **6,057,356** | **73 motif tags** | position-level | **CC0-1.0** |
| Lichess standard games | ~7.14B | `%eval`, NAGs | move-level eval | **CC0-1.0** |
| Lichess eval DB | **394,669,566** positions | SF cp/mate + PVs | position-level | **CC0-1.0** |
| **ChessBench** (Ruoss 2024) | 10M games / **15B** annotations | scalar win-prob — **no language** | move-level | CC-BY-4.0 |
| MATE (Wang 2025) | ~1M positions | **templated** phrases | move-level | MIT |
| ChessQA | 3,500 items (Semantic = 400) | structured gold answers | position-level | MIT |
| Icannos/chess_studies | 6,110 rows | PGN prose + arrows | yes | CC0 self-declared |
| Shogi Commentary Corpus | 4,327 games / **218,615** JP comments | free text + **factuality layer** | yes | not stated |

**The GameKnot corpus is downloadable after all.** Jhamtani's repo ships only a Python-2.7 and PyQt4 crawler, expects you to re-scrape gameknot.com, and its `license` field is `null`.

ChessGPT's release mirrors it: `chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz`, 57.6 MB, 12,770 GameKnot PGNs with move-aligned `{}` prose. Rights are grey either way — forum comments were never licensed by their authors.

**Correction worth recording:** Zang/Yu/Wan 2019 released *no* commentary-dataset extension. They use Jhamtani's data unchanged.

What they added is an *engine*-training corpus: FICS games with both players rated 2000+, giving 36M single-move records. They also drop the *General* category as "irrelevant to game analysis," running on 5 of 6.

**Lichess themes number 73.** That is 75 `categorized` entries in `PuzzleTheme.scala`, minus `mate` duplicated across two groups, minus the `mix` UI pseudo-theme. One hidden theme, `checkFirst`, is never exported.

The generator is `ornicar/lichess-puzzler`, `tagger/cook.py`, AGPL-3.0. Frequency skew is roughly 2,500:1 — `short` at 50.5% and `endgame` at 50.0% against `underPromotion` at 0.02%.

23 themes are rule-derived `staticThemes`; the rest are vote-refined and carry crowd noise. `cook.py` still emits an `overloading` tag that appears **zero** times in the live export.

**Label provenance is worse than it looks.** Jhamtani's category labels are SVM output on 297K of 298K rows — only 1K comments were hand-annotated, by two annotators.

The "Comparative" category falls back to a hand-written rule, the presence of the word "better." Roughly 30% of comments are "General" with no chess content, and 23% are five words or fewer.

MATE's "annotated by chess experts" means experts wrote about 20 template phrases per category, which were then applied by rule. It is not 1M hand-written explanations, and strategy classes are heavily imbalanced.

**ChessBench contains no natural language at all.** Its 15B "annotations" are Stockfish win-probabilities. It is an oracle, not an explanation corpus.

**What does not exist:**

No dataset pairs a position with a *verified causal* explanation.

No corpus of plausible-but-wrong chess explanations exists, so there are no negatives for training or evaluating a discriminator.

No human-adjudicated motif labels on arbitrary positions with reported inter-annotator agreement. Lichess themes are `cook.py` output on puzzles; Jhamtani's are SVM output.

No cleanly redistributable expert-prose corpus. GameKnot is grey, ChessBase Mega and NIC are proprietary, and Lichess studies' prose is user copyright even though the moves are CC0.

No move-aligned explanations at master strength under a permissive license, and no multimodal board-image-to-explanation pairing.

---

## 8. Evaluation and the faithfulness gap

**Is there any benchmark measuring causal faithfulness of chess move explanations? No.**

Every chess explanation artifact grades plausibility or chess-correctness: Jhamtani (BLEU plus validity judgements), Zang (BLEU/METEOR plus Likert), CCC's GCC-Eval, ChessQA Semantic, SARFA.

GCC-Eval deserves a note. It is an LLM judge *validated by correlation with human ratings*, which under Jacovi & Goldberg's guidelines makes it definitionally a plausibility metric, however much better than BLEU.

The word "faithful" appears in none of Jhamtani, Kim et al., or ChessQA.

Every causal intervention in chess targets internals, the board, or the training data — **never a stated reason**. No paper takes a natural-language chess explanation, intervenes on the factor it names, and checks whether the move changes.

### Nearest things that exist

**1. Pálsson & Björnsson, ECAI 2024**, FAIA vol. 392, pp. 874–881, DOI `10.3233/FAIA240574`. The closest precedent, and it builds a genuine causal ground truth.

Method: retrain the Stockfish NNUE with the concept's positions *skipped from training*, via a patched `make_skip_predicate` in `training_data_loader.cpp`, so minibatch size and hyperparameters are unchanged.

Then play the ablated net against the unmodified one — 3,000 games each at depths 14, 15 and 16, so 9,000 per agent — and convert the strength difference to Elo.

Probing methods are scored by Pearson correlation against that Elo drop: **0.69–0.73** for raw probe accuracy, rising to **0.83–0.92** for the differential between probes on the regular and concept-removed nets.

The published abstract states that interpreting linear-probe accuracy as concept importance is "somewhat unreliable," and proposes amnesic-style techniques instead.

It ablates training data rather than rules, and grades probes rather than stated reasons. But the gauntlet design is directly reusable.

**2. Atrey, Clary & Jensen, ICLR 2020** (arXiv:1912.05743). The falsification design, in Atari.

From 90 papers they extract 46 claims across 11 papers that read saliency as evidence about agent behaviour. **87% reason backwards from the saliency pattern to a hypothesis, and only 3 of 46 carry additional direct experimental evidence.**

Their key move is shifting the do-operator from pixels to game state. Pixel interventions "change the image in a way that is inconsistent with the generative process F"; latent-state interventions leave that process intact.

Three intervention classes — distortion, semantics-preserving, fat-hand — plus an invariance test: a static learned representation should give saliency invariant under semantics-preserving interventions. All three case studies come back negative.

Their verdict: "saliency maps cannot be trusted to reflect causal relationships between semantic concepts and agent behavior."

**3. Juozapaitis et al., IJCAI-19 XAI Workshop.** The only verified ground-truth explanation evaluation in RL, in a gridworld "small enough to solve for the ground truth decomposed Q-values."

It catches HRA producing values that do not reflect its own policy — a precedent for the claim that interpretable-by-construction is not automatically faithful.

**4–5.** Karvonen COLM 2024 (41% → 92% legal-move rate under a probe-directed residual edit) and Jenner et al. NeurIPS 2024 (1.88 vs 0.55 log-odds).

**6. SARFA, Puri et al., ICLR 2020** (arXiv:1912.12191) — the chess explanation benchmark that exists, on the wrong axis.

Its Chess Saliency Dataset is 100 Lichess puzzles with pieces marked relevant by three experts rated over 2200, majority vote as label, methods scored by ROC-AUC. SARFA reaches 0.92.

The human study runs N=40 at ELO 1600–2000 over 15 puzzles: solve accuracy rises from 56.67% with no saliency to 72.41% with SARFA.

Huber et al. name the problem in print: this "does not measure the saliency maps' fidelity to the agent's reasoning, but whether the saliency maps coincide with human reasoning."

### Two results the field should cite more

**Das & Chernova, IUI 2020** (arXiv:2002.04202), N=60, four conditions. The rationale generated *faithfully* from Stockfish's utility function produced no significant gain.

Only the hand-augmented variant — adding criteria the utility function does not represent, such as capture-next-move and mate-in-three — beat the baseline. Faithful does not imply useful, demonstrated in chess.

It is also the only chess paper cited in Lai et al.'s survey of 100+ AI-assisted decision-making studies.

**Shymanski et al. (arXiv:2511.03730)** ran a placebo control in chess, with chess players. Satisfaction ratings could not distinguish an actionable explanation from a vacuous one (p=0.851) while performance diverged (p=0.0236).

Any chess explanation evaluation resting on human judgement inherits this failure.

### The objection to pre-empt

**Parcalabescu & Frank, ACL 2024**: existing self-explanation faithfulness tests "do not investigate the correspondence between the LLM's explanation and its internal processes."

Instead they "design special LLM inputs and check whether the LLM returns self-consistent answers." A movegen-level intervention changes what the engine *can compute*, which is materially stronger than an input edit.

That argument should be made in their vocabulary, not around it.

Two further framings are worth reusing. Jacovi & Goldberg's graded-over-input-subspaces argument — "strictly faithful interpretation is a 'unicorn'" — makes a restricted-scope faithfulness score principled rather than a compromise.

And Lyu, Apidianaki & Callison-Burch (*Computational Linguistics* 50(2):657–723, 2024) have a white-box, ground-truth-referenced evaluation category that is the thinnest in their survey precisely because ground truth is rare. That is the slot a rule-defined domain fills.

---

## 9. The recognition/computation dissociation

ChessQA (arXiv:2510.23948) scores 23 models across five categories. **Semantic ranges 80% → 37% while Short Tactics ranges 77% → 0%.**

Mid-tier models often sit at 65–75% on Semantic and 15–25% on Short Tactics. They recognise plausible-sounding commentary far better than they can compute the tactics that commentary asserts.

Lomasov et al. (arXiv:2510.26025) corroborate from the inside. Named human concepts decode at up to 85% in early layers but drop to **50–65% in the deep layers that drive performance**.

A further 10–20% degrades on Chess960, once opening theory is removed. The CCC paradigm's grounding substrate is weakest exactly where the model is strongest.

Fluent, plausible, and tactically wrong is the modal failure — and it is precisely what preference-based and LLM-judge evaluation cannot see.

---

## 10. Delta vs `PAPERS.md`

Verified additions, absent from the current 64:

- Jhamtani et al., ACL 2018 — the foundational commentary dataset. Its absence is the biggest hole.
- Feng et al., ChessGPT, NeurIPS 2023 D&B, arXiv:2306.09200.
- Pálsson & Björnsson, ECAI 2024, FAIA 392:874–881 — **highest-value addition**.
- Pálsson & Björnsson, IJCAI 2023, pp. 4864–4872 — probes on Stockfish NNUE.
- Karvonen et al., NeurIPS 2024, arXiv:2408.00113 — SAE board-game benchmark.
- Puri et al., SARFA, ICLR 2020 — the only existing chess explanation benchmark.
- Das & Chernova, IUI 2020 — faithful does not imply useful.
- Wilkins, PARADISE, *AI* 14(2):165–203, 1980.
- Michie & Bratko Advice Language cluster, plus the Janičić/Marić/Maliković LMCS 2019 re-formalisation.
- Sadikov et al., Automated Chess Tutor, CG 2006 — CCC's direct ancestor.
- Možina et al., ABML, *AI* 171(10–15):922–937, 2007, and the chess applications (ECAI 2008, CG 2008, ACG 2009, ITS 2012).
- Costeff, CQL, *ICGA Journal* 27(4):217–225, 2004.
- Bizjak & Guid, ACG 2021.
- Cui, Ling & Ng, arXiv:2607.11486, 2026.
- Atrey et al., ICLR 2020, arXiv:1912.05743.
- Parcalabescu & Frank, ACL 2024; Lyu et al., *Computational Linguistics* 50(2), 2024.
- Kameko, Mori & Tsuruoka, IEEE CIG 2015 — the grounding problem.
- Edwards, PGN Specification, 1994 — NAGs.
- Zheng et al., arXiv:2506.17294 — AI-generated game commentary survey with a datasheet repository.
- Björnsson, "Chess and explainable AI," *ICGA Journal* 46(2):67–75, 2024 — the natural framing citation. Full content **UNVERIFIED** (paywalled).

Deliberately **excluded**: chess2vec (arXiv:2011.01014) shares no task, dataset, or evaluation with this literature.

Also excluded as a generation system: SentiMATE (arXiv:1907.08321) runs sentiment analysis *over* commentary to learn a move-evaluation function. The direction is text → chess. Cite it only as evidence that commentary carries recoverable move-quality signal.

Already present and unchanged: McGrath, Schut, Karvonen COLM, Jenner, Sandmann, Lin, Ruoss, Chessformer, Maia-2, Kim/CCC, Tang/C1, ChessQA, Lee, Zang, Spinnato, Caïssa, Guid CSEDU 2013.

---

## 11. Open ground

**Rule-level intervention is unoccupied.** Nothing in this survey modifies the rules of the game to test an explanation.

The only counterfactuals anywhere are Jenner's single-square piece corruptions, SHAP and SARFA piece removal, Lichess's null-move zugzwang test, and Pálsson & Björnsson's training-data ablation. Chess960 is used as a memorisation control, not an explanation probe.

**No witness-returning, validated motif detector exists.** Lichess returns `bool`; Caïssa returns witnesses but is unevaluated.

**No compositional motif language exists.** CQL queries, Bizjak & Guid tokenise, `cook.py` is imperative. Nobody has written a typed, compositional language whose expressions denote a move's reason.

**No concept-bottleneck model for chess** — a confirmed negative. The nearest neighbours are RL-general: SCoBots (NeurIPS 2024) and LICORICE (arXiv:2407.15786).

**Argumentation is applied to *learning* chess concepts, never to *justifying* a move.** ABML uses arguments to constrain rule induction.

Epstein's FORR (*Cognitive Science* 18(3):479–511, 1994), with multiple disagreeing Advisors collaborating per decision, is the only per-decision reasons architecture found — and it is game-general.

**No ASP or Prolog chess-explanation work between AL3 (1982) and Caïssa (2025)** could be found. A forty-year gap.

**No simulatability measurement exists in chess.** No study gives a human a chess explanation and measures their ability to predict the engine's move against a no-explanation control.

Jhamtani's Q2 — can the move be inferred from the commentary? — is the nearest, and it asks about the move, not the model.

**Comment-to-game-tree grounding is unsolved in chess**, per §3. It is upstream of the faithfulness question.

**NAGs are an unused output target.** An explainer emitting them would be gradeable against an existing standard vocabulary and comparable across systems.

### Primary-source holes worth a library request

PARADISE's production-rule syntax and concept vocabulary. *AI* 14(2) and SRI Technical Note 509 are both unreachable online.

The exact AL0/AL1 advice-clause syntax. Every primary text is paywalled or pre-digital; LMCS 2019 substitutes for AL's content, not its syntax.

Michie's "A theory of advice" (*Machine Intelligence* 8) versus "King and rook against king" (*Advances in Computer Chess 1*) — both cited as pp. 30–59, so one attribution is garbled.

Michie's original 1976 *Computer Bulletin* AL0 note. No bibliographic record was found at all.

Možina et al.'s bad-bishop worked example. `ailab.si/matej/bad_bishop.html` now returns 404.
