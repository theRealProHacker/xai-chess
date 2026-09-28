# The corpus in ten clusters

*87 works · 30 core / 46 supporting / 11 peripheral. Companion to the interactive mindmap.
Numbers are the fixed IEEE tags from `PAPERS.md`. Tiers mark distance from the causal question, not quality.*

**The lens.** Every work is read through one question: does it verify that a cited factor is causally
load-bearing for a decision, or does it stop short of that test? Four stop-points recur — correlational
probing, plausibility/simulatability, unverified-edit labels, and behavioural move-match.

---

**1 · Faithfulness & the intervention recipe** (7)
What the word means and how the field agreed to test it. Jacovi & Goldberg [30] separate *faithful* from
*plausible* and supply the definition but no metric — the silence the rest of the cluster fills. Turpin [28]
shows planted biases move answers while the reasoning never admits it; Lanham [29] generalises the
intervention to the trace itself; ICE [17] gives the recipe its full statistical form and finds faithfulness
essentially uncorrelated with plausibility; RFEval [16] reaches the same place from the reasoning-model side.
That several teams independently reinvented neutralize-and-retest makes it shared field property, not
anyone's contribution to claim.

**2 · Grading the graders** (4)
Where the recipe stalls, and the reason this project exists. BonaFide [14] builds verified labels by hand and
still finds the best metric caps at **0.70 AUROC** — the reference bar everything else is measured against.
Causal Diagnosticity [15] manufactures labels more cheaply but never confirms the model used the edit.
The bottleneck is the labels, not the arithmetic. Pálsson & Björnsson [66] is the exception and the nearest
precedent: real causal ground truth in chess, via an Elo gauntlet against a concept-ablated NNUE.

**3 · Simulatability & XAI-eval pitfalls** (11)
The other evaluation tradition — does the explanation help a human predict the model? A different axis from
faithfulness, routinely mistaken for it. ALMANACS [26] is the sharpest result: no method beat the
no-explanation control. Atrey [32] matters disproportionately here — it is the closest non-text causal
precedent, and what it lacks (a separate oracle, the statistical layer) defines what this project adds.

**4 · Inside the network** (19)
What chess models represent and whether they plan. The cluster splits on the survey's own axis.
*Probing (correlational)* — McGrath [1] recovers human concepts in AlphaZero and flags the
pin / can_capture_queen confound that only intervention resolves; Karvonen [3] finds a board state the model
was never given. Decodable, never demonstrably used. *Circuits (causal)* — Jenner [4] patches activations to
show real look-ahead; Sandmann [5] shows internal solutions get overridden before output, which is the
mechanism proving internal faithfulness and decision-level load-bearing genuinely come apart.

**5 · Oracles & human-aligned engines** (11)
The engines that supply decision quality, plus the calibration layer saying which positions are hard for whom.
Ruoss [7] is the live alternative oracle: a searchless value head giving calibrated win probability with no
centipawn conversion. The Maia line [20, 9, 48] is the instrument for stratifying positions by difficulty —
the natural fix for the saturation problem in the results.

**6 · Systems that explain a move** (15)
The explainers a benchmark would actually grade, across three generations: language-model commentary
[11, 12, 21, 22], the datasets under it [65, 80], tutoring and benchmarks [13, 70, 56], and neuro-symbolic
and commercial prior art. Two stand out. Caïssa [54] returns the *witness pieces* for a pin or fork rather
than a boolean — the structure a factor-level intervention needs. Chessmaster [87] shipped natural-language
advice in 1991 with no published method, which is worth knowing before claiming novelty.

**7 · Attribution on the board** (2)
Two papers, deliberately their own cluster: they are the nearest prior art the novelty claim must be
differenced against, and burying them elsewhere would hide them. SHAP-for-chess [19] already removes pieces
and reads the evaluation delta — board-level, no confound-free surface, no causal-faithfulness framing.
SARFA [67] attributes agent actions to specific and relevant features.

**8 · Symbolic & advice-language lineage** (8)
The oldest and least exploited thread. Wilkins [75] and Bratko & Michie [76] treated patterns, plans and
advice as first-class objects in 1980. Bizjak & Guid [71] report dynamic terms beating static ones
0.418 vs 0.252 on motif retrieval — direct evidence that a move's reason is not recoverable from the position
alone. CQL [81] is a query language, not an explanation. And PGN Numeric Annotation Glyphs [86] have been a
standard annotation vocabulary since 1994 that no system in this corpus emits.

**9 · Explaining search (MCTS)** (6)
A parallel explanation literature for tree search, with no faithfulness test attached to any of it.
Included because it is the same problem posed over a different computation.

**10 · Beyond chess** (4)
Whether the substrate argument generalises. It rests on two card-game works [57, 58], neither of which runs
the intervention test. Imperfect information removes the oracle and blurs the counterfactual, so this is a
conjecture, not a result — and porting the surface-agreement test outward is the proper unrun test.

---

*Resolved 2026-08-03: missing URLs for [80] and [86]; previously unverified titles for [67], [77], [82], [83].*
