# Phase 6b — Evaluator paper-visible scoring, round 2 (re-review)

Contract `academic_paper/evaluator_full/v1`, baseline v3.6.6, round 2 of 2 (last allowed). Draft: `paper.md` as rebuilt after `19_revision1_response.md`. Scored against the same `<phase6a_output>` Scoring Plan (`15_phase6a.md`). Writer `<phase4a_output>` (`13_phase4a.md`) rechecked: writer D2 (own-result claims point to a table) now holds throughout; writer D7 (terms used consistently) holds for family/group, with one residual label (N6).

Per the re-review rule, this round checks only the 18 issues of `18_phase6b.md` and material added in revision 1 (M1a/b/c, numeric thresholds, pin-edit construction, M8∘M2 round-trip remainder, M4 whitening, cost arithmetic).

Verification done this round:
- Table 2 and Figure 1 recomputed again from `lc0-probe/results_square_decoder.json`. All 55 printed values match.
- Table 1 row "value head applied to decoded L9 tokens" checked against `results_value_via_head.json`: dense2 0.815 vs 1.535, dense1 0.878 vs 1.343, n = 20,000 (3,960 test). Note on 17,831 bits checked against `results_value_decoder_mlp_n151614.json`.
- $m=28$ recomputed: the probe result files list 65 group names with 28 distinct top-level prefixes.
- Table 5 column sums still 356 and 331. 12 and 13 non-zero families confirmed.
- Cost arithmetic recomputed: $26\times(1.0+0.8+0.5+0.2+0.1)\times1{,}000\times10=6.76\times10^5$ pass-equivalents, which takes 62.6 min at 180/s. A Cholesky factorisation of dimension 14,629 costs $n^3/3\approx1.04\times10^{12}$ flops. Tapping 60,000 positions takes 5.6 min. `square_decoder.py` builds the Gram matrix from a sparse design and solves with `assume_a="pos"` (Cholesky), as the appendix states.
- M4 whitening derivation, M6 completeness, antisymmetry and orientation, M8∘M2 centring cancellation, and §9.3 coverage of M2 checked by hand.
- Pin-edit rules applied to the worked example. Pinner-off candidates for Bb5: a4 is on the pin line and is excluded; a6, c4 and d3 remain. Under a1 = 0 indexing, c4 (26) beats a6 (40). Interposition on d7: the minor-piece rule picks …Bd7 over …Qd7.
- Figure 1 re-inspected: (b) now annotates L2 at 0.76.

## Dimension Scores

### D1: originality

**pass.** The introduction names the gap. The contribution list now reads "an assembly of existing pieces for this system, not new theory", and the former contribution 5 is folded into contribution 2 as an observation. The closest lc0 work is now engaged rather than list-cited. §11.1 compares Lin et al.'s per-module explained variance (> 0.6, lower for mid-layer attention) with this vocabulary's total $R^2$ (0.80 at L2, 0.32 at L9). It says why the figures are not directly comparable and what the comparison does and does not show. Table 6 has dropped "first use" in favour of "no prior use on a game network found (Appendix E)". The round-1 warn trigger, prior work only weakly distinguished or list-cited, no longer applies.

### D2: methodological_rigor

**warn.** The design is sound, and all five round-1 math errors are fixed. M8∘M2 now uses the probe-decoder round trip and gives a stated tolerance. The M4 whitening is consistent and derived. The orientation argument is corrected. The faithfulness identity is restricted to complete pairwise methods, and per-family M2 reports an additivity gap. The side-to-move plane is accounted for. The decision rules now have numeric thresholds, three C1 seeds, a bootstrap half-width and a $\tau$ filter. Still, some protocol parameters are underspecified, in the first experiment that the paper itself offers for preregistration:
- The sample is "about 1,000 absolute pins" in §10.5 but "1,000 pairs" in the cost arithmetic. Each pin admits two edit types, so the sample size and cost are ambiguous by a factor of up to two (N1).
- The pinner-off tie-break "by square index" does not fix an indexing convention. Under a8 = 0 (FEN order) it picks Ba6, not the Bc4 of the worked example (N2).
- "Family-rank stability as the minimum pairwise Spearman correlation of per-family M2 ranks" does not say whether ranks are taken per pair or over pair-aggregated credit (N3).
- The M8∘M2 remainder $r_{L'}$ mixes content that leaves the vocabulary with round-trip error from the probe and decoder, and no calibration is specified (N4).

None reaches the block anchor. The protocol tests its hypotheses, the unrun status is explicit (§10.3 "None has been tested", §12), and limitations are discussed.

### D3: evidence_sufficiency

**pass.** Every claim about the paper's own results points to a table or file, and every number checked matches the result files. All seven secondary claims flagged in round 1 (13a–g) are fixed. The "named share near 1 at $e$" claim is now labelled a prediction conditional on an unfitted C2 decoder in §2.1, §5.2 item 4, §6.3, H1 and §12, and a C2 fit is added to Stage A0. Predictions remain labelled (H1–H6, "estimate", "proposed defaults"). Two rounding and reporting nits do not reach the warn anchor (N7).

### D4: argument_coherence

**warn.** The thesis is consistent across the abstract, §2 and §13. §13 now answers (a), (b), (c) and the first experiment in three sentences. The §2.2 versus §3.2 contradiction is gone, and the "per-family attribution comes from M2" sentence has left §11.1. Minor inconsistencies between section claims remain:
- H1 lists M1c among the methods run "with decoded deltas zeroed on unchanged-plane squares", but §6.2 says "M1c has no second board, so zeroing does not apply" (N5).
- The first experiment's unit is "pin edits" in the abstract and §13, "absolute pins" in §10.5, and "pairs" in Appendix A (N1).
- §11.3 opens with "Two tensions stay open" and then refers to "a third reason to chain every per-layer reading to the output" without naming the first two reasons.

### D5: writing_quality

**pass.** The register is academic and terse, and IEEE numeric citation style is uniform. Both figures and all tables (A, 1–6, B1, C1, D1, D2) are referenced in the text. The redundancy flagged in round 1 is cut: the abstract is 206 words, §13 is three sentences, Table 3 is merged into Table A, and the Appendix A question-to-method preamble is gone. "Family" is now defined once and used throughout. "Encoder" in "encoder direction" is disambiguated. One table label departs from the family definition (N6), and M1c's "zeroing" is in fact mean replacement in centred coordinates (N8). Both are single-word fixes that do not amount to "several word-choice issues".

## Failure Condition Checks

### F1
Not fired. No mandatory dimension (D2, D3) scores block.

### F2
**Fired.** Mandatory dimension D2 (methodological_rigor) scores warn. Action: `evaluator_decision=request_revision`.

### F3
Not fired. The high-priority dimensions D1 and D4 score pass and warn, not block.

### F6
Not fired. The normal-priority dimension D5 scores pass.

### F4
Not fired. D4 triggers warn on a high-priority dimension, but F4 requires that no mandatory dimension scores warn, and D2 does.

### F5
Not fired. This is round 2 after a writer revision, but no mandatory dimension scores block. D2 dropped from warn with five math errors to warn on four specification gaps, and D3 rose to pass. F5 requires a recurring block.

### F0
Not fired. D2 does not pass.

## Review Body

Revision 1 is substantive and honest. Every round-1 issue was addressed in `paper.md` itself, not just claimed in the response, and the two partial departures (issues 12 and 14) are justified. Using total $R^2$ for the Lin et al. comparison is the better choice. The new M1 forms, the decision rules and the pin-edit construction are mostly correct. What remains is a short list of specification gaps in the material that revision 1 added. Each needs one sentence to fix.

### Status of the 18 round-1 issues

| # | Issue | Status | Evidence in `paper.md` |
|---|---|---|---|
| 1 | M8∘M2 residual branch degenerate | **Resolved** | §8.3 and App. A M8∘M2 read $L'$ through $P_{L'}$ and decode. The degeneracy is stated as the reason. The tolerance (1% of the joint named change) replaces "$c_{G'}=0$". §2.3 and Table C1 are updated. New sub-issue N4. |
| 2 | M4 whitening one-sided | **Resolved** | App. A M4: $\nabla_{\tilde h}y=\Sigma^{1/2}\nabla_hy$ and $\tilde J^\top=\Sigma^{1/2}J^\top$. CG and the whitened residual follow correctly, and the residual ratio is in the whitened norm. [42] is cited. New nit N9. |
| 3 | Orientation argument wrong | **Resolved** | §7.2 uses the corrected denominator argument. App. A M6 shows the equivalence $\sum\mathrm{sign}(\Delta)\phi_n/\sum|\Delta|$. $\tau$ filter and bootstrap added. |
| 4 | Faithfulness identity overstated and misapplied | **Resolved** | §1 contribution 2 holds it as an observation. §9.2 restricts it to M6, pairwise M9 and M1b with a real-edit baseline. §9.3 adds per-family M2 with the additivity gap. The Appendix A derivation is deleted. |
| 5 | Side-to-move plane and mirroring | **Resolved** | §3.1 (mirroring, plane 108), §5.2 item 2, Table B1 C2, §11.3 rewritten. |
| 6 | Occlusion given one sentence | **Resolved** | §6.1 M1c card with zeroing and flipping baselines, off-manifold caveat [6], [8], on-manifold counterpart, and cost. Added to Table A, Table C1, Stage A and H4. New sub-issues N5 and N8. |
| 7 | Decision rules not preregistrable | **Resolved** | §10.5 gives $\tau=0.1$, three C1 seeds with a min–max range, $w_L$ from a 1,000-resample bootstrap, $\rho_L\ge0.5$, $\Gamma_L<0.2$, Spearman $\ge0.8$, $\Gamma_L>0.5$ with AUROC $\ge0.9$, and a drop of $\ge0.3$. All are labelled proposed defaults. Residual gap N3. |
| 8 | Pair construction underspecified | **Partially resolved** | §10.5 gives rules for both edit types, fixed state, and rejection rules, and they reproduce the worked example under a1 = 0. Open: the indexing convention (N2), pins versus pairs (N1), and "one legal move" for the side not to move (N2). |
| 9 | Cost omits refits | **Resolved** | §10.5 and App. A add 50 Cholesky solves and random-network tapping. Arithmetic verified: 63 min plus about 1 h gives about 2 h, propagated to the abstract, §2.5, Table 4 and §13. The 50 solves are an upper bound (see N1 note). |
| 10 | Taps ignore pin evidence | **Resolved** | L8 added, with one sentence for each tap's justification (§10.5). App. D and Table D1 include L8. |
| 11 | H2 near-guaranteed | **Resolved** | H2 is stated relative to the neighbour-augmented decoder. Its fit is placed in Stage A. |
| 12 | "Named share near 1" stated as fact | **Resolved** | Labelled a prediction in §2.1, §5.2(4), §6.3 and H1. C2 fit added to Stage A0 and Table 4. §12 says it is unfitted. |
| 13 | Secondary claims (a–g) | **Resolved** | (a) deleted; (b) "12 and 13 families"; (c) both bit counts given; (d) numbers given and match the JSON; (e) "no prior use … found"; (f) count dropped; (g) "this paper therefore places". |
| 14 | Lin et al. not engaged | **Resolved** | §11.1, with comparability limits stated. Nit in N7. |
| 15 | Families, groups, counts | **Resolved** | "Family" means top-level prefix, $m=28$ (verified), defined once in §3.2. "Group" is gone. "About 30" survives only as the literature's prescription (§4.6). Residual label N6. |
| 16 | "Not itself an attribution" | **Resolved** | §2.2 and §7.2: "a two-player attribution and a coverage measure". |
| 17 | Redundancy | **Resolved** | Abstract 206 words, §13 three sentences, Table 3 merged into Table A, preamble deleted, §1 quote paraphrased. |
| 18 | Minor (Fig. 1b, "encoder", ONNX Runtime) | **Resolved** | Figure 1(b) annotates L2 at 0.76, and the caption gives 0.765 vs 0.758. §3.2 and §3.1 are fixed. |

Summary: 17 resolved, 1 partially resolved (8), 0 unresolved.

### New issues from revision 1

Ordered by severity. N1–N5 carry the D2 and D4 warns. N6–N9 are nits.

**N1. Sample unit of the first experiment is ambiguous** (abstract "about 1,000 pin edits"; §10.5 "About 1,000 absolute pins"; App. A "With 1,000 pairs"). Each pin yields a pinner-off and an interposition edit, so 1,000 pins means up to 2,000 pairs and up to twice the 63-minute suffix cost. *Fix:* in §10.5, write "about 500 absolute pins, each giving one pinner-off and one interposition pair, about 1,000 pairs in total" (or keep 1,000 pins and double the suffix cost to about two hours, total about three). Use "pairs" in the abstract and §13. *Note on cost:* $G+\alpha I$ does not depend on the tap, because the design is the same features plus square one-hot at every tap, and the random networks share the training split's design. So about seven factorisations (five $\alpha$ values and two further splits), each with multi-column right-hand sides, replace the 50 solves. One clause in App. A could say that the hour is an upper bound.

**N2. Tie-break convention and "legal move" for the side not to move** (§10.5 *Pairs*). "Ties broken by square index" picks c4 under a1 = 0 (python-chess; c4 = 26 < a6 = 40) but a6 under a8 = 0 (FEN order; a6 = 16 < c4 = 34), which would contradict the worked example. In the interposition edit the pinned side is usually not to move, so "in one legal move" must mean a pseudo-legal move of that side. *Fix:* add "square index with a1 = 0, h8 = 63 (python-chess)". Replace "in one legal move" with "by a move that would be legal for the pinned side if it were to move". State the order "minor pieces first, then nearest".

**N3. Rank-stability statistic not fully defined** (§10.5 *Quantities*; Table B1 C3). It is not stated whether the Spearman correlation is over per-pair family ranks, which are noisy with 12–13 families per pair, or over ranks of pair-aggregated credit. *Fix:* "ranks of mean $|\Delta^{(L)}_G|$ over pairs, per tap; minimum pairwise Spearman across the five $\alpha$ values".

**N4. The M8∘M2 remainder is not calibrated** (§8.3; App. A M8∘M2). Because $D_{L'}\circ P_{L'}$ is not the identity, $r_{L'}$ is non-zero even for a change the vocabulary fully names. That is the intended fix for the column degeneracy, but it means $\pi_{G\to r}$ mixes "leaves the vocabulary" with round-trip error, including the soft-output mismatch the card already names. *Fix:* add a calibration control. Apply the same round trip to the real edit's change $h_{L'}(x')-h_{L'}(x)$ and report its remainder beside the exact decoder split $\varepsilon'_{L'}-\varepsilon_{L'}$ computed from true bits. Only the excess of $\pi_{G\to r}$ over this round-trip floor is read as content leaving the vocabulary.

**N5. H1 contradicts §6.2 on M1c** (§10.3 H1; §6.2; App. A M1). H1 runs "M1b, M1c, M6 and M9 under the full vocabulary" "with decoded deltas zeroed on unchanged-plane squares". §6.2 says zeroing does not apply to M1c. *Fix:* in H1, remove M1c from that clause and add "and M1c occlusion of local families agrees in sign and rank with M6 under C2", which matches App. A M1's control "agreement with M2 and M6 at $e$".

**N6. `square.control` row label** (Table 5). Families are defined as top-level prefixes (§3.2), and the prefix here is `square`. *Fix:* label the row `square` or add "(family `square`)".

**N7. Reporting nits** (§11.1; Table 1). Total $R^2$ at L2 is $1-0.19497=0.805$, so print 0.81, or write "about 0.8". In Table 1, "20,000 positions" for the value-head row is $n$, and the held-out set is 3,960. Write "3,960 held-out of 20,000".

**N8. "Zeroing" in M1c is mean replacement** (§6.1; App. A M1). $b^G_s=0$ in centred coordinates sets the family to its per-square mean, not to zero bits. *Fix:* call it "mean occlusion ($b^G_s=0$, the per-square mean)". Also define M1c's residual: "the output change not occluded" (Table A) and "the output change that occluding all families jointly does not produce" (App. A) have no stated reference quantity. Use $g_e(q_e)-g_e(q_e+\varepsilon_e)$, the further change from also removing $\varepsilon_e$.

**N9. $\Sigma_{hh}$ is singular at post-LN taps** (App. A M4). LayerNorm confines each token to an affine hyperplane, so $\Sigma_{hh}$ has rank at most 255 and $\Sigma_{hh}^{-1/2}$ does not exist. *Fix:* write $\tilde h=(\Sigma_{hh}^{1/2})^{+}(h-\mu)$, as M3 already does. The covector transforms use only $\Sigma^{1/2}$ and are unaffected.

### Math check of new material

- **M1a/b/c.** M1b completeness holds, with the dropped zeroed part moved into $\mathrm{IG}_\varepsilon$. M1c's ONNX cost of $m+1$ passes is correct. The reduction argument in §6.1 is sound.
- **Thresholds.** The rules are mutually consistent: a dropped tap cannot be promoted, since promotion requires $\rho_L$ above the C1 maximum plus $w_L$. $\Gamma_L$ is well defined on oriented pairs after the $\tau$ filter.
- **Pin-edit construction.** Correct under a1 = 0 (see N2).
- **M8∘M2.** The per-square centring of $D_{L'}$ cancels in the difference, as claimed. The decomposition is well defined but uncalibrated (N4).
- **M4 whitening.** Correct apart from the inverse (N9).

## Evaluator Decision

evaluator_decision=request_revision

Derived from F2 (severity 70), the highest-severity condition that fired: the mandatory dimension D2 scores warn. No dimension scores block, so F1, F3, F6 and F5 do not fire.

This was round 2 of 2, so the in-pair loop is exhausted (skill rule: max 2 rounds; remaining issues go to "Acknowledged limitations"). No further evaluator round follows. N1–N9 are one-sentence copy edits that need no new experiment. Whatever is applied before Phase 7 should be applied verbatim from the fixes above. Whatever is not applied goes into an "Acknowledged limitations" note:

1. **Sample unit (N1).** The first experiment's sample is stated as both 1,000 pins and 1,000 pairs. The cost estimate assumes 1,000 pairs.
2. **Edit-construction conventions (N2).** The square-index tie-break assumes a1 = 0. Interposition moves are pseudo-legal moves of the side not to move.
3. **Rank stability (N3).** The statistic is computed on pair-aggregated family credit (to be fixed at preregistration).
4. **M8∘M2 remainder (N4).** The remainder includes probe-decoder round-trip error. Until it is calibrated against the round trip on real edits, it is an upper bound on content leaving the vocabulary.
5. **H1 and M1c (N5).** M1c is compared with M6 under C2 by agreement, not by the zeroed-delta criterion.
6. **Nits (N6–N9).** Table 5's `square.control` is family `square`. L2 total $R^2$ is 0.805, and the value-head test set is 3,960 positions. M1c "zeroing" is mean occlusion. M4 whitening uses a pseudo-inverse, because post-LN covariance is singular.
