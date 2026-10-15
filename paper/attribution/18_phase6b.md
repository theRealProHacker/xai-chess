# Phase 6b — Evaluator paper-visible scoring

Contract `academic_paper/evaluator_full/v1`, baseline v3.6.6, round 1. Draft: `paper.md` (main text about 9,940 words including tables and math, Appendices A–E about 4,460). Scored against my `<phase6a_output>` Scoring Plan. Writer `<phase4a_output>` checked: all seven writer dimensions are paraphrased and acknowledged. Writer D2 (every own-result claim points to a table) holds except for issues 9, 10 and 12. Writer D7 (terms used consistently) fails on family/group (issue 15).

Verification done: every number in Table 2 and Figure 1 recomputed from `lc0-probe/results_square_decoder.json` (square-only, features-only and features+square FVU; features-add and partial $R^2$ at all 11 taps; all match to the printed precision). Table 1 checked against `results_value_decoder_mlp_n151614.json`, `results_value_via_head.json` and `results_square_decoder_mlp.json`. Table 6 column sums checked (356, 331). Appendix A cost arithmetic recomputed (6.24e5 suffix passes × 0.6 ≈ 3.7e5 pass-equivalents ≈ 35 min at 180/s). M6 completeness and antisymmetry, M9 endpoints, M3 (LEACE) and the 16-coalition policy split checked by hand. Feature orientation checked in `lc0-probe/square_probe.py` and `lc0_planes.py`. Both figures inspected.

## Dimension Scores

### D1: originality

**warn.** A contribution is stated, and parts of it are new: the reduction of embedding-decoder attribution to input saliency (§5.2, §6), the two-player named/residual split $\rho_L$ as a coverage measure (§7.2), feature paths with live attention (§8.3) and the identifiability statement (§9.1). The introduction names the gap (§1, the working-note dilemma) and Table 7 contrasts each gap with prior work. But engagement is deep for some prior sources and only a list citation for others. Lin et al. [39], the closest lc0 work, which replaces every module of a chess transformer and reports per-module explained variance above 0.6, appears only in §4.5 and as a single cell in Table 7. The paper never compares those figures with its own partial $R^2$ (Table 2), and that is the obvious external check. Contribution 5 ("a derivation") is an identity that follows from completeness in one line (issue 4). M6 is baseline Shapley with an error player, which §7.2 says itself. The originality lies in assembling these pieces for this system, and the contribution list should say that rather than present six separate contributions.

### D2: methodological_rigor

**warn.** The design fits the question, and the unrun status is stated clearly (§1, §10.3 "None has been tested", §12, Tables D1/D2 "nothing computed"). Limitations are discussed and not merely listed. The survey method is given in Appendix E. Some protocol parameters are underspecified or unjustified, and several method definitions contain mathematical errors. The decision rules in §10.5 use "high", "small", "large" and "sharply" with no thresholds, so as written the protocol cannot be preregistered unchanged. C1 is one random network, so the "C1 range" the first rule tests does not exist. No uncertainty estimate is defined for $\rho_L$ over pairs. M8∘M2's residual branch is degenerate as specified (issue 1). M4's whitening transforms only one side of the projection (issue 2). The orientation argument for $\rho_L$ is wrong (issue 3). The faithfulness identity is applied to per-family M2, which is not complete (issue 4). The side-to-move plane is missing from the $\varepsilon_e$ accounting and from C2, and the colour mirroring of labels is not disclosed (issue 5). None of these reaches the block anchor. The unrun protocol is never presented as having produced results, and limitations are present. Together they keep D2 at warn.

### D3: evidence_sufficiency

**warn.** Core claims are supported. Every Table 2 number matches the result file. Every survey claim in §4 carries a citation, and the citation audit (16) checked 41 of them. Predictions are labelled (H1–H6, "estimate", "predicts, but has not checked"). The survey also reports evidence against the paper's own preferences, for example Wu et al. [78] against the dormant-pathway criterion and the Pálsson/Sutter tension in §11.3. But some secondary claims lack support, or state more than the data show. §4.4 has an uncited, self-declared unverified claim ("A 2026 follow-up, not verified here"). §10.6 says "Each edit touches 13 groups", but Table 6 shows 12 for $x'_a$. Table 1's note "Decoders keep 14,565 bits" is wrong for the board-level decoder, which uses 17,831. "Value head applied to decoded L9 tokens: weak" withholds a number that exists (FVU 0.81 at `dense2`, against 1.54 for square only). Table 7 claims "first use on a game network" on the strength of a survey that Appendix E calls non-systematic. §5.2 consequence 4 and §6.3 state as fact that the named share at $e$ under C2 is near 1, but the C2 decoder has never been fitted. Lomasov et al.'s "six concepts" is flagged as uncertain in 16 §4 and is still printed without qualification.

### D4: argument_coherence

**warn.** The overall logic holds. The thesis is the same in the abstract, §2, §11.1 and §13. Each of (a), (b), (c) and the first experiment is posed in §1 and answered in §2, §6–§8, §10.5 and §13. Every section ends with an explicit transition. But in places a claim's strength or hedging changes slightly between sections. §2.2 says M6's $\rho_L$ "is not itself an attribution", while §3.2 defines an attribution as any assignment of an output share to the players $n^G$ and $\varepsilon$ under a removal rule, and M6 does exactly that. §11.1 says "Per-family attribution comes from M2", which sits uneasily with §9.1, which says per-family splits are not identifiable. §11.2 reconciles the two, but §11.1 should carry the qualifier itself. §11.3 recommends testing "side-to-move-relative bits", although the labels already are side-to-move relative (`square_probe.py`: black-to-move positions are colour-mirrored before extraction). §2.4 and §9.1 speak of "a dozen families", §3.2 of "about 30 families" and Table 6 of 13 "groups", with no mapping between the three.

### D5: writing_quality

**warn.** The register is academic and terse. There are no em dashes, no hype and few hedges, and grammar errors are rare. IEEE numeric citation style is uniform (16 §2). Both figures and all tables are referenced. There are minor citation-format or term-usage inconsistencies: family against group (D4), and "encoder" meaning a probe (§3.2) next to "encoder block" meaning a transformer layer, which the title's "Probes" and the user's "encoder at any layer" make ambiguous. The bigger problem for this author is redundancy. The answer is stated four times (abstract, §2, §11.1, §13), and method properties are tabulated three times (Table A, Table 3, Table C1), with the Appendix A preamble adding a fourth, prose, question-to-method map. The §1 working-note quote keeps an ASCII "->" and informal wording inside formal prose. Figure 1(b) labels L1 (0.76) as the post-embedding peak, but L2 is marginally higher (0.765 against 0.758).

## Failure Condition Checks

### F1
Not fired. No mandatory dimension (D2, D3) scores block.

### F2
**Fired.** Mandatory dimensions D2 (methodological_rigor) and D3 (evidence_sufficiency) score warn. Action: `evaluator_decision=request_revision`.

### F6
Not fired. The only normal-priority dimension, D5, scores warn, not block.

### F3
Not fired. The high-priority dimensions D1 and D4 score warn, not block.

### F4
Not fired. D1, D4 and D5 trigger warn, but F4 requires that no mandatory dimension scores warn, and D2 and D3 do.

### F5
Not applicable. This is round 1, and no mandatory dimension scores block.

### F0
Not fired. D2 and D3 do not pass.

## Review Body

The paper does what the user asked. It gives distinct, implementable approaches for (b) eight single-tap methods in two directions and (c) four cross-layer compositions plus a residual budget and a comparison map. It recommends a concrete first experiment with costs and decision rules. Every own number checks out against the data. Status tags are mostly disciplined. For (a), the paper delivers one method (M1) and an argued negative result. That is defensible, but the user explicitly named occlusion over feature bits and the paper gives it only one sentence (issue 6). What blocks acceptance is a set of mathematical and specification errors in the method definitions, plus a first-experiment protocol whose decision rules cannot yet be preregistered. Issues are ordered by severity.

**Method math**

1. **M8∘M2's residual branch is degenerate** (§8.3; Appendix A, M8∘M2; Table A; Table C1). The change at $L'$ is regressed on the columns of $W_{L'}$: 14,565 columns in $\mathbb R^{256}$, which almost surely span the whole space. Any 256-vector is then an exact combination of decoder columns, so $r_{L'}\to0$ as the ridge penalty goes to zero, and $\pi_{G\to r}$ ("how much leaves the vocabulary") only measures $\alpha$. This is the same surjectivity the paper identifies correctly for M4 at bit level (§7.3) but does not apply here. Also, under ridge no $c_{G'}$ is exactly zero, so "families with $c_{G'}=0$ are skipped" never triggers. *Fix:* define the named part at $L'$ through features, not columns. Read the patched token through the probe, $\hat f=P_{L'}(h^{\rm patch}_{L'})$, and decode, so that the named change is $D_{L'}(\hat f)-D_{L'}(P_{L'}(h_{L'}(x)))$ and $r_{L'}$ is the remainder. Alternatively, restrict each family to a rank-capped subspace (top-$r$ principal directions of $W^{G'}_{L'}$ with $\sum r<256$) so that the complement is non-trivial. State which, and replace "$c_{G'}=0$" with a stated tolerance.

2. **M4's whitening is inconsistent** (§7.3; Appendix A, M4). The card replaces $J_s^\top$ with $\Sigma_{hh}^{-1/2}J_s^\top$ but leaves $\nabla_{h}y$ unchanged, so $\rho^\perp=\nabla y-J^\top\mathrm{CG}$ mixes two coordinate systems. Columns of $J_s^\top$ and $\nabla_h y$ are both covectors. Under the change of variables $\tilde h=\Sigma^{-1/2}(h-\mu)$, both transform as $\Sigma^{1/2}(\cdot)$. *Fix:* transform both sides with the same matrix, write the derivation in two lines, and cite Park et al. [42] for the choice of inner product.

3. **The orientation argument for $\rho_L$ is wrong** (§7.2, after the $\rho_L$ display). "Without orientation, a complete vocabulary on edits of mixed sign would give $\rho_L\approx0$" is false. If $\phi_n=\Delta^{\rm true}$ on every pair, then $\sum\phi_n/\sum\Delta^{\rm true}=1$ whatever the signs. The real problem is that the unoriented denominator $\sum\Delta^{\rm true}$ can be near zero, which makes the ratio unstable. *Fix:* replace the sentence with "Without orientation the denominator $\sum\Delta^{\rm true}$ can cancel to near zero; orientation makes it $\sum|\Delta^{\rm true}|$ (Appendix A, M6)." Also add a minimum $|\Delta^{\rm true}|$ filter for pairs (see issue 7).

4. **The faithfulness identity is overstated and misapplied** (§1 contribution 5; §9.2; §9.3; Appendix A derivation). For complete pairwise methods, $\mathrm{err}=$ residual share is one line of algebra, not a derivation worth listing as a contribution. More importantly, the method list includes "M2's joint patch", which has one named player, so $\sum_{G\in\delta}A_G$ is just $\Delta_{\rm named}$ and the identity reduces to the definition of $\gamma_L$. Per-family M2 single patches are not complete, because $\sum_G\Delta^{(L)}_G\neq\Delta^{(L)}_{\rm named}$ when families interact [65]. For them the error is informative: it measures the interaction gap. Yet §9.3's "Error applies only where..." list leaves them out. *Fix:* call contribution 5 an observation and merge it into contribution 2. In §9.2 and §9.3, say that the identity covers M6, pairwise M9 and board-path M1. Add per-family M2 to the error list, with $\sum_G\Delta_G-\Delta_{\rm named}$ reported as the additivity gap.

5. **The side-to-move plane and the colour mirroring are not accounted for** (§3.1; §5.2 consequence 2; Appendix B C2). `square_probe.py` colour-mirrors black-to-move positions before extracting features, so the vocabulary is side-to-move relative. lc0 still receives a broadcast "black to move" plane (`lc0_planes.py`, plane 108) that no feature carries. That plane therefore lands in $\varepsilon_e$ on all 64 squares, and consequence 2 does not list it. *Fix:* state the mirroring in §3.1 (it is a replication detail). Add the side-to-move plane to consequence 2 and to the optional C2 additions. Delete or rewrite §11.3's "should test with side-to-move-relative bits", for example as "the vocabulary is already side-to-move relative; an absolute-colour variant would test Nanda et al.'s [85] point".

**First experiment and protocol**

6. **Part (a) gives the user's own occlusion idea one sentence** (§4.1 removal rule 1; §6). The working note names "integrated gradients / occlusion over feature bits". §6 develops only gradient forms, and bit-flip re-decoding appears only as an abstract removal rule. *Fix:* add to §6.1 a short card for bit or family occlusion at $e$: zero or flip $\tilde f^G$, re-decode, and run the suffix. State that it is M9's value function with a synthetic baseline, off-manifold [6], [8], and subject to the same locality reduction. This gives (a) three forms (point gradient, board-path IG, occlusion) rather than one.

7. **The decision rules cannot be preregistered as written** (§10.5). "High", "small", "large", "sharply" and "family-rank stability" have no thresholds. C1 is one network, so there is no "C1 range". No pair filter or uncertainty estimate is set. *Fix:* use at least three C1 seeds and define the range as their min–max. Set numeric thresholds, for example "$\rho_L$ above the C1 maximum by more than its 95% bootstrap CI half-width over pairs", "$\sum|\gamma_L|/\sum\Delta^{\rm true}<0.2$", "Spearman $\ge0.8$ across the five $\alpha$", "drop of $\ge0.3$ between L5 and L9". Each value is an illustrative default the author must fix. Add a filter $|\Delta^{\rm true}|>\tau$ with $\tau$ stated. These belong in the protocol, not as afterthoughts.

8. **The pair construction is underspecified** (§10.5 "Pairs"). The text does not say which piece is interposed or where it comes from (moving it changes a third square), how the pinner's destination is chosen, or whether side to move stays fixed. The worked example answers these for one position (bishop c8–d7, b5–c4), but no rule is given. *Fix:* one sentence per edit type: the interposer is the nearest same-side minor piece that can reach a line square in one move. The pinner moves to the nearest legal square off the line along its own ray. Side to move is unchanged. Material is preserved.

9. **The cost estimate omits the refits** (§10.5 "Cost"; Appendix A cost arithmetic). The one-hour figure counts suffix passes. It does not count refitting decoders at four taps for each of five $\alpha$ values and for each C1 network (a 14,565-dimensional ridge solve per fit), or tapping 20,000 positions through each random network. *Fix:* add both items with an estimate, or state that they are excluded and give the new total.

10. **The choice of taps ignores the cited pin evidence** (§10.5). The pin probe peaks at block 8 (Table 1), which is offered as a reason to start with pins, yet the taps are $e$, L2, L5 and L9. *Fix:* add L8 or say why not, and say in one sentence why L2 and L5 were chosen (the partial-$R^2$ plateau and midpoint, from Table 2).

11. **H2 is close to guaranteed** (§10.3). M6 counts information moved between squares as residual (Appendix A, M6), and that grows with depth because of attention, so $\rho_L$ falling after L2 is expected under cause (a) alone. *Fix:* state H2 relative to the neighbour-augmented decoder control ("$\rho_L$ falls after L2 even with the neighbour-augmented decoder"). Otherwise H2 cannot fail for an interesting reason.

**Evidence and claim status**

12. **"Named share near 1 at $e$" is stated as fact but rests on an unmeasured fit** (§2.1, §5.2 consequence 4, §6.3, §11.1). It holds only if the C2 decoder fits near 0.002, and the C2 decoder has never been fitted. *Fix:* label it a prediction tied to H1, and add a "C2 fit at $e$" line to Stage A0, which costs seconds.

13. **Unsupported or inaccurate secondary claims.** (a) §4.4: delete "A 2026 follow-up, not verified here, contests the AxBench SAE result", or cite it. (b) §10.6: "Each edit touches 13 groups" should read "12 and 13 groups" (Table 6, `rook` is 0 for $x'_a$). (c) Table 1 note: "Per-square decoders keep 14,565 bits; the board-level decoder keeps 17,831." (d) Table 1: give the number for "Value head applied to decoded L9 tokens" (test FVU 0.81 at `/value/dense2`, against 1.54 from square identity alone, `results_value_via_head.json`). (e) Table 7, first row: change "first use on a game network" to "no prior use on a game network found (Appendix E)". (f) §4.8: qualify "on six concepts" as 16 §4 recommends, or drop the count. (g) §4.4: "SAEs therefore belong on the residual of $D_L$" is a design choice drawn from LLM benchmarks. Write it as "this paper therefore places SAEs on the residual".

14. **Closest lc0 work is not engaged** (§4.5, §11, Table 7). *Fix:* add two sentences to §11.1 comparing Lin et al.'s [39] per-module explained variance (above 0.6, lower for mid-layer attention) with Table 2's partial $R^2$ (0.76 at L2, 0.24 at L9). Note that theirs is unsupervised with hand labels and covers a larger network. State what the comparison does and does not show.

**Coherence and prose**

15. **Families, groups and counts** (§2.4, §3.2, §9.1, §10.5, §10.6, Table 6, Table D2). Define once: "families" are the about 30 attribution players, "groups" are the extractor's top-level name prefixes, and a pin edit changes 12–13 groups, which map to $k\approx12$ families in the cost estimate. Or use one term throughout.

16. **"Not itself an attribution"** (§2.2, §7.2). This contradicts §3.2's definition. *Fix:* "$\rho_L$ is a two-player attribution; it does not say which family moved the output."

17. **Redundancy** (abstract, §2, §11.1, §13; Tables A, 3, C1; Appendix A preamble). *Fix:* cut §11.1 to the two sentences that add something (the L1–L5 pointer and the Lin et al. comparison from issue 14). Merge Table 3 into Table A by adding a "Question" column. Shorten §13 to three sentences. Cut the abstract by about a third, keeping the measured fits, the reduction at $e$, the identifiability limit and the first experiment. Replace the §1 quote with a one-sentence paraphrase and keep the quote in a footnote if wanted.

18. **Minor.** Figure 1(b): annotate L2 (0.765) as well as L1, or instead of it. §3.2: "this paper also calls probe-side methods the encoder direction" should state that "encoder" never means an lc0 encoder block in that sense. §3.1: "ONNX on CPU has no backward pass" should say "the ONNX Runtime inference build".

None of these changes needs a new experiment except the optional seconds-long C2 fit in issue 12. Issues 1–5 and 7 are required for D2 to pass. Issues 12 and 13 are required for D3 to pass.

## Evaluator Decision

evaluator_decision=request_revision

Derived from F2 (severity 70), the highest-severity condition that fired: the mandatory dimensions D2 and D3 score warn. No block on any dimension, so F1, F3 and F6 do not apply. Round 1, so F5 does not apply.
