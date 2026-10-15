## 10. Evaluation Protocol and First Experiment

### 10.1 Ground truth in three tiers

Ground truth is split by what it is true to [@chen2020] (Table 3).

**Table 3**
*Ground-truth tiers*

| Tier | Source | Measures | Used for |
|---|---|---|---|
| GT-M (true to lc0) | real board edits run through lc0 (removals, pin edits, pawn moves) | $\Delta^{\rm true}$; per-family feature change | grading every method |
| GT-C (true to chess) | MaskPinner and LineBlocker in patched Stockfish | engine's rule-level effect; "pin/line decisive" label | lc0's alignment with chess, not attribution faithfulness |
| GT-K (consistency) | decoder edit against board edit at the same layer | $\gamma_L$ (M2) | realisability of feature-space edits |

Edits are re-encoded with the same fen-only fill, so $\Delta^{\rm true}$ has no history artefact. Changed bits outside the target family are reported. Controls for line-blocking edits need a nonzero noise floor: in earlier project work, control squares chosen by geometry alone gave a near-zero floor that skewed verdicts towards "causal".

### 10.2 Metrics

Coverage (Section 9.3) is reported and never used to rank methods. Error and localisation, computed on $A_G(x')-A_G(x)$ against a shared baseline pool, apply to the methods listed in Section 9.3. Consistency is $\gamma_L$, alignment is IIA with Isolate, and stability is the Spearman correlation of family rankings across C3 settings and between linear and MLP maps. Control gaps subtract each metric under C1, C2 and C5, and the pool mixes puzzle and non-puzzle positions [@ramaswamy2023]. Ablation type, noising or denoising direction, and the target as a logit are fixed in advance, since circuit metrics move with these choices [@zhang2024], [@miller2024], [@heimersheim2024].

### 10.3 Pre-specified hypotheses

None has been tested.

- **H1.** At $e$, the C2 decoder fits with FVU close to 0.002, and under C2 $\rho_e$ is close to 1. With decoded deltas zeroed on unchanged-plane squares, M1b, M6 and M9 under the full vocabulary match the C2 control, and M1c occlusion of local families agrees in sign and rank with M6 under C2. A random-initialised embedding fits with FVU close to 0.002.
- **H2.** $\rho_L$ falls with depth after L2 even when M6 uses the neighbour-augmented decoder (Appendix A, M6), so the fall is not only information moved between squares, which M6 counts as residual. $\rho_e$ is excluded. The neighbour-augmented fit runs in Stage A.
- **H3.** In the M12 budget, non-locality explains a non-trivial share of $\varepsilon_L$ at L5 to L9.
- **H4.** Among non-pairwise methods, distributional M9 localises single-family edits better than M1a, M1c, M4 and M10, and all five beat M0.
- **H5.** LEACE use curves (M3) lag readability curves, and recomputation (M8) is high for occupancy-derived families.
- **H6.** Engine-derived families gain attribution with depth, resembling the late Go profile of Tomlin et al. [@tomlin2022], and lose most of it under C7.

### 10.4 Staging

**Table 4**
*Stages and order-of-magnitude CPU cost (estimates)*

| Stage | Methods | Needs | Estimated cost |
|---|---|---|---|
| A0 | M2 + M6 on pin-edit pairs, M0 on the same pairs, C2 fit at $e$ | ONNX split at five taps; three random networks | about two CPU hours, refits included |
| A | M3, M8, M8∘M2, M9, M2/M6 on all edit types | edit pairs of every type | about one CPU day |
| A | M1c | ONNX at $e$ | minutes |
| B | M1a, M1b, M4, M5, M7, M10 | verified PyTorch port | port plus one CPU day |
| C | M11, M12 | outputs of A and B; SAEs on $\eta_L$ | hours to days |

H1 to H6, edit construction, thresholds and metrics are to be preregistered before Stage A0.

### 10.5 What to try first

The first experiment is M2 and M6 on pin-edit pairs: the edits are clean, MaskPinner gives chess-side ground truth, and the project's pin probe peaks at AUROC 0.945 at block 8 (Table 1).

*Pairs.* About 500 absolute pins from the puzzle pool (a pinned piece cannot legally leave the line because its king stands behind it), each giving one pinner-off and one interposition pair, about 1,000 pin-edit pairs in total. Edits are board edits, not moves. Side to move, castling rights, en-passant square and rule-50 count stay unchanged, and no piece is captured. Square indices run a1 = 0 to h8 = 63 (python-chess). In the *pinner-off* edit, the pinner moves to the nearest square off the pin line (Chebyshev distance, ties broken by lower square index) that it could reach by a move that would be legal for its side if it were to move. In the *interposition* edit, a piece of the pinned side other than the pinned piece and the king moves to a line square between pinner and king, by a move that would be legal for the pinned side if it were to move. Candidates are ordered minor pieces first, then nearest to the target square, then by lower square index. Edits that put either king in check or leave a pin on the line are rejected. Changed bits and MaskPinner labels are recorded. Pairs with $|\Delta^{\rm true}|\le\tau$ are dropped and counted.

*Taps.* $e$ (C2 vocabulary), L2, L5, L8 and L9. Partial $R^2$ peaks at L2 (Table 2), L5 is the midpoint of its decline, the pin probe peaks at L8 (Table 1), and L9 is last.

*Quantities.* M2 and M6 outputs and per-family patches. Aggregated: $\rho_L$ with a 95% bootstrap interval over pairs (1,000 resamples, half-width $w_L$), $\Gamma_L=\sum|\gamma_L|/\sum\Delta^{\rm true}$, and family-rank stability: per tap, families are ranked by mean $|\Delta^{(L)}_G|$ over pairs, and the statistic is the minimum pairwise Spearman correlation of these ranks across the five $\alpha$ values.

*Controls.* C1 on three independently random-initialised copies of lc0, each with refitted decoders. The C1 range is the minimum to maximum over the three. C3 uses five $\alpha$ values from $10^{-2}$ to $10^2$ and two further train/test splits. A C2 decoder fit at $e$ tests H1's premise. M0 on the same pairs sets the floor.

*Cost.* About $2k+2\approx26$ suffix passes per pair, tap and setting for $k\approx12$ changed families, over ten settings (seven for C3, three for C1), take an estimated 63 minutes under ONNX. Refitting decoders at five taps per setting (50 ridge solves of dimension 14,629, an upper bound) and tapping 20,000 positions through each random network add an estimated hour. The total is about two CPU hours (Appendix A).

**Decision rules, fixed before running.** The numbers are proposed defaults, to be fixed at preregistration.

- $\tau=0.1$ in units of $y$ (a logit difference).
- A tap is dropped if trained-lc0 $\rho_L$ lies inside the C1 range widened by $w_L$ on each side.
- If, at L2 or L5, $\rho_L\ge0.5$, $\rho_L$ exceeds the C1 maximum by more than $w_L$, and $\Gamma_L<0.2$, then M8 and M8∘M2 are promoted up to L9. Per-family M2 credit is reported only where rank stability is $\ge0.8$. Otherwise families are merged until it is (Section 9.1).
- If $\Gamma_L>0.5$ at a tap where probes read the changed `xray` and `mobility` bits at AUROC $\ge0.9$, M5 is promoted, initialised from $W^G_L$ and probe directions.
- If $\rho_{\rm L5}-\rho_{\rm L9}\ge0.3$, the M12 budget runs on L5 to L9 before anything in $\varepsilon$ is named.

### 10.6 Worked example

No lc0 quantity has been computed for this opening example. Base $x$, after 1.e4 e5 2.Nf3 Nc6 3.Bb5 d6, White to move: `r1bqkbnr/ppp2ppp/2np4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4`. The bishop on b5 pins the knight on c6 to the king on e8. Both edits keep material. In $x'_a$ the pinner leaves the line (bishop b5 to c4). In $x'_b$ Black interposes (bishop c8 to d7).

With no pin bit, the pin shows in Table 5 as an x-ray bit on the pinner's square (`xray.knight.king.ne.1_2` on b5, which becomes `xray.knight.bishop.ne.1_1` under $x'_b$) and as absent mobility bits on the pinned knight: ten `mobility` bits on c6 switch on under both edits. Everything else is geometry the edit drags along. The edits touch 12 and 13 families, so a per-family claim about "the pin" would rest on a ridge split among them (Section 9.1). Appendix D gives the procedure.

**Table 5**
*Bits changed by the two edits, by family (own measurement, current extractor, engine lines off)*

| | $x\to x'_a$ (Bc4) | $x\to x'_b$ (…Bd7) |
|---|---|---|
| Squares changed | 27 | 21 |
| Bits changed | 356 | 331 |
| `route` | 203 | 180 |
| `mobility` | 40 | 36 |
| `control` | 34 | 36 |
| `role` | 14 | 24 |
| `square` | 16 | 20 |
| `pawn_shield` | 15 | 4 |
| `stability` | 15 | 3 |
| `blocked_by` | 3 | 10 |
| `blocks` | 8 | 5 |
| `xray` | 2 | 8 |
| `pawn` | 4 | 2 |
| `occupant` | 2 | 2 |
| `rook` | 0 | 1 |

Section 11 states what the protocol answers and leaves open.

## 11. Discussion

### 11.1 Where to attribute, and the closest lc0 work

The front/back choice becomes a choice of taps guided by $\rho_L$, and Table 2 points to L1 to L5, where partial $R^2$ is 0.53 to 0.76 and locality no longer holds. Lin et al. [@lin2026] report explained variance above 0.6 per module, lower for middle-layer attention, for unsupervised sparse replacements of a larger lc0 network with hand-labelled features. The fixed vocabulary here explains 0.81 of token variance at L2 and 0.32 at L9 (one minus the features-plus-square FVU, Table 2). The figures are not directly comparable. Lin et al. reconstruct module outputs of another network with a learned dictionary, while Table 2 reconstructs residual-stream tokens from a fixed vocabulary. Together they suggest that a learned dictionary retains structure at depth that the human vocabulary does not name, which M12 would test on this network. They do not show that the missing share is nameable.

### 11.2 The strongest objection

The strongest objection is that the paper answers how much of lc0 is nameable per layer, not which human feature caused a given value, that its front coverage is fixed by construction, and that its per-family numbers are a ridge split. The paper accepts most of this. The per-feature answer exists only jointly for collinear families, which is why Section 9.1 states it and why every decoder-split method carries C1 to C3. Where ranks are stable, M2 per changed family is the per-feature answer. Where they are not, M3, M5 and M8 carry the claims that do not follow from linear algebra, and M12 carries the residual. Complete attributions can be no better than random without an end-task [@bilodeau2024]. Here the end-task is a real board edit with GT-M ground truth, as in counterfactual benchmarks [@abraham2022], [@huang2024], [@mueller2025].

### 11.3 Open tensions

Two tensions stay open. Pálsson and Björnsson [@palsson2024] recommend nonlinear probes for chess agents, while Sutter et al. [@sutter2025] argue for linear maps. Nanda et al. [@nanda2023] find that a player-relative encoding can make a representation linear, and this vocabulary is already side-to-move relative (Section 3.1). An absolute-colour variant would test their point here. Freezing attention, which makes M7's edges linear, conflicts with the head-ablation evidence for look-ahead of Jenner et al. [@jenner2024]. Separately, late-layer overrides of mid-layer solutions [@sandmann2025] are why every per-layer reading here is chained to the output.

### 11.4 Gaps addressed

**Table 6**
*Gaps the protocol addresses*

| Gap | Evidence of absence | Claim of this paper |
|---|---|---|
| Attribution through a fitted decoder $f\to h$ | concept gradients use an encoder pseudo-inverse [@bai2023]; alignment methods work in the encode direction | no prior use on a game network found (Appendix E); M1 reduces to input saliency at $e$; the contribution is M2, M6, M9 |
| Supervised cross-layer replacement with error nodes | all cross-layer dictionaries are unsupervised (Section 4.5) | M7 combines existing parts; depends on Tier-2 sources |
| Residual share for a fixed external vocabulary | Lin et al. [@lin2026] report reconstruction per module, not a graph residual | $\rho_L$ and the M12 budget are new measurements, not new theory |
| Interventional validation of named families on lc0 | lc0 interventions target squares and heads [@jenner2024], steering directions [@sandmann2025] or unsupervised sparse features [@lin2026] | the protocol of Section 10 |
| Residual split by locality | Engels et al. [@engels2025] split error by linear predictability only | M12 budget; neighbour-augmented decoder |
| Interventional concept benchmarks outside NLP | CEBaB, RAVEL and MIB are NLP | a chess benchmark with exact board-level and engine-level operators |
| Attribution for binary, logically constrained features | closest: grouping [@aas2021] and on-manifold Shapley [@frye2021] | not solved; board edits sidestep it only where exact edits exist |

These claims are bounded by what has and has not been run (Section 12).

## 12. Limitations

Nothing in the method catalogue has been run. The empirical inputs are Tables 1, 2 and 5. Every statement about how a method behaves on lc0, including H1 to H6, the decision rules, their thresholds and the costs, is a hypothesis, a proposed default or an estimate. The C2 decoder at $e$ has not been fitted. The fits in Table 2 are single ridge fits ($\alpha=1$) on one 80/20 split, reported without confidence intervals or repeated splits.

The first experiment's specification has open choices. Its cost assumes 1,000 pin-edit pairs from 500 pins and scales linearly with the pair count. The edit construction fixes conventions (a1 = 0 tie-break, interposition by a move legal only if the pinned side were to move), and others would give other pairs. Rank stability is computed on pair-aggregated family credit, so per-pair stability is not measured. Until the round-trip calibration of Section 8.3 is run, $\pi_{G\to r}$ is an upper bound on content leaving the vocabulary. M1c is compared with M6 under C2 by sign and rank agreement only. M4 whitens through a pseudo-inverse, because post-LN covariance is singular.

All inputs are FENs with fen-only history fill. Conclusions hold for lc0 on synthetic history and need replication on game positions with real history. The vocabulary has no pin bit, so "pin credit" is the credit of the families an edit changes. The policy analysis needs a decoder at `/encoder9/ln2` that has not been fitted.

The survey is structured but not systematic, the field is preprint-heavy, the key cross-layer sources are Tier 2, and interventional benchmarks are all from NLP. Structural claims other than the embedding, heads and input encoding, which were checked against the ONNX graph, come from the project's description of the model. GT-C grades lc0's agreement with Stockfish's rule-level effects, not the faithfulness of an attribution. Engine-derived features partly encode the value target, which C7 controls.

The study covers one network, one extractor version and one puzzle-derived position set. Bit counts such as those in Table 5 change with extractor revisions, and family boundaries are set by the extractor. The MLP decoder results enter only as a trend, and the board-level decoder is still improving with data, so M0's floor may move. Bibliographic annotations were written by AI agents from abstracts and landing pages, not from full texts read by a human.

Within these limits, Section 13 restates the answer.

## 13. Conclusion

At the embedding, decoder gradients, integrated gradients and feature occlusion are input saliency in feature names, admissible only with a local vocabulary. At later taps, decoded substitution of real board edits gives the per-edit attribution, the named/residual split says where it is worth doing, erasure and interchange test use, four compositions combine layers, and only an edit's joint effect is identifiable. The first experiment, M2 and M6 on about 1,000 pin-edit pairs at five taps with random-initialisation and ridge controls, costs about two CPU hours and selects the next method by numeric rules fixed in advance.
