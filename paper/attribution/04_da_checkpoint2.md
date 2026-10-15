# Devil's Advocate Report — Checkpoint 2 (synthesis `03_synthesis.md`)

Reviewed against `00_context.md`, `01_scoping.md` (incl. DA Checkpoint 1), `02_biblio_T1..T8.md`, and one
coordinator-supplied fact read off the ONNX graph (`vendor/lc0-nets/t1-256x10.onnx`): the embedding tap is,
per square, $h_{e,s}=g_s\odot\mathrm{mish}(W[\text{planes}_s;\text{posenc}_s]+b)+a_s$ — elementwise-nonlinear,
strictly local (no cross-square information).

## Verdict: REVISE

One critical issue (the grading metric penalises the residual), seven major. All fixable in text; applied below.

---

## Critical

**C-1. The evaluation metric rewards hiding the residual.**
- Type: Method / bias against the project's stated value.
- Location: §6 ICaCE error, H4, M11 objective.
- Problem: $\mathrm{err}_k=\mathbb E|\sum_{G\in\delta}A_G-\Delta^{\rm true}|$. For any complete pairwise method
  (M2-joint, M6, M9 with $x_0=x'$, M1 board-path IG) unchanged families get zero by the dummy axiom, so
  $\mathrm{err}_k=|A_\varepsilon|/|\Delta^{\rm true}|$ exactly: the "error" *is* the residual share. The protocol
  would rank highest whichever method/layer/decoder pushes most credit into named families — the opposite of
  "measure what the vocabulary misses". M11 inherits it: fitting $w$ to make $\sum_G A^{\rm ens}_G$ match
  $\Delta^{\rm true}$ selects layers with small $\varepsilon$; including $\varepsilon$ instead makes every complete
  method's loss identically zero, so $w$ is unidentified. The unlearned variant $w_L\propto\rho_L\pi_L$ weights
  by named share — same circularity.
- Fix: report $A_\varepsilon/\Delta$ as coverage, never as error. Grade complete pairwise methods by $\gamma_L$,
  control gaps (C1-C5) and stability; apply localisation/ICaCE only to non-pairwise methods (point gradients,
  distributional baselines, relevance). Recast M11 as a disagreement map, not a fitted ensemble.

## Major

**M-1. Embedding premise is wrong (coordinator ONNX check).** §2.1, M1, H1, C1 note, §9 assume $h_e$ affine in
planes. It is local and mish-nonlinear. Consequence the synthesis misses: at $e$ a non-local family (x-ray,
threat, route, mate) can enter $h_{e,s}$ only via correlation with the square's own planes. In M1-IG/M6/M9 at
$e$, a board edit changes non-local bits on squares whose planes did not change; the decoder assigns
$W^G\Delta\tilde f^G_s\neq0$ there and $\varepsilon$ cancels it exactly — spurious named credit plus equal and
opposite residual, by construction. Fix: at $e$ restrict to a local vocabulary, or zero decoded deltas on
squares with unchanged planes; redefine C2 as "local vocabulary". The 0.002 FVU then says the vocabulary
contains each square's occupant; history/rule-50/castling broadcast planes go to $\varepsilon_e$.

**M-2. "Peak at L1-L2" depends on the denominator.** §2, Th8, H2. Features-add is a share of *total* variance,
capped by the square-only FVU. Normalised by non-square variance (partial $R^2$ = add / square-only FVU):
$e$ 0.99, L0 0.68, L1 0.76, L2 0.76, L5 0.53, L9 0.24 — highest at $e$, a local bump at L1-L2 relative to L0,
monotone after. "Informative single site is L1-L2" is not supported by the table alone. Fix: report both
quantities; drop the single-site claim.

**M-3. Confirmation bias in Th8.** Lomasov [T8.14] (probe accuracy, 6 concepts, different 270M model,
Creative AI track) is offered as matching the decoder-FVU decay. The project's own encoder result (families
readable at $e$, some *gaining* to L5) is the more relevant evidence and points the other way: features do not
become unreadable, they shrink as a share of token variance. Fix: say so; Lomasov supports probe decay on
another model, not our FVU decay.

**M-4. Missing alternative explanations for the decay.** §2.3 lists (a) non-locality, (b) map class,
(c) unnamed. Also: (d) dilution — features stay present (probe result) while other content grows;
(e) post-LN renormalisation; (f) output-specific content (policy move-scores per square) the value target
never uses; (g) fit capacity, ~30k bits at 20k positions. (d)-(f) mean FVU decay ≠ loss of explanatory power
*for $y$*; only M6/M9 measure the latter. Fix: add (d)-(g); M12 budget gets a "variance not used by $y$" read
via M6.

**M-5. M6 telescoping over-reads.** $\phi^{(L)}$ is exact at each tap, but $\rho_L$ and $\rho_{L+1}$ use
independently fitted decoders on different tokens; attention mixes squares and post-LN renormalises between
taps. $\kappa_L$ is a difference of two numbers, not a flow through block $L$; it also moves with decoder
quality (ridge α, data). $\rho_L$ is unstable when $\Delta^{\rm true}\approx0$. Compute: two new suffix passes
per tap, not 3-4. Fix: aggregate as $\sum\phi_n/\sum|\Delta|$; call $\kappa_L$ a descriptive difference; require
C3 stability before reading block-level structure.

**M-6. Method count over-claims distinctness and "multi".** M1, M2, M6, M9, M10 all apply different removal
operators to one decomposition $h=q+\sum n^G+\varepsilon$; M6 is M9 with players merged. M9 and M10 are per-layer
(M10 just sweeps in one pass), yet the table marks them "multi". Distinct approaches are ~6: decoder-split
(M1/M2/M6/M9/M10), erasure (M3/M8), probe gradient (M4), alignment (M5), replacement graph (M7),
residual analysis (M12); M11 aggregates. The user's "several decoders on several layers" in the strict sense
is only M7 and partly M8 (which uses probes). Fix: state the grouping; correct the Multi column; add a
decoder-read variant of M8.

**M-7. Family-level collinearity remains.** Grouping into families (Th6, M9) solves within-family
non-identifiability only. Families are collinear with each other (occupant vs every occupancy-derived
family), so the split of $W$ into $W^G$ is still ridge-dependent at family level. Fix: say so in M9; C3 must
report family-rank stability, not bit-rank.

## Minor

- Th1: Kumar [T1.7] critiques Shapley (off-manifold vs credit-to-correlates); the "uninformative unless tied to
  an end-task" claim is Bilodeau's [T1.17] alone. M9 cites [T1.7]'s conditional-Shapley critique for an
  interventional method; cite the off-manifold half.
- Th5: Lin et al. [T8.7] label features by hand and *check* against rule-based detectors (annotation), not
  "labelled with detectors".
- Th2/Th6: Pálsson & Björnsson [T8.3] recommend nonlinear probes; synthesis omits this while prescribing
  linear-primary. Note the tension.
- CP-003 resolution: Tomlin's terms include atari (board-computable, tactical), so "outcome-adjacent" is
  shaky. Mark the resolution as hypothesis.
- M10: LRP relevance is zero-referenced; the split ratio is baseline-differenced. Use one baseline throughout
  (DeepLIFT/G-DeepSHAP rescale).
- M1 point form is not complete; label it a sensitivity map, not an attribution with residual.
- GT-M edits: lc0 inputs carry 8 history frames; an edited board needs consistent history (apply the edit to
  every frame or use history-free inputs for both boards), else $\Delta^{\rm true}$ includes a history artefact.

## Spot-check of citations (12)

| Claim in 03 | Biblio annotation | Verdict |
|---|---|---|
| Zhou [T2.12] 65.9% residual | same | OK |
| Engels [T5.9] ~half predictable; earlier layers reduce most | same | OK |
| Sutter [T7.12] near-perfect IIA on random nets | same | OK |
| Ramaswamy [T2.15] ~32 concepts | human-study cap ~32 | OK |
| GCAV [T2.6] only cross-layer concept-map fusion | same | OK |
| Bai [T2.2] CG pseudo-inverse, residual dropped | same | OK |
| Sandmann [T8.6] late override | same, Tier 3 | OK |
| Jenner [T8.5] look-ahead in heads, causal validation | patching + head ablation | OK |
| Makelov [T4.15/T5.11] additive attribute dictionary | same | OK |
| Lomasov [T8.14] 85% → 50-65% | same; but probe accuracy, 6 concepts | Accurate, misapplied (M-3) |
| Lin [T8.7] labelled with detectors | hand-labelled, detector-checked | Minor error |
| Kumar [T1.7] uninformative unless end-task | that is Bilodeau | Minor misattribution |
| Pálsson [T8.3] linear accuracy unreliable | also recommends nonlinear probes | Selective |

## Strongest counter-argument

"Five of the twelve methods are one ridge decomposition read with different operators. Where the decomposition
is complete, faithfulness to lc0∘D is guaranteed and the only informative outputs are the residual and the
regulariser-dependent family split — so the paper's empirical content reduces to M6's curve plus stability
checks, and at the embedding even that curve is set by locality, not learning."

## What's missing

- Treatment of history planes (in the embedding and in edits).
- A decoder that conditions on other squares (listed as a gap, but it is also the direct test of cause (a)).
- Any evidence that FVU decay corresponds to decay in output-relevant content; only M6 can supply it.

## Stress test

| Test | Result |
|---|---|
| Remove strongest source (Makelov) — holds? | Yes; the decomposition is defined without it. |
| Flip RQ ("no route is faithful at depth") — credible? | Yes; M6 must be allowed to show $\rho_L\to0$. |
| Different context — generalises? | Yes, to any model with a symbolic extractor; locality argument is lc0-specific. |
| "So what?" | Yes, after C-1: otherwise the protocol would select against the residual. |

---

## Changes applied to `03_synthesis.md` (one revision loop)

1. §2 rewritten: embedding is local + mish (ONNX), normalised partial-$R^2$ row, L1-L2 single-site claim
   dropped, decay causes (d)-(g) added. (M-1, M-2, M-4)
2. Th8: Lomasov reframed; own probe result cited as counter-evidence. Matrix tier note. (M-3)
3. Th1 Kumar/Bilodeau, Th5 Lin wording, Th2 Pálsson tension, CP-003 row marked hypothesis. (Minor)
4. C2 redefined as local vocabulary; M1 failure modes/controls for locality and point-form completeness. (M-1)
5. M6: $\kappa_L$ descriptive, aggregate ratio, 2 passes/tap. (M-5)
6. M8: decoder-read variant. M9: family-level collinearity; [T1.7] use fixed. M10: one baseline. (M-6, M-7, Minor)
7. M11 recast as a disagreement map; learned/ρ-weighted ensemble removed. (C-1)
8. §3 intro: approach-family grouping; §4 Multi column and count corrected. (M-6)
9. §6: metrics split into coverage vs error; H1, H2, H4 rewritten; history handling in GT-M. (C-1, M-1, Minor)
10. §9: affine bullet replaced by verified locality.
