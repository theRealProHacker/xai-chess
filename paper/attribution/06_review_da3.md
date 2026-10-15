# Devil's Advocate Report — Checkpoint 3 (final report `05_report.md`)

Reviewed against `00_context.md`, `01_scoping.md`, `04_da_checkpoint2.md`. All DA2 fixes (C-1, M-1 to M-7, minors) are present in 05. The issues below are new.

## Verdict: REVISE

No critical issues. Four major issues: the report never says which method to run first, and its central framing answers a coverage question where the user asked an attribution question. Both can be fixed in text.

---

## Critical
No critical issues identified.

## Major

**MJ-1. No first recommendation.**
- Type: Scope / usefulness.
- Location: §9.4 Table 6, §10.
- Problem: Stage A bundles five methods (M6, M2, M3, M8, M9) as "one CPU day". The report never names the first experiment, the edit set it runs on, or the result that would change the plan. The user asked for several approaches and for a pick to try first. The report gives the several approaches and leaves out the pick.
- Recommendation: add a short "Run first" paragraph naming one experiment end to end. Suggested: M2 plus M6 on pin edits (move the pinner off the line, or interpose). Run it at taps e (with C2), L2, L5 and L9 on about 1,000 pairs, with C1 and C3. Output: the per-family $\Delta^{(L)}_G$, $\gamma_L$ and $\rho_L$. Pins are the obvious choice because they connect to MaskPinner (GT-C), to the existing pin probe (AUROC 0.945 at L8) and to clean edits. State what result would promote M8 next, and what result would promote M5 next.

**MJ-2. Central framing: $\rho_L$ is a coverage curve, not an attribution.**
- Type: Framing / substitution of question.
- Location: Abstract, §1 third finding, §5.4 last paragraph, §10 second answer ("That curve is the hybrid").
- Problem: M6 has two players, named and residual. It never says which human feature moved the value. The per-family answer the user wants comes from M2 and M9, and those carry the family-collinearity caveat (DA2 M-7). The front endpoint is also fixed by construction: at $e$ with local edits, $\varepsilon_e'-\varepsilon_e\approx0$, so $\rho_e\approx1$ whatever lc0 has learned. The back endpoint is close to the board-level-decoder result already in Table 1, which was weak. So "one curve replaces the hybrid" turns *where to attribute* into *how much is nameable*. It does not resolve *how to attribute*.
- Recommendation: present $\rho_L$ as the guide for choosing taps, not as the attribution. Say that per-family attribution comes from M2 or M9 at taps where $\rho_L$ is high and where $e$'s locality no longer holds, which predicts L1–L5. Say that $\rho_e\approx1$ holds by construction.

**MJ-3. Family-level credit is regulariser-dependent for exactly the families of interest, and the report does not say what is identifiable.**
- Type: Method / identifiability.
- Location: M2, M9, M1, Table 4 "Identifiability" column ("unique per joint patch").
- Problem: every real edit that changes a pin also changes occupancy, attacks and x-rays. $W^G_L$ is split among collinear families by ridge. So "pin got 0.12" from M2 or M9 is partly an artefact of $\alpha$. Only joint patches and the named/residual split are identifiable. C3 detects the instability but cannot remove it. The "So what?" for the researcher depends on this point and the report leaves it scattered.
- Recommendation: add one paragraph stating which quantities are identifiable (joint named effect, $\gamma_L$, $\rho_L$, LEACE effect of $G$ *together with its linear correlates*) and which are not (the single-family split among occupancy-derived families). Offer one remedy: attribute to the edit, describe the edit by the families it changes, and report family credit only where the C3 ranks are stable.

**MJ-4. The most obvious baseline is missing, and so is the policy side.**
- Type: Missing evidence / scope.
- Location: §6, §9; Table 1 row "Board-level decoder…".
- Problem: (a) A surrogate from features (or from probe outputs at $L$) to $y$, with Shapley over families and an explicit residual (post-hoc CBM-h, Yuksekgonul et al. 2023, already cited), is the first thing a reviewer will ask for. The project already has a version of it (FVU 0.52–0.62, still falling with data), yet it is not a method or a baseline. (b) Policy appears only as "$y$ = one move's logit". lc0's policy logit for a→b is formed from the from- and to-square tokens at the last layer, so a per-square decoder has a natural two-square localisation there. The report does not use this, and H1–H6 are all value-flavoured.
- Recommendation: add the surrogate as M0, the correlational baseline that every interventional method must beat (it is cheap and needs no port). Add one paragraph on policy: from/to-token attribution at L9, plus the fact that policy-relevant content may be the residual that the value target ignores (cause (f)).

## Minor
- §1 "First … 'mostly square identity' describes the front, not the back." The user's note says the back has "near-perfect reconstruction", which Table 1 contradicts (L9 FVU 0.68). The user may mean probes or the MLP decoder. Flag the ambiguity instead of correcting the user.
- §5.2 "a decoder would fit [a random embedding] almost as well" is stated as fact. It is a prediction for C1, so label it that way.
- History planes: the data are Lichess puzzle FENs, which carry no history, so lc0 already runs with synthetic history. This likely explains part of FVU 0.002 and belongs in the limitations. The GT-M rule "apply to every frame" is moot on FEN-only input.
- No concrete worked example of the output a researcher would get, e.g. one position: "$\Delta$value −0.31 at L5: named 0.18 (pin 0.09, hanging 0.05, …), residual 0.13". The user profile asks for concrete examples, and an illustrative mock-up flagged as hypothetical would do.
- Table 4 marks M6 "yes (sweep)" for combining layers. Per DA2 M-6 it applies independent decoders and composes nothing. Use "sweep" and drop the "yes".
- Combination omitted: compose M8's family-transfer tensor $T^{L\to L'}$ with M2's last-tap effects into a feature-level path graph that does not freeze attention, as an alternative to M7. This is a cheap, interesting multi-decoder idea under ONNX.
- Section 7 note: "Five run without a PyTorch port" includes M9 but omits M11. Trivial.

## Observations
- §10's reply to the strongest objection is honest and well placed.
- The limitations are genuine, not performative: nothing has been run, Tier-2 dependence is stated, and GT-C is kept apart from attribution faithfulness.
- The AI disclosure is adequate.

## Strongest counter-argument
"The report answers 'how much of lc0 is nameable per layer', not 'which human feature caused this value'. Its headline curve is trivially 1 at the front and repeats an existing weak result at the back. The per-feature attributions it does offer (M2, M9) split credit among collinear families according to a ridge penalty. A researcher who wants 'the eval dropped because of the pin' gets a regulariser-dependent number and no instruction on where to start."

## What's missing
- A named first experiment and the decision rule that follows it (MJ-1).
- A feature-to-output surrogate baseline (MJ-4a).
- A policy-specific treatment (MJ-4b).
- An explicit identifiability statement for single-family credit (MJ-3).
- One illustrative output example.

## Answers the user's three-part question?
| Part | Covered by | Verdict |
|---|---|---|
| Decoder at embedding | M1, §5.2 locality, C2 | Yes. The reduction to input saliency is clear and correct |
| Encoder (probe) at any layer | M3, M4, M5 | Yes |
| Combinations across layers | M6 (sweep), M8 (cascade, decoder-read), M7 (graph), M11, M12 | Partly. Three genuine combinations, plus one cheap omission (M8∘M2 path composition) |
| Several interesting, implementable approaches | Six families, costs, ONNX-vs-port split | Yes |
| Clear "try first" | none | **No** (MJ-1) |

## Stress test
| Test | Result |
|---|---|
| Remove the strongest source (Makelov) — does the argument hold? | Yes |
| Flip the RQ ("no route attributes per family identifiably") — is the opposing view credible? | Yes. That is MJ-3 |
| Different context — does the finding generalise? | Mostly. The locality argument is specific to lc0 |
| "So what?" — is the significance justified? | Partly. Coverage yes. Per-feature attribution needs MJ-1 to MJ-3 |
