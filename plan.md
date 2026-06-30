<!-- /autoplan restore point: /home/rharvey/.gstack/projects/xai-chess/unknown-autoplan-restore-20260625-000335.md -->
# ChessFaith — Final Plan

**Status:** APPROVED · Compiled 2026-06-24 · ⚠️ 2 OPEN items (O2 scope, O3 framing); O1 resolved 2026-06-25 — see §14
**Compiled from:** the office-hours design doc, the `/autoplan` review (CEO + Eng + DX), and the deep-research landscape report (all 26 cited works read in full).
**Full detail lives in:**
- Design doc + review + decision audit: `~/.gstack/projects/xai-chess/rharvey-unknown-design-20260624-104553.md`
- Landscape report + Full-Read Addendum: `~/.gstack/projects/xai-chess/chessfaith-landscape-deepresearch-20260624.md`

---

## 1. What ChessFaith is (scoped)

ChessFaith is a **benchmark for explanation engines.** Given a position, a justified move, and an explanation that cites typed factors (`pin`, `passed_pawn`, …), it measures whether each cited factor is **causally load-bearing for a strong oracle engine's decision** — by intervening on the board or rules and reading the change in the engine's *policy margin*. The chess engine is the **behavioral oracle of ground truth**; its internal computation is out of MVP scope.

What it measures is **grounding / causal-correctness of cited factors against an oracle** — NOT faithfulness to the explainer's own internals (BonaFide's quantity, which chess does not solve either). The defensible contribution, stated honestly:

> Chess supplies **verifiable decision-level causal ground truth** — whether a cited factor actually drives a strong oracle's move — which the text-domain faithfulness benchmarks (BonaFide, ICE, Causal Diagnosticity, RFEval) can only approximate with *unverified* proxies. ChessFaith is the testbed where their intervention-consistency machinery can be validated against checkable causal structure.

**Novelty claim (precise, don't over-concede):** first instantiation of intervention-consistency on an oracle-backed, non-text substrate (win-probability deltas as treatment effects) with multiple intervention surfaces whose agreement is itself measurable. Position with BonaFide's vocabulary; cite ICE / BonaFide / RFEval as the method ancestors.

## 2. Settled decisions (do not re-litigate)

| # | Decision | Source |
|---|----------|--------|
| 1 | Benchmark for explanation engines; behavioral oracle = ground truth; internal computation out of MVP | premise gate |
| 2 | Strength-vs-faithfulness is NOT the central question; peripheral (experiment C) | premise gate |
| 3 | MVP = behavioral surfaces only; mechanistic + surface-agreement in **backlog** | taste 1 |
| 4 | Keep rule-level surface, **gated on a board-level-agreement (Spearman) check** before any "confound-free" claim | taste 2 |
| 5 | Adopt BonaFide grounding/oracle vocabulary + cite it; project name stays ChessFaith | taste 3 |
| 6 | Approach A (ChessFaith) is the spine; mode SELECTIVE EXPANSION | CEO |

## 3. Architecture

```
  explanation record (JSONL, oracle-graded)
        │  {fen, justified_move_uci, cited_factors:[{type,args}], source}
        ▼
  parse ──► Factor DSL  ──detect/to_nl──►  remove(engine, board, factor)
        │   (typed predicates)                    │
        │                          ┌── rule-level ─► Stockfish (custom movegen)   [gated on agreement]
        │                          ├── board-level ─► any engine (MVP universal surface)
        │                          └── mechanistic ─► one pinned Leela net          [BACKLOG]
        ▼                                    │
  outcome reader: policy margin  ◄───────────┘   ──►  eff(f) = outcome(s,m) − outcome(s',m)
  WP(s,m) − max_{m'≠m} WP(s,m')   (cached)              │
        │                                              ▼
  meta-eval harness (labeled pairs) ──►  ICE stat stack ──► stratified AUROC + win-rates/CIs
                                          + matched placebo      vs BonaFide 0.70 bar + baselines
```

**Components**
- **Factor DSL** — typed predicates; per type: `detect(board)->[Factor]`, `to_nl(factor)->str`, `remove(engine,board,factor)->position`. **Tier factor types**: rule-clean vs board-requires-validity-proof. Ship `validate_factor_type()` that runs the edit-shock diagnostic and refuses to score a new type until shocks are within the pre-registered tolerance.
- **`Engine` protocol** — `win_prob(fen, move)` required; `supports_rule_level` / `supports_activation` capability flags. Oracle-swap = implement one interface.
- **Outcome reader** — policy margin (not raw WP). Cache `WP(engine, fen, move)` (cuts cost + enables zero-install repro).
- **Counterfactual generator** — edits must be legal, side-to-move-preserved, material-neutral; reject positions with no valid edit and **report the retention rate** (the real meta-eval N is what survives); add a same-engine off-distribution stability check.
- **Error contract** — invalid edit / net-hash mismatch / unparseable explanation → **quarantine with the reason in the output record, never a silent float.**

**Net pinning (corrected from full-text reads — these are three different nets):**
- Look-ahead (Jenner) + iterative-inference (Sandmann) bind to **Leela T82-768x15x24h (finetuned, no-history)** → this is the net for the mechanistic backlog.
- Lin 2026 circuit-tracing binds to **BT4**; Ruoss is a **separate action-value transformer** whose **AV head gives a clean per-move Q̂ usable directly as `wp()` — no centipawn conversion.**
- **Pick ONE net hash; every surface uses it.** Maia-2 SAEs cannot be ablated inside lc0 (different net) — do not mix.

## 4. The metric

- **Per-factor effect:** `eff(f) = outcome(s,m) − outcome(s',m)`, `outcome` = policy margin.
- **Statistics — adopt ICE's stack wholesale** (don't reinvent): Normalized Score Retention; randomization test (M≈50 perms); **win-rate as the primary statistic** + Cohen's d + bootstrap 95% CI (B≈200) + Benjamini-Hochberg FDR (α≈0.10); **≥2 intervention operators** (rule / board [/ mechanistic]) with **operator-comparative reporting** — agreement = evidence, gap = informative uncertainty. **Surface disagreement is a headline result, not a bug** (mechanism: Sandmann's internal≠behavior).
- **Placebo:** matched-pairs / propensity design — match placebo to cited factors on factor type, pieces involved, distance-to-move, raw saliency (removes the selection bias where cited factors sit near the action).
- **Edit-shock pilot:** pre-register the acceptance threshold (median material delta = 0; eval-shock on irrelevant factors below X cp) **before** running; demonstrate at least one structural factor passes it.

## 5. Validation / meta-evaluation (the heart — this is where the value is)

- **Position set:** forced-tactic (tablebase/forced-mate verified) **plus graded, non-saturated pairs** (eval +1..+3, cited factor load-bearing but not the only thing; unfaithful = a plausible-but-non-causal present factor). **Report stratified AUROC** by position type (forced-tactic / tactical / positional) — a metric that only works on forced mates validates nothing.
- **Power:** state the effect size and run the power calc *before* building; AUROC is primary; the per-explanation permutation test is descriptive only (discrete-null floor at small placebo sets).
- **Baselines to beat:** random-factor, a saliency baseline, **and the 8 BonaFide metrics ported as real baselines** (CC-SHAP, Filler Tokens, SCM, FUR, Early Answering, Adding Mistakes, Simulatability, Paraphrasing). Report AUROC at **step≈factor** and **CoT≈whole-explanation** with DeLong CIs; clear/cite the **0.70** field bar.
- **Determinism (precondition):** CI test asserting identical WP across two runs *before* any faithfulness value is trusted.

## 6. Inputs / explanation sources

- **Wire format:** frozen, versioned **JSONL** `{fen, justified_move_uci, cited_factors:[{type,args}], source}` + a JSON Schema + 2-3 golden records.
- **Two entry points:** `score(explanations)` (no ground truth, any positions — works day one) vs `meta_eval(labeled_pairs)` (needs ground truth).
- **Phase-1 sources:** hand-authored records (skeleton) → **C1 (Tang 2026, arXiv:2603.20510)** as the real source — it natively cites fork/pin/skewer with squares and self-admits its traces "may not faithfully capture the logic" — **plus a faithful-by-construction positive control.** **Do NOT use CCC for tagged extraction** (CCC = free-text prose over coarse Stockfish eval-terms; pin/fork were excluded for lack of labels — use only its probe-delta prioritization technique). Free-text factor extraction is deferred.

## 7. Reproducibility / DX (not deferrable)

- Pin net checkpoints by hash; **ship pre-computed WP caches** so the AUROC reproduces with zero engine installs.
- **CPU-only quickstart:** rule-level Stockfish + cached-WP meta-eval AUROC reproduces in ~10 min on a laptop (no GPU). This is the default `reproduce` target.
- **Dockerfile** with the movegen-patched Stockfish; `chessfaith reproduce --benchmark v0.1`.
- **Frozen versioned position set `v0.1`** (manifest hash) + a held-out split (leaderboard integrity); expected number + tolerance in the README.

## 8. Reuse map (don't reinvent) — with sources

| Prior work | Relation | Use it for | Code/data |
|---|---|---|---|
| **ICE** (2603.18579) | REUSE template | The full statistical stack (win-rate + d + bootstrap CI + BH-FDR + operator-comparative) | ICEBench (on arXiv page) |
| **BonaFide** (2605.25052) | REUSE ancestor | 8 baseline metrics; AUROC protocol; 0.70 bar | hf.co/yoavgurarieh/bonafide |
| **Causal Diagnosticity** (2502.18848) | IMPROVE-ON | The diagnosticity metric D(F); its ground truth is *unverified* → ChessFaith's edge | github.com/KeremZaman/CausalDiagnosticity |
| **RFEval** (2602.17053) | REUSE def | The κ "causal influence under output intervention" formalism; accuracy≠faithfulness | released (Apache-2.0) |
| **Ruoss** (2402.04494) / **Leela-CF** | DEPEND (oracle) | AV head = clean `wp()`; superhuman oracle | github.com/CSSLab/maia3 (Leela-CF) |
| **Maia-2 / Maia-3** | REUSE calibrator | Per-rating success probabilities = difficulty calibration (no NL channel) | github.com/CSSLab/maia3 |
| ALMANACS / ConSim / Chen / Hase&Bansal / Shymanski | DEFER (ChessSim) | Simulatability backbone + the ALMANACS cautionary null | — |

## 9. NOT in scope (deferred, written down)

- **Mechanistic surface + surface-agreement** — backlog; binds to net T82-768x15x24h; the genuinely hard-to-scoop result, pursued after the behavioral MVP lands.
- **`pip install` / CLI polish / public leaderboard** — defer (OK).
- **ChessSim teaching/simulatability track** — LLM student (per ALMANACS/ConSim), with the mandatory ALMANACS validation (confirm the student can exploit a known-good oracle explanation, and design the train/test shift so in-context matching can't substitute). Shymanski's actionable-explanation transfer metric is the design pattern.
- **Free-text factor extraction; multi-factor Shapley/joint ablation; diffuse factors (king safety).**
- **Strength/fluency-vs-faithfulness as a headline** — peripheral experiment C only.

## 10. Phased roadmap (the 13 tasks, mapped)

| Phase | Goal | Tasks | Done-criterion |
|---|---|---|---|
| **0. Setup & decisions** (days) | repo + env + engines; PIN one net hash; lock outcome=policy-margin, determinism settings, pre-registered edit-shock threshold | T1, T5 | engines run; net hash + settings recorded |
| **1. Walking skeleton** (days→1wk) | `pin` factor (detect/to_nl/rule-level remove) + `Engine`/Stockfish adapter + policy-margin reader + cache + `eff()` + error contract + JSONL schema + 10 golden positions + determinism test | T1, T6, T8, T10, T17 | **deterministic, clearly-positive `eff(pin)` on 10 positions** |
| **2. Board-level + agreement gate** (1-2wk) | board-level `pin` remove + counterfactual validity + retention rate; Spearman rule-vs-board → earn the "confound-free" claim or drop rule-level | T4, taste-2 | surfaces agree (or rule-level demoted) |
| **3. Meta-eval + statistics** (1-2wk) | graded non-saturated forced-tactic set (~50-100, verified) + matched placebo + ICE stat stack + BonaFide baselines → stratified AUROC; power calc first | T2, T3, T9 | **AUROC above chance, beats baselines, vs 0.70 bar** |
| **4. First result** (1-2wk) | fluency ≠ faithfulness: 2 real sources (C1 + 1) + faithful-by-construction control; faithfulness-vs-fluency plot | experiment C | the plot |
| **5. Reproducibility/packaging** | Docker (movegen-patched Stockfish), pinned weights, WP caches, frozen `v0.1` split, `reproduce` command, README expected number | T7, T11 | CPU repro in ~10 min |
| **Follow-on (backlog)** | mechanistic surface (net T82) + measured surface-agreement; ChessSim track | T12-T13 | — |

## 11. Success criteria (MVP gate)

1. A metric that, on the graded forced-tactic meta-eval, separates ground-truth-faithful from unfaithful **above chance** — stratified AUROC + per-explanation p + a power check — and **beats random + saliency + the BonaFide baselines**, reported through the ICE statistical layer.
2. The faithfulness-vs-fluency plot across 2 sources + the control.
3. A reusable artifact: position set `v0.1` + a C1-sourced explanation set + the intervention harness, **CPU-reproducible** with an expected number.
4. One workshop paper from Phases 1-4, positioned via BonaFide/ICE vocabulary.

## 12. Open items to confirm (before submission)

- Caïssa (Soliman & Ehab, Springer KI 2025) — paywalled, unread; confirm the DOI manually.
- Preprint venue attributions (NeurIPS/ICLR/NAACL/EMNLP/PNAS) — confirm against proceedings.
- The exact net hash for the pinned oracle/target net.

## 13. Citation fixes to apply to the design doc's references

1. **arXiv:2508.21380** = Sandmann, Lapuschkin & Samek, *Iterative Inference in a Chess-Playing NN* (v1; v3 retitled *The Algorithm Is Not the Behavior: Learned Priors Override Look-Ahead*). The doc's cite is **correct**; add the v3 title parenthetically.
2. **CCC (Kim, 2410.20811)** — reclassify as a probe-technique + coarse concept-list source, **not** a concept-tagged commentary corpus.
3. **Causal Diagnosticity (2502.18848)** — its ground truth is *unverified*; narrow the doc's "the field lacks ground truth" claim accordingly.
4. **Ruoss (2402.04494)** — retitled v1→v2; AV head = clean `wp()`.
5. **Chessformer (2605.19091)** — "Maia-3" is only its human-aligned instantiation; gloss as "Chessformer (Maia-3 = its human-aligned net)."
6. **Jiao Maia-2 SAE (OpenReview Wxl0JMgDoU)** — **rejected at ICLR 2025**; cite as an OpenReview submission, not "ICLR 2025."
7. **McGrath (2111.09259)** — explicitly correlational; do not credit with causal validation.
8. Pin the exact Leela nets (T82 vs BT4 vs Ruoss's transformer) wherever the doc says "Leela" generically.

## 14. Re-review open items — /autoplan re-run 2026-06-25 (single-voice; Codex usage-limited until 2026-07-18)

A second `/autoplan` ran on 2026-06-25. Codex was again unavailable (auth OK, but usage limit hit, resets 2026-07-18), so this was a **Claude single-voice pass, not the cross-model gauntlet.** A fresh independent CEO/strategist voice (no prior-review context) re-opened three of the "settled" decisions in §2. They are recorded here as **OPEN for an explicit decision, not auto-applied:** one model re-challenging an already-settled call is weaker than a two-model User Challenge, and these touch the contribution's core. **Defer the real cross-model re-review to after 2026-07-18.** Full review text lives in this session's transcript; the pre-re-review restore point is linked at the top of this file.

- **O1 [CRITICAL] — re-opens Decision #5 (name stays ChessFaith) + §1 framing.** The benchmark grades cited factors against an *oracle* engine's decision, explicitly not the explainer's own internals. BonaFide / ICE / RFEval define "faithfulness" as correspondence between an explanation and the computation that produced it; grading against an oracle the explainer never ran is *grounding / factor-correctness*, not faithfulness (a distinction the plan already cites via Shymanski; also Jacovi & Goldberg). Risk: a subfield reviewer reads "faithfulness," checks the definition, and rejects for a category error. §2 settled this as "adopt BonaFide vocabulary, keep the name." The fresh voice's objection: that knowingly ships a gap you've already spotted. **RESOLVED 2026-06-25 (user): scratch the word.** Drop "faithfulness" as the measured-property claim; coin no replacement for now; the project is called **xai-chess**. State plainly what it measures — "whether each cited factor is causally load-bearing for the oracle's decision" — with no property-noun to over-claim. (O2 and O3 remain open.)
- **O2 [CRITICAL] — re-opens Decision #3 (behavioral-only MVP; mechanistic + surface-agreement in backlog).** By the plan's own threat model (landscape §4.3) the un-scoopable result is multi-surface agreement against verified ground truth, and the scoopable result is the behavioral-only metric. The MVP ships the scoopable part first and defers the moat; a competitor with the public ICE/BonaFide stack can run the behavioral chess instantiation in ~3 weeks. The fix does **not** require pulling the mechanistic net forward: Phase 2 already has two behavioral operators (rule-level + board-level). Proposed fix: elevate the rule-vs-board *operator-agreement* result to the headline claim, not a Phase-2 validation gate; the mechanistic net becomes a third operator that strengthens, not founds, the result. **DECISION NEEDED:** keep the behavioral metric as headline, or promote operator-agreement.
- **O3 [CRITICAL] — re-opens Decision #1's positioning ("oracle ground truth = the contribution").** The verifiable ground truth is the *apparatus*, not the finding: the intervention is simultaneously the measurement and the label, so "we have ground truth" is not itself a result, and the meta-eval is partly circular on the saturated set. Proposed 10x reframe: position chess as the *calibration lab for the text field's own metrics* — run BonaFide's 8 metrics and ICE's protocol on chess where the true causal effect is computable, and report which text-domain faithfulness metrics are systematically biased and by how much. This uses the ground-truth property as a measuring instrument and is scoop-resistant. **DECISION NEEDED:** keep "reusable causal benchmark" as the contribution, or adopt "calibrate/correct the text-domain metrics" as the headline.

The three are mutually reinforcing: a single reframe (a verifiable causal-grounding apparatus, used to calibrate the text-domain faithfulness metrics, with operator-agreement in the MVP headline, named distinctly from "faithfulness") resolves all three.

**Secondary findings from the same voice (HIGH; do not re-open settled decisions — recorded for the cross-model pass):**
- The oracle-as-ground-truth premise can mark *correct human explanations unfaithful* where the oracle wins regardless of the cited factor. Run a ~20-position adversarial check (expert-validated good annotations on already-winning positions) before Phase 3, as a pre-registered kill-criterion. (H1)
- Name one external explanation engine scored end-to-end as an MVP success criterion; design the wire format to ingest C1's native fork/pin/skewer-with-squares output with zero translation. A benchmark with no external adopter is a null result. (H3)
- Before Phase 4, verify C1 actually emits a non-trivial rate of *non*-load-bearing cited factors on your position set, or the faithfulness-vs-fluency plot has no contrast. (M4)
