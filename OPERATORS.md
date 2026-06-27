# OPERATORS.md — Intervention operators & grading policy for xai-chess

Design doc (2026-06-27). Distilled from the "all possible reasons for a move" analysis.
**Status: design, not yet implemented.** Implemented today: `pin` (rule+board), `hanging` (board).

## 0. Core principle

The benchmark grades a cited *reason* by intervening on it and reading the oracle's policy-margin
change (`eff`). The intervention exists for one purpose: **to manufacture decision-level causal
ground truth where none otherwise exists.** Two consequences drive this whole doc:

1. One operator (`remove`) is not enough. Different *classes* of reason need different operators —
   together they form a small **operator algebra** (§2).
2. Where ground truth *already* exists (opening theory, tablebase endgames), the intervention is
   redundant; grade by **detection against the known reason** instead (§4).

The DSL interface stays: each factor type supplies `detect(board) -> [Factor]`,
`to_nl(factor) -> str`, and one or more *operators* below (generalizing today's
`board_level` / `rule_level`).

## 1. The operator algebra

| Operator | Mechanism | Engine cost | Confound profile |
|---|---|---|---|
| `remove-board` | material-neutral relocate / add-defender | none (board edit) | edit-shock (gate) |
| `remove-rule` | suspend the rule that creates the factor (`MaskPinner`) | movegen patch (have it) | **confound-free** |
| `obstruct` | inject a *magic blocker*: a phantom occupant that blocks a ray but is uncountable / uncapturable (`LineBlocker=<sq>`) | movegen+eval patch (new) | near-clean (occupies 1 square) |
| `tempo` | flip side-to-move = a null move / surrendered ply (`nullzug`) | none (FEN edit) | **confound-free** (no board change) |
| `threat` | forced-mate / threat search *before vs after* the move | none (search) | **confound-free** if solved independently |
| `restructure` | relocate a pawn keeping pawn count constant | none (board edit) | edit-shock (gate) |
| `ablate-piece` | delete a piece, net out its base value | none (board edit) | large shock (king/structure) |
| `book` | opening-book / theory lookup | none | detection only, not causal |
| `detect`-tier | factor existence / match against the *known* reason | none (+ Syzygy probe) | n/a — see §4 |

### Operator notes
- **`obstruct` (the magic blocker) is the dual of `MaskPinner`.** `MaskPinner` *removes* a line
  constraint; `obstruct` *adds* one, with no material change. It generalizes to **every line-based
  reason** with a single operator: open file (rook), open diagonal (bishop), battery, the rear line
  of a discovered attack, "the line to my king is open." Place the blocker on the interposition
  square that minimally perturbs other features; keep the edit-shock gate because it still occupies
  a square (can block a second line through it, shrink a king's escape set).
- **`tempo` ("nullzug" / "surrender a ply")** = edit the FEN's active color (and clear en-passant).
  Unlocks **zugzwang** (let the stuck side pass — if its eval jumps, zugzwang was load-bearing) and
  **initiative** (give the opponent the free ply — if the advantage evaporates, the initiative was
  load-bearing). **Legality guard:** reject if the side now *not* to move is left in check (that is
  an invalid position, not a pass). It grants a *full* tempo, so read it as a sign test, not a
  calibrated magnitude.
- **`threat` (prophylaxis, e.g. "prevents mate on h7")** = does the opponent have a forced threat
  *before* the move that is gone *after*? Grade faithful only if the threat existed pre-move and the
  move removes it. Solving with an **independent from-scratch mate-solver** (not the patched engine)
  makes the label **non-circular** — this is the verified-label route in the README's Next-steps #3.
- **`restructure`** rescues pawn structure from the material-neutrality wall: *relocate* a pawn
  rather than add/remove one (a-file isolani → b-file beside a friend; spread a doubled pair; retract
  a passed pawn). Define the feature on the standard scalars: **pawn-island count, isolated / doubled
  / backward flags, passed-pawn set.** Legality: no rank 1/8, resulting position legal, STM preserved.
- **`ablate-piece` (activity)** = `activity(P) = [WP(pos) − WP(pos − P)] − base_value(P)`; a large
  positive premium = an active piece. Cleaner for grading a *move's* reason is the `remove-board`
  variant: relocate the piece to its most passive legal square and read the move's value drop.
- **`book`** answers "is it theory," which is existence, not load-bearing (and a strong oracle already
  plays book-quality moves). Use as a tag/filter, never in the causal metric.

## 2. Reason → operator map

| Reason | Operator | Tier |
|---|---|---|
| pin (abs/rel) | `remove-rule` | confound-free ✅ done |
| hanging / en-prise | `remove-board` | board ✅ done |
| skewer, discovered attack/check | `remove-rule` | confound-free ⭐ next |
| fork / double attack | `remove-board` | board ⭐ next |
| passed pawn (promotion) | `remove-rule` (promotion mask) / `restructure` | confound-free / board ⭐ |
| open file / diagonal / line, battery | `obstruct` | near-clean ⭐ |
| remove-defender / overload / deflection / interference | `remove-board` | board |
| trapped piece | `remove-board` (open an escape) | board |
| zugzwang | `tempo` (let stuck side pass) | confound-free |
| initiative / momentum | `tempo` (give opponent a ply) | confound-free |
| prophylaxis ("prevents mate on h7", stops a threat) | `threat` | confound-free (independent solver) |
| pawn structure: isolated / doubled / backward / islands | `restructure` | board |
| piece activity | `ablate-piece` / passive `remove-board` | board (large shock) |
| outpost, 7th-rank rook, key square/diagonal | `remove-board` / `obstruct` | board |
| space, king safety, centre, development, coordination | `detect`-tier (degree scalar) | measurement only |
| opening "it's theory" | `book` + `detect`-tier | detection (see §4) |
| sequencing-only means (zwischenzug, opposition, triangulation) | `tempo` / not a factor | special |
| practical (clock, traps, opponent model) | — | out of scope (oracle has no clock/opponent) |

## 3. Confound discipline (unchanged, applied per operator)

Every *board edit* (`remove-board`, `restructure`, `ablate-piece`, partly `obstruct`) perturbs more
than its one target property → run the matched-placebo contrast + edit-shock gate before trusting
`eff`. The **confound-free** operators (`remove-rule`, `tempo`, `threat`) change no board material,
so they are the gold surfaces — and `tempo`/`threat` extend the confound-free story *beyond pins*
with **no engine patch at all**.

## 4. Two-band grading policy

**Band A — theoretical positions (openings, tablebase endgames): grade by DETECTION.**
The intervention manufactures ground truth; theory already has it, so detection against the *known
reason* suffices. The argument:

- **Endgames are airtight.** A ≤7-man tablebase returns the exact WDL/DTZ for every move, so
  load-bearing is computable exactly (no oracle, no proxy). Theoretical endgames are *named for their
  single load-bearing idea* (Lucena = bridge, Philidor = third-rank defence, opposition, Réti path),
  so the position's place in theory *certifies* the reason. This is the **verified, non-circular
  label** the project wants (README Next-steps #3) — a tablebase is the independent gold the Lichess
  `pin`-tag proxy is not.
- **It's strongest where the oracle is weakest.** `eff_wp` saturates in decided positions (golden
  case #3 emitted a false "red herring"). Theoretical positions *are* the saturated band, so
  detection-against-truth and intervention are **complementary instruments**, not a lowered bar.
- **No red herrings by construction.** Red herrings are a property of arbitrary positions with many
  coexisting factors; theoretical positions are curated down to their one deciding idea.
- **Openings are softer than endgames.** Opening theory is human consensus, not a solved game, and a
  book move is often good for several reasons at once. So detection-against-opening-theory cleanly
  catches **fabrication** (a factor not in the position) and **canonical-reason match**, but cannot
  fairly call a *real-but-secondary* reason unfaithful. Endgames: unconditional. Openings:
  fabrication/canonical sense only.

> Reconciliation with "existence isn't enough": here **detection = match against the *known* reason
> (tablebase-derived or codified theory), a gold label** — not mere board-occupancy of any factor.
> That is exactly why it holds for this class and not for arbitrary positions.

**Band B — middlegame (unsaturated, eval ≈ +1..+3): grade by INTERVENTION** (the operator algebra
above, with the confound discipline of §3).

Band selection: tablebase-reachable → Band A (verified); recognized opening/theory position →
Band A (consensus); else Band B.

## 5. Theoretical positions also CALIBRATE the operators

Band A is the test bed for the operators themselves: check that `obstruct`, `tempo`, `restructure`
return the **same verdict as the exact tablebase counterfactual**. This turns "do I trust this
operator?" from a judgment call into a measured agreement — `validate_factor_type` / the edit-shock
gate, applied at the operator level.

## 6. Open design subtleties

- **Exploit vs create.** Exploit a pre-existing factor → remove it. Create a factor → compare against
  the best move that does *not* create it. Today's `remove()` only handles exploit cleanly.
- **Positive vs prophylactic.** Positive = use your factor (`remove`/`obstruct`); prophylactic = stop
  the opponent's (`threat`). Distinct harnesses.
- **Material-neutrality.** `restructure`, `obstruct`, and `tempo` are the escapes from the
  add/remove-a-pawn wall that blocked structure and open-file factors.
- **Multi-factor moves.** A move good for several reasons needs joint ablation / Shapley over the
  cited set — deferred past single-factor.
- **Detection-tier honesty.** Where used outside Band A (diffuse strategic factors), report it as
  *anti-fabrication detection*, never as causal load-bearing — the analogue of the saliency baseline.

## 7. Priority

1. ✅ `pin` (rule), ✅ `hanging` (board)
2. ⭐ `skewer` (rule), `fork` (board) — closest wins, prove the surfaces generalize
3. ⭐ `tempo` operator — smallest, confound-free, no patch; unlocks zugzwang + initiative
4. ⭐ `obstruct` operator — one operator for all line factors (open file/diagonal/battery/discovery)
5. `passed_pawn` (rule via promotion mask) + `restructure` for pawn structure
6. `threat` operator + independent mate-solver → verified prophylaxis labels (Next-steps #3)
7. Band-A tablebase harness (verified endgame labels + operator calibration)
8. Deferred: space / king-safety (detection-tier), multi-factor joint ablation, mechanistic surface
