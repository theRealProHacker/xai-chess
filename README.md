# xai-chess

A research benchmark that grades chess move-explanations. Given an explanation
of the form *"move m is good because of factor X"*, xai-chess asks one question:
**is factor X actually causally responsible for m being good, or is it a red
herring the explainer name-dropped?**

The method is intervention, not correlation:

1. Detect the cited factor on the board (currently: an absolute **pin**), using
   python-chess so the detection is faithful by construction.
2. Neutralize that factor without changing material — two ways: physically
   (relocate the pinned-to king off the ray) and *logically* (suspend the pin in
   the engine's rules so the piece may move). See *Two intervention surfaces*.
3. Ask a strong oracle engine (Stockfish) how much the move's value changed.

```
eff = outcome(original, m) - outcome(neutralized, m)
  large positive  => removing the factor made m worse  => the factor was load-bearing (citation holds)
  ~ zero          => removing the factor changed nothing => the cited factor is a red herring
```

The primary outcome is `eff_wp`, the change in the engine's win-probability of
playing `m`. A cited factor that isn't real, or a position with no legal
material-neutral neutralization, is **quarantined** with a reason instead of
emitting a silent number.

**Reuse, not reinvention:** all chess mechanics come from python-chess and the
oracle is stock Stockfish. The only original code is the ~150-line verifier glue.

## Status

Phase-1 walking skeleton, now spanning **two factor types** (`pin` and `hanging`) on a
factor-typed DSL, verified end-to-end with **two independent intervention surfaces** (for
`pin`) and **three outcome scales**. Board-level `eff` on the three `pin` golden cases
(rule-level agrees on all three):

| Case                                                 | `eff_wp` | `eff_logit` | `eff_cp` | truth        |
|------------------------------------------------------|----------|-------------|----------|--------------|
| `d4` wins the pinned e5 knight (faithful)            | `+0.335` | `+1.62`     | `+441`   | load-bearing |
| `Rxa8` wins a rook, pin irrelevant (unfaithful)      | `-0.015` | `-0.15`     | `-43`    | red herring  |
| `d4` in an already-won position (saturated-faithful) | `+0.078` | `+0.73`     | `+197`   | load-bearing |

The first two separate cleanly at threshold `eff_wp ≥ 0.10`, and the two surfaces
**agree** on every case. The third case is the cautionary one: the pin genuinely
wins material, but White is already up a rook so **`eff_wp` saturates against the
[0,1] ceiling** (`+0.078`, *below* threshold → a false "red herring"). The
non-saturating scales recover it — `eff_cp ≈ +197` (≈ two pawns) and
`eff_logit ≈ +0.73`. Output is deterministic (single-thread, fixed nodes; two runs
byte-identical), though `eff` magnitude is mildly search-order dependent via the
shared transposition table.

### Second factor type: `hanging`

The verifier is a small **factor-typed DSL**, not a pin-only script: each factor type
supplies a human description, a material-neutral board-level counterfactual, and
*optionally* a rule-level surface. `hanging` is the second type — an enemy piece the move
wins because it is attacked and undefended. Its board-level neutralization relocates that
piece one square to safety (the parallel to relocating a pinned king). It is **board-level
only**: nothing "hanging" is a *rule* of chess, it is the absence of a defender, so there is
no rule-level surface to suspend.

| Case                                                  | `eff_wp` | `eff_logit` | `eff_cp` | truth        |
|-------------------------------------------------------|----------|-------------|----------|--------------|
| `Rxe5` wins the undefended e5 knight (faithful)       | `+0.407` | `+2.32`     | `+631`   | load-bearing |
| `Rxa8` wins a rook, e5 knight irrelevant (unfaithful) | `+0.000` | `+0.00`     | `+0`     | red herring  |
| e5 defended by the f6 pawn (factor not real)          | —        | —           | —        | quarantined  |

The faithful and red-herring cases separate cleanly, and the quarantine path fires when the
cited piece is actually defended. One caveat the board-only tier makes explicit: the board
edit can itself be a **confound**. An earlier red-herring draft (a forced-mate position) read
`eff_wp ≈ 0` correctly but `eff_cp ≈ +9000`, because relocating the knight changed the *mate
distance* — irrelevant to whether the piece was hanging. Relocating any piece moves more than
its one property; this is exactly the confound the rule-level surface removes for `pin`, and
which `hanging` (board-only) cannot.

### Three outcome scales

`eff = outcome(pin present) − outcome(pin removed)`, read three ways:

- **`eff_wp`** — Δ win-probability. Bounded and interpretable, the verdict scale —
  but it **saturates** near won/lost positions (the case-3 failure).
- **`eff_logit`** — Δ log-odds of the win-probability. De-saturates the tails while
  staying a monotone function of the same WP; the robust default.
- **`eff_cp`** — Δ centipawns (engine-native). Never saturates, but is *noisy* at
  the extremes (winning evals are unstable), so it is a reference, not the verdict.

### Two intervention surfaces

The pin is removed two different ways, and their agreement is the evidence:

- **board-level** — relocate the pinned-to king one square off the ray
  (material-neutral). Simple and engine-agnostic, but moving the king is a
  *king-safety confound*: `eff` conflates "removed the pin" with "nudged the king."
- **rule-level** ("logical") — leave the board untouched and *suspend the pin in
  the rules*: the pinned piece may move, and the pinning slider can neither check
  nor capture the king along that one ray. No piece moves, so there is **no
  confound**. This needs a movegen-patched Stockfish — a stock engine cannot free
  the piece without exposing the king to capture, which would *invert* the signal
  (the pin is load-bearing precisely because moving the piece loses the king).

When the two surfaces agree (as on all three golden cases), the board-level confound
is negligible — the gate for any "confound-free" claim.

## Meta-evaluation on a standard corpus

Beyond the golden cases, the apparatus is validated on the **Lichess puzzle database**
(CC0; natural human-game positions, theme-tagged). Records are labeled *independently of
the metric* by Lichess's tactic themes: **faithful** = `pin`-themed puzzles whose solution
exploits a pre-existing absolute pin; **red herring** = non-`pin` puzzles containing an
incidental pin the solution ignores. (`lichess_ingest.py` → JSONL; `corpus_eval.py` → AUROC
with bootstrap CIs.)

On a matched, non-saturated band — **N=164 (83 faithful / 81 red herring), 400k nodes**,
each scorer and both baselines run on the *same* records (`corpus_eval.py`, `baselines.py`):

| scorer | AUROC | 95% CI (bootstrap) |
|---|---|---|
| random | 0.57 | [0.49, 0.66] — CI includes 0.5 → chance |
| saliency (move targets pinned material) | 0.61 | [0.54, 0.68] |
| board-level `eff_wp` | 0.69 | [0.60, 0.77] |
| **rule-level** `eff_wp` | **0.82** | **[0.75, 0.88]** |

- The rule-level CI lies **entirely above the 0.70** reference bar (BonaFide's best metric).
- **Paired** bootstraps on the same positions — `Δ AUROC` (rule − X), all excluding 0:
  vs saliency **+0.21** [+0.12, +0.30] (P=1.00); vs board-level **+0.13** [+0.04, +0.23]
  (P=0.997). The confound-free rule-level intervention **significantly beats every baseline**.
  *Which intervention surface you use decides whether the metric works* — a contrast chess
  makes visible because the causal effect is computable.

**Honest scope.** (1) The Lichess `pin` tag is an *independent but noisy* proxy for "this pin
is load-bearing"; the rule≫baseline contrasts are robust to that noise (same labels for all),
but the absolute AUROC is against a proxy, not verified ground truth. (2) Only `random` and
`saliency` are ported — BonaFide's other 6 metrics are LLM-chain-of-thought-specific
(token perturbations / model internals / simulatability) with no faithful analogue in this
no-LLM, structured-factor + engine-oracle setting; the 0.70 line is therefore a cross-domain
*reference*, not a same-task baseline. (3) Single factor (`pin`), single oracle. (4) A
forced-mate *verified*-label probe (`verified_labels.py`) is **circular** for the rule-level
metric (label and metric share the counterfactual) — a cautionary artifact, not evidence.

## Next steps

1. **Add a second factor type.** ✅ Done — `hanging` (board-level only), shipped on the
   factor-typed DSL; see *Second factor type: `hanging`* above. Next: a factor that *also*
   has a rule-level surface (e.g. `passed_pawn` via a promotion mask) to show the rule-level
   intervention generalizes too, not just the DSL.
2. **Harden the verifier.** Add more golden cases (defended pin, relative pin, a position
   with no valid material-neutral neutralization; for `hanging`, a trapped piece with no safe
   relocation), confirm the quarantine paths actually fire, and stress the board-level
   confound to find where it breaks.
3. **Independent verified labels.** Replace the noisy Lichess proxy with a from-scratch
   variant mate-solver (not the patched engine) so the forced-mate label is verified *and*
   non-circular — the only route that keeps the "verifiable causal ground truth" claim.
4. **Difficulty-stratified AUROC + fuller meta-eval.** Use the Lichess `Rating` to report
   AUROC by difficulty band (the ICE/BonaFide stratification) with DeLong CIs; assemble a
   graded, non-saturated position set with matched placebo and a pre-registered power
   calculation before trusting the `0.10` threshold.

(Two strategic framing questions are parked until 2026-07-18 so they can be
decided with cross-model review rather than a single voice — they are not part
of the working next steps.)

## How to run

This machine has no `python3-venv` and no passwordless sudo, so dependencies go
into a local `./.pylibs` directory and scripts run with `PYTHONPATH` pointed at it.

**1. Install Python dependencies:**

```bash
pip install --target ./.pylibs -r requirements.txt
```

**2. The oracle engine is vendored** (not a pip package). Two builds live under
`vendor/stockfish/`:

- `stockfish-ubuntu-x86-64-avx2` — stock Stockfish 18 (AVX2). Drives the
  **board-level** surface only.
- `stockfish-xai-rulelevel` — the same source with a small movegen patch that
  adds the opt-in `MaskPinner` UCI option (see *The rule-level patch* below).
  Drives **both** surfaces. With `MaskPinner` unset it behaves as stock Stockfish.

**3. Run the verifier on the golden set** (use the patched build for both surfaces):

```bash
PYTHONPATH=./.pylibs python3 pin_verify.py golden.jsonl \
  --engine vendor/stockfish/stockfish-xai-rulelevel \
  --nodes 400000
```

The verifier auto-detects `MaskPinner`: point `--engine` at the stock binary and
it transparently falls back to the board-level surface only.

Flags: `--nodes` fixes the search size for determinism; `--threshold`
(default `0.10`) is the `eff_wp` cutoff for calling a factor load-bearing.

### The rule-level patch

`stockfish-xai-rulelevel` is built from the vendored Stockfish 18 source with a
~20-line, opt-in patch (`position.{h,cpp}`, `movegen.cpp`, `engine.cpp`). A new
UCI option `MaskPinner=<square>` *suspends the pin* cast by the slider on that
square: the pinned piece becomes legal to move, and that slider can neither check
nor capture the king along its ray. Default (`64` = `SQ_NONE`) is byte-for-byte
stock behavior, so the existing board-level path is untouched. To rebuild:

```bash
cd vendor/stockfish/src && make -j"$(nproc)" build ARCH=x86-64-avx2
cp stockfish ../stockfish-xai-rulelevel
```

### Input format

`golden.jsonl` is one JSON record per line:

```json
{"fen": "4k3/8/5p2/4n3/8/8/3P4/4R1K1 w - - 0 1",
 "justified_move_uci": "d2d4",
 "cited_factors": [{"type": "pin", "args": {"pinned": "e5"}}],
 "source": "faithful: d4 wins the e5 knight because the pin stops it fleeing"}
```
