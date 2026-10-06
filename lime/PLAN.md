# LIME on Leela's positional evaluation

For one position: perturb it with equal numbers of white and black pawns, ask Leela for Q on every perturbation,
fit a small NN from our binary features to Q, read each feature family's relevance off that NN.
Output is one JSON per position, which is all the web demo needs.

One script, `lime/lime_001.py`. No new dependencies: `chess`, `numpy`, `onnxruntime`, `scikit-learn`
are in `.pylibs`; `lc0-probe/lc0_planes.py` encodes; `features/` extracts.

    python lime/lime_001.py "<fen>" [--n 3000] [--out lime/out]

## What "symmetric" means

Symmetric in material, not in geometry: each perturbation adds `k` white and `k` black pawns,
every one on its own random square. The balance of the base position is kept. Mirrored squares would put both new pawns on the same file every time, so the
sample could never contain a new passed pawn, half-open file, or one-sided space or island
change, and those families would come out as "not varied".

## Steps

1. **Perturb.** Draw `k` from a geometric distribution (mean ~3). Place `k` white and `k`
   black pawns on distinct random empty squares of ranks 2-7. More than 8 pawns a side is
   allowed. Reject if the board is invalid for any reason other than pawn or piece count
   (`board.status()` minus the `TOO_MANY_*` flags) and dedupe by FEN. Sample 0 is the
   unperturbed position.
   - Leela takes 9- and 10-pawn boards without complaint (tried: Q moves 0.01 -> 0.03 -> 0.22
     on one test position). It was never trained on them, so each result records the share of
     samples over 8 pawns, and we compare fidelity with and without them on the first run.
   - `features/` rejects them today ("too much material"). `position()` in
     `features/src/lib.rs` needs shakmaty's `ignore_too_much_material()`; `cargo test` must
     still pass.
   - Review probe (8 book positions): median |dQ| per sample is 0.45-0.78, 0.26-0.32 even at
     k=1, so the geometric draw gives no locality on its own. 56-83% of samples have a new pawn
     one push from promotion, a new pawn hitting a piece, or a check, and those carry 75-94%
     of the summed dQ²: the dropped pawns are tactics, not position. So a sample is rejected
     when any new pawn stands one push from promotion, attacks an enemy piece, or gives check
     (python-chess `attacks()`, a few lines). With that filter the probe's MLP R² was 0.27-0.64,
     so the fidelity floor stays open.
2. **Leela.** `encode_fens` -> `vendor/lc0-nets/t1-256x10.onnx` -> `/output/wdl`, target
   `Q = W - L` (raw net, no search, side to move's view). ~180 pos/s on CPU, so 3000 samples
   is about 17 s.
3. **Features.** Pipe the FENs through `cargo run --release --bin features -- --all`, turn
   white/black into us/them. Copy `sae/value_sae.py:43-56`; that file is a script and cannot be
   imported. Rows the CLI returns an error for (shakmaty rejects a double check by two pawns,
   ~1% of samples) are dropped, not asserted on.
   - No PV is passed, so the initiative family is dropped. Add it when a per-sample lc0 search
     is affordable.
   - Drop columns that never change across the sample.
4. **Surrogate.** `MLPRegressor(hidden_layer_sizes=(64, 64, 64, 64), early_stopping=True)` on
   an 80/20 split: four hidden layers plus the output layer. Report held-out R² as `fidelity`.
   Fit a `Ridge` on the same split and report its R² next to it, so we can see what the depth buys.
   Fit 5 seeds; relevance is the mean, with the spread and a flag for families whose sign
   disagrees across seeds (probe: `centre.control.them` went -0.09 to +0.11 across 4 seeds).
   Probe fidelity as planned was MLP 0.15-0.44, Ridge 0.22-0.41; the fidelity floor is open.
5. **Relevance, per family.** Families are the `families()` ids from `lib.rs`, side-split into
   `.us`/`.them` where the buckets carry a side. A relative family (`equal` unsided, `plusN`
   sided) is one family with one state, not three. One-hot buckets cannot be flipped one bit at
   a time without leaving the data, so ablate the whole family with values the sample took,
   over the samples J where the family's state differs from the base:

       rel[fam] = f(y0) - mean_{j in J} f(y0 with fam's columns taken from sample j)

   Report `|J|/n` next to it. Positive means the family's current state helps the side to move.
   Families with empty J get `varied: false`, not a relevance of 0: pawns cannot change the
   bishop pair, so LIME says nothing about it.
6. **Check.** `python lime/lime_001.py --selftest`: every perturbation has equal pawn counts added
   for both sides, pawns only on ranks 2-7, original pieces untouched, CLI-error rows dropped;
   every ablated row has exactly one bucket on per one-hot family; and on a synthetic target
   `Q = 0.3 * (one bucket of a relative family)`, black to move, the top family is that one
   with the right sign.

## Output, the demo contract

`lime/out/<fen-slug>.json`:

    { "fen": ..., "q": 0.12, "n": 3000, "over8": 0.3,
      "fidelity": { "mlp": ..., "ridge": ... },
      "families": [ { "id": "pawns.islands", "side": "us", "state": ["pawns.islands.3"],
                      "relevance": -0.04, "spread": 0.01, "sign_stable": true,
                      "varied": 0.04, "squares": ["a2", ...], "why": "..." }, ... ] }

`state` is the list of on-buckets (empty for an off boolean, several for multi-sector families);
`squares`, `why` come straight from the extractor so the demo can highlight without recomputing.
Sorted by `|relevance|`.

## Positions

The demo's positions are the book cases in `features/examples.json`: 92 distinct positions, each
already carrying game, move, quote and source. `python lime/lime_001.py --examples` runs them all
and writes `lime/out/index.json`: one row per FEN with game, move, path to the result, and its
cases (quote, source, family, tp/tn/fp/fn). 7 FENs carry more than one quote, and 37 of the 99
cases are fp/fn, where the book disagrees with the extractor; the demo must show those as such.

All 92 are usable now that the 8-pawn cap is gone; 14 of them have a full set of pawns on one
side, so every perturbation of those goes over 8. The demo lists only positions above a
fidelity floor we pick after the first run.

## Web demo, later

`docs/lime.html`: pick a book position from `index.json`, board on the left with the book's
quote under it, families as a signed bar list on the right, hover a bar to highlight its squares.
Titles and rules come from the existing features wasm `families()`.

Skipped: live LIME in the browser (onnxruntime-web + a ~3000-position batch + in-browser training).
Add it if "try your own position" turns out to matter; the features wasm already exists and the
MLP forward pass is ten lines of JS.

## Not doing

- Search-based Q, other nets, other perturbation types (piece moves, removals).
- A global model across positions. This is one model per position, as in the journey.
- Caching, config files, a package layout. One script, arguments on the command line.

## GSTACK REVIEW REPORT

Target: `lime/PLAN.md` (plan-eng-review, independent run, 2026-10-06, commit ff8d834). Findings only; the plan body above is unchanged. "Ran" = measured with a probe (scripts in the reviewer's scratchpad; 8 book positions, 1000 to 3000 perturbations each, sampling exactly as Step 1); "read" = inferred from code.

**Findings**

- [P1] (9/10, ran) L15-16 "The balance of the base position is kept, so a change in Q is positional." Refuted. 56-83% of perturbations contain a new pawn on its 7th rank, a new pawn hitting an enemy piece, or a check on the side to move, and those samples carry 75-94% of the summed dQ². Median |dQ| is 0.45-0.78 overall, 0.25-0.38 without them. At k=1 the median |dQ| is already 0.26-0.32, so "small perturbations dominate" (L33-35) does not give locality either. Consequence seen in the fit: `pawns.passed.us` comes out as the top family (-0.10 to -0.14) on 4r1k1/pb3p1p/1pN5/4P3/7p/2R5/PP1r1PP1/1R4K1 w because samples gaining a white passer average Q +0.10 against -0.36; most of those passers are pawns dropped on the 7th.
- [P1] (9/10, ran) L49 "A family is the id minus its last segment, plus .us/.them". Does not match the ids the CLI emits. `pieces.out_of_play.us` ... `pieces.rook_seventh.us` (7 families in `families()`, lib.rs:120-130) all collapse into `pieces.us`; `material.bishop_pair` and `material.opposite_bishops` into `material`. Relative families are split three ways: `centre.control.equal` has side None and `plus*` buckets are sided, likewise `material.balance`, `local.*`, `pieces.development`, `space_pawns*`. Swapping only `centre.control.us` from sample j left two buckets on in 626 of 1989 rows (31%): the off-manifold input L50-51 says the method avoids.
- [P2] (8/10, ran) L52 relevance averages over all j, including rows where the family did not change. The share that changed ranged from 0.00 (`local.centre.us`) through 0.04 (`pawns.islands.us`) to 0.32 (`pawns.passed.us`), so the ranking mostly reflects how often a dropped pawn changes the family, not how much Leela cares.
- [P2] (8/10, ran) L45-47 surrogate. Held-out R², MLP(64x4) / Ridge: 0.41/0.41, 0.44/0.37, 0.15/0.22 (≤8-pawn samples only, since the CLI still rejects 9+). Over 4 seeds on one position the top 5 families change, and `centre.control.them` flips sign (-0.091 to +0.110). The plan has no stability check. The example JSON's `"mlp": 0.81, "ridge": 0.64` (L67) looks like a result but is not one.
- [P2] (8/10, ran) L24-25 and L31. python-chess `status()` passes a double check by two pawns; shakmaty rejects it as "impossible check" (37 of 3001 samples on 1k2r3/1ppr2pp/p1p2p2/5R2/4P3/1P1P3P/P1P3P1/5RK1 w). The copied `assert "error" not in r` (sae/value_sae.py:47) would abort the run. `ignore_too_much_material()` alone does not cover it; shakmaty 0.30.1 also has `ignore_impossible_check()`.
- [P2] (7/10, ran) L68-73 `state` is one string, but many families have zero or several buckets on: boolean `pieces.*` when off, `pawns.passed.1` together with `.protected`, multi-sector `pawns.majority`, `space`, multiple `material.imbalance` trades, both `pieces.coordination` bits. When the family is off, `squares` is empty (lib.rs:60), so hovering highlights nothing.
- [P3] (8/10, ran) L77-79. Verified: 92 distinct FENs, 14 with eight pawns on one side. 99 cases have a FEN, so 7 FENs carry more than one quote. 37 of the 99 are `fp`/`fn`, where the book contradicts the extractor, and a demo would show the extractor's `why` next to a quote that disagrees with it.
- [P2] (8/10, read) L57-60 the selftest would not catch any of the problems above. A target of `0.3 * feature` on independent bits passes even with the grouping bug, cannot see the tactical confound, and does not exercise us/them on a black-to-move position.

**Verified / unverified claims in the plan**: verified: 92 / 14 counts; 211 columns; ~175-180 samples/s for Leela plus sampling; Ridge and MLP already in `.pylibs` (sklearn 1.9.0). Not reproduced: "Q moves 0.01 -> 0.03 -> 0.22" (no position given). Wrong: the "positional" claim, "small perturbations dominate", and "ten lines" (it is value_sae.py:43-56, 14 lines). Untested: everything about samples with more than 8 pawns, since the features CLI still rejects them; that covers all samples of the 14 full-pawn positions.

**Test coverage (proposed `--selftest`, nothing exists yet)**
```
perturb()  [GAP] equal k per side   [GAP] ranks 2-7   [GAP] originals untouched   [GAP] CLI-error rows dropped
families   [GAP] grouping = families() ids; ablated rows stay one-hot per family
relevance  [GAP] synthetic target on a relative-family bucket, black to move: right family AND sign
surrogate  [GAP] seed spread reported; sign-unstable families flagged
features   [★★ cargo test, tests/examples.rs] must still pass after ignore_too_much_material
```
**Performance**: no issues. About 1 min per position, 92 positions; no caching needed.
**Outside voice**: unavailable. Codex preflight returned model_unusable (gpt-6-astra), and the native fallback needs TaskOutput, which this host does not provide.
**NOT in scope**: what the plan already lists. No TODOS.md in the repo; no TODOs proposed.

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside Review | codex (`/plan-eng-review` outside voice) | Independent 2nd opinion | 1 | unavailable | model_unusable; native fallback unavailable |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | issues_open | 8 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **OUTSIDE COVERAGE:** codex, plan-review phase, unavailable (model_unusable); no native fallback ran.
- **VERDICT:** NOT CLEARED, issues open; eng review required.

**UNRESOLVED DECISIONS:**
- D1 tactical confound (P1). RESOLVED: A. A) Keep "ranks 2-7" as the square pool but reject a sample when a new pawn stands one push from promotion, hits an enemy piece, or gives check, a few lines with python-chess `attacks()` (recommended: removes the 75-94% of dQ² that no feature describes; on the remaining samples MLP R² was still only 0.27-0.64, so the fidelity floor stays open). B) Keep those samples and add tactic indicator columns as unreported surrogate inputs. C) As planned. This changes the user's literal "ranks 2-7" sampling, so it is the user's call.
- D2 family key (P1). A) Use the `families()` ids (lib.rs:99-155) as keys; a relative family (equal / plusN.us / plusN.them) is one family with one state, and per-side families stay split into us/them (recommended: the only option that keeps ablated rows on the data manifold). B) The L49 rule.
- D3 relevance average (P2). A) Average only over the j where the family's state differs from the base, and report that share next to the relevance (recommended: separates "how often it varied" from "how much it matters"). B) As planned.
- D4 surrogate stability (P2). A) Fit 5 seeds of the same 4-layer MLP, report mean relevance and spread, and mark families whose sign disagrees across seeds (recommended: one sign flipped across seeds in the probe; this keeps the NN requirement and costs a few seconds). B) Single fit.
- D5 CLI errors (P2). A) Treat CLI error rows as rejected samples in Python (recommended: no Rust change beyond `ignore_too_much_material`). B) Also call `ignore_impossible_check()` in `position()` and feed the double-pawn-check boards to Leela.
- D6 output `state` (P2). A) `state` becomes the list of on-buckets, possibly empty (recommended: matches what the extractor emits). B) Single string as planned.
- D7 demo positions (P3). A) One `index.json` row per FEN holding the list of its cases with `case` (tp/tn/fp/fn), so the demo can mark the 37 cases where the book disagrees with the extractor (recommended). B) As planned.
