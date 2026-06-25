# xai-chess

A research benchmark that grades chess move-explanations. Given an explanation
of the form *"move m is good because of factor X"*, xai-chess asks one question:
**is factor X actually causally responsible for m being good, or is it a red
herring the explainer name-dropped?**

The method is intervention, not correlation:

1. Detect the cited factor on the board (currently: an absolute **pin**), using
   python-chess so the detection is faithful by construction.
2. Neutralize that factor in a material-neutral way (for a pin: relocate the
   pinned-to king off the pin ray, leaving material unchanged).
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

Phase-1 walking skeleton for the `pin` factor, verified end-to-end. On the two
golden cases it separates a faithful citation from a red herring:

| Case                                          | `eff_wp` | Verdict          |
|-----------------------------------------------|----------|------------------|
| `d4` wins the pinned e5 knight (faithful)     | `+0.335` | load-bearing     |
| `Rxa8` wins a rook, pin irrelevant (unfaithful) | `-0.005` | red herring      |

Both classify correctly at threshold `0.10`. Output is deterministic
(single-thread, fixed nodes).

## Next three steps

1. **Harden the pin verifier.** Add more golden pins (defended pin, relative
   pin, a position with no valid material-neutral neutralization), confirm the
   quarantine paths actually fire, and stress the king-safety confound on a
   quiet/positional pin to find where the board-level operator breaks.
2. **Add a second factor type.** Widen the DSL past `pin` with one more factor
   (e.g. `hanging` or `passed_pawn`) to show the apparatus generalizes.
3. **Build toward the real meta-eval.** Assemble a graded, non-saturated
   position set with matched placebo cases and report stratified AUROC; run a
   power calculation before trusting the `0.10` threshold.

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

**2. The oracle engine is vendored** (not a pip package) at
`vendor/stockfish/stockfish-ubuntu-x86-64-avx2` (Stockfish 18, AVX2 build).

**3. Run the verifier on the golden set:**

```bash
PYTHONPATH=./.pylibs python3 pin_verify.py golden.jsonl \
  --engine vendor/stockfish/stockfish-ubuntu-x86-64-avx2 \
  --nodes 400000
```

Flags: `--nodes` fixes the search size for determinism; `--threshold`
(default `0.10`) is the `eff_wp` cutoff for calling a factor load-bearing.

### Input format

`golden.jsonl` is one JSON record per line:

```json
{"fen": "4k3/8/5p2/4n3/8/8/3P4/4R1K1 w - - 0 1",
 "justified_move_uci": "d2d4",
 "cited_factors": [{"type": "pin", "args": {"pinned": "e5"}}],
 "source": "faithful: d4 wins the e5 knight because the pin stops it fleeing"}
```
