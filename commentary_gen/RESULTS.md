# Results — 2026-09-14

Target: `gemini-3.5-flash-lite`, default thinking. Judge: Sonnet subagents, rubric in `JUDGE.md`
(1–5 on faithful / relevant / human / overall). Dev = 300 held-out comments, stratified by source.
Screening on a fixed 40-item dev minibatch; the winner confirmed on full dev.

## Winner: `candidates/best.json` (= r2_ttw_plain_voice_hedged_D0)

| | overall | faithful | relevant | human | chrF |
|---|---|---|---|---|---|
| baseline (one-line instruction, no demos) | 2.70 | 3.02 | 3.13 | 2.78 | 0.256 |
| winner | **3.06** | 3.17 | 3.48 | 3.83 | 0.236 |

Paired on 256 dev items: 104 wins, 89 ties, 63 losses, mean +0.30. Gain is uniform across the
four sources (+0.22 to +0.36). Overall distribution of the winner: 1:10 2:74 3:77 4:83 5:13.

The winning instruction: silent board check first, then 1–4 plain sentences in an annotator's
voice, no markdown or hype adjectives, prefer plans and ideas over concrete tactics, at most one
verified concrete threat, no "who stands better". Three labelled demos (one GameKnot, one lichess
study, one PathToChessMastery, 150–350 chars).

## What moved the needle, what did not

- **Voice is fully steerable.** Plain-voice rules take human from 2.8 to ~4.0 in one step.
- **Faithfulness is capped by board reading, not by prompting.** Every variant sits at 2.3–3.2
  faithful. Prompts that ask for more checkable claims (checklists, "name the piece and square",
  contrastive alternatives) get *worse*: the model asserts more and misreads more. The only lever
  that helped was hedging, i.e. saying fewer checkable things.
- **Over-hedging collapses relevance.** Banning all threat language (r3_zero_threat) scored 1.62,
  the worst of 21 candidates: content-free prose.
- **Demos matter as much as the instruction.** Same instruction, demo set D0 3.02 vs D2 2.33;
  6 demos (D0+D1) worse than 3. Not fully separable from judge noise (see below).
- **Round 3 converged.** All four children of the winner scored below it on the minibatch.

## Caveats

- Judge noise is large: two judges on halves of the same run differ by up to 0.45 in mean overall.
  Minibatch rankings below ±0.15 are not reliable; the full-dev paired result is.
- chrF is uninformative here (all runs 0.23–0.32, and it prefers long output).
- 293 lichess-study examples with a set-up FEN were removed from the dataset mid-run; the dev set
  was rebuilt and 163 baseline scores carried over by (fen, comment) key.
- Residual faithfulness failures (wrong-piece 18, wrong-square 17, hallucinated-threat 19 of 257)
  would need a different input, e.g. a computed list of pieces per square or attacked pieces,
  not a different prompt. That is a task change and was not tried.

## Cost

~5,000 Gemini calls, about $1. 60 Sonnet judge/proposer subagents, ~7M tokens.

# Round 4 — 2026-09-15: lessons as context, thinking visible, open-ended diagnosis

Run `r4_best_lessons_think`: best.json unchanged, plus `chess_school_intro.md` (three ICS lessons,
~11k tokens) prepended verbatim, and Gemini thinking switched on (`thinkingLevel: medium`; the
previous run had *no* thinking, Flash-Lite's default). Judges (12 Sonnet, `JUDGE2.md`) saw the
thought summary and wrote free-text `mistakes` and `tools`; categories below were built from
reading all 300 notes, then made reproducible in `categorize_r4.py`.

| | overall | faithful | relevant | human | thinking_faithful |
|---|---|---|---|---|---|
| r2 winner (no lessons, no thinking) | 3.06 | 3.17 | 3.47 | 3.83 | — |
| r4 | 3.04 | **3.93** | 3.20 | 3.41 | 3.77 |

Paired on 257 items: overall −0.02 (83 W / 85 T / 89 L); faithful +0.77; relevant −0.28;
human −0.42. Per-batch overall ranges 2.36–3.32, i.e. judge noise as before. Two changes are
conflated (lessons + thinking); no control run.

Top-level split of the 300: 96 outputs contain a false claim, 101 are true but generic (miss
the point), 103 accepted (overall ≥ 4). Faithfulness rose because the model asserts less; the
human score fell because the lessons' vocabulary leaks ("qualitative value of my pieces", "to do
list") and voice drifts toward the textbook.

## Mistake categories (multi-label, n of 224 items with a note)

| n | category |
|---|---|
| 128 | true but generic: misses the hanging piece / fork / mate / tempo attack / plan that the reference is about |
| 37 | phantom or vanished piece: wrong occupant, wrong origin square, captured piece misidentified (Nxe7 "takes a rook", knight "trapped on a8" two moves after it was traded, "my e2 bishop" that is Black's) |
| 35 | geometry: piece said to attack/defend a square it cannot reach or that is blocked (Nf3 "defends e4", Rh2 "hits h5" through own h4 pawn, Bh6 "attacks g8") |
| 32 | evaluation inverted: praises a move the reference calls a mistake, or wrong side has the edge |
| 31 | material / count wrong: "rooks" with one rook, bishop pair with one bishop, down a rook unnoticed, "development complete" with pieces at home |
| 30 | square called undefended/weak/vacant that is covered or occupied |
| 28 | side / direction backwards: narrates from the wrong side, weakness assigned to the wrong side, attacks the wing where the king is not |
| 28 | thinking–output gap (see below) |
| 22 | light/dark bishop swapped (Stonewall and French "bad bishop" backwards in both texts) |
| 20 | pawn-structure mislabel: isolated vs doubled, backward, "passed" |
| 18 | "open file" with a pawn still on it |
| 15 | "forced" / "threat" where other legal replies exist or the capture is illegal (Bxc4 through own e2 pawn; Qxf7 with Nf6 interposed) |
| 15 | rooks "connected" / back rank "cleared" with pieces still between |
| 14 | lesson vocabulary leaked into the output |

Thinking vs output: in 25 items the thinking contains an error the output dropped or blurred
(the model's hedging instruction is doing its job); in 12 the output asserts something the
thinking never checked (e.g. a knight "in the corner" when Black has no knight). The thought
summaries are short (median 1.4k thought tokens) and rarely show square-by-square verification;
when they do, the misreads are already there.

## Tools the judges asked for (multi-label)

| n | tool |
|---|---|
| 98 | per-piece attack/defence map with blockers: what this piece reaches; what attacks/defends this square |
| 82 | piece inventory / occupancy: what sits on square X, king locations, material count, capture log |
| 68 | short forcing-line search: checks, forks, mates, discovered attacks, is candidate move X safe |
| 28 | hanging-piece scan (attackers vs defenders per piece, pin-aware) |
| 25 | pawn-structure classifier (isolated / doubled / backward / passed, weak squares) |
| 21 | board predicates: back-rank connectivity, development checklist, castling squares |
| 19 | opening book / named plans for the structure |
| 18 | square-colour / bishop-identity lookup |
| 17 | per-file pawn census (open / half-open / closed) |
| 12 | legal-move list / check-reply enumerator |
| 7 | engine eval / move comparison |
| 94 | none (76 clean items + 18 where the defect is voice, side of narration or judgement) |

Reading: the first two rows together cover nearly every factual error and are cheap
(python-chess, no search). The third row is what separates "true but generic" from the reference
and needs a shallow tactics search. Engine evaluation is almost never what the judges wanted.

## Cost

300 Gemini calls × ~12k prompt tokens ≈ 3.7M input tokens (< $1). 12 Sonnet judges ≈ 2.2M tokens.
