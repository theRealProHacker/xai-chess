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

# Round 5 — 2026-09-16: recipe prompt + computed facts, optimised on train

Prompt rebuilt as one flow: lessons (positional-only trim, `chess_school_positional.md`), a
five-step recipe replacing the r2 instruction, and a `board_tools.report_after_move` block after
every board (demos included): the moved piece's reach and attackers, piece list and material,
hanging pieces with the winning capture, every check and capture with its net, pawn structure
and file status. ~8.6k prompt tokens. Thinking medium. Judge: `JUDGE3.md` (JUDGE2 plus
`facts_used` and `facts_contradicted`).

Optimisation on a 40-item **train** minibatch (seed 0), two rounds, three candidates each:

| round | candidate | overall | faithful | relevant | human | contradicts facts |
|---|---|---|---|---|---|---|
| 1 | recipe + lessons + facts | 3.10 | 3.85 | 3.18 | 3.05 | 13/40 |
| 1 | same, no lessons (control) | 3.28 | 4.40 | 3.43 | 3.33 | 5/40 |
| 1 | + voice step | 3.20 | 4.13 | 3.73 | 3.13 | 11/40 |
| 2 | + check-against-facts step | 3.10 | 4.08 | 3.43 | 2.50 | 11/40 |
| 2 | + check + specificity rule | 3.18 | 4.10 | 3.58 | 3.18 | 8/40 |
| 2 | **+ check + voice** (`best_r5.json`) | **3.38** | 4.20 | 3.43 | 3.65 | 7/40 |

Batch halves differ by up to 0.55 in mean overall, so the ranking is indicative only.

## Winner on full dev (300)

| | overall | faithful | relevant | human | thinking_faithful |
|---|---|---|---|---|---|
| r2 winner (no lessons, no thinking, no facts) | 3.06 | 3.17 | 3.47 | 3.83 | — |
| r4 (lessons + thinking) | 3.04 | 3.93 | 3.20 | 3.41 | 3.77 |
| **r5** (recipe + lessons + facts + thinking) | **3.37** | **4.02** | 3.66 | 3.80 | 3.99 |

Paired: vs r2 (257 items) overall +0.37 (111 W / 86 T / 60 L), faithful +0.93, human −0.01.
vs r4 (300) overall +0.38 (134 / 104 / 62), relevant +0.48, human +0.39. Gain is uniform across
sources (3.29–3.48). Overall distribution 1:1 2:61 3:95 4:112 5:31. Split: 97 false claim,
60 true but generic, 143 accepted (r4: 96 / 101 / 103). Lesson-word leakage 125 → 13 items.
Median thought tokens 1.4k → 0.8k; the recipe replaces most of the thinking.

## Does the model use the facts?

Yes, unevenly. Judges mark a fact section as used in 291/300 items (moved-piece line 248,
hanging 118, forcing 92, inventory 78, pawns 43). Lexically (`tool_usage.py`), the output names
the hanging square in 115 of the 127 positions where a piece hangs, and a moved-piece
attack/attacker square in 110 of 179; a listed forcing move appears in 16 outputs (78 thinkings)
of 256, a pawn-structure label in 23 of 233.

Per r4 wish (items whose r4 judge asked for that tool → r5 faithful on the same items):

| r4 asked for | n | r4 faithful | r5 faithful | uses it |
|---|---|---|---|---|
| piece inventory / material | 82 | 2.93 | 3.93 | via inventory line |
| attack/defence map | 98 | 3.21 | 3.94 | 46 name a square from it |
| file status | 17 | 3.12 | 4.41 | 6 |
| hanging scan | 28 | 3.64 | 4.04 | 19 |
| pawn structure | 25 | 3.72 | 4.20 | 9 |
| forcing search | 68 | 3.84 | 3.85 | rarely: 16 outputs |
| none (r4 clean) | 94 | 4.81 | 4.15 | — |

The cheap lookups (inventory, attack map, file status) fixed what they were asked to fix. The
forcing list did not: the model reads a capture's *existence* but not its *net*, and presents
net-negative captures as threats or options. That is the largest remaining contradiction class.

## What the model still gets wrong

`facts_contradicted` is non-empty in 81/300 (mean faithful 2.72 vs 4.51 elsewhere):
net-loss capture presented as a threat or good option 38, "undefended"/"hanging" with defenders
listed 26, material count 20, reach/geometry 19, open vs half-open file 10. The check step (4)
did not remove these; the model paraphrases a fact and loses the qualifier.

Mistake categories (n of 193 items with a note): true but generic 75, material / count wrong 43,
evaluation inverted 40, undefended/weak square that is covered 21, phantom piece 16, open file 15,
forced/threat claimed 12, geometry 11. Tools still wanted: forcing-line search with a verdict
(51), inventory 30, attack map 23; "none" 190.

Not done: the no-lessons control on dev (train minibatch says lessons cost faithfulness and
buy nothing measurable); a facts block that spells out the net verdict in words ("Bxa4 loses
a pawn") instead of `net -2`.

## Cost

~420 Gemini calls (train screens + dev). 25 Sonnet judge subagents, ~4M tokens; 12 dev judges
in parallel hit the session limit, 5 were rerun.
