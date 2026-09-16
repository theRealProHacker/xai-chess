# Judge rubric (round 5: computed facts in the prompt)

You grade machine-written chess commentary against the human annotator's comment on the same
move. Each item in a batch file has the moves so far, the board after the last move, the
COMPUTED FACTS the model was given (exact: what the moved piece reaches, piece list, hanging
pieces, checks and captures with net material, pawn structure), the human REFERENCE, the
model's MODEL THINKING (a summary returned by the API; the user never sees it) and the
GENERATED text.

Score each item 1–5 on four axes. Verify claims against the board, the move list and the facts
yourself; do not trust the reference, the thinking or the generation.

- **faithful** — every concrete claim about the position, threats, pieces and lines is true.
  5 = all true. 3 = one wrong or unverifiable claim. 1 = mostly wrong, wrong piece/square, or
  talks about a different move.
- **relevant** — it is about the move just played and what matters now, at the same level of
  abstraction the reference uses (a plan, a tactic, an evaluation). 1 = generic opening
  boilerplate or filler that would fit any position.
- **human** — reads like the corpus annotator: a person talking about the game, with a point
  of view and normal length. Penalise engine-voice, bold headers, bullet lists, textbook tone,
  and exaggerated adjectives ("thematic", "ambitious", "solidify", "signals intent").
- **overall** — would a strong club player accept this in place of the reference? Weigh
  faithfulness most.

Then diagnose, in your own words.

- **thinking_faithful** — 1–5: are the claims inside the MODEL THINKING itself true?
- **mistakes** — free text, one to three sentences. Where the output went wrong: what the model
  misread, miscalculated, assumed, skipped, or dropped between thinking and output. Cite squares
  and pieces. Empty string if nothing went wrong.
- **facts_used** — list of the fact sections the thinking or output demonstrably draws on, from:
  "moved" (the moved piece's reach/attackers line), "inventory", "hanging", "forcing", "pawns".
  Empty list if the text could have been written without the facts.
- **facts_contradicted** — free text: a claim in the output that the COMPUTED FACTS already
  refute (e.g. calls a defended piece hanging, names a capture the facts list as losing, calls a
  file open that the facts mark closed). Empty string if none.
- **tools** — free text, one to three sentences: which further lookup, had the model been able
  to call it where it went wrong, would have prevented the mistake, and what it would return.
  "none" if no tool would help because the defect is voice, relevance or judgement.

Output: write `<batch>.scores.json` next to the batch file — a JSON array, one object per item,
in the batch's order:

    {"id": "train-0003", "faithful": 3, "relevant": 4, "human": 4, "overall": 3,
     "thinking_faithful": 2,
     "mistakes": "The output says the e-file is open; the facts mark it half-open for White.",
     "facts_used": ["moved", "hanging"],
     "facts_contradicted": "e-file called open; facts: e closed.",
     "tools": "none"}

Nothing else in the file. Do not skip items.
