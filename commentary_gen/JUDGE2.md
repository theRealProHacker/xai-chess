# Judge rubric (round 4: thinking visible, diagnosis required)

You grade machine-written chess commentary against the human annotator's comment on the same
move. Each item in a batch file has the moves so far, the board after the last move, the human
REFERENCE, the model's MODEL THINKING (a summary of its reasoning returned by the API; the user
never sees it) and the GENERATED text.

Score each item 1–5 on four axes. Verify claims against the board and move list yourself; do
not trust any of the three texts.

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

Then diagnose, in your own words. There is no fixed vocabulary; describe what you actually see.

- **thinking_faithful** — 1–5: are the claims inside the MODEL THINKING itself true? (5 = all true.)
- **mistakes** — free text, one to three sentences. Trace where the output went wrong: what the
  model misread, miscalculated, assumed, skipped or dropped between thinking and output, and
  where the thinking and the output diverge. Cite squares and pieces. Empty string if nothing
  went wrong.
- **tools** — free text, one to three sentences. Which concrete tool, lookup or extra input,
  had the model been able to call it at the point where it went wrong, would most plausibly
  have prevented the mistake? Be specific about what the tool would have to return. Write
  "none" if no tool would help because the defect is voice, relevance or judgement.

Output: write `<batch>.scores.json` next to the batch file — a JSON array, one object per item,
in the batch's order:

    {"id": "dev-0003", "faithful": 3, "relevant": 4, "human": 4, "overall": 3,
     "thinking_faithful": 2,
     "mistakes": "The thinking puts the black queen on d8, but it is on c7, so the asserted Nd5 fork of queen and rook does not exist; the output repeats the fork.",
     "tools": "A per-piece attack list for the knight on d5 (which enemy pieces it attacks) would have shown no queen among them."}

Nothing else in the file. Do not skip items.
