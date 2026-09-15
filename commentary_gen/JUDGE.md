# Judge rubric

You grade machine-written chess commentary against the human annotator's comment on the same
move. Each item in a batch file has the moves so far, the board after the last move, the human
REFERENCE and the GENERATED text.

Score each item 1–5 on four axes. Verify claims against the board and move list yourself; do
not trust either text.

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

Add a **fail** tag (one or two words) for the main defect when overall ≤ 3: e.g. `wrong-move`,
`hallucinated-threat`, `generic`, `engine-voice`, `too-long`, `formatting`, `misses-point`.

Output: write `<batch>.scores.json` next to the batch file — a JSON array, one object per item,
in the batch's order:

    {"id": "dev-0003", "faithful": 4, "relevant": 3, "human": 2, "overall": 3, "fail": "engine-voice"}

Nothing else in the file. Do not skip items.
