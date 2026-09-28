# Judge rubric (round 6: fact sheet + three lookups)

You grade machine-written chess commentary against the human annotator's comment on the same
move. Each item has the moves so far, the board after the last move, the FACTS sheet the model
was given (numbered F1, F2, ...; computed by python-chess, exact), the TOOL CALLS it made
(numbered [1], [2], ...; exact), the model's written EVALUATION (its STEP 1/2, not shown to the
user; absent if it skipped them), the human REFERENCE, the MODEL THINKING (an API summary; the
user never sees it) and the GENERATED text.

The model's tools are the facts sheet and the three lookups `after(moves)`, `legal(move)`,
`attack_map(square)`. Score each item 1–5 on five axes. Verify claims against the board, the move
list, the facts and the call results yourself; do not trust the reference, the thinking or the
generation.

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
- **tool_use** — how well the model used the facts and the lookups to reach its point.
  5 = the note's point rests on the fact(s) that decide the position, read correctly, and any
  line or square the note names beyond the sheet was checked with a lookup; no wasted calls.
  4 = sound use with one small gap (a named line unchecked but true, or a redundant call).
  3 = the facts were read but the deciding one was passed over for a lesser one, or a lookup
  that mattered was not made.
  2 = a fact or call result is misread or stretched in the note, or the note ignores a fact that
  decides the game (mate, hanging piece, lost material).
  1 = the note contradicts a fact or call result, or was written without regard to either.
  Score the use, not the outcome: a true note that got lucky without checking is not a 5.

Then diagnose, in your own words.

- **thinking_faithful** — 1–5: are the claims inside the MODEL THINKING itself true?
- **mistakes** — free text, one to three sentences: what the model misread, miscalculated,
  assumed, skipped, or dropped between thinking and output. Cite squares and pieces. Empty
  string if nothing went wrong.
- **facts_used** — list of fact ids (e.g. "F3") the output demonstrably relies on. Empty list if
  none.
- **calls_used** — list of call numbers whose result the output demonstrably relies on. Empty
  list if none.
- **calls_contradicted** — free text: a claim in the output that a fact or call result already
  refutes; name the fact id or call number. Empty string if none.
- **calls_missing** — free text: a lookup the model should have made (tool and argument), or a
  fact it should have used, that would have prevented the mistake or found the reference's point;
  or "none".
- **tool_use_notes** — one or two sentences justifying the tool_use score: which facts or calls
  mattered, whether they were chosen, read and carried into the note correctly.

Output: write `<batch>.scores.json` next to the batch file — a JSON array, one object per item,
in the batch's order:

    {"id": "dev-0003", "faithful": 3, "relevant": 4, "human": 4, "overall": 3, "tool_use": 2,
     "thinking_faithful": 2,
     "mistakes": "The output says the e-file is open; F21 marks it half-open for White.",
     "facts_used": ["F1", "F21"], "calls_used": [1],
     "calls_contradicted": "e-file called open; F21 says half-open.",
     "calls_missing": "after('Rxe5') to test the capture the reference is about",
     "tool_use_notes": "Picked F21 over F4 (the hanging knight) and misread it; the one call checked an irrelevant square."}

Nothing else in the file. Do not skip items.
