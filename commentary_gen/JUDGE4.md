# Judge rubric (round 5: tool calls)

You grade machine-written chess commentary against the human annotator's comment on the same
move. Each item has the moves so far, the board after the last move, the TOOL CALLS the model
made (numbered; the results are computed by python-chess and exact), the human REFERENCE, the
MODEL THINKING (an API summary; the user never sees it) and the GENERATED text.

Score each item 1–5 on four axes. Verify claims against the board, the move list and the tool
results yourself; do not trust the reference, the thinking or the generation.

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
- **mistakes** — free text, one to three sentences: what the model misread, miscalculated,
  assumed, skipped, or dropped between thinking and output. Cite squares and pieces. Empty
  string if nothing went wrong.
- **calls_used** — list of call numbers whose result the output demonstrably relies on (a
  square, move, count or verdict that appears in the output and came from that result). Empty
  list if the output could have been written without any call.
- **calls_contradicted** — free text: a claim in the output that one of the tool results already
  refutes; name the call number. Empty string if none.
- **calls_missing** — free text: a call the model should have made (name the tool and the
  argument) that would have prevented the mistake, or "none".

Output: write `<batch>.scores.json` next to the batch file — a JSON array, one object per item,
in the batch's order:

    {"id": "train-0003", "faithful": 3, "relevant": 4, "human": 4, "overall": 3,
     "thinking_faithful": 2,
     "mistakes": "The output says the e-file is open; call [6] marks it half-open for White.",
     "calls_used": [1, 3],
     "calls_contradicted": "e-file called open; [6] says half-open.",
     "calls_missing": "none"}

Nothing else in the file. Do not skip items.
