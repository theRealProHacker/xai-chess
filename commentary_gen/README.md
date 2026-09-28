# commentary_gen — DSPy-style prompt optimisation, run by hand

Task: `(moves so far, ASCII board) -> human commentary on the last move`. Target model: `LLM_MODEL`
(default `gpt-5-nano`; OpenAI or Gemini, keys in `../.env`). The optimiser is not DSPy: instruction proposals, demo
selection and the judge are Claude subagents driven from the session; this directory holds the
deterministic parts and every artifact they produce.

    build_dataset.py   corpus -> data/{pool,train,dev}.jsonl   (best 2000, stratified: 700/700/400/200)
    program.py         candidate json (instruction + demo ids) -> runs/<name>/<split>.jsonl
    metric.py          chrF + length ratio; packs runs into judge batches, unpacks judge scores
    llm.py             REST client (OpenAI / Gemini); sleeps through rate limits and billing outages
    board_tools.py     tactical lookups (attack map, inventory, hanging, forcing, legality, pawns)
    positional.py      positional lookups (what the move changed, alternatives, piece quality, king, space)
    facts.py           the two above as one numbered sheet, in reading order ("facts": true)
    verify.py          checks a written note against the board; no judge, no model
    tools.py           function-calling ToolBox ("functions": true, or a list of names)
    sandbox.py         code-execution alternative: run_python cells with a call logger ("code": true)
    tool_usage.py      tool use from the call log, joined with judge scores and the r4 wish list
    candidates/        every prompt tried, c0_baseline first; best.json = r2 winner,
                       r6_algorithm.json = current

Loop (MIPROv2 shape): baseline on dev -> judge -> bootstrap demos from train generations the
judge rates >=4 -> subagent proposes K instructions from the failures -> screen candidates on a
40-example minibatch (`screen.py --split train` since round 5) -> full dev for the winner ->
repeat with the winner as parent. Judges: JUDGE.md (r1-3), JUDGE2.md (r4), JUDGE4.md (r5, sees the
call log; calls_used / calls_contradicted / calls_missing), JUDGE6.md (r6 dev, also sees the fact
sheet and written evaluation; adds a 1-5 tool_use score with notes).

Round 6 splits the job differently: anything computable without a choice the model has to make is
computed and injected (`facts.py`, ~1.1k tokens, seven sections ordered most game-deciding first),
and only the three lookups that need a model-chosen move or square stay callable. The instruction is
a procedure — evaluate tactically then strategically citing fact ids, rank the findings by how much
they decide the game, check the winner with a lookup, write one point — and the model writes all
three steps out, with `"split_on": "NOTE:"` keeping only the note as the comment and the rest in
`eval`. `verify.py` then checks the note against python-chess: on its high-precision classes the
human reference comments are 21/300 and r5_tools 15/300, so board facts are no longer the gap.

`data/` and `runs/` are git-ignored: they contain corpus text (see ../commentary/LICENSING.md).
