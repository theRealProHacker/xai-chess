# commentary_gen — DSPy-style prompt optimisation, run by hand

Task: `(moves so far, ASCII board) -> human commentary on the last move`. Target model: `LLM_MODEL`
(default `gpt-5-nano`; OpenAI or Gemini, keys in `../.env`). The optimiser is not DSPy: instruction proposals, demo
selection and the judge are Claude subagents driven from the session; this directory holds the
deterministic parts and every artifact they produce.

    build_dataset.py   corpus -> data/{pool,train,dev}.jsonl   (best 2000, stratified: 700/700/400/200)
    program.py         candidate json (instruction + demo ids) -> runs/<name>/<split>.jsonl
    metric.py          chrF + length ratio; packs runs into judge batches, unpacks judge scores
    llm.py             REST client (OpenAI / Gemini); sleeps through rate limits and billing outages
    board_tools.py     python-chess lookups (attack map, inventory, hanging, forcing, legality, pawns)
    tool_usage.py      did the model use the facts block? lexical check + join with r4 judge wishes
    candidates/        every prompt tried, c0_baseline first; best.json = r2 winner, best_r5.json = current

Loop (MIPROv2 shape): baseline on dev -> judge -> bootstrap demos from train generations the
judge rates >=4 -> subagent proposes K instructions from the failures -> screen candidates on a
40-example minibatch (`screen.py --split train` since round 5) -> full dev for the winner ->
repeat with the winner as parent. Candidate keys: `context_file` (lessons), `tools` (facts
block from board_tools after every board), `thinking`, `max_tokens`. Judges: JUDGE.md (r1-3),
JUDGE2.md (r4), JUDGE3.md (r5, adds facts_used / facts_contradicted).

`data/` and `runs/` are git-ignored: they contain corpus text (see ../commentary/LICENSING.md).
