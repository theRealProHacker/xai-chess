# commentary_gen — DSPy-style prompt optimisation, run by hand

Task: `(moves so far, ASCII board) -> human commentary on the last move`. Target model: `LLM_MODEL`
(default `gpt-5-nano`; OpenAI or Gemini, keys in `../.env`). The optimiser is not DSPy: instruction proposals, demo
selection and the judge are Claude subagents driven from the session; this directory holds the
deterministic parts and every artifact they produce.

    build_dataset.py   corpus -> data/{pool,train,dev}.jsonl   (best 2000, stratified: 700/700/400/200)
    program.py         candidate json (instruction + demo ids) -> runs/<name>/<split>.jsonl
    metric.py          chrF + length ratio; packs runs into judge batches, unpacks judge scores
    llm.py             REST client (OpenAI / Gemini); sleeps through rate limits and billing outages
    candidates/        every prompt tried, c0_baseline first

Loop (MIPROv2 shape): baseline on dev -> judge -> bootstrap demos from train generations the
judge rates >=4 -> subagent proposes K instructions from the failures -> screen candidates on a
60-example dev minibatch -> full dev for the top two -> repeat with the winner as parent.

`data/` and `runs/` are git-ignored: they contain corpus text (see ../commentary/LICENSING.md).
