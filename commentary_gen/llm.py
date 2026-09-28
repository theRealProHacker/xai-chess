"""Target-model client: OpenAI or Gemini over plain HTTPS, chosen by LLM_MODEL.

  LLM_MODEL=gpt-5-nano          (default; cheapest/fastest OpenAI text model)
  LLM_MODEL=gemini-3.8-flash    (any name starting with "gemini" routes to Google)
  LLM_MODEL=chessgpt-chat-v1    (Waterhorse/ChessGPT, local CPU via transformers; also chessgpt-base-v1.
                                 2048-token context, no tools; run with ../chessgpt/.venv/bin/python)

Rate-limit 429s sleep for the server's retry hint and retry forever. Billing 429s
("no credits", "credits are depleted") poll every 10 minutes until the account works again.
"""
import json, os, re, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
for env in (ROOT / ".env", Path(__file__).parent / ".env"):
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))

MODEL = os.environ.get("LLM_MODEL", "gpt-5-nano")
BILLING = ("credits are depleted", "no credits remaining", "insufficient_quota", "billing")


def _post(url, body, headers, log):
    data = json.dumps(body).encode()
    backoff = 5
    while True:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 429 and any(b in msg.lower() for b in BILLING):
                print("[llm] billing exhausted; polling again in 10 min", file=log, flush=True)
                time.sleep(600)
                continue
            if e.code == 429 or e.code >= 500:
                m = re.search(r"retryDelay\"?:\s*\"?(\d+)|try again in (\d+(?:\.\d+)?)s", msg)
                wait = (float(m.group(1) or m.group(2)) + 1) if m else backoff
                print(f"[llm] {e.code}; sleeping {wait:.0f}s", file=log, flush=True)
                time.sleep(min(wait, 300))
                backoff = min(backoff * 2, 120)
                continue
            raise RuntimeError(f"{e.code}: {msg[:500]}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[llm] {e}; retry in {backoff}s", file=log, flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)


_LOCAL = {}
_LOCAL_LOCK = __import__("threading").Lock()


def _chessgpt(prompt, model, system, temperature, max_tokens, tools, prefill, stop, repetition_penalty):
    """ChessGPT (arXiv 2306.09200): GPT-NeoX 2.8B, one model in memory, one generation at a time.
    prefill starts the model's own turn (e.g. "NOTE:"); generation stops at the first `stop` string."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if tools is not None:
        raise ValueError("ChessGPT has no function calling")
    with _LOCAL_LOCK:
        if model not in _LOCAL:
            repo = "Waterhorse/" + model
            _LOCAL[model] = (AutoTokenizer.from_pretrained(repo), AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval())
        tok, lm = _LOCAL[model]
        text = f"{system}\n\n{prompt}" if system else prompt
        if "chat" in model:  # the model card's chat format
            text = f"A friendly, helpful chat between some humans.<|endoftext|>Human 0: {text}<|endoftext|>Human 1:"
        if prefill:
            text += " " + prefill
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] + max_tokens > lm.config.max_position_embeddings:
            raise ValueError(f"prompt {ids.shape[1]} + max_tokens {max_tokens} tokens exceeds the "
                             f"{lm.config.max_position_embeddings}-token context")
        with torch.inference_mode():
            sample = dict(do_sample=True, temperature=temperature, top_p=0.7, top_k=50) if temperature > 0 else {}
            out = lm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                              repetition_penalty=repetition_penalty, stop_strings=[stop], tokenizer=tok,
                              pad_token_id=tok.eos_token_id, **sample)
        new = out[0, ids.shape[1]:]
    stopped = new[-1].item() == tok.eos_token_id
    text = tok.decode(new, skip_special_tokens=True)
    stopped = stopped or stop in text
    return {"text": text.split("\n---")[0].split(stop)[0].strip(), "thoughts": "",
            "usage": {"prompt_tokens": ids.shape[1], "completion_tokens": len(new)},
            "finish": "stop" if stopped else "length", "calls": []}


def generate(prompt, *, model=MODEL, system=None, temperature=1.0, max_tokens=600, log=sys.stderr,
             thinking=None, tools=None, max_rounds=10, prefill=None, stop="\n\n", repetition_penalty=1.15):
    """Returns {"text", "thoughts", "usage", "finish", "calls"}.

    thinking: None = model default; an int = thinkingBudget; "low"/"medium"/"high" = thinkingLevel.
    Gemini returns thought *summaries* in `thoughts` (raw thinking is not exposed by the API).
    tools: an object with .declarations() and .call(name, args) -> str (Gemini only). The model may
    call functions for up to max_rounds turns; every call is logged in `calls`."""
    if model.startswith("chessgpt"):
        return _chessgpt(prompt, model, system, temperature, max_tokens, tools, prefill, stop, repetition_penalty)
    if prefill:
        raise ValueError("prefill is only implemented for local models")
    if model.startswith("gemini"):
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if tools is not None:
            body["tools"] = [{"functionDeclarations": tools.declarations()}]
        if thinking is None and "LLM_THINKING" in os.environ:
            thinking = os.environ["LLM_THINKING"]
        if thinking is not None:
            cfg = {"includeThoughts": True}
            if str(thinking).lstrip("-").isdigit():
                cfg["thinkingBudget"] = int(thinking)
            else:
                cfg["thinkingLevel"] = str(thinking)
            body["generationConfig"]["thinkingConfig"] = cfg
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
               f"?key={os.environ['GEMINI_API_KEY']}")
        calls, thoughts, usage, nudged = [], [], {}, False
        rounds, extra, last_failed = 0, 0, False
        while rounds < max_rounds + extra:
            rounds += 1
            try:
                out = _post(url, body, {}, log)
            except RuntimeError as e:  # models without a thinking mode reject thinkingConfig
                if "INVALID_ARGUMENT" not in str(e) or "thinkingConfig" not in body["generationConfig"]:
                    raise
                del body["generationConfig"]["thinkingConfig"]
                out = _post(url, body, {}, log)
            for k, v in out.get("usageMetadata", {}).items():
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v
            cands = out.get("candidates") or []
            content = cands[0].get("content", {}) if cands else {}
            parts = content.get("parts", [])
            thoughts += [p["text"] for p in parts if p.get("thought") and p.get("text")]
            fcs = [p["functionCall"] for p in parts if "functionCall" in p]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if not fcs and not text and tools is not None and calls and not nudged:
                nudged = True  # a tool round ended with no text: ask once for the answer
                body["contents"].append(content)
                body["contents"].append({"role": "user", "parts": [{"text": "Now write the commentary."}]})
                continue
            if not fcs or tools is None:
                return {"text": text,
                        "thoughts": "\n".join(thoughts).strip(), "usage": usage,
                        "finish": cands[0].get("finishReason") if cands else out.get("promptFeedback"),
                        "calls": calls}
            body["contents"].append(content)  # verbatim, keeps thought signatures
            responses, last_failed = [], False
            for fc in fcs:
                res = tools.call(fc["name"], fc.get("args") or {})
                # a call that did not happen must not look like a call that returned a fact
                failed = res.startswith("Error:")
                last_failed |= failed
                rec = {"name": fc["name"], "args": fc.get("args") or {}, "result": res, "failed": failed}
                if getattr(tools, "last_functions", None):
                    rec["functions"] = tools.last_functions
                calls.append(rec)
                responses.append({"functionResponse": {"name": fc["name"],
                                                      "response": {"error": res} if failed else {"result": res}}})
            body["contents"].append({"role": "user", "parts": responses})
            if last_failed and extra < 2:
                extra += 1  # do not spend the budget on a call that never happened
        body.pop("tools", None)  # rounds exhausted: one last turn, no tools, must answer
        body["contents"].append({"role": "user", "parts": [{"text":
            "The last lookup failed, so ignore it and use the facts you already have. Write the commentary now."
            if last_failed else "Stop checking. Write the commentary now."}]})
        out = _post(url, body, {}, log)
        parts = (out.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return {"text": "".join(p.get("text", "") for p in parts if not p.get("thought")).strip(),
                "thoughts": "\n".join(thoughts).strip(), "usage": usage, "finish": "MAX_ROUNDS", "calls": calls}

    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_completion_tokens": max_tokens}
    if re.match(r"gpt-5|o\d", model):  # reasoning models: no temperature, minimal thinking
        body["reasoning_effort"] = os.environ.get("LLM_REASONING", "minimal")
    else:
        body["temperature"] = temperature
    out = _post("https://api.openai.com/v1/chat/completions", body,
                {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, log)
    ch = out["choices"][0]
    return {"text": (ch["message"].get("content") or "").strip(), "thoughts": "", "usage": out.get("usage", {}),
            "finish": ch.get("finish_reason")}


if __name__ == "__main__":
    print(generate(" ".join(sys.argv[1:]) or "Say OK.", max_tokens=50))
