"""Target-model client: OpenAI or Gemini over plain HTTPS, chosen by LLM_MODEL.

  LLM_MODEL=gpt-5-nano          (default; cheapest/fastest OpenAI text model)
  LLM_MODEL=gemini-3.8-flash    (any name starting with "gemini" routes to Google)

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


def generate(prompt, *, model=MODEL, system=None, temperature=1.0, max_tokens=600, log=sys.stderr,
             thinking=None, tools=None, max_rounds=10):
    """Returns {"text", "thoughts", "usage", "finish", "calls"}.

    thinking: None = model default; an int = thinkingBudget; "low"/"medium"/"high" = thinkingLevel.
    Gemini returns thought *summaries* in `thoughts` (raw thinking is not exposed by the API).
    tools: an object with .declarations() and .call(name, args) -> str (Gemini only). The model may
    call functions for up to max_rounds turns; every call is logged in `calls`."""
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
        for _ in range(max_rounds):
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
            responses = []
            for fc in fcs:
                res = tools.call(fc["name"], fc.get("args") or {})
                rec = {"name": fc["name"], "args": fc.get("args") or {}, "result": res}
                if getattr(tools, "last_functions", None):
                    rec["functions"] = tools.last_functions
                calls.append(rec)
                responses.append({"functionResponse": {"name": fc["name"], "response": {"result": res}}})
            body["contents"].append({"role": "user", "parts": responses})
        body.pop("tools", None)  # rounds exhausted: one last turn, no tools, must answer
        body["contents"].append({"role": "user", "parts": [{"text": "Stop checking. Write the commentary now."}]})
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
