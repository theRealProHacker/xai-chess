"""GRPO on the stage-2 oracle with Qwen3.5's own thinking mode: grpo.py with enable_thinking=True. The prompt ends in
"<think>\\n"; the model writes free reasoning, "</think>", then the answer. No reasoning traces are given: the reward
only sees the answer after "</think>", and an answer without "</think>" (e.g. cut off at --max-new) counts as unparsed (-1).
Not R1-Zero: Qwen3.5-0.8B is post-trained and already thinks. Default start: a stage-1 checkpoint's projector + a fresh
LoRA, so the LLM reasons as Qwen shipped it and the KL reference is that model. A stage-2 checkpoint continues its own LoRA.
Rollouts and evals use left-padded HF generate, so prompts of different lengths share a batch.
Sampling follows the Qwen3.5 card for thinking on text (top-p 0.95, top-k 20, presence penalty 1.5 on generated
tokens); the update's log-probs are the raw policy's, so rollouts are slightly off-policy. Evals are greedy plus the
presence penalty. The card asks for 32,768 output tokens; --max-new is lower for cost.

Each step: --prompts train positions, each with a random question (qa.sample); --group answers per prompt sampled at
--temp; reward per answer; advantage = (reward - group mean) / group std; one policy-gradient update on the LoRA
(projector frozen) with a k3 KL penalty (--beta) to the starting LoRA weights.

Reward:  unparsed -1;  parsed but a format the question doesn't allow, or the wrong subject -0.5;  otherwise accuracy in [0, 1]:
  pawns (A, E)   exp(-|pawns - true| / 2)          bin (D, E)   1 exact, 0.5 off by one, 0 else   (E: mean of both)
  win % (B)      0 wrong side, else max(0, 1 - 2|win - true|)
  score (C)      max(0, 1 - 2|score - true|)

Checkpoints (last.pt, final.pt) load in train.py --eval-only. Val eval every --eval-every steps, test once at the end.
  python oracle/grpo_think.py --data oracle/data_g50k --out oracle/runs/g50k_think --init-ckpt oracle/runs/q08b/epoch9.pt
"""
import argparse, contextlib, json, math, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oracle"))
import numpy as np, torch, torch.nn as nn
import qa

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "oracle/data_g50k"))
ap.add_argument("--out", default=str(ROOT / "oracle/runs/g50k_think"))
ap.add_argument("--init-ckpt", required=True, metavar="CKPT", help="projector + mu/sd; a stage-2 checkpoint also gives the LoRA (KL reference = starting LoRA)")
ap.add_argument("--lora", type=int, default=16, metavar="R", help="fresh LoRA rank when --init-ckpt is stage 1")
ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj")
ap.add_argument("--dtype", default="auto", help="auto = bf16 on Ampere+, fp32 on older cards")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--prompts", type=int, default=32, help="positions per step")
ap.add_argument("--group", type=int, default=8, help="sampled answers per position")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--presence", type=float, default=1.5, help="subtracted from the logit of every token already generated")
ap.add_argument("--beta", type=float, default=0.04, help="KL penalty to the starting LoRA")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--micro", type=int, default=8, help="sequences per forward/backward in the update")
ap.add_argument("--gen-batch", type=int, default=256)
ap.add_argument("--train-n", type=int, default=0)
ap.add_argument("--eval-n", type=int, default=300)
ap.add_argument("--eval-batch", type=int, default=150)
ap.add_argument("--max-new", type=int, default=4096, help="reasoning + answer tokens")
ap.add_argument("--save-every", type=int, default=50)
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--test-n", type=int, default=1000)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed)

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# ---- data ----
D = np.load(Path(a.data) / "labels.npz")
wdl, black, split = D["wdl"], D["black"], D["split"]
raw = torch.from_numpy(np.load(Path(a.data) / "acts.npy").astype(np.float32))        # (N, 64, 256)
tr = np.flatnonzero(split == 0)
if a.train_n: tr = tr[:a.train_n]
val = [json.loads(l) for l in open(Path(a.data) / "val.jsonl")][:a.eval_n]
test = [json.loads(l) for l in open(Path(a.data) / "test.jsonl")][:a.test_n] if a.test_n else []

# ---- model ----
init = torch.load(a.init_ckpt, map_location="cpu")
lcfg = init.get("lora") or dict(r=a.lora, alpha=2 * a.lora, dropout=0.0, targets=a.lora_targets)  # fresh: lora_B = 0, i.e. the base LLM
model_name = init["model"]
mu, sd = init["mu"], init["sd"]; acts = (raw - mu) / sd                               # the checkpoint's normalisation
cuda = torch.cuda.is_available()
if a.dtype == "auto":
    dt = torch.bfloat16 if cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
else:
    dt = getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(model_name)
kw = dict(dtype=dt, device_map="auto" if cuda else None)
try:
    llm = AutoModelForCausalLM.from_pretrained(model_name, **kw)
except ValueError:                                                                     # multimodal-only config
    from transformers import AutoModelForImageTextToText
    llm = AutoModelForImageTextToText.from_pretrained(model_name, **kw)
llm.requires_grad_(False).eval()
emb = llm.get_input_embeddings()
dev = emb.weight.device
names = set(lcfg["targets"].split(","))
tgt = [n for n, m in llm.named_modules() if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in names
       and "visual" not in n and "vision" not in n]
llm = get_peft_model(llm, LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"], target_modules=tgt))
if init.get("lora"): set_peft_model_state_dict(llm, {k: v.to(dev) for k, v in init["lora_sd"].items()})
llm.train()
lora = [(n, p) for n, p in llm.named_parameters() if p.requires_grad]
ref = [p.detach().clone() for _, p in lora]                                             # KL reference = starting LoRA
print(f"LoRA r={lcfg['r']} on {len(tgt)} modules: {sum(p.numel() for _, p in lora):,} params, {'from' if init.get('lora') else 'fresh, projector from'} {a.init_ckpt}", flush=True)
END = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token
end_id = tok.convert_tokens_to_ids(END)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else end_id
think_end = tok.convert_tokens_to_ids("</think>")

PH = "@@BOARD@@"
chat = tok.apply_chat_template([{"role": "user", "content": f"Position: <board>{PH}</board>\n" + "@@Q@@"}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=True)
pre_s, post_s = chat.split(PH)
PRE = tok(pre_s, add_special_tokens=False)["input_ids"]
P = len(PRE)
assert P == init["P"], f"prompt prefix is {P} tokens, the checkpoint was trained with {init['P']}"


def prompt_ids(question):
    return PRE + [pad_id] * 64 + tok(post_s.replace("@@Q@@", question), add_special_tokens=False)["input_ids"]


class Projector(nn.Module):
    """As in train.py; frozen here."""
    def __init__(self, d_in, d_hid, d_out, rms):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(64, d_in))
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))
        self.norm = nn.LayerNorm(d_out, elementwise_affine=False)
        self.scale = nn.Parameter(torch.tensor(float(rms)))

    def forward(self, x):
        return self.norm(self.net(x + self.pos)) * self.scale


proj = Projector(256, init["args"]["hidden"], emb.weight.shape[1], 0.0).to(dev)
proj.load_state_dict({k: v.to(dev) for k, v in init["proj"].items()}); proj.requires_grad_(False).eval()
opt = torch.optim.AdamW([p for _, p in lora], lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
start, rng_state = 0, None
if (out / "last.pt").exists():
    ck = torch.load(out / "last.pt", map_location=dev)
    set_peft_model_state_dict(llm, ck["lora_sd"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start, rng_state = ck["step"], ck["rng"]
    torch.set_rng_state(ck["trng"].cpu())
    if "crng" in ck and cuda: torch.cuda.set_rng_state(ck["crng"].cpu())
    print(f"resumed from step {start}/{a.steps}", flush=True)
rng = np.random.default_rng(a.seed)
if rng_state: rng.bit_generator.state = rng_state


def embed(ids_list, soft):
    """Right-padded (B, L, d) inputs with soft tokens at [P, P+64) and the attention mask."""
    L = max(map(len, ids_list))
    ids = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_list), L), dtype=torch.long)
    for k, s in enumerate(ids_list):
        ids[k, :len(s)] = torch.tensor(s); mask[k, :len(s)] = 1
    x = emb(ids.to(dev)).clone()
    x[:, P:P + 64] = soft.to(x.dtype)
    return x, mask.to(dev)


@contextlib.contextmanager
def ref_weights():
    """Swap the starting LoRA in for a forward pass."""
    cur = [p.detach().clone() for _, p in lora]
    with torch.no_grad():
        for (_, p), r in zip(lora, ref): p.copy_(r)
    try: yield
    finally:
        with torch.no_grad():
            for (_, p), c in zip(lora, cur): p.copy_(c)


class Presence(LogitsProcessor):
    """Presence penalty as in vLLM/OpenAI. Custom processors run before HF's temperature/top-k/top-p, as there.
    With inputs_embeds, generate passes only the generated ids, so the prompt is not penalised."""
    def __call__(self, ids, scores):
        if not ids.shape[1]: return scores
        return scores - a.presence * torch.zeros_like(scores, dtype=torch.bool).scatter_(1, ids, True)


@torch.no_grad()
def generate(prompts, idx, temp=0.0, bs=None):
    """Token ids (up to and including END) per prompt; temp 0 = greedy. Left-padded, so any prompt lengths batch together."""
    was_training = llm.training
    llm.eval()
    gc = GenerationConfig(max_new_tokens=a.max_new, do_sample=temp > 0, temperature=temp or None,
                          top_k=a.top_k if temp > 0 else None, top_p=a.top_p if temp > 0 else 1.0,
                          eos_token_id=end_id, pad_token_id=pad_id)
    lp_proc = [Presence()] if a.presence else None
    outs = []
    for s in range(0, len(prompts), bs or a.eval_batch):
        pp = prompts[s:s + (bs or a.eval_batch)]
        L = max(map(len, pp))
        ids = torch.full((len(pp), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(pp), L), dtype=torch.long)
        for k, p in enumerate(pp):
            ids[k, L - len(p):] = torch.tensor(p); mask[k, L - len(p):] = 1
        x = emb(ids.to(dev)).clone()
        soft = proj(acts[idx[s:s + len(pp)]].to(dev)).to(x.dtype)
        for k, p in enumerate(pp):
            x[k, L - len(p) + P:L - len(p) + P + 64] = soft[k]
        gen = llm.generate(inputs_embeds=x, attention_mask=mask.to(dev), generation_config=gc, logits_processor=lp_proc).tolist()
        outs += [g[:g.index(end_id) + 1] if end_id in g else g for g in gen]
    llm.train(was_training)
    return outs


def text(g):
    """The answer after </think>; "" if the model never closed its reasoning."""
    if think_end not in g: return ""
    return tok.decode([t for t in g[g.index(think_end) + 1:] if t != end_id], skip_special_tokens=True).strip()


def thought(g):
    return tok.decode(g[:g.index(think_end)] if think_end in g else g, skip_special_tokens=True).strip()


def logps(prompts, comps, idx):
    """(B, T) log-probs of the answer tokens and their mask; T spans from the shortest prompt's end."""
    ids = [p + c for p, c in zip(prompts, comps)]
    x, mask = embed(ids, proj(acts[idx].to(dev)))
    s0, L = min(map(len, prompts)) - 1, mask.shape[1]
    logits = llm(inputs_embeds=x, attention_mask=mask, logits_to_keep=slice(s0, L - 1), use_cache=False).logits.float()
    y = torch.full((len(ids), L), pad_id, dtype=torch.long)
    cm = torch.zeros((len(ids), L - 1 - s0))
    for k, (p, c) in enumerate(zip(prompts, comps)):
        y[k, :len(p) + len(c)] = torch.tensor(p + c); cm[k, len(p) - 1 - s0:len(p) - 1 - s0 + len(c)] = 1
    y = y[:, s0 + 1:L].to(logits.device)
    lp = logits.gather(-1, y[..., None])[..., 0] - logits.logsumexp(-1)
    return lp, cm.to(logits.device)


def truth(r):
    w, d, l, _ = qa.view(wdl[r["i"]], black[r["i"]], r["q"], r["colour"])
    p = qa.pawns(w, l)
    return dict(w=w, d=d, l=l, pawns=p, bin=qa.bin_of(p))


def reward(r, answer):
    p = qa.parse(answer)
    if p is None: return -1.0
    subject = r["colour"] if r["q"] == qa.COLOUR_Q else "the side to move"
    if p["fmt"] not in qa.QUESTIONS[r["q"]][1] or (p.get("subject") is not None and p["subject"].lower() != subject.lower()):
        return -0.5
    t = truth(r)
    acc = []
    if "pawns" in p: acc.append(math.exp(-abs(p["pawns"] - t["pawns"]) / 2))
    if p["fmt"] in "DE":
        acc.append({0: 1.0, 1: 0.5}.get(abs(p["bin"] - t["bin"]), 0.0))
    if p["fmt"] == "B":
        side = 1 if t["w"] >= t["l"] else -1
        acc.append(0.0 if p["side"] != side else max(0.0, 1 - 2 * abs(p["win"] - (t["w"] if side == 1 else t["l"]))))
    if p["fmt"] == "C":
        acc.append(max(0.0, 1 - 2 * abs(p["score"] - (t["w"] + t["d"] / 2))))
    return float(np.mean(acc))


def score(rows, preds):
    m = {k: [] for k in ("parsed", "exact", "format_ok", "subject_ok", "side", "side_q8", "bin", "bin_pm1",
                         "pawn_mae", "win_mae", "score_mae", "reward")}
    for r, a_ in zip(rows, preds):
        t, p = truth(r), qa.parse(a_)
        m["reward"].append(reward(r, a_))
        m["parsed"].append(p is not None); m["exact"].append(a_ == r["answer"])
        if p is None: continue
        m["format_ok"].append(p["fmt"] in qa.QUESTIONS[r["q"]][1])
        if p.get("subject") is not None:
            m["subject_ok"].append(p["subject"].lower() == (r["colour"] if r["q"] == qa.COLOUR_Q else "the side to move").lower())
        if "bin" in p:
            ps, ts = np.sign(p["bin"]), np.sign(t["bin"])
            m["bin"].append(p["bin"] == t["bin"]); m["bin_pm1"].append(abs(p["bin"] - t["bin"]) <= 1)
        elif "win" in p:
            ps, ts = p["side"], (1 if t["w"] >= t["l"] else -1)
            m["win_mae"].append(abs(p["win"] - (t["w"] if p["side"] == 1 else t["l"])))
        else:
            ps, ts = np.sign(p["score"] - 0.5), np.sign(t["w"] + t["d"] / 2 - 0.5)
            m["score_mae"].append(abs(p["score"] - (t["w"] + t["d"] / 2)))
        m["side"].append(ps == ts)
        if r["q"] == qa.COLOUR_Q: m["side_q8"].append(ps == ts)
        if "pawns" in p: m["pawn_mae"].append(abs(p["pawns"] - t["pawns"]))
    return {k: (round(float(np.mean(v)), 4) if v else None) for k, v in m.items()}


mean_wdl = wdl[np.flatnonzero(split == 0)].mean(0)
base_mean = lambda rows: score(rows, [qa.render(mean_wdl, black[r["i"]], r["q"], r["fmt"], r["colour"]) for r in rows])
print(f"{model_name} {dt}, {len(tr):,} train positions, {a.steps} steps x {a.prompts} prompts x {a.group} samples, "
      f"eval on {len(val)} val", flush=True)


def reported(step, report):
    if not report.exists(): return False
    done = set()
    for l in open(report):
        try: done.add(json.loads(l)["step"])
        except (ValueError, KeyError): pass                                             # a line cut off by a kill
    return step in done


def save(ck, path):
    torch.save(ck, path.with_suffix(".tmp")); os.replace(path.with_suffix(".tmp"), path)


def ckpt(step):
    """train.py's checkpoint fields (so --eval-only reads it) plus GRPO state."""
    return dict(proj=proj.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=0, step=step,
                rng=rng.bit_generator.state, trng=torch.get_rng_state(), crng=torch.cuda.get_rng_state() if cuda else None,
                mu=mu, sd=sd, model=model_name, P=P, args={**init["args"], "grpo": vars(a)},
                lora=lcfg, lora_sd=get_peft_model_state_dict(llm))


def evaluate(step, t0, report, rows=None, name="val"):
    rows = rows or val; idx = [r["i"] for r in rows]
    t1 = time.time()
    prompts = [prompt_ids(r["question"]) for r in rows]
    gens = generate(prompts, idx)
    preds = [text(g) for g in gens]
    shuf = [text(g) for g in generate(prompts, list(np.random.default_rng(step).permutation(idx)))]
    rep = dict(epoch=0, step=step, data=name, train_min=round((t1 - t0) / 60, 1) if t0 else None,
               eval_min=round((time.time() - t1) / 60, 1), model=score(rows, preds), shuffled=score(rows, shuf), mean=base_mean(rows))
    with open(report, "a") as f: f.write(json.dumps(rep) + "\n")
    tag = "test" if name == "test" else f"step{step}"
    with open(out / f"samples_{tag}.jsonl", "w") as f:
        for r, p_, s_, g in zip(rows, preds, shuf, gens):
            f.write(json.dumps(dict(question=r["question"], answer=r["answer"], pred=p_, shuffled=s_, fen=r["fen"],
                                    think=thought(g), think_len=g.index(think_end) if think_end in g else None)) + "\n")
    print(f"\n== {name} step {step}/{a.steps} ==", flush=True)
    keys = ["reward", "parsed", "format_ok", "subject_ok", "exact", "side", "side_q8", "bin", "bin_pm1", "pawn_mae", "win_mae", "score_mae"]
    print(f"{'':12}" + "".join(f"{k:>11}" for k in keys))
    for name in ("model", "shuffled", "mean"):
        print(f"{name:12}" + "".join(f"{'-' if rep[name][k] is None else rep[name][k]:>11}" for k in keys))
    for r, p_, g in list(zip(rows, preds, gens))[:5]:
        print(f"  Q: {r['question']}\n     think: {thought(g)[:300]!r}\n     true: {r['answer']}\n     pred: {p_}")
    print(flush=True)


def step_fn():
    """One GRPO step; returns stats."""
    pos = rng.choice(tr, a.prompts, replace=False)
    rows = []
    for i in pos:
        q, _, colour, qt, at = qa.sample(rng, wdl[i], black[i])
        rows += [dict(i=int(i), q=q, colour=colour, question=qt)] * a.group
    idx = [r["i"] for r in rows]
    prompts = [prompt_ids(r["question"]) for r in rows]
    comps = generate(prompts, idx, a.temp, a.gen_batch)
    R = torch.tensor([reward(r, text(c)) for r, c in zip(rows, comps)]).view(a.prompts, a.group)
    A = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).flatten()
    opt.zero_grad(set_to_none=True)
    kls, n = 0.0, len(rows)
    for s in range(0, n, a.micro):
        sl = slice(s, s + a.micro)
        with torch.no_grad(), ref_weights():
            ref_lp, _ = logps(prompts[sl], comps[sl], idx[sl])
        lp, cm = logps(prompts[sl], comps[sl], idx[sl])
        d = ref_lp - lp
        kl = d.exp() - d - 1                                                            # k3 estimator, >= 0
        per_tok = -torch.exp(lp - lp.detach()) * A[sl].to(lp.device)[:, None] + a.beta * kl
        per_seq = (per_tok * cm).sum(1) / cm.sum(1).clamp(min=1)
        (per_seq.sum() / n).backward()
        kls += ((kl.detach() * cm).sum(1) / cm.sum(1).clamp(min=1)).sum().item()
    gn = torch.nn.utils.clip_grad_norm_([p for _, p in lora], 1.0).item()
    opt.step(); sched.step()
    return dict(reward=R.mean().item(), parsed=(R > -1).float().mean().item(), closed=float(np.mean([think_end in c for c in comps])), flat=(R.std(1) < 1e-6).float().mean().item(),
                kl=kls / n, len=float(np.mean([len(c) for c in comps])), gn=gn)


t0, logs = time.time(), []
if start and start % a.eval_every == 0 and start < a.steps and not reported(start, out / "report.jsonl"):
    evaluate(start, None, out / "report.jsonl")                                         # killed during this eval
if start == 0 and not reported(0, out / "report.jsonl"):
    evaluate(0, None, out / "report.jsonl")                                             # the starting checkpoint
for s in range(start, a.steps):
    logs.append(step_fn())
    if (s + 1) % a.log_every == 0 or s + 1 == a.steps:
        el = (time.time() - t0) / (s + 1 - start)
        m = {k: np.mean([l[k] for l in logs]) for k in logs[0]}
        print(f"step {s + 1}/{a.steps} reward {m['reward']:+.3f} parsed {m['parsed']:.3f} closed {m['closed']:.3f} flat-groups {m['flat']:.2f} "
              f"kl {m['kl']:.4f} len {m['len']:.1f} gnorm {m['gn']:.2f} {el:.2f}s/step eta {el * (a.steps - s - 1) / 60:.1f}min", flush=True)
        logs = []
    if (s + 1) % a.eval_every == 0 or s + 1 == a.steps:
        save(ckpt(s + 1), out / "last.pt"); evaluate(s + 1, t0, out / "report.jsonl")
    elif (s + 1) % a.save_every == 0:
        save(ckpt(s + 1), out / "last.pt")
if start < a.steps or not (out / "final.pt").exists():
    save(ckpt(a.steps), out / "final.pt")
if a.steps and not reported(a.steps, out / "report.jsonl"):
    evaluate(a.steps, None, out / "report.jsonl")                                        # killed during the last eval
if test and not reported(a.steps, out / "report_test.jsonl"):
    evaluate(a.steps, None, out / "report_test.jsonl", test, "test")
