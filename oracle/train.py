"""Stage 1 eval oracle: train a projector from Leela's layer-9 tokens into a frozen LLM's input embeddings.

Each sample is   <chat prefix> 64 soft tokens <question> -> <answer>,  loss on the answer only.
Train questions are resampled every epoch; val is the fixed oracle/data/val.jsonl.
After every epoch: checkpoint (epoch{e}.pt, last.pt), greedy answers on --eval-n val rows
(samples_epoch{e}.jsonl), and one line in report.jsonl. Baselines in the report: the training-mean WDL
rendered in the asked format ("mean"), and the same model fed another position's tokens ("shuffled").

  python oracle/train.py --out oracle/runs/q08b            # resumes from last.pt if it exists
Stage 2 (--lora R): projector + LoRA on the LLM's linear layers, same loss; --save-every keeps last.pt fresh mid-epoch.
--eval-every N adds mid-epoch checkpoint + val evals; test.jsonl is scored once, after the last epoch (report_test.jsonl).
  python oracle/train.py --out oracle/runs/q08b_lora --lora 16 --init-proj oracle/runs/q08b/epoch9.pt
"""
import argparse, contextlib, json, math, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oracle"))
import numpy as np, torch, torch.nn as nn
import qa

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "oracle/data"))
ap.add_argument("--out", default=str(ROOT / "oracle/runs/q08b"))
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--dtype", default="auto", help="auto = bf16 on Ampere+, fp32 on older cards (Pascal fp16 is ~64x slow)")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--train-n", type=int, default=0, help="use only the first n train positions (0 = all)")
ap.add_argument("--eval-n", type=int, default=300)
ap.add_argument("--eval-batch", type=int, default=32)
ap.add_argument("--max-new", type=int, default=24)
ap.add_argument("--grad-ckpt", action="store_true", help="recompute LLM activations in backward: less memory, ~30%% slower")
ap.add_argument("--no-cache",action="store_true", help="generate without KV cache (slow; for checking the cached path)")
ap.add_argument("--eval-only", nargs="+", metavar="CKPT", help="score saved checkpoints on --data's val split and exit")
ap.add_argument("--lora", type=int, default=0, metavar="R", help="LoRA rank on the LLM (0 = frozen LLM, stage 1)")
ap.add_argument("--lora-alpha", type=float, help="default 2R")
ap.add_argument("--lora-dropout", type=float, default=0.0)
ap.add_argument("--lora-lr", type=float, default=2e-4)
# full attention q/k/v/o, gated-delta-net in_proj_qkv/in_proj_z/out_proj, MLPs; not in_proj_a/b (16-wide gate heads)
ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj")
ap.add_argument("--init-proj", metavar="CKPT", help="warm-start the projector (and its mu/sd) from a checkpoint; ignored when resuming")
ap.add_argument("--init-ckpt", metavar="CKPT", help="warm-start projector, mu/sd and LoRA weights from a checkpoint (LoRA flags must match)")
ap.add_argument("--save-every", type=int, default=200, help="steps between mid-epoch last.pt saves (0 = epoch end only)")
ap.add_argument("--eval-every", type=int, default=0, help="also checkpoint + val eval every N steps within an epoch")
ap.add_argument("--test-n", type=int, default=1000, help="test rows scored once after the last epoch (0 = none)")
ap.add_argument("--log-every", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed)

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- data ----
D = np.load(Path(a.data) / "labels.npz")
wdl, black, split = D["wdl"], D["black"], D["split"]
acts = torch.from_numpy(np.load(Path(a.data) / "acts.npy").astype(np.float32))       # (N, 64, 256)
tr = np.flatnonzero(split == 0)
if a.train_n: tr = tr[:a.train_n]
raw = acts
if len(tr):
    mu, sd = acts[tr].mean((0, 1)), acts[tr].std((0, 1)) + 1e-6                      # per channel, train only
    acts = (acts - mu) / sd
val = [json.loads(l) for l in open(Path(a.data) / "val.jsonl")][:a.eval_n]
test = [json.loads(l) for l in open(Path(a.data) / "test.jsonl")][:a.test_n] if a.test_n else []

# ---- model ----
cuda = torch.cuda.is_available()
if a.dtype == "auto":
    dt = torch.bfloat16 if cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
else:
    dt = getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(a.model)
kw = dict(dtype=dt, device_map=("balanced" if torch.cuda.device_count() > 1 else "auto") if cuda else None)  # "auto" fills GPU 0 first
try:
    llm = AutoModelForCausalLM.from_pretrained(a.model, **kw)
except ValueError:                                                                     # multimodal-only config
    from transformers import AutoModelForImageTextToText
    llm = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
llm.requires_grad_(False).eval()
if a.grad_ckpt:                                   # HF only checkpoints in train mode; the LLM has no dropout
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()
emb = llm.get_input_embeddings()
dev = emb.weight.device
lcfg = dict(r=a.lora, alpha=a.lora_alpha or 2 * a.lora, dropout=a.lora_dropout, targets=a.lora_targets) if a.lora else None
if a.eval_only:                                   # the checkpoints' LoRA config wins over the flags
    lcs = {json.dumps(c) for c in (torch.load(c, map_location="cpu").get("lora") for c in a.eval_only) if c}
    assert len(lcs) <= 1, f"checkpoints disagree on LoRA config: {lcs}"
    lcfg = json.loads(lcs.pop()) if lcs else None
lora_params = []
if lcfg:
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    names = set(lcfg["targets"].split(","))
    tgt = [n for n, m in llm.named_modules() if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in names
           and "visual" not in n and "vision" not in n]
    llm = get_peft_model(llm, LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"], target_modules=tgt))
    llm.train()                                   # LoRA dropout; generate() switches to eval and back
    lora_params = [p for p in llm.parameters() if p.requires_grad]
    print(f"LoRA r={lcfg['r']} on {len(tgt)} modules: {sum(p.numel() for p in lora_params):,} params", flush=True)
END = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token
end_id = tok.convert_tokens_to_ids(END)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else end_id

PH = "@@BOARD@@"
chat = tok.apply_chat_template([{"role": "user", "content": f"Position: <board>{PH}</board>\n" + "@@Q@@"}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
pre_s, post_s = chat.split(PH)
PRE = tok(pre_s, add_special_tokens=False)["input_ids"]
P = len(PRE)


def prompt_ids(question):
    return PRE + [pad_id] * 64 + tok(post_s.replace("@@Q@@", question), add_special_tokens=False)["input_ids"]


class Projector(nn.Module):
    """Per square: (token + learned square embedding) -> MLP -> LayerNorm * scale (init: LLM embedding RMS)."""
    def __init__(self, d_in, d_hid, d_out, rms):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(64, d_in))
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))
        self.norm = nn.LayerNorm(d_out, elementwise_affine=False)
        self.scale = nn.Parameter(torch.tensor(float(rms)))

    def forward(self, x):
        return self.norm(self.net(x + self.pos)) * self.scale


rms = emb.weight.float().pow(2).mean(1).sqrt().mean().item()
proj = Projector(256, a.hidden, emb.weight.shape[1], rms).to(dev)
groups = [dict(params=list(proj.parameters()), lr=a.lr)] + ([dict(params=lora_params, lr=a.lora_lr)] if lora_params else [])
opt = torch.optim.AdamW(groups, weight_decay=a.wd)
steps_per_epoch = max(1, math.ceil(len(tr) / a.batch))                          # eval-only sets have no train split
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=a.epochs * steps_per_epoch, pct_start=0.05)
start, s0, rng_state = 0, 0, None
if (out / "last.pt").exists() and not a.eval_only:
    ck = torch.load(out / "last.pt", map_location=dev)
    assert ck.get("lora") == lcfg, f"last.pt has LoRA config {ck.get('lora')}, flags say {lcfg}"
    proj.load_state_dict(ck["proj"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    if lcfg: set_peft_model_state_dict(llm, ck["lora_sd"])
    mu, sd = ck["mu"].cpu(), ck["sd"].cpu(); acts = (raw - mu) / sd
    start, s0, rng_state = ck["epoch"], ck.get("step", steps_per_epoch), ck.get("rng")  # old checkpoints: epoch end
    if "trng" in ck: torch.set_rng_state(ck["trng"].cpu())
    print(f"resumed from epoch {start} step {s0}/{steps_per_epoch}", flush=True)
elif (a.init_ckpt or a.init_proj) and not a.eval_only:
    ck = torch.load(a.init_ckpt or a.init_proj, map_location=dev); proj.load_state_dict(ck["proj"])
    mu, sd = ck["mu"].cpu(), ck["sd"].cpu(); acts = (raw - mu) / sd
    if a.init_ckpt:
        assert ck.get("lora") == lcfg, f"{a.init_ckpt} has LoRA config {ck.get('lora')}, flags say {lcfg}"
        if lcfg: set_peft_model_state_dict(llm, ck["lora_sd"])
    print(f"{'projector + LoRA' if a.init_ckpt else 'projector'} from {a.init_ckpt or a.init_proj}", flush=True)


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


def batch(idx, rng):
    ids, labels = [], []
    for i in idx:
        _, _, _, qt, at = qa.sample(rng, wdl[i], black[i])
        p = prompt_ids(qt); y = tok(at + END, add_special_tokens=False)["input_ids"]
        ids.append(p + y); labels.append([-100] * len(p) + y)
    x, mask = embed(ids, proj(acts[idx].to(dev)))
    lab = torch.full(mask.shape, -100, dtype=torch.long)
    for k, l in enumerate(labels): lab[k, :len(l)] = torch.tensor(l)
    # logits only where some row predicts an answer token: the 248k-wide output layer dominates memory
    # a slice, not a tensor: accelerate moves tensor arguments to GPU 0, but the hidden states end on the last GPU
    keep = slice(min(l.index(next(t for t in l if t != -100)) for l in labels) - 1, mask.shape[1] - 1)
    return x, mask, lab.to(dev), keep


def loss_fn(x, mask, lab, keep):
    logits = llm(inputs_embeds=x, attention_mask=mask, logits_to_keep=keep, use_cache=False).logits
    y = lab[:, keep.start + 1:keep.stop + 1].flatten().to(logits.device)                  # with 2 GPUs the logits sit on the last one
    return nn.functional.cross_entropy(logits.float().flatten(0, 1), y, ignore_index=-100)


@torch.no_grad()
def generate(rows, idx):
    """Greedy answers; rows of equal prompt length run together, so no padding and no KV cache needed."""
    was_training = llm.training
    proj.eval(); llm.eval()                         # train mode + grad ckpt makes HF drop the KV cache
    outs = [None] * len(rows)
    groups = {}
    for k, r in enumerate(rows): groups.setdefault(len(prompt_ids(r["question"])), []).append(k)
    for ks in groups.values():
        for s in range(0, len(ks), a.eval_batch):
            kk = ks[s:s + a.eval_batch]
            x, _ = embed([prompt_ids(rows[k]["question"]) for k in kk], proj(acts[[idx[k] for k in kk]].to(dev)))
            gen = torch.zeros((len(kk), 0), dtype=torch.long, device=dev)
            o = llm(inputs_embeds=x, use_cache=not a.no_cache, logits_to_keep=1)
            for _ in range(a.max_new):
                nxt = o.logits[:, -1].argmax(-1).to(dev)
                gen = torch.cat([gen, nxt[:, None]], 1)
                if (gen == end_id).any(1).all() or gen.shape[1] == a.max_new: break
                if a.no_cache:
                    x = torch.cat([x, emb(nxt)[:, None]], 1)
                    o = llm(inputs_embeds=x, use_cache=False, logits_to_keep=1)
                else:
                    o = llm(inputs_embeds=emb(nxt)[:, None], past_key_values=o.past_key_values, use_cache=True,
                            logits_to_keep=1)
            for k, g in zip(kk, gen.tolist()):
                outs[k] = tok.decode(g[:g.index(end_id)] if end_id in g else g, skip_special_tokens=True).strip()
    proj.train(); llm.train(was_training)
    return outs


def truth(r):
    w, d, l, _ = qa.view(wdl[r["i"]], black[r["i"]], r["q"], r["colour"])
    p = qa.pawns(w, l)
    return dict(w=w, d=d, l=l, pawns=p, bin=qa.bin_of(p))


def score(rows, preds):
    m = {k: [] for k in ("parsed", "exact", "format_ok", "subject_ok", "side", "side_q8", "bin", "bin_pm1",
                         "pawn_mae", "win_mae", "score_mae")}
    for r, a_ in zip(rows, preds):
        t, p = truth(r), qa.parse(a_)
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


mean_wdl = wdl[tr].mean(0) if len(tr) else wdl.mean(0)                              # eval-only set: its own mean
base_mean = lambda rows: score(rows, [qa.render(mean_wdl, black[r["i"]], r["q"], r["fmt"], r["colour"]) for r in rows])
print("baseline mean:", json.dumps(base_mean(val)), flush=True)
print(f"{a.model} {dt}, {len(tr):,} train positions, {steps_per_epoch} steps/epoch, eval on {len(val)} val", flush=True)

def reported(ep, step, report):
    if not report.exists(): return False
    done = set()
    for l in open(report):
        try: r = json.loads(l); done.add((r["epoch"], r.get("step", steps_per_epoch)))
        except (ValueError, KeyError): pass                                             # a line cut off by a kill
    return (ep, step) in done


def save(ck, path):
    torch.save(ck, path.with_suffix(".tmp")); os.replace(path.with_suffix(".tmp"), path)


def ckpt(ep, step, rng):
    return dict(proj=proj.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, step=step,
                rng=rng.bit_generator.state, trng=torch.get_rng_state(), mu=mu, sd=sd, model=a.model, P=P, args=vars(a),
                lora=lcfg, lora_sd=get_peft_model_state_dict(llm) if lcfg else None)


def evaluate(ep, step, t0, report, rows=None, name="val"):
    rows = rows or val; idx = [r["i"] for r in rows]
    t1 = time.time()
    preds = generate(rows, idx)
    shuf = generate(rows, list(np.random.default_rng(ep * 100003 + step).permutation(idx)))
    rep = dict(epoch=ep, step=step, data=name, train_min=round((t1 - t0) / 60, 1) if t0 else None,
               eval_min=round((time.time() - t1) / 60, 1), model=score(rows, preds), shuffled=score(rows, shuf), mean=base_mean(rows))
    with open(report, "a") as f: f.write(json.dumps(rep) + "\n")
    tag = "test" if name == "test" else f"epoch{ep}" + ("" if step == steps_per_epoch else f"_step{step}")
    with open(out / f"samples_{tag}.jsonl", "w") as f:
        for r, p_, s_ in zip(rows, preds, shuf):
            f.write(json.dumps(dict(question=r["question"], answer=r["answer"], pred=p_, shuffled=s_, fen=r["fen"])) + "\n")
    print(f"\n== {name} epoch {ep} step {step}/{steps_per_epoch} ==", flush=True)
    keys = ["parsed", "format_ok", "subject_ok", "exact", "side", "side_q8", "bin", "bin_pm1", "pawn_mae", "win_mae", "score_mae"]
    print(f"{'':12}" + "".join(f"{k:>11}" for k in keys))
    for name in ("model", "shuffled", "mean"):
        print(f"{name:12}" + "".join(f"{'-' if rep[name][k] is None else rep[name][k]:>11}" for k in keys))
    for r, p_ in list(zip(rows, preds))[:5]:
        print(f"  Q: {r['question']}\n     true: {r['answer']}\n     pred: {p_}")
    print(flush=True)


if a.eval_only:
    for c in a.eval_only:
        ck = torch.load(c, map_location=dev); proj.load_state_dict(ck["proj"])
        acts = (raw - ck["mu"].cpu()) / ck["sd"].cpu()                                # the checkpoint's train stats
        if ck.get("lora"): set_peft_model_state_dict(llm, ck["lora_sd"])
        with llm.disable_adapter() if lcfg and not ck.get("lora") else contextlib.nullcontext():
            evaluate(ck["epoch"], ck.get("step", steps_per_epoch), None, out / "report_rescored.jsonl")
    sys.exit()

for ep in range(start, a.epochs):
    rng = np.random.default_rng(a.seed * 1000 + ep)
    order = rng.permutation(tr)
    first = s0 if ep == start else 0
    if first and rng_state: rng.bit_generator.state = rng_state                       # question draws continue mid-epoch
    t0, run, n = time.time(), 0.0, 0
    for s in range(first, steps_per_epoch):
        if s == first and s and a.eval_every and s % a.eval_every == 0 and not reported(ep, s, out / "report.jsonl"):
            evaluate(ep, s, None, out / "report.jsonl")                                # killed during this eval
        loss = loss_fn(*batch(order[s * a.batch:(s + 1) * a.batch], rng))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step(); sched.step()
        run += loss.item(); n += 1
        if (s + 1) % a.log_every == 0 or s + 1 == steps_per_epoch:
            el = time.time() - t0
            print(f"ep {ep} step {s + 1}/{steps_per_epoch} loss {run / n:.4f} "
                  f"{el / (s + 1 - first):.2f}s/step eta {el / (s + 1 - first) * (steps_per_epoch - s - 1) / 60:.1f}min", flush=True)
            run, n = 0.0, 0
        if s + 1 < steps_per_epoch and a.eval_every and (s + 1) % a.eval_every == 0:
            save(ckpt(ep, s + 1, rng), out / "last.pt"); evaluate(ep, s + 1, t0, out / "report.jsonl")
        elif s + 1 < steps_per_epoch and a.save_every and (s + 1) % a.save_every == 0:
            save(ckpt(ep, s + 1, rng), out / "last.pt")
    if first < steps_per_epoch:
        ck = ckpt(ep, steps_per_epoch, rng)
        save(ck, out / f"epoch{ep}.pt"); save(ck, out / "last.pt")
    # a kill during eval resumes at this epoch's end and re-evaluates; the report check keeps it to one line
    if not reported(ep, steps_per_epoch, out / "report.jsonl"):
        evaluate(ep, steps_per_epoch, t0 if first < steps_per_epoch else None, out / "report.jsonl")
if test and not reported(a.epochs - 1, steps_per_epoch, out / "report_test.jsonl"):
    evaluate(a.epochs - 1, steps_per_epoch, None, out / "report_test.jsonl", test, "test")
