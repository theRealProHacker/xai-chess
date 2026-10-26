"""Stage 1 eval oracle: train a projector from Leela's layer-9 tokens into a frozen LLM's input embeddings.

Each sample is   <chat prefix> 64 soft tokens <question> -> <answer>,  loss on the answer only.
Train questions are resampled every epoch; val is the fixed oracle/data/val.jsonl.
After every epoch: checkpoint (epoch{e}.pt, last.pt), greedy answers on --eval-n val rows
(samples_epoch{e}.jsonl), and one line in report.jsonl. Baselines in the report: the training-mean WDL
rendered in the asked format ("mean"), and the same model fed another position's tokens ("shuffled").

  python oracle/train.py --out oracle/runs/q08b            # resumes from last.pt if it exists
"""
import argparse, json, math, sys, time
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
ap.add_argument("--eval-only", nargs="+", metavar="CKPT", help="re-score saved checkpoints (e.g. epoch*.pt) and exit")
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
mu, sd = acts[tr].mean((0, 1)), acts[tr].std((0, 1)) + 1e-6                          # per channel, train only
acts = (acts - mu) / sd
val = [json.loads(l) for l in open(Path(a.data) / "val.jsonl")][:a.eval_n]

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
opt = torch.optim.AdamW(proj.parameters(), lr=a.lr, weight_decay=a.wd)
steps_per_epoch = math.ceil(len(tr) / a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.05)
start = 0
if (out / "last.pt").exists():
    ck = torch.load(out / "last.pt", map_location=dev)
    proj.load_state_dict(ck["proj"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start = ck["epoch"] + 1
    print(f"resumed from epoch {ck['epoch']}", flush=True)


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


val_idx = [r["i"] for r in val]
mean_wdl = wdl[tr].mean(0)
base_mean = score(val, [qa.render(mean_wdl, black[r["i"]], r["q"], r["fmt"], r["colour"]) for r in val])
print("baseline mean:", json.dumps(base_mean), flush=True)
print(f"{a.model} {dt}, {len(tr):,} train positions, {steps_per_epoch} steps/epoch, eval on {len(val)} val", flush=True)

def evaluate(ep, t0, report):
    t1 = time.time()
    preds = generate(val, val_idx)
    shuf = generate(val, list(np.random.default_rng(ep).permutation(val_idx)))
    rep = dict(epoch=ep, train_min=round((t1 - t0) / 60, 1) if t0 else None, eval_min=round((time.time() - t1) / 60, 1),
               model=score(val, preds), shuffled=score(val, shuf), mean=base_mean)
    with open(report, "a") as f: f.write(json.dumps(rep) + "\n")
    with open(out / f"samples_epoch{ep}.jsonl", "w") as f:
        for r, p_, s_ in zip(val, preds, shuf):
            f.write(json.dumps(dict(question=r["question"], answer=r["answer"], pred=p_, shuffled=s_, fen=r["fen"])) + "\n")
    print(f"\n== epoch {ep} ==", flush=True)
    keys = ["parsed", "format_ok", "subject_ok", "exact", "side", "side_q8", "bin", "bin_pm1", "pawn_mae", "win_mae", "score_mae"]
    print(f"{'':12}" + "".join(f"{k:>11}" for k in keys))
    for name in ("model", "shuffled", "mean"):
        print(f"{name:12}" + "".join(f"{'-' if rep[name][k] is None else rep[name][k]:>11}" for k in keys))
    for r, p_ in list(zip(val, preds))[:5]:
        print(f"  Q: {r['question']}\n     true: {r['answer']}\n     pred: {p_}")
    print(flush=True)


if a.eval_only:
    for c in a.eval_only:
        ck = torch.load(c, map_location=dev); proj.load_state_dict(ck["proj"])
        evaluate(ck["epoch"], None, out / "report_rescored.jsonl")
    sys.exit()

for ep in range(start, a.epochs):
    rng = np.random.default_rng(a.seed * 1000 + ep)
    order = rng.permutation(tr)
    t0, run, n = time.time(), 0.0, 0
    for s in range(steps_per_epoch):
        loss = loss_fn(*batch(order[s * a.batch:(s + 1) * a.batch], rng))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item(); n += 1
        if (s + 1) % a.log_every == 0 or s + 1 == steps_per_epoch:
            el = time.time() - t0
            print(f"ep {ep} step {s + 1}/{steps_per_epoch} loss {run / n:.4f} "
                  f"{el / (s + 1):.2f}s/step eta {el / (s + 1) * (steps_per_epoch - s - 1) / 60:.1f}min", flush=True)
            run, n = 0.0, 0
    ck = dict(proj=proj.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep,
              mu=mu, sd=sd, model=a.model, P=P, args=vars(a))
    torch.save(ck, out / f"epoch{ep}.pt"); torch.save(ck, out / "last.pt")

    evaluate(ep, t0, out / "report.jsonl")
