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

AO-style options (Karvonen et al. 2025); the defaults are the setup above:
  --data D1 D2 .. --layers L1 L2 ..   one oracle over several Leela layers (same positions); each sample draws one,
                                      the prompt starts "Layer L:", each layer keeps its own mu/sd
  --placeholder pad|q|image|square|piece   the token each square's vector goes on: pad, " ?", <|image_pad|>,
                                      "e4 ?" (real-board square name, then " ?"), or "Ne4 ?" (FEN piece letter + square;
                                      empty squares "e4 ?"); square/piece list the squares in FEN order
  --inject embed|add --inject-layer N  embed: replace the input embedding (projector ends in LayerNorm * scale);
                                      add: h + |h| v/|v| on the residual after decoder layer N (no LayerNorm)
  --squares                           random square subsets: 1 square with prob --single-frac, else k ~ U{1..64};
                                      val is scored at each k in --eval-squares
  python oracle/train.py --data oracle/data_g50k_l2 oracle/data_g50k_l5 oracle/data_g50k --layers 2 5 9 \\
      --placeholder piece --inject add --squares --lora 16 --out oracle/runs/g50k_ao_piece

Supervised tasks (tasks.py, rows from build_tasks.py): --tasks T1 .. Tn trains on Tn and, with prob --replay,
draws a sample from T1 .. Tn-1 instead; "eval" is the WDL question above. Val/test score every listed task.
oracle/curriculum.sh chains the stages. --sched cosine: cosine decay from --lr to 0, no warmup.
"""
import argparse, contextlib, json, math, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "oracle"))
import numpy as np, torch, torch.nn as nn
import qa

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", default=[str(ROOT / "oracle/data")])
ap.add_argument("--layers", nargs="+", type=int, help="Leela layer of each --data dir; adds the \"Layer L:\" prompt tag")
ap.add_argument("--placeholder", choices=["pad", "q", "image", "square", "piece"], default="pad")
ap.add_argument("--inject", choices=["embed", "add"], default="embed")
ap.add_argument("--inject-layer", type=int, default=1, help="--inject add: decoder layer whose output gets the vectors")
ap.add_argument("--stm", action=argparse.BooleanOptionalAction, default=True, help="state the side to move in the prompt (\"White to move.\"); --no-stm = v1/v2 prompt")
ap.add_argument("--squares", action="store_true", help="random square subsets instead of all 64")
ap.add_argument("--single-frac", type=float, default=1 / 3)
ap.add_argument("--eval-squares", default="64,8,1", help="--squares: subset sizes scored on val")
ap.add_argument("--no-square-emb", action="store_true", help="no learned square embedding in the projector")
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
ap.add_argument("--tasks", nargs="+", help="tasks.py tasks or eval: train on the last, replay the earlier ones")
ap.add_argument("--sf", default=str(ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2"), help="Stockfish for multipv scoring")
ap.add_argument("--replay", type=float, default=0.2, help="--tasks: share of samples from the earlier tasks")
ap.add_argument("--sched", choices=["onecycle", "cosine"], default="onecycle",
                help="onecycle: 5%% warmup, cosine anneal, cycled momentum; cosine: cosine decay to 0")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed)

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- data ----
assert a.layers is None or len(a.layers) == len(a.data), "--layers needs one layer per --data dir"
assert len(a.data) == 1 or a.layers, "several --data dirs need --layers"
D = np.load(Path(a.data[0]) / "labels.npz")
wdl, black, split, fens = D["wdl"], D["black"], D["split"], D["fens"]
for d in a.data[1:]:                                                                   # same positions, split and labels
    E = np.load(Path(d) / "labels.npz")
    assert (E["fens"] == fens).all() and (E["split"] == split).all() and np.allclose(E["wdl"], wdl), f"{d} differs from {a.data[0]}"
raws = [np.load(Path(d) / "acts.npy", mmap_mode="r") for d in a.data]                 # each (N, 64, 256) float16, on disk
tr = np.flatnonzero(split == 0)
if a.train_n: tr = tr[:a.train_n]


def norm_stats(x, rows, chunk=1024):
    """Per-channel mean and (unbiased) std over x[rows], read in chunks: no full float copy in RAM."""
    s = q = 0; n = 0
    for k in range(0, len(rows), chunk):
        c = torch.from_numpy(np.asarray(x[rows[k:k + chunk]], dtype=np.float64)).flatten(0, 1)
        s, q, n = s + c.sum(0), q + c.pow(2).sum(0), n + len(c)
    m = s / n
    return m.float(), ((q - n * m ** 2) / (n - 1)).sqrt().float()


def set_norm(mu_, sd_):
    """mu/sd: one (256,) pair, or (n_layers, 256) stacked; soft() normalises each layer's rows with them."""
    global mu, sd, norms
    mu, sd = mu_, sd_
    norms = list(zip([mu_] if mu_.dim() == 1 else list(mu_), [sd_] if sd_.dim() == 1 else list(sd_)))


norms = [(torch.zeros(256), torch.ones(256))] * len(raws)
if len(tr):
    st = [norm_stats(r, tr) for r in raws]; ms = [m for m, _ in st]; ss = [s + 1e-6 for _, s in st]  # per channel, train only
    set_norm(ms[0], ss[0]) if len(raws) == 1 else set_norm(torch.stack(ms), torch.stack(ss))
val = [json.loads(l) for l in open(Path(a.data[0]) / "val.jsonl")][:a.eval_n]
test = [json.loads(l) for l in open(Path(a.data[0]) / "test.jsonl")][:a.test_n] if a.test_n else []
if a.tasks:
    import tasks, chess.engine                                                         # needs python-chess
    assert set(a.tasks) <= {"eval", *tasks.TASKS}, f"--tasks: pick from eval {' '.join(tasks.TASKS)}"
    sf = {}                                                                            # best/pv labels; may be partial
    if (Path(a.data[0]) / "sf.jsonl").exists():
        for l in open(Path(a.data[0]) / "sf.jsonl"):
            if l.endswith("\n"): r = json.loads(l); sf[r["i"]] = r["lines"]
    POOL = {t: tr[[int(i) in sf for i in tr]] if t in tasks.SF_TASKS else tr for t in a.tasks}
    tr = POOL[a.tasks[-1]]

    def task_rows(name, n):
        """{task: first n rows} from tasks_{name}.jsonl, eval from {name}.jsonl."""
        rows = [json.loads(l) for l in open(Path(a.data[0]) / f"tasks_{name}.jsonl")]
        if "eval" in a.tasks: rows += [dict(json.loads(l), task="eval") for l in open(Path(a.data[0]) / f"{name}.jsonl")]
        return {t: [r for r in rows if r["task"] == t][:n] for t in dict.fromkeys(a.tasks)}
    val = task_rows("val", a.eval_n)
    test = task_rows("test", a.test_n) if a.test_n else {}

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
LAYERS = a.layers or [None]
PRES, post_s = [], None
for L in LAYERS:
    tag = f"Layer {L}:\n" if L is not None else ""
    chat = tok.apply_chat_template([{"role": "user", "content": f"{tag}Position: <board>{PH}</board>\n" + "@@Q@@"}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pre_s, post_s = chat.split(PH)
    PRES.append(tok(pre_s, add_special_tokens=False)["input_ids"])
PRE = PRES[0]
P = len(PRE)
ALL = list(range(64))
Q_ID = tok(" ?", add_special_tokens=False)["input_ids"]
assert len(Q_ID) == 1
SLOT = {"pad": pad_id, "q": Q_ID[0], "image": tok.convert_tokens_to_ids("<|image_pad|>")}.get(a.placeholder, Q_ID[0])


def real_square(k, b):
    """Net token k (side to move at the bottom) -> (file, rank) on the real board; Leela flips ranks for Black."""
    f, r = k % 8, k // 8
    return f, (7 - r if b else r)


if a.placeholder in ("square", "piece"):
    import chess
    NAME = {}                                                                          # (" e4 ?" or " Ne4 ?") ids per text
    def square_text(i, k):
        f, r = real_square(k, black[i])
        s = "abcdefgh"[f] + str(r + 1)
        if a.placeholder == "piece":
            pc = chess.Board(str(fens[i])).piece_at(chess.square(f, r))
            if pc: s = pc.symbol() + s
        return f" {s} ?"


def prompt_order(i, sq):
    """Squares in prompt order: net order, or FEN order (a8..h8, ..., a1..h1) for named squares."""
    if a.placeholder not in ("square", "piece"): return sorted(sq)
    return sorted(sq, key=lambda k: (-real_square(k, black[i])[1], real_square(k, black[i])[0]))


def prompt_ids(question, i=None, j=0, sq=None):
    """Token ids and the positions that carry square vectors, for position i, layer j, squares sq (None = all)."""
    sq = ALL if sq is None else sq
    if a.stm and i is not None: question = f"{'Black' if black[i] else 'White'} to move.\n{question}"
    ids = list(PRES[j]); pos = []
    for k in (prompt_order(i, sq) if i is not None else sq):
        if a.placeholder in ("square", "piece"):
            t = square_text(i, k)
            if t not in NAME: NAME[t] = tok(t, add_special_tokens=False)["input_ids"]
            ids += NAME[t]
        else:
            ids.append(SLOT)
        pos.append(len(ids) - 1)
    return ids + tok(post_s.replace("@@Q@@", question), add_special_tokens=False)["input_ids"], pos


def sq_list(i, sq=None):
    """The squares in the order prompt_ids places their vectors."""
    return prompt_order(i, ALL if sq is None else sq)


class Projector(nn.Module):
    """Per square: (token + learned square embedding) -> MLP -> LayerNorm * scale (init: LLM embedding RMS).
    --inject add drops LayerNorm * scale (the injection matches norms); --no-square-emb drops the square embedding."""
    def __init__(self, d_in, d_hid, d_out, rms, norm=True, square_emb=True):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(64, d_in)) if square_emb else None
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))
        if norm:
            self.norm = nn.LayerNorm(d_out, elementwise_affine=False)
            self.scale = nn.Parameter(torch.tensor(float(rms)))
        self.use_norm = norm

    def forward(self, x, sq=None):
        """x: (..., 64, d_in) with sq None, or (n, d_in) with sq the n square indices."""
        if self.pos is not None: x = x + (self.pos if sq is None else self.pos[sq])
        y = self.net(x)
        return self.norm(y) * self.scale if self.use_norm else y


rms = emb.weight.float().pow(2).mean(1).sqrt().mean().item()
proj = Projector(256, a.hidden, emb.weight.shape[1], rms, a.inject == "embed", not a.no_square_emb).to(dev)
INJ = None                                                                             # (rows, positions, vectors) for --inject add
if a.inject == "add":
    blocks = [m for n, m in llm.named_modules() if n.endswith(".layers") and isinstance(m, nn.ModuleList)
              and "visual" not in n and "vision" not in n]
    assert len(blocks) == 1, [type(b) for b in blocks]

    def inject_hook(mod, inp, out):
        """h' = h + |h| v / |v| at the square positions; skipped on cached one-token decode steps."""
        if INJ is None: return None
        h = out[0] if isinstance(out, tuple) else out
        b, p, v = (t.to(h.device) for t in INJ)
        if h.shape[1] <= int(p.max()): return None
        hv = h[b, p]
        h = h.index_put((b, p), hv + hv.norm(dim=-1, keepdim=True) * (v / v.norm(dim=-1, keepdim=True)).to(h.dtype))
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    blocks[0][a.inject_layer].register_forward_hook(inject_hook)
groups = [dict(params=list(proj.parameters()), lr=a.lr)] + ([dict(params=lora_params, lr=a.lora_lr)] if lora_params else [])
opt = torch.optim.AdamW(groups, weight_decay=a.wd)
steps_per_epoch = max(1, math.ceil(len(tr) / a.batch))                          # eval-only sets have no train split
if a.sched == "cosine":
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * steps_per_epoch)
else:
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=a.epochs * steps_per_epoch, pct_start=0.05)
start, s0, rng_state = 0, 0, None
if (out / "last.pt").exists() and not a.eval_only:
    ck = torch.load(out / "last.pt", map_location=dev)
    assert ck.get("lora") == lcfg, f"last.pt has LoRA config {ck.get('lora')}, flags say {lcfg}"
    assert ck["args"].get("sched", "onecycle") == a.sched, f"last.pt has --sched {ck['args'].get('sched', 'onecycle')}"
    proj.load_state_dict(ck["proj"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    if lcfg: set_peft_model_state_dict(llm, ck["lora_sd"])
    set_norm(ck["mu"].cpu(), ck["sd"].cpu())
    start, s0, rng_state = ck["epoch"], ck.get("step", steps_per_epoch), ck.get("rng")  # old checkpoints: epoch end
    if "trng" in ck: torch.set_rng_state(ck["trng"].cpu())
    print(f"resumed from epoch {start} step {s0}/{steps_per_epoch}", flush=True)
elif (a.init_ckpt or a.init_proj) and not a.eval_only:
    ck = torch.load(a.init_ckpt or a.init_proj, map_location=dev); proj.load_state_dict(ck["proj"])
    set_norm(ck["mu"].cpu(), ck["sd"].cpu())
    if a.init_ckpt:
        assert ck.get("lora") == lcfg, f"{a.init_ckpt} has LoRA config {ck.get('lora')}, flags say {lcfg}"
        if lcfg: set_peft_model_state_dict(llm, ck["lora_sd"])
    print(f"{'projector + LoRA' if a.init_ckpt else 'projector'} from {a.init_ckpt or a.init_proj}", flush=True)


def soft(i, j, sq):
    """Projected vectors of position i's layer-j tokens at squares sq (in prompt order): (len(sq), d)."""
    t = torch.as_tensor(sq); m, s = norms[j]
    x = torch.from_numpy(np.asarray(raws[j][i][np.asarray(sq)], dtype=np.float32))
    return proj(((x - m) / s).to(dev), t.to(dev))


def embed(ids_list, softs, poss):
    """Right-padded (B, L, d) inputs and the attention mask. --inject embed: the vectors replace the input
    embeddings at their positions; add: inputs stay tokens and INJ holds the vectors for the hook."""
    global INJ
    L = max(map(len, ids_list))
    ids = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_list), L), dtype=torch.long)
    for k, s in enumerate(ids_list):
        ids[k, :len(s)] = torch.tensor(s); mask[k, :len(s)] = 1
    x = emb(ids.to(dev)).clone()
    b = torch.cat([torch.full((len(p),), k) for k, p in enumerate(poss)])
    p = torch.cat([torch.tensor(p) for p in poss])
    v = torch.cat(softs)
    if a.inject == "embed":
        x[b.to(dev), p.to(dev)] = v.to(x.dtype)
    else:
        INJ = (b, p, v)
    return x, mask.to(dev)


def draw(rng):
    """Layer and squares for one training sample; draws nothing with the default flags."""
    j = int(rng.integers(len(a.data))) if len(a.data) > 1 else 0
    if not a.squares: return j, None
    k = 1 if rng.random() < a.single_frac else int(rng.integers(1, 65))
    return j, sorted(rng.choice(64, k, replace=False).tolist())


def sample_row(i, rng):
    """(position, question, answer) of one training sample; a best/pv sample moves to a position with labels."""
    if not a.tasks: return i, *qa.sample(rng, wdl[i], black[i])[3:]
    prev = a.tasks[:-1]
    t = prev[int(rng.integers(len(prev)))] if prev and rng.random() < a.replay else a.tasks[-1]
    if t == "eval": return i, *qa.sample(rng, wdl[i], black[i])[3:]
    if t in tasks.SF_TASKS and int(i) not in sf: i = int(rng.choice(POOL[t]))
    r = tasks.sample(t, rng, str(fens[i]), sf.get(int(i)))
    return i, r["question"], r["answer"]


def batch(idx, rng):
    ids, labels, softs, poss = [], [], [], []
    for i in idx:
        i, qt, at = sample_row(i, rng)
        j, sq = draw(rng)
        p, pos = prompt_ids(qt, i, j, sq); y = tok(at + END, add_special_tokens=False)["input_ids"]
        ids.append(p + y); labels.append([-100] * len(p) + y); poss.append(pos)
        softs.append(soft(i, j, sq_list(i, sq)))
    x, mask = embed(ids, softs, poss)
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
def generate(rows, idx, specs=None, max_new=None):
    """Greedy answers; rows of equal prompt length run together, so no padding and no KV cache needed.
    Row k's prompt is its own position's; the vectors come from position idx[k], layer and squares from specs[k]."""
    global INJ
    was_training = llm.training
    proj.eval(); llm.eval()                         # train mode + grad ckpt makes HF drop the KV cache
    specs = specs or [(0, None)] * len(rows)
    max_new = max_new or a.max_new
    prompts = [prompt_ids(r["question"], r["i"], *sp) for r, sp in zip(rows, specs)]
    outs = [None] * len(rows)
    groups = {}
    for k, (p, _) in enumerate(prompts): groups.setdefault(len(p), []).append(k)
    for ks in groups.values():
        for s in range(0, len(ks), a.eval_batch):
            kk = ks[s:s + a.eval_batch]
            x, _ = embed([prompts[k][0] for k in kk],
                         [soft(idx[k], specs[k][0], sq_list(rows[k]["i"], specs[k][1])) for k in kk], [prompts[k][1] for k in kk])
            gen = torch.zeros((len(kk), 0), dtype=torch.long, device=dev)
            o = llm(inputs_embeds=x, use_cache=not a.no_cache, logits_to_keep=1)
            for _ in range(max_new):
                nxt = o.logits[:, -1].argmax(-1).to(dev)
                gen = torch.cat([gen, nxt[:, None]], 1)
                if (gen == end_id).any(1).all() or gen.shape[1] == max_new: break
                if a.no_cache:
                    x = torch.cat([x, emb(nxt)[:, None]], 1)
                    o = llm(inputs_embeds=x, use_cache=False, logits_to_keep=1)
                else:
                    o = llm(inputs_embeds=emb(nxt)[:, None], past_key_values=o.past_key_values, use_cache=True,
                            logits_to_keep=1)
            for k, g in zip(kk, gen.tolist()):
                outs[k] = tok.decode(g[:g.index(end_id)] if end_id in g else g, skip_special_tokens=True).strip()
    INJ = None
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
if not a.tasks: print("baseline mean:", json.dumps(base_mean(val)), flush=True)
print(f"{a.model} {dt}, {len(tr):,} train positions, {steps_per_epoch} steps/epoch, eval on {sum(map(len, val.values())) if a.tasks else len(val)} val", flush=True)

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


EVAL_K = [int(k) for k in a.eval_squares.split(",")] if a.squares else [64]
COMBOS = [(j, k) for j in range(len(a.data)) for k in EVAL_K]                         # the first one fills model/shuffled


def eval_squares(i, k):
    """Fixed k-square subset of position i (all 64 at k = 64)."""
    return None if k >= 64 else sorted(np.random.default_rng(i * 65 + k).choice(64, k, replace=False).tolist())


def combo_name(j, k):
    return f"L{LAYERS[j]}" * (LAYERS[j] is not None) + (f"/k{k}" if a.squares else "")


def task_score(rows, preds):
    """Means of tasks.score's metrics (WDL score() for eval); exact counts an unparsed answer as wrong."""
    if rows[0]["task"] == "eval": return score(rows, preds)
    m = {}
    # multipv scores evals with Stockfish, open only here: python-chess's engine thread keeps Python from exiting
    with chess.engine.SimpleEngine.popen_uci(a.sf) if rows[0]["task"] == "multipv" else contextlib.nullcontext() as eng:
        if eng: eng.configure({"Threads": 1, "Hash": 16})
        for r, p_ in zip(rows, preds):
            s = tasks.score(r, p_, eng)
            if not s["parsed"] and r["task"] not in ("pv", "multipv"): s["exact"] = False
            for k, v in s.items():
                if k != "task": m.setdefault(k, []).append(float(v))
    return {k: round(float(np.mean(v)), 4) for k, v in m.items()}


def evaluate_tasks(ep, step, t0, report, sets=None, name="val"):
    """--tasks: every task's rows at every combo, model and shuffled; one report line, one samples file."""
    sets = sets or val
    t1 = time.time()
    by, samples = {}, []
    for t, rows in sets.items():
        if not rows: continue
        idx = [r["i"] for r in rows]
        perm = list(np.random.default_rng(ep * 100003 + step).permutation(idx))
        mn = max(len(tok(r["answer"] + END, add_special_tokens=False)["input_ids"]) for r in rows) + 8
        for j, k in COMBOS:
            specs = [(j, eval_squares(i, k)) for i in idx]
            pr, sh = generate(rows, idx, specs, mn), generate(rows, perm, specs, mn)
            key = t + (f" {combo_name(j, k)}" if len(COMBOS) > 1 else "")
            by[key] = dict(model=task_score(rows, pr), shuffled=task_score(rows, sh))
            samples += [dict(task=t, combo=combo_name(j, k), question=r["question"], answer=r["answer"], pred=p_,
                             shuffled=s_, fen=r["fen"]) for r, p_, s_ in zip(rows, pr, sh)]
    rep = dict(epoch=ep, step=step, data=name, tasks=a.tasks, train_min=round((t1 - t0) / 60, 1) if t0 else None,
               eval_min=round((time.time() - t1) / 60, 1), by=by)
    with open(report, "a") as f: f.write(json.dumps(rep) + "\n")
    tag = "test" if name == "test" else f"epoch{ep}" + ("" if step == steps_per_epoch else f"_step{step}")
    with open(out / f"samples_{tag}.jsonl", "w") as f:
        for r in samples: f.write(json.dumps(r) + "\n")
    print(f"\n== {name} epoch {ep} step {step}/{steps_per_epoch} ==", flush=True)
    for key, v in by.items():
        for t_ in ("model", "shuffled"):
            print(f"{key:24}{t_:10}" + "  ".join(f"{k} {x}" for k, x in v[t_].items() if x is not None))
    seen = set()
    for r in samples:
        if r["task"] in seen: continue
        seen.add(r["task"]); print(f"  Q: {r['question']}\n     true: {r['answer']}\n     pred: {r['pred']}")
    print(flush=True)


def evaluate(ep, step, t0, report, rows=None, name="val"):
    if a.tasks: return evaluate_tasks(ep, step, t0, report, rows, name)
    rows = rows or val; idx = [r["i"] for r in rows]
    t1 = time.time()
    perm = list(np.random.default_rng(ep * 100003 + step).permutation(idx))
    res = {}
    for j, k in COMBOS:
        specs = [(j, eval_squares(i, k)) for i in idx]
        res[j, k] = generate(rows, idx, specs), generate(rows, perm, specs)
    preds, shuf = res[COMBOS[0]]
    rep = dict(epoch=ep, step=step, data=name, train_min=round((t1 - t0) / 60, 1) if t0 else None,
               eval_min=round((time.time() - t1) / 60, 1), model=score(rows, preds), shuffled=score(rows, shuf), mean=base_mean(rows))
    if len(COMBOS) > 1:
        rep["by"] = {combo_name(j, k): dict(model=score(rows, pr), shuffled=score(rows, sh)) for (j, k), (pr, sh) in res.items()}
    with open(report, "a") as f: f.write(json.dumps(rep) + "\n")
    tag = "test" if name == "test" else f"epoch{ep}" + ("" if step == steps_per_epoch else f"_step{step}")
    with open(out / f"samples_{tag}.jsonl", "w") as f:
        for (j, k), (pr, sh) in res.items():
            extra = dict(combo=combo_name(j, k)) if len(COMBOS) > 1 else {}
            for r, p_, s_ in zip(rows, pr, sh):
                f.write(json.dumps(dict(question=r["question"], answer=r["answer"], pred=p_, shuffled=s_, fen=r["fen"], **extra)) + "\n")
    print(f"\n== {name} epoch {ep} step {step}/{steps_per_epoch} ==", flush=True)
    keys = ["parsed", "format_ok", "subject_ok", "exact", "side", "side_q8", "bin", "bin_pm1", "pawn_mae", "win_mae", "score_mae"]
    print(f"{'':16}" + "".join(f"{k:>11}" for k in keys))
    lines = [("model", rep["model"]), ("shuffled", rep["shuffled"]), ("mean", rep["mean"])]
    if len(COMBOS) > 1:
        lines = [(f"{t} {c}", v[t]) for c, v in rep["by"].items() for t in ("model", "shuffled")] + [("mean", rep["mean"])]
    for n_, v in lines:
        print(f"{n_:16}" + "".join(f"{'-' if v[k] is None else v[k]:>11}" for k in keys))
    for r, p_ in list(zip(rows, preds))[:5]:
        print(f"  Q: {r['question']}\n     true: {r['answer']}\n     pred: {p_}")
    print(flush=True)


if a.eval_only:
    for c in a.eval_only:
        ck = torch.load(c, map_location=dev); proj.load_state_dict(ck["proj"])
        set_norm(ck["mu"].cpu(), ck["sd"].cpu())                                    # the checkpoint's train stats
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
