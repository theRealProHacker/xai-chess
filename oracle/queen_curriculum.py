"""Queen's domain-adaptation curriculum (Bhaskar, Cheng, Chen 2026, arXiv:2610.03695, Sec. 3.2 and App. A) on the oracle:
Leela layer-9 tokens -> fresh projector -> Qwen3.5-0.8B with fresh LoRA r16, as in train.py --inject embed.

Stages (Table 8 replay mix, Table 9 steps scaled 50k/60k/30k/100k -> 5k/6k/3k/10k at batch 16, LRs as given):
  1 static-current   2 dynamic-current   3 static-future   4 dynamic-future
Question types: the 13 of Table 7, three paired (question, answer) formats each; "future" asks about the position
after 1-8 random legal moves, given as square tokens. 64 square and 12 piece tokens are new; their embeddings
(tied in and out, as Qwen ties them) train at 0.1x the stage LR. Pieces are named from the side to move in the
position asked about: <PIECE_MN> my knight, <PIECE_OK> opponent's king. Squares: <SQUARE_1> = a1 .. <SQUARE_64> = h8.
Per stage: AdamW, 5% warmup, cosine to 5% of peak; val every --eval-every steps on the stage's own questions,
early stopping after --patience evals without gain, best.pt starts the next stage. After each stage the test set of
every stage so far is scored (exact match of the whole answer), as Table 10.
Each stage dir resumes from last.pt (scavenger requeue); a stage with report_test.jsonl is done.

  python oracle/queen_curriculum.py --data oracle/data_g50k --out oracle/runs/queen
"""
import argparse, json, math, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import numpy as np, torch, torch.nn as nn
import chess

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "oracle/data_g50k"))
ap.add_argument("--out", default=str(ROOT / "oracle/runs/queen"))
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--steps", default="5000,6000,3000,10000", help="max training steps per stage")
ap.add_argument("--lrs", default="1e-4,8e-5,7.5e-5,2e-4", help="peak LR of projector and LoRA per stage")
ap.add_argument("--token-lr", type=float, default=0.1, help="new token embeddings: this times the stage LR")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--lora", type=int, default=16)
ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj")
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--patience", type=int, default=3, help="evals without a new best before the stage stops")
ap.add_argument("--val-n", type=int, default=35, help="val questions per question type")
ap.add_argument("--test-n", type=int, default=150, help="test questions per question type")
ap.add_argument("--eval-batch", type=int, default=32)
ap.add_argument("--train-n", type=int, default=0, help="only the first n train positions (smoke tests)")
ap.add_argument("--stages", default="1,2,3,4")
ap.add_argument("--log-every", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed)

# ---- notation ----
SQ = [f"<SQUARE_{s + 1}>" for s in range(64)]
PIECES = [f"<PIECE_{w}{p}>" for w in "MO" for p in "PNBRQK"]
NEW = SQ + PIECES
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def pc(b, piece):
    """Piece token, mine (M) or the opponent's (O) from b's side to move."""
    return f"<PIECE_{'M' if piece.color == b.turn else 'O'}{piece.symbol().upper()}>"


def at(b, s):
    return f"{pc(b, b.piece_at(s))} on {SQ[s]}"


def mv(m):
    return SQ[m.from_square] + SQ[m.to_square] + (PIECES[m.promotion - 1] if m.promotion else "")


def lst(xs, none):
    return ", ".join(xs) if xs else none


# ---- question types (Table 7); each returns (question, answer) in format f, or None if it does not apply ----
def piece_on(b, rng, f):
    s = int(rng.integers(64)); p = b.piece_at(s)
    q = [f"What piece is on {SQ[s]}?", f"What stands on {SQ[s]}?", f"Piece on {SQ[s]}?"][f]
    if p is None: return q, [f"There is no piece on {SQ[s]}.", f"{SQ[s]} is empty.", "empty"][f]
    return q, [f"There is {pc(b, p)} on {SQ[s]}.", f"{pc(b, p)} stands on {SQ[s]}.", pc(b, p)][f]


def locate(b, rng, f):
    present = sorted({(p.color, p.piece_type) for p in b.piece_map().values()})
    col, kind = present[int(rng.integers(len(present)))] if rng.random() < 0.9 else \
        (bool(rng.integers(2)), int(rng.integers(1, 7)))
    t = pc(b, chess.Piece(kind, col)); sqs = [SQ[s] for s in sorted(b.pieces(kind, col))]
    q = [f"Where is {t}?", f"Which squares hold {t}?", f"Locate {t}."][f]
    if not sqs: return q, [f"{t} is not on the board.", f"No square holds {t}.", "none"][f]
    return q, [f"{t} occupies {' and '.join(sqs)}.", f"{t} is on {', '.join(sqs)}.", " ".join(sqs)][f]


def counted(b, sqs):
    """'1 <PIECE_MP> and 2 <PIECE_OR>' over the pieces on sqs, mine then the opponent's, P N B R Q K."""
    n = {}
    for s in sqs:
        if (p := b.piece_at(s)): n[pc(b, p)] = n.get(pc(b, p), 0) + 1
    return " and ".join(f"{n[t]} {t}" for t in PIECES if t in n)


def walked(b, sqs):
    return ", ".join(f"{pc(b, b.piece_at(s))} {SQ[s]}" for s in sqs if b.piece_at(s))


def on_file(b, rng, f):
    fi = int(rng.integers(8)); sqs = [chess.square(fi, r) for r in range(8)]
    q = [f"Pieces on file {SQ[sqs[0]]}--{SQ[sqs[-1]]}?", f"What is on the file from {SQ[sqs[0]]} to {SQ[sqs[-1]]}?",
         f"Count the pieces on file {SQ[sqs[0]]}--{SQ[sqs[-1]]}."][f]
    c = counted(b, sqs)
    return q, [f"{c}." if c else "The file is empty.", f"On the file: {c}." if c else "Nothing is on the file.", c or "empty"][f]


def on_rank(b, rng, f):
    r = int(rng.integers(8)); sqs = [chess.square(fi, r) for fi in range(8)]
    q = [f"Pieces on rank {SQ[sqs[0]]}--{SQ[sqs[-1]]}?", f"Walk the rank from {SQ[sqs[0]]} to {SQ[sqs[-1]]}.",
         f"What is on rank {SQ[sqs[0]]}--{SQ[sqs[-1]]}?"][f]
    w = walked(b, sqs)
    return q, [f"Walking the rank: {w}." if w else "Walking the rank: empty.", f"{w}." if w else "The rank is empty.", w or "empty"][f]


DIAGS = []                                      # every diagonal and anti-diagonal of length >= 3, a-file end first
for s0 in range(64):
    for df, dr in ((1, 1), (1, -1)):
        f0, r0 = chess.square_file(s0), chess.square_rank(s0)
        if 0 <= f0 - df < 8 and 0 <= r0 - dr < 8: continue                              # not the line's start
        line = [chess.square(f0 + k * df, r0 + k * dr) for k in range(8) if 0 <= f0 + k * df < 8 and 0 <= r0 + k * dr < 8]
        if len(line) >= 3: DIAGS.append(line)


def on_diag(b, rng, f):
    sqs = DIAGS[int(rng.integers(len(DIAGS)))]
    q = [f"Pieces on diagonal {SQ[sqs[0]]}--{SQ[sqs[-1]]}?", f"What is on the diagonal from {SQ[sqs[0]]} to {SQ[sqs[-1]]}?",
         f"Walk the diagonal {SQ[sqs[0]]}--{SQ[sqs[-1]]}."][f]
    w = walked(b, sqs)
    return q, [f"{w}." if w else "open diagonal", f"On the diagonal: {w}." if w else "The diagonal is open.", w or "open"][f]


def counts(b, rng, f):
    side = lambda col: "".join(f"{pc(b, chess.Piece(k, col))}{len(b.pieces(k, col))}" for k in range(1, 7) if b.pieces(k, col))
    me, op = side(b.turn), side(not b.turn)
    q = ["Piece counts of both sides?", "How many pieces of each type does each side have?", "Count the pieces."][f]
    return q, [f"player: {me}; opponent: {op}.", f"Player has {me}, opponent has {op}.", f"{me} {op}"][f]


def material(b, rng, f):
    pts = lambda col: sum(VALUE[p.piece_type] for p in b.piece_map().values() if p.color == col)
    me, op = pts(b.turn), pts(not b.turn)
    q = ["Material points per side?", "How much material does each side have?", "Count the material."][f]
    return q, [f"player {me}, opponent {op}.", f"Player has {me} points, opponent has {op}.", f"{me} {op}"][f]


def legal_of(b, rng, f):
    own = sorted(chess.SquareSet(b.occupied_co[b.turn]))
    s = own[int(rng.integers(len(own)))]
    to = sorted({m.to_square for m in b.legal_moves if m.from_square == s})
    q = [f"Legal moves for {at(b, s)}?", f"Where can {at(b, s)} move?", f"List the legal moves of {at(b, s)}."][f]
    t = ", ".join(SQ[x] for x in to)
    return q, [f"{t}." if to else "none.", f"It can move to {t}." if to else "It cannot move.", t or "none"][f]


def attack_defend(b, rng, f):
    sqs = sorted(b.piece_map()); s = sqs[int(rng.integers(len(sqs)))]
    col = b.color_at(s)
    att = lst([at(b, x) for x in sorted(b.attackers(not col, s))], "nothing")
    dfn = lst([at(b, x) for x in sorted(b.attackers(col, s))], "nothing")
    q = [f"Attack/defend {at(b, s)}?", f"Who attacks and who defends {at(b, s)}?", f"Attackers and defenders of {at(b, s)}?"][f]
    return q, [f"attacked by {att}, defended by {dfn}.", f"It is attacked by {att} and defended by {dfn}.", f"{att}; {dfn}"][f]


def captures(b, rng, f):
    cs = [m for m in b.legal_moves if b.is_capture(m)]
    cs.sort(key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
    def name(m):
        s = chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)) if b.is_en_passant(m) else m.to_square
        return f"{at(b, m.from_square)} takes {at(b, s)}" + (f" promoting to {PIECES[m.promotion - 1]}" if m.promotion else "")
    q = ["What captures can player make?", "List every capture for the side to move.", "Captures?"][f]
    t = "; ".join(map(name, cs))
    return q, [f"Player can capture with {t}." if cs else "Player has no captures.",
               f"{t}." if cs else "No captures.", " ".join(map(mv, cs)) or "none"][f]


def checks(b, rng, f):
    cs = sorted((m for m in b.legal_moves if b.gives_check(m)), key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
    k = pc(b, chess.Piece(chess.KING, not b.turn))
    name = lambda m: f"{at(b, m.from_square)} to {SQ[m.to_square]}" + (f" promoting to {PIECES[m.promotion - 1]}" if m.promotion else "")
    q = ["What checks can player give?", "List every check for the side to move.", "Checks?"][f]
    t = "; ".join(map(name, cs))
    return q, [f"The {k} can be checked by {t}." if cs else f"The {k} cannot be checked.",
               f"{t}." if cs else "No checks.", " ".join(map(mv, cs)) or "none"][f]


def escapes(b, rng, f):
    if not b.is_check(): return None
    ms = sorted(b.legal_moves, key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
    king = b.king(b.turn)
    cap = [m for m in ms if b.is_capture(m) and m.from_square != king]
    kmv = [m for m in ms if m.from_square == king]
    blk = [m for m in ms if m not in cap and m not in kmv]
    nm = lambda xs: lst([f"{at(b, m.from_square)} to {SQ[m.to_square]}" for m in xs], "none")
    q = ["How to get out of check?", "List every legal way out of check.", "Escapes from check?"][f]
    if not ms: return q, ["There is no way out: checkmate.", "None; it is checkmate.", "none"][f]
    return q, [f"capture: {nm(cap)}; block: {nm(blk)}; king-move: {nm(kmv)}.",
               f"Capture with {nm(cap)}. Block with {nm(blk)}. Move the king {nm(kmv)}.",
               " | ".join(" ".join(map(mv, xs)) or "-" for xs in (cap, blk, kmv))][f]


def mate(b, rng, f):
    if not b.is_check(): return None
    k = b.king(b.turn); kt = f"{pc(b, b.piece_at(k))} at {SQ[k]}"
    q = ["Is it checkmate?", "Is the side to move checkmated?", "Checkmate?"][f]
    if b.is_checkmate(): return q, [f"Yes --- {kt} is mated.", f"Yes, {kt} is checkmated.", "yes"][f]
    return q, ["No.", "No, it is only check.", "no"][f]


STATIC = [piece_on, locate, on_file, on_rank, on_diag, counts, material]
DYNAMIC = [legal_of, attack_defend, captures, checks, escapes, mate]
STAGES = {1: (STATIC, False), 2: (DYNAMIC, False), 3: (STATIC, True), 4: (DYNAMIC, True)}
MIX = {1: [1.0], 2: [0.10, 0.90], 3: [0.06, 0.08, 0.86], 4: [0.06, 0.06, 0.08, 0.80]}       # Table 8
NEEDS_CHECK = (escapes, mate)


def future(b, rng, want_check, tries=200):
    """(board, moves) after 1-8 random legal moves; want_check: retry until the end position is in check."""
    for _ in range(tries):
        c, ms = b.copy(), []
        for _ in range(int(rng.integers(1, 9))):
            if c.is_game_over(): break
            m = list(c.legal_moves)[int(rng.integers(c.legal_moves.count()))]
            ms.append(m); c.push(m)
        if ms and (c.is_check() or not want_check) and (want_check or not c.is_game_over()):
            return c, ms
    return None


def question(stage, i, rng, fen, typ=None):
    """Row (stage, type, question, answer, i, fen) of `stage` on position i, or None if the type does not apply."""
    types, fut = STAGES[stage]
    fn = typ or types[int(rng.integers(len(types)))]
    f = int(rng.integers(3))
    b = chess.Board(fen)
    pre = ""
    if fut:
        r = future(b, rng, fn in NEEDS_CHECK)
        if r is None: return None
        b, ms = r
        pre = ["After " + " ".join(map(mv, ms)) + ": ", "Play " + " ".join(map(mv, ms)) + ". ", "Moves " + " ".join(map(mv, ms)) + ". "][f]
    qa_ = fn(b, rng, f)
    if qa_ is None: return None
    return dict(stage=stage, type=fn.__name__, fmt=f, question=pre + qa_[0], answer=qa_[1], i=int(i), fen=fen)


# ---- data ----
D = np.load(Path(a.data) / "labels.npz")
black, split, fens = D["black"], D["split"], D["fens"]
raw = np.load(Path(a.data) / "acts.npy", mmap_mode="r")
tr = np.flatnonzero(split == 0)
if a.train_n: tr = tr[:a.train_n]
CHECKED = np.array([i for i in tr if chess.Board(str(fens[i])).is_check()])   # current-position escapes/mate draw from these


def fixed_rows(k, n_per_type, seed):
    """n questions per (stage, type) on split k's positions; check-only types use its positions in check."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(split == k)
    chk = [i for i in idx if chess.Board(str(fens[i])).is_check()]
    rows = {}
    for st, (types, fut) in STAGES.items():
        rows[st] = []
        for fn in types:
            pool = chk if fn in NEEDS_CHECK and not fut else idx
            got, tries = 0, 0
            while got < n_per_type and tries < 50 * n_per_type:
                i = pool[int(rng.integers(len(pool)))]; tries += 1
                r = question(st, i, rng, str(fens[i]), fn)
                if r: rows[st].append(r); got += 1
    return rows


VAL, TEST = fixed_rows(1, a.val_n, 1), fixed_rows(2, a.test_n, 2)
for st in STAGES:
    print(f"stage {st}: val {len(VAL[st])}, test {len(TEST[st])} questions", flush=True)

# ---- model ----
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
cuda = torch.cuda.is_available()
dt = torch.bfloat16 if cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
tok = AutoTokenizer.from_pretrained(a.model)
B0 = len(tok)
tok.add_tokens(NEW)
NEW_IDS = tok.convert_tokens_to_ids(NEW)
assert NEW_IDS == list(range(B0, B0 + len(NEW))), "new tokens not contiguous"
try:
    llm = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map="auto" if cuda else None)
except ValueError:
    from transformers import AutoModelForImageTextToText
    llm = AutoModelForImageTextToText.from_pretrained(a.model, dtype=dt, device_map="auto" if cuda else None)
llm.requires_grad_(False).eval()
emb, head = llm.get_input_embeddings(), llm.get_output_embeddings()
assert head.weight.data_ptr() == emb.weight.data_ptr(), "expects tied embeddings"
assert B0 + len(NEW) <= emb.weight.shape[0], "no spare embedding rows for the new tokens"
dev = emb.weight.device


def text_init(t):
    """New token's start embedding: mean embedding of its plain-text name ('e4', 'my knight')."""
    if t.startswith("<SQUARE_"):
        s = int(t[8:-1]) - 1; name = chess.square_name(s)
    else:
        name = ("my " if t[7] == "M" else "opponent's ") + chess.piece_name(chess.PIECE_SYMBOLS.index(t[8].lower()))
    ids = tok(" " + name, add_special_tokens=False)["input_ids"]
    return emb.weight[ids].float().mean(0)


new_emb = nn.Parameter(torch.stack([text_init(t) for t in NEW]).to(dev))


def head_hook(mod, inp, o):
    """Tied output rows of the new tokens: their logits come from new_emb."""
    nl = (inp[0].float() @ new_emb.T).to(o.dtype)
    return torch.cat([o[..., :B0], nl, o[..., B0 + len(NEW):]], -1)


head.register_forward_hook(head_hook)
tgt = [n for n, m in llm.named_modules() if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in set(a.lora_targets.split(","))
       and "visual" not in n and "vision" not in n]
llm = get_peft_model(llm, LoraConfig(r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.0, target_modules=tgt))
llm.train()
lora_params = [p for p in llm.parameters() if p.requires_grad]
print(f"LoRA r={a.lora} on {len(tgt)} modules: {sum(p.numel() for p in lora_params):,} params; {len(NEW)} new tokens", flush=True)
END = "<|im_end|>"; end_id = tok.convert_tokens_to_ids(END); pad_id = tok.pad_token_id if tok.pad_token_id is not None else end_id
chat = tok.apply_chat_template([{"role": "user", "content": "Position: <board>@@BOARD@@</board>\n@@Q@@"}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
pre_s, post_s = chat.split("@@BOARD@@")
PRE = tok(pre_s, add_special_tokens=False)["input_ids"]


def prompt_ids(question):
    ids = PRE + [pad_id] * 64
    return ids + tok(post_s.replace("@@Q@@", question), add_special_tokens=False)["input_ids"], list(range(len(PRE), len(PRE) + 64))


class Projector(nn.Module):
    """train.py's: per square (token + learned square embedding) -> MLP -> LayerNorm * scale."""
    def __init__(self, d_in, d_hid, d_out, rms):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(64, d_in))
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))
        self.norm = nn.LayerNorm(d_out, elementwise_affine=False)
        self.scale = nn.Parameter(torch.tensor(float(rms)))

    def forward(self, x):
        return self.norm(self.net(x + self.pos)) * self.scale


def norm_stats(rows, chunk=1024):
    s = q = 0; n = 0
    for k in range(0, len(rows), chunk):
        c = torch.from_numpy(np.asarray(raw[rows[k:k + chunk]], dtype=np.float64)).flatten(0, 1)
        s, q, n = s + c.sum(0), q + c.pow(2).sum(0), n + len(c)
    m = s / n
    return m.float(), ((q - n * m ** 2) / (n - 1)).sqrt().float() + 1e-6


mu, sd = norm_stats(tr)
rms = emb.weight.float().pow(2).mean(1).sqrt().mean().item()
proj = Projector(256, a.hidden, emb.weight.shape[1], rms).to(dev)


def soft(i):
    x = torch.from_numpy(np.asarray(raw[i], dtype=np.float32))
    return proj(((x - mu) / sd).to(dev))


def embed(ids_list, idx):
    """Right-padded inputs: token embeddings (new tokens from new_emb), the 64 pads replaced by projected squares."""
    L = max(map(len, ids_list))
    ids = torch.full((len(ids_list), L), pad_id, dtype=torch.long); mask = torch.zeros_like(ids)
    for k, s in enumerate(ids_list): ids[k, :len(s)] = torch.tensor(s); mask[k, :len(s)] = 1
    ids = ids.to(dev); isnew = (ids >= B0) & (ids < B0 + len(NEW))
    x = emb(torch.where(isnew, pad_id, ids)).clone()
    x[isnew] = new_emb[ids[isnew] - B0].to(x.dtype)
    x[:, len(PRE):len(PRE) + 64] = torch.stack([soft(i) for i in idx]).to(x.dtype)
    return x, mask.to(dev)


def sample(stage, i, rng):
    """One training row: replay stage drawn by Table 8, current-position check types move to a checked position."""
    while True:
        st = 1 + int(rng.choice(stage, p=MIX[stage]))
        types, fut = STAGES[st]
        fn = types[int(rng.integers(len(types)))]
        j = int(rng.choice(CHECKED)) if fn in NEEDS_CHECK and not fut else i
        r = question(st, j, rng, str(fens[j]), fn)
        if r: return r
        i = int(rng.choice(tr))


def batch(idx, stage, rng):
    rows = [sample(stage, i, rng) for i in idx]
    ids, labels = [], []
    for r in rows:
        p, _ = prompt_ids(r["question"]); y = tok(r["answer"] + END, add_special_tokens=False)["input_ids"]
        ids.append(p + y); labels.append([-100] * len(p) + y)
    x, mask = embed(ids, [r["i"] for r in rows])
    lab = torch.full(mask.shape, -100, dtype=torch.long)
    for k, l in enumerate(labels): lab[k, :len(l)] = torch.tensor(l)
    keep = slice(min(l.index(next(t for t in l if t != -100)) for l in labels) - 1, mask.shape[1] - 1)
    return x, mask, lab.to(dev), keep


def loss_fn(x, mask, lab, keep):
    logits = llm(inputs_embeds=x, attention_mask=mask, logits_to_keep=keep, use_cache=False).logits
    y = lab[:, keep.start + 1:keep.stop + 1].flatten().to(logits.device)
    return nn.functional.cross_entropy(logits.float().flatten(0, 1), y, ignore_index=-100)


@torch.no_grad()
def generate(rows):
    """Greedy (text, token ids) per row; rows of equal prompt length batched together (no padding), KV cache."""
    proj.eval(); llm.eval()
    prompts = [prompt_ids(r["question"])[0] for r in rows]
    outs = [None] * len(rows); groups = {}
    for k, p in enumerate(prompts): groups.setdefault(len(p), []).append(k)
    for ks in groups.values():
        for s in range(0, len(ks), a.eval_batch):
            kk = ks[s:s + a.eval_batch]
            mn = max(len(tok(rows[k]["answer"] + END, add_special_tokens=False)["input_ids"]) for k in kk) + 8
            x, _ = embed([prompts[k] for k in kk], [rows[k]["i"] for k in kk])
            gen = torch.zeros((len(kk), 0), dtype=torch.long, device=dev)
            o = llm(inputs_embeds=x, use_cache=True, logits_to_keep=1)
            for _ in range(mn):
                nxt = o.logits[:, -1].argmax(-1).to(dev)
                gen = torch.cat([gen, nxt[:, None]], 1)
                if (gen == end_id).any(1).all(): break
                isnew = (nxt >= B0) & (nxt < B0 + len(NEW))
                e = emb(torch.where(isnew, pad_id, nxt)).clone()
                e[isnew] = new_emb[nxt[isnew] - B0].to(e.dtype)
                o = llm(inputs_embeds=e[:, None], past_key_values=o.past_key_values, use_cache=True, logits_to_keep=1)
            for k, g in zip(kk, gen.tolist()):
                g = g[:g.index(end_id)] if end_id in g else g
                outs[k] = (tok.decode(g, skip_special_tokens=True).strip(), g)
    proj.train(); llm.train()
    return outs


def evaluate(sets, stage, step, path, samples_path):
    """Exact-match accuracy (answer token ids) per stage and question type; one report line, answers to samples_path."""
    t1 = time.time(); rep = dict(stage=stage, step=step, by={}); samples = []
    for st, rows in sets.items():
        gen = generate(rows); pr = [t for t, _ in gen]
        ok = [g == tok(r["answer"], add_special_tokens=False)["input_ids"] for r, (_, g) in zip(rows, gen)]
        types = {}
        for r, o in zip(rows, ok): types.setdefault(r["type"], []).append(o)
        rep["by"][st] = dict(acc=round(float(np.mean(ok)), 4), types={t: round(float(np.mean(v)), 4) for t, v in types.items()})
        samples += [dict(r, pred=p, ok=o) for r, p, o in zip(rows, pr, ok)]
    rep["eval_min"] = round((time.time() - t1) / 60, 1)
    with open(path, "a") as f: f.write(json.dumps(rep) + "\n")
    with open(samples_path, "w") as f:
        for r in samples: f.write(json.dumps(r) + "\n")
    print(f"== {path.stem} stage {stage} step {step}: " + "  ".join(f"S{st} {v['acc']}" for st, v in rep["by"].items()), flush=True)
    for st, v in rep["by"].items(): print(f"   S{st}", "  ".join(f"{t} {x}" for t, x in v["types"].items()), flush=True)
    return rep


def state():
    return dict(proj=proj.state_dict(), new_emb=new_emb.detach().cpu(), lora_sd=get_peft_model_state_dict(llm), mu=mu, sd=sd,
                tokens=NEW, model=a.model, args=vars(a))


def load(ck):
    proj.load_state_dict(ck["proj"]); new_emb.data.copy_(ck["new_emb"].to(dev)); set_peft_model_state_dict(llm, ck["lora_sd"])


def save(obj, path):
    torch.save(obj, path.with_suffix(".tmp")); os.replace(path.with_suffix(".tmp"), path)


# ---- stages ----
STEPS = [int(x) for x in a.steps.split(",")]; LRS = [float(x) for x in a.lrs.split(",")]
NAMES = {1: "static_current", 2: "dynamic_current", 3: "static_future", 4: "dynamic_future"}
prev = None
for stage in map(int, a.stages.split(",")):
    sd_ = out / f"{stage}_{NAMES[stage]}"; sd_.mkdir(exist_ok=True)
    if (sd_ / "report_test.jsonl").exists():
        prev = sd_ / "best.pt"; print(f"stage {stage} done", flush=True); continue
    n, lr = STEPS[stage - 1], LRS[stage - 1]
    opt = torch.optim.AdamW([dict(params=list(proj.parameters()) + lora_params, lr=lr),
                             dict(params=[new_emb], lr=lr * a.token_lr)], weight_decay=a.wd)
    warm = max(1, round(0.05 * n))
    lam = lambda s: (s + 1) / warm if s < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    rng = np.random.default_rng(a.seed * 100 + stage)
    step, best, bad = 0, -1.0, 0
    if (sd_ / "last.pt").exists():
        ck = torch.load(sd_ / "last.pt", map_location=dev, weights_only=False); load(ck)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["trng"]); step, best, bad = ck["step"], ck["best"], ck["bad"]
        print(f"stage {stage}: resumed at step {step}, best {best}, bad {bad}", flush=True)
    elif prev:
        load(torch.load(prev, map_location=dev, weights_only=False)); print(f"stage {stage}: from {prev}", flush=True)
    print(f"stage {stage} {NAMES[stage]}: {n} steps, lr {lr}, mix {MIX[stage]}", flush=True)
    t0, run, k = time.time(), 0.0, 0
    while step < n and bad < a.patience:
        idx = rng.choice(tr, a.batch)
        loss = loss_fn(*batch(idx, stage, rng))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(proj.parameters()) + lora_params + [new_emb], 1.0)
        opt.step(); sched.step(); step += 1
        run += loss.item(); k += 1
        if step % a.log_every == 0:
            el = (time.time() - t0) / k
            print(f"stage {stage} step {step}/{n} loss {run / k:.4f} {el:.2f}s/step", flush=True); t0, run, k = time.time(), 0.0, 0
        if step % a.eval_every == 0 or step == n:
            acc = evaluate({stage: VAL[stage]}, stage, step, sd_ / "report.jsonl", sd_ / f"samples_step{step}.jsonl")["by"][stage]["acc"]
            if acc > best: best, bad = acc, 0; save(state(), sd_ / "best.pt")
            else: bad += 1
            save(dict(state(), opt=opt.state_dict(), sched=sched.state_dict(), rng=rng.bit_generator.state,
                      trng=torch.get_rng_state(), step=step, best=best, bad=bad), sd_ / "last.pt")
            t0, run, k = time.time(), 0.0, 0
    print(f"stage {stage}: stopped at step {step}, best val {best}", flush=True)
    load(torch.load(sd_ / "best.pt", map_location=dev, weights_only=False))
    evaluate({st: TEST[st] for st in range(1, stage + 1)}, stage, step, sd_ / "report_test.jsonl", sd_ / "samples_test.jsonl")
    prev = sd_ / "best.pt"
