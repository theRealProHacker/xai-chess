"""Who gains from each centre type (features/src/centre.rs kind()) in Leela T1-256x10, and how faithfully the
per-square decoder tracks it.

Every board is mirrored so the side to move is White, with any en-passant square cleared; v = Leela's
P(win) - P(loss) for the side to move. kind() is ported below (checked equal to the binary's pawn.centre
bit on 5,000 cached rows); "mobile+" is a mobile centre of the side to move, "mobile-" of the opponent.

Stages, each writing lc0-probe/results_centre_type_<stage>.json:
  leela    Leela on the 100k game positions (sf-probe/data/fens_games.txt), cached for regress.
  relabel  Decoder only: rewrite the pawn.centre bit on every c-f pawn to each kind, decode, run the net's
           rest; how far the value moves. Held-out cached rows (fens.txt), one decoder at a time.
  edit     Board edits. From quiet game positions (Stockfish depth 12: best minus third-best <= --gap cp),
           remove one white and one black c-f pawn so the kind changes (edit), one white and one black
           a/b/g/h pawn, away from the own king's file when possible (flank control, kind unchanged), and,
           where one exists, a c-f pair that keeps the kind (centre control).
           tau = dv(edit) - dv(control); tau > 0: the side to move gains from the new kind.
  fidelity Features at --nodes for the edit stage's boards (the square binary, --jobs processes), then the
           decoders' dv and tau against real Leela's.
  shap     Exact Shapley over the 6 players {White d/e pawns, Black d/e pawns, White N, White B, Black N,
           Black B}, everything else always on (64 Leela runs per position), stratified by kind; from the
           same table the games with both sides' d/e pawns merged into one player C ({C, WN, WB, BN, BB} and
           {C, White minors, Black minors}) and with minors merged per side. Grabisch-Roubens pairwise interactions.
  regress  Observational: v on kind dummies with exact material-signature fixed effects, then with
           flank-pawn, castling, king-square, ply and rating controls, then also c-f pawns per file (which
           nearly fix the kind, so that spec over-controls). Games (cache from leela) and the cached puzzle
           rows (lc0-probe/data/wdl.npy).

  python lc0-probe/centre_type_effect.py --stage leela
  python lc0-probe/centre_type_effect.py --stage relabel
  python lc0-probe/centre_type_effect.py --stage edit --per 150 --gap 50
  python lc0-probe/centre_type_effect.py --stage fidelity --jobs 2
  python lc0-probe/centre_type_effect.py --stage shap --per 50
  python lc0-probe/centre_type_effect.py --stage regress
"""
import argparse, collections, itertools, json, subprocess, sys, time, types
from math import factorial
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import chess, chess.engine, numpy as np, onnxruntime as ort
from lc0_planes import encode_fens

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["leela", "relabel", "edit", "fidelity", "shap", "regress"])
ap.add_argument("--per", type=int, default=150, help="edit: rows per transition; shap: positions per kind")
ap.add_argument("--gap", type=int, default=50, help="quiet gate, centipawns")
ap.add_argument("--nodes", type=int, default=200000)
ap.add_argument("--jobs", type=int, default=2, help="fidelity: parallel square processes")
ap.add_argument("--positions", type=int, default=600, help="relabel: held-out rows")
ap.add_argument("--decoders", default="encoder0_ln2_n100000,encoder2_ln1_n100000,encoder2_ln2_n100000,encoder5_ln1_n100000,"
                "attn_body_rehape_n100000,encoder5_ln2_n20000,encoder7_ln2_n20000,encoder9_ln1_n20000,encoder9_ln2_n20000")
ap.add_argument("--fid-decoders", default="encoder0_ln2_n100000,encoder2_ln2_n100000")
ap.add_argument("--scratch", default=str(ROOT / "lc0-probe/data"), help="where the fidelity feature rows are cached")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
OUT = lambda s: ROOT / f"lc0-probe/results_centre_type_{s}.json"
GAMES = ROOT / "sf-probe/data/fens_games.txt"
LEELA_GAMES = ROOT / "lc0-probe/data/centre_type_games_leela.npy"
SF = ROOT / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2"
SQUARE = ROOT / "features/square/target/release/square"
NET = ROOT / "vendor/lc0-nets/t1-256x10.onnx"
W, B = chess.WHITE, chess.BLACK

def kind(b):
    """centre.rs kind() for board b; mobile split by side ('mobile+' = White's). None without a c-f pawn."""
    P = lambda f, r, c: b.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, c)
    has = lambda f, c: any(P(f, r, c) for r in range(8))
    if not any(has(f, c) for f in range(2, 6) for c in (W, B)): return None
    if all(any(P(f, r, W) and P(f, r + 1, B) for r in range(7)) for f in (3, 4)): return "closed"
    for c in (W, B):
        ps = [(chess.square_file(s), chess.square_rank(s)) for s in b.pieces(chess.PAWN, c) if 2 <= chess.square_file(s) <= 5]
        fw = 1 if c else -1
        if len(ps) >= 2 and any(f in (3, 4) for f, _ in ps) and not has(3, not c) and not has(4, not c) \
           and all(0 <= r + fw < 8 and b.piece_at(chess.square(f, r + fw)) is None for f, r in ps):
            return "mobile+" if c else "mobile-"
    op = [f for f in range(2, 6) if not has(f, W) and not has(f, B)]
    half = [f for f in range(2, 6) if has(f, W) != has(f, B)]
    return "open" if len(op) >= 2 else "static" if op or len(half) >= 2 else "dynamic"

def stm(fen):
    b = chess.Board(fen); b = b if b.turn else b.mirror(); b.ep_square = None
    return b

so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
_sess = None
def leela(fens):
    """P(win) - P(loss) for the side to move, batched."""
    global _sess
    _sess = _sess or ort.InferenceSession(str(NET), so, providers=["CPUExecutionProvider"])
    out = [_sess.run(["/output/wdl"], {"/input/planes": encode_fens(fens[s:s + 256])})[0] for s in range(0, len(fens), 256)]
    w = np.concatenate(out) if out else np.zeros((0, 3))
    return w[:, 0] - w[:, 2]

def boot(x, n=2000, seed=0):
    """Mean and 95% percentile bootstrap CI."""
    x = np.asarray(x, float)
    if len(x) < 2: return {"n": len(x), "mean": float(x.mean()) if len(x) else None}
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return {"n": len(x), "mean": float(x.mean()), "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))],
            "median": float(np.median(x)), "sd": float(x.std())}

_sf = None
def gap(b):
    """Stockfish depth 12: best minus third-best move in cp; None with a mate score or under three moves."""
    global _sf
    if _sf is None:
        _sf = chess.engine.SimpleEngine.popen_uci(str(SF)); _sf.configure({"Threads": 1, "Hash": 64})
    info = _sf.analyse(b, chess.engine.Limit(depth=12), multipv=3)
    cp = [x["score"].relative.score() for x in info]
    return None if len(cp) < 3 or None in cp else cp[0] - cp[2]

def game_boards():
    fens = [l for l in GAMES.read_text().split("\n") if l]
    return fens, [stm(f) for f in fens]

def pawns(b, c, files):
    return [s for s in b.pieces(chess.PAWN, c) if chess.square_file(s) in files]

def minus(b, sqs):
    e = b.copy(); [e.remove_piece_at(s) for s in sqs]
    return e

# ---------------------------------------------------------------- leela
if a.stage == "leela":
    fens, boards = game_boards()
    t0 = time.time(); v = np.concatenate([leela([x.fen() for x in boards[s:s + 4096]]) for s in range(0, len(boards), 4096)])
    np.save(LEELA_GAMES, v.astype(np.float32)); print(f"{len(v)} positions, {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- decoder helpers
def load_decoder(name):
    """(value(pre-activations) -> v, col map bit -> input row, E, base pre-activation) for one checkpoint."""
    import torch, onnx
    tv = types.ModuleType("torchvision"); tv.ops = types.ModuleType("torchvision.ops")
    tv.ops.roi_align = tv.ops.nms = tv.ops.batched_nms = None
    sys.modules.update({"torchvision": tv, "torchvision.ops": tv.ops})
    from onnx2torch import convert
    torch.set_num_threads(a.threads)
    ck = torch.load(ROOT / f"lc0-probe/data/square_decoder_{name}.pt", weights_only=False)
    keep, mu, sd = ck["keep"], torch.from_numpy(ck["mu"]).float(), float(ck["sd"])
    tap = "/" + name.rsplit("_n", 1)[0].replace("attn_body_rehape", "attn_body/rehape").replace("_ln", "/ln")
    E, bias = ck["state"]["inp.weight"], ck["state"]["bias"]
    out = torch.nn.Linear(*ck["state"]["out.weight"].shape[::-1])
    out.load_state_dict({"weight": ck["state"]["out.weight"], "bias": ck["state"]["out.bias"]})
    sub = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(onnx.load(NET))).extract_model([tap], ["/output/wdl"])
    assert [i.name for i in sub.graph.input] == [tap]
    flat = len(sub.graph.input[0].type.tensor_type.shape.dim) == 2
    tail = convert(sub).eval()
    col = np.full(E.shape[0] + 100000, -1); col[keep] = np.arange(len(keep))
    base = E[len(keep):] + bias
    @torch.no_grad()
    def value(P):
        ys = []
        for s in range(0, len(P), 64):
            T = out(torch.nn.functional.gelu(P[s:s + 64])) * sd + mu
            w = tail(T.reshape(-1, mu.shape[0]) if flat else T)
            ys.append(w[:, 0] - w[:, 2])
        return torch.cat(ys).numpy()
    def pre(squares):
        """{square name: [bit indices]} -> (64, hidden) pre-activation."""
        P = base.clone()
        for s, v in squares.items():
            c = col[np.asarray(v)]; c = c[c >= 0]
            if len(c): P[chess.parse_square(s)] += E[torch.from_numpy(c)].sum(0)
        return P
    return value, pre, col, E, base, tap

def schema_labels():
    return json.loads(subprocess.run([SQUARE, "--schema"], capture_output=True, text=True, check=True).stdout)["labels"]

# ---------------------------------------------------------------- relabel
if a.stage == "relabel":
    import torch, gc
    L = schema_labels(); KB = {l.split(".")[-1]: i for i, l in enumerate(L) if l.startswith("pawn.centre.")}
    te = np.random.default_rng(0).random(100000) < 0.2                      # held out by every n>=20000 decoder
    wdl = np.load(ROOT / "lc0-probe/data/wdl.npy"); y_all = wdl[:, 0] - wdl[:, 2]
    rows = []
    with open(ROOT / "lc0-probe/data/square_bits_n100000_nodes200000.jsonl") as f:
        for i, l in enumerate(f):
            if i >= 20000 or len(rows) >= a.positions: break
            if not te[i]: continue
            r = json.loads(l); k = next((L[j].split(".")[-1] for v in r["squares"].values() for j in v if j in KB.values()), None)
            if k: rows.append((r["squares"], k, float(y_all[i])))
    res = {"positions": len(rows), "kinds": dict(collections.Counter(k for _, k, _ in rows)), "decoders": {}}
    for name in a.decoders.split(","):
        t0 = time.time()
        value, pre, col, E, base, tap = load_decoder(name)
        miss = [k for k, j in KB.items() if col[j] < 0]
        V = []
        for sq, k, y in rows:
            P0 = pre(sq); cen = [(chess.parse_square(s), KB[k]) for s, v in sq.items() if KB[k] in v]
            vs = [P0]
            for k2 in list(KB) + [None]:                                   # each kind, then no centre bit at all
                P = P0.clone()
                for s, j in cen:
                    if col[j] >= 0: P[s] -= E[col[j]]
                    if k2 and col[KB[k2]] >= 0: P[s] += E[col[KB[k2]]]
                vs.append(P)
            V.append(value(torch.stack(vs)))
        V = np.array(V); y = np.array([r[2] for r in rows]); K = list(KB)
        spread = V[:, 1:6].max(1) - V[:, 1:6].min(1)
        res["decoders"][name] = {"tap": tap, "kinds_not_in_decoder": miss,
            "abs_err_vs_leela": float(np.abs(V[:, 0] - y).mean()),
            "spread_over_kinds": boot(spread), "spread_max": float(spread.max()),
            "spread_balanced": boot(spread[np.abs(y) <= 0.5]),
            "abs_closed_minus_open": boot(np.abs(V[:, 1 + K.index("closed")] - V[:, 1 + K.index("open")])),
            "closed_minus_open": boot(V[:, 1 + K.index("closed")] - V[:, 1 + K.index("open")]),
            "abs_remove_bit": boot(np.abs(V[:, 6] - V[:, 0])),
            "by_kind_spread": {k: float(spread[[r[1] == k for r in rows]].mean()) for k in res["kinds"]}}
        print(name, f"{time.time() - t0:.0f}s", json.dumps(res["decoders"][name]["spread_over_kinds"]), flush=True)
        del value, pre, col, E, base; gc.collect()
        json.dump(res, open(OUT("relabel"), "w"), indent=1)

# ---------------------------------------------------------------- edit
if a.stage == "edit":
    rnd = np.random.default_rng(a.seed)
    fens, boards = game_boards()
    order = rnd.permutation(len(boards))
    count, rows, gated, quiet = collections.Counter(), [], 0, 0
    def flank(b):
        """A white and a black a/b/g/h pawn, away from the own king's file if possible; (pair, shelter flag)."""
        pick = []
        for c in (W, B):
            ps = pawns(b, c, (0, 1, 6, 7)); kf = chess.square_file(b.king(c))
            far = [s for s in ps if abs(chess.square_file(s) - kf) > 1]
            pick.append((far or ps, not far))
        if not pick[0][0] or not pick[1][0]: return None, None
        return (int(rnd.choice(pick[0][0])), int(rnd.choice(pick[1][0]))), pick[0][1] or pick[1][1]
    t0 = time.time()
    for n_seen, i in enumerate(order):
        b = boards[i]; k0 = kind(b)
        if k0 is None or not b.is_valid(): continue
        cands, same = collections.defaultdict(list), []
        for x, y in itertools.product(pawns(b, W, range(2, 6)), pawns(b, B, range(2, 6))):
            e = minus(b, (x, y))
            if not e.is_valid(): continue
            k1 = kind(e) or "none"
            (same if k1 == k0 else cands[f"{k0}>{k1}"]).append((x, y))
        need = [t for t in cands if count[t] < a.per]
        if not need: continue
        fl, shelter = flank(b)
        if fl is None or not minus(b, fl).is_valid(): continue
        gated += 1; g0 = gap(b)
        if g0 is None or g0 > a.gap: continue
        quiet += 1
        ctrl = minus(b, fl); assert kind(ctrl) == k0
        sm = minus(b, same[rnd.integers(len(same))]) if same else None
        g_ctrl = gap(ctrl)
        for t in need:
            pr = cands[t][rnd.integers(len(cands[t]))]; e = minus(b, pr)
            rows.append({"i": int(i), "base": b.fen(), "t": t, "edit": e.fen(), "ctrl": ctrl.fen(), "same": sm.fen() if sm else None,
                         "removed": [chess.square_name(s) for s in pr], "ctrl_removed": [chess.square_name(s) for s in fl],
                         "ctrl_shelter": bool(shelter), "gap": g0, "gap_edit": gap(e), "gap_ctrl": g_ctrl})
            count[t] += 1
        if gated % 200 == 0: print(n_seen, gated, quiet, f"{time.time() - t0:.0f}s", dict(count), flush=True)
    vb = leela([r["base"] for r in rows]); ve = leela([r["edit"] for r in rows]); vc = leela([r["ctrl"] for r in rows])
    hs = [j for j, r in enumerate(rows) if r["same"]]; vs = dict(zip(hs, leela([rows[j]["same"] for j in hs])))
    for j, r in enumerate(rows):
        r.update(v=float(vb[j]), d_edit=float(ve[j] - vb[j]), d_ctrl=float(vc[j] - vb[j]), tau=float(ve[j] - vc[j]))
        if j in vs: r.update(d_same=float(vs[j] - vb[j]), tau_same=float(ve[j] - vs[j]))
    def summ(R):
        q = lambda r: r["gap_edit"] is not None and r["gap_edit"] <= a.gap and r["gap_ctrl"] is not None and r["gap_ctrl"] <= a.gap
        return {"tau": boot([r["tau"] for r in R]), "d_edit": boot([r["d_edit"] for r in R]), "d_ctrl": boot([r["d_ctrl"] for r in R]),
                "tau_same": boot([r["tau_same"] for r in R if "tau_same" in r]),
                "tau_edit_ctrl_quiet": boot([r["tau"] for r in R if q(r)]),
                "tau_ctrl_no_shelter": boot([r["tau"] for r in R if not r["ctrl_shelter"]]),
                "tau_balanced": boot([r["tau"] for r in R if abs(r["v"]) <= 0.5])}
    by = collections.defaultdict(list)
    for r in rows: by[r["t"]].append(r)
    res = {"gap": a.gap, "per": a.per, "bases_seen": n_seen + 1, "gated": gated, "quiet": quiet,
           "summary": {t: summ(R) for t, R in sorted(by.items(), key=lambda x: -len(x[1]))}, "rows": rows}
    json.dump(res, open(OUT("edit"), "w"), indent=1)
    for t, s in res["summary"].items(): print(t, s["tau"])

# ---------------------------------------------------------------- fidelity
if a.stage == "fidelity":
    import torch
    E_ = json.load(open(OUT("edit"))); rows = E_["rows"]
    fens = list(dict.fromkeys(f for r in rows for f in (r["base"], r["edit"], r["ctrl"])))
    cache = Path(a.scratch) / f"centre_type_bits_nodes{a.nodes}.jsonl"
    have = {json.loads(l)["fen"] for l in open(cache)} if cache.exists() else set()
    todo = [f for f in fens if f not in have]
    print(f"{len(fens)} boards, {len(todo)} to extract", flush=True)
    if todo:
        parts = [todo[j::a.jobs] for j in range(a.jobs)]
        ps = [subprocess.Popen(["nice", "-n", "19", SQUARE, "--indices", "--engine", SF, "--nodes", str(a.nodes)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) for _ in parts]
        import threading
        lock = threading.Lock()
        def feed(p, part): p.stdin.write("\n".join(part) + "\n"); p.stdin.close()
        def drain(p):
            for l in p.stdout:
                with lock, open(cache, "a") as f: f.write(l)
        th = [threading.Thread(target=fn, args=args) for p, part in zip(ps, parts) for fn, args in ((feed, (p, part)), (drain, (p,)))]
        [t.start() for t in th]; [t.join() for t in th]; [p.wait() for p in ps]
    bits = {}
    for l in open(cache):
        r = json.loads(l); bits[r["fen"]] = r["squares"]
    real = dict(zip(fens, leela(fens)))
    res = {"nodes": a.nodes, "boards": len(fens), "decoders": {}}
    for name in a.fid_decoders.split(","):
        value, pre, *_ = load_decoder(name)
        dec = {}
        for s in range(0, len(fens), 64):
            ch = fens[s:s + 64]; dec.update(zip(ch, value(torch.stack([pre(bits[f]) for f in ch]))))
        def fit(p, y):
            p, y = np.asarray(p), np.asarray(y)
            return {"n": len(y), "corr": float(np.corrcoef(p, y)[0, 1]), "fvu": float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                    "mae": float(np.abs(y - p).mean())}
        D = lambda r, k: (dec[r[k]] - dec[r["base"]], real[r[k]] - real[r["base"]])
        out = {"value": fit([dec[f] for f in fens], [real[f] for f in fens]),
               "d_edit": fit(*zip(*[D(r, "edit") for r in rows])), "d_ctrl": fit(*zip(*[D(r, "ctrl") for r in rows])),
               "tau": fit([dec[r["edit"]] - dec[r["ctrl"]] for r in rows], [real[r["edit"]] - real[r["ctrl"]] for r in rows]),
               "by_transition": {}}
        by = collections.defaultdict(list)
        for r in rows: by[r["t"]].append(r)
        for t, R in by.items():
            td = [dec[r["edit"]] - dec[r["ctrl"]] for r in R]; tr = [real[r["edit"]] - real[r["ctrl"]] for r in R]
            out["by_transition"][t] = {"tau_real": boot(tr), "tau_dec": boot(td), "fit": fit(td, tr)}
        res["decoders"][name] = out
        print(name, json.dumps({k: out[k] for k in ("value", "d_edit", "d_ctrl", "tau")}), flush=True)
        del value, pre
    json.dump(res, open(OUT("fidelity"), "w"), indent=1)

# ---------------------------------------------------------------- shap
def shapley_gr(u, m):
    """Exact Shapley values and Grabisch-Roubens pairwise interactions of an m-player game u(frozenset)."""
    N = range(m); phi, I = [], {}
    for i in N:
        o = [q for q in N if q != i]
        phi.append(sum(factorial(r) * factorial(m - r - 1) / factorial(m) * (u(frozenset(S) | {i}) - u(frozenset(S)))
                       for r in range(m) for S in itertools.combinations(o, r)))
    for i, j in itertools.combinations(N, 2):
        o = [q for q in N if q not in (i, j)]
        I[(i, j)] = sum(factorial(r) * factorial(m - r - 2) / factorial(m - 1) *
                        (u(frozenset(S) | {i, j}) - u(frozenset(S) | {i}) - u(frozenset(S) | {j}) + u(frozenset(S)))
                        for r in range(m - 1) for S in itertools.combinations(o, r))
    return phi, I

if a.stage == "shap":
    rnd = np.random.default_rng(a.seed + 1)
    fens, boards = game_boards()
    P6 = ["WP", "BP", "WN", "WB", "BN", "BB"]
    def players(b):
        return [pawns(b, W, (3, 4)), pawns(b, B, (3, 4)), list(b.pieces(chess.KNIGHT, W)), list(b.pieces(chess.BISHOP, W)),
                list(b.pieces(chess.KNIGHT, B)), list(b.pieces(chess.BISHOP, B))]
    KINDS = ["closed", "open", "static", "dynamic"]                          # mobile cannot have d/e pawns on both sides
    count, picked, t0 = collections.Counter(), [], time.time()
    for i in rnd.permutation(len(boards)):
        b = boards[i]; k = kind(b)
        if k not in KINDS or count[k] >= a.per or not b.is_valid(): continue
        pl = players(b)
        if not (pl[0] and pl[1] and (pl[2] or pl[3]) and (pl[4] or pl[5])): continue
        coal = [minus(b, [s for j in range(6) if not (m >> j) & 1 for s in pl[j]]) for m in range(64)]
        if not all(c.is_valid() for c in coal): continue
        g = gap(b)
        if g is None or g > a.gap: continue
        count[k] += 1; picked.append((int(i), b, k, g, coal))
        if all(count[x] >= a.per for x in KINDS): break
    print(dict(count), f"{time.time() - t0:.0f}s", flush=True)
    V = leela([c.fen() for _, _, _, _, coal in picked for c in coal]).reshape(len(picked), 64)
    games = {"6": [[0], [1], [2], [3], [4], [5]], "5": [[0, 1], [2], [3], [4], [5]], "4": [[0], [1], [2, 3], [4, 5]], "3": [[0, 1], [2, 3], [4, 5]]}
    names = {"6": P6, "5": ["C"] + P6[2:], "4": ["WP", "BP", "WM", "BM"], "3": ["C", "WM", "BM"]}
    rows = []
    for (i, b, k, g, _), v in zip(picked, V):
        r = {"i": i, "fen": b.fen(), "kind": k, "gap": g, "v_all": float(v[63]), "v_none": float(v[0]), "v": [round(float(x), 5) for x in v]}
        for gname, parts in games.items():
            u = lambda S: float(v[sum(1 << j for p in S for j in parts[p])])
            phi, I = shapley_gr(u, len(parts)); nm = names[gname]
            r[f"phi{gname}"] = dict(zip(nm, phi)); r[f"I{gname}"] = {f"{nm[x]}x{nm[y]}": val for (x, y), val in I.items()}
        rows.append(r)
    summ = {}
    for gname in games:
        for key in (f"phi{gname}", f"I{gname}"):
            for p in rows[0][key]:
                s = summ.setdefault(f"{key}.{p}", {})
                for k in KINDS: s[k] = boot([r[key][p] for r in rows if r["kind"] == k])
                xc = [r[key][p] for r in rows if r["kind"] == "closed"]; xo = [r[key][p] for r in rows if r["kind"] == "open"]
                bs = np.random.default_rng(0)
                d = [np.mean(bs.choice(xc, len(xc))) - np.mean(bs.choice(xo, len(xo))) for _ in range(2000)]
                s["closed_minus_open"] = {"mean": float(np.mean(xc) - np.mean(xo)), "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
    json.dump({"gap": a.gap, "per": a.per, "count": dict(count), "summary": summ, "rows": rows}, open(OUT("shap"), "w"), indent=1)
    for key in ("I3.CxWM", "I3.CxBM", "I5.CxWN", "I5.CxWB", "I5.CxBN", "I5.CxBB", "I6.WPxWN", "I6.WPxWB", "I6.BPxBN", "I6.BPxBB"):
        print(key, {k: round(summ[key][k]["mean"], 4) for k in KINDS}, summ[key]["closed_minus_open"])

# ---------------------------------------------------------------- regress
def design(boards, extra):
    """Kind labels, material-signature group ids, control matrix."""
    kinds = [kind(b) or "none" for b in boards]
    sig = [tuple(len(b.pieces(p, c)) for c in (W, B) for p in range(1, 6)) for b in boards]
    gid = np.unique(np.array(sig), axis=0, return_inverse=True)[1].ravel()
    X = []
    for b, e in zip(boards, extra):
        f = [len(pawns(b, c, fs)) for c in (W, B) for fs in ((0, 1), (2,), (3,), (4,), (5,), (6, 7))]
        f += [b.has_kingside_castling_rights(W), b.has_queenside_castling_rights(W), b.has_kingside_castling_rights(B),
              b.has_queenside_castling_rights(B)]
        f += [chess.square_file(b.king(c)) for c in (W, B)] + [chess.square_rank(b.king(W)), 7 - chess.square_rank(b.king(B))]
        X.append(f + list(e))
    return kinds, gid, np.array(X, float)

def fe_ols(y, D, gid):
    """OLS of y on D after demeaning within groups; coefficients and HC1 SEs. Singleton groups drop out."""
    cnt = np.bincount(gid); m = cnt[gid] > 1
    y, D, gid = y[m], D[m], np.unique(gid[m], return_inverse=True)[1]
    dm = lambda Z: Z - (np.stack([np.bincount(gid, Z[:, j]) for j in range(Z.shape[1])], 1) / np.bincount(gid)[:, None])[gid]
    yd, Dd = dm(y[:, None])[:, 0], dm(D)
    keep = Dd.std(0) > 1e-9; Dd = Dd[:, keep]
    beta, *_ = np.linalg.lstsq(Dd, yd, rcond=None); e = yd - Dd @ beta
    n, k = Dd.shape; k += gid.max() + 1
    XtXi = np.linalg.pinv(Dd.T @ Dd); cov = XtXi @ (Dd.T * e ** 2) @ Dd @ XtXi * n / max(n - k, 1)
    b_full = np.full(len(keep), np.nan); se = np.full(len(keep), np.nan)
    b_full[keep] = beta; se[keep] = np.sqrt(np.diag(cov))
    C = np.full((len(keep), len(keep)), np.nan); C[np.ix_(keep, keep)] = cov
    return b_full, se, C, int(m.sum())

if a.stage == "regress":
    KS = ["open", "closed", "mobile+", "mobile-", "dynamic", "none"]          # reference: static
    res = {}
    fens, boards = game_boards()
    meta = [l.split("\t") for l in (ROOT / "sf-probe/data/fens_games_meta.tsv").read_text().split("\n")[1:] if l]
    assert [m[0] for m in meta] == fens
    elo = lambda m, b_turn: (float(m[3]), float(m[4])) if b_turn else (float(m[4]), float(m[3]))
    ex_g = [[int(m[2]), *(lambda s, o: (s + o, s - o))(*elo(m, chess.Board(f).turn))] for m, f in zip(meta, fens)]
    sets = {"games": (boards, np.load(LEELA_GAMES).astype(float), ex_g)}
    pf = [l for l in (ROOT / "sf-probe/data/fens.txt").read_text().split("\n") if l][:100000]
    wdl = np.load(ROOT / "lc0-probe/data/wdl.npy")[:100000]
    pm = [l.split("\t") for l in (ROOT / "sf-probe/data/meta.tsv").read_text().split("\n")[1:] if l][:100000]
    sets["puzzles"] = ([stm(f) for f in pf], wdl[:, 0] - wdl[:, 2], [[float(m[3]), chess.Board(f).fullmove_number] for m, f in zip(pm, pf)])
    for nm, (bs, y, ex) in sets.items():
        kinds, gid, X = design(bs, ex)
        D = np.array([[k == x for x in KS] for k in kinds], float)
        out = {"n": len(y), "kinds": dict(collections.Counter(kinds)),
               "raw_mean_v": {k: float(y[np.array(kinds) == k].mean()) for k in set(kinds)}}
        cf = [1, 2, 3, 4, 7, 8, 9, 10]                                       # c-f pawn counts per file: they define the kind
        specs = (("material_fe", D), ("material_fe+controls", np.hstack([D, np.delete(X, cf, 1)])),
                 ("material_fe+controls+cf_file_pawns", np.hstack([D, X])))      # last one over-controls
        for spec, Z in specs:
            b, se, C, n = fe_ols(y, Z, gid)
            o, c = KS.index("open"), KS.index("closed")
            out[spec] = {"n_used": n, "coef_vs_static": {k: [float(b[j]), float(se[j])] for j, k in enumerate(KS)},
                         "closed_minus_open": [float(b[c] - b[o]), float(np.sqrt(C[c, c] + C[o, o] - 2 * C[c, o]))]}
        res[nm] = out; print(nm, json.dumps({s: out[s]["closed_minus_open"] for s, _ in specs}), flush=True)
    json.dump(res, open(OUT("regress"), "w"), indent=1)

if _sf: _sf.quit()
