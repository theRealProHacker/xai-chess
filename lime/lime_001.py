"""LIME on Leela's raw-net Q for one position (see lime/PLAN.md).

Perturb by removing r and adding k pawns per side, ask Leela for Q on every perturbation, fit a 4-layer MLP
from the features/ bits to Q, read each family's relevance off the MLP by ablation.

  python lime/lime_001.py "<fen>" [--n 3000] [--out lime/out]
  python lime/lime_001.py --examples        # the book positions in features/examples.json
  python lime/lime_001.py --selftest
"""
import argparse, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / ".pylibs"), str(ROOT / "lc0-probe")]
import chess, numpy as np, onnxruntime as ort
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from lc0_planes import encode_fens

TOO_MANY = (chess.STATUS_TOO_MANY_WHITE_PAWNS | chess.STATUS_TOO_MANY_BLACK_PAWNS
            | chess.STATUS_TOO_MANY_WHITE_PIECES | chess.STATUS_TOO_MANY_BLACK_PIECES)
SEEDS = 5


# --- 1. perturb ---------------------------------------------------------------------------------

def tactical(b, sq, c):
    """A new pawn of colour c on sq that is one push from promotion or hits an enemy piece."""
    if chess.square_rank(sq) == (6 if c == chess.WHITE else 1):
        return True
    return any(b.color_at(t) == (not c) and b.piece_type_at(t) != chess.PAWN for t in b.attacks(sq))

def perturb(fen, n, rng):
    """n distinct FENs: the base first, then r white + r black original pawns removed (r in 0-2) and
    k white + k black pawns added on other empty squares of ranks 2-7 (k >= 1 when nothing is removed)."""
    base = chess.Board(fen)
    ok = TOO_MANY | base.status()
    out, seen, tries = [fen], {base.board_fen()}, 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        r = rng.integers(3)
        k = rng.geometric(1 / 3) - (r > 0)
        own = [list(base.pieces(chess.PAWN, c)) for c in (chess.WHITE, chess.BLACK)]
        empty = [s for s in chess.SQUARES if 1 <= chess.square_rank(s) <= 6 and base.piece_at(s) is None]
        if 2 * k > len(empty) or r > min(map(len, own)):
            continue
        sq = rng.choice(empty, 2 * k, replace=False)
        b = base.copy(stack=False)
        for p in own:
            for s in rng.choice(p, r, replace=False):
                b.remove_piece_at(int(s))
        for i, s in enumerate(sq):
            b.set_piece_at(int(s), chess.Piece(chess.PAWN, chess.WHITE if i < k else chess.BLACK))
        if b.status() & ~ok or b.is_check():
            continue
        if any(tactical(b, int(s), chess.WHITE if i < k else chess.BLACK) for i, s in enumerate(sq)):
            continue
        if b.board_fen() in seen:
            continue
        seen.add(b.board_fen()); out.append(b.fen())
    return out


# --- 2. Leela -----------------------------------------------------------------------------------

_sess = None
def leela_q(fens):
    """Raw-net Q = W - L, side to move's view."""
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(str(ROOT / "vendor/lc0-nets/t1-256x10.onnx"), providers=["CPUExecutionProvider"])
    q = []
    for s in range(0, len(fens), 1024):
        wdl = _sess.run(["/output/wdl"], {"/input/planes": encode_fens(fens[s:s + 1024])})[0]
        q.append(wdl[:, 0] - wdl[:, 2])
    return np.concatenate(q).astype(np.float64)


# --- 3. features --------------------------------------------------------------------------------

_fams = None
def family_ids():
    """families() ids from lib.rs, longest first so a prefix match finds the most specific."""
    global _fams
    if _fams is None:
        src = (ROOT / "features/src/lib.rs").read_text().split("pub fn families()")[1]
        _fams = sorted(re.findall(r'f\("([a-z_.]+)",', src), key=len, reverse=True)
    return _fams

def family_of(fid):
    return next(f for f in family_ids() if fid == f or fid.startswith(f + "."))

def features(fens):
    """Per FEN the CLI's feature list, with sides turned into us/them; None where the CLI errors."""
    run = subprocess.run(["cargo", "run", "--release", "-q", "--bin", "features", "--", "--all"],
                         cwd=ROOT / "features", input="\n".join(fens) + "\n", capture_output=True, text=True, check=True)
    rows = []
    for l in run.stdout.splitlines():
        r = json.loads(l)
        if "error" in r:
            rows.append(None); continue
        us = "white" if r["fen"].split()[1] == "w" else "black"
        for f in r["features"]:
            f["side"] = None if f["side"] is None else ("us" if f["side"] == us else "them")
            f["col"] = f["id"] if f["side"] is None else f"{f['id']}.{f['side']}"
        rows.append(r["features"])
    assert len(rows) == len(fens)
    return rows

def groups(cols):
    """Family key -> column indices. A family is split into .us/.them only when every bucket is sided;
    one with an unsided bucket (equal / plusN, imbalance 'other') is one family."""
    by = {}
    for i, (fid, side) in enumerate(cols):
        by.setdefault(family_of(fid), []).append((i, side))
    g = {}
    for fam, ms in by.items():
        if all(s is not None for _, s in ms):
            for side in ("us", "them"):
                g[f"{fam}.{side}"] = [i for i, s in ms if s == side]
        else:
            g[fam] = [i for i, _ in ms]
    return {k: v for k, v in g.items() if v}


# --- 4-5. surrogate and relevance ---------------------------------------------------------------

def ablate(Y, idx):
    """For one family's columns: the samples J whose state differs from the base, and the base row with
    the family's columns taken from each of them."""
    J = np.flatnonzero((Y[1:, idx] != Y[0, idx]).any(1)) + 1
    A = np.repeat(Y[:1], len(J), 0); A[:, idx] = Y[J][:, idx]
    return J, A

def fit(Y, q, g):
    n = len(q)
    te = np.random.default_rng(0).random(n) < 0.2; te[0] = False
    var = np.flatnonzero(Y.min(0) != Y.max(0))
    X = Y[:, var]
    ridge = Ridge(1.0).fit(X[~te], q[~te]).score(X[te], q[te])
    rel, mlp = {k: [] for k in g}, []
    for seed in range(SEEDS):
        m = MLPRegressor(hidden_layer_sizes=(64, 64, 64, 64), early_stopping=True, max_iter=500, random_state=seed)
        m.fit(X[~te], q[~te]); mlp.append(m.score(X[te], q[te]))
        f0 = m.predict(X[:1])[0]
        for k, idx in g.items():
            J, A = ablate(Y, idx)
            if len(J):
                rel[k].append(f0 - m.predict(A[:, var]).mean())
    return {"mlp": float(np.mean(mlp)), "mlp_spread": float(np.std(mlp)), "ridge": float(ridge)}, rel


def run(fen, n, rng, target=None, max_dq=None):
    """One position, end to end. `target` replaces Leela (selftest). `max_dq` drops samples whose Q moved more
    than that from the base; four times n are drawn so that n are left."""
    fens = perturb(fen, n if max_dq is None else 4 * n, rng)
    if max_dq is not None:
        dq = np.abs(leela_q(fens) - leela_q(fens[:1])[0])
        fens = [f for f, d in zip(fens, dq) if d <= max_dq][:n]
    rows = features(fens)
    if rows[0] is None:
        raise SystemExit(f"features/ rejects the base position: {fen}")
    keep = [i for i, r in enumerate(rows) if r is not None]
    fens, rows = [fens[i] for i in keep], [rows[i] for i in keep]
    cols = sorted({(f["id"], f["side"]) for r in rows for f in r if family_of(f["id"]) != "initiative"},
                  key=lambda c: (c[0], c[1] or ""))
    ci = {c: i for i, c in enumerate(cols)}
    Y = np.zeros((len(rows), len(cols)), np.float32)
    for j, r in enumerate(rows):
        for f in r:
            if f["on"] and (f["id"], f["side"]) in ci:
                Y[j, ci[(f["id"], f["side"])]] = 1
    q = leela_q(fens) if target is None else target(Y, cols)
    g = groups(cols)
    fid, rel = fit(Y, q, g)

    base = {(f["id"], f["side"]): f for f in rows[0]}
    fams = []
    for k, idx in g.items():
        split = k.endswith((".us", ".them"))
        on = [base[cols[i]] for i in idx if base[cols[i]]["on"]]
        J, _ = ablate(Y, idx)
        e = {"id": k.rsplit(".", 1)[0] if split else k, "side": k.rsplit(".", 1)[1] if split else None,
             "state": [f["id"] if split else f["col"] for f in on],
             "squares": sorted({s for f in on for s in f["squares"]}),
             "why": " ".join(dict.fromkeys(f["why"] for f in (on or [base[cols[idx[0]]]])))}
        if len(J):
            r = np.array(rel[k])
            e.update(relevance=float(r.mean()), spread=float(r.std()),
                     sign_stable=bool((r > 0).all() or (r < 0).all()), varied=len(J) / len(fens))
        else:
            e.update(relevance=None, varied=False)
        fams.append(e)
    fams.sort(key=lambda e: -abs(e["relevance"] or 0))
    pawns = lambda f, c: chess.Board(f).pieces(chess.PAWN, c)
    over8 = np.mean([len(pawns(f, chess.WHITE)) > 8 or len(pawns(f, chess.BLACK)) > 8 for f in fens])
    return {"fen": fen, "q": float(q[0]), "n": len(fens), "over8": float(over8), "fidelity": fid, "families": fams}, (fens, Y, cols, g)


def slug(fen):
    return "_".join(fen.split()[:2]).replace("/", "-")

def save(res, out):
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{slug(res['fen'])}.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return p


# --- 6. selftest --------------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(1)
    fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2p5/2N1PN2/PP3PPP/R2QKB1R b KQ - 0 8"
    base = chess.Board(fen)
    fens = perturb(fen, 400, rng)
    assert fens[0] == fen and len(set(fens)) == len(fens)
    removed = 0
    for f in fens[1:]:
        b = chess.Board(f)
        gone = [s for s, p in base.piece_map().items() if b.piece_at(s) != p]
        assert all(base.piece_type_at(s) == chess.PAWN and b.piece_at(s) is None for s in gone), f  # only pawns go
        assert len(gone) <= 4 and sum(base.color_at(s) == chess.WHITE for s in gone) * 2 == len(gone), f
        removed += bool(gone)
        new = [s for s in chess.SQUARES if b.piece_at(s) and not base.piece_at(s)]
        assert all(b.piece_type_at(s) == chess.PAWN and 1 <= chess.square_rank(s) <= 6 for s in new), f
        assert sum(b.color_at(s) == chess.WHITE for s in new) * 2 == len(new), f       # k white, k black
        assert not b.is_check() and not any(tactical(b, s, b.color_at(s)) for s in new), f
        assert gone or new, f
    assert removed > len(fens) / 3, removed
    # a double check by two pawns: python-chess passes it, the CLI errors, the row is dropped
    bad = "4k3/8/8/8/8/8/3p1p2/4K3 w - - 0 1"
    assert features([fen, bad])[1] is None

    # synthetic target on one on-bucket of a relative family, black to move
    rows = features([fen])[0]
    pick = next(f["col"] for f in rows if f["on"] and f["side"] == "us" and family_of(f["id"]) == "space_pawns")
    def target(Y, cols):
        return 0.3 * Y[:, [i for i, (fid, s) in enumerate(cols) if f"{fid}.{s}" == pick][0]]
    res, (fens, Y, cols, g) = run(fen, 1500, rng, target)
    top = res["families"][0]
    assert top["id"] == "space_pawns" and top["relevance"] > 0, (pick, res["families"][:3])
    # ablated rows stay one-hot wherever the samples are one-hot
    for k, idx in g.items():
        _, A = ablate(Y, idx)
        prefix = {}
        for i in idx:
            prefix.setdefault((cols[i][0].rsplit(".", 1)[0], cols[i][1]), []).append(i)
        for p in prefix.values():
            if (Y[:, p].sum(1) == 1).all():
                assert (A[:, p].sum(1) == 1).all(), (k, p)
    print(f"selftest ok: {pick} -> top {top['id']} {top['relevance']:+.3f}, fidelity {res['fidelity']}")


# --- main ---------------------------------------------------------------------------------------

def examples(n, out):
    cases = json.loads((ROOT / "features/examples.json").read_text())
    by = {}
    for c in cases:
        if c.get("fen"):
            by.setdefault(c["fen"], []).append(c)
    index = []
    for i, (fen, cs) in enumerate(by.items()):
        res, _ = run(fen, n, np.random.default_rng(0))
        p = save(res, out)
        print(f"{i + 1}/{len(by)} {fen}  mlp {res['fidelity']['mlp']:.2f} ridge {res['fidelity']['ridge']:.2f}", flush=True)
        index.append({"fen": fen, "game": cs[0].get("game"), "move": cs[0].get("move"), "path": p.name,
                      "fidelity": res["fidelity"], "over8": res["over8"],
                      "cases": [{k: c.get(k) for k in ("quote", "source", "family", "feature", "side", "case")} for c in cs]})
        (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fen", nargs="?")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "lime" / "out"))
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--max-dq", type=float, help="drop samples whose Leela Q moved more than this from the base")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.examples:
        examples(a.n, Path(a.out))
    elif a.fen:
        res, _ = run(a.fen, a.n, np.random.default_rng(0), max_dq=a.max_dq)
        res["max_dq"] = a.max_dq
        print(save(res, Path(a.out)))
        print(f"q {res['q']:+.3f}  n {res['n']}  over8 {res['over8']:.2f}  fidelity {res['fidelity']}")
        for e in res["families"][:10]:
            print(f"  {e['id']:28} {e['side'] or '':5} {e['relevance']:+.3f} ±{e['spread']:.3f}  varied {e['varied']:.2f}"
                  f"{'' if e['sign_stable'] else '  sign unstable'}  {e['state']}")
    else:
        ap.error("give a FEN, --examples or --selftest")
