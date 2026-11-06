"""Serves the annotation page and saves annotations to annotate/data/annotations.jsonl.

  python annotate/server.py [--port 8765]

GET  /api/game[?id=ID]  a game (no id: a random one with no annotations), its annotations by ply and
                        Stockfish evals of every position, White's view: {cp} or {mate}, mate 0 = mated,
                        and pv, the best line in SAN
GET  /review            the agent run: cases from cases.py, tries from attempt.py
GET  /api/review        {cases, attempts, votes}: votes by "case/n": wrong, irrelevant (does not explain), partial or correct
GET  /api/schema        the DSL's types, read from dsl_verifier/src/dsl_claude.rs, for the editor
POST /api/annotation    {game, ply, fen, mov, dsl, ...}; appended, the latest per (game, ply) wins;
                        an empty dsl deletes
GET  /api/tempo         the tempo examples (data/tempo.jsonl) and my votes on them
POST /api/tempo/check   {fen, mov, dsl}: the native verifier's verdict
POST /api/tempo/vote    {id, vote}; appended to data/tempo_votes.jsonl, the latest wins
POST /api/vote          {case, round, n, vote: "wrong" | "irrelevant" | "partial" | "correct" | null}; appended to data/votes.jsonl, the latest wins
"""
import argparse, json, os, random, re, subprocess, sys, threading, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "docs/vendor"
DATA = HERE / os.environ.get("ANNOTATE_DATA", "data")  # another run: ANNOTATE_DATA=data_2800
GAMES = DATA / "games.jsonl"
ANNOTATIONS = DATA / "annotations.jsonl"
EVALS = DATA / "evals.jsonl"
CASES, ATTEMPTS = DATA / "cases.jsonl", DATA / "attempts.jsonl"  # the agent run, see attempt.py
VOTES = DATA / "votes.jsonl"  # my votes on the agent's tries
TEMPO, TEMPO_VOTES = HERE / "data/tempo.jsonl", HERE / "data/tempo_votes.jsonl"
VERIFIER = HERE.parent / "dsl_verifier/target/release/dsl_verifier"
DSL = HERE.parent / "dsl_verifier/src/dsl_claude.rs"
STOCKFISH = HERE.parent / "vendor/stockfish/stockfish-ubuntu-x86-64-avx2"
NODES = 300_000
sys.path.insert(0, str(HERE.parent / ".pylibs"))
import chess, chess.engine


def games():
    return {g["id"]: g for g in map(json.loads, GAMES.open())} if GAMES.exists() else {}


def annotations():
    """Latest annotation per game and ply."""
    out = {}
    if ANNOTATIONS.exists():
        for a in map(json.loads, ANNOTATIONS.open()):
            out.setdefault(a["game"], {})[a["ply"]] = a
    return {g: {p: a for p, a in by.items() if a["dsl"].strip()} for g, by in out.items()}


def split_top(s):
    """Splits at commas outside brackets."""
    out, depth, cur = [], 0, ""
    for c in s:
        depth += c in "(<[{"
        depth -= c in ")>]}"
        if c == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += c
    return [x for x in out + [cur.strip()] if x]


def ty(t):
    """A Rust argument type as the editor names it: Box<Reason> → Reason, Vec<Sq> → [Sq]."""
    t = t.strip()
    if m := re.match(r'#\[serde\((?:default, )?deserialize_with = "(\w+)"\)\]\s*(.*)', t):
        return "[Role]" if m[1] == "roles" else "Role"
    if m := re.match(r'#\[serde\(default\)\]\s*(.*)', t):
        return ty(m[1])
    if m := re.fullmatch(r"Box<(.*)>", t):
        return ty(m[1])
    if m := re.fullmatch(r"Vec<(.*)>", t):
        return f"[{ty(m[1])}]"
    if t.startswith("("):
        return "(" + ", ".join(ty(x) for x in split_top(t[1:-1])) + ")"
    return t


def schema():
    """{type: [{name, doc, group, args: [{name?, type}], fields: bool}]} for the DSL's enums."""
    out = {}
    for name, body in re.findall(r"pub\(crate\) enum (\w+) \{\n(.*?)\n\}", DSL.read_text(), re.S):
        # A struct variant over several lines, as one line.
        body = re.sub(r"\{\n(.*?)\n\s*\},", lambda m: "{ " + " ".join(m[1].split()) + " },", body, flags=re.S)
        vs, doc, group = [], [], 0
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                group += bool(vs)
            elif line.startswith("///"):
                doc.append(line[3:].strip())
            elif m := re.fullmatch(r"(\w+)\s*(?:\((.*)\)|\{(.*)\})?,?", line):
                if m[3] is not None:
                    # "#[serde(..)] get: Vec<Role>": the attribute goes with the type.
                    fs = [re.fullmatch(r"\s*(#\[.*?\]\s*)?(\w+)\s*:\s*(.*)", f, re.S) for f in split_top(m[3])]
                    args = [{"name": f[2], "type": ty((f[1] or "") + f[3])} for f in fs]
                else:
                    args = [{"type": ty(a)} for a in split_top(m[2] or "")]
                vs.append({"name": m[1], "doc": " ".join(doc).replace("`", ""), "group": group, "args": args, "fields": m[3] is not None})
                doc = []
        out[name] = vs
    return out


PV = 10  # plies of the principal variation kept
# Searched like dsl_verifier: 1 thread, default hash, cleared before each position, given as a bare FEN,
# NODES nodes. Same position, same numbers, so the annotator's grading and the verifier's Rated agree.
# Entries from an older search are evaluated again.
SEARCH = 3
_evals = {e["id"]: e["evals"] for e in map(json.loads, EVALS.open()) if e.get("search") == SEARCH} \
    if EVALS.exists() else {}
_engine, _lock = None, threading.Lock()


def search(engine, board):
    """One position as dsl_verifier searches it: White's view, {cp} or {mate}, and pv in SAN."""
    if board.is_checkmate():
        return {"mate": 0, "pv": []}
    if board.is_stalemate() or board.is_insufficient_material():
        return {"cp": 0, "pv": []}
    b = chess.Board(board.fen())  # no move history, as the verifier sends it
    # The last info line that is not a bound, as dsl_verifier reads it; new game: hash cleared.
    info = {}
    with engine.analysis(b, chess.engine.Limit(nodes=NODES), game=object()) as an:
        for i in an:
            if "score" in i and not i.get("upperbound") and not i.get("lowerbound"):
                info = i
    s, pv = info["score"].white(), []
    for mv in info.get("pv", [])[:PV]:
        pv.append(b.san(mv)); b.push(mv)
    return {"mate": s.mate(), "pv": pv} if s.is_mate() else {"cp": s.score(), "pv": pv}


def search_game(engine, moves):
    """Evals of the position before each move and after the last."""
    board, out = chess.Board(), [search(engine, chess.Board())]
    for m in moves:
        board.push_san(m)
        out.append(search(engine, board))
    return out


def evals(game):
    """Evals of the position before each move and after the last, computed once and cached."""
    global _engine
    if game["id"] in _evals:
        return _evals[game["id"]]
    with _lock:
        if game["id"] in _evals or not STOCKFISH.exists():
            return _evals.get(game["id"])
        if _engine is None:
            _engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
            _engine.configure({"Threads": 1})
        out = search_game(_engine, game["moves"])
        _evals[game["id"]] = out
        with EVALS.open("a") as f:
            f.write(json.dumps({"id": game["id"], "search": SEARCH, "evals": out}) + "\n")
        return out


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        p = urlparse(path).path
        if p.startswith("/lib/"):
            return str(HERE / "lib" / p[len("/lib/"):])
        if p.startswith("/vendor/"):
            return str(VENDOR / p[len("/vendor/"):])
        return str(HERE / {"/": "index.html", "/review": "review.html", "/tempo": "tempo.html"}.get(p, "_none"))

    def end_headers(self):
        # Revalidate every file, so a rebuilt wasm or edited page is picked up on reload.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/schema":
            return self.send_json(schema())
        if u.path == "/api/review":
            read = lambda f: [json.loads(l) for l in f.open()] if f.exists() else []
            votes = {f"{v['case']}/{v['round']}/{v['n']}" if v.get("round", 1) > 1 else f"{v['case']}/{v['n']}": v["vote"]
                     for v in read(VOTES)}
            return self.send_json({"cases": read(CASES), "attempts": read(ATTEMPTS),
                                   "votes": {k: v for k, v in votes.items() if v}})
        if u.path == "/api/tempo":
            read = lambda f: [json.loads(l) for l in f.open()] if f.exists() else []
            votes = {v["id"]: v["vote"] for v in read(TEMPO_VOTES)}
            return self.send_json({"examples": read(TEMPO), "votes": {k: v for k, v in votes.items() if v}})
        if u.path != "/api/game":
            return super().do_GET()
        gs, ann = games(), annotations()
        gid = parse_qs(u.query).get("id", [None])[0]
        if gid is None:
            fresh = [g for g in gs if g not in ann]
            if not fresh:
                return self.send_json({"error": "No unannotated games left. Run sample_games.py."}, 404)
            gid = random.choice(fresh)
        if gid not in gs:
            return self.send_json({"error": f"No game {gid}"}, 404)
        self.send_json({"game": gs[gid], "annotations": ann.get(gid, {}), "evals": evals(gs[gid]),
                        "left": sum(g not in ann for g in gs)})

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/annotation", "/api/vote", "/api/tempo/check", "/api/tempo/vote"):
            return self.send_json({"error": "not found"}, 404)
        a = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if path == "/api/tempo/check":
            r = subprocess.run([str(VERIFIER), "check"], input=json.dumps({k: a[k] for k in ("fen", "mov", "dsl")}),
                               capture_output=True, text=True, timeout=600)
            lines = r.stdout.strip().splitlines()
            return self.send_json(json.loads(lines[-1]) if lines else {"error": r.stderr.strip()[-500:]})
        if path == "/api/tempo/vote":
            if "id" not in a or a.get("vote") not in ("wrong", "irrelevant", "partial", "correct", None):
                return self.send_json({"error": "needs id, vote: wrong, irrelevant, partial, correct or null"}, 400)
            with TEMPO_VOTES.open("a") as f:
                f.write(json.dumps({"id": a["id"], "vote": a["vote"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            return self.send_json({"ok": True})
        if path == "/api/vote":
            if not {"case", "n", "vote"} <= a.keys() or a["vote"] not in ("wrong", "irrelevant", "partial", "correct", None):
                return self.send_json({"error": "needs case, n, vote: wrong, irrelevant, partial, correct or null"}, 400)
            with VOTES.open("a") as f:
                f.write(json.dumps({"case": a["case"], "round": a.get("round", 1), "n": a["n"], "vote": a["vote"],
                                    "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            return self.send_json({"ok": True})
        if not {"game", "ply", "fen", "mov", "dsl"} <= a.keys():
            return self.send_json({"error": "needs game, ply, fen, mov, dsl"}, 400)
        a["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with ANNOTATIONS.open("a") as f:
            f.write(json.dumps(a) + "\n")
        self.send_json({"ok": True})

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    print(f"http://localhost:{port}")
    # Evaluate the queue in the background so games open without waiting.
    threading.Thread(target=lambda: [evals(g) for g in games().values()], daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
