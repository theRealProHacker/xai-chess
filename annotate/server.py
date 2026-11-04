"""Serves the annotation page and saves annotations to annotate/data/annotations.jsonl.

  python annotate/server.py [--port 8765]

GET  /api/game[?id=ID]  a game (no id: a random one with no annotations), its annotations by ply and
                        Stockfish evals of every position, White's view: {cp} or {mate}, mate 0 = mated,
                        and pv, the best line in SAN
GET  /review            the agent run: cases from cases.py, tries from attempt.py
GET  /api/review        {cases, attempts}
GET  /api/schema        the DSL's types, read from dsl_verifier/src/dsl_claude.rs, for the editor
POST /api/annotation    {game, ply, fen, mov, dsl, ...}; appended, the latest per (game, ply) wins;
                        an empty dsl deletes
"""
import argparse, json, random, re, sys, threading, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "docs/vendor"
GAMES = HERE / "data/games.jsonl"
ANNOTATIONS = HERE / "data/annotations.jsonl"
EVALS = HERE / "data/evals.jsonl"
CASES, ATTEMPTS = HERE / "data/cases.jsonl", HERE / "data/attempts.jsonl"  # the agent run, see attempt.py
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
    if m := re.match(r'#\[serde\(deserialize_with = "(\w+)"\)\]\s*(.*)', t):
        return "[Role]" if m[1] == "roles" else "Role"
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
        vs, doc, group = [], [], 0
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                group += bool(vs)
            elif line.startswith("///"):
                doc.append(line[3:].strip())
            elif m := re.fullmatch(r"(\w+)\s*(?:\((.*)\)|\{(.*)\})?,?", line):
                if m[3] is not None:
                    args = [{"name": k.strip(), "type": ty(v)} for k, v in (f.split(":", 1) for f in split_top(m[3]))]
                else:
                    args = [{"type": ty(a)} for a in split_top(m[2] or "")]
                vs.append({"name": m[1], "doc": " ".join(doc).replace("`", ""), "group": group, "args": args, "fields": m[3] is not None})
                doc = []
        out[name] = vs
    return out


PV = 10  # plies of the principal variation kept
# Entries from before the PV was stored are evaluated again.
_evals = {e["id"]: e["evals"] for e in map(json.loads, EVALS.open()) if all("pv" in x for x in e["evals"])} \
    if EVALS.exists() else {}
_engine, _lock = None, threading.Lock()


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
            _engine.configure({"Threads": 4, "Hash": 128})
        board, out = chess.Board(), []
        for m in [None] + game["moves"]:
            if m:
                board.push_san(m)
            if board.is_checkmate():
                out.append({"mate": 0, "pv": []})
            elif board.is_game_over():
                out.append({"cp": 0, "pv": []})
            else:
                info = _engine.analyse(board, chess.engine.Limit(nodes=NODES))
                s, b, pv = info["score"].white(), board.copy(), []
                for mv in info.get("pv", [])[:PV]:
                    pv.append(b.san(mv)); b.push(mv)
                out.append({"mate": s.mate(), "pv": pv} if s.is_mate() else {"cp": s.score(), "pv": pv})
        _evals[game["id"]] = out
        with EVALS.open("a") as f:
            f.write(json.dumps({"id": game["id"], "evals": out}) + "\n")
        return out


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        p = urlparse(path).path
        if p.startswith("/lib/"):
            return str(HERE / "lib" / p[len("/lib/"):])
        if p.startswith("/vendor/"):
            return str(VENDOR / p[len("/vendor/"):])
        return str(HERE / {"/": "index.html", "/review": "review.html"}.get(p, "_none"))

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
            return self.send_json({"cases": read(CASES), "attempts": read(ATTEMPTS)})
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
        if urlparse(self.path).path != "/api/annotation":
            return self.send_json({"error": "not found"}, 404)
        a = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
