"""Run model-written Python against a position, with a call logger on board_tools and chess.Board.

    s = Session(fen); out, calls = s.run("print(hanging(board))")
Cells persist: every run re-executes all earlier cells (they are short) in a fresh `python -I`
process with CPU/memory limits. `calls` counts the tool and Board methods the cell used.
"""
import json, resource, site, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).parent
BOARD_METHODS = ("piece_at", "attackers", "attacks", "is_pinned", "is_attacked_by", "push_san", "push", "pop",
                 "san", "parse_san", "is_check", "gives_check", "is_capture", "is_checkmate", "is_legal", "copy")
TOOLS = ("attack_map", "inventory", "hanging", "forcing", "legal", "pawn_structure", "report", "report_after_move", "see")
LIMIT = 2000

PRELUDE = r'''
import sys; sys.path[:0] = %(paths)r
import json, functools, pydoc, chess, board_tools
_LOG = {}
def _wrap(obj, name, label):
    f = getattr(obj, name)
    @functools.wraps(f)
    def w(*a, **k):
        if sys._getframe(1).f_code.co_filename == "<cell>":  # only calls written by the model
            _LOG[label] = _LOG.get(label, 0) + 1
        return f(*a, **k)
    setattr(obj, name, w)
for _n in %(tools)r: _wrap(board_tools, _n, _n)
for _n in %(methods)r: _wrap(chess.Board, _n, "Board." + _n)
from board_tools import *
board = chess.Board(%(fen)r)
def mover(b=None):
    """A copy of the position with the side that just moved to move again; push its moves on it to test a threat."""
    b = b or board
    return board_tools._as_mover(b, not b.turn)
def threats(b=None):
    """Checks and captures for the side that just moved, as if it could move again."""
    return forcing(mover(b))
_wrap(sys.modules[__name__], "threats", "threats"); _wrap(sys.modules[__name__], "mover", "mover")
def docs(name=None):
    """docs() lists the tools; docs('chess.Board.attackers') shows any symbol's documentation."""
    if name is None:
        return "\n".join(f"{n}: {(getattr(board_tools, n).__doc__ or '').strip().splitlines()[0] if getattr(board_tools, n).__doc__ else ''}" for n in %(tools)r) + "\nthreats(): " + threats.__doc__ + "\nmover(): " + mover.__doc__
    return pydoc.render_doc(pydoc.locate(name) or name, renderer=pydoc.plaintext)
_cells = json.loads(sys.stdin.read())
import io, contextlib, traceback
_out = io.StringIO()
for _i, _c in enumerate(_cells):
    _last = _i == len(_cells) - 1
    _buf = _out if _last else io.StringIO()
    with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
        try:
            exec(compile(_c, "<cell>", "exec"), globals())
        except Exception:
            if _last: traceback.print_exc(limit=1, file=_buf)
    if not _last: _LOG.clear()
sys.stdout.write(_out.getvalue()[:%(limit)d])
sys.stdout.write("\n\x00" + json.dumps(_LOG))
'''


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (768 << 20, 768 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class Session:
    def __init__(self, fen):
        self.fen, self.cells = fen, []
        self.tmp = tempfile.mkdtemp(prefix="sbx_")
        sitedirs = [d for d in [site.getusersitepackages(), *site.getsitepackages()] if Path(d, "chess").exists()]
        self.paths = [str(HERE), *sitedirs]  # -I ignores PYTHONPATH, so the prelude extends sys.path

    def run(self, code):
        prelude = PRELUDE % {"tools": TOOLS, "methods": BOARD_METHODS, "fen": self.fen, "limit": LIMIT, "paths": self.paths}
        try:
            p = subprocess.run([sys.executable, "-I", "-c", prelude], input=json.dumps(self.cells + [code]),
                               capture_output=True, text=True, timeout=8, cwd=self.tmp, env={"PYTHONDONTWRITEBYTECODE": "1"}, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return "Error: cell exceeded 5 s of CPU time", {}
        out, _, log = p.stdout.rpartition("\n\x00")
        if p.returncode != 0 and not log:
            return "Error: " + (p.stderr.strip().splitlines() or ["sandbox failed"])[-1][:LIMIT], {}
        self.cells.append(code)
        calls = json.loads(log) if log else {}
        return (out.strip() or "(no output)"), calls


if __name__ == "__main__":  # self-check
    s = Session("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    out, calls = s.run("x = 1\nprint(hanging(board)); print(len(list(board.legal_moves)))")
    assert calls == {"hanging": 1} and "No piece is hanging" in out and "27" in out, (out, calls)
    out, calls = s.run("print(x); board.push_san('Nxe5'); print(board.attackers(chess.BLACK, chess.E5))")
    assert out.startswith("1") and calls == {"Board.push_san": 1, "Board.attackers": 1}, (out, calls)
    out, calls = s.run("1/0")
    assert "ZeroDivisionError" in out, out
    out, calls = s.run("print(docs('chess.Board.attackers')[:60])")
    assert "attackers" in out, out
    out, _ = s.run("while True: pass")
    assert out.startswith("Error"), out
    print("ok")
