"""Callable board tools for function calling. Bound to one position; the model never passes a FEN.

    tb = ToolBox(fen_after_move)
    tb.declarations()          -> Gemini functionDeclarations
    tb.call(name, args)        -> str (never raises)
"""
import chess
import board_tools as bt

DECL = [
    {"name": "attack_map", "description": "For one square: which piece stands there, every square it reaches "
             "(blockers respected), what it attacks and defends, and which pieces attack or defend it.",
     "parameters": {"type": "OBJECT", "properties": {"square": {"type": "STRING", "description": "e.g. e4"}},
                    "required": ["square"]}},
    {"name": "inventory", "description": "All pieces of both sides by square, material count, king squares, side to move.",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "hanging", "description": "Pieces the opponent can capture for a net material gain, with the capture, "
             "attackers and defenders. Pin-aware.",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "forcing", "description": "Every check and capture available to the side to move, with the net "
             "material after the exchange on that square (negative = loses material).",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "threats", "description": "What the side that just moved threatens: its checks and captures if it "
             "were to move again, with net material.",
     "parameters": {"type": "OBJECT", "properties": {}}},
    {"name": "legal", "description": "Is this move legal for the side to move? If not, why (blocked, pinned, "
             "wrong side, piece does not reach).",
     "parameters": {"type": "OBJECT", "properties": {"move": {"type": "STRING", "description": "SAN, e.g. Nxe5 or O-O"}},
                    "required": ["move"]}},
    {"name": "after", "description": "Play a sequence of moves from the current position and report the resulting "
             "position: material, hanging pieces, checks and captures. Use to check a line.",
     "parameters": {"type": "OBJECT", "properties": {"moves": {"type": "STRING", "description": "SAN moves separated by spaces, e.g. 'Nxe5 Nxe5 Qxe5'"}},
                    "required": ["moves"]}},
    {"name": "pawn_structure", "description": "Isolated, doubled, passed and backward pawns for both sides, "
             "open/half-open/closed files, holes.",
     "parameters": {"type": "OBJECT", "properties": {}}},
]


class ToolBox:
    def __init__(self, fen):
        self.board = chess.Board(fen)

    def declarations(self):
        return DECL

    def call(self, name, args):
        b = self.board
        try:
            if name == "attack_map":
                return bt.attack_map(b, str(args.get("square", "")).strip().lower())
            if name == "inventory":
                return bt.inventory(b)
            if name == "hanging":
                return bt.hanging(b)
            if name == "forcing":
                return bt.forcing(b)
            if name == "threats":
                return bt.forcing(bt._as_mover(b, not b.turn))
            if name == "legal":
                return bt.legal(b, str(args.get("move", "")))
            if name == "after":
                b2 = b.copy(stack=False)
                played = []
                for san in str(args.get("moves", "")).split():
                    try:
                        b2.push_san(san.strip(",;"))
                        played.append(san)
                    except ValueError:
                        return (f"After {' '.join(played) or 'nothing'}: {bt.legal(b2, san)}")
                return f"After {' '.join(played)}:\n{bt.report(b2)}"
            if name == "pawn_structure":
                return bt.pawn_structure(b)
            return f"Error: unknown tool {name}"
        except Exception as e:  # tools must never break the generation loop
            return f"Error: {e}"


class CodeToolBox:
    """One tool: run_python(code) in sandbox.Session; the call record carries the functions the cell used."""
    DECL = [{"name": "run_python", "description": "Run Python in a session where `board` is the current position "
             "(python-chess Board, after the last move), the board tools are imported, and docs(name) shows "
             "documentation. Variables persist between calls. print() what you want to see.",
             "parameters": {"type": "OBJECT", "properties": {"code": {"type": "STRING"}}, "required": ["code"]}}]

    def __init__(self, fen):
        import sandbox
        self.session = sandbox.Session(fen)
        self.last_functions = {}

    def declarations(self):
        return self.DECL

    def call(self, name, args):
        if name != "run_python":
            return f"Error: unknown tool {name}"
        out, self.last_functions = self.session.run(str(args.get("code", "")))
        return out
