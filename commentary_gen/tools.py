"""Callable board tools for function calling. Bound to one position; the model never passes a FEN.

    tb = ToolBox(fen_after_move)           # all eight
    tb = ToolBox(fen, ("after", "legal", "attack_map"))   # only these
    tb.declarations()          -> Gemini functionDeclarations
    tb.call(name, args)        -> str (never raises)

Since round 6 everything that needs no argument from the model is injected by facts.py instead, and
only the three argument-taking lookups are left callable. A move for the side that is not to move is
answered on the flipped position rather than refused: that was 71 of the failed calls in round 5.
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
    {"name": "after", "description": "Play a line from the current position and report what it reaches: material, "
             "hanging pieces, checks and captures. Use it before claiming that a line works, a move is possible, "
             "a piece is trapped or a plan gets somewhere. A line for the side that is not to move is played anyway, "
             "with a note saying so.",
     "parameters": {"type": "OBJECT", "properties": {"moves": {"type": "STRING", "description": "SAN moves separated by spaces, e.g. 'Nxe5 Nxe5 Qxe5'"}},
                    "required": ["moves"]}},
    {"name": "pawn_structure", "description": "Isolated, doubled, passed and backward pawns for both sides, "
             "open/half-open/closed files, holes.",
     "parameters": {"type": "OBJECT", "properties": {}}},
]


class ToolBox:
    def __init__(self, fen, names=None):
        self.board = chess.Board(fen)
        self.names = names

    def declarations(self):
        return [d for d in DECL if not self.names or d["name"] in self.names]

    def call(self, name, args):
        b = self.board
        try:
            if name == "attack_map":
                sq = str(args.get("square", "")).strip().lower()
                if not chess.SQUARE_NAMES.count(sq):
                    return (f"Error: {sq!r} is not a square. Give one square as a file and a rank, "
                            f"like e4 or d5 — no piece letter. The call did not happen; call again.")
                return bt.attack_map(b, sq)
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
                return self._after(str(args.get("moves", "")))
            if name == "pawn_structure":
                return bt.pawn_structure(b)
            return (f"Error: there is no tool called {name}. The tools are: "
                    f"{', '.join(d['name'] for d in self.declarations())}. Call one of those.")
        except Exception as e:  # tools must never break the generation loop
            return f"Error: {e}. The call did not happen; fix the argument and call again."

    def _after(self, moves):
        """Play a line. If the first move belongs to the other side, play it for them and say so."""
        b, note = self.board.copy(stack=False), ""
        sans = [s.strip(",;") for s in moves.split() if not s[0].isdigit()]
        if not sans:
            return "No moves given."
        try:
            b.parse_san(sans[0])
        except ValueError:
            other = bt._as_mover(b, not b.turn)
            try:
                other.parse_san(sans[0])
                note = (f"It is {bt._side(b.turn)}'s move; playing this line for "
                        f"{bt._side(not b.turn)} as if they had a free move.\n")
                b = other
            except ValueError:
                pass
        played = []
        for san in sans:
            try:
                b.push_san(san)
                played.append(san)
            except ValueError:
                return note + f"After {' '.join(played) or 'nothing'}: {bt.legal(b, san)}"
        return note + f"After {' '.join(played)}:\n{bt.report(b)}"


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
