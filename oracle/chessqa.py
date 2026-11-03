"""ChessQA structural tasks (Wen, Tang & Anderson 2025, arXiv:2510.23948): prompts verbatim, answers rebuilt with
python-chess. The prompt keeps the FEN; the model also gets the position's Leela tokens.

Prompts are ChessQA's with CONTEXT_PLACEHOLDER empty (their add_context=False) and FORMAT_EXAMPLE_PLACEHOLDER the
first format example. The target is "FINAL ANSWER: <answer>" with no reasoning. Scoring follows ChessQA: the text after
the last "FINAL ANSWER:", exact match for single answers, set match on comma-separated items for multi.

  sample(task, rng, fen, moves=None) -> row (task, question, answer, target, answer_type) or None
  score(row, text) -> {parsed, exact, f1 (multi) / fen4 (state tracking: first four FEN fields)}
  python oracle/chessqa.py CHESSQA_JSON   checks every gold answer of the benchmark against this file's rules
"""
import json, re, sys
import chess

TASKS = ["cqa_arrangement", "cqa_legal_piece", "cqa_legal_all", "cqa_check_det", "cqa_check_in_1", "cqa_capture",
         "cqa_control", "cqa_protect", "cqa_track_short", "cqa_track_mid", "cqa_track_long"]
CHESSQA = {"cqa_arrangement": "structural_piece_arrangement", "cqa_legal_piece": "structural_legal_move_piece",
           "cqa_legal_all": "structural_legal_move_all", "cqa_check_det": "structural_check_detection",
           "cqa_check_in_1": "structural_check_in_1", "cqa_capture": "structural_capture_squares",
           "cqa_control": "structural_control_squares", "cqa_protect": "structural_protect_squares",
           "cqa_track_short": "structural_state_tracking_short", "cqa_track_mid": "structural_state_tracking_mid",
           "cqa_track_long": "structural_state_tracking_long"}
TRACK = {"cqa_track_short": (1, 5), "cqa_track_mid": (6, 10), "cqa_track_long": (11, 15)}
SINGLE = {"cqa_arrangement", *TRACK}
NAME = {chess.KING: "King", chess.QUEEN: "Queen", chess.ROOK: "Rook", chess.BISHOP: "Bishop", chess.KNIGHT: "Knight",
        chess.PAWN: "Pawn"}
COLOUR = {chess.WHITE: "White", chess.BLACK: "Black"}

TAIL = ("Analyze step by step and explain your reasoning.\nFinish with a single line formatted EXACTLY as:\n"
        "FINAL ANSWER: <answer>\n")
EX = "Example final answer: {ex}"
UCI_LIST = "List the moves in UCI format, separated by commas and spaces.\n"
ARRANGE_EX = ("White King: ['e1'], White Queen: ['d1'], White Rook: ['a1', 'h1'], White Bishop: ['c1', 'f1'], "
              "White Knight: ['b1', 'g1'], White Pawn: ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'], "
              "Black King: ['e8'], Black Queen: ['d8'], Black Rook: ['a8', 'h8'], Black Bishop: ['c8', 'f8'], "
              "Black Knight: ['b8', 'g8'], Black Pawn: ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7']")


def head(fen):
    return f"You are given a chess position in FEN: {fen}.\n"


def question(task, fen, sq=None, piece=None, moves=None):
    if task == "cqa_arrangement":
        return (head(fen) + "Provide the complete piece arrangement of this position. List all pieces with their colors, "
                "types, and squares.\nFormat: 'Color PieceType: [square1, square2, ...]', separated by commas and spaces "
                "for different piece types.\nList the pieces in the order of White pieces first (King, Queen, Rook, "
                "Bishop, Knight, Pawn) followed by Black pieces (King, Queen, Rook, Bishop, Knight, Pawn).\nIf a piece "
                "type has no pieces on the board, skip it in the listing.\nList the squares for each piece type in "
                "alphabetical order.\n" + TAIL + EX.format(ex=ARRANGE_EX))
    if task == "cqa_legal_piece":
        return head(fen) + f"Find all legal moves for the piece on square {sq}. " + UCI_LIST + TAIL + EX.format(ex="e2e3, e2e4")
    if task == "cqa_legal_all":
        return head(fen) + "Find all legal moves in this position. " + UCI_LIST + TAIL + EX.format(ex="e2e4, c2b1q")
    if task == "cqa_check_in_1":
        return head(fen) + "Find all moves that put the opponent in check. " + UCI_LIST + TAIL + EX.format(ex="e2e4, c2b1q")
    if task == "cqa_check_det":
        return (head(fen) + "In this position, the side to move is in check. Identify the piece(s) that is delivering the "
                "check.\n" + TAIL + "List each checking piece with its color, type, and square (e.g., White Queen at e5). "
                "Separate multiple pieces with commas and spaces if applicable.")
    p = f"{piece} on {sq}"
    if task == "cqa_capture":
        body = (f"Find all squares that the {p} can capture (i.e. every square that has an opponent piece such that the "
                f"{p} could legally move to that square and capture the piece).\nExclude captures if the {p} is pinned to "
                "its king and thus cannot move.\n")
    elif task == "cqa_control":
        body = (f"Find all squares that the {p} controls (i.e. every empty square that the {p} could legally move to, "
                f"excluding squares occupied by any piece).\nExclude control if the {p} is pinned to its king and thus "
                "cannot move.\n")
    elif task == "cqa_protect":
        body = (f"Find all squares that contain pieces that the {p} protects (i.e. every square that contains a piece such "
                f"that the {p} could legally recapture if an enemy piece captured it, excluding the king since it can't "
                f"be captured).\nExclude protection if the {p} is pinned to its king and thus cannot move.\n")
    else:
        return ("Given an initial FEN and a sequence of UCI moves, apply the moves in order and output the exact resulting "
                f"FEN.\nInitial FEN: {fen}\nMoves (UCI): {' '.join(moves)}\n" + TAIL
                + EX.format(ex="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    return head(fen) + body + TAIL + EX.format(ex="e4, f5")


# ---- answers ----
def names(sqs):
    return sorted(chess.square_name(s) for s in sqs)


def arrangement(b):
    parts = []
    for c in (chess.WHITE, chess.BLACK):
        for k in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            sq = names(b.pieces(k, c))
            if sq: parts.append(f"{COLOUR[c]} {NAME[k]}: [{', '.join(repr(s) for s in sq)}]")
    return ", ".join(parts)


def piece_squares(task, b, s):
    """capture / control / protect squares of the (non-pawn, non-king) piece on s; empty if it is pinned."""
    c = b.color_at(s)
    if b.is_pinned(c, s): return []
    at = b.attacks(s)
    if task == "cqa_control": return names(at & ~b.occupied)
    if task == "cqa_capture": return names(at & b.occupied_co[not c] & ~b.kings)
    return names(at & b.occupied_co[c] & ~b.kings)


def answer(task, b, sq=None, moves=None):
    if task == "cqa_arrangement": return arrangement(b)
    if task == "cqa_legal_piece":
        return sorted(m.uci() for m in b.legal_moves if m.from_square == sq)
    if task == "cqa_legal_all": return sorted(m.uci() for m in b.legal_moves)
    if task == "cqa_check_in_1": return sorted(m.uci() for m in b.legal_moves if b.gives_check(m))
    if task == "cqa_check_det":
        return [f"{COLOUR[b.color_at(s)]} {NAME[b.piece_type_at(s)]} at {chess.square_name(s)}" for s in b.checkers()]
    if task in TRACK:
        a = b.copy(stack=False)
        for u in moves: a.push_uci(u)
        return a.fen()
    return piece_squares(task, b, sq)


def text(task, ans):
    return ans if task in SINGLE else ", ".join(ans)


# ---- sampling ----
def sample(task, rng, fen, moves=None):
    """One ChessQA-format row for this position, or None when ChessQA would skip it (empty answer, no fitting piece,
    game too short). Like ChessQA, every row has a non-empty answer; check detection only in check."""
    b = chess.Board(fen)
    row = dict(task=task, fen=fen, answer_type="single" if task in SINGLE else "multi")
    sq = piece = mv = None
    if task in TRACK:
        lo, hi = TRACK[task]
        if not moves or len(moves) < lo: return None
        mv = list(moves[:int(rng.integers(lo, min(hi, len(moves)) + 1))])
    elif task == "cqa_legal_piece":
        frm = sorted({m.from_square for m in b.legal_moves})
        if not frm: return None
        sq = frm[int(rng.integers(len(frm)))]
    elif task in ("cqa_capture", "cqa_control", "cqa_protect"):
        ok = [s for s in b.piece_map() if b.piece_type_at(s) not in (chess.PAWN, chess.KING) and piece_squares(task, b, s)]
        if not ok: return None
        sq = ok[int(rng.integers(len(ok)))]
        piece = f"{COLOUR[b.color_at(sq)]} {NAME[b.piece_type_at(sq)]}"
    elif task == "cqa_check_det" and not b.is_check():
        return None
    ans = answer(task, b, sq, mv)
    if not ans: return None
    t = text(task, ans)
    s = chess.square_name(sq) if sq is not None else None
    return dict(row, question=question(task, fen, s, piece, mv), target=t, answer="FINAL ANSWER: " + t,
                **({"moves": mv} if mv else {}))


# ---- scoring (ChessQA: last FINAL ANSWER line; exact for single, set match for multi) ----
_FA = re.compile(r"FINAL ANSWER:\s*(.*)")


def extract(t):
    m = _FA.findall(t)
    return m[-1].strip() if m else None


def items(t):
    return {x.strip() for x in t.split(",") if x.strip()}


def score(row, t):
    p = extract(t)
    if p is None: return dict(parsed=False, exact=False)
    gold = row["target"]
    if row["answer_type"] == "single":
        out = dict(parsed=True, exact=p == gold)
        if row["task"] in TRACK: out["fen4"] = p.split()[:4] == gold.split()[:4]
        return out
    ps, gs = items(p), items(gold)
    tp = len(ps & gs)
    return dict(parsed=True, exact=ps == gs, f1=2 * tp / (len(ps) + len(gs)) if tp else 0.0)


def from_chessqa(r):
    """A ChessQA structural item as a row: our prompt (placeholders resolved), its gold answer."""
    task = {v: k for k, v in CHESSQA.items()}[r["task_type"]]
    m = json.loads(r["metadata_json"])
    fen, *mv = r["input"].split(" | ")
    mv = mv[0].split() if mv else None
    sq, piece = m.get("target_square"), m.get("piece")
    gold = r["correct_answer"]
    return dict(task=task, fen=fen, answer_type=r["answer_type"], question=question(task, fen, sq, piece, mv),
                target=gold, answer="FINAL ANSWER: " + gold, chessqa_id=r["task_id"], **({"moves": mv} if mv else {}))


if __name__ == "__main__":
    bad = 0
    for r in json.load(open(sys.argv[1])):
        row = from_chessqa(r)
        b = chess.Board(row["fen"])
        sq = chess.parse_square(json.loads(r["metadata_json"])["target_square"]) if "target_square" in r["metadata_json"] else None
        mine = text(row["task"], answer(row["task"], b, sq, row.get("moves")))
        want = r["question"].replace("CONTEXT_PLACEHOLDER", "").replace(
            "FORMAT_EXAMPLE_PLACEHOLDER", r["format_examples"][0])
        ok_a = score(row, "FINAL ANSWER: " + mine)["exact"] and (row["answer_type"] == "multi" or mine == r["correct_answer"])
        ok_q = row["question"] == want
        if not (ok_a and ok_q):
            bad += 1
            if bad <= 8: print(r["task_id"], "answer" if not ok_a else "", "question" if not ok_q else "",
                               "\n  gold:", r["correct_answer"][:150], "\n  mine:", mine[:150])
    print("mismatches:", bad)
