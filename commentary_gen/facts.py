"""The fact sheet: everything about the position that can be computed, numbered, in reading order.

Round 5 gave the model eight functions and it still contradicted its own results in 88 of 300 items.
Here nothing derivable is left to the model: every lookup that does not need an argument the model
has to invent is computed here and injected, one numbered line per fact, ordered most game-deciding
first. The model selects facts and cites their ids; verify.py checks the note against the board.

    sheet(board_before, san) -> (text, index)      index: id -> {"section", "text", "squares"}

CLI:  python facts.py "<fen_before>" <san>
"""
import re
import sys
import chess
import board_tools as bt
import positional as ps

SQ = re.compile(r"\b[a-h][1-8]\b")


class Sheet:
    def __init__(self):
        self.n, self.lines, self.index, self.section = 0, [], {}, ""

    def head(self, title):
        self.section = title
        self.lines.append("")
        self.lines.append(title)

    def add(self, text, src=None):
        for t in str(text).split("\n"):
            t = t.strip()
            if not t:
                continue
            if t.endswith(":"):  # a lead-in, not a fact
                self.lines.append(t)
                continue
            self.n += 1
            fid = f"F{self.n}"
            self.lines.append(f"{fid}  {t}")
            self.index[fid] = {"section": self.section, "src": src or self.section,
                               "text": t, "squares": SQ.findall(t)}

    def text(self):
        return "\n".join(self.lines).strip()


def _status(board):
    n = board.legal_moves.count()
    if board.is_checkmate():
        return f"CHECKMATE: {bt._side(not board.turn)} has won; {bt._side(board.turn)} has no legal move."
    if board.is_stalemate():
        return "Stalemate: the side to move has no legal move and is not in check. The game is drawn."
    if board.is_check():
        return (f"{bt._side(board.turn)} is in check from {bt._sqs(board.checkers())} and must answer it; "
                f"everything else waits. {n} legal repl{'y' if n == 1 else 'ies'}.")
    return f"{bt._side(board.turn)} to move, not in check, with {n} legal moves."


def _replies(board, limit=14):
    moves = list(board.legal_moves)
    shown = " ".join(board.san(m) for m in moves[:limit])
    if len(moves) == 1:
        return f"Only one legal move: {shown}. It is forced."
    return (f"The {len(moves)} legal replies are: {shown}"
            + (f" and {len(moves) - limit} more." if len(moves) > limit else "."))


def _exchange(before, after, san, s):
    """A capture leaves the count mid-exchange; state what it comes to once the square is settled."""
    m = before.parse_san(san)
    if not before.is_capture(m) or after.is_checkmate():
        return None
    net = bt.see(before, m)
    if bt._see_square(after, m.to_square) == 0:
        return None  # nothing can take back on that square: the count above is already final
    mover = before.turn
    diff = bt.inventory_data(after)["material_diff_white_minus_black"]
    diff += (-1 if mover == chess.WHITE else 1) * (bt._gain_of(before, m) - net)
    state = "equal" if diff == 0 else f"{'White' if diff > 0 else 'Black'} up {abs(diff)}"
    s.add(f"{san} is a capture and the exchange on {chess.square_name(m.to_square)} is not finished, so the "
          f"count above is mid-exchange. Counted over the whole exchange, {san} is "
          f"{bt.trade_words(net)} for {bt._side(mover)}; once it is settled, material is {state}.", "inventory")
    return {"square": chess.square_name(m.to_square), "move": san, "net": net}


def sheet(board_before, san, pieces_both_sides=True):
    """Facts for the position after `san`, ordered: what decides the game first."""
    before = bt._board(board_before)
    after = before.copy(stack=False)
    after.push(after.parse_san(san))
    mover = not after.turn
    s = Sheet()

    s.head("1. STATUS AND MATERIAL — if this decides the game, it is the point.")
    s.add(_status(after), "status")
    s.add(bt.inventory(after), "inventory")
    completes = _exchange(before, after, san, s)

    s.head(f"2. MATERIAL ON THE MOVE — what either side can win right now.")
    s.add("Hanging: " + bt.hanging(after, completes), "hanging")
    s.add(bt.forcing(after, completes=completes), "forcing")
    if after.is_check() or after.legal_moves.count() <= 3:
        s.add(_replies(after), "status")
    if after.is_check():
        s.add(f"{bt._side(after.turn)} must answer the check, so {bt._side(mover)} has no free move "
              f"to threaten anything else yet.", "status")
    else:
        s.add(f"If it were {bt._side(mover)}'s move again ({bt._side(mover)} just played {san}), "
              f"{bt._side(mover)} would have:")
        s.add(bt.forcing(bt._as_mover(after, mover)), "threats")

    s.head(f"3. WHAT {san} CHANGED — the move's plus and its minus.")
    s.add(ps.change(before, san), "change")
    s.add(f"Was it necessary, was it careless (captures and mate only, the move and the reply):")
    s.add(ps.alternatives(before, san), "alternatives")

    s.head("4. KING SAFETY.")
    s.add(ps.king_safety(after), "king_safety")

    s.head("5. THE PIECES — mobility, what each one is doing, whether it can be driven off.")
    s.add(ps.piece_quality(after, None if pieces_both_sides else mover), "piece_quality")

    s.head("6. PAWNS, FILES AND HOLES — the permanent part of the position.")
    s.add(bt.pawn_structure(after), "pawn_structure")

    s.head("7. SPACE.")
    s.add(ps.space(after), "space")
    return s.text(), s.index


TOOLS = ("status", "inventory", "material", "exchange", "hanging", "replies", "threats", "change", "alternatives",
         "king", "pieces", "pawns", "space")


def short_sheet(board_before, san, tools=TOOLS):
    """The sheet for a 2048-token model: only the named lookups, no headings, no ids, same order."""
    before = bt._board(board_before)
    after = before.copy(stack=False)
    after.push(after.parse_san(san))
    mover, side = not after.turn, bt._side
    out, completes = [], None
    if "status" in tools:
        out.append(_status(after))
    if "inventory" in tools:  # piece lists and the count; stands in for the diagram
        out.append(bt.inventory(after))
    elif "material" in tools:
        out.append(bt.inventory(after).splitlines()[-1])
    if "exchange" in tools:
        s = Sheet()
        completes = _exchange(before, after, san, s)
        out += s.lines
    if "hanging" in tools:
        out.append("Hanging: " + bt.hanging(after, completes))
    if "replies" in tools:
        out.append(bt.forcing(after, completes=completes))
    if "threats" in tools and not after.is_check():
        out.append(f"If {side(mover)} could move again: " + bt.forcing(bt._as_mover(after, mover)))
    if "change" in tools:
        out.append(ps.change(before, san))
    if "alternatives" in tools:
        out.append(ps.alternatives(before, san))
    if "king" in tools:
        out.append(ps.king_safety(after))
    if "pieces" in tools:
        out.append(ps.piece_quality(after))
    if "pawns" in tools:
        out.append(bt.pawn_structure(after))
    if "space" in tools:
        out.append(ps.space(after))
    return "\n".join(l.strip() for t in out for l in str(t).splitlines() if l.strip())


def board_before(example):
    """Rebuild the position before the example's last move from its move list."""
    b = chess.Board()
    for tok in example["moves"].split():
        if not tok[0].isdigit():
            b.push_san(tok)
    b.pop()
    return b


def for_example(e, tools=None):
    if tools is not None:
        return short_sheet(board_before(e), e["move"], tools), None
    return sheet(board_before(e), e["move"])


if __name__ == "__main__":
    if sys.argv[1].endswith(".jsonl") or sys.argv[1].startswith("dev-") or sys.argv[1].startswith("train-"):
        import program
        pool = program.by_id("train", "dev")
        e = pool[sys.argv[1]]
        print(program.fmt_input(e), "\n")
        print(for_example(e)[0])
    else:
        print(sheet(chess.Board(sys.argv[1]), sys.argv[2])[0])
