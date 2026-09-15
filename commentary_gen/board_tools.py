"""Board lookups the r4 judges asked for (RESULTS.md, round 4). Pure python-chess, no engine.

Every tool takes a FEN (or a chess.Board) and returns plain text meant to be pasted into a
prompt; the *_data variants return the underlying dicts. `report(fen)` bundles them.

    attack_map     what each piece reaches, blockers respected; attackers/defenders of a square
    inventory      pieces by square for both sides, material, king squares
    hanging        pieces the opponent can capture for a net gain (static exchange, pin-aware)
    forcing        checks and captures for the side to move, one ply, with a net verdict
    legal          is a named move legal here; if not, why
    pawn_structure isolated / doubled / backward / passed pawns, file status, holes

CLI:  python board_tools.py "<fen>" [last_move_san]
"""
import re
import sys
import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
        chess.QUEEN: "queen", chess.KING: "king"}
LETTER = {"N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN, "K": chess.KING}

class PositionError(ValueError):
    pass

def _board(b):
    try:
        board = b if isinstance(b, chess.Board) else chess.Board(b)
    except ValueError as e:
        raise PositionError(f"not a valid FEN: {e}")
    if not board.is_valid():
        status = board.status()
        why = [n for n, flag in (("a side has no king", chess.STATUS_NO_WHITE_KING | chess.STATUS_NO_BLACK_KING),
                                 ("more than one king", chess.STATUS_TOO_MANY_KINGS),
                                 ("the side not to move is in check", chess.STATUS_OPPOSITE_CHECK),
                                 ("pawns on the first or last rank", chess.STATUS_PAWNS_ON_BACKRANK),
                                 ("too many pieces", chess.STATUS_TOO_MANY_WHITE_PIECES | chess.STATUS_TOO_MANY_BLACK_PIECES),
                                 ("castling rights without king and rook in place", chess.STATUS_BAD_CASTLING_RIGHTS),
                                 ("invalid en-passant square", chess.STATUS_INVALID_EP_SQUARE))
               if status & flag]
        raise PositionError("position is not legal: " + ("; ".join(why) or str(status)))
    return board

def safe(fn):
    """Wrap a tool so a bad position or move returns one error line instead of raising."""
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except (PositionError, ValueError) as e:
            return f"Error: {e}"
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped

def _sq(s):
    return chess.parse_square(s) if isinstance(s, str) else s

def _side(c):
    return "White" if c else "Black"

def _piece(board, sq):
    p = board.piece_at(sq)
    return f"{_side(p.color)} {NAME[p.piece_type]}" if p else "empty"

def _sqs(squares):
    return " ".join(chess.square_name(s) for s in sorted(squares)) or "-"

def _as_mover(board, color):
    b = board.copy(stack=False)
    if b.turn != color:
        b.turn = color
        b.ep_square = None
    return b

def attack_map_data(board, square=None):
    """Per piece: squares reached (blockers respected), enemy pieces attacked, own pieces defended,
    and who attacks / defends it. With `square`: only that square."""
    board = _board(board)
    out = {}
    squares = [_sq(square)] if square is not None else list(chess.SQUARES)
    for sq in squares:
        p = board.piece_at(sq)
        if not p:
            continue
        att = board.attacks(sq)
        pinned = board.is_pinned(p.color, sq)
        out[chess.square_name(sq)] = {
            "piece": _piece(board, sq),
            "pinned": pinned,
            "reaches": [chess.square_name(s) for s in att],
            "attacks": [chess.square_name(s) for s in att if board.piece_at(s) and board.piece_at(s).color != p.color],
            "defends": [chess.square_name(s) for s in att if board.piece_at(s) and board.piece_at(s).color == p.color],
            "attacked_by": [chess.square_name(s) for s in board.attackers(not p.color, sq)],
            "defended_by": [chess.square_name(s) for s in board.attackers(p.color, sq)],
        }
    return out

def attack_map(board, square=None):
    board = _board(board)
    lines = []
    if square is not None:
        sq = _sq(square)
        w, b = board.attackers(chess.WHITE, sq), board.attackers(chess.BLACK, sq)
        lines.append(f"{chess.square_name(sq)}: {_piece(board, sq)}; attacked by White from {_sqs(w)}; "
                     f"attacked by Black from {_sqs(b)}")
    for name, d in attack_map_data(board, square).items():
        lines.append(f"{name} ({d['piece']}{', pinned to its king' if d['pinned'] else ''}): "
                     f"reaches {' '.join(d['reaches']) or '-'}"
                     + (f"; attacks {' '.join(d['attacks'])}" if d["attacks"] else "")
                     + (f"; defends {' '.join(d['defends'])}" if d["defends"] else "")
                     + f"; attacked by {' '.join(d['attacked_by']) or 'nothing'}"
                     + f"; defended by {' '.join(d['defended_by']) or 'nothing'}")
    return "\n".join(lines)

def inventory_data(board):
    board = _board(board)
    d = {}
    for color in (chess.WHITE, chess.BLACK):
        pieces = {}
        for pt in (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            sqs = board.pieces(pt, color)
            if sqs:
                pieces[NAME[pt]] = [chess.square_name(s) for s in sqs]
        d[_side(color)] = {"pieces": pieces,
                           "material": sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color == color),
                           "king": chess.square_name(board.king(color)) if board.king(color) is not None else "none"}
    d["material_diff_white_minus_black"] = d["White"]["material"] - d["Black"]["material"]
    d["side_to_move"] = _side(board.turn)
    return d

def inventory(board):
    d = inventory_data(board)
    lines = []
    for side in ("White", "Black"):
        s = d[side]
        parts = [f"{k} {' '.join(v)}" for k, v in s["pieces"].items()]
        lines.append(f"{side} (material {s['material']}): " + "; ".join(parts))
    diff = d["material_diff_white_minus_black"]
    lines.append("Material: " + ("equal" if diff == 0 else f"{'White' if diff > 0 else 'Black'} up {abs(diff)}")
                 + f". {d['side_to_move']} to move.")
    return "\n".join(lines)

def _captures_on(board, sq):
    out = []
    for m in board.legal_moves:
        if m.to_square == sq and board.is_capture(m):
            out.append(m)
        elif board.is_en_passant(m) and m.to_square == board.ep_square\
                and sq == m.to_square + (-8 if board.turn else 8):
            out.append(m)

    return [m for m in out if not m.promotion or m.promotion == chess.QUEEN]

def _gain_of(board, m):
    g = 0
    if board.is_en_passant(m):
        g = 1
    elif board.is_capture(m):
        g = VAL[board.piece_type_at(m.to_square)]
    if m.promotion:
        g += VAL[m.promotion] - 1
    return g

def _see_square(board, sq):
    best = 0
    for m in _captures_on(board, sq):
        b = board.copy(stack=False)
        b.push(m)
        best = max(best, _gain_of(board, m) - _see_square(b, sq))
    return best

def see(board, m):
    """Net material of playing `m` (any move) and then the exchange on its destination square."""
    if not isinstance(board, chess.Board):
        board = _board(board)
    b = board.copy(stack=False)
    b.push(m)
    return _gain_of(board, m) - _see_square(b, m.to_square)

def hanging_data(board):
    """Every piece the opponent can capture for a net gain, with the capture and the reason."""
    board = _board(board)
    out = []
    for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
            continue
        b = _as_mover(board, not p.color)
        caps = _captures_on(b, sq)
        if not caps:
            continue
        best = max(caps, key=lambda m: see(b, m))
        gain = see(b, best)
        if gain <= 0:
            continue
        b2 = b.copy(stack=False)
        b2.push(best)
        recap = _captures_on(b2, best.to_square)
        nominal = [chess.square_name(s) for s in board.attackers(p.color, sq)]
        reason = (("undefended" if not nominal else "defended, but the recapture is not legal")
                  if not recap else
                  "attacked by a cheaper piece" if VAL[b.piece_type_at(best.from_square)] < VAL[p.piece_type] else
                  "outnumbered")
        out.append({"square": chess.square_name(sq), "piece": _piece(board, sq), "reason": reason,
                    "capture": b.san(best), "wins": gain,
                    "attackers": [chess.square_name(m.from_square) for m in caps],
                    "defenders": nominal,
                    "legal_recapture_from": [chess.square_name(m.from_square) for m in recap]})
    return out

def hanging(board):
    rows = hanging_data(board)
    if not rows:
        return "No piece is hanging."
    board = _board(board)
    lines = []
    for r in rows:
        dfn = ("nothing" if not r["defenders"] else
               " ".join(r["defenders"]) if r["legal_recapture_from"] else
               f"{' '.join(r['defenders'])} but no legal recapture")
        own = r["piece"].startswith(_side(board.turn))
        lines.append(f"{r['square']} {r['piece']}: {r['reason']}, {r['capture']} wins {r['wins']}"
                     f"{'' if not own else ' if it were ' + _side(not board.turn) + chr(39) + 's move'} "
                     f"(attacked from {' '.join(r['attackers'])}; defended from {dfn})")
    return "\n".join(lines)

def forcing_data(board):
    """Checks and captures for the side to move (one ply), each with its net material."""
    board = _board(board)
    out = []
    for m in board.legal_moves:
        is_cap, is_chk = board.is_capture(m), board.gives_check(m)
        if not (is_cap or is_chk):
            continue
        b2 = board.copy(stack=False)
        b2.push(m)
        if m.promotion and m.promotion != chess.QUEEN and not b2.is_checkmate():
            continue
        row = {"move": board.san(m), "check": is_chk, "capture": is_cap, "mate": b2.is_checkmate(),
               "net": see(board, m)}
        if is_cap:
            row["takes"] = "pawn (en passant)" if board.is_en_passant(m) else _piece(board, m.to_square)
        p = b2.piece_at(m.to_square)
        targets = [s for s in b2.attacks(m.to_square) if b2.piece_at(s) and b2.piece_at(s).color != p.color
                   and b2.piece_type_at(s) != chess.PAWN]
        if len(targets) >= 2 or (is_chk and len(targets) >= 1 and not is_cap):
            row["fork"] = [chess.square_name(s) for s in targets if b2.piece_type_at(s) != chess.KING]
            if not row["fork"]:
                del row["fork"]
        out.append(row)
    out.sort(key=lambda r: (not r["mate"], -r["net"], not r["check"], r["move"]))
    return out

def forcing(board):
    board = _board(board)
    rows = forcing_data(board)
    if not rows:
        return f"{_side(board.turn)} has no checks or captures."
    lines = [f"{_side(board.turn)} to move; checks and captures (net = material after the exchange on that square):"]
    for r in rows:
        tags = []
        if r["mate"]:
            tags.append("MATE")
        elif r["check"]:
            tags.append("check")
        if r["capture"]:
            tags.append(f"takes {r['takes']}")
        if r.get("fork"):
            tags.append("then hits " + " ".join(r["fork"]))
        tags.append(f"net {r['net']:+d}")
        lines.append(f"  {r['move']}: " + ", ".join(tags))
    return "\n".join(lines)

def _legal_reason(board, m):
    if board.is_check():
        return "the king is in check and this move does not get out of it"
    if board.piece_type_at(m.from_square) == chess.KING:
        att = board.attackers(not board.turn, m.to_square)
        return f"{chess.square_name(m.to_square)} is attacked from {_sqs(att)}"
    if board.is_pinned(board.turn, m.from_square):
        return f"the {NAME[board.piece_type_at(m.from_square)]} on {chess.square_name(m.from_square)} is pinned to the king"
    return "it would leave the king in check"

def _castling_reason(board, san):
    king = board.king(board.turn)
    kingside = san.count("O") == 2
    rank = 0 if board.turn else 7
    if not (board.has_kingside_castling_rights(board.turn) if kingside else board.has_queenside_castling_rights(board.turn)):
        return "no castling rights on that side (king or rook has moved)"
    path = [chess.square(f, rank) for f in ((5, 6) if kingside else (1, 2, 3))]
    blocked = [chess.square_name(s) for s in path if board.piece_at(s)]
    if blocked:
        return f"{' '.join(blocked)} occupied"
    if board.is_check():
        return "the king is in check"
    through = [chess.square(f, rank) for f in ((5, 6) if kingside else (2, 3))]
    att = [chess.square_name(s) for s in through if board.is_attacked_by(not board.turn, s)]
    return f"the king would pass through or land on an attacked square ({' '.join(att)})" if att else "unknown"

def legal(board, san):
    """Is `san` legal here? One-line verdict, with the reason when it is not."""
    board = _board(board)
    me = _side(board.turn)
    clean = san.strip().replace("+", "").replace("#", "").replace("!", "").replace("?", "")
    if not clean:
        return "No move given."
    try:
        m = board.parse_san(clean)
        notes = []
        if "x" in clean and not board.is_capture(m):
            notes.append(f"it is not a capture, {chess.square_name(m.to_square)} is empty")
        b2 = board.copy(stack=False)
        b2.push(m)
        if "#" in san and not b2.is_checkmate():
            notes.append("it is not mate" if b2.is_check() else "it does not give check")
        elif "+" in san and not b2.is_check():
            notes.append("it does not give check")
        elif "+" not in san and "#" not in san and b2.is_check():
            notes.append("it gives " + ("mate" if b2.is_checkmate() else "check"))
        return f"{san} is legal for {me}" + (", but " + " and ".join(notes) if notes else "") + "."
    except chess.AmbiguousMoveError:
        return f"{san} is ambiguous; more than one {me} piece can play it."
    except chess.InvalidMoveError:
        return f"{san} is not a move."
    except chess.IllegalMoveError:
        pass
    if clean in ("O-O", "O-O-O", "0-0", "0-0-0"):
        return f"{san} is not legal: {_castling_reason(board, clean.replace('0', 'O'))}."
    mm = re.fullmatch(r"([NBRQK])?([a-h])?([1-8])?(x)?([a-h][1-8])(=?[NBRQ])?", clean)
    if not mm:
        return f"{san} is not legal here."
    pt = LETTER.get(mm.group(1), chess.PAWN)
    to = chess.parse_square(mm.group(5))
    if mm.group(6) and (pt != chess.PAWN or chess.square_rank(to) not in (0, 7)):
        return f"{san} is not legal: a promotion suffix only applies to a pawn reaching the last rank."
    cands = [s for s in board.pieces(pt, board.turn)
             if (not mm.group(2) or chess.square_file(s) == ord(mm.group(2)) - 97)
             and (not mm.group(3) or chess.square_rank(s) == int(mm.group(3)) - 1)]
    if pt == chess.PAWN and not mm.group(2):
        cands = [s for s in cands if chess.square_file(s) == chess.square_file(to)]
    target = board.piece_at(to)
    if target and target.color == board.turn:
        return f"{san} is not legal: {mm.group(5)} holds {me}'s own {NAME[target.piece_type]}."
    if not cands:
        other = _as_mover(board, not board.turn)
        try:
            other.parse_san(clean)
            return f"{san} is not legal: it is {me}'s move and that would be a {_side(not board.turn)} move."
        except Exception:
            return f"{san} is not legal: {me} has no {NAME[pt]} that could play it."

    if pt == chess.PAWN and chess.square_rank(to) in (0, 7) and not mm.group(6):
        for s in cands:
            if board.is_legal(chess.Move(s, to, promotion=chess.QUEEN)):
                return f"{san} is legal but needs a promotion piece, e.g. {clean}=Q."
    reasons = []
    for s in cands:
        promo = chess.QUEEN if pt == chess.PAWN and chess.square_rank(to) in (0, 7) else None
        m = chess.Move(s, to, promotion=promo)
        if board.is_pseudo_legal(m):
            reasons.append(_legal_reason(board, m))
            continue
        src = chess.square_name(s)
        if pt == chess.PAWN:
            if chess.square_file(s) == chess.square_file(to):
                dist = (chess.square_rank(to) - chess.square_rank(s)) * (1 if board.turn else -1)
                start = chess.square_rank(s) == (1 if board.turn else 6)
                if dist == 1 or (dist == 2 and start):
                    ahead = chess.SquareSet.between(s, to) | chess.SquareSet([to])
                    occ = [chess.square_name(x) for x in ahead if board.piece_at(x)]
                    reasons.append(f"the pawn is blocked, {' '.join(occ)} is occupied")
                else:
                    reasons.append(f"a pawn on {src} cannot advance to {mm.group(5)} in one move")
            elif to in board.attacks(s):
                reasons.append(f"{mm.group(5)} is empty; a pawn moves diagonally only when capturing"
                               + (" (no en passant available)" if chess.square_rank(to) in (2, 5) else ""))
            else:
                reasons.append(f"the pawn on {src} does not reach {mm.group(5)}")
        elif pt in (chess.KNIGHT, chess.KING):
            reasons.append(f"a {NAME[pt]} on {src} does not reach {mm.group(5)}")
        else:
            on_line = bool(chess.SquareSet.ray(s, to)) and to in chess.SquareSet(
                chess.BB_ALL & ~chess.BB_SQUARES[s]) and (
                (pt != chess.ROOK and chess.square_file(s) != chess.square_file(to) and chess.square_rank(s) != chess.square_rank(to)) or
                (pt != chess.BISHOP and (chess.square_file(s) == chess.square_file(to) or chess.square_rank(s) == chess.square_rank(to))))
            blockers = [chess.square_name(x) for x in chess.SquareSet.between(s, to) if board.piece_at(x)]
            if on_line and blockers:
                reasons.append(f"the line from {src} to {mm.group(5)} is blocked at {' '.join(blockers)}")
            else:
                reasons.append(f"{mm.group(5)} is not on a line from the {NAME[pt]} on {src}")
    return f"{san} is not legal: " + "; ".join(dict.fromkeys(reasons)) + "."

def pawn_structure_data(board):
    board = _board(board)
    d = {}
    files = {c: {chess.square_file(s) for s in board.pieces(chess.PAWN, c)} for c in (chess.WHITE, chess.BLACK)}
    for color in (chess.WHITE, chess.BLACK):
        own = board.pieces(chess.PAWN, color)
        enemy = board.pieces(chess.PAWN, not color)
        iso, dbl, passed, backward = [], [], [], []
        by_file = {}
        for s in own:
            by_file.setdefault(chess.square_file(s), []).append(s)
        for f, sqs in by_file.items():
            if f - 1 not in files[color] and f + 1 not in files[color]:
                iso += sqs
            if len(sqs) > 1:
                dbl += sqs
        for s in own:
            f, r = chess.square_file(s), chess.square_rank(s)
            ahead = range(r + 1, 8) if color else range(r - 1, -1, -1)
            blockers = [chess.square(ff, rr) for rr in ahead for ff in (f - 1, f, f + 1) if 0 <= ff < 8]
            own_ahead = any(chess.square(f, rr) in own for rr in ahead)
            if not any(x in enemy for x in blockers) and not own_ahead:
                passed.append(s)

            behind = range(0, r + 1) if color else range(r, 8)
            support = [chess.square(ff, rr) for rr in behind for ff in (f - 1, f + 1) if 0 <= ff < 8]
            stop = s + (8 if color else -8)
            if 0 <= stop < 64 and s not in iso and not any(x in own for x in support) and\
                    any(board.piece_type_at(a) == chess.PAWN for a in board.attackers(not color, stop)):
                backward.append(s)
        d[_side(color)] = {"isolated": [chess.square_name(s) for s in iso],
                           "doubled": [chess.square_name(s) for s in dbl],
                           "passed": [chess.square_name(s) for s in passed],
                           "backward": [chess.square_name(s) for s in backward]}
    status = {}
    for f in range(8):
        w, b = f in files[chess.WHITE], f in files[chess.BLACK]
        status["abcdefgh"[f]] = "open" if not (w or b) else "closed" if (w and b) else\
            f"half-open for {'Black' if w else 'White'}"
    d["files"] = status

    holes = {}
    for color in (chess.WHITE, chess.BLACK):
        own = board.pieces(chess.PAWN, color)
        camp = range(2, 4) if color else range(4, 6)
        hs = []
        for r in camp:
            for f in range(2, 6):
                sq = chess.square(f, r)
                if sq in own:
                    continue
                behind = range(0, r) if color else range(r + 1, 8)
                guards = [chess.square(ff, rr) for rr in behind for ff in (f - 1, f + 1) if 0 <= ff < 8]
                if not any(g in own for g in guards):
                    hs.append(chess.square_name(sq))
        holes[_side(color)] = hs
    d["holes"] = holes
    return d

def pawn_structure(board):
    d = pawn_structure_data(board)
    lines = []
    for side in ("White", "Black"):
        parts = [f"{k} {' '.join(v)}" for k, v in d[side].items() if v]
        lines.append(f"{side} pawns: " + ("; ".join(parts) if parts else "no isolated, doubled, passed or backward pawns"))
    lines.append("Files: " + ", ".join(f"{f} {s}" for f, s in d["files"].items()))
    for side in ("White", "Black"):
        if d["holes"][side]:
            lines.append(f"Central squares in {side}'s camp no {side} pawn can ever guard: {' '.join(d['holes'][side])}")
    return "\n".join(lines)

def report(board):
    """Inventory, hanging pieces, forcing moves and pawn structure for a position."""
    board = _board(board)
    state = ("CHECKMATE, " + _side(not board.turn) + " has won." if board.is_checkmate() else
             "Stalemate." if board.is_stalemate() else
             f"{_side(board.turn)} is in check from {_sqs(board.checkers())}." if board.is_check() else "")
    return "\n".join(([state] if state else []) +
                     [inventory(board), "", "Hanging: " + hanging(board).replace("\n", "\n  "), "",
                      forcing(board), "", pawn_structure(board)])

def report_after_move(board_before, san):
    """Report for the position after `san`, plus what the moved piece now attacks and what attacks it."""
    b = _board(board_before).copy(stack=False)
    try:
        m = b.parse_san(san)
    except ValueError:
        return f"Cannot play {san}: {legal(b, san)}"
    b.push(m)
    moved = attack_map(b, m.to_square).split("\n")[1] if b.piece_at(m.to_square) else ""
    return f"After {san}:\n{moved}\n\n{report(b)}"

attack_map, inventory, hanging, forcing, legal, pawn_structure, report, report_after_move = map(
    safe, (attack_map, inventory, hanging, forcing, legal, pawn_structure, report, report_after_move))

if __name__ == "__main__":
    fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN
    if len(sys.argv) > 2:
        print(report_after_move(fen, sys.argv[2]))
    else:
        print(report(fen))
