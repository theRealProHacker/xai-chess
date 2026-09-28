"""The positional half of the lookups: what the move changed, and how good every piece is.

board_tools answers tactical questions about a position (who hangs, what is forcing). These answer
the two questions the ICS lessons ask after every move — what did it change, and what is each piece
worth now — and they are what a positional evaluation is written from. Pure python-chess.

    change          the move's plus and minus: squares given up and taken, lines opened and blocked,
                    defenders pulled away, pieces newly attacked, what a pawn move makes permanent
    piece_quality   per piece: mobility, safe mobility, role, what ties it down, stability, bad bishop
    king_safety     shield, holes in it, files bearing on the king, flight squares, attackers nearby
    space           squares held in the opponent's half, by wing

CLI:  python positional.py "<fen_before>" <san>
"""
import sys
import chess
import board_tools as bt
from board_tools import VAL, NAME, _board, _piece, _side, _sqs, _as_mover, safe, see

WINGS = (("queenside", range(0, 3)), ("centre", range(3, 5)), ("kingside", range(5, 8)))


def _name_on(board, sq):
    p = board.piece_at(sq)
    return f"{_side(p.color)} {NAME[p.piece_type]} on {chess.square_name(sq)}" if p else chess.square_name(sq)


def _blockers(board, sq):
    """The pieces that stop a slider on `sq`: python-chess attacks() ends on the first occupant."""
    p = board.piece_at(sq)
    if not p or p.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return {}
    return {s: board.piece_at(s) for s in board.attacks(sq) if board.piece_at(s)}


def _pawn_guard_squares(color, sq):
    """The two squares a pawn on `sq` guards."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    r2 = r + 1 if color else r - 1
    if not 0 <= r2 < 8:
        return []
    return [chess.square(ff, r2) for ff in (f - 1, f + 1) if 0 <= ff < 8]


def _would_hang(board, sq):
    """Can the other side win material by capturing on `sq`?"""
    p = board.piece_at(sq)
    if not p:
        return False
    b = _as_mover(board, not p.color)
    return any(see(b, m) > 0 for m in bt._captures_on(b, sq))


# --------------------------------------------------------------------------- change

def change_data(board_before, san):
    """What the move gave up and what it gained. Everything is a diff of before and after."""
    before = _board(board_before).copy(stack=False)
    try:
        m = before.parse_san(san)
    except ValueError as e:
        raise ValueError(f"cannot play {san}: {e}")
    mover = before.turn
    piece = before.piece_at(m.from_square)
    captured = before.piece_at(m.to_square)
    after = before.copy(stack=False)
    after.push(m)

    d = {"move": san, "mover": _side(mover), "piece": NAME[piece.piece_type],
         "from": chess.square_name(m.from_square), "to": chess.square_name(m.to_square),
         "captured": NAME[captured.piece_type] if captured else None,
         "check": after.is_check(), "mate": after.is_checkmate(), "castles": before.is_castling(m)}

    # squares the moved piece guarded and no longer does (nobody else of its colour covers them)
    was = set(before.attacks(m.from_square))
    now = set(after.attacks(m.to_square)) if after.piece_at(m.to_square) else set()
    lost = [s for s in sorted(was - now) if not after.attackers(mover, s)]
    gained = [s for s in sorted(now - was) if not before.attackers(mover, s)]
    d["squares_given_up"] = [chess.square_name(s) for s in lost]
    d["squares_taken"] = [chess.square_name(s) for s in gained]
    # of those, the ones the opponent can actually use: empty and reachable by an enemy piece
    d["given_up_and_reachable"] = [chess.square_name(s) for s in lost
                                   if not after.piece_at(s) and after.attackers(not mover, s)]

    # own pieces that lost their only defender by the move
    undefended = []
    for s in was:
        p = after.piece_at(s)
        if p and p.color == mover and s != m.to_square and not after.attackers(mover, s) \
                and after.attackers(not mover, s):
            undefended.append(chess.square_name(s))
    d["left_undefended"] = sorted(undefended)

    # lines: sliders whose reach changed because from_square emptied or to_square filled
    opened, blocked = [], []
    for color in (chess.WHITE, chess.BLACK):
        for s in list(before.pieces(chess.BISHOP, color)) + list(before.pieces(chess.ROOK, color)) \
                + list(before.pieces(chess.QUEEN, color)):
            if s in (m.from_square, m.to_square):
                continue
            b_reach, a_reach = set(before.attacks(s)), set(after.attacks(s))
            if a_reach - b_reach:
                opened.append({"piece": _name_on(after, s), "gains": _sqs(sorted(a_reach - b_reach))})
            if b_reach - a_reach:
                blocked.append({"piece": _name_on(after, s), "loses": _sqs(sorted(b_reach - a_reach))})
    d["lines_opened"] = opened
    d["lines_blocked"] = blocked

    # what the piece hits now that nothing of its colour hit before
    d["now_attacks"] = [_name_on(after, s) for s in sorted(now)
                        if after.piece_at(s) and after.piece_at(s).color != mover
                        and not before.attackers(mover, s)]
    d["now_defends"] = [_name_on(after, s) for s in sorted(now)
                        if after.piece_at(s) and after.piece_at(s).color == mover
                        and not before.attackers(mover, s)]

    # a rook or queen landing on a file: is the file open, and who stands on it
    if piece.piece_type in (chess.ROOK, chess.QUEEN):
        fl = chess.square_file(m.to_square)
        letter = "abcdefgh"[fl]
        d["file"] = {"file": letter, "status": bt.pawn_structure_data(after)["files"][letter],
                     "enemy_on_it": [_name_on(after, s) for s in chess.SQUARES
                                     if chess.square_file(s) == fl and after.piece_at(s)
                                     and after.piece_at(s).color != mover]}

    # a pawn move is permanent: the two squares it guarded are given up for good
    if piece.piece_type == chess.PAWN:
        d["pawn_permanent"] = [{"square": chess.square_name(s),
                                "still_covered_by": [_name_on(after, a) for a in after.attackers(mover, s)]}
                               for s in _pawn_guard_squares(mover, m.from_square) if s not in now]
        d["pawn_now_guards"] = [chess.square_name(s) for s in _pawn_guard_squares(mover, m.to_square)]

    # is the piece safe where it landed
    d["landed_safe"] = not _would_hang(after, m.to_square) if after.piece_at(m.to_square) else True
    d["attacked_on_arrival"] = _sqs(after.attackers(not mover, m.to_square)) if \
        after.attackers(not mover, m.to_square) else ""

    # tempo: had this piece already moved, and is a back-rank piece still at home
    stack = getattr(board_before, "move_stack", [])
    d["piece_moved_before"] = any(mv.to_square == m.from_square for mv in stack)
    rank = 0 if mover else 7
    d["still_at_home"] = [chess.square_name(s) for s in
                          (chess.square(f, rank) for f in (1, 2, 3, 5, 6))
                          if after.piece_at(s) and after.piece_at(s).color == mover
                          and after.piece_type_at(s) in (chess.KNIGHT, chess.BISHOP, chess.QUEEN)]
    d["castled"] = not after.has_castling_rights(mover) and after.king(mover) not in (chess.E1, chess.E8)
    return d


def change(board_before, san):
    d = change_data(board_before, san)
    L = [f"{d['mover']} played {d['move']}: the {d['piece']} went {d['from']}-{d['to']}"
         + (f", taking the {d['captured']}" if d["captured"] else "")
         + (" with mate" if d["mate"] else ", with check" if d["check"] else "") + "."]
    if d.get("file"):
        f = d["file"]
        L.append(f"It stands on the {f['file']}-file, which is {f['status']}"
                 + (f"; on that file: {', '.join(f['enemy_on_it'])}." if f["enemy_on_it"] else "."))
    if d["squares_taken"]:
        L.append("Squares it guards that nothing of its colour guarded before: "
                 + " ".join(d["squares_taken"]) + ".")
    if d["now_attacks"]:
        L.append("It now attacks " + "; ".join(d["now_attacks"]) + " (nothing of its colour did before).")
    if d["now_defends"]:
        L.append("It now defends " + "; ".join(d["now_defends"]) + ".")
    if d["attacked_on_arrival"]:
        L.append(f"On {d['to']} it is attacked from {d['attacked_on_arrival']}"
                 + ("." if d["landed_safe"] else " and can be won there."))
    if d.get("pawn_permanent"):
        gave = ", ".join(f"{g['square']} (still covered by {', '.join(g['still_covered_by'][:2])}"
                         + (" and others)" if len(g["still_covered_by"]) > 2 else ")")
                         if g["still_covered_by"] else f"{g['square']} (covered by nothing)"
                         for g in d["pawn_permanent"])
        L.append(f"A pawn cannot go back: it will never guard {gave} again"
                 + (f"; it guards {' '.join(d['pawn_now_guards'])} instead." if d.get("pawn_now_guards") else "."))
    elif d["squares_given_up"]:
        L.append("Squares it guarded and now nothing of its colour guards: " + " ".join(d["squares_given_up"]) + ".")
    if d["given_up_and_reachable"]:
        L.append("Of those the opponent can reach " + " ".join(d["given_up_and_reachable"]) + ".")
    if d["left_undefended"]:
        L.append("It was the only defender of " + " ".join(d["left_undefended"])
                 + f", now undefended and attacked.")
    for o in d["lines_opened"]:
        L.append(f"Line opened: the {o['piece']} now also reaches {o['gains']}.")
    for b in d["lines_blocked"]:
        L.append(f"Line blocked: the {b['piece']} no longer reaches {b['loses']}.")
    if d["piece_moved_before"]:
        L.append("This piece had already moved.")
    if d["still_at_home"] and not d["castles"]:
        L.append(f"{d['mover']} still has pieces at home: {' '.join(d['still_at_home'])}.")
    return "\n".join(L)


# --------------------------------------------------------------------------- alternatives

MATE = 100


def _best_grab(board):
    """Most material the side to move can win right now: mate counts as everything."""
    best = 0
    for m in board.legal_moves:
        b2 = board.copy(stack=False)
        b2.push(m)
        if b2.is_checkmate():
            return MATE
        if board.is_capture(m):
            best = max(best, see(board, m))
    return best


def _grab_words(v):
    return "mate" if v >= MATE else "nothing" if v <= 0 else bt.material_words(v).replace("wins ", "")


NET_WORDS = {1: "a pawn", 3: "a piece", 5: "a rook", 9: "a queen"}


def _net_words(v):
    """Net result for the mover: v > 0 wins, v < 0 loses. No 'the exchange' here: a net 2 is
    rarely rook-for-minor once the moves are added up."""
    if v <= -MATE:
        return "allows mate"
    if v >= MATE:
        return "mates"
    if v == 0:
        return "comes out even"
    return ("wins " if v > 0 else "loses ") + NET_WORDS.get(abs(v), f"{abs(v)} points of material") + f" ({v:+d})"


def _recaptured(before, san):
    """If the move takes back on the square the opponent's last move captured on: (their move, gain)."""
    if not before.move_stack:
        return None
    m = before.parse_san(san)
    prev = before.move_stack[-1]
    if prev.to_square != m.to_square or not before.is_capture(m):
        return None
    b0 = before.copy()
    b0.pop()
    if not b0.is_capture(prev):
        return None
    return b0.san(prev), bt._gain_of(b0, prev)


def alternatives_data(board_before, san, limit=5):
    """Was the move necessary, and did it hold the loss as well as the alternatives did?

    No engine: 'good' here means only 'the opponent cannot win material or mate in reply'.
    Each move is scored over two plies: what it captures, minus the most the reply can win back,
    so a capture that is simply taken back counts as a trade, not as material given up.
    """
    before = _board(board_before)
    mover = before.turn
    threat = _best_grab(_as_mover(before, not mover))  # if the mover could do nothing at all
    scores, gains, grabs = {}, {}, {}
    for m in before.legal_moves:
        b2 = before.copy(stack=False)
        b2.push(m)
        k = before.san(m)
        if b2.is_checkmate() and b2.turn != mover:
            gains[k], grabs[k], scores[k] = 0, 0, -MATE  # the mover delivered mate: best possible
            continue
        gains[k], grabs[k] = bt._gain_of(before, m), _best_grab(b2)
        scores[k] = MATE if grabs[k] >= MATE else grabs[k] - gains[k]
    played = scores.get(san.strip("+#!?"), scores.get(san))
    if played is None:
        for k in scores:
            if k.rstrip("+#") == san.rstrip("+#!?"):
                played, san = scores[k], k
                break
    else:
        san = san if san in scores else san.strip("+#!?")
    floor = min(scores.values())
    better = sorted([s for s, v in scores.items() if v < played], key=lambda s: scores[s])
    equal = sorted([s for s, v in scores.items() if v == played and s != san])
    return {"move": san, "concedes": played, "gain": gains[san], "grab": grabs[san],
            "threat_before": threat, "best_available": floor,
            "better_moves": better[:limit], "n_better": len(better),
            "equal_moves": equal[:limit], "n_equal": len(equal), "n_legal": len(scores)}


def alternatives(board_before, san):
    d = alternatives_data(board_before, san)
    mover = bt._side(_board(board_before).turn)
    other = bt._side(not _board(board_before).turn)
    L = []
    if d["threat_before"] > 0:
        L.append(f"Before {d['move']}, {other} was ready to take {_grab_words(d['threat_before'])}"
                 f", so {mover} had something to answer.")
    rec = _recaptured(_board(board_before), d["move"]) if d["gain"] > 0 else None
    if rec:  # it finishes an exchange the opponent started: count from before that capture
        prev, prev_gain = rec
        ex = d["gain"] - prev_gain
        L.append(f"{d['move']} takes back on the square {prev} captured on: that exchange "
                 + ("is even." if ex == 0 else f"{_net_words(ex)} for {mover}."))
        if d["grab"] >= MATE:
            L.append(f"But it allows mate.")
        elif d["grab"] > 0:
            L.append(f"{other} can then take {_grab_words(d['grab'])}: counted from before {prev}, "
                     f"{mover} {_net_words(ex - d['grab'])} unless that capture can be answered.")
        else:
            L.append(f"{other} cannot win anything straight away.")
    elif d["gain"] > 0:  # a capture: judge it over the capture and the reply, not the reply alone
        took = _grab_words(d["gain"])
        if d["grab"] >= MATE:
            L.append(f"{d['move']} took {took}, but it allows mate.")
        elif d["grab"] > 0:
            L.append(f"{d['move']} took {took} and {other} can take {_grab_words(d['grab'])} back: "
                     f"over the two moves {mover} {_net_words(-d['concedes'])}.")
        else:
            L.append(f"{d['move']} took {took} and {other} cannot win anything back straight away.")
    elif d["concedes"] <= 0:
        if d["n_better"] == 0 and d["n_equal"] == 0:
            L.append(f"{d['move']} was the only one of {d['n_legal']} legal moves that leaves "
                     f"{other} nothing to take: every other move drops material or allows mate.")
        elif d["n_equal"] <= 2:
            L.append(f"{d['move']} leaves {other} nothing to take, and so would "
                     f"{' and '.join(d['equal_moves'])} — those were the only moves that did.")
        else:
            L.append(f"{d['move']} leaves {other} no capture that wins material and no mate; "
                     f"{d['n_equal']} of the {d['n_legal']} legal moves were as safe as that.")
    else:
        L.append(f"After {d['move']}, {other} can take {_grab_words(d['concedes'])} straight away.")
    if d["concedes"] > 0 or (d["gain"] > 0 and d["n_better"]):
        if d["n_better"]:
            L.append(f"{d['n_better']} other move{'' if d['n_better'] == 1 else 's'} did better over the "
                     f"two moves, the best of them {_net_words(-d['best_available'])}: "
                     f"{' '.join(d['better_moves'])}.")
        elif d["concedes"] > 0:
            L.append(f"No legal move avoided it; {mover} had to give up that much whatever "
                     f"{mover} played.")
    return "\n".join(L)


# --------------------------------------------------------------------------- piece quality

def _mobility(board, sq, color):
    b = _as_mover(board, color)
    moves = [m for m in b.legal_moves if m.from_square == sq]
    safe_moves = [m for m in moves if see(b, m) >= 0]
    return moves, safe_moves


def _driven_off_by(board, sq):
    """Can a cheaper enemy piece, or an enemy pawn in one move, attack this square?"""
    p = board.piece_at(sq)
    if not p or p.piece_type == chess.PAWN:
        return []
    out = []
    for a in board.attackers(not p.color, sq):
        if VAL[board.piece_type_at(a)] < VAL[p.piece_type]:
            out.append(f"already attacked by the {NAME[board.piece_type_at(a)]} on {chess.square_name(a)}")
    b = _as_mover(board, not p.color)
    for m in b.legal_moves:
        if b.piece_type_at(m.from_square) != chess.PAWN or b.is_capture(m):
            continue
        b2 = b.copy(stack=False)
        b2.push(m)
        if sq in b2.attacks(m.to_square) and see(b, m) >= 0:
            out.append(f"{b.san(m)} drives it off")
    return out[:3]


def piece_quality_data(board, color=None):
    board = _board(board)
    out = []
    colors = (chess.WHITE, chess.BLACK) if color is None else (color,)
    for c in colors:
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            for sq in sorted(board.pieces(pt, c)):
                moves, safe_moves = _mobility(board, sq, c)
                blockers = {chess.square_name(s): _piece(board, s) for s, p in _blockers(board, sq).items()
                            if p.color == c}
                attacks = [s for s in board.attacks(sq) if board.piece_at(s) and board.piece_at(s).color != c]
                defends = [s for s in board.attacks(sq) if board.piece_at(s) and board.piece_at(s).color == c]
                # tied down: an own piece it defends would hang without it
                tied = []
                for s in defends:
                    b2 = board.copy(stack=False)
                    b2.remove_piece_at(sq)
                    if _would_hang(b2, s) and not _would_hang(board, s):
                        tied.append(chess.square_name(s))
                row = {"square": chess.square_name(sq), "side": _side(c), "piece": NAME[pt],
                       "moves": len(moves), "safe_moves": len(safe_moves),
                       "blocked_by_own": blockers,
                       "attacks": [_name_on(board, s) for s in sorted(attacks)],
                       "defends": [chess.square_name(s) for s in sorted(defends)],
                       "tied_to": tied,
                       "pinned": board.is_pinned(c, sq),
                       "driven_off": _driven_off_by(board, sq)}
                if pt == chess.BISHOP:
                    same = [s for s in board.pieces(chess.PAWN, c)
                            if (chess.BB_LIGHT_SQUARES >> s) & 1 == (chess.BB_LIGHT_SQUARES >> sq) & 1]
                    row["own_pawns_on_its_colour"] = [chess.square_name(s) for s in sorted(same)]
                    row["colour"] = "light" if (chess.BB_LIGHT_SQUARES >> sq) & 1 else "dark"
                row["role"] = ("attacking and defending" if attacks and tied else
                               "attacking" if attacks else
                               "tied to defence" if tied else
                               "out of play" if len(safe_moves) <= 1 else "free")
                out.append(row)
    return out


def piece_quality(board, color=None):
    rows = piece_quality_data(board, color)
    if not rows:
        return "No pieces besides kings and pawns."
    L = []
    for r in rows:
        bits = [f"{r['moves']} legal move{'' if r['moves'] == 1 else 's'} "
                f"({r['safe_moves']} without losing material)"]
        if r["blocked_by_own"]:
            bits.append("own " + ", ".join(f"{v.split()[1]} on {k}" for k, v in r["blocked_by_own"].items())
                        + " in its way")
        if r["attacks"]:
            bits.append("attacks " + ", ".join(r["attacks"]))
        if r["tied_to"]:
            bits.append("must stay to defend " + " ".join(r["tied_to"]))
        elif r["defends"]:
            bits.append("defends " + " ".join(r["defends"]))
        if r["pinned"]:
            bits.append("pinned to its king")
        if r["driven_off"]:
            bits.append(r["driven_off"][0])
        if r.get("own_pawns_on_its_colour"):
            bits.append(f"{len(r['own_pawns_on_its_colour'])} own pawns on its {r['colour']} squares"
                        f" ({' '.join(r['own_pawns_on_its_colour'])})")
        L.append(f"{r['side']} {r['piece']} on {r['square']}: " + "; ".join(bits) + f". Role: {r['role']}.")
    return "\n".join(L)


# --------------------------------------------------------------------------- king safety

def _zone(board, color):
    k = board.king(color)
    return [s for s in chess.SQUARES if chess.square_distance(s, k) <= 1] if k is not None else []


def king_safety_data(board):
    board = _board(board)
    d = {}
    for c in (chess.WHITE, chess.BLACK):
        k = board.king(c)
        if k is None:
            continue
        f, r = chess.square_file(k), chess.square_rank(k)
        shield, missing = [], []
        for ff in (f - 1, f, f + 1):
            if not 0 <= ff < 8:
                continue
            own = [s for s in board.pieces(chess.PAWN, c) if chess.square_file(s) == ff]
            if not own:
                missing.append("abcdefgh"[ff] + "-file")
                continue
            sq = min(own, key=lambda s: abs(chess.square_rank(s) - r))
            home = 1 if c else 6
            shield.append(chess.square_name(sq) + ("" if chess.square_rank(sq) == home else " (advanced)"))
        zone = _zone(board, c)
        flight = [chess.square_name(s) for s in zone
                  if s != k and not board.piece_at(s) and not board.is_attacked_by(not c, s)]
        attackers = sorted({chess.square_name(a) for s in zone for a in board.attackers(not c, s)
                            if board.piece_type_at(a) != chess.PAWN})
        d[_side(c)] = {"king": chess.square_name(k),
                       "castled": k in (chess.G1, chess.C1, chess.B1, chess.G8, chess.C8, chess.B8),
                       "rights": board.has_castling_rights(c),
                       "shield": shield, "no_pawn_on": missing,
                       "flight_squares": flight,
                       "enemy_pieces_bearing_on_the_zone": attackers,
                       "in_check": board.is_check() and board.turn == c}
    return d


def king_safety(board):
    d = king_safety_data(board)
    L = []
    for side in ("White", "Black"):
        if side not in d:
            continue
        s = d[side]
        bits = [f"king on {s['king']}" + (", castled" if s["castled"] else
                                          ", can still castle" if s["rights"] else ", has not castled")]
        bits.append("shield " + (" ".join(s["shield"]) if s["shield"] else "gone"))
        if s["no_pawn_on"]:
            bits.append("no pawn left on the " + ", ".join(s["no_pawn_on"]))
        bits.append(("flight squares " + " ".join(s["flight_squares"])) if s["flight_squares"]
                    else "no flight square")
        if s["enemy_pieces_bearing_on_the_zone"]:
            bits.append("enemy pieces bearing on the king zone from "
                        + " ".join(s["enemy_pieces_bearing_on_the_zone"]))
        L.append(f"{side}: " + "; ".join(bits) + ".")
    return "\n".join(L)


# --------------------------------------------------------------------------- space

def space_data(board):
    board = _board(board)
    d = {}
    for c in (chess.WHITE, chess.BLACK):
        far = range(4, 8) if c else range(0, 4)
        held = {}
        for name, files in WINGS:
            held[name] = sum(1 for r in far for f in files
                             if board.is_attacked_by(c, chess.square(f, r)))
        pawns = [s for s in board.pieces(chess.PAWN, c)
                 if (chess.square_rank(s) >= 4 if c else chess.square_rank(s) <= 3)]
        d[_side(c)] = {"squares_held_in_the_far_half": held,
                       "pawns_past_the_middle": [chess.square_name(s) for s in sorted(pawns)]}
    return d


def space(board):
    d = space_data(board)
    L = []
    for side in ("White", "Black"):
        s = d[side]
        h = s["squares_held_in_the_far_half"]
        L.append(f"{side} holds {sum(h.values())} squares in the opponent's half "
                 f"(queenside {h['queenside']}, centre {h['centre']}, kingside {h['kingside']})"
                 + (f"; pawns already there: {' '.join(s['pawns_past_the_middle'])}."
                    if s["pawns_past_the_middle"] else "."))
    return "\n".join(L)


change, piece_quality, king_safety, space, alternatives = map(
    safe, (change, piece_quality, king_safety, space, alternatives))

if __name__ == "__main__":
    fen = sys.argv[1]
    b = chess.Board(fen)
    if len(sys.argv) > 2:
        print(change(b, sys.argv[2]), "\n")
        b.push_san(sys.argv[2])
    print(piece_quality(b), "\n")
    print(king_safety(b), "\n")
    print(space(b))
