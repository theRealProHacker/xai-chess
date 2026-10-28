"""Supervised board tasks beside qa.py's eval: question templates, answer formats, parsers, scores.

sample(task, rng, fen, sf) -> row dict (task, fmt, question, answer + whatever score needs), computed from the
fen on the fly so train resamples the piece/move every epoch; sf is the position's sf.jsonl "lines" (best, pv).
score(row, text, engine=None) -> metrics; best's cp loss off the top 5 and every pv ply need a Stockfish engine.

  best         best move (SAN)                   exact, top5, cp loss (Stockfish depth 18 MultiPV 5 labels)
  pv           first 6 plies of the top line     accurate prefix: plies until one is illegal or loses > 50 cp
  control      squares piece x controls          set F1; direct attacks plus batteries (features/src/geo.rs)
  attacks      enemy pieces piece x attacks      set F1 on squares (+ piece kinds); no x-rays; king included;
                                                 half the pieces asked about attack something
  phase        opening / middlegame / endgame    accuracy; lichess Divider on this position alone (context.rs)
  move_facts   rule facts of a move              set F1: captures, en passant, check, mate, stalemate, castles
  move_attacks attack changes of a move          set F1: attacks, discovers, leaves undefended
  move_board   board after a move                exact placement field + per-square accuracy

Moves in move_* questions are named by piece and squares ("the knight on g5 to f6"), never SAN, which would
give away capture and check.
"""
import re
import chess, chess.engine

TASKS = ["best", "pv", "control", "attacks", "phase", "move_facts", "move_attacks", "move_board"]
PV_LEN = 6
PV_TOL = 50          # cp a pv move may lose against the best move and still count
PV_DEPTH = 12        # depth of the per-ply check at score time
MATE = 10000
NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
         chess.QUEEN: "queen", chess.KING: "king"}
KINDS = {v: k for k, v in NAMES.items()}
PHASES = ["opening", "middlegame", "endgame"]

QUESTIONS = {
    "best": [("What is the best move?", "AB"), ("What should the side to move play?", "AB"),
             ("Find the strongest move.", "AB")],
    "pv": [("What is the main line?", "AB"), ("Give the principal variation.", "AB"),
           ("How should play continue?", "AB")],
    "control": [("Which squares does {piece} control?", "AB"), ("What does {piece} control?", "AB")],
    "attacks": [("Which pieces does {piece} attack?", "ABC"), ("What is {piece} attacking?", "ABC")],
    "phase": [("What stage of the game is this?", "AB"), ("Is this the opening, middlegame or endgame?", "AB")],
    "move_facts": [("What happens after {move}?", "A"), ("What does {move} do?", "A")],
    "move_attacks": [("What changes in the attacks after {move}?", "A"),
                     ("Which attacks does {move} create or give up?", "A")],
    "move_board": [("What does the board look like after {move}?", "A"),
                   ("Give the piece placement after {move}.", "A")],
}


# ---- names ----
def piece_name(b, s):
    return f"the {NAMES[b.piece_type_at(s)]} on {chess.square_name(s)}"


def move_name(b, m):
    t = f"{piece_name(b, m.from_square)} to {chess.square_name(m.to_square)}"
    return t + f" promoting to a {NAMES[m.promotion]}" if m.promotion else t


def join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def split(t):
    return [x.strip() for x in re.split(r",\s*|\s+and\s+", t.strip()) if x.strip()]


_PIECE = re.compile(r"the (pawn|knight|bishop|rook|queen|king) on ([a-h][1-8])")


def pieces_of(t):
    """'the rook on d6 and the pawn on c5' -> {(square, kind)} or None if any item does not parse."""
    out = set()
    for x in split(t):
        m = _PIECE.fullmatch(x)
        if not m:
            return None
        out.add((chess.parse_square(m[2]), KINDS[m[1]]))
    return out


def f1(pred, true):
    if not pred and not true:
        return 1.0
    tp = len(pred & true)
    return 2 * tp / (len(pred) + len(true)) if tp else 0.0


# ---- control: direct attacks plus batteries, per piece ----
DIRS = {"straight": [(1, 0), (-1, 0), (0, 1), (0, -1)], "diag": [(1, 1), (1, -1), (-1, 1), (-1, -1)]}


def _slides(kind, line):
    return kind == chess.QUEEN or kind == (chess.ROOK if line == "straight" else chess.BISHOP)


def control(b, s):
    """Squares the piece on s controls: geo.rs's counting rule read per piece. A slider keeps going through a
    friendly slider of the same line, and one square past a friendly pawn that captures along its diagonal."""
    p = b.piece_at(s)
    out = set(b.attacks(s))
    for line, dirs in DIRS.items():
        if not _slides(p.piece_type, line):
            continue
        for df, dr in dirs:
            f, r = chess.square_file(s) + df, chess.square_rank(s) + dr
            while 0 <= f < 8 and 0 <= r < 8:
                q = b.piece_at(chess.square(f, r))
                out.add(chess.square(f, r))
                if q is not None:
                    if q.color != p.color:
                        break
                    if _slides(q.piece_type, line):
                        pass
                    elif q.piece_type == chess.PAWN and line == "diag" and dr == (1 if q.color else -1):
                        f2, r2 = f + df, r + dr
                        if 0 <= f2 < 8 and 0 <= r2 < 8:
                            out.add(chess.square(f2, r2))
                        break
                    else:
                        break
                f, r = f + df, r + dr
    return out


def control_counts(b, sq):
    """[white, black] attacker count of sq, a literal port of geo.rs Geo::control (for checking control())."""
    n = [0, 0]
    f, r = chess.square_file(sq), chess.square_rank(sq)
    at = lambda ff, rr: b.piece_at(chess.square(ff, rr)) if 0 <= ff < 8 and 0 <= rr < 8 else None
    for df in (-1, 1):
        if (p := at(f + df, r - 1)) and p == chess.Piece(chess.PAWN, chess.WHITE): n[0] += 1
        if (p := at(f + df, r + 1)) and p == chess.Piece(chess.PAWN, chess.BLACK): n[1] += 1
    for df, dr in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
        if (p := at(f + df, r + dr)) and p.piece_type == chess.KNIGHT: n[0 if p.color else 1] += 1
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if (df, dr) != (0, 0) and (p := at(f + df, r + dr)) and p.piece_type == chess.KING: n[0 if p.color else 1] += 1
    for line, dirs in DIRS.items():
        for df, dr in dirs:
            ff, rr, col, step = f + df, r + dr, None, 1
            while 0 <= ff < 8 and 0 <= rr < 8:
                p = at(ff, rr)
                if p is not None:
                    if col is not None and col != p.color:
                        break
                    if _slides(p.piece_type, line):
                        n[0 if p.color else 1] += 1; col = p.color
                    elif line == "diag" and step == 1 and p.piece_type == chess.PAWN and dr == (-1 if p.color else 1):
                        col = p.color
                    else:
                        break
                ff, rr, step = ff + df, rr + dr, step + 1
    return n


# ---- phase: lichess Divider (scalachess), this position alone; port of features/src/context.rs ----
def _mix_score(y, w, k):
    return {(0, 1): 1 + y, (0, 2): 2 + (6 - y) if y < 6 else 0, (0, 3): 3 + (7 - y) if y < 7 else 0,
            (0, 4): 3 + (7 - y) if y < 7 else 0, (1, 0): 1 + (8 - y), (1, 1): 5 + abs(4 - y),
            (1, 2): 4 + (7 - y), (1, 3): 5 + (7 - y), (2, 0): 2 + (y - 2) if y > 2 else 0,
            (2, 1): 4 + (y - 1), (2, 2): 7, (3, 0): 3 + (y - 1) if y > 1 else 0,
            (4, 0): 3 + (y - 1) if y > 1 else 0, (3, 1): 5 + (y - 1)}.get((w, k), 0)


def phase(b):
    white, black = b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK]
    mm = chess.popcount(b.occupied & ~(b.kings | b.pawns))
    mix = sum(_mix_score(y + 1, chess.popcount(white & (0x0303 << (x + 8 * y))), chess.popcount(black & (0x0303 << (x + 8 * y))))
              for y in range(7) for x in range(7))
    if mm <= 6:
        return "endgame"
    if mm <= 10 or chess.popcount(white & chess.BB_RANK_1) < 4 or chess.popcount(black & chess.BB_RANK_8) < 4 or mix > 150:
        return "middlegame"
    return "opening"


# ---- moves ----
def pick_move(rng, b):
    """Half the time a capture, check or castle when there is one; otherwise uniform."""
    moves = list(b.legal_moves)
    sharp = [m for m in moves if b.is_capture(m) or b.gives_check(m) or b.is_castling(m)]
    pool = sharp if sharp and rng.random() < 0.5 else moves
    return pool[int(rng.integers(len(pool)))]


def move_facts(b, m):
    """Set of fact strings, in the order they are rendered."""
    out = []
    if b.is_castling(m):
        out.append("It castles kingside." if chess.square_file(m.to_square) > chess.square_file(m.from_square) else "It castles queenside.")
    if b.is_en_passant(m):
        out.append(f"It captures the pawn on {chess.square_name(m.to_square ^ 8)} en passant.")
    elif b.is_capture(m):
        out.append(f"It captures {piece_name(b, m.to_square)}.")
    a = b.copy(stack=False); a.push(m)
    if a.is_checkmate():
        out.append("It is checkmate.")
    elif a.is_check():
        out.append("It gives check.")
    elif a.is_stalemate():
        out.append("It is stalemate.")
    return out


def _attacked_by(b, squares, col):
    """{(target square, kind)} of `col`'s opponent pieces attacked from `squares`."""
    out = set()
    for s in squares:
        for t in b.attacks(s) & b.occupied_co[not col]:
            out.add((t, b.piece_type_at(t)))
    return out


def move_attacks(b, m):
    """(attacks, discovers, undefended): enemy pieces newly attacked by the moved piece(s); enemy pieces newly
    attacked by the mover's other pieces; the mover's other pieces (not the king) defended before and not after."""
    col = b.turn
    a = b.copy(stack=False); a.push(m)
    if b.is_castling(m):
        rank = chess.square_rank(m.from_square)
        ks = chess.square_file(m.to_square) > chess.square_file(m.from_square)
        rook = m.to_square if b.piece_type_at(m.to_square) == chess.ROOK else chess.square(7 if ks else 0, rank)
        moved_before = {m.from_square, rook}
        moved_after = {a.king(col), chess.square(5 if ks else 3, rank)}
    else:
        moved_before, moved_after = {m.from_square}, {m.to_square}
    others_b = [s for s in chess.SquareSet(b.occupied_co[col]) if s not in moved_before]
    others_a = [s for s in chess.SquareSet(a.occupied_co[col]) if s not in moved_after]
    attacks = _attacked_by(a, moved_after, col) - _attacked_by(b, moved_before, col)
    discovers = _attacked_by(a, others_a, col) - _attacked_by(b, others_b, col) - attacks
    undefended = {(s, b.piece_type_at(s)) for s in others_b if b.piece_type_at(s) != chess.KING
                  and b.is_attacked_by(col, s) and not a.is_attacked_by(col, s) and s in others_a}
    return attacks, discovers, undefended


def _pl(items):
    return join([f"the {NAMES[k]} on {chess.square_name(s)}" for s, k in sorted(items)])


def render_move_attacks(at, di, un):
    out = []
    if at: out.append(f"Attacks {_pl(at)}.")
    if di: out.append(f"Discovers an attack on {_pl(di)}.")
    if un: out.append(f"Leaves {_pl(un)} undefended.")
    return " ".join(out) or "Nothing changes."


_MA = re.compile(r"(?:Attacks (?P<at>[^.]+)\.)?\s*(?:Discovers an attack on (?P<di>[^.]+)\.)?\s*(?:Leaves (?P<un>[^.]+) undefended\.)?")


# ---- sample / render ----
def _san_line(b, ucis, numbered):
    b = b.copy(stack=False); out = []
    for k, u in enumerate(ucis):
        m = chess.Move.from_uci(u)
        if numbered and (b.turn or k == 0):
            out.append(f"{b.fullmove_number}." if b.turn else f"{b.fullmove_number}...")
        out.append(b.san(m)); b.push(m)
    return " ".join(out)


def sample(task, rng, fen, sf=None):
    b = chess.Board(fen)
    q = int(rng.integers(len(QUESTIONS[task])))
    qt, fmts = QUESTIONS[task][q]
    fmt = fmts[int(rng.integers(len(fmts)))]
    row = dict(task=task, q=q, fmt=fmt, fen=fen)
    if task in ("best", "pv"):
        if not sf:
            return None
        if task == "best":
            san = b.san(chess.Move.from_uci(sf[0]["uci"]))
            ans = san if fmt == "A" else f"The best move is {san}."
            row.update(target=sf[0]["uci"], top=[[l["uci"], cp(l)] for l in sf])
        else:
            pv = sf[0]["pv"][:PV_LEN]
            ans = _san_line(b, pv, fmt == "B")
            row.update(target=pv)
        return dict(row, question=qt, answer=ans)
    if task == "phase":
        ph = phase(b)
        return dict(row, question=qt, target=ph, answer=ph.capitalize() if fmt == "A" else f"This is the {ph}.")
    if task in ("control", "attacks"):
        sqs = sorted(b.piece_map())
        if task == "attacks":                          # half the time a piece that attacks something
            hit = [x for x in sqs if b.attacks(x) & b.occupied_co[not b.color_at(x)]]
            sqs = hit if hit and rng.random() < 0.5 else sqs
        s = sqs[int(rng.integers(len(sqs)))]
        name = piece_name(b, s)
        if task == "control":
            c = sorted(control(b, s))
            names = [chess.square_name(t) for t in c]
            ans = ", ".join(names) if fmt == "A" else f"{name[0].upper() + name[1:]} controls {join(names)}."
            return dict(row, question=qt.format(piece=name), square=s, target=c, answer=ans)
        col = b.color_at(s)
        tg = sorted(t for t in b.attacks(s) if b.color_at(t) == (not col))
        if not tg:
            ans = "Nothing."
        elif fmt == "A":
            ans = join([piece_name(b, t) for t in tg]) + "."
        elif fmt == "B":
            ans = ", ".join((b.piece_at(t).symbol().upper() if b.piece_type_at(t) != chess.PAWN else "") + chess.square_name(t) for t in tg)
        else:
            ans = ", ".join(chess.square_name(t) for t in tg)
        return dict(row, question=qt.format(piece=name), square=s, target=[[t, b.piece_type_at(t)] for t in tg], answer=ans)
    m = pick_move(rng, b)
    row.update(question=qt.format(move=move_name(b, m)), move=m.uci())
    if task == "move_facts":
        fx = move_facts(b, m)
        return dict(row, target=fx, answer=" ".join(fx) or "Nothing special.")
    if task == "move_attacks":
        at, di, un = move_attacks(b, m)
        return dict(row, target=[sorted(map(list, x)) for x in (at, di, un)], answer=render_move_attacks(at, di, un))
    a = b.copy(stack=False); a.push(m)
    return dict(row, target=a.board_fen(), answer=a.board_fen())


# ---- parse ----
_SQ = re.compile(r"[a-h][1-8]")


def parse(row, text):
    """Model answer -> the value score() compares with row["target"], or None if it does not parse."""
    t, task, fmt, b = text.strip(), row["task"], row["fmt"], chess.Board(row["fen"])
    try:
        if task == "best":
            m = re.fullmatch(r"(?:The best move is )?(\S+?)\.?", t)
            return b.parse_san(m[1]).uci() if m else None
        if task == "pv":
            out = []
            for tok in t.split():
                if re.fullmatch(r"\d+\.(\.\.)?", tok):
                    continue
                try:
                    mv = b.parse_san(tok)
                except ValueError:
                    out.append(None); break             # illegal: the line ends here, unscored past this
                out.append(mv.uci()); b.push(mv)
            return out
        if task == "phase":
            m = re.fullmatch(r"(?:This is the )?(opening|middlegame|endgame)\.?", t, re.I)
            return m[1].lower() if m else None
        if task == "control":
            m = re.fullmatch(r"The \w+ on [a-h][1-8] controls (.+)\.", t)
            items = split(m[1] if m else t)
            return sorted(chess.parse_square(x) for x in items) if all(_SQ.fullmatch(x) for x in items) else None
        if task == "attacks":
            if t == "Nothing.":
                return []
            if fmt == "A":
                p = pieces_of(t.rstrip("."))
                return None if p is None else sorted([s, k] for s, k in p)
            out = []
            for x in split(t):
                m = re.fullmatch(r"([NBRQK]?)([a-h][1-8])" if fmt == "B" else r"()([a-h][1-8])", x)
                if not m:
                    return None
                kind = chess.Piece.from_symbol(m[1]).piece_type if m[1] else (chess.PAWN if fmt == "B" else None)
                out.append([chess.parse_square(m[2]), kind])
            return sorted(out, key=lambda x: x[0])
        if task == "move_facts":
            if t == "Nothing special.":
                return []
            return re.findall(r"It [^.]+\.", t) if re.fullmatch(r"(It [^.]+\.\s*)+", t) else None
        if task == "move_attacks":
            if t == "Nothing changes.":
                return [[], [], []]
            m = _MA.fullmatch(t)
            if not m or not t:
                return None
            out = []
            for g in ("at", "di", "un"):
                p = pieces_of(m[g]) if m[g] else set()
                if p is None:
                    return None
                out.append(sorted([s, k] for s, k in p))
            return out
        if task == "move_board":
            chess.BaseBoard(t)
            return t
    except (ValueError, KeyError, IndexError):
        return None
    raise ValueError(task)


# ---- score ----
def cp(line):
    """Stockfish line score in cp from the side to move, mates mapped to +-(MATE - plies)."""
    if "mate" in line:
        return MATE - abs(line["mate"]) if line["mate"] > 0 else -MATE + abs(line["mate"])
    return line["cp"]


def _cp(score):
    return MATE - abs(score.mate()) if score.is_mate() and score.mate() > 0 else (
        -MATE + abs(score.mate()) if score.is_mate() else score.score())


def move_loss(engine, b, m, depth=PV_DEPTH):
    """cp the move loses against the engine's best at `depth`, capped at 1000."""
    lim = chess.engine.Limit(depth=depth)
    best = _cp(engine.analyse(b, lim)["score"].pov(b.turn))
    got = _cp(engine.analyse(b, lim, root_moves=[m])["score"].pov(b.turn))
    return min(1000, max(0, best - got))


def score(row, text, engine=None):
    p = parse(row, text)
    task, tg = row["task"], row["target"]
    out = dict(task=task, parsed=p is not None)
    if p is None:
        return out
    if task == "best":
        top = dict(row["top"])
        out.update(exact=p == tg, top5=p in top)
        if p in top:
            out["cp_loss"] = min(1000, top[tg] - top[p])
        elif engine is not None:
            out["cp_loss"] = move_loss(engine, chess.Board(row["fen"]), chess.Move.from_uci(p), 18)
    elif task == "pv":
        b, n = chess.Board(row["fen"]), 0
        for u in p[:PV_LEN]:
            if u is None or engine is None:
                break
            m = chess.Move.from_uci(u)
            if move_loss(engine, b, m) > PV_TOL:
                break
            b.push(m); n += 1
        if engine is not None:
            full = len(tg) if len(tg) < PV_LEN else PV_LEN                 # a shorter label line ends in mate
            out.update(acc_len=n, full=n >= full)
        out.update(prefix=next((k for k, (x, y) in enumerate(zip(p, tg)) if x != y), min(len(p), len(tg))))
    elif task == "phase":
        out["exact"] = p == tg
    elif task == "control":
        out.update(exact=p == tg, f1=f1(set(p), set(tg)))
    elif task == "attacks":
        ps, ts = {s for s, _ in p}, {s for s, _ in tg}
        out.update(exact=ps == ts, f1=f1(ps, ts))
        if row["fmt"] != "C":
            out["kind_ok"] = {tuple(x) for x in p} == {tuple(x) for x in tg}
    elif task == "move_facts":
        out.update(exact=set(p) == set(tg), f1=f1(set(p), set(tg)))
    elif task == "move_attacks":
        ps = {(g, s, k) for g, xs in enumerate(p) for s, k in xs}
        ts = {(g, s, k) for g, xs in enumerate(tg) for s, k in xs}
        out.update(exact=ps == ts, f1=f1(ps, ts))
    elif task == "move_board":
        pb, tb = chess.BaseBoard(p), chess.BaseBoard(tg)
        out.update(exact=p == tg, square_acc=sum(pb.piece_at(s) == tb.piece_at(s) for s in chess.SQUARES) / 64)
    return out
