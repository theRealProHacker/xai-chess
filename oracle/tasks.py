"""Supervised board tasks beside qa.py's eval: question templates, answer formats, parsers, scores.

sample(task, rng, fen, sf) -> row dict (task, fmt, question, answer + whatever score needs), computed from the
fen on the fly so train resamples the piece/move every epoch; sf is the position's sf.jsonl "lines" (best, pv).
score(row, text, engine=None) -> metrics; best's cp loss off the top 5 and every pv ply need a Stockfish engine.

  best         best move (SAN)                   exact, top5, cp loss (Stockfish depth 18 MultiPV 5 labels)
  pv           first 6 plies of the top line     accurate prefix: plies until one is illegal or loses > 50 cp
  multipv      top 3 lines of 6 plies + evals     each stated eval vs Stockfish depth 12 at the end of the line as
                                                 written (needs an engine): pawn MAE, bin right
  control      squares the piece on x controls   set F1; direct attacks plus batteries (features/src/geo.rs);
                                                 the question names only the square, empty with prob CONTROL_EMPTY
  attacks      enemy pieces piece x attacks      set F1 on squares (+ piece kinds); no x-rays; king included;
                                                 half the pieces asked about attack something
  phase        opening / middlegame / endgame    accuracy; lichess Divider on this position alone (context.rs)
  move_facts   rule facts of a move              set F1: captures, en passant, check, mate, stalemate, castles
  move_attacks attack changes of a move          set F1: attacks, discovers, leaves undefended
  move_board   board after a move                exact placement field + per-square accuracy
Minimal v1 ladder (moves in UCI, colours absolute):
  inventory    pieces per colour and kind        exact + share of the 12 counts right
  lookup       what is on a square               exact; half the squares asked about are occupied
  render       piece placement (FEN field 1)     exact + per-square accuracy
  piece_moves  legal moves of the piece on x     set F1; side to move's pieces, half of them with a move
  legal_moves  all legal moves                   set F1
  apply_move   squares a UCI move changes        set F1 on (square, new content)
  result_fen   FEN after a UCI move              exact on the first four fields (no move counters) + per-square
  check        is the side to move in check      exact + F1 on the checkers; POOLED: half the samples in check
  mate1        all mates in one                  set F1; POOLED: half the samples have one

Moves in move_* questions are named by piece and squares ("the knight on g5 to f6"), never SAN, which would
give away capture and check.
"""
import re
import chess, chess.engine
import qa

TASKS = ["best", "pv", "control", "attacks", "phase", "move_facts", "move_attacks", "move_board", "multipv",
         "inventory", "lookup", "render", "piece_moves", "legal_moves", "apply_move", "result_fen", "check", "mate1"]
V1_TASKS = TASKS[9:]
SF_TASKS = ("best", "pv", "multipv")    # need sf.jsonl labels
PV_LEN = 6
CONTROL_EMPTY = 0.1     # share of control questions about an empty square
MPV_N = 3
PV_TOL = 50          # cp a pv move may lose against the best move and still count
PV_DEPTH = 12        # depth of the per-ply check at score time
MATE = 10000
NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
         chess.QUEEN: "queen", chess.KING: "king"}
KINDS = {v: k for k, v in NAMES.items()}
PHASES = ["opening", "middlegame", "endgame"]
KIND_ORDER = "KQRBNP"
POOL_SHARE = 0.5        # share of a POOLED task's train samples drawn from positions where the answer is not "no"

QUESTIONS = {
    "best": [("What is the best move?", "AB"), ("What should the side to move play?", "AB"),
             ("Find the strongest move.", "AB")],
    "pv": [("What is the main line?", "AB"), ("Give the principal variation.", "AB"),
           ("How should play continue?", "AB")],
    "control": [("Which squares does the piece on {square} control?", "AB"), ("What does the piece on {square} control?", "AB")],
    "attacks": [("Which pieces does {piece} attack?", "ABC"), ("What is {piece} attacking?", "ABC")],
    "multipv": [("What are the three best lines and their evaluations?", "A"),
                ("Give the top three lines with evaluations.", "A")],
    "phase": [("What stage of the game is this?", "AB"), ("Is this the opening, middlegame or endgame?", "AB")],
    "move_facts": [("What happens after {move}?", "A"), ("What does {move} do?", "A")],
    "move_attacks": [("What changes in the attacks after {move}?", "A"),
                     ("Which attacks does {move} create or give up?", "A")],
    "move_board": [("What does the board look like after {move}?", "A"),
                   ("Give the piece placement after {move}.", "A")],
    "inventory": [("List all pieces.", "A"), ("Which pieces are on the board?", "A")],
    "lookup": [("What is on {square}?", "A"), ("Which piece stands on {square}?", "A")],
    "render": [("Give the board as FEN.", "A"), ("Write the piece placement.", "A")],
    "piece_moves": [("List the legal moves of the piece on {square}.", "A"), ("Where can the piece on {square} move?", "A")],
    "legal_moves": [("List all legal moves.", "A"), ("Which moves are legal?", "A")],
    "apply_move": [("Play {uci}. Which squares change?", "A"), ("What changes after {uci}?", "A")],
    "result_fen": [("Play {uci}. Give the resulting FEN.", "A"), ("What is the FEN after {uci}?", "A")],
    "check": [("Is the side to move in check?", "A"), ("Is the king in check?", "A")],
    "mate1": [("Find all mates in one.", "A"), ("Which moves give checkmate?", "A")],
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


def fmt_eval(c):
    """cp from the side to move -> '+0.45', or '#3' / '#-3' for mates."""
    if abs(c) >= MATE - 1000:
        return f"#{MATE - c}" if c > 0 else f"#-{MATE + c}"
    return f"{c / 100:+.2f}"


def mates_in_one(b):
    out = []
    for m in b.legal_moves:
        b.push(m)
        if b.is_checkmate(): out.append(m.uci())
        b.pop()
    return sorted(out)


POOLED = {"check": lambda b: b.is_check(), "mate1": lambda b: bool(mates_in_one(b))}


def content(b, s):
    p = b.piece_at(s)
    return f"{'white' if p.color else 'black'} {NAMES[p.piece_type]}" if p else "empty"


def inventory(b):
    out = []
    for col, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        n = [(k, len(b.pieces(chess.Piece.from_symbol(k).piece_type, col))) for k in KIND_ORDER]
        out.append(name + " " + " ".join(f"{k}{c}" for k, c in n if c))
    return "; ".join(out)


def fen4(b):
    return " ".join(b.fen().split()[:4])


def _v1(task, row, qt, rng, b):
    if task == "inventory":
        t = inventory(b); return dict(row, question=qt, target=t, answer=t)
    if task == "render":
        return dict(row, question=qt, target=b.board_fen(), answer=b.board_fen())
    if task == "lookup":
        occ = sorted(b.piece_map())
        sqs = occ if rng.random() < 0.5 else sorted(set(chess.SQUARES) - set(occ))
        s = sqs[int(rng.integers(len(sqs)))]
        c = content(b, s)
        return dict(row, question=qt.format(square=chess.square_name(s)), square=s, target=c,
                    answer="Nothing." if c == "empty" else f"A {c}.")
    if task == "piece_moves":
        own = sorted(chess.SquareSet(b.occupied_co[b.turn]))
        mov = sorted({m.from_square for m in b.legal_moves})
        sqs = mov if mov and rng.random() < 0.5 else own
        s = sqs[int(rng.integers(len(sqs)))]
        ms = sorted(m.uci() for m in b.legal_moves if m.from_square == s)
        return dict(row, question=qt.format(square=chess.square_name(s)), square=s, target=ms, answer=" ".join(ms) or "None.")
    if task == "legal_moves":
        ms = sorted(m.uci() for m in b.legal_moves)
        return dict(row, question=qt, target=ms, answer=" ".join(ms) or "None.")
    if task == "check":
        cs = sorted(b.checkers())
        ans = f"Yes, by {join([piece_name(b, x) for x in cs])}." if cs else "No."
        return dict(row, question=qt, target=cs, answer=ans)
    if task == "mate1":
        ms = mates_in_one(b)
        return dict(row, question=qt, target=ms, answer=" ".join(ms) or "None.")
    if not any(b.legal_moves): return None                       # apply_move, result_fen
    m = pick_move(rng, b)
    a = b.copy(stack=False); a.push(m)
    row.update(question=qt.format(uci=m.uci()), move=m.uci())
    if task == "apply_move":
        ch = [[s, content(a, s)] for s in chess.SQUARES if b.piece_at(s) != a.piece_at(s)]
        return dict(row, target=ch, answer=", ".join(f"{chess.square_name(s)} {c}" for s, c in ch))
    return dict(row, target=fen4(a), answer=fen4(a))


def sample(task, rng, fen, sf=None):
    b = chess.Board(fen)
    q = int(rng.integers(len(QUESTIONS[task])))
    qt, fmts = QUESTIONS[task][q]
    fmt = fmts[int(rng.integers(len(fmts)))]
    row = dict(task=task, q=q, fmt=fmt, fen=fen)
    if task in V1_TASKS:
        return _v1(task, row, qt, rng, b)
    if task in SF_TASKS:
        if not sf:
            return None
        if task == "multipv":
            ls = sf[:MPV_N]
            ans = "; ".join(f"{_san_line(b, l['pv'][:PV_LEN], False)} ({fmt_eval(cp(l))})" for l in ls)
            return dict(row, question=qt, target=[cp(l) for l in ls], answer=ans)
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
        if task == "control" and rng.random() < CONTROL_EMPTY:
            sqs = sorted(set(chess.SQUARES) - set(sqs))
        s = sqs[int(rng.integers(len(sqs)))]
        if task == "control":                          # the question names the square, never the piece
            q = qt.format(square=chess.square_name(s))
            if b.piece_at(s) is None:
                return dict(row, question=q, square=s, target="empty", answer=f"There is no piece on {chess.square_name(s)}.")
            name = piece_name(b, s)
            c = sorted(control(b, s))
            names = [chess.square_name(t) for t in c]
            ans = ", ".join(names) if fmt == "A" else f"{name[0].upper() + name[1:]} controls {join(names)}."
            return dict(row, question=q, square=s, target=c, answer=ans)
        name = piece_name(b, s)
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
_MPV = re.compile(r"(.*?)\s*\(([+-]\d+\.\d+|#-?\d+)\)")


def _line(b, t):
    """SAN line -> uci list; an illegal move ends it as None."""
    b, out = b.copy(stack=False), []
    for tok in t.split():
        if re.fullmatch(r"\d+\.(\.\.)?", tok):
            continue
        try:
            mv = b.parse_san(tok)
        except ValueError:
            out.append(None); break
        out.append(mv.uci()); b.push(mv)
    return out


def _attacks(t, fmt):
    """Attacks answer in format fmt -> sorted [square, kind] (kind None when fmt C names no piece), or None."""
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


def parse(row, text):
    """Model answer -> the value score() compares with row["target"], or None if it does not parse."""
    t, task, fmt, b = text.strip(), row["task"], row["fmt"], chess.Board(row["fen"])
    try:
        if task == "best":
            m = re.fullmatch(r"(?:The best move is )?(\S+?)\.?", t)
            return b.parse_san(m[1]).uci() if m else None
        if task == "pv":
            return _line(b, t)                          # illegal: the line ends there, unscored past it
        if task == "multipv":
            out = []
            for x in t.split(";"):
                m = _MPV.fullmatch(x.strip())
                if not m:
                    return None
                e = m[2]
                c = (MATE - int(e[1:]) if not e.startswith("#-") else -MATE + int(e[2:])) if e[0] == "#" else round(float(e) * 100)
                out.append([_line(b, m[1]), c])
            return out
        if task == "phase":
            m = re.fullmatch(r"(?:This is the )?(opening|middlegame|endgame)\.?", t, re.I)
            return m[1].lower() if m else None
        if task == "control":
            if re.fullmatch(r"There is no piece on [a-h][1-8]\.", t):
                return "empty"
            m = re.fullmatch(r"The \w+ on [a-h][1-8] controls (.+)\.", t)
            items = split(m[1] if m else t)
            return sorted(chess.parse_square(x) for x in items) if all(_SQ.fullmatch(x) for x in items) else None
        if task == "attacks":                           # the question does not name the format: any one parses
            if t == "Nothing.":
                return []
            for f in [fmt] + [f for f in "ABC" if f != fmt]:
                if (p := _attacks(t, f)) is not None:
                    return p
            return None
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
        if task in ("move_board", "render"):
            chess.BaseBoard(t)
            return t
        if task == "inventory":
            m = re.fullmatch(r"White((?: [KQRBNP]\d+)*); Black((?: [KQRBNP]\d+)*)", t)
            return [{x[0]: int(x[1:]) for x in g.split()} for g in (m[1], m[2])] if m else None
        if task == "lookup":
            if t == "Nothing.": return "empty"
            m = re.fullmatch(r"An? ((?:white|black) (?:pawn|knight|bishop|rook|queen|king))\.", t)
            return m[1] if m else None
        if task in ("piece_moves", "legal_moves", "mate1"):
            if t == "None.": return []
            xs = t.split()
            return sorted(set(xs)) if all(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", x) for x in xs) else None
        if task == "apply_move":
            out = []
            for x in t.split(", "):
                m = re.fullmatch(r"([a-h][1-8]) (empty|(?:white|black) (?:pawn|knight|bishop|rook|queen|king))", x)
                if not m: return None
                out.append([chess.parse_square(m[1]), m[2]])
            return out
        if task == "result_fen":
            f = t.split()
            if len(f) != 4: return None
            chess.BaseBoard(f[0])
            return t
        if task == "check":
            if t == "No.": return []
            m = re.fullmatch(r"Yes, by (.+)\.", t)
            ps = pieces_of(m[1]) if m else None
            return sorted(s for s, _ in ps) if ps is not None else None
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


def line_eval(engine, b, ucis):
    """cp at the end of the legal prefix of ucis, from b's side to move, Stockfish at PV_DEPTH."""
    root, b = b.turn, b.copy(stack=False)
    for u in ucis:
        if u is None:
            break
        b.push(chess.Move.from_uci(u))
    if b.is_checkmate():
        return -MATE if b.turn == root else MATE
    if b.is_game_over():
        return 0
    return _cp(engine.analyse(b, chess.engine.Limit(depth=PV_DEPTH))["score"].pov(root))


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
    elif task == "multipv":
        out["lines"] = min(len(p), MPV_N) / MPV_N
        if engine is not None:
            b, err, ok = chess.Board(row["fen"]), [], []
            for ucis, c in p[:MPV_N]:
                if not ucis or ucis[0] is None:
                    continue                            # no legal first move: no line to evaluate
                pw = lambda x: max(-qa.PAWN_CLIP, min(qa.PAWN_CLIP, x / 100))
                t = pw(line_eval(engine, b, ucis))
                err.append(abs(pw(c) - t)); ok.append(qa.bin_of(pw(c)) == qa.bin_of(t))
            if err:
                out.update(pawn_mae=sum(err) / len(err), bin=sum(ok) / len(ok))
    elif task == "phase":
        out["exact"] = p == tg
    elif task == "control":
        if "empty" in (p, tg):
            out.update(exact=p == tg, f1=float(p == tg))
        else:
            out.update(exact=p == tg, f1=f1(set(p), set(tg)))
    elif task == "attacks":
        ps, ts = {s for s, _ in p}, {s for s, _ in tg}
        out.update(exact=ps == ts, f1=f1(ps, ts))
        if all(k is not None for _, k in p):
            out["kind_ok"] = {tuple(x) for x in p} == {tuple(x) for x in tg}
    elif task == "move_facts":
        out.update(exact=set(p) == set(tg), f1=f1(set(p), set(tg)))
    elif task == "move_attacks":
        ps = {(g, s, k) for g, xs in enumerate(p) for s, k in xs}
        ts = {(g, s, k) for g, xs in enumerate(tg) for s, k in xs}
        out.update(exact=ps == ts, f1=f1(ps, ts))
    elif task in ("move_board", "render", "result_fen"):
        pb, tb = chess.BaseBoard(p.split()[0]), chess.BaseBoard(tg.split()[0])
        out.update(exact=p == tg, square_acc=sum(pb.piece_at(s) == tb.piece_at(s) for s in chess.SQUARES) / 64)
    elif task == "inventory":
        tg_ = parse(dict(row, task="inventory"), tg)
        out.update(exact=p == tg_, counts=sum(p[c].get(k, 0) == tg_[c].get(k, 0) for c in (0, 1) for k in KIND_ORDER) / 12)
    elif task == "lookup":
        out["exact"] = p == tg
    elif task in ("piece_moves", "legal_moves", "mate1", "check"):
        out.update(exact=set(p) == set(tg), f1=f1(set(p), set(tg)))
    elif task == "apply_move":
        ps, ts = {tuple(x) for x in p}, {tuple(x) for x in tg}
        out.update(exact=ps == ts, f1=f1(ps, ts))
    return out
