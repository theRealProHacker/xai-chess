"""Check a written note against the board, mechanically. No judge, no model.

Every claim a note makes about squares, pieces, lines and files is either true of the position or
it is not, and python-chess knows which. This finds the claims it can recognise and checks them;
what it cannot recognise (judgement, plans, forecasts) it leaves alone and counts as unchecked.

    check(fen, text) -> [{"kind", "claim", "why"}]        violations only
    score(fen, text) -> {"checked", "violations", ...}

CLI:  python verify.py runs/<name>/dev.jsonl
"""
import json
import re
import sys
from pathlib import Path

import chess
import board_tools as bt

PIECES = {"knight": chess.KNIGHT, "bishop": chess.BISHOP, "rook": chess.ROOK,
          "queen": chess.QUEEN, "king": chess.KING, "pawn": chess.PAWN}
PW = "|".join(PIECES)
VERBS = r"attacks|hits|eyes|covers|guards|defends|protects|controls|bears on|aims at"

PIECE_AT = re.compile(rf"\b({PW})\s+(?:on|at)\s+([a-h][1-8])\b", re.I)
AT_PIECE = re.compile(rf"\b([a-h][1-8])[-\s]({PW})\b", re.I)
# third person only, so "under attack by", "control of" and "to defend" do not match
ATTACKS = re.compile(rf"\b({VERBS})\s+(?:the\s+)?(?:[\w'-]+\s+){{0,3}}?(?:on\s+|at\s+)?([a-h][1-8])\b", re.I)
LOOSE = re.compile(rf"\b([a-h][1-8])(?:\s+(?:{PW}))?\s+(?:is|was|stands?)\s+(?:now\s+|still\s+)?"
                   r"(hanging|en prise|undefended|unprotected)\b", re.I)
LOOSE2 = re.compile(rf"\b(hanging|undefended|unprotected|loose)\s+(?:{PW})\s+(?:on\s+)?([a-h][1-8])\b", re.I)
FILE_IS = re.compile(r"\b(?:the\s+)?([a-h])[-\s]file\s+(?:is\s+|stands\s+)?(open|half-open|closed)\b", re.I)
IS_FILE = re.compile(r"\b(open|half-open|closed)\s+([a-h])[-\s]file\b", re.I)
PAWN_LABEL = re.compile(r"\b(isolated|doubled|backward|passed)\s+(?:\w+\s+){0,2}?pawn\s+(?:on\s+)?([a-h][1-8])\b", re.I)
SAN = re.compile(r"(?<![\w.])(O-O-O|O-O|[NBRQK][a-h1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?|[a-h]x[a-h][1-8](?:=[NBRQ])?[+#]?)(?![\w])")


def _occupants(board):
    return {chess.square_name(s): p for s, p in board.piece_map().items()}


# Claims about the position as it stands. The human reference comments trip these rarely.
HARD = ("wrong piece", "geometry", "false hanging", "file status", "pawn label", "phantom piece")
# Annotators name moves from lines and from the future all the time, so this one is a diagnostic,
# never a reason to delete a sentence: 128 of the 300 human comments contain such a move.
SOFT = ("illegal move",)


def check(fen, text, kinds=None):
    board = chess.Board(fen)
    occ = _occupants(board)
    out = []

    def bad(kind, claim, why):
        if kinds is None or kind in kinds:
            out.append({"kind": kind, "claim": claim.strip(), "why": why})

    # 1. a named piece is standing on a named square
    seen = []
    for m in list(PIECE_AT.finditer(text)) + list(AT_PIECE.finditer(text)):
        word, sq = (m.group(1), m.group(2)) if m.re is PIECE_AT else (m.group(2), m.group(1))
        want, p = PIECES[word.lower()], occ.get(sq.lower())
        seen.append((m.start(), sq.lower()))
        if p is None:
            bad("phantom piece", m.group(0), f"{sq} is empty")
        elif p.piece_type != want:
            bad("wrong piece", m.group(0), f"{sq} holds a {bt.NAME[p.piece_type]}")
    seen.sort()

    # 2. X attacks / defends / covers Y — the subject is the last piece named in the same sentence
    for m in ATTACKS.finditer(text):
        target = m.group(2).lower()
        head = text[:m.start()]
        cut = max(head.rfind("."), head.rfind(";"), m.start() - 60)
        subj = [sq for pos, sq in seen if cut < pos < m.start()]
        if not subj or subj[-1] == target:
            continue
        src = chess.parse_square(subj[-1])
        if board.piece_at(src) is None:
            continue
        if chess.parse_square(target) not in board.attacks(src):
            bad("geometry", text[max(0, m.start() - 40):m.end()],
                f"the piece on {subj[-1]} does not reach {target}")

    # 3. hanging / undefended
    for m in list(LOOSE.finditer(text)) + list(LOOSE2.finditer(text)):
        sq = (m.group(1) if m.re is LOOSE else m.group(2)).lower()
        p = board.piece_at(chess.parse_square(sq))
        if p is None:
            continue
        defenders = board.attackers(p.color, chess.parse_square(sq))
        attackers = board.attackers(not p.color, chess.parse_square(sq))
        if defenders and not any(r["square"] == sq for r in bt.hanging_data(board)):
            bad("false hanging", m.group(0),
                f"{sq} is defended from {bt._sqs(defenders)} and cannot be won")
        elif not attackers and not defenders:
            bad("false hanging", m.group(0), f"nothing attacks {sq}")

    # 4. file status
    files = bt.pawn_structure_data(board)["files"]
    for m in list(FILE_IS.finditer(text)) + list(IS_FILE.finditer(text)):
        fl, said = (m.group(1), m.group(2)) if m.re is FILE_IS else (m.group(2), m.group(1))
        truth = files[fl.lower()]
        if not truth.startswith(said.lower()):
            bad("file status", m.group(0), f"the {fl}-file is {truth}")

    # 5. pawn labels
    st = bt.pawn_structure_data(board)
    for m in PAWN_LABEL.finditer(text):
        label, sq = m.group(1).lower(), m.group(2).lower()
        p = board.piece_at(chess.parse_square(sq))
        if p is None or p.piece_type != chess.PAWN:
            bad("phantom piece", m.group(0), f"there is no pawn on {sq}")
        elif sq not in st[bt._side(p.color)][label]:
            bad("pawn label", m.group(0), f"the pawn on {sq} is not {label}")

    # 6. named moves have to be playable: from the position, or as the continuation of a line
    line = board.copy(stack=False)
    for m in SAN.finditer(text):
        san = m.group(1).rstrip("!?")
        for b in (line, bt._as_mover(line, not line.turn)):
            try:
                b.push_san(san)          # a line the note is spelling out
                line = b
                break
            except ValueError:
                pass
        else:
            for b in (board, bt._as_mover(board, not board.turn)):
                try:
                    b.copy(stack=False).push_san(san)   # a single move, either side
                    line = board.copy(stack=False)
                    break
                except ValueError:
                    pass
            else:
                bad("illegal move", san, bt.legal(board, san))
                line = board.copy(stack=False)
    return out


def score(fen, text):
    v = check(fen, text)
    kinds = {}
    for r in v:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {"violations": len(v), "kinds": kinds, "clean": not v}


def main():
    path = Path(sys.argv[1])
    import program
    pool = program.by_id("train", "dev")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n_bad, kinds, examples = 0, {}, []
    for r in rows:
        e = pool[r["id"]]
        v = check(e["fen"], r.get("gen", ""))
        if v:
            n_bad += 1
            examples.append((r["id"], v))
        for x in v:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    print(f"{path}: {len(rows)} notes, {n_bad} with at least one false claim "
          f"({100 * n_bad / max(1, len(rows)):.0f}%), {sum(kinds.values())} claims wrong")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {k}")
    if "-v" in sys.argv:
        for i, v in examples[:20]:
            print(f"\n{i}")
            for x in v:
                print(f"  [{x['kind']}] {x['claim']!r} -> {x['why']}")


if __name__ == "__main__":
    main()
