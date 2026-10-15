"""Eval questions, answer formats, and the parser that scores answers back.

All labels are Leela's value-head WDL from the side to move (lc0 mirrors the board so the side to
move is "us"). Question 8 names a colour; its answer is from that colour's view.

Formats:
  A  pawns                     +1.35
  B  better side's win chance  48% win for the side to move
  C  expected score            0.69 expected score for the side to move
  D  verbal                    The side to move is clearly better.
  E  verbal + pawns            Slightly better for the side to move (+0.45).
"""
import math, re

QUESTIONS = [
    ("What is the evaluation of this position?", "AE"),
    ("What's the eval?", "AE"),
    ("Who is better here, and by how much?", "BE"),
    ("How good is this position for the side to move?", "CDE"),
    ("What are the winning chances?", "BC"),
    ("Assess this position.", "DE"),
    ("Is the side to move better, worse, or equal?", "DE"),
    ("How is the position for {colour}?", "ADE"),
]
COLOUR_Q = 7
PAWN_CLIP = 10.0
BINS = [  # (upper bound in pawns, D phrase, E phrase); mirrored for negative evals
    (0.3, "equal", "Equal"),
    (0.7, "slightly better", "Slightly better"),
    (2.0, "clearly better", "Clearly better"),
    (math.inf, "winning", "Winning"),
]
WORSE = {"slightly better": "slightly worse", "clearly better": "clearly worse", "winning": "losing",
         "Slightly better": "Slightly worse", "Clearly better": "Clearly worse", "Winning": "Losing"}


def pawns(w, l):
    """lc0's Q -> centipawn formula, in pawns, clipped."""
    q = max(-0.999, min(0.999, w - l))
    return max(-PAWN_CLIP, min(PAWN_CLIP, 0.9 * math.tan(1.5637541897 * q)))


def bin_of(p):
    """Signed bin index -3..3."""
    i = next(k for k, (ub, _, _) in enumerate(BINS) if abs(p) < ub)
    return i if p >= 0 else -i


def view(wdl, black_to_move, q, colour):
    """(w, d, l, subject) from the perspective the question asks for."""
    w, d, l = wdl
    if q != COLOUR_Q:
        return w, d, l, "the side to move"
    stm = "Black" if black_to_move else "White"
    return (w, d, l, colour) if colour == stm else (l, d, w, colour)


def render(wdl, black_to_move, q, fmt, colour="White"):
    w, d, l, who = view(wdl, black_to_move, q, colour)
    p = pawns(w, l)
    b = bin_of(p)
    if fmt == "A":
        return f"{p:+.2f}"
    if fmt == "B":
        return f"{round(100 * w)}% win for the side to move" if w >= l else f"{round(100 * l)}% win for the opponent"
    if fmt == "C":
        return f"{w + d / 2:.2f} expected score for the side to move"
    _, dp, ep = BINS[abs(b)]
    if fmt == "D":
        if b == 0:
            return "The position is equal."
        who_cap = who[0].upper() + who[1:]
        return f"{who_cap} is {dp if b > 0 else WORSE[dp]}."
    if fmt == "E":
        if b == 0:
            return f"Equal ({p:+.2f})."
        return f"{ep if b > 0 else WORSE[ep]} for {who} ({p:+.2f})."
    raise ValueError(fmt)


def question(q, colour="White"):
    return QUESTIONS[q][0].format(colour=colour)


def sample(rng, wdl, black_to_move):
    """Random (q, fmt, colour, question text, answer text) for one position."""
    q = int(rng.integers(len(QUESTIONS)))
    fmt = QUESTIONS[q][1][int(rng.integers(len(QUESTIONS[q][1])))]
    colour = "White" if rng.random() < 0.5 else "Black"
    return q, fmt, colour, question(q, colour), render(wdl, black_to_move, q, fmt, colour)


# ---- parsing ----
_SUBJ = r"(the side to move|White|Black)"
_D_PHR = "|".join(re.escape(s) for s in ["slightly better", "clearly better", "winning",
                                          "slightly worse", "clearly worse", "losing"])
_E_PHR = "|".join(re.escape(s) for s in ["Slightly better", "Clearly better", "Winning",
                                          "Slightly worse", "Clearly worse", "Losing"])
PATTERNS = {
    "A": re.compile(r"([+-]\d+\.\d\d)"),
    "B": re.compile(r"(\d{1,3})% win for (the side to move|the opponent)"),
    "C": re.compile(r"(0\.\d\d|1\.00) expected score for the side to move"),
    "D": re.compile(rf"(?:The position is equal|({_SUBJ[1:-1]}) is ({_D_PHR}))\.", re.I),
    "E": re.compile(rf"(?:Equal|({_E_PHR}) for {_SUBJ}) \(([+-]\d+\.\d\d)\)\."),
}
_BIN_IDX = {"equal": 0, "slightly better": 1, "clearly better": 2, "winning": 3,
            "slightly worse": -1, "clearly worse": -2, "losing": -3}


def parse(text):
    """-> dict(fmt, pawns?, bin?, win?, side?, score?, subject?) or None. Whole answer must match."""
    t = text.strip()
    for fmt in "EDBCA":                                     # E before A: E contains a number
        m = PATTERNS[fmt].fullmatch(t)
        if not m:
            continue
        if fmt == "A":
            p = float(m[1]); return dict(fmt=fmt, pawns=p, bin=bin_of(p))
        if fmt == "B":
            return dict(fmt=fmt, win=int(m[1]) / 100, side=1 if m[2] == "the side to move" else -1)
        if fmt == "C":
            return dict(fmt=fmt, score=float(m[1]))
        if fmt == "D":
            if m[1] is None:
                return dict(fmt=fmt, bin=0, subject=None)
            return dict(fmt=fmt, bin=_BIN_IDX[m[2].lower()], subject=m[1])
        p = float(m[3])
        return dict(fmt=fmt, pawns=p, bin=_BIN_IDX[m[1].lower()] if m[1] else 0, subject=m[2])
    return None
