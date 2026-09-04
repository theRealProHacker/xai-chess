"""Lc0 INPUT_CLASSICAL_112_PLANE encoder for a single FEN, matching lc0's
EncodePositionForNN with HistoryFill=fen_only (the default when a position is
given as a FEN without moves): every history slot holds the current board, an
en-passant pawn is undone in the filled slots, and the board is mirrored so
the side to move is always "us" looking up the board.

Token / plane bit index = rank*8 + file from the side-to-move's view; that is
also python-chess's square index for white to move, and `sq ^ 56` for black.
"""
import numpy as np
import chess

PLANES = 112
PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def stm_square(sq: int, black_to_move: bool) -> int:
    """python-chess square -> token index in the network's 64-square sequence."""
    return sq ^ 56 if black_to_move else sq


def _bits(mask: int) -> np.ndarray:
    """64-bit mask -> (8, 8) float32, bit rank*8+file -> [rank, file]."""
    return np.unpackbits(np.array([mask], dtype="<u8").view(np.uint8),
                         bitorder="little").astype(np.float32).reshape(8, 8)


def encode(board: chess.Board, out: np.ndarray | None = None) -> np.ndarray:
    """(112, 8, 8) float32 input planes for `board`."""
    x = out if out is not None else np.zeros((PLANES, 8, 8), np.float32)
    x[:] = 0
    us = board.turn
    flip = us == chess.BLACK

    def bb(piece, color):
        m = board.pieces_mask(piece, color)
        return chess.flip_vertical(m) if flip else m

    ours = [bb(p, us) for p in PIECES]
    theirs = [bb(p, not us) for p in PIECES]
    cur = ours + theirs                       # 12 piece planes, current board

    # Filled history: the same board, except a just-double-pushed enemy pawn is
    # put back on its start square (lc0 undoes the move it can infer from ep).
    hist = list(cur)
    if board.ep_square is not None:
        f = chess.square_file(board.ep_square)
        tp = hist[6] & ~(1 << (32 + f)) | (1 << (48 + f))   # their pawn: rank 5 -> rank 7
        hist[6] = tp

    for i in range(8):
        src = cur if i == 0 else hist
        base = 13 * i
        for j, m in enumerate(src):
            if m:
                x[base + j] = _bits(m)
        # plane base+12 = repetitions = 0

    c = board.castling_rights
    us_back, them_back = (chess.BB_RANK_8, chess.BB_RANK_1) if flip else (chess.BB_RANK_1, chess.BB_RANK_8)
    if c & us_back & chess.BB_FILE_A:   x[104] = 1     # we can O-O-O
    if c & us_back & chess.BB_FILE_H:   x[105] = 1     # we can O-O
    if c & them_back & chess.BB_FILE_A: x[106] = 1
    if c & them_back & chess.BB_FILE_H: x[107] = 1
    if flip:                            x[108] = 1     # black to move
    x[109] = board.halfmove_clock                     # rule-50 ply count, raw
    # 110: zeros (former move-count plane)
    x[111] = 1                                        # board-edge helper
    return x


def encode_fens(fens, out: np.ndarray | None = None) -> np.ndarray:
    x = out if out is not None else np.zeros((len(fens), PLANES, 8, 8), np.float32)
    for i, f in enumerate(fens):
        encode(chess.Board(f), x[i])
    return x
