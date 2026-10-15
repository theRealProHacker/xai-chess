"""One random position per Lichess game, from the monthly standard rated database, streamed.

Reads PGN on stdin (the decompressed database file, so only its first part need be fetched) and
keeps one position per game, uniformly among the plies 18..120 (moves 10 to 60), or 18..last ply
if the game ends earlier. Skips games shorter than 18 plies, games from a set-up position, and
positions already taken. Writes sf-probe/data/fens_games.txt (as fens.txt) and fens_games_meta.tsv.

  curl -s https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst \
    | zstd -dc | python lc0-probe/lichess_games_fens.py --month 2024-01
"""
import argparse, collections, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".pylibs"))
import chess, chess.pgn

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=100000)
ap.add_argument("--month", default="2024-01", help="database month piped in, only recorded in the log")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lo", type=int, default=18)
ap.add_argument("--hi", type=int, default=120)
a = ap.parse_args()

rng = random.Random(a.seed)
out = ROOT / "sf-probe/data/fens_games.txt"
seen, skipped, read = set(), collections.Counter(), 0
with open(out, "w") as f, open(out.with_name("fens_games_meta.tsv"), "w") as m:
    m.write("fen\turl\tply\twhite_elo\tblack_elo\ttime_control\n")
    while len(seen) < a.games:
        game = chess.pgn.read_game(sys.stdin)
        if game is None:
            break
        read += 1
        h = game.headers
        if "FEN" in h or "SetUp" in h:
            skipped["set-up position"] += 1; continue
        if game.errors:
            skipped["parse error"] += 1; continue
        moves = list(game.mainline_moves())
        if len(moves) < a.lo:
            skipped[f"under {a.lo} plies"] += 1; continue
        ply = rng.randint(a.lo, min(a.hi, len(moves)))
        board = chess.Board()
        for mv in moves[:ply]:
            board.push(mv)
        fen = board.fen()
        if fen in seen:
            skipped["duplicate"] += 1; continue
        seen.add(fen)
        f.write(fen + "\n")
        m.write("\t".join([fen, h.get("Site", ""), str(ply), h.get("WhiteElo", ""), h.get("BlackElo", ""),
                           h.get("TimeControl", "")]) + "\n")
        if read % 10000 == 0:
            print(f"read {read}, kept {len(seen)}", flush=True)
print(f"month {a.month}: read {read} games, kept {len(seen)}; skipped {dict(skipped)}", flush=True)
