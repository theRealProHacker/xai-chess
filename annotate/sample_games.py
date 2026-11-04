"""Random games from the monthly Lichess database into annotate/data/games.jsonl, streamed.

Reads PGN on stdin and keeps each game with probability --p until --games are kept, so the sample
spreads over the stream. Skips set-up positions, parse errors, games under --min-plies and games
whose lower rating is under --min-elo. Appends; games already in the file are skipped.

  curl -s https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst \
    | zstd -dc | python annotate/sample_games.py --games 500
"""
import argparse, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".pylibs"))
import chess.pgn

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=500)
ap.add_argument("--p", type=float, default=0.01)
ap.add_argument("--min-elo", type=int, default=0)
ap.add_argument("--min-plies", type=int, default=20)
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()

rng = random.Random(a.seed)
out = ROOT / "annotate/data/games.jsonl"
out.parent.mkdir(exist_ok=True)
have = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
kept = 0
with out.open("a") as f:
    while kept < a.games:
        if rng.random() >= a.p:
            if not chess.pgn.skip_game(sys.stdin):
                break
            continue
        game = chess.pgn.read_game(sys.stdin)
        if game is None:
            break
        h = game.headers
        gid = h.get("Site", "").rsplit("/", 1)[-1]
        if "FEN" in h or "SetUp" in h or game.errors or not gid or gid in have:
            continue
        elos = [int(h[k]) if h.get(k, "?").isdigit() else 0 for k in ("WhiteElo", "BlackElo")]
        if min(elos) < a.min_elo:
            continue
        board, sans = game.board(), []
        for mv in game.mainline_moves():
            sans.append(board.san(mv))
            board.push(mv)
        if len(sans) < a.min_plies:
            continue
        f.write(json.dumps({"id": gid, "url": h["Site"], "white": h.get("White"), "black": h.get("Black"),
                            "white_elo": elos[0], "black_elo": elos[1], "time_control": h.get("TimeControl"),
                            "result": h.get("Result"), "moves": sans}) + "\n")
        have.add(gid)
        kept += 1
print(f"kept {kept}; {len(have)} games in {out.relative_to(ROOT)}")
