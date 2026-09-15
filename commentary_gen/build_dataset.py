"""Build the commentary-generation dataset from the cleaned corpus.

One example per mainline move that carries a human comment:
  moves   SAN movetext up to and including the commented move
  board   ASCII board after that move (white at the bottom)
  fen     FEN after that move
  comment the annotator's prose (tool/annotator markup stripped)

Writes data/pool.jsonl (every candidate with its quality score) and
data/train.jsonl / data/dev.jsonl (the best N, stratified by source).
"""
import argparse, json, random, re, sys
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE.parent / "cleaned_commentary"
sys.path.insert(0, str(CORPUS))
import filter_games as fg  # noqa: E402  open_pgn, read_games, prose, pgn_files, SOURCES

import chess  # noqa: E402

QUOTA = {"gameknot": 700, "lichess_studies": 700, "pgnlib": 400, "pathtochessmastery": 200}

STOP = set("the a an and to of in is it that for with on this as be by at or not but from "
           "have has was are will can would if his her their he she they we you".split())
CHESS = re.compile(r"\b(?:pawn|knight|bishop|rook|queen|king|castl|file|rank|diagonal|centre|"
                   r"center|square|attack|defen|threat|pin|fork|sacrific|exchange|trade|tempo|"
                   r"develop|control|pressure|weak|strong|open|initiative|counterplay|endgame|"
                   r"middlegame|opening|checkmate|mate|check|material|space|outpost|passed|"
                   r"isolated|doubled|majority|structure|plan|idea)\w*", re.I)
WHY = re.compile(r"\b(?:because|since|so that|in order|threat|idea|plan|otherwise|prevent|"
                 r"avoid|prepar|aim|intend|point|forces?|wins?|loses?|control|pressure|weak)\w*",
                 re.I)
BAD = re.compile(r"https?://|\bwww\.|\[%|\bTODO\b|thanks for|please comment|rate this|"
                 r"\bLOL\b|:-?\)|\bgg\b", re.I)


def score(text):
    """Higher = more likely a substantive, well-formed English comment about the move."""
    n = len(text)
    if n < 60 or n > 700 or BAD.search(text):
        return 0.0
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 10:
        return 0.0
    alpha = sum(c.isalpha() or c.isspace() for c in text) / n
    if alpha < 0.75:
        return 0.0
    stop = sum(w.lower() in STOP for w in words) / len(words)
    if stop < 0.15:  # not English prose
        return 0.0
    chess_hits = len(CHESS.findall(text))
    why_hits = len(WHY.findall(text))
    # sentence hygiene: starts with a capital or a move, ends with punctuation
    tidy = bool(re.match(r"^[A-Z0-9(\[]", text)) + text.rstrip().endswith((".", "!", "?"))
    caps_ratio = sum(w.isupper() and len(w) > 2 for w in words) / len(words)
    length_bonus = min(n, 350) / 350
    return (2 * min(chess_hits, 6) + 3 * min(why_hits, 4) + 2 * tidy + 4 * length_bonus
            + 3 * stop - 10 * caps_ratio)


def ascii_board(board):
    return str(board)


def examples(source):
    for path in fg.pgn_files(source):
        with fg.open_pgn(path) as fh:
            for gi, game in enumerate(fg.read_games(fh)):
                if game is None or "FEN" in game.headers:  # set-up positions: the movetext is not the game
                    continue
                board = game.board()
                san = []
                node = game
                ply = 0
                while node.variations:
                    node = node.variations[0]
                    ply += 1
                    try:
                        s = board.san(node.move)
                        board.push(node.move)
                    except Exception:
                        break
                    san.append(f"{(ply + 1) // 2}. {s}" if ply % 2 else s)
                    text = fg.prose(node.comment or "")
                    if not text:
                        continue
                    yield {
                        "source": source, "file": path.name, "game": gi, "ply": ply,
                        "white": game.headers.get("White", ""), "black": game.headers.get("Black", ""),
                        "moves": " ".join(san), "fen": board.fen(), "board": ascii_board(board),
                        "move": s, "side": "White" if ply % 2 else "Black",
                        "comment": text, "score": round(score(text), 2),
                    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=300, help="held-out size carved from the best N")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = HERE / "data"
    out.mkdir(exist_ok=True)

    pool, per = [], {}
    with open(out / "pool.jsonl", "w", encoding="utf-8") as fh:
        for source in fg.SOURCES:
            k = 0
            for ex in examples(source):
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                k += 1
                if ex["score"] > 0:
                    pool.append(ex)
            per[source] = k
            print(f"{source:<20} {k:>8,} mainline comments", flush=True)

    best = []
    for source, quota in QUOTA.items():
        cands = sorted((e for e in pool if e["source"] == source), key=lambda e: -e["score"])
        # one comment per game so 2000 examples are 2000 games' worth of voices, not 50 games'
        seen, take = set(), []
        for e in cands:
            key = (e["file"], e["game"])
            if key in seen:
                continue
            seen.add(key)
            take.append(e)
            if len(take) == quota:
                break
        best += take
        print(f"{source:<20} scored>0 {len(cands):>7,}  took {len(take):>4}  "
              f"score range {take[-1]['score']:.1f}–{take[0]['score']:.1f}")

    random.Random(args.seed).shuffle(best)
    dev, train = best[:args.dev], best[args.dev:]
    for name, rows in (("train", train), ("dev", dev)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for i, e in enumerate(rows):
                e = dict(e, id=f"{name}-{i:04d}")
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}")
    json.dump({"mainline_comments": per, "quota": QUOTA, "train": len(train), "dev": len(dev)},
              open(out / "dataset_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
