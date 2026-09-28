#!/usr/bin/env python3
"""Parse the ChessGPT annotated-PGN corpus and emit a standalone HTML browser.

Corpus: Feng et al. 2023 (ChessGPT, NeurIPS D&B), Waterhorse/chess_data,
chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz, extracted under data/.

Every game is parsed to produce the corpus-wide counts; a seeded sample of games
is embedded in the HTML.  The whole game tree is walked, so comments inside
analysis sidelines are kept.
"""

import argparse, base64, gzip, io, json, logging, random, re, statistics, sys, time
from itertools import zip_longest
from pathlib import Path

import chess, chess.pgn

sys.setrecursionlimit(20000)
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)  # errors are counted, not printed

DATA = Path(__file__).parent / "data"
SOURCES = ["gameknot", "lichess_studies", "pgnlib", "pathtochessmastery"]
ANNOT = re.compile(r"\[%[^\]]*\]")


def prose(comment):
    """Comment text with engine/clock annotations removed; '' if only annotations."""
    return " ".join(ANNOT.sub(" ", comment).split())


def pgn_files(source):
    d = DATA / source
    if source == "gameknot":  # one file per game, 12k of them: sort numerically
        return sorted(d.glob("*.pgn"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    return sorted(d.glob("*.pgn"))


def open_pgn(path):
    """Seven of the pgnlib files are cp1252, the rest utf-8 — decode each on its own terms."""
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            return io.StringIO(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return io.StringIO(raw.decode("utf-8", "replace"))


def iter_games(source):
    """Yield (game_id, game | None) per game; None means it failed to parse."""
    for path in pgn_files(source):
        with open_pgn(path) as fh:
            if source == "gameknot":
                gid = path.stem.replace("game_", "")
                try:
                    game = chess.pgn.read_game(fh)
                except Exception:
                    game = None
                yield gid, game
                continue
            i = 0
            while True:
                try:
                    game = chess.pgn.read_game(fh)
                except Exception:
                    yield f"{path.stem}#{i}", None
                    i += 1
                    continue
                if game is None:
                    break
                yield f"{path.stem}#{i}", game
                i += 1


def san_label(board, move):
    n = board.fullmove_number
    return f"{n}." if board.turn == chess.WHITE else f"{n}..."


def walk(game):
    """DFS over the whole game tree -> flat node list, in reading order.

    Each node: s=SAN, m=move-number label, d=sideline depth, and when it carries
    prose: c=comment, f=FEN before the move, g=FEN after, u=UCI (from/to squares).
    """
    nodes, words, n_engine_only = [], [], 0

    def emit(child, board, depth, sibling):
        nonlocal n_engine_only
        rec = {"s": board.san(child.move), "m": san_label(board, child.move), "d": depth}
        if sibling:
            rec["v"] = 1  # first move of a sibling variation
        for raw in (child.starting_comment, child.comment):
            if not raw:
                continue
            text = prose(raw)
            if not text:
                n_engine_only += 1
                continue
            words.append(len(text.split()))
            rec.setdefault("c", []).append(text)
        after = board.copy(stack=False)
        after.push(child.move)
        if "c" in rec:
            rec["c"] = " ".join(rec["c"])
            rec.update(f=board.fen(), g=after.fen(), u=child.move.uci())
        nodes.append(rec)
        return after

    def visit(node, board, depth):
        """PGN reading order: a move, then its sidelines, then the line continues."""
        while node.variations:
            main = node.variations[0]
            after = emit(main, board, depth, sibling=False)
            for alt in node.variations[1:]:
                visit_alt(alt, board, depth + 1)
            node, board = main, after

    def visit_alt(alt, board, depth):
        visit(alt, emit(alt, board, depth, sibling=True), depth)

    root = game.board()
    intro = prose(game.comment or "")
    if intro:
        words.append(len(intro.split()))
        nodes.append({"s": "", "m": "", "d": 0, "c": intro, "f": root.fen(), "g": root.fen()})
    elif game.comment:
        n_engine_only += 1
    visit(game, root, 0)
    return nodes, words, n_engine_only


def game_url(source, game_id, headers):
    site = headers.get("Site", "")
    if source == "lichess_studies" and site.startswith("http"):
        return site
    if source == "gameknot":
        return f"https://gameknot.com/annotation.pl/x?gm={game_id}"
    return ""


def pack(source, game_id, game, nodes):
    h = game.headers
    rec = {"src": source, "id": game_id, "n": nodes}
    for key, tag in (("w", "White"), ("b", "Black"), ("ev", "Event"), ("dt", "Date"), ("r", "Result")):
        v = h.get(tag, "").strip()
        if key == "dt":
            v = re.sub(r"[.\-]\?\?", "", v)  # "1931.??.??" is just 1931
        if v and v.strip("?.-") and v != "-":
            rec[key] = v
    for key, tag in (("we", "WhiteElo"), ("be", "BlackElo")):
        v = h.get(tag, "").strip()
        if v.isdigit():
            rec[key] = int(v)
    url = game_url(source, game_id, h)
    if url:
        rec["url"] = url
    start = game.board().fen()
    if start != chess.STARTING_FEN:
        rec["root"] = start
    return rec


def build(per_source, seed):
    stats, sample = {}, []
    for source in SOURCES:
        t0 = time.time()
        rng = random.Random(seed)
        keep, seen_ok = [], 0
        games = skipped = comments = engine_only = 0
        words = []
        for game_id, game in iter_games(source):
            games += 1
            if game is None or game.errors:
                skipped += 1
                if game is None:
                    continue
            nodes, wc, n_e = walk(game)
            comments += len(wc)
            engine_only += n_e
            words += wc
            if not wc:
                continue
            # reservoir sample over games that actually carry commentary
            seen_ok += 1
            packed = pack(source, game_id, game, nodes)
            if len(keep) < per_source:
                keep.append(packed)
            else:
                j = rng.randrange(seen_ok)
                if j < per_source:
                    keep[j] = packed
        stats[source] = {
            "games": games, "games_skipped": skipped, "games_with_commentary": seen_ok,
            "comments": comments, "engine_only_dropped": engine_only,
            "words_mean": round(statistics.mean(words), 1) if words else 0,
            "words_median": statistics.median(words) if words else 0,
            "sampled_games": len(keep),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{source:20s} {stats[source]}", flush=True)
        sample.append(keep)
    # round-robin the sources so the list opens on a mix, not 500 rows of one site
    return stats, [g for row in zip_longest(*sample) for g in row if g]


def emit(out, stats, sample, per_source, seed, compress):
    payload = json.dumps({"stats": stats, "games": sample, "per_source": per_source, "seed": seed},
                         ensure_ascii=False, separators=(",", ":"))
    template = (Path(__file__).parent / "viewer.template.html").read_text(encoding="utf-8")
    if compress:
        blob = base64.b64encode(gzip.compress(payload.encode("utf-8"), 9)).decode("ascii")
        html = template.replace('"__DATA__"', json.dumps(blob))
    else:
        # plain JSON: bigger, but greppable and independent of DecompressionStream
        html = template.replace('"__DATA__"', json.dumps(payload))
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{'gzip+base64' if compress else 'plain'}; payload {len(payload) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-source", type=int, default=500, help="games sampled per source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "commentary.html")
    ap.add_argument("--no-compress", action="store_true",
                    help="embed plain JSON instead of gzip+base64 (larger file)")
    ap.add_argument("--stats", type=Path, default=DATA / "stats.json",
                    help="cache the corpus-wide counts here")
    a = ap.parse_args()
    missing = [s for s in SOURCES if not (DATA / s).is_dir()]
    if missing:
        sys.exit(f"missing {missing} under {DATA} — see README for the download step")
    stats, sample = build(a.per_source, a.seed)
    a.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    emit(a.out, stats, sample, a.per_source, a.seed, not a.no_compress)


if __name__ == "__main__":
    main()
