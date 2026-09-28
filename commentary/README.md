# commentary

A local browser for the human-written game commentary in the **ChessGPT** corpus
(Feng et al., 2023, NeurIPS Datasets & Benchmarks) — `Waterhorse/chess_data`,
`chessclip_data/annotated_pgn/`. See [Licence and citation](#licence-and-citation).

```
build_viewer.py        parse the PGNs, sample games, emit the viewer
viewer.template.html   the viewer (HTML/CSS/JS, no dependencies)
commentary.html        the built viewer — open it over file://
data/                  the extracted corpus + stats.json (git-ignored, ~200 MB)
```

## Get the data

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Waterhorse/chess_data", repo_type="dataset",
                  allow_patterns="chessclip_data/annotated_pgn/*", local_dir="data")
PY
tar -xzf data/chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz -C data/
```

57 MB compressed. Do **not** use `load_dataset()` — the Hub's parquet conversion for
this repo fails with a schema cast error. The other copy of the same text at
`chessgpt_data/annotated_pgn/` is two JSONL shards with an empty `metadata` field,
so it loses provenance; the tarball keeps the per-source directories.

## Build

```bash
.venv/bin/python build_viewer.py                    # 500 games per source, seed 0
.venv/bin/python build_viewer.py --per-source 1000 --seed 7 --out big.html
.venv/bin/python build_viewer.py --no-compress      # plain JSON instead of gzip+base64
```

Every game in the corpus is parsed on each run (~5 minutes) to produce the
counts below; only the sampled games are embedded. Sampling is a seeded reservoir over
the games that carry at least one prose comment, so the same seed gives the same sample.
The payload is embedded as gzip+base64 and inflated in the page with the browser's own
`DecompressionStream` — still one file, still no CDN. `--no-compress` embeds plain JSON
(bigger, greppable) for browsers without it.

## What the parse produced

| Source | Games | Parse errors | Comments | Engine-only dropped | Words/comment (mean / median) | Sampled |
|---|---|---|---|---|---|---|
| `gameknot/` | 12,769 | 1 | 353,954 | 469 | 19.6 / 12 | 500 |
| `lichess_studies/` | 71,593 | 0 | 324,182 | 1,328,033 | 13.0 / 6 | 500 |
| `pgnlib/` | 3,187 | 20 | 42,941 | 105 | 16.1 / 10 | 500 |
| `pathtochessmastery/` | 365 | 0 | 14,520 | 17 | 15.8 / 13 | 365 |
| **total** | **87,914** | **21** | **735,597** | **1,328,624** | | **1,865** |

`pathtochessmastery` has only 365 games, so `--per-source 500` takes all of them.
The default build is 1,865 games / 39,355 comments in a 5.2 MB `commentary.html`.

- **Parse errors** are games where python-chess rejected at least one move (illegal SAN,
  ambiguous SAN, a stray `1-0` in the movetext) — bad transcriptions upstream, concentrated
  in `pgnlib`. Such a game is still shown up to the point it parsed; none are silently dropped.
  Only 1 of the 21 (in `gameknot`) actually raises and yields no game; the other 20 parse up to
  the bad move and keep their commentary.
- **Engine-only** counts brace blocks that held nothing but `[%eval]`, `[%clk]`, `[%emt]`,
  `[%csl]`, `[%cal]`. `lichess_studies` is mostly these. Any such marker is also stripped out
  of a comment that does carry prose.
- **The whole game tree is walked**, so comments inside analysis sidelines are kept, in PGN
  reading order (a move, then its sidelines, then the line continues). A node's
  `starting_comment` and `comment` count as two comments and display as one block.
- **Not all of it is human writing.** 82,888 `lichess_studies` comments — a quarter of that
  slice — are nothing but lichess's own computer annotation ("Inaccuracy. Bf4 was best.",
  "Checkmate is now unavoidable. Qxh7+ was best."), and another 14,015 are its result line
  ("1-0 White wins by checkmate."). 1,144 open with one and then continue in the annotator's own
  words. `../cleaned_commentary/prefilter.py` drops the pure ones, keeps the mixed ones whole, and
  reports every rule's catch.
- Seven `pgnlib` files are cp1252, the rest of the corpus is UTF-8; each file is decoded on its
  own terms, so accented names survive ("Lilienthal, André").
- Comment text is verbatim. The upstream corpus lost some spaces at line joins
  ("the previousknight move"); that is the data, not the parser.

## The viewer

One game at a time: its full commentary listed in move order with sidelines indented,
the board beside it. Clicking a comment jumps the board to that position and highlights
the move's from/to squares (the origin as a ring, the destination filled); the Position
control under the board switches between the position before and after the move. Filter by source, search the comment text (games are filtered to
matches and the matches are highlighted), or jump to a random game.

Boards are Unicode glyphs in a CSS grid — no chessboard library, no sprites. Only commented
moves are clickable: FENs are stored for those alone, which is what keeps the file at 5 MB.
The sampled games are interleaved across sources, so the list opens on a mix rather than 500
rows of one site.

Each `lichess_studies` game links back to its `https://lichess.org/study/<study>/<chapter>`
page, which is live. GameKnot links are rendered too, but gameknot.com answers 403 to
automated requests and may not resolve.

## Licence and citation

The Hub repo is tagged `apache-2.0`, but that tag covers the release as a whole. For
`annotated_pgn` specifically the dataset card claims no licence of its own and points at the
four upstream sites' terms instead:
[PGNlib](https://www.angelfire.com/games3/smartbridge/),
[lichess](https://lichess.org/terms-of-service),
[GameKnot](https://gameknot.com/pg/pol_eula.htm),
[Path to Chess Mastery](https://www.pathtochessmastery.com/). The commentary is
user-written text scraped from those sites; treat redistribution as governed by them,
not by Apache-2.0. [LICENSING.md](LICENSING.md) compiles what each of the four actually says —
the short version is that none of them licenses redistribution, so keep the built
`commentary.html` local and rebuild it per machine.

```bibtex
@article{feng2023chessgpt,
  title={ChessGPT: Bridging Policy Learning and Language Modeling},
  author={Feng, Xidong and Luo, Yicheng and Wang, Ziyan and Tang, Hongrui and Yang, Mengyue
          and Shao, Kun and Mguni, David and Du, Yali and Wang, Jun},
  journal={arXiv preprint arXiv:2306.09200},
  year={2023}
}
```

That is the entry the dataset card gives; the paper was published at NeurIPS 2023 Datasets
and Benchmarks, so cite the proceedings version if the venue matters.
