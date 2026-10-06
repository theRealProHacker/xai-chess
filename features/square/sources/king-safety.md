# King-safety label sources

Goal: ≥2000 positions, each with a judgement on one named king (safe / exposed / in danger, or graded).
Checked 2026-10-06. **[V]** = text opened and example checked; **[U]** = not opened, from catalogue or memory.

## Summary

| Source | Label type | Usable positions | Access | Verified |
|---|---|---|---|---|
| ChessCLIP annotated-PGN dump (GameKnot + Lichess studies + pgnlib + ChessAdmin) | free-text expert or amateur comment, attached to a move | ~1.5k explicit "king is safe/exposed/…" phrases; ~14k broader king-related comments | free download, mixed licences | V |
| Lichess puzzle DB themes | rule-based tactic tags (`exposedKing`, `kingsideAttack`, …) | 191k / 555k / 96k / 47k | CC0 | V |
| Stockfish ≤15 classical eval, `King safety` term | numeric score per side | unlimited | GPL-3 | V |
| Aagaard, *Attacking Manual 2*, ch. 4 "King Safety" | GM prose, one chapter per weakness type | ~100 pp., est. 100–150 positions | commercial; free 15-page excerpt | V (contents page only) |
| Vukovic, *The Art of Attack in Chess* | IM prose, positions grouped by attack type | est. 150–250 | commercial; partial Lichess transcription | partial |
| Lichess user studies titled "King safety" / "Attacking the king" | chapter title (implicit), few comments | dozens of chapters, mostly amateur | Lichess API PGN export | V |
| Chess.com articles | prose | small, low quality | free online | V (weak) |
| Silman, Dvoretsky, Kotov, Nimzowitsch, ICS | prose | — | commercial | U |

Bottom line: no single expert-written source reaches 2000 positions with explicit labels. A feasible plan:
(1) mine explicit king-safety sentences from the annotated-PGN dump (~1.5k high-precision positions, plus several thousand weaker ones), labelled with an LLM or regex and spot-checked;
(2) take book positions (Aagaard, Vukovic) as a small, high-trust test set;
(3) use SF15 `King safety` and the puzzle tags only as weak labels or baselines. They are hand-coded rules, so a probe that agrees with them shows only that it recovers the rule, not that the label is valid.

---

## 1. ChessCLIP / ChessGPT annotated-PGN dump [V]

- URL: https://huggingface.co/datasets/Waterhorse/chess_data, file `chessclip_data/annotated_pgn/annotated_pgn_free.tar.gz` (57 MB, 259 MB unpacked).
- Authors: Feng et al. (ChessGPT, NeurIPS 2023 Datasets & Benchmarks). Licence: the dataset card gives CC0 for the Lichess parts; for the other sources it says only "individual terms apply". GameKnot and pgnlib text is third-party, so use it for research only and do not redistribute.
- Contents. Counts are mine; brace-comment counts are approximate.

  | Subdir | Games | Comments | Explicit phrase¹ | Broad king comment² |
  |---|---|---|---|---|
  | `gameknot/` | 12,769 | ~354k | 987 | 8,048 |
  | `lichess_studies/game_studies_new.pgn` | 71,593 chapters | ~1.65M | 784 | 4,863 |
  | `pgnlib/` (35 files: Informant, Kasparov, Wijk 2003/04, great masters, …) | 3,182 | ~44k | 101 | 652 |
  | `pathtochessmastery/` (ChessAdmin blog, Class A/Expert with Komodo check) | 365 | ~14.5k | 43 | 260 |

  ¹ Matches `king (is|was|looks|seems|remains) [adv] (safe|exposed|weak|vulnerable|in danger|naked|airy|secure|insecure)`, `king's safety`, `exposed king`, `weakened king/kingside`.
  ² A comment containing `(king|kingside)` followed within 40 characters by safe/exposed/weak/vulnerable/danger/open/attack/shelter/naked/airy/secure. Noisy.
- Label balance in GameKnot (the predicate word after "king is/was …"): safe 186, exposed 106, vulnerable 38, weak 27, in danger 19, secure 6, naked 4, airy 3. Both "safe" and "unsafe" classes are present.
- `gameknot/` is the source scraped for the Jhamtani et al. ACL 2018 dataset (https://aclanthology.org/P18-1154/; 11,578 games and 298k move–comment pairs in the paper; crawler at https://github.com/harsh19/ChessCommentaryGeneration). Annotators are amateur club players, so labels are noisy. Usually the comment does name which king.
- Caveats:
  - The FEN comes from replaying the PGN up to the commented move.
  - A comment can describe a variation rather than the game position. Keep only comments attached to mainline moves.
  - The pgnlib annotator field is sometimes an editor (e.g. "Goldsby (edit.)").
- Verified examples (I re-checked the mainline move in the PGN):
  - `pgnlib/great_masters.pgn`, Euwe–Maroczy 1936, annotator Ramkrishna Kashelkar, after **20.Bxh6!!**: "White obtains only two pawns for the piece but Black's king is exposed and white enjoys a long term initiative." → Black king: exposed.
  - `pgnlib/kasp_top.pgn`, Kasparov–Topalov, Wijk aan Zee 1999-01-20, annotator R. Byrne, after **13…O-O-O**: "the black King was safe on the other wing." → Black king: safe.
  - `pgnlib/great_masters.pgn`, Janowsky–Rubinstein 1907, after **41.Kg4?**: "The king is vulnerable here. It will be better to retreat the king to g2." → White king: vulnerable.
  - `pathtochessmastery`, Kramnik–Fedoseev 2017-07-15, after **21…h6**: "Once the knight is pushed back and Black follows with …g6, his king is safe." This is conditional on a future move, so it shows the kind of comment a filter must drop.
  - `lichess_studies`, Janowski–Marshall 1912 (study by Jonnyx), after **15.gxf3 Nd7**: "whites king is still in danger."

## 2. Lichess puzzle database, king themes [V]

- URL: https://database.lichess.org/#puzzles (6,157,341 puzzles; columns `PuzzleId,FEN,Moves,Rating,…,Themes,GameUrl,…`). Licence: CC0.
- Theme strings, from https://github.com/lichess-org/lila/blob/master/translation/source/puzzleTheme.xml:
  - `exposedKing`: "A tactic involving a king with few defenders around it, often leading to checkmate."
  - `kingsideAttack`: "An attack of the opponent's king, after they castled on the king side."
  - `queensideAttack`: the same, for queenside castling.
  - `attackingF2F7`: "An attack focusing on the f2 or f7 pawn, such as in the fried liver opening."
- Counts on https://lichess.org/training/themes: exposedKing 191,379; kingsideAttack 554,720; queensideAttack 95,912; attackingF2F7 46,807. Mate-in-N and back-rank-mate themes give a further ~2M "king in danger" positions.
- **How the tags are assigned.** They are not expert judgements. They are rules in `tagger/cook.py` (https://github.com/ornicar/lichess-puzzler):
  - `exposed_king`: the defending king is on its own back three ranks, has no own pawn on the three squares directly in front of it, and the solution contains a check.
  - `side_attack`: the king is on its back rank in the corner files, the board has at least 20 pieces (18 for queenside), the solution contains a check, and a score over captures near the corner is ≥ 2.
- Who is "in danger": always the side to move after the first move in `Moves`. The CSV FEN is the position *before* the opponent's move, so apply `Moves[0]` before labelling.
- There is no "safe" class. Negatives must come from elsewhere, for example puzzles without these themes where the king is castled behind an intact pawn shield. This choice adds selection bias.
- Example from the CSV head: `003cs`, FEN `2r1kbnr/pp4pp/4p3/3pP1N1/3q4/1P2B3/P3Q1PP/nN3RK1 b k - 1 16`, themes `advantage attraction discoveredAttack exposedKing middlegame sacrifice veryLong`, game https://lichess.org/PmRTKflq/black#32. In a 58,767-row sample, 3.0% were tagged exposedKing and 8.7% kingsideAttack.

## 3. Stockfish classical eval, `King safety` term [V]

- Source: https://github.com/official-stockfish/Stockfish/blob/sf_15/src/evaluate.cpp (`Evaluation::king()`, around line 545) and `pawns.cpp` (`ShelterStrength`, `UnblockedStorm`, `BlockedStorm`).
  - Running `eval` in a build of SF 15 or earlier prints a per-side `|King safety |` row (line 1164).
  - The hand-crafted eval was removed from Stockfish in 2023, so pin `sf_15` (my repo memory: SF 18 is patched here, so build a separate binary).
- What it measures: pawn shelter and pawn storm, plus a `kingDanger` sum:
  - attacker count × weight (`KingAttackWeights = {0,0,76,46,45,14}`)
  - 183 × weak squares in the king ring
  - 148 × unsafe checks
  - safe checks (`SafeCheck` table)
  - 98 × pinned or blocking pieces
  - 69 × attacks on the king ring
  - kingFlankAttack²
  - minus 873 if the opponent has no queen, and other corrections

  The penalty is `kingDanger²/4096` in the middlegame when `kingDanger > 100`. `PawnlessFlank` gives S(19,97).
- Usage: thresholds on `kingDanger` give a graded, cheap label for any FEN, at any scale. McGrath et al. 2022 (PNAS; arXiv 2111.09259 §"Concepts", Table 2) used SF8 `king_safety` as a probe target on 10^5 positions from the ChessBase Mega DB ("depends on the number and type of pieces that attack squares around the king, shelter strength, number of pawns around the king, penalties for being on pawnless flank").
- Caveat: these are hand-coded heuristics, not human judgement. Use them as a baseline or weak label, not as ground truth.

## 4. Aagaard, *Attacking Manual 2* (Quality Chess, 2010) [V: contents only]

- Author: GM Jacob Aagaard, British champion 2007, FIDE Senior Trainer.
- Free excerpt: https://www.scacco.it/data/attachments/1949/attacking_manual_2.pdf (15 pages; the qualitychess.co.uk link returned HTML).
- Chapter 4 "King Safety", pp. 251–352: Destroying the defensive structure; Line clearance; Cutting off the defences; **Weak kings** (p. 281); Drawing the king into the open; Kings on the run; Cutting off the escape route; **The king stuck in the centre** (p. 315); Breakthrough in the centre; Two thematic sacrifices. An index of games starts on p. 457, so positions can be rebuilt from it.
- Vol. 1 excerpt: https://www.scacco.it/data/attachments/1966/attackingmanual1.pdf (16 pages; no king-safety sentences in it).
- Label: section membership (the king is weak or in danger) plus prose. Nearly every position is "in danger", so the book needs negatives from elsewhere. Commercial; the full text was not opened.

## 5. Vukovic, *The Art of Attack in Chess* (1965; Everyman algebraic edition 1998) [partial]

- Author: IM Vladimir Vuković.
- Chapters are organised by king situation (castled king, uncastled king, king in the centre, attacking along the e-file, …) [U from memory; contents not opened].
- Partial Lichess transcription: https://lichess.org/study/JrIJGAdI (user endwithff213), 5 chapters, exportable via `https://lichess.org/api/study/JrIJGAdI.pgn`. It includes the chapter "Attacks on an uncastled king – along the e file", Chigorin–Burn, Hansen–Lundin (FEN `2kr1b1r/ppp2ppp/2n5/8/3PqBb1/2P2N2/PP1QB1PP/R3K2R b - - 0 1`). It has no prose comments.
- Estimate: 150–250 positions [U]. Commercial.

## 6. Lichess user studies [V]

- Search: https://lichess.org/study/search?q=king+safety and `?q=attacking+the+king`. Export with `https://lichess.org/api/study/{id}.pgn`.
- I sampled 12 studies (`FDFFqYbN wrYLquWU h4ZQmV5J MmHpmkqQ ZUtA0WT4 AWKeSpqR rmQdYVsu 19pxNhU0 A9bCy1il WWMhI0Se hQaY2QJ0 G9iUyVhF`): 1–16 chapters each, mostly 0–5 comments, all from untitled amateur authors.
- The best was `rmQdYVsu` (Thepawnslayer, 13 chapters: Carlsen–Chakkravarthy, Greek Gift, Em. Lasker–Bauer, Ed. Lasker–Thomas). The label is implicit in the chapter title, and comments are mostly arrows.
- Verdict: a handful of curated positions; not a label source at scale.

## 7. Chess.com articles [V, weak]

- https://www.chess.com/article/view/kings-sanctuary (Spike_Mason, 2007, untitled): "White's king is secure."
- https://www.chess.com/article/view/king-exposure (Patzer24, untitled; GM Yemelin–IM Kaplan, 12th Bavarian Open): generic prose only.
- These are user-written articles, not titled authors. I found no titled-author king-safety series with per-position judgements. Low yield.

## 8. Not opened [U]

- Silman, *How to Reassess Your Chess* (4th ed., 2010; IM). King safety is one of his listed imbalances. Commercial.
- Dvoretsky & Yusupov, *Attack and Defence* (School of Future Champions 4). Commercial.
- Kotov, *Think Like a Grandmaster* / *Play Like a Grandmaster*. Commercial.
- Nimzowitsch, *My System*. The 1925 German edition may be public domain; check archive.org.
- Edward Lasker, *Chess Strategy* (Project Gutenberg #5614, public domain). Opened and grepped: only 3 generic "castled King" passages, so negligible.
- ICS (Internet Chess School) course material: not found online.
- Pálsson & Björnsson 2023, "Unveiling concepts learned by a world-class chess-playing agent" (IJCAI). Probes Leela with Stockfish eval concepts. Relevant as prior work, not as a label source.
