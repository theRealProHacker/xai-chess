# Piece-role labels: sources

Target: per-piece labels {out of play, defensive, offensive, both} for N/B/R/Q, ≥2000 positions per piece type.

Bottom line: no expert source comes near 2000 per piece type. Explicit expert labels number in the tens to low hundreds per class. The four-class scheme is used verbatim only by ICS. Everything else says "out of play", "passive", "tied to the defence", "joins the attack" and has to be mapped to the classes. To reach the target, weak-label the commentary corpora (§3), adjudicate the hits, and keep the book and ICS positions as a clean test set.

Status: **V** = text opened and quote checked this session. **U** = not verified (no free text).

## 1. Definition source

**International Chess School (ICS), Core Course, Month 1 "Think Like a Strong Player"**: chessmasterschool.com. Course syllabus PDF: https://www.chessmasterschool.com/download/core-program-strategy.pdf (V). Author: ICS (Romanian school; trainers not named in the PDF). Local copy of the free intro lesson: `chess_school_intro.md` (V).
- Labels: the four roles, verbatim ("1. A piece out of play … far away from a local battlefield and … unable to quickly arrive there. 2. … defensive role … 3. … offensive role … 4. … simultaneously plays an offensive role and one or more defensive roles").
- Size: the free intro has 1 role diagram. Months 3–4 "Master Piece Play" cover "an opponent's piece out of play" with a "large set of problems" (syllabus). Paid, count unknown (U).
- Example (V): Diagram 4, FEN `b5k1/5qp1/2p5/1pPp1p2/1P1B4/5P2/RP3KQ1/7R w`. Quote: "Ra2 and, particularly, Ba8 are out of play. White's pieces play offensive roles … the double role played by Bd4 which helps the attack on the kingside and simultaneously protects the pawn on b2". This one diagram yields all four classes.
- **Transcription bug:** the a2 rook in that FEN must be *black* (`r`). The text calls it out of play, Bd4 guards b2 against it, and with a black rook the material is level. Fix before use.

## 2. Expert text with explicit per-piece labels

| Source | Author / credentials | Labels | Usable positions | Access | Status |
|---|---|---|---|---|---|
| *Chess Fundamentals* (1921), Gutenberg #33870 | J.R. Capablanca, World Champion | "out of action", "out of the game" | ~5–10 | Public domain, plain text | V |
| *Chess Strategy* (1915), Gutenberg #5614 | Edward Lasker, IM-strength master | "out of play" (8 hits on named pieces) | ~10–15 | Public domain, plain text | V |
| *My System* (1930 English ed.), Wikimedia Commons `File:My_System.pdf` (330 pp., OCR layer) | A. Nimzowitsch | aggressive vs passive rook, "out of play", "defender of the 8th row" | ~20–40 | Public domain in the US (pre-1931). Diagrams are images, so FENs must be set up by hand | V |
| ChessBase "Understanding before Moving" series (en.chessbase.com) | IM Herman Grooten, trainer and author | piece activity, octopus knight, bad pieces; embedded replayable PGN | ~1–3 per column, 50+ columns | Free to read | V (No. 56) |
| Lichess studies, search "bad piece", "improve a bad piece", "out of play" | Mostly club players and coaches; credentials per study, mostly unknown | worst/useless piece, often marked with a `[%csl]` square highlight | Hundreds of studies; ~1–10 positions each | Free. `GET https://lichess.org/api/study/{id}.pgn` returns comments and arrows | V (export tested) |
| Silman, *How to Reassess Your Chess* 4th ed.; Aagaard, *Excelling at Positional Chess* / *GM Preparation: Positional Play* ("which is the worst piece?"); Dvoretsky & Yusupov, *Positional Play*; Shereshevsky, *Endgame Strategy* ("improve the worst piece"); Kotov, *Think Like a Grandmaster*; Karpov's annotated games | GMs, IMs and top trainers | worst piece, bad bishop, passive defender, activation | ~50–200 per book | Copyrighted, paid, no free text | U |
| Silman's chess.com articles | IM Jeremy Silman | imbalances, bad pieces | low | Free to read | V for one article only ("Death by Dark Squares"); its labels are weak ("the Knight actually blocks the c-file") |

Verified examples:
- Capablanca, Morrison–Capablanca, New York 1918 (*Chess Fundamentals*, Game 13). After 17.h4 f4, White's bishop on g3 (from 13.Bg3) gets the comment: "The Bishop is now out of action." Labels: out of play, bishop.
- E. Lasker, Capablanca–Alekhine (*Chess Strategy*, Game No. 42, diag. 160 line). After 16.Qd2 Ng8 (Nf6–g8): "The text move covers the weak square, but at the same time brings the Kt out of play … the Knight cannot return in time." The knight both defends h6 and is out of play, so this is a borderline case: the ICS classes are not mutually exclusive in practice.
- Nimzowitsch, *My System*, Part I, endgame rook section, Diagram LXXa: "The White R has the aggressive, the Black R the passive position." Labels: offensive rook vs defensive rook. FEN must be set up from the diagram image.
- Grooten, "Understanding before Moving 56", Karpov–Kasparov, WCh 1985 game 16. After 16...Nd3: "a black knight to d3 that paralysed the white pieces … Kasparov later called the knight on d3 an 'octopus'". Labels: offensive knight.
- Lichess study `WVnqRUEQ` "Improve a bad piece" (Martini99), ch. 3, FEN `2r1kb1r/2qbnn1p/1p2p1p1/pP1pPp2/P1pP3P/2P2NN1/2B2PP1/R1BQR1K1 w k - 0 19`. Comment: "what is the worst/ most useless piece for white" with `[%csl Gg3]`. Labels: knight g3, out of play / passive. The author's credentials are unknown.

## 3. Weak labelling at scale (verified on local corpora)

The local corpora are the ChessCLIP annotated-PGN release (HF `Waterhorse/chess_data`, Lichess/puzzle parts CC0). Its book, forum and blog parts were withheld for legal reasons.
- `cleaned_commentary/gameknot/`: 12,762 games. This is the same GameKnot source as Jhamtani et al. ACL 2018 (298K move–comment pairs, 11K games). That repo ships only the crawler (`harsh19/ChessCommentaryGeneration`, `Data/crawler/`) and has no licence. The annotators are amateurs, roughly 1300–2000 Elo.
- `cleaned_commentary/lichess_studies/game_studies_new.pgn`: 34,828 chapters.
- `cleaned_commentary/pgnlib/`: 35 files, 3,971 games, including expert-annotated ones (`GM_games`, `hartwig`, `d00_chess_informant`, `great_masters`, `newyork1924`). `pathtochessmastery/`: 2 files.

Rule: a comment matches when a piece name is followed, within 40 characters and in the same sentence, by
- out: `out of play|out of the game|offside|out of action|doing nothing|has no scope|has no future`
- def: `tied (down) to (the) defence|passive defender|defensive (role|duties|task)|defends? the|guarding the`
- off: `joins/joined/into the attack|attacking the king`

Positions come from `python-chess` at the commented node.

| Corpus | out N/B/R/Q | def N/B/R/Q | off N/B/R/Q |
|---|---|---|---|
| gameknot | 65/62/45/59 | 83/75/88/82 | 20/25/44/23 |
| lichess studies | 16/20/8/7 | 106/85/58/74 | 11/8/25/13 |
| pgnlib + PtCM | 12/10/6/6 | 29/25/37/29 | 3/2/3/2 |
| **total** | **93/92/59/72** | **218/185/183/185** | **34/35/72/38** |

Raw phrase counts, any context: "bad bishop" 706, "out of play" 247, "out of the game" 239, "passive <piece>" 138.

Caveats:
- Precision has not been measured. By eye, about 2/3 of 24 sampled hits are correct about the matched piece. A typical false positive: "up a full knight and also attacking the kingside".
- The matched piece name is not resolved to a square. Resolve it from "on g6", from `[%csl]` highlights, or by matching against the moved piece.
- Comments often refer to the position *after* the move, or to a hypothetical line.

Verified hits:
- GM_games.pgn, Pelletier–Karpov, after 27...Nd5: "White is now forced to accept the offer of the a6-pawn, after which his rook is out of play." Labels: out of play, rook (Karpov annotation source not stated in the file).
- great_masters.pgn, Steinitz–Sellman, after 32.Rxc7: "his Rook is tied down to the defence of the Knight and his Bishop has no safe square." Labels: defensive rook; bishop with no squares.
- lichess studies, Grischuk–Bauer, French League, after 16.f5: "the bishop on g6 … is trapped by its own pawn … currently doing nothing productive." FEN `r3k2r/pp1qbp1p/1np1p1b1/5p2/3P3N/1BP2QP1/PP1B1P1P/R4RK1 w kq - 0 16`. Labels: out of play, bishop.
- gameknot game_10358, after 15...Qxf3: "The queen is totally out of play." Labels: out of play, queen (amateur annotator).

**Algorithmic (not expert) defensive labels from the Lichess puzzle DB** (CC0; theme counts from lichess.org/training/themes, V):
- "Capture the defender": 43,535 puzzles.
- "Deflection" ("distracts an opposing piece from another duty"): 277,612.
- "Trapped piece": 74,256.

The tagger (`ornicar/lichess-puzzler`, `tagger/cook.py::capturing_defender`, AGPL) identifies the defender deterministically: `defender_square = prev.move.to_square`. Re-running it gives piece-level "defensive" labels for 40K+ positions. Caveats:
- This is a short-range tactical guard, not the ICS strategic role. Use it as an auxiliary or pretraining signal only.
- The puzzle DB has no offensive or out-of-play signal. "trappedPiece" is not the same as "out of play".

## 4. Recommended route

1. Test set: the ICS intro (with the FEN fixed), Capablanca, Lasker, Nimzowitsch and Grooten positions, plus the pgnlib expert hits. That gives ~100–200 clean, hand-checked positions.
2. Training set: §3 regex hits (~1.3K) plus more Lichess studies fetched through the API (search queries "bad piece", "worst piece", "passive", "out of play", "piece activity"). Resolve each label to a square. Adjudicate each hit, by hand or with an LLM, against the ICS definition.
3. The "both" and "offensive" classes are the bottleneck (~35–70 hits per piece type). Commentators rarely name dual roles. Expect to need a rule-based labeller built from the ICS definition, scored against step 1.

Sources: [ICS syllabus](https://www.chessmasterschool.com/download/core-program-strategy.pdf) · [Jhamtani ACL 2018](https://aclanthology.org/P18-1154/) · [harsh19/ChessCommentaryGeneration](https://github.com/harsh19/ChessCommentaryGeneration) · [Waterhorse/chess_data](https://huggingface.co/datasets/Waterhorse/chess_data) · [Gutenberg 33870](https://www.gutenberg.org/ebooks/33870) · [Gutenberg 5614](https://www.gutenberg.org/ebooks/5614) · [My System PDF](https://commons.wikimedia.org/wiki/File:My_System.pdf) · [Grooten, UbM 56](https://en.chessbase.com/post/understanding-before-moving-56) · [Lichess study WVnqRUEQ](https://lichess.org/study/WVnqRUEQ) · [Lichess puzzle themes](https://lichess.org/training/themes) · [lichess-puzzler cook.py](https://github.com/ornicar/lichess-puzzler/blob/master/tagger/cook.py) · [Silman, Death by Dark Squares](https://www.chess.com/article/view/death-by-dark-squares)
