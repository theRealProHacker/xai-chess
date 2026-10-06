# Queen safe/active vs. misplaced/exposed/passive/trapped: label sources

Target: ≥2000 positions labelled for a supervised queen probe.
✔ = I opened the text and checked the quoted example. ✘ = not verified.

## Bottom line

No single expert source has 2000 explicit queen verdicts. Practical mix:
1. **Negative "trapped" class at scale:** Lichess puzzle DB, `trappedPiece` theme, where the solver wins the queen. Tens of thousands, CC0.
2. **Expert prose labels, all classes:** regex-mined comments in the ChessGPT annotated-PGN dump (Apache-2.0), especially the `pgnlib_*` master-annotated subset. Comments must be hand-filtered because the regex picks up false hits.
3. **Small, clean gold set:** public-domain books on Gutenberg (ASCII diagrams, so the FEN is exact) plus hand-picked verdicts from (2).
4. **Engine weak labels** (Stockfish 11 queen eval terms). These help with augmentation and sanity checks but are not expert judgement.

## 1. Lichess puzzle database: `trappedPiece` ✔
- URL: https://database.lichess.org/#puzzles. Theme definitions: https://github.com/lichess-org/lila/blob/master/translation/source/puzzleTheme.xml
- Author: Lichess, auto-generated from games by Stockfish analysis and voted on by users. These are not expert prose.
- Labels: `trappedPiece` = "A piece is unable to escape capture as it has limited moves." No theme names the piece. Find the queen cases by replaying `Moves` and checking whether a solver move captures a queen.
- Note: `queensideAttack` is **not** about the queen. It means "An attack of the opponent's king, after they castled on the queen side." Useless here. `exposedKing` is also about the king.
- Size: 6,157,341 puzzles. In a 599,512-row sample, 6,596 were `trappedPiece` and 4,445 of those had the solver capturing a queen. My heuristic overcounts because it does not exclude queen trades. Extrapolated: roughly 45k candidate trapped queens, before filtering.
- Licence: CC0. CSV columns: FEN, Moves (UCI), Rating, Popularity, Themes, GameUrl.
- Example: puzzle `008nF`, FEN `2rq1rk1/7p/1n4pb/1R2p3/pPpP1P2/P1B5/3NQ1PP/2R3K1 w - - 0 31`, moves `e2e5 f8e8 e5e8 d8e8`, themes `crushing master middlegame short trappedPiece`. The labelled position is the one after `e2e5`, with the opponent to move. The FEN given is the position before the blunder.
- Caveat: these are tactical outcomes, not "passive" or "active". They give the trapped extreme only, and are biased toward lost-queen positions. The `Popularity` and `NbPlays` columns can be used as a quality filter.

## 2. ChessGPT annotated-PGN corpus ✔
- URL: https://huggingface.co/datasets/Waterhorse/chess_data/tree/main/chessgpt_data/annotated_pgn. Two JSONL shards, about 87 MB each. Paper: Feng et al., NeurIPS 2023 D&B, https://arxiv.org/abs/2306.09200
- Content: full PGNs with `{comments}`, keyed by source. Shard 0: 35,760 Lichess studies, about 1.7k `pgnlib_*` files (master annotations: Wijk aan Zee 2003/04 annotated, Linares, New York 1924, "middleg" lessons, Informant D00, and others), `pathtochess_*`, and GameKnot.
- Licence: the dataset is tagged Apache-2.0. The underlying annotations (Henderson, Hathaway, Informant, and so on) are third-party copyright. Use them for internal training and do not redistribute the text.
- Size, shard 0 only: regex hits for a queen comment plus a positive word (active/dominating/centralized/strong/well placed) = 458. Queen plus a negative word (trapped/misplaced/offside/out of play/passive/exposed/too early/harassed/wasted time/tempo) = 936. In the `pgnlib` subset, the tight negative set (trapped/misplaced/offside/out of play/stranded/exposed) gave 16 hits. Expect about 2× across both shards, and roughly half of the hits to be real queen verdicts. The FEN is exact because the full movetext is present.
- Verified examples (shard 0):
  - Leko–Bologan, Wijk aan Zee 2004, annotated by Henderson. Comment in the line after 15.Qh4: "Normally white would be much better because of black's king position, but here white's queen is also misplaced (imagine it would be on e2!)".
  - Bhend–Petrosian, Zurich 1961, file `pgnlib_middleg` ch. "46: Eagerness to Checkmate". After 14.g4: "Only the White Queen is rather misplaced, blocking Bc1; but this is a temporary matter and White can now bring the Queen to f2 or g2".
  - Lehmann–Smyslov 1966, `pgnlib_immortal_games`. After 17.Bg2: "White gains time attacking Black's exposed queen."
  - J. Watson–I. Bell, BPCTC 1997/98, annotated by J. Watson (IM). Final position after 51.Kxg2: "and the Black Queen is trapped. 1-0".

## 3. Lichess studies (raw) ✔
- Export: `GET https://lichess.org/api/study/{id}.pgn` returns the whole study as PGN, with chapter comments and `[%csl]`/`[%cal]` arrows. No authentication is needed for public studies. Verified on `nEGI6dm9`.
- Discovery: the `https://lichess.org/study/search?q=...` HTML can be scraped. For "early queen", "trapped queen" and "queen activity" it returns dozens of studies each (e.g. `QfkuSN20` "Punishing Early Queen", `zWfiAf7g` "Queen Activity", `rtyLgrii` "...why moving queen early isn't good").
- Quality: mostly anonymous club players. The ones I opened were thin: trap lines, or exercises like "capture all the pawns with the queen". To filter for experts, check the author's `title` via `GET /api/user/{name}`.
- Licence: user content under Lichess ToS. It is already included in source 2, so mine it there first.
- Example: study `nEGI6dm9`, chapter "Trap 4: QGA oh no my queen!" (author Bosburp, untitled). Line 1.d4 d5 2.c4 dxc4 3.e3 Nf6 4.Bxc4 c5 5.Nf3 Bg4? 6.Ne5! with the comment `{ oh no my queen! }`. The comment refers to White's own queen on d1, which the knight move leaves en prise to the bishop on g4. The point is that 6...Bxd1 runs into 7.Bxf7#. It is a trap, not a queen-placement verdict. This is typical of the noise level.

## 4. Jhamtani et al. 2018 GameKnot commentary ✔ (paper and crawler), ✘ (no queen count)
- Paper: "Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data", ACL 2018, https://aclanthology.org/P18-1154/. Code: https://github.com/harsh19/ChessCommentaryGeneration
- Content: 298K move–comment pairs from about 11.6k games on gameknot.com. The repo ships the 11,578 game links plus a crawler (`Data/crawler/`). The text itself is not redistributed, so it has to be re-crawled. The paper tags comments as MoveDesc / Quality / Comparative / Planning / Context.
- Annotators: GameKnot members, mostly amateurs, with the rating shown on the page. Weak labels only.
- Overlap: ChessGPT (source 2) already contains about 1 GameKnot game per key.
- Example ✔: https://gameknot.com/annotation.pl/my-first-time-against-the-polish?gm=6572, annotated by belacqua17 (1200). 1.b4 e5 2.Bb2 d6 3.e3 Bf5 4.Qf3 Bxc2 5.Qxb7 Nd7 6.Bb5 Ngf6 7.a3 Rb8: "the queen is almost trapped, if he takes the pawn then 8. Be4 basically wins the queen for a rook".

## 5. Public-domain books (Gutenberg) ✔: small gold set
Exact text, with ASCII diagrams, so the FEN can be reconstructed without ambiguity. Free to use.
- **Emanuel Lasker, *Chess and Checkers: The Way to Mastership* (1918)**, Gutenberg #4913, https://www.gutenberg.org/ebooks/4913. Lasker was world champion 1894–1921.
  - Diagram 57, Capablanca–Bernstein, San Sebastian 1911, after 1.Ne2: "Black … should not have risked to capture the Pawn a2, getting his Queen quite out of play."
  - Diagram 56: "it would be very risky for White to take the Pawn a6. Black would play P-c4, cutting off the retreat of White's Queen".
  - Opening section: "after (2) …, Pxd4; (3) Qxd4 the white Queen is exposed to the attack of the Knight b8, which means loss of time."
- **Edward Lasker, *Chess Strategy* (1915)**, Gutenberg #5614, https://www.gutenberg.org/ebooks/5614. Edward Lasker was an IM-strength US champion contender.
  - Diag. 17 (1.d4 d5 2.c4 Bf5 3.cxd5 Bxb1 4.Rxb1 Qxd5): "Black's Queen cannot easily be driven away from her commanding position" (positive class).
  - Line around ln. 7770: "White … having both Rooks in play and his Queen better placed … plays the Queen back into the same dominating position".
- Capablanca, *Chess Fundamentals* (Gutenberg #33870): grep found no explicit queen-placement verdicts.
- Yield: a few dozen positions in total. Useful as a held-out test set, not for volume.

## 6. Engine weak labels: Stockfish 11 classical eval ✔
- https://github.com/official-stockfish/Stockfish/blob/sf_11/src/evaluate.cpp (GPL-3).
- Queen terms:
  - `WeakQueen = S(49,15)`: "Penalty if any relative pin or discovered attack against the queen".
  - `KnightOnQueen = S(16,12)`: safe knight attack squares on the queen.
  - `SliderOnQueen = S(59,18)`: safe, doubly attacked slider squares on the queen.
  - Queen mobility via `MobilityBonus[QUEEN]`, from S(-39,-36) up to S(116,212).
- Use: compute these terms on any FEN to get an unlimited number of "harassable/exposed" and "mobile" scores. This is not expert judgement. Treat it as a baseline the probe must beat, or as a sanity check, not as ground truth.

## 7. MATE dataset (Wang et al., NAACL 2025) ✔ (paper), ✘ (data not opened)
- https://arxiv.org/abs/2411.06655, data at https://mate-chess.github.io/
- About 1M Lichess positions. The "strategy" text comes from **rules written by experts and applied automatically**: "We ask chess experts, including world champion-level players, to formulate the rules". There is no separate queen concept, only a generic "piece activity". The labels are templated, so this is effectively another rule-based labeller. Low value.

## 8. Web articles ✔/✘: few positions, mostly opening principle
- Chess.com, "Are You Making This Beginner's Queen Mistake?" by chessvictory (no title listed). https://www.chess.com/article/view/are-you-making-this-beginners-queen-mistake ✔. Spielmann–Prokes, Prague, Vienna Game. On 5.Qf3: "Principles of chess do not recommend developing the Queen so early in the opening."
- Chess.com, "The Queen in the Opening" by GM Gregory Serper. https://www.chess.com/article/view/a-queen-in-the-center ✔ (text only). "While White was developing his pieces, Black had to waste time again and again by moving his Queen which was constantly attacked by White pieces." The game diagrams are not in the fetched HTML, so I could not recover move numbers ✘.
- ChessBase, Jan Markos (GM), "Middlegame Secrets Vol.1: The Power of the Queen". https://en.chessbase.com/newsroom/post/markos-middlegame-secrets-queen-1 ✔. The page contains only promotional text ("dangerous attackers, but poor defenders"). The positions are in the paid course ✘.

## 9. Commercial books: not verified ✘
Real expert sources, but no free text, so each position has to be keyed by hand. Not opened.
- Silman, *How to Reassess Your Chess* (4th ed.) and *The Amateur's Mind*: piece-activity/imbalance chapters, no queen chapter that I could confirm.
- Nimzowitsch, *My System*: centralisation chapter. Queen centralisation examples are reportedly endgame-focused. The 1929 Hereford translation may be PD in some jurisdictions; check archive.org.
- Markos, *Middlegame Secrets Vol.1* (ChessBase video, GM). This is the most directly on-topic expert source (queen placement, timing, exchange decisions), but it is paid.
- Dvoretsky/Yusupov *Positional Play*, Shereshevsky *Endgame Strategy*: no queen-specific chapter that I could confirm.
- "ICS course material": could not identify it. Not found.

## Suggested pipeline to 2000
1. Trapped (negative): sample about 800 `trappedPiece` puzzles with a queen capture, deduplicated by opening and game.
2. Expert prose: regex over both ChessGPT shards, then hand-verify about 1–1.5k hits into misplaced / exposed / passive / active. Use `pgnlib` first for annotator quality.
3. Positive "safe and active" is the scarcest class. Supplement it with Lichess/pgnlib comments containing "dominating/centralized/strong queen", then filter out positions where Stockfish 11 `WeakQueen`/`SliderOnQueen` fire.
4. Hold out the Gutenberg positions as a clean test set.
