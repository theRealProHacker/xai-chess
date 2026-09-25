# YouTube Shorts → annotated PGN: findings

Goal: turn chess Shorts into (position, move, comment) data in the same shape as `commentary/data/`. Background survey of tools, legal status and prior work: `paper/yt-shorts-extraction.md`.

## 1. Which channels explain rather than react

Searched YouTube (yt-dlp `ytsearch`) with six queries for explanatory chess Shorts, listed the `/shorts` tab of the 12 most frequent channels, and read the auto-captions of 3–4 random Shorts from six of them. Small sample; first impressions only.

| Channel | Shorts | Captions |
|---|---|---|
| **You Me And Chess** (`UCLoKcyINfdnxMlBR23a-rQA`) | 127 | Pure explanation; every move named in notation. Traps, puzzles, endgames. |
| Chessscape (`UCrFUYBN778jDUQ6rWgrl2ig`) | 149 | Explanation, notation-heavy, with reasons. Some dramatic framing and off-topic titles. |
| Remote Chess Academy (`UCsKZ2yOsgfNxln8xH5WkGvg`) | 400+ | Clean explanation with lots of *why*, but moves are pointed at ("deliver this check") rather than named. |
| Chess Vibes (`UChDxbOUQRXEZ1zdI14Zyx9w`) | 72 | Explanatory; many cut from longer videos, start mid-sentence. |
| thechesswebsite (`UCHz5JQAUSkjxrosDIWCtEdw`) | 90 | Mostly explanation cut from long videos; one sample was an announcement. |
| ChessCoach Andras (`UCcYZTGsTO5TbCaA1O0wcBzw`) | 395 | Reaction and banter. Unusable. |

You Me And Chess is the best starting point: spoken moves can be checked against moves read from the board.

## 2. Worked example: "Fishing Pole Trap"

Short [`1h0LdL3M2EE`](https://www.youtube.com/shorts/1h0LdL3M2EE), 37 s. Output: `commentary/data/yt_shorts/1h0LdL3M2EE.pgn` (git-ignored, like the rest of `commentary/data/`). python-chess parses it with no errors; the final position is checkmate.

### Pipeline

1. **Download.** `yt-dlp -f "bv*[height<=720]+ba" --write-auto-subs --sub-langs en-orig --sub-format json3`. Without a JS runtime (deno) yt-dlp only got 360p; that was enough.
2. **Frames.** `ffmpeg -vf fps=4` → 148 frames.
3. **Board reading.** `tsoj/Chess_diagram_to_FEN`, the tool recommended in the survey, failed on this channel's purple theme, both on the full frame and on a cropped, upscaled board. It produced placements like `8/8/8/8/6N1/8/6NQ/8`. Replaced it with template matching:
   - fixed grid: 45 px squares at y=140; the board is shown flipped (h-file left, rank 1 on top);
   - templates for each piece are taken from the starting-position frame;
   - per square: masks of white and black "ink" (low-saturation pixels that are very bright or very dark), compared by IoU; under 2 % ink counts as empty.

   A first version compared raw pixels and misread pieces on the olive last-move highlight. The ink-mask IoU fixed that. Arrows and mid-move frames still cause single-frame errors.
4. **Frames → moves.** From the start position, take the legal move whose resulting placement differs least from the frame. Accept it only if at most 1 square is wrong, it explains the frame better than the current board, and the next frame agrees (allowing one reply move in between). This throws out frames caught mid-move.
   Result: all 18 plies, ending in checkmate, matching the narration. 9. a3 is never spoken ("a random move") and is known only from the video.
5. **Alignment.** Each move appears on the board within about 0.3 s of the narrator saying it, so json3 word timestamps tie speech to moves directly.
6. **Comment cleanup: by hand, for this one example.**
   - Cutting the transcript at move times splits clauses ("attacking | the pawn"), so I set sentence breaks myself.
   - Pointing words ("over here", "that square") were replaced with the square the on-screen arrows show (h2, h2, f2), checked against the frames.
   - Speech-to-text fixes: "why it gets happy" → "White gets happy", "picking away" → "takes away". Added capitals and punctuation.
   - Removed words that only repeat the move ("plays Knight to F6") and filler.

### Result

```
{ The Fishing Pole Trap starts with: } 1. e4 e5 2. Nf3 Nc6 3. Bb5
{ White attacks Black's knight. } 3... Nf6 { Black ignores it. } 4. O-O Ng4
{ Attacking the pawn on h2. } 5. h3 { White gets extremely uncomfortable. }
5... h5 { Instead of moving the knight. } 6. hxg4
{ White gets happy and captures the knight. } 6... hxg4
{ Black captures back, and now the knight on f3 is under attack. } 7. Ne1
{ So it moves to e1. } 7... Qh4
{ The crushing move, threatening mate on h2. } 8. f3
{ To create room for the king on f2. But it's too late: } 8... g3
{ Black takes away that square. } 9. a3
{ Now White has to make a random move. } 9... Qh2# { Checkmate. } 0-1
```

## 3. What this shows

- **Moves from video are reliable on a digital board.** Legal-move matching absorbs the reader's single-square errors; no ML model was needed.
- **Generic board recognizers don't carry over to every theme.** Per-channel templates are cheap because a channel keeps the same layout; learn them once from a start-position frame.
- **The video fills gaps the speech leaves.** Unspoken moves (a3) come from the board; "over here" is resolved from arrows.
- **Comment cleanup is the part that doesn't scale yet.** Sentence splitting, pointing-word resolution and transcription fixes were done by hand. At scale this needs an LLM pass given the transcript, the move list and the arrow squares, plus a check that every square it names matches the board.

## 4. Open issues

- The grid geometry (y=140, 45 px, flipped) is hard-coded for this video. Untested on other Shorts from the channel; puzzle Shorts don't begin from the start position, so templates must come from elsewhere.
- Arrows are read by eye, not detected. Detecting them (colour-segment red/green, fit a line between square centres) is needed to resolve pointing words automatically.
- Positions not starting from move 1 need side to move and castling rights, which the video doesn't show.
- Legal: see `paper/yt-shorts-extraction.md` §risks. Only derived data (IDs, timestamps, PGN) is publishable; no media is stored in the repo.
