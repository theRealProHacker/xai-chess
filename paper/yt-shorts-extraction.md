# YouTube Shorts → (position, commentary) pairs

*Research note, 2026-09-25. Question: can chess YouTube Shorts be turned into a paired
(position/move sequence, spoken commentary) corpus alongside `commentary/` and `cleaned_commentary/`?*

## 0. Target format

The existing corpora (`commentary/`, from ChessGPT's `annotated_pgn`: GameKnot, lichess studies,
PGNlib, Path to Chess Mastery) are PGN with `{comment}` after a move, so every comment binds to one
FEN + one move + a source. A Shorts-derived record should reduce to the same thing:

```
[Event "yt:<video_id>"] [Site "https://www.youtube.com/shorts/<id>"] [FEN "<first recognised position>"] [SetUp "1"]
1... e1=Q { speech t=4.8–6.2s, words aligned to this ply } 2. g4 { ... }
```

plus sidecar fields: channel, upload date, licence field, per-ply recogniser confidence, and the
time window each comment was cut from. Unlike the PGN corpora, the start position is usually
mid-game, so `[SetUp]/[FEN]` is the norm, not the exception.

## 1. Downloading (yt-dlp)

Checked against yt-dlp 2026.08.19 (installed locally) and its
[README](https://github.com/yt-dlp/yt-dlp/blob/master/README.md).

- **Listing a channel's Shorts.** A channel's `/shorts` tab is a `youtubetab` playlist.
  `yt-dlp --flat-playlist --print "%(id)s %(title)s" https://www.youtube.com/@GothamChess/shorts`
  returned IDs and `https://www.youtube.com/shorts/<id>` URLs in seconds (tested). `duration` came
  back `NA` in flat mode, so filter on duration after a full `-J` extraction, or fetch per-ID.
  The README: "Channel URLs download all uploads of the channel, including shorts and live".
- **Metadata / captions without media.** `-J --skip-download` on one Short (tested) gave 1080×1920
  video-only formats (avc1/vp9/av01), audio-only m4a/opus (incl. `-drc` variants), no manual
  subtitles, and `automatic_captions` in 157 languages including `en-orig`.
- **Format selection.** README: `ba`/`bestaudio` = "best quality **audio-only** format"; default
  selector is `bv*+ba/b`. For this pipeline: `-f ba` for ASR, `-f "bv*[height>=1280]"` video-only
  for frames (audio is not needed in the video file). A `bv*[height<=720][ext=mp4]` request
  silently returned 360×640 in my test; check `ffprobe` resolution after download.
- **Captions.** `--write-auto-subs --sub-langs en-orig --sub-format json3 --skip-download`.
  Tested: the `json3` file carries **per-word offsets** (`tStartMs` per event + `tOffsetMs` per
  word), i.e. free word timestamps. Casing is unreliable (half the words came back upper-case).
- **Bot-detection / JS.** README: a JS runtime (deno recommended) plus `yt-dlp-ejs` is "highly
  recommended"; without one yt-dlp warns that "some formats may be missing". `player_client` and
  `po_token` extractor-args exist because some clients require Proof-of-Origin tokens. My
  unauthenticated test worked without deno; bulk runs likely won't (unverified).

### ToS and licensing

YouTube [Terms of Service](https://www.youtube.com/t/terms) (dated 15 Dec 2023), "Permissions and
Restrictions" — you are not allowed to:

> "access, reproduce, download, distribute, transmit, broadcast, display, sell, license, alter,
> modify or otherwise use any part of the Service or any Content except: (a) as expressly
> authorized by the Service; or (b) with prior written permission from YouTube and, if applicable,
> the respective rights holders"

> "access the Service using any automated means (such as robots, botnets or scrapers) except (a) in
> the case of public search engines, in accordance with YouTube's robots.txt file; or (b) with
> YouTube's prior written permission"

> "collect or harvest any information that might identify a person (for example, usernames or
> faces), unless permitted by that person"

So yt-dlp bulk download is a ToS breach regardless of purpose; Shorts facecams make the
face clause relevant too. What can override the ToS is statute, not contract, and only for some uses:

- **German/EU TDM exception.** [§60d UrhG](https://www.gesetze-im-internet.de/urhg/__60d.html)
  permits reproductions for text and data mining for scientific research by research
  organisations and by "individual researchers pursuing non-commercial research"; copies may be
  shared only with a defined research group / for verification, and kept only as long as the
  research needs. It permits *making* the corpus, not *publishing* the media or transcripts.
  Whether §60d beats the ToS's contractual download ban, and whether yt-dlp's signature handling
  counts as circumventing a technical measure (§95a UrhG), is not settled here. Not legal advice.
- **Creative Commons uploads.** YouTube offers only CC BY 4.0
  ([help page](https://support.google.com/youtube/answer/2797468)): reuse with attribution. yt-dlp
  exposes it as the `license` field (my test Short: `None`). Filtering to CC BY Shorts gives the
  only cleanly redistributable subset; expect it to be small.
- **Precedent.** ChessGPT built an 83k-video YouTube transcript set and did not release it:
  "Because of the legal issue, for ChessGPT dataset, we do not open-source the chess-book,
  chess-forum, chess-blog, and Youtube transcript datasets"
  ([dataset card](https://huggingface.co/datasets/Waterhorse/chess_data)).

**Consequence:** release only derived, non-media artefacts: video ID + timestamps + FEN/PGN +
(optionally) short commentary spans, or release only the pipeline. Keep media local and transient.
This matches how `commentary/LICENSING.md` already treats the PGN corpora (no redistribution rights).

## 2. Speech-to-text with word timestamps

| Option | Word timings | Vocabulary biasing | Notes |
|---|---|---|---|
| YouTube auto-captions (`json3`) | yes, per word (tested) | none | Free, no GPU; quality on notation untested; casing noisy. Absent on some videos. |
| [openai/whisper](https://github.com/openai/whisper) | `word_timestamps=True` (cross-attention + DTW, [`timing.py`](https://github.com/openai/whisper/blob/main/whisper/timing.py)) | `initial_prompt`; `carry_initial_prompt` re-applies it to every window ([`transcribe.py`](https://github.com/openai/whisper/blob/main/whisper/transcribe.py)) | Prompt is truncated to the last `n_ctx//2 − 1` = 223 tokens ([`decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py)). |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `word_timestamps=True` | `initial_prompt`, `hotwords` ("Hotwords/hint phrases… Has no effect if prefix is not None", [source](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)) | Silero VAD via `vad_filter=True`; default only drops >2 s silence. CTranslate2, fastest on CPU. |
| [WhisperX](https://github.com/m-bain/whisperX) | wav2vec2 forced alignment ("Accurate word-level timestamps using wav2vec2 alignment") | inherits Whisper's | README limitation: words "which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing". "e5" keeps its letter; a bare "5" loses timing. Also "Overlapping speech is not handled particularly well". |

**Chess notation.** Speakers say "knight takes e5", "queen b7 check", "h5!"; Whisper will
emit a mix of "Knight takes E5", "Nxe5", "night takes e five". Nothing in the primary sources
measures Whisper on chess notation — **untested**. Plan:
1. Bias with `initial_prompt`/`hotwords` built from chess terms plus SAN-in-words
   ("knight takes e5, bishop to g7, queen b7 check, castles, en passant, promotes, Nf3, Bxh7+").
2. Don't trust ASR to be exact. Normalise to spoken form (letters+digits → square tokens) and
   **match against the legal moves** of the recognised position (§5). A 20–40-way legal-move
   choice absorbs most ASR errors, so transcripts double as a move-disambiguation signal.
3. Music/SFX. Shorts often have music under the voice and meme sound effects. Options: VAD
   (faster-whisper/WhisperX), source separation with
   [Demucs](https://github.com/adefossez/demucs) (the `facebookresearch/demucs` repo is archived;
   the maintained fork is `adefossez/demucs`), and Whisper's `hallucination_silence_threshold`.
   Effect on notation accuracy is untested.

Recommendation: take YouTube `json3` captions first (zero cost); run faster-whisper large-v3
with a chess prompt + `word_timestamps=True` where captions are missing or disagree with the legal
move set. Use WhisperX only if timing precision below ~0.3 s turns out to matter. Given Shorts'
fast cuts it probably doesn't, since alignment is to board-change events (§4), not to frames.

## 3. Board recognition from frames

### Digital vs physical

- **Digital (lichess/chess.com screen capture, Chessbase, streamer overlays):** fixed geometry
  within a video, flat rendering, finite theme set. Once the board's bbox and theme are known, per-square
  template matching against 12 piece sprites + empty works (unverified at scale, standard
  technique). A learned model is only needed for localisation, theme variety, and overlays.
- **Physical (OTB, broadcast cameras):** perspective, occlusion by hands, lighting. Much harder:
  on the ChessReD benchmark (10,800 smartphone photos) Masouris & van Gemert reach 15.26% of test
  boards fully correct ([arXiv:2310.04086](https://arxiv.org/abs/2310.04086)); ChessQueries
  (Seytre, [arXiv:2608.30762](https://arxiv.org/abs/2608.30762), Aug 2026, ViT + DETR-style
  decoder) reports 99.2% with 0.01 wrong squares/board and promises code, weights and a harder
  "SLCC" set. Not released at time of writing (unverified).

### Tools

| Tool | Input | Localisation | Orientation | Licence / status |
|---|---|---|---|---|
| [tsoj/Chess_diagram_to_FEN](https://github.com/tsoj/Chess_diagram_to_FEN) | diagrams, screenshots | yes (detect, bbox, warp) | image rotation + "whites or blacks perspective" net | MIT, active (pushed 2026-03). **Tested, see below.** |
| [Elucidation/tensorflow_chessbot](https://github.com/Elucidation/tensorflow_chessbot) | "online chessboard screenshots" | gradient + Hough line grid | none | MIT, 2023; TF1-era. ChessGPT reused its line-detection notebook. |
| [mcdominik/board_to_fen](https://github.com/mcdominik/board_to_fen) | "digital chessboard image" (pre-cropped) | no | manual `black_view` flag | MIT, 2023, TF/Keras |
| [notnil/fenify](https://github.com/notnil/fenify) | book diagrams | "No localization" | "No board orientation detection"; also "No move detection (common on online boards with highlighted squares)" | MIT, 2023; 99.8% per-square on its (unreleased) test set |
| [notnil/fenify-3D](https://github.com/notnil/fenify-3D) | physical, any angle | assumes crop | — | MIT, 2023; Unity synthetic + MTurk data |
| [georg-wolflein/chesscog](https://github.com/georg-wolflein/chesscog) | physical photos | corner/line detection | — | MIT; Wölflein & Arandjelović, J. Imaging 2021 |
| [davidmallasen/LiveChess2FEN](https://github.com/davidmallasen/LiveChess2FEN) | physical, live camera | yes | — | **AGPL-3.0**; Jetson-Nano oriented ([arXiv:2012.06858](https://arxiv.org/abs/2012.06858)) |
| [oliverfrost1/chess-video-move-detection](https://github.com/oliverfrost1/chess-video-move-detection) | physical video | YOLOv11 seg | yes | MIT, 2024; hand detector filters occluded frames; naive frame-diff move inference |
| [Chessvision.ai](https://chessvision.ai/) | screenshots, books, YouTube | yes | — | Closed; its "Video App" finds YouTube videos matching a position. No public API or accuracy numbers on the site. |

### Test on one Short (local, not committed)

GothamChess Short `pKuCQA3HXck`: 47 s, 9:16 layout, facecam on top, chess.com board (~50% of
frame width) in the middle, second facecam below, burned-in captions, mouse cursor, last-move
highlight, and a facecam-only final ~10 s.

Chess_diagram_to_FEN with default settings (`auto_rotate_image=True`) on **full, uncropped**
frames:
- 1080×1920 frame at t=1 s → `8/3k3p/4p3/1p2P1K1/1P1P3P/8/2P1p1P1/8`, exact match to the
  hand-read board. The same frame at **360×640** (≈21 px squares) was also exact.
- 2 fps over the video: boards returned for the first 73 frames (~0.3 s/frame CPU after warm-up);
  `get_fen` returns `None` on the facecam-only tail.
- Transient errors are almost all **piece-in-flight** frames: a king drawn on two squares
  (`…2KK…`), a moving pawn on both origin and destination.
- Passing `auto_rotate_image=False` produced garbage (the default pipeline's rotation step is
  load-bearing).

So localisation, facecam overlays and a 9:16 crop are **not** blockers for digital boards, at least
for chess.com's default theme. Untested: lichess themes, arrows drawn over pieces, flipped
boards, Chessbase/streamer overlays, boards < 25% of frame width.

### Specific hazards

- **Animations/premoves:** pieces mid-slide produce two-square or zero-square pieces. Drop frames
  whose FEN is illegal (two kings, pawns on back rank) and require N-frame stability.
- **Highlights/arrows/cursor:** tinted squares tolerated in the test; arrows untested. The
  last-move highlight is also a *feature*: it gives the side that just moved (ChessGPT used
  Colorthief on tile colours for exactly this).
- **Orientation:** Chess_diagram_to_FEN predicts flip; cross-check with coordinates, or let legality
  resolve it (a flipped board rarely yields a legal continuation).
- **Themes:** template matching needs one template set per theme; cluster by board colours and
  bootstrap templates from high-confidence model outputs.
- **Resolution:** Shorts are delivered up to 1080×1920 (tested); 360p still sufficed in one case.

## 4. Temporal alignment

1. **Board-change events, not scenes.** Scene detectors
   ([PySceneDetect](https://github.com/Breakthrough/PySceneDetect): `ContentDetector`,
   `AdaptiveDetector`, BSD-3) find cuts between facecam and board, which is useful for gating
   board presence, but a move changes two squares out of 64 and won't register. Instead: crop to the
   localised board and diff the 8×8 **square grid** (mean-abs-diff per square) at 5–10 fps. A move
   = 2–4 squares change and then stay stable for ≥ k frames. This also catches moves the 2 fps sampling
   missed (the test lost plies at the end; see §5).
2. **Map moves to words.** Each recovered ply gets a time `t_i` (first stable frame). Assign
   transcript words to plies by nearest-preceding or window `[t_i − δ, t_{i+1})`. Speakers
   typically *announce* before or *react* after, so learn δ per channel. Stronger: when a spoken
   move string matches a legal move from the current position (§2), use it as a hard anchor and
   interpolate between anchors (DTW over the ply timeline vs. anchor words).
3. **Commentary granularity.** In Shorts one comment often spans many plies ("he's going to get
   a queen… 10 seconds"). Store comments per *segment* (ply range), not forced per ply.
   Agadmator-2K did the same with GPT-4o labelling which moves a segment refers to (§7).

## 5. Recovering legal move sequences from noisy per-frame FENs

**Tested on the Short above** (python-chess, greedy, ~30 lines, scratch only): starting from the
first recognised FEN, for each new placement search all legal sequences of ≤ 2 plies that
reproduce it; if none, discard the frame as noise. Result: a 32-ply fully legal sequence
(`h5 e1=Q g4 Qg1 h6 Qc1+ Kh4 … Qxc2+ Kf6 Qf2+ Kg6 b4 Kg7 b3`), 23 noisy frames rejected, zero
ambiguous matches. Plies that fell between 2 fps samples later broke the chain (a ≥ 3-ply gap), so
the greedy search is not enough. Not checked against the real game score.

Design for the real thing:
- **State:** python-chess `Board`. Side-to-move, castling rights and e.p. are unknown for the first
  frame. Try both sides and choose by the longer legal continuation. Castling rights: infer from
  king/rook home squares, then let legality prune.
- **Emission model:** per-square class probabilities from the recogniser (not the argmax FEN).
  Score a hypothesis board by Σ log p(square class).
- **Transition model:** legal moves (1 ply, and 2–3 ply for missed frames, with a gap penalty),
  plus a low-probability **reset** transition to "any legal-looking position". That covers
  Shorts that cut between positions, show a puzzle, or set up a new board.
- **Decode:** Viterbi / beam search over (time × position) with a beam of ~20. Branching ≈ 35
  legal moves/ply keeps this cheap. Segments between resets are the unit of output.
- **ASR as a second emission:** a transcript move token near `t_i` that matches a legal move
  raises that transition's score (§2).
- **Existing code:** no open implementation of HMM/Viterbi decoding of chess video onto legal moves
  found. What exists does naive consecutive-frame diffing (chess-video-move-detection README: "If
  a piece disappears from one square and appears on another, it's considered a move") or
  single-image FEN. Agadmator-2K's "sliding-window move-tracking buffer" aligns to a *known* PGN
  and is not described further. **Gap: this component has to be written.**
- **Non-contiguous content:** puzzles ("white to play"), "what would you play here", replays of
  the same position. Treat each reset segment as its own record. Duplicate segments
  (same FEN re-shown) merge.

## 6. Validation and context recovery

- **Legality** (python-chess) is the first gate: a segment that decodes to ≥ 2 consecutive legal
  plies is almost certainly a correctly recognised position.
- **Lichess Opening Explorer** ([API spec](https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml)):
  host `explorer.lichess.org`, endpoints `/masters`, `/lichess`, `/player`, `/masters/pgn/{gameId}`.
  The spec marks `/masters` with `security: OAuth2`, so it needs a token. Useful for opening-phase
  segments: `topGames` → `/masters/pgn/{id}` recovers the full game score and names for famous
  games (Shorts love Morphy/Kasparov/Carlsen). Mid/endgame positions rarely hit the explorer.
- **Cloud eval** `/api/cloud-eval?fen=` (no auth in spec): "about 320 million positions";
  a hit is weak evidence the position occurred in real play. Bulk use should go through the
  [eval export](https://database.lichess.org/#evals) as the spec asks.
- **Puzzle DB** (public domain per the API spec, [download](https://database.lichess.org/#puzzles)):
  exact-FEN lookup against the puzzle CSV catches tactic Shorts and yields the solution line and
  source game URL. This project already uses the puzzle DB as its corpus.
- **Engine:** Stockfish eval before/after each recovered ply. A commentary "blunder!" landing on a
  ~0-cp move flags misalignment.
- **Streamer games:** chess.com/lichess player-game APIs can confirm a game when the Short names
  the players (e.g. `Jynxzi` vs `Tyler1`). Unverified path.

## 7. Prior work

- **Jhamtani et al. 2018** ([ACL P18-1154](https://aclanthology.org/P18-1154/),
  [code/data](https://github.com/harsh19/ChessCommentaryGeneration)): 298K move–comment pairs
  from 11K GameKnot games. Text forum, no video. It is the GameKnot share of `commentary/`.
- **ChessGPT** (Feng et al. 2023, [arXiv:2306.09200](https://arxiv.org/abs/2306.09200), App. E.5):
  **the closest precedent.** They collected ~83K chess videos via
  [scrapetube](https://github.com/dermasmid/scrapetube), filtered by a zero-shot NLI classifier,
  localised boards with GLIP (prompt "Chess, chessboard") + aspect-ratio filtering, split tiles via
  tensorflow_chessbot's gradient/Hough method, classified tiles with ResNet18 trained on the Kaggle
  [chess-positions](https://www.kaggle.com/datasets/koryakinp/chess-positions) set, and took side
  to move from the highlighted tile. Output: "million-scale English transcripts and board-language
  pairs". It used YouTube's timestamped transcripts, not ASR, and no legal-move decoding. They
  judged it too noisy: "it consistently contains more noise compared to the annotated PGN dataset",
  so ChessCLIP used PGN only. **Not released** (legal).
- **Agadmator-2K** (Harini S I et al., "Exploring Collaboration between a language and a
  non-language agent", [arXiv:2609.00474](https://arxiv.org/abs/2609.00474), Aug 2026): 1,900
  narrated Agadmator games (~500 h). Whisper-v3-large transcripts, timestamps aligned to a
  **known** PGN, GPT-4o labels which moves each segment references, and segments kept only if
  inferred order matches the PGN exactly. The first *move-aligned* video-commentary set. No board
  recognition: the PGN is given. Release status not stated in the parts I read (unverified).
- **MATE** (Wang et al., [arXiv:2411.06655](https://arxiv.org/abs/2411.06655), NAACL 2025
  short): 1M Lichess positions with strategy/tactic explanations. Text, not video.
- **Surveys.** The AI game-commentary survey ([arXiv:2506.17294](https://arxiv.org/abs/2506.17294),
  Table 1) lists chess datasets (Jhamtani, SentiMATE, ASSESS) with no video hours. Video-based
  commentary datasets there are sports/esports.
- **Shorts specifically:** no dataset derived from YouTube Shorts, or from any short-form chess
  video, was found. Nothing public pairs *recognised* positions with *spoken* commentary: ChessGPT's
  set is unreleased and Agadmator-2K starts from known PGNs.

## 8. Recommended pipeline (simplest first)

**Scope v0:** English Shorts from a hand-picked list of channels whose Shorts are digital-board
screen captures (analysis/tactic channels, not reaction clips). Physical boards are out of scope.

| Stage | Tool | Notes |
|---|---|---|
| List | `yt-dlp --flat-playlist --print id <channel>/shorts` | per channel; then `-J --skip-download` for duration, licence, caption availability |
| Captions | `yt-dlp --write-auto-subs --sub-langs en-orig --sub-format json3 --skip-download` | per-word timings for free |
| Media | `yt-dlp -f "bv*[height>=1280]"` + `-f ba` into a tmp dir, delete after processing | keep only derived data |
| Board present? | Chess_diagram_to_FEN at 1 fps (returns `None` when no board) | gates segments |
| Board track | fix bbox per segment; square-grid diff at 5–10 fps; recognise only on stable frames | template matching per theme once confident |
| FEN → moves | python-chess beam/Viterbi: legal 1–3-ply transitions, reset transition, per-square log-probs | the one component to write |
| ASR fallback | faster-whisper large-v3, `word_timestamps=True`, chess `initial_prompt`/`hotwords`, VAD | only if captions missing/poor |
| Align | ply times ↔ word times; spoken moves matched against legal moves as anchors; per-segment comments | |
| Validate | legality; Lichess puzzle DB / cloud-eval / masters explorer; Stockfish sanity | drop unverifiable single-frame positions |
| Emit | PGN with `[SetUp]/[FEN]`, `{comment}` per ply range, `yt:<id>` + timestamps | same shape as `cleaned_commentary/` |

First milestone: 50 Shorts from 3 channels, hand-check 20 decoded segments against the video,
report per-segment exact-sequence rate and comment-to-position relevance rate before scaling.

## 9. Risks

1. **Legal.** Downloading breaches YouTube ToS (automated access, download, face harvesting).
   §60d UrhG covers non-commercial research copies but not publication. ChessGPT withheld its
   YouTube set for this reason. Release derived data (IDs, timestamps, PGN) or code only; CC BY
   Shorts are the only clean subset.
2. **Content mix.** Many chess Shorts are reaction/meme clips: the test Short's audio is "HOLY
   MOLY… TICK TOCK… I win on time?" over a correctly recognised endgame. The position is right and
   the commentary carries no chess content. Expect a large relevance-filter loss. Needs a classifier
   (LLM judge, as in `commentary_gen/`) before anything counts as a pair.
3. **Commentary not about the shown position.** Speakers narrate ahead ("he's going to get a
   queen") or behind, or talk about the previous clip. Alignment windows mis-assign. Segment-level
   pairing + move-mention anchors mitigate but don't remove it.
4. **Recognition accuracy.** One chess.com-theme Short was exact on stable frames. Arrows,
   non-default themes, tiny boards, Chessbase/broadcast overlays and physical boards are untested.
   Physical boards are an order of magnitude harder (ChessReD 15% full-board accuracy until
   ChessQueries, which is unreleased).
5. **Missed plies.** Fast (bullet) play outruns low-fps sampling. Needs grid-diff event detection
   and multi-ply transitions, or segments break.
6. **Ambiguous/unknown state.** First-frame side-to-move, castling and e.p. rights are unobservable.
   Puzzles and set-up positions have no history. Recovered "games" are fragments.
7. **ASR on notation.** Whisper's accuracy on spoken SAN is unmeasured. Music/SFX raise
   hallucination risk. YouTube captions are casing-noisy and may be missing.
8. **Tooling fragility.** YouTube bot-detection/PO-token changes break yt-dlp periodically. A
   JS runtime is already "highly recommended".
9. **Selection bias.** Shorts over-represent brilliancies, blunders and famous games. The
   commentary is entertainment register, unlike GameKnot/lichess-study prose, so it may not
   transfer to the DSL-induction use in `paper/dsl-scoping.md`.
