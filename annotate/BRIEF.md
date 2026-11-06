# Agent brief

The instructions each Sonnet agent gets. Round 1 of `data/` and `data_2800/` used the earlier brief; the
changes since are listed at the end.

---

# Explain chess mistakes in the Reason DSL

You annotate moves from real Lichess games between players rated 2800+ that Stockfish grades as mistakes or
blunders. For each move you write why it is bad in the Reason DSL, a typed language for chess explanations,
and a verifier checks the claims against the board and Stockfish. Working directory:
/home/rharvey/Dokumente/Programmieren/xai-chess

## Your tool (the only thing you write with)

    ANNOTATE_DATA=data_2800 ROUND=2 FRESH=1 python3 annotate/attempt.py show CASE
    ANNOTATE_DATA=data_2800 ROUND=2 FRESH=1 python3 annotate/attempt.py try CASE --english "what you claim, in plain words" --dsl 'Because(...)' --note "..."

`show` prints the board, FEN, the move, the eval before and after, Stockfish's best line before the move
("Best") and after it ("Then"). `try` checks your reason, records it, and prints the verdict per part. Every
try is recorded for a human reviewer. A case ends when a try **holds** or is **unknown**, or after 5 tries.
On tries 2-5 use `--note` to say in one sentence what you changed and why. Put the DSL in single quotes.
Run `show` before every case and `try` at least once on every case.

Do not edit any file. Do not run a chess engine. You may use python-chess to inspect positions and play
out lines: `python3 -c "import sys; sys.path.insert(0, '.pylibs'); import chess; ..."`.

## What to write

First decide, as a strong player, why the move loses what the eval says it loses. Then write that reason.

- The reason must account for the loss. A claim that is true but would hold for many other moves too
  (an attack on a defended piece, a pawn won that comes back, a bare `Initiative`) does not explain it.
- When your real reason fails, change how you write it, not what you claim. Do not fall back to a weaker
  claim just because it holds: a hollow pass is worse than an honest failure. If the DSL cannot say it,
  let the case fail and name the gap in your report.
- Shape: `Because(Rated(Blunder), WHY)` or `Because(Rated(Mistake), WHY)` (or Dubious), the grade `show` prints.

Pick the claim that names your point:

| The point | Write |
|---|---|
| Wins material | `Material { get: [Knight] }` inside the `Allows`/`After` line that wins it |
| Wins material for material | `Material { get: [Knight, Bishop], give: [Rook] }` |
| Is N pawns up or down (the standing balance) | `Material { get: [Pawn, Pawn], total: true }` |
| Throws away a won position / a forced mate | `Loses(Win)` / `Loses(Mate)` |
| A better move wins something | `Instead(Bxe5, Material { get: [Pawn] })` (counts what Bxe5 itself takes) |
| A pawn queens | `Promote(sq)`: the pawn's square or its queening square |
| A draw by perpetual check | `Perpetual` (e.g. `Allows([Qh5+], Perpetual)`) |
| A bad or good piece | `Quality(sq, Bad)` / `Quality(sq, Good)`; `Better { pieces: [..], than: [..] }` |
| Gives up the bishop pair | `Loses(BishopPair)` |
| Weakens squares or pawns | `Concedes(Weakened(..))`, `Concedes(Hole(..))` (they must be new after the move) |
| A defender walks away | `Loses(Defend { by: sq, target: sq })` |
| Material left en prise | `Allows([], Hanging(..))` |
| A lasting edge | `Permanent(r)`: r still holds 8 plies along the engine's line |
| One factor beats another | `Outweighs(mine, theirs)`: they point opposite ways and the eval sides with the first; `Balances`: the eval is level |

Material: name only what each side loses inside the counted span: after the played move (it is not
counted), or from before m under `Instead(m, ..)`. Two pawns are `[Pawn, Pawn]`. Do not use `total` for a
change; it is the whole balance on the board.

## Rules the verifier enforces

- `Rated` uses Lichess's bands on the winning chances lost against the best move: Blunder 0.26 or more,
  Mistake 0.16-0.34, Dubious 0.06-0.24, Good under 0.14 (neighbouring grades overlap).
- Under `Rated(Blunder/Mistake/Dubious)`, `Because` checks that the material your reason names
  (`Material` in an `Allows` or an `Instead`) nets the mover a loss. Zero or a gain fails.
  Under `Instead(m, ..)` material counts from before m, so what m itself takes counts.
- Every move of an `After` or `Allows` line must be (nearly) best play for its mover: within 0.1 winning
  chances of that side's best move. Use the engine's line or one as good. Passes (`--`) are exempt.
- `Pin` needs something to win: the piece behind is worth more, or falls once the front piece moves.
- `Mate` holds on a mate on the board or a forced mate for the owner.
- `Material`: every piece named comes off, and the settled net has the sign of get minus give.

## The DSL

Read `dsl_verifier/src/dsl_claude.rs` lines 1-310 first: every claim and connective with its doc comment.
Lines 357 onwards are worked examples from real annotated games; skim several for the syntax.
Text syntax is Rust Debug form without quotes:

- Moves and squares bare: `Allows([Qxf2+, Kh1, Qxg3], Material { get: [Pawn, Pawn] })`. Never `"Qxf2+"`.
- Vec arguments are lists in brackets: `[e4]`, `Material { get: [Knight] }`, `Squares([f7, g7])`.
- Struct variants: `Attack { by: e5, targets: [d7, g6] }`, `KingZone { opponent: true }`, `Pin { by: b5, front: c6, behind: e8 }`.
- Unit variants bare: `Mate`, `Check`, `Initiative`. Levels: `Degree(Decisive, ...)` with Slight / Clear / Decisive.
- In a line, `--` is a pass (needed in `Threatens`, where the owner plays all moves: `Threatens([Rf2, --, Qh2], Mate)`).
- Squares name pieces where the reason is evaluated: inside a line, at the end of that line.
- "At some point in a line": `After(l1, And([r1, After(l2, r2)]))`.

Ownership (who a claim is about):
- The top-level reason belongs to the side that played the move.
- `Allows`, `Prevents`, `Removes`, `Concedes` hand the reason to the opponent; inside them, `opponent: true`
  means the mover's side.
- Lines start with the side to move after the played move. `Allows(line, r)`: after the line the opponent
  has r. `After(line, r)`: after the line the owner has r. `Threatens(line, r)`: the owner plays the line unopposed.
- `Instead(Move, r)`: what an alternative move (from the position before the played move) would have achieved.
  A line inside it starts with the reply to that move.

Example that holds (28.g3, a blunder; Then line 28...Qxf2+ 29.Kh1 Qxg3):

    Because(Rated(Blunder), And([Allows([Qxf2+, Kh1, Qxg3], And([Material { get: [Pawn, Pawn] }, Degree(Decisive, Pressure(KingZone { opponent: true }))])), Instead(Rd1, Allows([Qxf2+], Material { get: [Pawn] }))]))

## Report back

When all your cases are done, reply with one line per case: `CASE: holds|unknown|failed after N tries`,
then at most five bullets on what the DSL could not express or where the verifier seemed wrong
(name the case). Nothing else.

---

## Changes from the round-1 brief

- Decide the reason first; it must account for the loss. No fallback to a weaker claim that holds; let it
  fail and report the gap. (Round 1: about 10 holds in `data_2800` were hollow by the agents' own account.)
- Named the positional claims the agents never used (`Quality`, `Better`; `BishopPair` 0 times in `data_2800`)
  and the new ones, as a table of point → claim (`Loses(Mate)`, `Material { get, give, total }`, `Promote`'s
  squares, `Perpetual`, `Permanent`, `Outweighs`/`Balances`).
- `WinMaterial`, `WinTrade` and `Material(n)` are now one claim, `Material { get, give, total }`; the old forms
  still parse.
- The rules for `Instead` material, `Pin` and `Mate`, as the verifier now checks them.
- `show` and `try` on every case (round 1: two cases skipped, one reported without a try).
