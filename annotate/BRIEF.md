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
  Nor do `Instead(m, Rated(Stronger))` alone, an `Attack` without the material it wins, or a `Not(..)` about
  the alternative. A blunder that loses far more than a pawn is rarely explained by one pawn: find what else goes.
- When your real reason fails, change how you write it, not what you claim. Do not fall back to a weaker
  claim just because it holds: a hollow pass is worse than an honest failure. If the DSL cannot say it,
  let the case fail and name the gap in your report.
- Shape: `Because(Rated(Blunder), WHY)` or `Because(Rated(Mistake), WHY)` (or Dubious), the grade `show` prints.

Pick the claim that names your point:

| The point | Write |
|---|---|
| Wins material | `Allows([Rxe8+], Material { get: [Rook] })`: the line ends with the capture of every piece named |
| Wins material for material | `Material { get: [Queen], give: [Rook, Pawn] }`: `give` includes what the played move itself took |
| Is N pawns up or down (the standing balance) | `Material { get: [Pawn, Pawn], total: true }` |
| Allows mate / throws away a forced mate | `Allows([Qxg2#], Mate)` / `Loses(Mate)`, with `Instead(Re8+, Mate)` for the move that mated |
| Throws away a win, allows a draw | `Loses(Win)`; `Allows(line, Draw)` |
| A better move wins something | `Instead(m, After(line, Material { .. }))`: the line runs to the capture; `Instead(m, Material { .. })` counts only m itself, so a piece m takes and loses back nets 0 |
| A pawn queens | `Allows([b7], Promote(b8))`: the pawn's square or its queening square |
| A draw by perpetual check | `Perpetual` only when, along the engine's line, every move of the checking side is a check and a position repeats within 12 plies; otherwise `Draw` |
| A bad or good piece | `Gains(Quality(g7, Bad))` when the move itself makes it bad, `Allows(line, Quality(d6, Bad))` when the line does; `Better { pieces: [g2], than: [g7] }` |
| Gives up the bishop pair | `After([gxf6], Loses(BishopPair))`: the pair goes when the bishop is taken, so wrap it in the line (`Loses(BishopPair)` alone fails: the bishop is still on f6) |
| Wrecks the pawns | `Allows([Bxf3, gxf3], Doubled(Kingside))` (also `Isolated`, `Backward`) |
| Weakens squares or pawns | `Concedes(Hole(Squares([d5])))`: no pawn can ever guard it; `Concedes(Weakened(..))`: one side still can. Both must be new after the move |
| A defender walks away | `Loses(Defend { by: h7, target: h6 })`, with the `Allows` line that wins the piece |
| Material left en prise | `Allows([], Hanging(Squares([e8])))` |
| A lasting edge | `Permanent(r)`: r still holds 8 plies along the engine's line |
| The king is under attack | `Pressure(KingZone { opponent: .. })`: an attack on the king's ring (weighted pieces, weak squares, safe checks) within 8 plies of the engine's line |
| More control over the centre | `Outnumber(Centre)`: the owner outnumbers the opponent's attackers on more of d4, e4, d5, e5 than the other way round (`GreaterCentre` for c3-f6) |
| One factor beats another | `Outweighs(mine, theirs)` / `Balances(mine, theirs)`: they point opposite ways |

Material: name everything each side loses from before the played move (before m under `Instead(m, ..)`)
to the end of the line, including what the played move itself takes. Two pawns are `[Pawn, Pawn]`. Do not use `total` for a
change; it is the whole balance on the board.

## Rules the verifier enforces

- `Rated` uses Lichess's bands on the winning chances lost against the best move: Blunder 0.26 or more,
  Mistake 0.16-0.34, Dubious 0.06-0.24, Good under 0.14 (neighbouring grades overlap).
- Under `Rated(Blunder/Mistake/Dubious)`, `Because` checks that the material your reason names
  (`Material` in an `Allows` or an `Instead`) nets the mover a loss. Zero or a gain fails.
  Material counts from before the played move (before m under `Instead(m, ..)`), so what that move takes counts.
  It reads only `Allows(line, ..)`, `Instead(m, ..)` and `Instead(m, After/Allows(line, ..))`, alone or as items
  of the top `And`. Inside `Instead(m, And([..]))` it counts m alone: put `After` straight under `Instead`.
- Every move of an `After` or `Allows` line must be (nearly) best play for its mover: within 0.15 winning
  chances of that side's best move, up to 0.3 in decided positions. Use the engine's line or one as good. Passes (`--`) are exempt.
- `Pin` needs something to win: the piece behind is worth more, or falls once the front piece moves.
- `Mate` holds on a mate on the board or a forced mate for the owner.
- `Material`: every piece named is off the board at the end of the line, and the net once the exchanges
  settle has the sign of get minus give.
- `Loses`, `Gains`, `Concedes` compare the position before the played move with the one right after it.
  For something lost in the reply, wrap it: `After([gxf6], Loses(BishopPair))`.
- `Attack`, `Pin`, `Defend` read the board at the end of their line: `by` must be the owner's piece there.
  End a fork's line on the forking move, and give the capture its own line (example 4 below).
- `Tempo` takes the free moves, not a number: `Tempo([Be6])`.
- `Initiative`: along the whole engine line, the annotated move first, the owner forces first, or from its first
  forcing move on forces more often than the opponent. Forcing: check, capture (not a recapture), attack on a
  bigger or loose piece, aiming at more of the king's squares, pawn lever. The annotated move must itself be
  forcing and the owner's next move too: a quiet move can't take the initiative. A losing side can still have it: write what outweighs it,
  `Allows([], Outweighs(Material(3), Initiative))` ("Black has the initiative, but White's extra knight counts for more").

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

## Examples

Each holds under the current verifier unless marked.

1. `r5k1/2p4p/1p1p2p1/p2Pp3/1PPq1r2/P6P/5PP1/R1Q1R1K1 w - - 4 28`, 28.g3??. Qxf2+ and Qxg3 win two pawns
   and open the king; 28.Rd1 loses only f2.

       Because(Rated(Blunder), And([Allows([Qxf2+, Kh1, Qxg3], And([Material { get: [Pawn, Pawn] }, Degree(Decisive, Pressure(KingZone { opponent: true }))])), Instead(Rd1, Allows([Qxf2+], Material { get: [Pawn] }))]))

2. `5r1k/p5q1/1pp1Q3/3pPp1p/3P4/2PB1P1P/PP6/1K2R3 w - - 1 36`, 36.Qxf5??. It takes a pawn and loses the
   queen for rook and pawn. The pawn Qxf5 took counts, so it goes in `give`.

       Because(Rated(Blunder), Allows([Rxf5, Bxf5], Material { get: [Queen], give: [Rook, Pawn] }))

3. `r2qk3/pp1nbp2/2pp1r2/4pPQ1/7p/1PNP3P/1PP2PPB/R3K2R b KQq - 0 19`, 19...Qb6?. 19...Rg6 hits the queen
   and wins g2. Holds as written; wrapped as `Instead(Rg6, And([Rated(Stronger), After(..)]))` it fails
   ("nets the mover +0 pawns"), because the line under `And` is not counted.

       Because(Rated(Mistake), Instead(Rg6, After([Qh5, Rxg2], Material { get: [Pawn] })))

4. `3r2k1/5pp1/p3p3/2P5/3n4/5P1B/1PK2P1P/6R1 w - - 0 25`, 25.Kc1??. Ne2+ forks king and rook, Nxg1 takes it.
   With the fork on the long line it fails: on e2 there is no knight any more.

       Because(Rated(Blunder), And([Allows([Ne2+], Attack { by: e2, targets: [c1, g1] }), Allows([Ne2+, Kc2, Nxg1], Material { get: [Rook] })]))

5. `5qk1/3p2p1/1Qp1p3/6r1/3P3p/2P5/2P1B2P/5N1K b - - 1 32`, 32...Qf4??. Black throws away the win;
   Qd8+ and Bd3+ draw. `Allows([], Perpetual)` fails: the engine's line plays Nd2, no check.

       Because(Rated(Blunder), And([Loses(Win), Allows([Qd8+, Kh7, Bd3+], Draw)]))

6. `r1bq1rk1/pp1nnpbp/4p1p1/3p4/3p3P/2P2NP1/PP1BPPB1/RN1QK2R w KQ - 0 10`, 10.Qc1?. It ignores the
   tension; ...e5 gives Black the centre.

       Because(Rated(Mistake), Allows([e5], Outnumber(Centre)))

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
- King `Pressure`: no winning chances any more. `Degree` on it is unchecked, and a refuted attack still holds.
- King `Pressure` looks 8 plies along the engine's line instead of 4.
- Added `Outnumber(Centre)` for "more control over the centre".
- Hollow shapes named (`Instead(m, Rated(Stronger))` alone, a bare `Attack`, `Not(..)` about the alternative,
  one pawn for a big blunder). Votes in `data/`: attack-only reasons 5 of 22 correct, pawn-only 18 of 37, piece won 24 of 33.
- Table rows now show a tested shape each: material lines end on the capture, `give` holds what the move took,
  `Loses(Mate)` with `Instead(m, Mate)`, `Loses(Win)` / `Allows(line, Draw)`, `Promote` on the queening square,
  `Gains`/`Allows` for `Quality`, `Better`, `Doubled`, `Hole` vs `Weakened`, `Loses(Defend)`, `Hanging`.
- `Instead(m, Material)` without a line counts m alone (15 failed tries, e.g. `CjWb3hXw-75`, `8ahw0DRs-19`): use `Instead(m, After(line, ..))`.
- `Because` does not count a line under `Instead(m, And([..]))`: true reasons failed on it (`eFugMq3R-80`, `2FEnRcK7-37`, `1gURpSly-24`).
- `Perpetual` only for a pure check sequence; it failed on all three perpetual-like cases (`n04msaiF-63`, `hgpVUOCL-43`, `oVAuIb3u-92`), where `Draw` holds.
- `Loses(BishopPair)` and other `Loses` on a reply go under `After(line, ..)` (`pMNre5W8-34`).
- Rules for squares read at the end of a line (22 failed tries name squares that do not fit the board there) and for `give` pieces taken only in the settling (4 tries).
- `Tempo` takes moves (4 tries wrote a number); `Initiative`'s test stated (14 of 18 tries failed).
- Six worked examples, each rechecked; three show the failing variant.
