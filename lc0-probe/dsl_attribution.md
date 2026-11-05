# Feature attribution on DSL trees

Spec, approved in design 2026-11-06. Nothing has run.

The book claims (square_shap_textbook.js, 43 positions; concepts in claim_concepts.md) are written as dsl_verifier
Reason trees. Integrated gradients through the /encoder0/ln2 decoder then do four things with them:

1. **Leaf verdict:** a network verdict on each positional leaf, next to eval.rs's verdict.
2. **Because link:** whether the leaf a tree gives as the reason for a move carries the move's policy logit.
3. **Unnamed remainder:** the move-logit attribution that falls outside every leaf in the tree.
4. **Rank:** a tree's leaves ordered by their share in (2).

## Shared core: leaf → bits

- **Rust (dsl_verifier):** `leaves` mode. It takes a position and a tree, and writes one JSON line per positional
  leaf: kind, owner, resolved squares, and the path in the tree. Squares come from the same Region resolver eval.rs
  uses. Tactical leaves (WinMaterial, Attack, Pin, Check, Mate, Material, Trade) are written with `"tactical": true`
  and are not attributed, because eval.rs already checks them against the engine.
- **Python (lc0-probe/dsl_concepts.py):** maps each leaf kind to a bit pattern over those squares. The patterns are
  claim_concepts.md's table (Space, Control/Centre, Hold, Outpost/Hole, KingSafety/Weakened, Activity/Better/Quality,
  Passer, BishopPair, Pressure/Threat, Isolated/Doubled/Hanging → Weakness). Bits are signed for the owner. Leaf kinds
  with no bits (Develop, Backward, Initiative, Tempo) are reported as "no bits", not scored.
- **IG:** square_attribution.py with 256 steps and the square baseline. Each IG result is reused by every part that
  attributes the same target on the same board.

## Targets

| part | target | board |
|---|---|---|
| 1 | W − L, side to move | the claim's board (and the with/without boards in claim_concepts.md) |
| 2–4 | lc0 policy logit of the cited move | the board before that move |

Policy: square_attribution.py gets `--target policy --move <uci>`. It cuts the graph at `/output/policy` instead of
`/output/wdl` and takes that move's index, using lc0's 1858-move mapping for the side to move.

## Parts

**1. Leaf verdict.** The leaf's signed attribution S. With/without claims use the claim_concepts.md test unchanged.
For descriptive claims, S is compared with a null: the same leaf kind on 200 random square sets of the same size and
shape (same rank band and file count for regions, same piece type for piece leaves) on the same board. The verdict is
holds if S is above the null's 90th percentile (or below the 10th for a negative claim), unknown in between. Scored
against the author: agree / disagree / unknown, and shown next to eval.rs's verdict for the same leaf.

**2. Because link.** For `Because(m, r)`, `Instead(m, r)`, `Allows(m, r)` and `Threatens(m, r)` trees on the claims
below: r's share = Σ|IG| over r's bits / Σ|IG| over all bits, for the move-logit target. Null: 200 random bit sets of
the same size, drawn from the same feature families. Reported as the share and its null percentile. No threshold is
fixed before the first run.

**3. Unnamed remainder.** 1 − (share of the union of all of the tree's leaf bits), with the remaining |IG| grouped by
feature family and by square, top 10 listed per claim.

**4. Rank.** Leaves ordered by their (2) share, only for trees with two or more positional leaves.

## Claims for parts 2–4

Only claims that cite a move. The move and the board before it come from the quote. Each row is confirmed against the
quote before the run.

| # | move | board before |
|---|---|---|
| 13 | ...b4 | b-pawn taken back to b5 |
| 15 | ...Bh3 | bishop taken back to d7 |
| 16 | 12.O-O-O | Ke1, Ra1 |
| 19 | 8...h6 | pawn taken back to h7 |
| 22 | d7 or Qd6+ (Threatens) | the claim's board |
| 26 | ...b6 | to check in the quote |
| 37 | ...Rd8 / Qxd8 | to check in the quote |

Claims 6, 17, 20 and 35 name a square or a move in the quote, but not as the claim's cause. They are left out
unless the quote shows otherwise.

## Tests

- cargo: for each Region form (Squares, Kingside, Queenside, Centre, GreaterCentre, Half, KingWing, KingZone, Line),
  the `leaves` squares equal the squares eval.rs uses.
- pytest: the IG completeness gap is under 0.05 at 256 steps for both targets, on 5 book boards.
- pytest: the policy target's logit at the full input equals ONNX's /output/policy for that move to within 1e-4.

## Outputs

- lc0-probe/data/dsl_trees.jsonl: the book claims as trees (written by hand, one per claim; reviewed before the run).
- lc0-probe/results_dsl_attribution.json, one record per claim: leaves, eval.rs verdicts, S and null percentiles,
  shares, remainder.
- A table in journey.md only if asked.
