# Feature attribution on DSL trees

Does lc0's attribution agree with what a DSL leaf says? Start with one leaf on one board, and add more only if the
answer is readable.

## PoC: one leaf, one board

Book claim 28 (Znosko-Borovsky): "Black's isolated d5 pawn is weak and subject to attack."

- Board: `2rq1rk1/1b2bppp/pp3n2/2np2B1/B7/2N1PN2/PP2QPPP/2R2RK1 w - - 0 14`
- Tree: `Isolated(Squares([d5]))`, owner White
- Steps:
  1. **DSL verdict:** `dsl_verifier check` on the tree. Expected: holds.
  2. **IG:** `square_attribution.py --fens <that FEN> --ig 256 --ig-base square`, one board.
  3. **Leaf sum:** the Weakness bits on d5 (`pawn.isolated`, `weak.black`, `stability.*` on d5, `role.attacks` naming
     d5), signed for White.
  4. **Compare:** the same sum on each of Black's other pawns (a6, b6, f7, g7, h7).
- Output: one table with each pawn's sum and the DSL verdict.
- Readable means d5's sum ranks first among Black's pawns. If it doesn't, stop and look at which bits carry d5 before
  going further.
- Code: lc0-probe/dsl_poc.py, about 50 lines, reusing claim_concepts.py's View. No Rust change.

## Then, one step at a time

Each step only after the one before reads clearly.

1. **More leaves:** the other descriptive book claims, each written as a hand tree in data/dsl_trees.jsonl.
   Nulls are space_null.py's strict null for with/without claims, and matched square sets for descriptive ones.
2. **Leaf dump from Rust:** `Ev::node` records each leaf (kind, owner, squares, FEN). This replaces the hand-written
   squares.
3. **Because link:** IG of the cited move's legal-move log-softmax, on claims 13, 16 and 19. Two things must pass
   first:
   - lc0's move table, ported and tested against lc0's own output;
   - a decoder-fidelity check on that move's log-prob.

Details and review notes from the longer draft are in
~/.gstack/projects/xai-chess/rharvey-main-dsl-attribution-full-spec-20261106.md.
