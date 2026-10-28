# Claim players: Shapley over the features each quote names

**Game.** Decoder value v (win − loss) through the /encoder0/ln2 decoder, as in square_shap.py. Players are sets of
feature bits; a player is never removed. Switching it off replaces its bits by the average bits of the same occupant
(or emptiness) on the same square, over 16,040 training positions (claim_shap_means_n20000.pt); every other bit stays
on. Exact Shapley over the named players (at most 2).

**pos(sq)** = every bit on sq except `occupant`, `material*`, `material_win`, `mate`, `check`, `threat*` and the
`role.*` bits naming another square, plus the `role.{defends,needed_for,attacks,protected_by,needs}` bits on other
squares whose target is sq. Each bit belongs to one square, so players don't overlap. The `bishop_pair` /
`unlike_bishops` bits sit on the bishop squares, so "pair bits" are inside pos of the bishops. Occupant and tactic
bits always stay on.

**Centre** = pos(d4, e4, d5, e5). **Shield(K)** = pos of the king's square (with its `pawn_shield` and `king.castle`
bits) plus pos of the king's own pawns on the adjacent files up to three ranks ahead.

**Verdict** (fixed before the first run, which removed bits instead; results_claim_shap_drop.json). Each player's
φ = v(all) − v(player averaged), the same quantity the norms use, ranked against its kind (one knight, one king's shield, one side's isolated pawns, the centre …) in
the 300 corpus positions. s = φ signed for the player's owner. Owner = favoured side: s > 0 and percentile ≥ 75.
Owner = the other side: s < 0 and percentile ≤ 25. No owner (centre, empty square): φ signed for the favoured side,
norms pooled over both signs, > 0 and ≥ 75. "Low" players (out of play, inactive, backward): percentile ≤ 25.
Claim 27: |s| percentile ≤ 25. Claims 11 and 14 also need s(A) > s(B) in the joint game. A claim is supported
iff all its players pass. The joint Shapley over the named players is reported as the decomposition.

Squares are real-board squares (orig_fen); the run mirrors them for Black to move.

| # | claim | players | expected |
|---|---|---|---|
| 1 | W more space: two centre pawns, knight on 4th | A = pos(e4, f4) · B = pos(Nd4) | φ(A), φ(B) > 0 |
| 2 | B space: pawns e4, f5, knight e5 | A = pos(e4, f5) · B = pos(ne5) | φ(A), φ(B) > 0 |
| 3 | W space: pawns d5, f4, queen d4 | A = pos(d5, f4) · B = pos(Qd4) | φ(A), φ(B) > 0 |
| 4 | B space on the queenside | A = pos(b4, c3) | φ(A) > 0 |
| 5 | B cramped on the queen's wing | A = pos(W a3, b4, c4, d5) · B = pos(B b6, c6, d6, bb7, nd7) | φ(A) > 0, φ(B) > 0 (B hurts Black) |
| 6 | W dominates the centre | A = Centre | φ(A) > 0 |
| 7 | B superior in the centre | A = Centre | φ(A) > 0 |
| 8 + 43 | B holds the centre, d5 secured | A = pos(d5) (empty square) | φ(A) > 0 |
| 9 + 12 | B better: two bishops and the e4 knight | A = `bishop_pair` + `unlike_bishops` bits + pos(bc8, bf8) · B = pos(ne4) | φ(A), φ(B) > 0 |
| 10 | W's bishops decisive; B's knights lack bases | A = pair bits + pos(Be3, Bb3) · B = pos(nd7, ne5) | φ(A) > 0, φ(B) > 0 (B hurts Black) |
| 11 | B's two bishops beat bishop and knight | A = pair bits + pos(ba6, bb6) · B = pos(Bb4, Nc6) | φ(A) > φ(B) for Black |
| 13 | ...b4 supports B's c3 knight | A = pos(nc3) · B = pos(b4) | φ(A), φ(B) > 0 |
| 14 + 40 | W's unassailable f4 knight outweighs B's bishops | A = pos(Nf4) · B = pair bits + pos(bc8, bg7) | φ(A) > 0, φ(A) > −φ(B) |
| 15 | W cannot castle (…Bh3) | A = Shield(Ke1) | φ(A) > 0 (hurts White) |
| 16 | O-O-O brought W's king into safety | A = Shield(Kc1) | φ(A) > 0 |
| 17 | B forced to weaken the kingside | A = Shield(kg8) incl. f6, g5, h6 | φ(A) > 0 (hurts Black) |
| 18 | W king exposed, B king entrenched | A = Shield(Kg2) · B = Shield(kg8) | φ(A), φ(B) > 0 |
| 19 | ...h6 weakened B's king | A = Shield(kg8) (h6 is in it) | φ(A) > 0 (hurts Black) |
| 20 | W king exposed, B takes the initiative | A = Shield(Kd2) | φ(A) > 0 (hurts White) |
| 21 | W passed pawn and commanding king | A = pos(e4) · B = pos(Ke3) | φ(A), φ(B) > 0 |
| 22 | W passed d6 threatens d7, Qd6+ | A = pos(d6) + `threat.*` bits on d6 | φ(A) > 0 |
| 23 + 32 | W protected passer; rook cuts the king off | A = pos(g5) (its protection by f4 is in it) · B = pos(Rh7) | φ(A), φ(B) > 0 |
| 24 | B's three isolated pawns | A = pos(a5, d6, f5) | φ(A) > 0 (hurts Black) |
| 25 | B's isolated pawns need defending | A = pos(a5, c7, e5, e6) | φ(A) > 0 (hurts Black) |
| 26 | B's doubled c-pawn | A = pos(c6, c7) | φ(A) > 0 (hurts Black) |
| 27 | B's extra pawn doubled, of no account | A = pos(c4, c6) + their occupant bits | φ(A) for Black near 0 |
| 28 | B's isolated d5 is weak | A = pos(d5) | φ(A) > 0 (hurts Black) |
| 29 | B's isolated h7 is weak | A = pos(h7) | φ(A) > 0 (hurts Black) |
| 30 | B's rook on the only open file; B king more mobile | A = pos(rd8) · B = pos(kg8) | φ(A), φ(B) > 0 |
| 31 | W commands the 7th; W e3 isolated, g2 backward | A = pos(Ra7) · B = pos(e3, g2) | φ(A) > 0, φ(B) < 0 |
| 33 | B's a2 rook out of play | A = pos(ra2) | φ(A) for Black ≤ 25th pct |
| 39 | B's queenside space not real | A = pos(b4, c5, a6) | φ(A) > 0 (hurts Black) |
| 41 | B's b7 bishop inactive, keeping the pair a mistake | A = pos(bb7) · B = pair bits | φ(A) for Black ≤ 25th pct, φ(B) not > 0 |
| 42 | W's d5 knight can't be maintained | A = pos(Nd5) | φ(A) > 0 (W knight worth little) |

## Development: no clean mapping

34–38 (superior development, ahead in development, one move behind) name no squares. Closest:
A = pos(favoured side's knights, bishops, queen) · B = pos(the other side's), expected φ(A) > 0, φ(B) > 0 (B hurts
its owner). That tests piece activity, not development as tempo count. Keep or drop?

## Merged

8 = 43, 9 + 12, 14 = 40, 23 = 32: same position, the quote names both, so one game with both players.
