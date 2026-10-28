# Claim concepts: attribution of the claim's concept, with and without the credited pieces

Draft for review. Nothing has run.

**Measure.** Integrated gradients through the /encoder0/ln2 decoder (square_attribution.py --ig 32), per bit.
A concept's attribution = sum over its bits, signed for the side named. Two kinds of test:

- **With / without**: the quote credits a piece (or a move) with the concept. The concept's attribution on the real
  board vs the board with that piece taken off (or the move taken back). Each credited piece separately. Reported raw
  and as a share of the board's total |attribution|, because taking material off shifts every attribution.
- **Descriptive**: the quote only states the concept. Its attribution for the named side, against the same concept
  for the other side where the quote compares them.

Squares are real-board squares.

## Concepts (bit sets)

| concept | bits |
|---|---|
| Space(X, files) | X's `control.reach.t`, `control.geometry.t` with t in the opponent's half (on those files), plus `square.control.X*` on those t |
| Centre(X) | same, t in d4 e4 d5 e5 |
| Hold(X, s) | X's control bits targeting s, `square.control.X*` and `strong.X.*` on s |
| Outpost(s) | `strong.<owner>.*` on s, `stability.*` on s (an enemy pawn or piece can hit it), `role.protected_by` on s |
| KingSafety(K) | on K's square and its 8 neighbours: `pawn_shield.*`, `king.castle`, `pawn_cover.<owner>*`, `square.control.*`; the enemy's control bits targeting them; `role.attacks` bits naming K |
| KingMobility(K) | K's `mobility.*`, `control.*`, `blocked_by.own.*` |
| Pair(X) | `bishop_pair.*`, `unlike_bishops.*` on X's bishops |
| Activity(piece) | the piece's `control.*`, `mobility.*`, `xray.*`, `role.*` |
| Passer(p) | `pawn.passed`, `pawn.rank4_6`, `pawn.rank7` on p; `role.protected_by` on p; `rook.behind_passer` naming p |
| Threat(s) | `threat.*` on s |
| Weakness(pawns) | `pawn.isolated`, `pawn.doubled_with.*`, `pawn.hanging`, `weak.<owner>` on the pawns' squares; `role.attacks` and `stability.*` naming them; for 25 also the owner's `role.defends` bits naming them (pieces tied down) |

## With / without

| # | claim | concept | taken off / back | expected |
|---|---|---|---|---|
| 1 | W space: two centre pawns, knight on the 4th | Space(W) | Pe4, Pf4, Nd4 | drops for each |
| 2 | B space: e4, f5, knight e5 | Space(B) | pe4, pf5, ne5 | drops for each (ne5: −0.039 raw, done) |
| 3 | W space: d5, f4, queen d4 | Space(W) | Pd5, Pf4, Qd4 | drops for each |
| 4 | B space on the queenside | Space(B, a–d) | pb4, pc3 | drops for each |
| 5 | B cramped on the queen's wing | Activity of B's pieces on a–d (mobility only) | Pa3, Pb4, Pc4, Pd5 | rises for each |
| 13 | ...b4 supports the c3 knight | Outpost(c3) | pb4 | drops |
| 14 + 40 | unassailable Nf4; B's bishops have little to say | Pair(B) | Nf4 | rises |
| 15 | W cannot castle (...Bh3) | KingSafety(Ke1) incl. `king.castle` | bd7 | rises |
| 16 | O-O-O brought W's king into safety | KingSafety(W king) | take 12.O-O-O back (Ke1, Ra1) | lower before |
| 19 | ...h6 weakened B's king | KingSafety(kg8) | take ...h6 back (pawn to h7) | higher before |
| 22 | d6 threatens d7 and Qd6+ | Threat(d6), Passer(d6) | Qd3 | drops |
| 23 | protected passed pawn | Passer(g5) | Pf4 (the protector) | drops |
| 30 | B king more mobile, W king hampered by own pawns | KingMobility(Kg1) | Pf2, Pg2, Ph2 | rises |
| 32 | rook cuts the B king off | KingMobility(kf8), signed for Black | Rh7 | rises |

## Descriptive

| # | claim | concept | expected |
|---|---|---|---|
| 6 | W dominates the centre | Centre(W) vs Centre(B) | W > 0, W > B |
| 7 | B superior in the centre | Centre(B) vs Centre(W) | B > 0, B > W |
| 8 + 43 | d5 secured for B | Hold(B, d5) | > 0 |
| 9 + 12 | B's two bishops; strongly posted ne4 | Pair(B), Outpost(e4) | both > 0 |
| 10 | W's bishops decisive; B's knights lack bases | Pair(W); Outpost(nd7), Outpost(ne5) | Pair(W) > 0; outposts ≤ 0 |
| 11 | two bishops beat bishop and knight | Pair(B) + Activity(ba6, bb6) vs Activity(Bb4, Nc6) | B > W |
| 17 | B forced to weaken the kingside | KingSafety(kg8), signed for Black | < 0 |
| 18 | W king exposed, B king entrenched | KingSafety(Kg2), KingSafety(kg8) | W < 0, B > 0 |
| 20 | W king exposed | KingSafety(Kd2) | < 0 |
| 21 | W passed pawn, commanding king | Passer(e4), Activity(Ke3) | both > 0 |
| 24 | three isolated pawns, attacked | Weakness(a5, d6, f5), signed for Black | < 0 |
| 25 | isolated pawns tied to pieces | Weakness(a5, c7, e5, e6) incl. defenders | < 0 |
| 26 | doubled c-pawn | Weakness(c6, c7) | < 0 |
| 27 | extra pawn doubled, of no account | `material.up` + Weakness(c4, c6), signed for Black | sum ≈ 0 |
| 28 | isolated d5 weak | Weakness(d5) | < 0 |
| 29 | isolated h7 weak | Weakness(h7) | < 0 |
| 31 | W commands the 7th; e3 isolated, g2 backward | `rook.seventh` + Activity(Ra7); Weakness(e3, g2) | > 0; < 0 |
| 33 | ra2 out of play | Activity(ra2) | ≈ 0 |
| 39 | B's queenside space not real | Space(B, a–c) + Weakness(b4, c5, a6) | sum ≤ 0 |
| 41 | bb7 inactive, keeping the pair a mistake | Activity(bb7); Pair(B) | ≈ 0; ≤ 0 |
| 42 | Nd5 can't be maintained | `stability.*` on d5, signed for White | < 0 |

## Gaps

- **Development (34–38):** no bits encode development (moves spent, pieces off the back rank). Drop, or add a feature.
- **Backward pawn (31's g2):** no `backward` bit; Weakness has only isolated/doubled/hanging and the attacks.
- **Absences:** "unassailable" (14) and "lack bases" (10) are the *absence* of `stability.*` / `strong.*` bits, and
  attribution can't credit a bit that isn't there. 14 is tested through the with/without on Pair(B) instead.
- **"≈ 0" and "> 0" thresholds** for descriptive claims are not fixed yet.
