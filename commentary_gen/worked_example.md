--- Worked example: the whole procedure once, on a real position ---

Moves so far: 1. c4 e5 2. Nc3 Nc6 3. g3 g6 4. Bg2 Bg7 5. d3 d6 6. Nf3 h6 7. O-O Be6 8. Rb1 Qd7
9. b4 Nge7 10. Re1 Bh3 11. Bh1 O-O 12. b5 Nd8 13. a4 f5 14. Qb3 Kh8 15. Ba3 Ne6 16. Nd5 g5 17. e3

From the fact sheet for that position (an extract):
F5   Hanging: No piece is hanging.
F6   Nxd5: takes the knight, an even trade, no material change
F14  A pawn cannot go back: it will never guard d3 (now covered by White queen on b3),
     f3 (now covered by White bishop on h1) again; it guards d4 f4 instead.
F15  Line opened: the White rook on e1 now also reaches e3.
F16  White: king on g1, castled; shield f2 g3 (advanced) h2; no flight square; enemy pieces
     bearing on the king zone from h3.
F21  White bishop on h1: 1 legal move; own knight on f3 in its way; 4 own pawns on its light
     squares (d3 a4 c4 b5). Role: out of play.
F30  Black knight on e6: 4 legal moves (2 without losing material); defends g5 c7 g7 f8. Role: free.
F35  Central squares in White's camp no White pawn can ever guard: c3 f3

STEP 1 — EVALUATE
Tactical
  [F5] Nothing is hanging for either side.
  [F6] Black's only capture is Nxd5, an even trade.
  [F14] e3 takes d4 and f4 and gives up d3 and f3 for good.
  [F15] The rook on e1 now looks up the e-file, behind its own pawn.
Strategic
  [F14, F30] d4 was the square the knight on e6 wanted.
      attack_map("d4") -> d4: empty; attacked by White from e3 f3; attacked by Black from e5 e6
      after("Nd4 exd4") -> White up 3; Black's best is exd4, White stays up 2.
      So the knight cannot stand there any more.
  [F14] d3 is now held only by the queen on b3.
  [F16] White's king has no flight square and the bishop on h3 already looks at the zone.
  [F21, F35] The bishop on h1 has one move with four of its own pawns on its colour; c3 and f3
      are squares no White pawn can ever cover.

STEP 2 — PICK THE POINT
No mate and no check (level 1). Nothing hanging, no capture that wins material (level 2).
Level 3: the move's own permanent change, d4 and f4 taken, d3 and f3 given up [F14]. That is
what e3 did, so that is the point. The bishop on h1 [F21] is true but the move did not cause it.

STEP 3 — WRITE

NOTE:
The real point of e3 is d4: the knight on e6 was heading there, and now it cannot, because the
pawn simply takes it. The price is that a pawn does not come back, so d3 and f3 are given up for
good and the queen on b3 is the only piece still covering d3. If the e-pawns ever come off, the
rook on e1 is already sitting on the file.
