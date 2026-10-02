//! 0. Context: game phase and side to move.

use shakmaty::{Bitboard, Chess, Color, Position, Rank};

use crate::Out;
use crate::geo::Geo;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase { Opening, Middlegame, Endgame }

/// scalachess `Divider.score`: how mixed a 2x2 region is, for `y` = 1..7 counted from White.
fn score(y: i32, white: u32, black: u32) -> i32 {
    match (white, black) {
        (0, 1) => 1 + y,
        (0, 2) => if y < 6 { 2 + (6 - y) } else { 0 },
        (0, 3) | (0, 4) => if y < 7 { 3 + (7 - y) } else { 0 },
        (1, 0) => 1 + (8 - y),
        (1, 1) => 5 + (4 - y).abs(),
        (1, 2) => 4 + (7 - y),
        (1, 3) => 5 + (7 - y),
        (2, 0) => if y > 2 { 2 + (y - 2) } else { 0 },
        (2, 1) => 4 + (y - 1),
        (2, 2) => 7,
        (3, 0) | (4, 0) => if y > 1 { 3 + (y - 1) } else { 0 },
        (3, 1) => 5 + (y - 1),
        _ => 0,
    }
}

fn mixedness(g: &Geo) -> i32 {
    let mut acc = 0;
    for y in 0..7 { for x in 0..7 {
        let region = Bitboard(0x0303u64 << (x + 8 * y));
        acc += score(y + 1, (g.b.white() & region).count() as u32, (g.b.black() & region).count() as u32);
    }}
    acc
}

/// The phase as lichess's Divider assigns it, applied to this position alone.
pub fn phase(g: &Geo) -> (Phase, String) {
    let mm = (g.b.occupied() & !(g.b.kings() | g.b.pawns())).count();
    let back_w = (g.b.white() & Bitboard::from_rank(Rank::First)).count();
    let back_b = (g.b.black() & Bitboard::from_rank(Rank::Eighth)).count();
    let mix = mixedness(g);
    if mm <= 6 {
        (Phase::Endgame, format!("{mm} pieces besides kings and pawns; six or fewer is an endgame"))
    } else if mm <= 10 {
        (Phase::Middlegame, format!("{mm} pieces besides kings and pawns; ten or fewer is a middlegame"))
    } else if back_w < 4 || back_b < 4 {
        (Phase::Middlegame, format!("a back rank has emptied ({back_w} white and {back_b} black pieces left on them)"))
    } else if mix > 150 {
        (Phase::Middlegame, "the two armies have come into contact".to_string())
    } else {
        (Phase::Opening, format!("{mm} pieces besides kings and pawns, both back ranks still hold four or more, the armies still apart"))
    }
}

pub fn run(g: &Geo, pos: &Chess, o: &mut Out) -> Phase {
    let (p, why) = phase(g);
    let hit = match p { Phase::Opening => 0, Phase::Middlegame => 1, Phase::Endgame => 2 };
    o.one_hot("context.phase", None, &["opening", "middlegame", "endgame"], hit, vec![], &why);
    let w = pos.turn() == Color::White;
    o.global("context.turn.white", w, vec![], if w { "White to move" } else { "Black to move" });
    p
}
