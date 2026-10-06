//! Static exchange: what a side wins by capturing on a square and trading off the best way.
//! Same values as commentary_gen/board_tools.see, with the king too dear to give.

use shakmaty::{Board, Color, Role, Square};

pub fn value(r: Role) -> i32 {
    match r { Role::Pawn => 1, Role::Knight | Role::Bishop => 3, Role::Rook => 5, Role::Queen => 9, Role::King => 100 }
}

fn cheapest(b: &Board, sq: Square, c: Color, occ: shakmaty::Bitboard) -> Option<Square> {
    (b.attacks_to(sq, c, occ) & occ).into_iter().min_by_key(|&s| value(b.role_at(s).unwrap()))
}

/// What `by` gains by capturing on `sq` first; 0 when there is nothing to take or nothing takes.
pub fn see(b: &Board, sq: Square, by: Color) -> i32 {
    let Some(target) = b.piece_at(sq) else { return 0 };
    if target.color == by { return 0; }
    let Some(mut from) = cheapest(b, sq, by, b.occupied()) else { return 0 };
    let mut occ = b.occupied();
    let mut gains = vec![value(target.role)];
    let mut on_sq = value(b.role_at(from).unwrap());
    let mut side = by;
    loop {
        occ = occ.without(from);
        side = side.other();
        let Some(f) = cheapest(b, sq, side, occ) else { break };
        gains.push(on_sq - gains.last().unwrap());
        on_sq = value(b.role_at(f).unwrap());
        from = f;
    }
    while gains.len() > 1 {
        let g = gains.pop().unwrap();
        let last = gains.last_mut().unwrap();
        *last = -(-*last).max(g);
    }
    gains[0]
}

/// The opponent of the piece on `sq` wins material by capturing it.
pub fn hanging(b: &Board, sq: Square) -> bool {
    b.color_at(sq).is_some_and(|c| see(b, sq, c.other()) > 0)
}
