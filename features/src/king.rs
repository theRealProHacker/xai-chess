//! 3. King safety: where the king stands, its pawn shield, attackers against defenders.

use shakmaty::{Bitboard, Color, Role};

use crate::Out;
use crate::geo::{Geo, fr, fwd, name, names, on, title};

/// The king's square and the squares next to it.
pub fn zone(g: &Geo, c: Color) -> Bitboard {
    match g.king(c) { Some(k) => shakmaty::attacks::king_attacks(k).with(k), None => Bitboard::EMPTY }
}

pub fn run(g: &Geo, o: &mut Out) {
    for c in Color::ALL {
        let Some(k) = g.king(c) else { continue };
        let (kf, kr) = fr(k);
        let hit = match kf { 6 | 7 => 0, 0..=2 => 1, _ => 2 };
        o.one_hot("king.castled", Some(c), &["kingside", "queenside", "uncastled"], hit, vec![k.to_string()],
                  &format!("{} king on {k}", title(c)));

        let mut missing = vec![];
        for f in kf - 1..=kf + 1 {
            if !on(f, 0) { continue; }
            if !(1..=2).any(|d| g.pawn(f, kr + d * fwd(c), c)) { missing.push(format!("{}-file", (b'a' + f as u8) as char)); }
        }
        let why = if missing.is_empty() { format!("pawns in front of the king on {k}") }
                  else { format!("no pawn one or two ranks in front of the king on the {}", missing.join(", ")) };
        let sq: Vec<String> = (kf - 1..=kf + 1).filter(|&f| on(f, 0))
            .flat_map(|f| (1..=2).map(move |d| (f, kr + d * fwd(c)))).filter(|&(f, r)| on(f, r)).map(|(f, r)| name(f, r)).collect();
        o.side("king.shield.breached", c, !missing.is_empty(), sq, why);

        let z = zone(g, c);
        let att: Vec<_> = g.pieces(c.other()).into_iter().filter(|(s, _)| g.attacks(*s).intersects(z)).map(|(s, _)| s).collect();
        let def: Vec<_> = g.pieces(c).into_iter().filter(|(s, _)| g.attacks(*s).intersects(z) || z.contains(*s)).map(|(s, _)| s).collect();
        let d = att.len() as i32 - def.len() as i32;
        let hit = match d { d if d < 0 => 0, 0 => 1, 1 => 2, 2 => 3, 3 => 4, _ => 5 };
        let why = format!("{} attackers ({}) against {} defenders ({})", att.len(), names(att.clone()).join(" "),
                          def.len(), names(def.clone()).join(" "));
        let _ = Role::King;
        o.one_hot("king.attack", Some(c), &["more_defenders", "equal", "attackers_plus1", "attackers_plus2", "attackers_plus3", "attackers_plus4"],
                  hit as usize, names(att.into_iter().chain(def)), &why);
    }
}
