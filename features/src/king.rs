//! 3. King safety: where the king stands, its pawn shield, attackers against defenders.

use shakmaty::{Bitboard, Color};

use crate::Out;
use crate::geo::{Geo, fr, fwd, name, names, rel, title};
use crate::pawns::SECTORS;
use crate::space::forces;

/// The king's square and the squares next to it.
pub fn zone(g: &Geo, c: Color) -> Bitboard {
    match g.king(c) { Some(k) => shakmaty::attacks::king_attacks(k).with(k), None => Bitboard::EMPTY }
}

/// The castled box the king stands in: g/h files (kingside) or a–c files (queenside), on its
/// first two ranks.
fn castled(g: &Geo, c: Color) -> Option<bool> {
    let (f, r) = fr(g.king(c)?);
    if rel(c, r) > 1 { return None; }
    match f { 6 | 7 => Some(true), 0..=2 => Some(false), _ => None }
}

pub fn run(g: &Geo, o: &mut Out) {
    for c in Color::ALL {
        let Some(k) = g.king(c) else { continue };
        let (kf, _) = fr(k);
        let side = castled(g, c);
        let hit = match side { Some(true) => 0, Some(false) => 1, None => 2 };
        o.one_hot("king.castled", Some(c), &["kingside", "queenside", "uncastled"], hit, vec![k.to_string()],
                  &format!("{} king on {k}", title(c)));

        // the shield: the two outer files must keep an own pawn, and the second rank in front of
        // the king at least two of its three
        let (files, row): ([i32; 2], [i32; 3]) = match side {
            Some(true) => ([6, 7], [5, 6, 7]),
            Some(false) => ([1, 0], [0, 1, 2]),
            None => {
                o.side("king.shield.breached", c, false, vec![], format!("the {} king is not castled", title(c).to_lowercase()));
                continue;
            }
        };
        let fl = |f: i32| (b'a' + f as u8) as char;
        let r2 = if c.is_white() { 1 } else { 6 };
        let bare: Vec<i32> = files.into_iter().filter(|&f| !(0..8).any(|r| g.pawn(f, r, c))).collect();
        let home: Vec<i32> = row.into_iter().filter(|&f| g.pawn(f, r2, c)).collect();
        let mut why = vec![];
        if !bare.is_empty() { why.push(format!("no own pawn on the {}", bare.iter().map(|&f| format!("{}-file", fl(f))).collect::<Vec<_>>().join(" or "))); }
        if home.len() <= 1 { why.push(format!("{} of {} still on the {} rank", home.len(),
            row.iter().map(|&f| name(f, r2)).collect::<Vec<_>>().join(" "), if c.is_white() { "second" } else { "seventh" })); }
        let on = !why.is_empty();
        let sq: Vec<String> = row.iter().map(|&f| name(f, r2)).chain(bare.iter().map(|&f| name(f, r2 + fwd(c)))).collect();
        o.side("king.shield.breached", c, on, sq,
               if on { why.join("; ") } else { format!("own pawns on the {}- and {}-files, and {} of {} still home", fl(files[0]), fl(files[1]), home.len(), row.iter().map(|&f| name(f, r2)).collect::<Vec<_>>().join(", ")) });

        // attackers and defenders on the king's wing
        let wing = SECTORS.iter().find(|(_, fs)| fs.contains(&kf)).unwrap().1.clone();
        let z = zone(g, c);
        let att = forces(g, c.other(), &wing, z, true);
        let def = forces(g, c, &wing, z.without(k), false);
        let d = att.len() as i32 - def.len() as i32;
        let hit = match d { d if d < 0 => 0, 0 => 1, 1 => 2, 2 => 3, 3 => 4, _ => 5 };
        let why = format!("{} attackers ({}) against {} defenders ({})", att.len(), names(att.clone()).join(" "),
                          def.len(), names(def.clone()).join(" "));
        o.one_hot("king.attack", Some(c), &["more_defenders", "equal", "attackers_plus1", "attackers_plus2", "attackers_plus3", "attackers_plus4"],
                  hit as usize, names(att.into_iter().chain(def)), &why);
    }
}
