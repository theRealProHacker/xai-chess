//! 2. Local superiority and 6. Space, per sector.

use std::ops::RangeInclusive;

use shakmaty::{Bitboard, Color, Role, Square};

use crate::Out;
use crate::geo::{Geo, fr, name, names, rel, title};
use crate::pawns::SECTORS;

/// A rook still on its back rank with an own pawn on its file cannot reach the fight yet.
pub fn shut_in(g: &Geo, c: Color, s: Square) -> bool {
    let (f, r) = fr(s);
    g.b.role_at(s) == Some(Role::Rook) && rel(c, r) == 0 && (0..8).any(|rr| g.pawn(f, rr, c))
}

/// `c`'s knights, bishops, rooks and queens that take part on `files`: they stand on those files
/// (a shut-in rook does not count), or hit a square of `zone`, or (with `far`) hit three of their
/// squares in the opponent's half.
pub fn forces(g: &Geo, c: Color, files: &RangeInclusive<i32>, zone: Bitboard, far: bool) -> Vec<Square> {
    let mut v: Vec<Square> = g.pieces(c).into_iter().map(|x| x.0).filter(|&s| {
        let a = g.attacks(s);
        (files.contains(&fr(s).0) && !shut_in(g, c, s)) || a.intersects(zone)
            || far && a.into_iter().filter(|&t| { let (f, r) = fr(t); files.contains(&f) && rel(c, r) >= 4 }).count() >= 3
    }).collect();
    v.sort_by_key(|s| s.to_string());
    v
}

/// `c`'s knights, bishops, rooks and queens that stand on `files` (not a minor piece at home, not a
/// shut-in rook), or that attack an enemy unit or three squares there in the opponent's half.
fn bearing(g: &Geo, c: Color, files: &RangeInclusive<i32>) -> Vec<Square> {
    let mut v: Vec<Square> = g.pieces(c).into_iter().map(|x| x.0).filter(|&s| {
        let a: Vec<Square> = g.attacks(s).into_iter().filter(|&t| { let (f, r) = fr(t); files.contains(&f) && rel(c, r) >= 4 }).collect();
        let (f, r) = fr(s);
        let home = rel(c, r) == 0 && matches!(g.b.role_at(s), Some(Role::Knight | Role::Bishop)) && [1, 2, 5, 6].contains(&f);
        (files.contains(&f) && !shut_in(g, c, s) && !home) || a.len() >= 3 || a.iter().any(|&t| g.b.by_color(c.other()).contains(t))
    }).collect();
    v.sort_by_key(|s| s.to_string());
    v
}

pub fn local(g: &Geo, o: &mut Out) {
    for (sector, files) in SECTORS {
        let (fw, fb) = (bearing(g, Color::White, &files), bearing(g, Color::Black, &files));
        let d = fw.len() as i32 - fb.len() as i32;
        let why = format!("{sector}: White {} ({}), Black {} ({})", fw.len(), names(fw.clone()).join(" "), fb.len(), names(fb.clone()).join(" "));
        o.global(format!("local.{sector}.equal"), d == 0, vec![], why.clone());
        for c in Color::ALL {
            let mine = if c.is_white() { d } else { -d };
            let hit = if mine > 0 { (mine - 1).min(2) as usize } else { 99 };
            // both sides' pieces, so the page can mark the defenders too
            let sq = names(fw.iter().chain(&fb).copied());
            o.one_hot(&format!("local.{sector}"), Some(c), &["plus1", "plus2", "plus3plus"], hit, sq, &why);
        }
    }
}

pub fn run(g: &Geo, o: &mut Out) {
    for (sector, files) in SECTORS {
        let held = |c: Color| -> Vec<String> {
            files.clone().flat_map(|f| (0..8).map(move |r| (f, r)))
                .filter(|&(_, r)| rel(c, r) >= 4)
                .filter(|&(f, r)| g.holder(f, r) == Some(c)).map(|(f, r)| name(f, r)).collect()
        };
        let (hw, hb) = (held(Color::White), held(Color::Black));
        for c in Color::ALL {
            let (mine, theirs) = if c.is_white() { (&hw, &hb) } else { (&hb, &hw) };
            let on = mine.len() >= theirs.len() + 2;
            o.side(format!("space.{sector}"), c, on, if on { mine.clone() } else { vec![] },
                   format!("{sector}: {} holds {} squares in the opponent's half, {} holds {}", title(c), mine.len(),
                           title(c.other()), theirs.len()));
        }
    }
}

/// Space (pawn advance): own pawns on the 4th, 5th or 6th rank, against the opponent's.
pub fn pawn_advance(g: &Geo, o: &mut Out) {
    let adv = |c: Color| -> Vec<String> {
        let mut v: Vec<_> = g.pawns(c).into_iter().filter(|&(_, r)| (3..=5).contains(&rel(c, r))).collect();
        v.sort();
        v.into_iter().map(|(f, r)| name(f, r)).collect()
    };
    let (aw, ab) = (adv(Color::White), adv(Color::Black));
    let d = aw.len() as i32 - ab.len() as i32;
    let why = format!("pawns on their own 4th to 6th rank: White {} ({}), Black {} ({})",
                      aw.len(), aw.join(" "), ab.len(), ab.join(" "));
    o.global("space_pawns.equal", d == 0, vec![], why.clone());
    for c in Color::ALL {
        let mine = if c.is_white() { d } else { -d };
        let hit = if mine > 0 { (mine - 1).min(2) as usize } else { 99 };
        // both sides' advanced pawns, so the page can mark the other side's too
        o.one_hot("space_pawns", Some(c), &["plus1", "plus2", "plus3plus"], hit, aw.iter().chain(&ab).cloned().collect(), &why);
    }
}
