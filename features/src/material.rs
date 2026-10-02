//! 1. Material: balance, unlike material, bishop pair, unlike bishops.

use shakmaty::{Color, Role};

use crate::Out;
use crate::geo::{Geo, names, title, val};

fn total(g: &Geo, c: Color) -> i32 {
    [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen].iter().map(|&r| val(r) * g.count(c, r) as i32).sum()
}

/// Counts of (knights, bishops, rooks, queens) left after cancelling identical pieces.
type Set = [i32; 4];

fn residue(g: &Geo) -> (Set, Set) {
    let roles = [Role::Knight, Role::Bishop, Role::Rook, Role::Queen];
    let (mut w, mut b) = ([0; 4], [0; 4]);
    for (i, &r) in roles.iter().enumerate() {
        let (x, y) = (g.count(Color::White, r) as i32, g.count(Color::Black, r) as i32);
        let m = x.min(y);
        w[i] = x - m;
        b[i] = y - m;
    }
    (w, b)
}

fn words(s: Set) -> String {
    let names = ["knight", "bishop", "rook", "queen"];
    let parts: Vec<String> = s.iter().zip(names).filter(|(n, _)| **n > 0)
        .map(|(n, w)| if *n == 1 { w.to_string() } else { format!("{n} {w}s") }).collect();
    if parts.is_empty() { "nothing".into() } else { parts.join(" and ") }
}

/// Imbalance types as (id, has, against): `has` is the side the feature is about.
fn kind(a: Set, b: Set) -> Option<&'static str> {
    let [an, ab, ar, aq] = a;
    let [bn, bb, br, bq] = b;
    let (am, bm) = (an + ab, bn + bb);
    Some(match () {
        _ if a == [0, 1, 0, 0] && b == [1, 0, 0, 0] => "bishop_vs_knight",
        _ if ar == 1 && aq == 0 && am == 0 && b[2..] == [0, 0] && bm == 1 => "exchange",
        _ if am == 2 && ar == 0 && aq == 0 && b == [0, 0, 1, 0] => "two_minors_vs_rook",
        _ if a == [2, 0, 0, 0] && b == [0, 1, 0, 0] => "two_knights_vs_bishop",
        _ if a == [0, 0, 2, 0] && b == [0, 0, 0, 1] => "two_rooks_vs_queen",
        _ if am == 3 && ar == 0 && aq == 0 && b == [0, 0, 0, 1] => "three_minors_vs_queen",
        _ if a == [0, 0, 0, 1] && br == 1 && bm == 1 && bq == 0 => "queen_vs_rook_minor",
        _ => return None,
    })
}

const KINDS: [&str; 8] = ["bishop_vs_knight", "exchange", "two_minors_vs_rook", "two_knights_vs_bishop",
    "two_rooks_vs_queen", "three_minors_vs_queen", "queen_vs_rook_minor", "piece_vs_pawns"];

pub fn run(g: &Geo, o: &mut Out) {
    let (tw, tb) = (total(g, Color::White), total(g, Color::Black));
    o.global("material.balance.equal", tw == tb, vec![], format!("{tw} against {tb}"));
    for c in Color::ALL {
        let d = if c.is_white() { tw - tb } else { tb - tw };
        let hit = match d { 1 => Some(0), 2 => Some(1), d if d >= 3 => Some(2), _ => None };
        for (i, b) in ["up1", "up2", "up3plus"].iter().enumerate() {
            let on = hit == Some(i);
            o.side(format!("material.balance.{b}"), c, on, vec![],
                   if on { format!("{} {tw} against {tb}", title(c)) } else { String::new() });
        }
    }

    let (w, b) = residue(g);
    let pawns = |c| g.count(c, Role::Pawn) as i32;
    for c in Color::ALL {
        let (mine, theirs) = if c.is_white() { (w, b) } else { (b, w) };
        let mut k = kind(mine, theirs);
        // A lone minor piece for pawns: the other side has nothing extra but two or more pawns.
        if k.is_none() && mine[0] + mine[1] == 1 && mine[2..] == [0, 0] && theirs == [0; 4]
            && pawns(c.other()) - pawns(c) >= 2 {
            k = Some("piece_vs_pawns");
        }
        let why = format!("{} keeps {} against {}", title(c), words(mine), words(theirs));
        for id in KINDS {
            let on = k == Some(id);
            o.side(format!("material.imbalance.{id}"), c, on, vec![], if on { why.clone() } else { String::new() });
        }
    }
    let other = w != b && kind(w, b).is_none() && kind(b, w).is_none() && w != [0; 4] && b != [0; 4];
    o.global("material.imbalance.other", other, vec![],
             if other { format!("White keeps {} against {}", words(w), words(b)) } else { String::new() });


    for c in Color::ALL {
        let bs: Vec<_> = g.b.by_piece(shakmaty::Piece { color: c, role: Role::Bishop }).into_iter().collect();
        let pair = |bs: &[shakmaty::Square]| bs.iter().any(|s| s.is_light()) && bs.iter().any(|s| s.is_dark());
        let theirs: Vec<_> = g.b.by_piece(shakmaty::Piece { color: c.other(), role: Role::Bishop }).into_iter().collect();
        let on = pair(&bs) && !pair(&theirs);
        o.side("material.bishop_pair", c, on, if on { names(bs.clone()) } else { vec![] },
               if on { format!("{} has both bishops, {} does not", title(c), title(c.other()).to_lowercase()) }
               else if pair(&bs) { "both sides have the pair".into() } else { String::new() });
    }
    let wb: Vec<_> = g.b.by_piece(shakmaty::Piece { color: Color::White, role: Role::Bishop }).into_iter().collect();
    let bb: Vec<_> = g.b.by_piece(shakmaty::Piece { color: Color::Black, role: Role::Bishop }).into_iter().collect();
    let ocb = wb.len() == 1 && bb.len() == 1 && wb[0].is_light() != bb[0].is_light();
    o.global("material.opposite_bishops", ocb, if ocb { names([wb[0], bb[0]]) } else { vec![] },
             if ocb { format!("bishops on {} and {}", wb[0], bb[0]) }
             else if wb.len() == 1 && bb.len() == 1 { "the bishops share a colour".into() }
             else { format!("White has {}, Black {} bishops", wb.len(), bb.len()) });
}
