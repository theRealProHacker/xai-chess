//! 1. Material: balance, unlike material, bishop pair, unlike bishops.

use shakmaty::{Color, Role};

use crate::Out;
use crate::geo::{Geo, names, title, val};

fn total(g: &Geo, c: Color) -> i32 {
    [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen].iter().map(|&r| val(r) * g.count(c, r) as i32).sum()
}

/// Counts of (knights, bishops, rooks, queens) left after cancelling identical pieces.
pub type Set = [i32; 4];

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

/// The named trades as (id, what the side has, what it faces), counts of (N, B, R, Q).
const TRADES: [(&str, Set, Set); 16] = [
    ("bishop_vs_knight", [0, 1, 0, 0], [1, 0, 0, 0]),
    ("exchange", [0, 0, 1, 0], [1, 0, 0, 0]),
    ("exchange", [0, 0, 1, 0], [0, 1, 0, 0]),
    ("two_minors_vs_rook", [2, 0, 0, 0], [0, 0, 1, 0]),
    ("two_minors_vs_rook", [1, 1, 0, 0], [0, 0, 1, 0]),
    ("two_minors_vs_rook", [0, 2, 0, 0], [0, 0, 1, 0]),
    ("two_knights_vs_bishop", [2, 0, 0, 0], [0, 1, 0, 0]),
    ("two_rooks_vs_queen", [0, 0, 2, 0], [0, 0, 0, 1]),
    ("three_minors_vs_queen", [3, 0, 0, 0], [0, 0, 0, 1]),
    ("three_minors_vs_queen", [2, 1, 0, 0], [0, 0, 0, 1]),
    ("three_minors_vs_queen", [1, 2, 0, 0], [0, 0, 0, 1]),
    ("three_minors_vs_queen", [0, 3, 0, 0], [0, 0, 0, 1]),
    ("queen_vs_rook_minor", [0, 0, 0, 1], [1, 0, 1, 0]),
    ("queen_vs_rook_minor", [0, 0, 0, 1], [0, 1, 1, 0]),
    ("piece_vs_pawns", [1, 0, 0, 0], [0, 0, 0, 0]),
    ("piece_vs_pawns", [0, 1, 0, 0], [0, 0, 0, 0]),
];

pub const KINDS: [&str; 8] = ["bishop_vs_knight", "exchange", "two_minors_vs_rook", "two_knights_vs_bishop",
    "two_rooks_vs_queen", "three_minors_vs_queen", "queen_vs_rook_minor", "piece_vs_pawns"];

fn sub(a: Set, b: Set) -> Option<Set> {
    let mut o = a;
    for i in 0..4 { o[i] -= b[i]; if o[i] < 0 { return None; } }
    Some(o)
}

/// A trade in the split: (white has it, id, what that side gives, what it faces), counts of (N, B, R, Q).
pub type Trade = (bool, &'static str, Set, Set);

/// Split the two armies into identical pairs and named trades, every piece used once. Among the
/// splits, keep the one with the most identical pairs, then the fewest trades. Each trade is
/// (white has it, id). `lone[c]` allows a lone minor piece against pawns for that side.
fn split(w: Set, b: Set, lone: [bool; 2]) -> Option<(usize, Vec<Trade>)> {
    if w == [0; 4] && b == [0; 4] { return Some((0, vec![])); }
    let mut best: Option<(usize, Vec<Trade>)> = None;
    let mut keep = |cand: Option<(usize, Vec<Trade>)>| {
        if let Some(c) = cand {
            if best.as_ref().is_none_or(|b| (c.0, std::cmp::Reverse(c.1.len())) > (b.0, std::cmp::Reverse(b.1.len()))) { best = Some(c); }
        }
    };
    // the first piece type still on the board: either it pairs off with its like, or it goes into a trade
    let i = (0..4).find(|&i| w[i] > 0 || b[i] > 0).unwrap();
    if w[i] > 0 && b[i] > 0 {
        let mut one = [0; 4]; one[i] = 1;
        keep(split(sub(w, one).unwrap(), sub(b, one).unwrap(), lone).map(|(n, t)| (n + 1, t)));
    }
    for (id, has, against) in TRADES {
        for white in [true, false] {
            if id == "piece_vs_pawns" && !lone[if white { 0 } else { 1 }] { continue; }
            let (mine, theirs) = if white { (w, b) } else { (b, w) };
            // every split is found once: the trade must use that first piece type
            if has[i] == 0 && against[i] == 0 { continue; }
            let (Some(m), Some(t)) = (sub(mine, has), sub(theirs, against)) else { continue };
            let (nw, nb) = if white { (m, t) } else { (t, m) };
            keep(split(nw, nb, lone).map(|(n, mut ts)| { ts.push((white, id, has, against)); (n, ts) }));
        }
    }
    best
}

fn phrase(id: &str) -> &'static str {
    match id {
        "bishop_vs_knight" => "bishop against knight",
        "exchange" => "rook against a minor piece",
        "two_minors_vs_rook" => "two minor pieces against a rook",
        "two_knights_vs_bishop" => "two knights against a bishop",
        "two_rooks_vs_queen" => "two rooks against the queen",
        "three_minors_vs_queen" => "three minor pieces against the queen",
        "queen_vs_rook_minor" => "queen against rook and minor piece",
        _ => "a minor piece against pawns",
    }
}

fn count_set(g: &Geo, c: Color) -> Set {
    [Role::Knight, Role::Bishop, Role::Rook, Role::Queen].map(|r| g.count(c, r) as i32)
}

/// The trades the armies split into; None when no split exists (the imbalance is "other").
pub fn trades(g: &Geo) -> Option<Vec<Trade>> {
    // a lone minor piece for pawns: the other side has two or more pawns more
    let pawns = |c| g.count(c, Role::Pawn) as i32;
    let lone = [pawns(Color::Black) - pawns(Color::White) >= 2, pawns(Color::White) - pawns(Color::Black) >= 2];
    let (w, b) = residue(g);
    if w == [0; 4] && b == [0; 4] { Some(vec![]) } else { split(count_set(g, Color::White), count_set(g, Color::Black), lone).map(|x| x.1) }
}

pub(crate) fn run(g: &Geo, o: &mut Out) {
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
    let trades = trades(g);
    for c in Color::ALL {
        let mine: Vec<&str> = trades.iter().flatten().filter(|t| t.0 == c.is_white()).map(|t| t.1).collect();
        let why = format!("{}: {}", title(c), mine.iter().map(|id| phrase(id)).collect::<Vec<_>>().join(", "));
        for id in KINDS {
            let on = mine.contains(&id);
            o.side(format!("material.imbalance.{id}"), c, on, vec![], if on { why.clone() } else { String::new() });
        }
    }
    let other = trades.is_none();
    o.global("material.imbalance.other", other, vec![],
             if other { format!("White keeps {} against {}", words(w), words(b)) } else { String::new() });

    // material off the board: what each side has lost from its starting set
    for c in Color::ALL {
        let gone = |r: Role, start: usize| start.saturating_sub(g.count(c, r));
        let (q, r, m) = (gone(Role::Queen, 1), gone(Role::Rook, 2), gone(Role::Knight, 2) + gone(Role::Bishop, 2));
        let why = format!("{} has lost {} queen, {} rook{} and {} minor piece{}", title(c), q, r, if r == 1 { "" } else { "s" }, m, if m == 1 { "" } else { "s" });
        o.side("material.traded.queen", c, q == 1, vec![], why.clone());
        o.one_hot("material.traded", Some(c), &["one_rook", "two_rooks"], r.wrapping_sub(1), vec![], &why);
        o.one_hot("material.traded", Some(c), &["one_minor", "two_minors", "three_minors", "four_minors"], m.wrapping_sub(1), vec![], &why);
    }

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
