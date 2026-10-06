//! 4. Quality of pieces: out of play, the per-piece strengths and weaknesses the journey lists,
//! development, and two coordination proxies.

use shakmaty::{Bitboard, Color, Piece, Role, Square};

use crate::Out;
use crate::context::Phase;
use crate::geo::{Geo, fr, fwd, name, names, on, rel, sq, title, val};
use crate::pawns::{backward, passed};

fn home(c: Color, s: Square, role: Role) -> bool {
    let (f, r) = fr(s);
    rel(c, r) == 0 && match role {
        Role::Knight => f == 1 || f == 6,
        Role::Bishop => f == 2 || f == 5,
        Role::Rook => f == 0 || f == 7,
        Role::Queen => f == 3,
        _ => false,
    }
}

/// A move to `to` loses nothing at once: no enemy pawn or cheaper piece hits the square, and if
/// anything hits it, an own piece guards it.
pub fn safe_square(g: &Geo, c: Color, role: Role, from: Square, to: Square) -> bool {
    let (f, r) = fr(to);
    if g.pawn_attacked(f, r, c.other()) { return false; }
    let occ = g.b.occupied().without(from);
    let enemy = g.b.attacks_to(to, c.other(), occ);
    if enemy.into_iter().any(|a| g.b.role_at(a).is_some_and(|ar| val(ar) < val(role))) { return false; }
    enemy.is_empty() || g.b.attacks_to(to, c, occ).without(from).any()
}

fn line_lengths(g: &Geo, s: Square) -> [(usize, usize); 2] {
    // squares reached along a1-h8 and along h1-a8, and how many of them lie past the middle
    let (f, r) = fr(s);
    let c = g.b.color_at(s).unwrap();
    let mut out = [(0, 0); 2];
    for (i, dirs) in [[(1, 1), (-1, -1)], [(1, -1), (-1, 1)]].iter().enumerate() {
        for (df, dr) in dirs {
            let (mut ff, mut rr) = (f + df, r + dr);
            while on(ff, rr) {
                out[i].0 += 1;
                if rel(c, rr) >= 4 { out[i].1 += 1; }
                if g.piece(ff, rr).is_some() { break; }
                ff += df; rr += dr;
            }
        }
    }
    out
}

/// Up its file the rook reaches a square on the opponent's second or back rank that no enemy
/// piece or pawn attacks.
fn entry(g: &Geo, c: Color, s: Square) -> bool {
    let (f, r) = fr(s);
    let mut rr = r + fwd(c);
    while on(f, rr) {
        if rel(c, rr) >= 6 && g.control(f, rr)[if c.is_white() { 1 } else { 0 }] == 0 { return true; }
        if g.piece(f, rr).is_some() { return false; }
        rr += fwd(c);
    }
    false
}

/// The first pawn up the rook's file, past own pieces, is its own, has left its starting rank and
/// can step forward: the rook stands behind a lever.
fn rook_behind_lever(g: &Geo, c: Color, s: Square) -> bool {
    let (f, r) = fr(s);
    let mut rr = r + fwd(c);
    while on(f, rr) {
        if let Some(p) = g.piece(f, rr) {
            if p.role == Role::Pawn {
                return p.color == c && rel(c, rr) >= 2 && on(f, rr + fwd(c)) && g.piece(f, rr + fwd(c)).is_none();
            }
            if p.color != c || p.role == Role::King { return false; }
        }
        rr += fwd(c);
    }
    false
}

pub(crate) fn run(g: &Geo, phase: Phase, o: &mut Out) {
    let centre = Bitboard(0x3c3c_3c3c_u64 << 16);
    for c in Color::ALL {
        let them = c.other();
        let ek = g.king(them).map(|k| shakmaty::attacks::king_attacks(k).with(k)).unwrap_or(Bitboard::EMPTY);
        let enemy = g.b.by_color(them);

        // out of play. Own pieces other than pawns and the king will step aside, so lines run
        // through them.
        let mut oop = vec![];
        let solid = g.b.pawns() | g.b.kings() | g.b.by_color(them);
        for (s, role) in g.pieces(c) {
            // in the opening a piece on its back rank is undeveloped, which the development family covers
            if phase == Phase::Opening && rel(c, fr(s).1) == 0 { continue; }
            let a = shakmaty::attacks::attacks(s, Piece { color: c, role }, solid);
            if role == Role::Rook && rook_behind_lever(g, c, s) { continue; }
            let safe = (a & !g.b.by_color(c)).into_iter().filter(|&t| safe_square(g, c, role, s, t)).count();
            if safe <= 2 && !a.intersects(centre) && !a.intersects(ek) && !a.intersects(enemy) { oop.push((s, safe)); }
        }
        let why = oop.iter().map(|(s, n)| format!("{} on {s}: {n} safe move{}, reaches nothing that matters", format!("{:?}", g.b.role_at(*s).unwrap()).to_lowercase(), if *n == 1 { "" } else { "s" })).collect::<Vec<_>>().join("; ");
        o.side("pieces.out_of_play", c, !oop.is_empty(), names(oop.iter().map(|x| x.0)), why);

        // knight outposts
        let mut posts = vec![];
        for s in g.b.by_piece(Piece { color: c, role: Role::Knight }) {
            let (f, r) = fr(s);
            if !(3..=5).contains(&rel(c, r)) || !g.pawn_attacked(f, r, c) { continue; }
            // an enemy pawn on a neighbouring file, ahead of the knight, with no pawn in its way
            // to the square from which it would hit the knight
            let kickable = g.pawns(them).iter().any(|&(pf, pr)| (pf - f).abs() == 1 && rel(c, pr) > rel(c, r)
                && (1..(rel(c, pr) - rel(c, r))).all(|d| g.b.pawns().is_disjoint(Bitboard::from_square(sq(pf, pr - d * fwd(c))))));
            if !kickable { posts.push(s); }
        }
        o.side("pieces.knight_outpost", c, !posts.is_empty(), names(posts.clone()),
               if posts.is_empty() { String::new() } else { format!("knight on {}: guarded by a pawn, no enemy pawn can come to attack it", names(posts).join(", ")) });

        // bishops
        let mut open = vec![];
        let mut bad = vec![];
        for s in g.b.by_piece(Piece { color: c, role: Role::Bishop }) {
            let l = line_lengths(g, s);
            if l.iter().any(|&(n, far)| n >= 5 && far >= 3) { open.push(s); }
            // own centre pawns (c- to f-file) on its colour, blocked by an enemy pawn, ahead of it
            let fixed: Vec<Square> = g.pawns(c).into_iter().filter(|&(f, r)| sq(f, r).is_light() == s.is_light()
                && (2..=5).contains(&f) && rel(c, r) > rel(c, fr(s).1)
                && on(f, r + fwd(c)) && g.pawn(f, r + fwd(c), them)).map(|(f, r)| sq(f, r)).collect();
            if fixed.len() >= 2 { bad.push((s, fixed)); }
        }
        o.side("pieces.bishop_diagonal", c, !open.is_empty(), names(open.clone()),
               if open.is_empty() { String::new() } else { format!("bishop on {} sweeps a long diagonal into the enemy half", names(open).join(", ")) });
        o.side("pieces.bad_bishop", c, !bad.is_empty(), bad.iter().flat_map(|(s, p)| std::iter::once(*s).chain(p.iter().copied())).map(|s| s.to_string()).collect(),
               bad.iter().map(|(s, p)| format!("bishop on {s}; own pawns fixed on its colour: {}", names(p.clone()).join(" "))).collect::<Vec<_>>().join("; "));

        // rooks
        let rooks: Vec<Square> = g.b.by_piece(Piece { color: c, role: Role::Rook }).into_iter().collect();
        let file_pawns = |f: i32, who: Color| (0..8).filter(|&r| g.pawn(f, r, who)).count();
        let (mut open_f, mut half_f, mut seventh, mut behind, mut blocked) = (vec![], vec![], vec![], vec![], vec![]);
        for &s in &rooks {
            let (f, r) = fr(s);
            let (mine, theirs) = (file_pawns(f, c), file_pawns(f, them));
            // an enemy pawn guarded by a pawn closes the file
            let solid = (0..8).any(|rr| g.pawn(f, rr, them) && g.pawn_attacked(f, rr, them));
            if mine == 0 && theirs == 0 && entry(g, c, s) { open_f.push(s); } else if mine == 0 && !solid { half_f.push(s); }
            if rel(c, r) == 6 { seventh.push(s); }
            for rr in 0..8 {
                if !g.pawn(f, rr, c) || rel(c, rr) <= rel(c, r) { continue; }
                if passed(g, c, f, rr) { behind.push(s); }
                if g.pawn(f, rr + fwd(c), them) || backward(g, c, f, rr) { blocked.push(s); }
            }
        }
        behind.dedup(); blocked.dedup();
        let at = |v: &Vec<Square>, what: &str| if v.is_empty() { String::new() } else { format!("rook on {} {what}", names(v.clone()).join(", ")) };
        o.global_or_side("pieces.rook_file.open", c, &open_f, at(&open_f, "on an open file"));
        o.global_or_side("pieces.rook_file.half_open", c, &half_f, at(&half_f, "on a half-open file"));
        o.global_or_side("pieces.rook_seventh", c, &seventh, at(&seventh, "on the seventh"));
        o.global_or_side("pieces.rook_behind_passer", c, &behind, at(&behind, "behind its passed pawn"));
        o.global_or_side("pieces.rook_blocked", c, &blocked, at(&blocked, "behind a blocked or backward own pawn"));

        // queen
        let mut good = vec![];
        for s in g.b.by_piece(Piece { color: c, role: Role::Queen }) {
            let far = g.attacks(s).into_iter().filter(|&t| rel(c, fr(t).1) >= 4).count();
            let threat = g.attackers(s, them).into_iter().any(|a| g.b.role_at(a).is_some_and(|r| r != Role::Queen && r != Role::King));
            if far >= 5 && !threat { good.push((s, far)); }
        }
        o.side("pieces.queen.active_safe", c, !good.is_empty(), names(good.iter().map(|x| x.0)),
               good.iter().map(|(s, n)| format!("queen on {s} hits {n} squares in the enemy half and no cheaper piece attacks it")).collect::<Vec<_>>().join("; "));

        // coordination proxies
        let connected = rooks.len() >= 2 && rooks.iter().enumerate().any(|(i, &a)| rooks[i + 1..].iter().any(|&b|
            shakmaty::attacks::rook_attacks(a, g.b.occupied()).contains(b)));
        o.side("pieces.coordination.rooks_connected", c, connected, if connected { names(rooks.clone()) } else { vec![] },
               if connected { "the rooks guard each other".into() } else { String::new() });
        let loose: Vec<Square> = g.pieces(c).into_iter().map(|x| x.0).filter(|&s| g.attackers(s, c).is_empty()).collect();
        o.side("pieces.coordination.loose_piece", c, !loose.is_empty(), names(loose.clone()),
               if loose.is_empty() { String::new() } else { format!("nothing guards {}", names(loose).join(", ")) });
    }

    // development, opening only
    let dev = |c: Color| {
        let mut n = 0;
        for role in [Role::Knight, Role::Bishop] {
            for s in g.b.by_piece(Piece { color: c, role }) { if !home(c, s, role) { n += 1; } }
        }
        let castled = g.king(c).is_some_and(|k| { let (f, r) = fr(k); rel(c, r) == 0 && (f == 6 || f == 2 || f == 1) });
        n + castled as i32
    };
    let (dw, db) = (dev(Color::White), dev(Color::Black));
    let opening = phase == Phase::Opening;
    o.global("pieces.development.equal", opening && dw == db, vec![],
             if opening { format!("White has {dw} developed, Black {db}") } else { "not the opening".into() });
    for c in Color::ALL {
        let lead = if c.is_white() { dw - db } else { db - dw };
        let hit = if !opening { None } else { match lead { 1 => Some(0), 2 => Some(1), l if l >= 3 => Some(2), _ => None } };
        for (i, b) in ["plus1", "plus2", "plus3plus"].iter().enumerate() {
            let on = hit == Some(i);
            o.side(format!("pieces.development.{b}"), c, on, vec![],
                   if on { format!("{} {} developed against {}", title(c), dw.max(db), dw.min(db)) } else { String::new() });
        }
    }
    let _ = name;
}

impl Out {
    fn global_or_side(&mut self, id: &str, c: Color, v: &[Square], why: String) {
        self.side(id, c, !v.is_empty(), names(v.iter().copied()), why);
    }
}
