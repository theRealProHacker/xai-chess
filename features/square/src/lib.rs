//! Binary features per square: what the piece on it does, and what the square is. The rules are
//! the ones settled in the plan for the per-square list (journey.md, "Features"); `schema` names
//! every bit.
//!
//! Perspective: "own" and "enemy" are the occupant's colour and the other one. Board-only groups
//! (square control, pawn cover, weak/strong, files, shields, the engine lines) fire on empty
//! squares too.

pub mod schema;
pub mod see;

use std::collections::VecDeque;

use features::geo::{Geo, fr, fwd, on, rel, sq};
use features::pawns::{isolated, passed};
use shakmaty::uci::UciMove;
use shakmaty::{Bitboard, Board, Chess, Color, Move, Piece, Position, Role, Square};

use schema::{DIRS, PLIES, offset, pair};

/// The bits that are on, per square (index = `Square as usize`), as indices into the schema.
pub type Bits = Vec<Vec<usize>>;

/// Engine lines in UCI: `best` from the position, `null` from the position with the side to move
/// passing. Without them the tactical groups stay off.
#[derive(Default)]
pub struct Lines<'a> {
    pub best: Option<&'a [String]>,
    pub null: Option<&'a [String]>,
}

struct Out(Bits);

impl Out {
    fn set(&mut self, s: Square, group: &str, idx: &[usize]) {
        let (off, dims) = offset(group);
        let mut i = 0;
        for (d, &x) in dims.iter().zip(idx) {
            debug_assert!(x < d.size(), "{group}: {x} out of {}", d.size());
            i = i * d.size() + x;
        }
        let v = &mut self.0[s as usize];
        if !v.contains(&(off + i)) { v.push(off + i); }
    }
}

fn ti(r: Role) -> usize { r as usize - 1 }
/// White 0, Black 1.
fn ci(c: Color) -> usize { if c.is_white() { 0 } else { 1 } }
fn sqs(b: Bitboard) -> impl Iterator<Item = Square> { b.into_iter() }

/// The null-move position, if the side to move may pass (not in check).
pub fn null_position(pos: &Chess) -> Option<Chess> {
    if pos.is_check() { return None; }
    pos.clone().swap_turn().ok()
}

pub fn extract(pos: &Chess, lines: &Lines) -> Bits {
    let g = Geo { b: pos.board().clone() };
    let mut o = Out(vec![vec![]; 64]);
    tactics(pos, lines, &mut o);
    for s in Square::ALL {
        board_square(&g, s, &mut o);
        if let Some(p) = g.b.piece_at(s) { piece(&g, pos, s, p, &mut o); }
    }
    material(&g, &mut o);
    for v in &mut o.0 { v.sort_unstable(); }
    o.0
}

// ------------------------------------------------------------------------------- tactics

fn play(pos: &Chess, line: &[String]) -> Vec<(Move, Chess)> {
    let mut out = vec![];
    let mut p = pos.clone();
    for u in line.iter().take(PLIES) {
        let Some(m) = u.parse::<UciMove>().ok().and_then(|u| u.to_move(&p).ok()) else { break };
        p = p.clone().play(m).unwrap();
        out.push((m, p.clone()));
    }
    out
}

fn balance(b: &Board, c: Color) -> i32 {
    let total = |c: Color| [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen].iter()
        .map(|&r| features::geo::val(r) * (b.by_piece(Piece { color: c, role: r })).count() as i32).sum::<i32>();
    total(c) - total(c.other())
}

/// Mate and material wins along `line` from `pos`, marked on every square a move of it touches.
fn line_bits(pos: &Chess, line: &[String], mate: &str, win: &str, o: &mut Out) {
    let played = play(pos, line);
    let Some((_, last)) = played.last() else { return };
    let c = pos.turn();
    let mut groups = vec![];
    if last.is_checkmate() { groups.push(mate); }
    if balance(last.board(), c) > balance(pos.board(), c) { groups.push(win); }
    for g in groups {
        for (k, (m, _)) in played.iter().enumerate() {
            for s in m.from().into_iter().chain([m.to()]) { o.set(s, g, &[k]); }
        }
    }
}

fn tactics(pos: &Chess, lines: &Lines, o: &mut Out) {
    let b = pos.board();
    for s in sqs(b.occupied()) {
        xray(b, s, s, "xray", o);
        let p = b.piece_at(s).unwrap();
        if b.king_of(p.color.other()).is_some_and(|k| b.attacks_from(s).contains(k)) { o.set(s, "check", &[0]); }
    }
    if let Some(line) = lines.best { line_bits(pos, line, "mate", "material_win", o); }
    if let (Some(line), Some(null)) = (lines.null, null_position(pos)) {
        line_bits(&null, line, "threat.mate", "threat.material_win", o);
        if let Some((m, after)) = play(&null, line).into_iter().next() {
            let from = m.from().unwrap();
            o.set(from, "threat.move", &[m.to() as usize]);
            if after.is_check() { o.set(from, "threat.check", &[0]); }
            xray(after.board(), m.to(), from, "threat.xray", o);
        }
    }
}

/// The slider on `s`: along each of its rays, the first two pieces and their distances. Marked on `at`.
fn xray(b: &Board, s: Square, at: Square, group: &str, o: &mut Out) {
    let role = b.role_at(s).unwrap();
    let (f, r) = fr(s);
    for (di, &(df, dr)) in DIRS.iter().enumerate() {
        let diag = df != 0 && dr != 0;
        let slides = role == Role::Queen || (role == Role::Rook && !diag) || (role == Role::Bishop && diag);
        if !slides { continue; }
        let mut hits = vec![];
        let (mut ff, mut rr, mut d) = (f + df, r + dr, 1);
        while on(ff, rr) && hits.len() < 2 {
            if let Some(p) = b.piece_at(sq(ff, rr)) { hits.push((p.role, d)); }
            ff += df; rr += dr; d += 1;
        }
        if let [(r1, d1), (r2, d2)] = hits[..] {
            o.set(at, group, &[ti(r1), ti(r2), di, pair(d1 as usize, (d2 - d1) as usize)]);
        }
    }
}

// ---------------------------------------------------------------------------- material

fn material(g: &Geo, o: &mut Out) {
    let b = &g.b;
    let kinds = features::material::KINDS;
    let roles = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen];
    for c in Color::ALL {
        // up by d: every own piece of a type the side has more of
        let d = balance(b, c);
        if d >= 1 {
            let bucket = (d as usize - 1).min(2);
            for &r in &roles {
                if g.count(c, r) > g.count(c.other(), r) {
                    for s in b.by_piece(Piece { color: c, role: r }) { o.set(s, "material.up", &[bucket]); }
                }
            }
        }
    }
    // trades: both sides' pieces of the types in it; the side giving more, smaller units has its
    // rooks and queens outside the trade as play-coordinating pieces
    let nbrq = [Role::Knight, Role::Bishop, Role::Rook, Role::Queen];
    for (white, id, has, against) in features::material::trades(g).unwrap_or_default() {
        let k = kinds.iter().position(|x| *x == id).unwrap();
        let giver = if white { Color::White } else { Color::Black };
        for (c, set) in [(giver, has), (giver.other(), against)] {
            for (i, &r) in nbrq.iter().enumerate() {
                if set[i] > 0 { for s in b.by_piece(Piece { color: c, role: r }) { o.set(s, "material.trade", &[k]); } }
            }
        }
        let pawns_side = id == "piece_vs_pawns";
        if pawns_side { for s in b.by_piece(Piece { color: giver.other(), role: Role::Pawn }) { o.set(s, "material.trade", &[k]); } }
        let units = |x: features::material::Set| x.iter().sum::<i32>();
        let more = if pawns_side || units(against) > units(has) { Some((giver.other(), against)) }
                   else if units(has) > units(against) { Some((giver, has)) } else { None };
        if let Some((c, set)) = more {
            for (i, r) in [(2, Role::Rook), (3, Role::Queen)] {
                if g.count(c, r) as i32 > set[i] {
                    for s in b.by_piece(Piece { color: c, role: r }) { o.set(s, "material.coordinating", &[0]); }
                }
            }
        }
    }
    // bishop pair and unlike bishops: the other bishop's square
    let bishops = |c: Color| b.by_piece(Piece { color: c, role: Role::Bishop }).into_iter().collect::<Vec<_>>();
    let pair = |v: &[Square]| v.iter().any(|s| s.is_light()) && v.iter().any(|s| s.is_dark());
    for c in Color::ALL {
        let (mine, theirs) = (bishops(c), bishops(c.other()));
        if pair(&mine) && !pair(&theirs) {
            for &s in &mine { for &t in &mine { if s != t && s.is_light() != t.is_light() { o.set(s, "bishop_pair", &[t as usize]); } } }
        }
    }
    let (w, k) = (bishops(Color::White), bishops(Color::Black));
    if w.len() == 1 && k.len() == 1 && w[0].is_light() != k[0].is_light() {
        o.set(w[0], "unlike_bishops", &[k[0] as usize]);
        o.set(k[0], "unlike_bishops", &[w[0] as usize]);
    }
}

// ------------------------------------------------------------------------ the square itself

/// How many pawns of `c` guard the square now.
fn pawn_guards(g: &Geo, s: Square, c: Color) -> usize {
    let (f, r) = fr(s);
    [-1, 1].iter().filter(|&&df| g.pawn(f + df, r - fwd(c), c)).count()
}

/// A pawn of `c` guards the square now, or can walk up a neighbouring file to guard it with no
/// pawn in its way (the outpost rule in features/src/pieces.rs).
pub fn pawn_can_guard(g: &Geo, s: Square, c: Color) -> bool {
    let (f, r) = fr(s);
    let guard = r - fwd(c);
    if !on(f, guard) { return false; }
    [-1, 1].iter().any(|&df| {
        let ff = f + df;
        (0..8).any(|pr| g.pawn(ff, pr, c) && rel(c, pr) <= rel(c, guard)
            && (1..=(rel(c, guard) - rel(c, pr))).all(|d| g.piece(ff, pr + d * fwd(c)).is_none_or(|p| p.role != Role::Pawn)))
    })
}

fn board_square(g: &Geo, s: Square, o: &mut Out) {
    let (f, r) = fr(s);
    o.set(s, "centralisation", &[f.min(7 - f).min(r).min(7 - r) as usize]);

    let [w, b] = g.control(f, r);
    let d = w as i32 - b as i32;
    o.set(s, "square.control", &[match d { 0 => 0, d if d > 0 => d.min(3) as usize, d => 3 + (-d).min(3) as usize }]);

    for (c, now, ever, strong) in [(Color::White, "pawn_cover.white", "pawn_cover.white_ever", "strong.white"),
                                   (Color::Black, "pawn_cover.black", "pawn_cover.black_ever", "strong.black")] {
        let n = pawn_guards(g, s, c);
        o.set(s, now, &[n]);
        if pawn_can_guard(g, s, c) { o.set(s, ever, &[0]); } else { o.set(s, "weak", &[ci(c)]); }
        if n > 0 && !pawn_can_guard(g, s, c.other()) { o.set(s, strong, &[n - 1]); }
    }

    let pawns_on = |c: Color| (0..8).any(|rr| g.pawn(f, rr, c));
    if !pawns_on(Color::White) && !pawns_on(Color::Black) { o.set(s, "open_file", &[f as usize]); }
    if f == r { o.set(s, "long_diagonal", &[0]); }
    if f + r == 7 { o.set(s, "long_diagonal", &[1]); }

    // pawns that stop lines into the square: the first piece along a ray is a pawn
    for (di, &(df, dr)) in DIRS.iter().enumerate() {
        let (mut ff, mut rr) = (f + df, r + dr);
        while on(ff, rr) {
            if let Some(p) = g.piece(ff, rr) {
                if p.role == Role::Pawn { o.set(s, "pawn_shield", &[sq(ff, rr) as usize, di]); }
                break;
            }
            ff += df; rr += dr;
        }
    }
}

// ------------------------------------------------------------------------------ the piece

/// Where a piece on `from` can go on a frozen board with occupancy `occ`: its attacks minus own
/// pieces; a pawn pushes (twice from home) into empty squares and takes enemy pieces.
fn steps(b: &Board, occ: Bitboard, p: Piece, from: Square) -> Bitboard {
    let own = b.by_color(p.color) & occ;
    if p.role != Role::Pawn { return shakmaty::attacks::attacks(from, p, occ) & !own; }
    let (f, r) = fr(from);
    let d = fwd(p.color);
    let mut out = shakmaty::attacks::pawn_attacks(p.color, from) & occ & !own;
    if on(f, r + d) && !occ.contains(sq(f, r + d)) {
        out = out.with(sq(f, r + d));
        if rel(p.color, r) == 1 && !occ.contains(sq(f, r + 2 * d)) { out = out.with(sq(f, r + 2 * d)); }
    }
    out
}

/// Shortest number of moves from `start` to every square on a frozen board, landing only on
/// squares `ok` allows; an enemy piece can be taken but not passed.
fn bfs(b: &Board, occ: Bitboard, p: Piece, start: Square, ok: &dyn Fn(Square) -> bool) -> [u8; 64] {
    let mut dist = [u8::MAX; 64];
    dist[start as usize] = 0;
    let mut q = VecDeque::from([start]);
    while let Some(x) = q.pop_front() {
        if x != start && occ.contains(x) { continue; }
        for y in steps(b, occ, p, x) {
            if dist[y as usize] == u8::MAX && ok(y) {
                dist[y as usize] = dist[x as usize] + 1;
                q.push_back(y);
            }
        }
    }
    dist
}

/// The piece's legal destinations, as if its side were to move.
fn legal(pos: &Chess, s: Square, p: Piece) -> Bitboard {
    let mover = if pos.turn() == p.color { Some(pos.clone()) } else { pos.clone().swap_turn().ok() };
    match mover {
        Some(m) => m.legal_moves().iter().filter(|mv| mv.from() == Some(s))
            .map(|mv| match mv.castling_side() { Some(cs) => cs.king_to(p.color), None => mv.to() }).collect(),
        None => steps(pos.board(), pos.board().occupied(), p, s),
    }
}

fn safe(g: &Geo, p: Piece, from: Square, to: Square) -> bool {
    features::pieces::safe_square(g, p.color, p.role, from, to)
}

fn without(b: &Board, s: Square) -> Board { let mut b = b.clone(); b.discard_piece_at(s); b }

fn piece(g: &Geo, pos: &Chess, s: Square, p: Piece, o: &mut Out) {
    let b = &g.b;
    let (c, them) = (p.color, p.color.other());
    let (f, r) = fr(s);
    let own = b.by_color(c);
    let enemy = b.by_color(them);
    o.set(s, "occupant", &[ci(c) * 6 + ti(p.role)]);

    // stability
    for e in sqs(b.pawns() & enemy) {
        let (ef, er) = fr(e);
        if (ef - f).abs() != 1 || rel(c, er) <= rel(c, r) { continue; }
        let dist = rel(c, er) - rel(c, r);
        if (1..dist).all(|d| g.piece(ef, er - d * fwd(c)).is_none_or(|x| x.role != Role::Pawn)) {
            o.set(s, "stability.pawn_ever", &[e as usize]);
        }
        // directly: hits it now, or after one push into empty squares
        let ep = Piece { color: them, role: Role::Pawn };
        let pushes = steps(b, b.occupied(), ep, e) & !b.occupied();
        if dist == 1 || pushes.into_iter().any(|t| shakmaty::attacks::pawn_attacks(them, t).contains(s)) {
            o.set(s, "stability.pawn_direct", &[e as usize]);
        }
    }
    let worth = |x: Role| if x == Role::King { i32::MAX } else { features::geo::val(x) };
    for a in sqs(enemy & !b.kings()) {
        let ar = b.role_at(a).unwrap();
        if features::geo::val(ar) > worth(p.role) { continue; }
        let ap = Piece { color: them, role: ar };
        let occ = b.occupied().without(a);
        let dist = bfs(b, occ, ap, a, &|t| !occ.contains(t) && safe(g, ap, a, t));
        let best = Square::ALL.iter().filter(|&&t| dist[t as usize] < PLIES as u8 && shakmaty::attacks::attacks(t, ap, occ).contains(s))
            .map(|&t| dist[t as usize]).min();
        if let Some(n) = best { o.set(s, "stability.attacked_in", &[n as usize, ti(ar), a as usize]); }
    }

    // control
    let reach = b.attacks_from(s);
    for t in reach { o.set(s, "control.reach", &[t as usize]); }
    for t in geometry(g, s, p) { o.set(s, "control.geometry", &[t as usize]); }

    // mobility
    for t in legal(pos, s, p) {
        o.set(s, "mobility.abs", &[t as usize]);
        if safe(g, p, s, t) { o.set(s, "mobility.rel", &[t as usize]); }
    }
    let occ = b.occupied().without(s);
    for (group, rel_only) in [("route.abs", false), ("route.rel", true)] {
        let ok = |t: Square| !rel_only || safe(g, p, s, t);
        let firsts: Vec<Square> = steps(b, occ, p, s).into_iter().filter(|&t| ok(t)).collect();
        let dists: Vec<[u8; 64]> = firsts.iter().map(|&m| bfs(b, occ, p, m, &ok)).collect();
        for x in Square::ALL {
            if x == s { continue; }
            let best = dists.iter().map(|d| d[x as usize]).min().unwrap_or(u8::MAX);
            if best == u8::MAX { continue; }
            for (m, d) in firsts.iter().zip(&dists) {
                if d[x as usize] == best { o.set(s, group, &[x as usize, *m as usize]); }
            }
        }
    }
    for t in blockers(g, s, p) {
        let x = b.piece_at(t).unwrap();
        o.set(s, if x.color == c { "blocked_by.own" } else { "blocked_by.enemy" }, &[ti(x.role), t as usize]);
    }

    // files
    let pawns_on = |who: Color| (0..8).any(|rr| g.pawn(f, rr, who));
    if !pawns_on(c) && pawns_on(them) { o.set(s, "half_open_file", &[f as usize]); }

    // role
    let hangs = see::hanging(b, s);
    for t in sqs(reach & own & !b.kings()) {
        o.set(s, "role.defends", &[ti(b.role_at(t).unwrap()), t as usize]);
        if !see::hanging(b, t) && see::hanging(&without(b, s), t) {
            o.set(s, "role.needed_for", &[ti(b.role_at(t).unwrap()), t as usize]);
        }
    }
    for t in sqs(reach & enemy) { o.set(s, "role.attacks", &[ti(b.role_at(t).unwrap()), t as usize]); }
    for e in sqs(enemy) {
        let ep = b.piece_at(e).unwrap();
        for q in steps(b, b.occupied(), ep, e) {
            if reach.contains(q) && !safe(g, ep, e, q) { o.set(s, "role.restricts", &[ti(ep.role), q as usize]); }
        }
    }
    for d in sqs(b.attacks_to(s, c, b.occupied()) & own) {
        o.set(s, "role.protected_by", &[ti(b.role_at(d).unwrap()), d as usize]);
        if p.role != Role::King && !hangs && see::hanging(&without(b, d), s) {
            o.set(s, "role.needs", &[ti(b.role_at(d).unwrap()), d as usize]);
        }
    }
    if (1..=6).contains(&f) { o.set(s, "blocks.line", &[r as usize]); }
    if (1..=6).contains(&r) { o.set(s, "blocks.line", &[8 + f as usize]); }
    if (1..=6).contains(&f) && (1..=6).contains(&r) {
        o.set(s, "blocks.diagonal", &[(f - r + 7) as usize]);
        o.set(s, "blocks.diagonal", &[15 + (f + r) as usize]);
    }
    for pc in Color::ALL {
        if g.pawn(f, r - fwd(pc), pc) { o.set(s, "blocks.pawn", &[sq(f, r - fwd(pc)) as usize]); }
    }

    match p.role {
        Role::Rook => rook(g, s, c, o),
        Role::Pawn => pawn(g, s, c, o),
        _ => {}
    }
}

/// Squares reached counting batteries: a slider sees through an own slider of its kind and onto
/// the square an own pawn in front of it attacks (the rule of `Geo::control`).
fn geometry(g: &Geo, s: Square, p: Piece) -> Bitboard {
    let mut out = g.b.attacks_from(s);
    let (f, r) = fr(s);
    for &(df, dr) in &DIRS {
        let diag = df != 0 && dr != 0;
        let slides = p.role == Role::Queen || (p.role == Role::Rook && !diag) || (p.role == Role::Bishop && diag);
        if !slides { continue; }
        let (mut ff, mut rr) = (f + df, r + dr);
        while on(ff, rr) {
            out = out.with(sq(ff, rr));
            match g.piece(ff, rr) {
                None => {}
                Some(x) if x.color == p.color && (x.role == Role::Queen || x.role == if diag { Role::Bishop } else { Role::Rook }) => {}
                Some(x) if x.color == p.color && x.role == Role::Pawn && diag && dr == fwd(p.color) => {
                    if on(ff + df, rr + dr) { out = out.with(sq(ff + df, rr + dr)); }
                    break;
                }
                Some(_) => break,
            }
            ff += df; rr += dr;
        }
    }
    out
}

/// Pieces that stop it: the first piece on each ray of a slider, own pieces on a knight's or
/// king's squares, the piece in front of a pawn.
fn blockers(g: &Geo, s: Square, p: Piece) -> Vec<Square> {
    let b = &g.b;
    let (f, r) = fr(s);
    match p.role {
        Role::Knight | Role::King => (b.attacks_from(s) & b.by_color(p.color)).into_iter().collect(),
        Role::Pawn => g.piece(f, r + fwd(p.color)).map(|_| sq(f, r + fwd(p.color))).into_iter().collect(),
        _ => DIRS.iter().filter_map(|&(df, dr)| {
            let diag = df != 0 && dr != 0;
            if !(p.role == Role::Queen || (p.role == Role::Rook) != diag) { return None; }
            let (mut ff, mut rr) = (f + df, r + dr);
            while on(ff, rr) { if g.piece(ff, rr).is_some() { return Some(sq(ff, rr)); } ff += df; rr += dr; }
            None
        }).collect(),
    }
}

/// Two own pawns side by side on adjacent files within c–f, no own pawn on the files beside the
/// pair, no enemy pawn on either of the pair's files.
pub fn hanging_pawn(g: &Geo, c: Color, f: i32, r: i32) -> bool {
    let has = |ff: i32, who: Color| (0..8).any(|rr| g.pawn(ff, rr, who));
    [-1, 1].iter().any(|&df| {
        let (a, z) = (f.min(f + df), f.max(f + df));
        (2..=5).contains(&a) && (2..=5).contains(&z) && g.pawn(f + df, r, c)
            && !has(a - 1, c) && !has(z + 1, c) && !has(a, c.other()) && !has(z, c.other())
    })
}

fn rook(g: &Geo, s: Square, c: Color, o: &mut Out) {
    let b = &g.b;
    let (f, r) = fr(s);
    let open = !(0..8).any(|rr| g.piece(f, rr).is_some_and(|x| x.role == Role::Pawn));
    let enemy_rook = (0..8).any(|rr| g.is(f, rr, c.other(), Role::Rook));
    if open && !enemy_rook { o.set(s, "rook.open_file", &[f as usize]); }
    if rel(c, r) == 6 { o.set(s, "rook.seventh", &[0]); }
    if rel(c, r) == 7 { o.set(s, "rook.eighth", &[0]); }
    for t in shakmaty::attacks::rook_attacks(s, b.occupied()) & b.by_color(c) {
        match b.role_at(t) {
            Some(Role::Rook) => o.set(s, "rook.battery", &[0, t as usize]),
            Some(Role::Queen) => o.set(s, "rook.battery", &[1, t as usize]),
            _ => {}
        }
    }
    for rr in 0..8 {
        if !g.pawn(f, rr, c) || rel(c, rr) <= rel(c, r) { continue; }
        if passed(g, c, f, rr) { o.set(s, "rook.behind_passer", &[sq(f, rr) as usize]); }
        if hanging_pawn(g, c, f, rr) { o.set(s, "rook.behind_hanging", &[sq(f, rr) as usize]); }
    }
}

/// The own pawns linked to this one by pawn guards, itself included.
fn chain(g: &Geo, s: Square, c: Color) -> Vec<Square> {
    let pawns = g.b.pawns() & g.b.by_color(c);
    let mut seen = vec![s];
    let mut i = 0;
    while i < seen.len() {
        let x = seen[i];
        // the pawns it guards and the pawns guarding it
        let near = (shakmaty::attacks::pawn_attacks(c, x) | shakmaty::attacks::pawn_attacks(c.other(), x)) & pawns;
        for y in near { if !seen.contains(&y) { seen.push(y); } }
        i += 1;
    }
    seen
}

fn sector(f: f64) -> usize {
    // nearest of a–c, d–e, f–h; a tie goes to the centre
    if f < 2.5 { 0 } else if f <= 4.5 { 1 } else { 2 }
}

fn pawn(g: &Geo, s: Square, c: Color, o: &mut Out) {
    let (f, r) = fr(s);
    let ch = chain(g, s, c);
    if ch.len() >= 2 {
        let pawns = g.b.pawns() & g.b.by_color(c);
        for &t in &ch { if t != s { o.set(s, "pawn.chain", &[t as usize]); } }
        let guards = (shakmaty::attacks::pawn_attacks(c, s) & pawns).any();
        let guarded = (shakmaty::attacks::pawn_attacks(c.other(), s) & pawns).any();
        match (guards, guarded) {
            (false, true) => o.set(s, "pawn.chain_pos", &[0]),
            (true, true) => o.set(s, "pawn.chain_pos", &[1]),
            (true, false) => o.set(s, "pawn.chain_pos", &[2]),
            _ => {}
        }
        // straight: the file the front pawn points at; bent: the average file
        let links: Vec<i32> = ch.iter().flat_map(|&x| (shakmaty::attacks::pawn_attacks(c, x) & pawns).into_iter()
            .map(move |y| fr(y).0 - fr(x).0)).collect();
        let dir = if links.iter().all(|&d| d == links[0]) {
            let front = *ch.iter().max_by_key(|&&x| rel(c, fr(x).1)).unwrap();
            sector((fr(front).0 + links[0]).clamp(0, 7) as f64)
        } else {
            sector(ch.iter().map(|&x| fr(x).0 as f64).sum::<f64>() / ch.len() as f64)
        };
        o.set(s, "pawn.chain_dir", &[dir]);
    }
    if passed(g, c, f, r) { o.set(s, "pawn.passed", &[0]); }
    if hanging_pawn(g, c, f, r) { o.set(s, "pawn.hanging", &[0]); }
    if isolated(g, c, f) { o.set(s, "pawn.isolated", &[0]); }
    for rr in 0..8 { if rr != r && g.pawn(f, rr, c) { o.set(s, "pawn.doubled_with", &[rr as usize]); } }
    for k in 1..=6 {
        if let Some(x) = g.piece(f, r + k * fwd(c)) { o.set(s, "pawn.blocked_by", &[k as usize - 1, ti(x.role)]); }
    }
    if (2..=5).contains(&f) { o.set(s, "pawn.centre", &[features::centre::kind(g).0]); }
    if (3..=5).contains(&rel(c, r)) { o.set(s, "pawn.rank4_6", &[0]); }
    if rel(c, r) == 6 { o.set(s, "pawn.rank7", &[0]); }
}

pub fn chess(fen: &str) -> Result<Chess, String> { features::position(fen) }
