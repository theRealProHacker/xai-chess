//! Evaluates a `Reason` against the board.
//!
//! Tier 1: board facts, shakmaty only. Tier 2: Stockfish over UCI (`STOCKFISH`, default the vendored
//! binary). Tier 3: heuristics ported from commentary_gen/board_tools.py and positional.py.
//!
//! Caveats:
//! - `Because(a, b)` holds when `a` and `b` both hold. Under `Rated(Dubious/Mistake/Blunder)` the material `b`
//!   names must net the mover a loss; a named gain fails (`explains`). Otherwise that `b` causes `a` is not checked.
//! - `Material` counts from before the annotated move (before m under `Instead(m, ..)`), so what that move takes counts.
//! - `Outweighs(a, b)` and `Balances(a, b)`: a and b point opposite ways for the owner (`polarity`); the weighing is not checked.
//! - `Mate` holds on a mate on the board or the engine's forced mate for the owner.
//! - `Promote(s)` follows the engine's line `PROMOTE_PLIES` ahead. `Pin` needs something to win behind the front piece.
//! - `Quality` is the eval minus the median eval with the piece on its other safe squares (`relocated`); `Better` compares that.
//! - `Eventually(r)` holds when `r` fails now and holds after the engine's line, `EVENTUALLY_PLIES` ahead.
//! - `Only(r)` and `NoMove(r)` ignore moves that achieve `r` but score `ONLY_MARGIN` or more below the played move.
//! - `Material` follows the engine's line (`SETTLE_NODES`) until the exchanges settle. `Sacrifice` uses static exchange evaluation.
//! - `But(a, b)`: a and b point opposite ways for the owner (`polarity`).
//! - `Zugzwang` and `Faster` are measured by passing (`zugzwang`, `faster`).
//! - `Tempo(moves)` counts the opponent's moves since the position before the annotated move (`tempo`).
//! - Lines in `After` and `Allows` must be best play: each move within `LINE_MARGIN` of its mover's best, widening to `LINE_MARGIN_EDGE` in decided positions.
//! - `Prevents`, `Loses`, `Gains` and `Concedes` compare with the position before the enclosing move or line.

#![cfg_attr(target_arch = "wasm32", allow(unused_imports, dead_code))]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufRead, BufReader, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::uci::UciMove;
use shakmaty::{
    Bitboard, CastlingMode, CastlingSide, Chess, Color, EnPassantMode, Move, Piece, Position, Rank, Role, Square, attacks,
};

use crate::dsl_claude::{Annotation, Grade, Move_, Pieces, QUESTIONABLE, Reason, Reason::*, Region, Sq, examples};

/// Centipawns an alternative may trail the played move and still count against `Only` and `NoMove`.
const ONLY_MARGIN: i32 = 30;

// Thresholds in centipawns, set leniently: a comment's claim should hold unless the board clearly disagrees.
const WIN: i32 = 200;
/// `Pressure` on a king: the attack measure (`king_attack`) it needs, and how far along the engine's line it may come.
const KING_PRESSURE: i32 = 8;
const KING_PLIES: usize = 8;
/// `Promote`: how far along the engine's line the pawn may queen.
const PROMOTE_PLIES: usize = 20;
/// `Permanent` for reasons other than pawn structure: how far along the engine's line they must still hold.
const PERMANENT_PLIES: usize = 8;
/// `Perpetual`: how far along the engine's line a position must repeat.
const PERPETUAL_PLIES: usize = 12;
/// The engine's score for a forced mate.
const MATE: i32 = 10000;
const DRAW: i32 = 100;
/// A threat's first move may score this far below the position as it stands.
const THREAT_MARGIN: i32 = 50;
/// `Rated(Best)`: the engine's move, or this close to it.
const BEST_MARGIN: i32 = 10;
/// `Rated`, Lichess's grading: the winning chances lost against the best move make the move an
/// inaccuracy (Dubious), a mistake or a blunder from these up. Good loses less than an inaccuracy.
const INACCURACY: f64 = 0.1;
const MISTAKE: f64 = 0.2;
const BLUNDER: f64 = 0.3;
/// Neighbouring grades overlap by this much on each side of a boundary: the loss moves by up to ~0.04
/// winning chances between searches of 200k, 300k and 450k nodes (all 120 annotate cases).
const GRADE_MARGIN: f64 = 0.04;
/// `Rated(Brilliant)`, chess.com's rule: a best move that sacrifices a piece, not already completely
/// winning without it (owner's winning chances after the best other move below this), and not bad after it (above minus this).
const BRILLIANT_WINNING: f64 = 0.8;
const BRILLIANT_BAD: f64 = 0.2;
/// Each move of a line in `After` and `Allows` may lose at most this many winning chances against its mover's best move.
/// `LINE_MARGIN` in a level position; it widens linearly to `LINE_MARGIN_EDGE` as the best move's winning
/// chances approach ±1, where a move costs little in winning chances that matter.
const LINE_MARGIN: f64 = 0.15;
const LINE_MARGIN_EDGE: f64 = 0.3;

/// `Because(Rated(..), b)`: the material `b` names must explain at least this share of the winning chances lost.

/// `Quality` and `Better`: search size per relocation, and the fewest other squares that give a median.
const RELOC_NODES: u64 = 50_000;
const RELOC_MIN: usize = 5;

/// Search size for settling exchanges in `Material`.
const SETTLE_NODES: u64 = 1_000_000;

/// `Simplify`: how far along the engine's line the material may come off.
const SIMPLIFY_PLIES: usize = 4;

/// `Zugzwang`: the most positions the opponent's moves-alone search visits.
const SHUFFLE_LIMIT: usize = 50_000;

/// How far ahead `Eventually` looks along the engine's line.
const EVENTUALLY_PLIES: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum V {
    Holds,
    Fails,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub v: V,
    pub why: String,
}

fn ok() -> Verdict { Verdict { v: V::Holds, why: String::new() } }
fn fail(why: impl Into<String>) -> Verdict { Verdict { v: V::Fails, why: why.into() } }
fn unknown(why: impl Into<String>) -> Verdict { Verdict { v: V::Unknown, why: why.into() } }
fn check(b: bool, why: impl Into<String>) -> Verdict { if b { ok() } else { fail(why) } }

/// Fails beats Unknown beats Holds.
fn all(vs: impl IntoIterator<Item = Verdict>) -> Verdict {
    let vs: Vec<_> = vs.into_iter().collect();
    vs.iter().find(|x| x.v == V::Fails).or(vs.iter().find(|x| x.v == V::Unknown)).cloned().unwrap_or_else(ok)
}

/// Holds beats Unknown beats Fails.
fn any(vs: Vec<Verdict>, why: &str) -> Verdict {
    vs.iter().find(|x| x.v == V::Holds).or(vs.iter().find(|x| x.v == V::Unknown)).cloned().unwrap_or_else(|| fail(why))
}

fn not(v: Verdict) -> Verdict {
    match v.v {
        V::Holds => fail(format!("not: {}", if v.why.is_empty() { "it holds" } else { &v.why })),
        V::Fails => ok(),
        V::Unknown => v,
    }
}

// --------------------------------------------------------------------------- board helpers

thread_local! {
    /// Inside `Eventually`: where each piece, named by its square at the start of the line, stands now.
    /// None: captured along the line.
    static TRACK: RefCell<Option<[Option<Square>; 64]>> = const { RefCell::new(None) };
    /// Set when a reason names a piece that was captured along the line.
    static GONE: Cell<bool> = const { Cell::new(false) };
}

fn sq(s: Sq) -> Square {
    let s0: Square = s.parse().unwrap();
    TRACK.with(|t| match &*t.borrow() {
        None => s0,
        Some(m) => m[s0 as usize].unwrap_or_else(|| { GONE.set(true); s0 }),
    })
}

/// Move every tracked piece that `m` moves; drop the one it captures.
fn track(map: &mut [Option<Square>; 64], m: &Move) {
    if let Move::Castle { king, rook } = *m {
        let short = rook.file() > king.file();
        let (k, r) = if short { (shakmaty::File::G, shakmaty::File::F) } else { (shakmaty::File::C, shakmaty::File::D) };
        for e in map.iter_mut() {
            if *e == Some(king) { *e = Some(Square::from_coords(k, king.rank())) }
            else if *e == Some(rook) { *e = Some(Square::from_coords(r, rook.rank())) }
        }
        return;
    }
    let taken = match *m {
        Move::EnPassant { from, to } => Some(Square::from_coords(to.file(), from.rank())),
        _ if m.is_capture() => Some(m.to()),
        _ => None,
    };
    for e in map.iter_mut() {
        if e.is_some() && *e == taken { *e = None }
        else if *e == m.from() { *e = Some(m.to()) }
    }
}

pub(crate) fn val(r: Role) -> i32 {
    match r {
        Role::Pawn => 1,
        Role::Knight | Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 0,
    }
}

/// Centipawns, for `Better`: two minors outweigh rook and pawn.
fn cp(r: Role) -> i32 {
    match r {
        Role::Pawn => 100,
        Role::Knight | Role::Bishop => 325,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 0,
    }
}

fn worth(rs: &[Role]) -> i32 { rs.iter().map(|r| val(*r)).sum() }
fn material(p: &Chess, c: Color) -> i32 { p.board().by_color(c).into_iter().map(|s| val(p.board().role_at(s).unwrap())).sum() }
fn count(p: &Chess, c: Color, r: Role) -> usize { (p.board().by_color(c) & p.board().by_role(r)).count() }
fn attackers(p: &Chess, s: Square, c: Color) -> Bitboard { p.board().attacks_to(s, c, p.board().occupied()) }
fn pawns(p: &Chess, c: Color) -> Bitboard { p.board().pawns() & p.board().by_color(c) }
fn fen(p: &Chess) -> String { Fen::from_position(p, EnPassantMode::Legal).to_string() }

/// The king whose zone or wing `r` is: it does not count among the defenders of its own surroundings.
fn zone_king(r: &Region, p: &Chess, owner: Color) -> Bitboard {
    match r {
        Region::KingZone { opponent } => {
            let c = if *opponent { !owner } else { owner };
            p.board().kings() & p.board().by_color(c)
        }
        _ => Bitboard::EMPTY,
    }
}

/// How hard `owner` attacks the other king (CPW attack units): N and B 2, R 3, Q 5 for each piece aimed at
/// the king's ring (the king's 3x3, moved off the edge), 1 per ring square the owner hits that only the king
/// or queen defends, 2 per safe check. 0 with fewer than two pieces aimed; 99 when mated.
fn king_attack(p: &Chess, owner: Color) -> i32 {
    let them = !owner;
    if p.is_checkmate() && p.turn() == them { return 99; }
    let b = p.board();
    let Some(k) = b.king_of(them) else { return 0 };
    let c = Square::from_coords(shakmaty::File::new(k.file() as u32).min(shakmaty::File::G).max(shakmaty::File::B),
                                shakmaty::Rank::new(k.rank() as u32).min(Rank::Seventh).max(Rank::Second));
    let ring = attacks::king_attacks(c) | Bitboard::from(c);
    let pieces = (b.knights() | b.bishops() | b.rooks() | b.queens()) & b.by_color(owner);
    let aimed = ring.into_iter().fold(Bitboard::EMPTY, |acc, s| acc | (attackers(p, s, owner) & pieces));
    if aimed.count() < 2 { return 0; }
    let units: i32 = aimed.into_iter().map(|s| match b.role_at(s) { Some(Role::Queen) => 5, Some(Role::Rook) => 3, _ => 2 }).sum();
    let guards = !(b.kings() | b.queens());
    let weak = ring.into_iter().filter(|s| attackers(p, *s, owner).any() && (attackers(p, *s, them) & guards).is_empty()).count() as i32;
    let checks = as_mover(p, owner).map_or(0, |q| q.legal_moves().into_iter()
        .filter(|m| attackers(p, m.to(), them).is_empty() && q.clone().play(m.clone()).unwrap().is_check()).count() as i32);
    units + weak + 2 * checks
}

/// Attackers minus defenders at the other king, the rule of thumb behind "decisive": the owner's N, B, R and Q
/// aimed at the king's ring (through the owner's own sliders, so a battery counts twice), against the defender's
/// N, B, R and Q within two squares of its king, plus 1 for an intact shelter (a pawn on each square directly in
/// front of the king). Other pawns do not count.
fn king_count(p: &Chess, owner: Color) -> i32 {
    let them = !owner;
    let b = p.board();
    let Some(k) = b.king_of(them) else { return 0 };
    let c = Square::from_coords(shakmaty::File::new(k.file() as u32).min(shakmaty::File::G).max(shakmaty::File::B),
                                shakmaty::Rank::new(k.rank() as u32).min(Rank::Seventh).max(Rank::Second));
    let ring = attacks::king_attacks(c) | Bitboard::from(c);
    let pieces = |col: Color| (b.knights() | b.bishops() | b.rooks() | b.queens()) & b.by_color(col);
    let sliders = (b.bishops() | b.rooks() | b.queens()) & b.by_color(owner);
    let aimed = pieces(owner).into_iter().filter(|s| {
        let occ = b.occupied() & !(sliders & !Bitboard::from(*s));
        let piece = b.piece_at(*s).unwrap();
        (attacks::attacks(*s, piece, occ) & ring).any()
    }).count() as i32;
    let defenders = pieces(them).into_iter().filter(|s| s.distance(k) <= 2).count() as i32;
    let front = k.rank() as i32 + if them.is_white() { 1 } else { -1 };
    let intact = (0..8).contains(&front) && (k.file() as i32 - 1..=k.file() as i32 + 1).filter(|f| (0..8).contains(f))
        .all(|f| pawns(p, them).contains(Square::from_coords(shakmaty::File::new(f as u32), Rank::new(front as u32))));
    aimed - defenders - intact as i32
}

/// Pass if `c` is not to move. None when that side is in check.
fn as_mover(p: &Chess, c: Color) -> Option<Chess> {
    if p.turn() == c { Some(p.clone()) } else { p.clone().swap_turn().ok() }
}

/// The positions in which `c` moves next: `p`, `p` after a pass, or, when the other side is in check
/// and cannot pass, `p` after each of its legal replies. "`c` could do X" must then hold after every
/// reply. Empty when the other side is mated.
fn movers(p: &Chess, c: Color) -> Vec<Chess> {
    match as_mover(p, c) {
        Some(q) => vec![q],
        None => p.legal_moves().into_iter().map(|m| p.clone().play(m).unwrap()).collect(),
    }
}

/// The reply that leads from `p` to `q`, in SAN.
fn reply(p: &Chess, q: &Chess) -> String {
    p.legal_moves().into_iter().find(|m| p.clone().play(m.clone()).unwrap().board() == q.board())
        .map_or("?".into(), |m| San::from_move(p, m).to_string())
}

/// `f` holds in every position in which `c` moves next (see `movers`), and there is one.
fn every_mover(p: &Chess, c: Color, f: impl Fn(&Chess) -> bool) -> bool {
    let ms = movers(p, c);
    !ms.is_empty() && ms.iter().all(f)
}

pub(crate) fn san(p: &Chess, m: Move_) -> Result<Move, String> {
    San::from_ascii(m.as_bytes()).map_err(|_| format!("bad SAN {m}"))?.to_move(p).map_err(|_| format!("illegal {m}"))
}

/// Play `line`. With `owner` set, a pass first if the owner is not to move. "--" passes.
fn play(p: &Chess, owner: Option<Color>, line: &[Move_]) -> Result<Chess, String> {
    let pass = |p: Chess| p.swap_turn().map_err(|_| "cannot pass in check".to_string());
    let mut p = p.clone();
    if owner.is_some_and(|c| c != p.turn()) {
        p = pass(p)?;
    }
    for m in line {
        p = if *m == "--" { pass(p)? } else { let mv = san(&p, m)?; p.play(mv).map_err(|_| format!("illegal {m}"))? };
    }
    Ok(p)
}

/// The position after `hist` from `start` (None: a pass).
fn replay(start: &Chess, hist: &[Option<Move>]) -> Option<Chess> {
    hist.iter().try_fold(start.clone(), |q, m| match m { Some(m) => q.play(m.clone()).ok(), None => q.swap_turn().ok() })
}

fn side(c: Color) -> &'static str { if c.is_white() { "White" } else { "Black" } }

/// The moves `play` makes, a pass as None.
fn moves(p: &Chess, owner: Option<Color>, line: &[Move_]) -> Option<Vec<Option<Move>>> {
    let mut p = p.clone();
    let mut out = vec![];
    if owner.is_some_and(|c| c != p.turn()) { p = p.swap_turn().ok()?; out.push(None); }
    for m in line {
        if *m == "--" { p = p.swap_turn().ok()?; out.push(None); continue; }
        let mv = san(&p, m).ok()?;
        p = p.play(mv.clone()).ok()?;
        out.push(Some(mv));
    }
    Some(out)
}

/// Lichess's winning chances in [-1, 1] for a score in centipawns.
fn chances(cp: i32) -> f64 { 2.0 / (1.0 + (-0.00368208 * cp as f64).exp()) - 1.0 }

/// The side to move in `p` moves alone while the other side passes at every turn. Some(true): no
/// position it can reach this way offers it a check, capture or promotion, so it can only shuffle.
/// Some(false): it can get through. None: more than `SHUFFLE_LIMIT` positions.
fn shuffles(p: &Chess) -> Option<bool> {
    let mut seen = std::collections::HashSet::new();
    let mut todo = vec![p.clone()];
    while let Some(q) = todo.pop() {
        // The board and the side to move; the move counters would make every position new.
        if !seen.insert((q.board().clone(), q.turn())) { continue; }
        if seen.len() > SHUFFLE_LIMIT { return None; }
        for m in q.legal_moves() {
            if m.is_capture() || m.is_promotion() { return Some(false); }
            let next = q.clone().play(m).unwrap();
            if next.is_check() { return Some(false); }
            // The other side passes: the mover is to move again.
            if let Ok(back) = next.swap_turn() { todo.push(back); }
        }
    }
    Some(true)
}

fn gain(m: &Move) -> i32 { m.capture().map_or(0, val) + m.promotion().map_or(0, |r| val(r) - 1) }

/// The side to move's best net gain from starting an exchange on `s`, least valuable attacker first.
fn see_square(p: &Chess, s: Square) -> i32 {
    let m = p.capture_moves().into_iter()
        .filter(|m| m.to() == s && m.promotion().is_none_or(|r| r == Role::Queen))
        .min_by_key(|m| val(m.role()));
    m.map_or(0, |m| (gain(&m) - see_square(&p.clone().play(m).unwrap(), s)).max(0))
}

/// Net material of `m` and the exchange on its destination.
fn see(p: &Chess, m: &Move) -> i32 { gain(m) - see_square(&p.clone().play(m.clone()).unwrap(), m.to()) }

/// Can the other side win material by capturing on `s`?
fn would_hang(p: &Chess, s: Square) -> bool {
    let Some(c) = p.board().color_at(s) else { return false };
    every_mover(p, !c, |b| b.board().color_at(s) == Some(c) && see_square(b, s) > 0)
}

/// Best net gain for the side to move from any capture.
fn best_capture(p: &Chess) -> i32 { p.capture_moves().iter().map(|m| see(p, m)).max().unwrap_or(0).max(0) }

fn safe_mobility(p: &Chess, s: Square) -> usize {
    let Some(c) = p.board().color_at(s) else { return 0 };
    movers(p, c).iter().map(|b| b.legal_moves().iter().filter(|m| m.from() == Some(s) && see(b, m) >= 0).count()).min().unwrap_or(0)
}

fn rank(s: Square, c: Color) -> u32 { c.relative_rank(s.rank()) as u32 }

fn adjacent_files(s: Square) -> Bitboard {
    let f = s.file() as i32;
    [f - 1, f + 1].into_iter().filter(|f| (0..8).contains(f)).map(|f| Bitboard::from_file(shakmaty::File::new(f as u32))).fold(Bitboard::EMPTY, |a, b| a | b)
}

fn file_bb(s: Square) -> Bitboard { Bitboard::from_file(s.file()) }

/// Squares strictly ahead of `s` for `c`.
fn ahead(s: Square, c: Color) -> Bitboard { Bitboard::FULL.into_iter().filter(|t| rank(*t, c) > rank(s, c)).collect() }
fn behind(s: Square, c: Color) -> Bitboard { Bitboard::FULL.into_iter().filter(|t| rank(*t, c) < rank(s, c)).collect() }

fn passed(p: &Chess, s: Square, c: Color) -> bool {
    (pawns(p, !c) & ahead(s, c) & (file_bb(s) | adjacent_files(s))).is_empty() && (pawns(p, c) & ahead(s, c) & file_bb(s)).is_empty()
}

fn isolated(p: &Chess, s: Square, c: Color) -> bool { (pawns(p, c) & adjacent_files(s)).is_empty() }
fn doubled(p: &Chess, s: Square, c: Color) -> bool { (pawns(p, c) & file_bb(s)).count() > 1 }

/// No own pawn beside or behind to support it, and an enemy pawn guards its stop square.
fn backward(p: &Chess, s: Square, c: Color) -> bool {
    let support = pawns(p, c) & adjacent_files(s) & (behind(s, c) | Bitboard::from_rank(s.rank()));
    let stop = s.offset(if c == Color::White { 8 } else { -8 });
    !isolated(p, s, c) && support.is_empty()
        && stop.is_some_and(|t| (attackers(p, t, !c) & p.board().pawns()).any())
}

/// On one of the files beside `s`, no pawn of `c` can ever guard it.
fn weakened(p: &Chess, s: Square, c: Color) -> bool {
    adjacent_files(s).into_iter().map(|t| t.file()).collect::<std::collections::BTreeSet<_>>().into_iter()
        .any(|f| (pawns(p, c) & Bitboard::from_file(f) & behind(s, c)).is_empty())
}

/// No pawn of `c` can ever guard `s`.
fn hole(p: &Chess, s: Square, c: Color) -> bool { (pawns(p, c) & adjacent_files(s) & behind(s, c)).is_empty() }

/// The squares of `r` for `owner`.
fn region(r: &Region, p: &Chess, owner: Color) -> Result<Vec<Square>, String> {
    let files = |fs: std::ops::RangeInclusive<u32>| Bitboard::FULL.into_iter().filter(|s| fs.contains(&(s.file() as u32))).collect::<Vec<_>>();
    let box_ = |lo: u32, hi: u32| Bitboard::FULL.into_iter().filter(|s| (lo..=hi).contains(&(s.file() as u32)) && (lo..=hi).contains(&(s.rank() as u32))).collect();
    let side = |opponent: bool| if opponent { !owner } else { owner };
    Ok(match r {
        Region::Squares(ss) if ss.is_empty() => return Err("no squares".into()),
        Region::Squares(ss) => ss.iter().map(|s| sq(s)).collect(),
        Region::Kingside => files(5..=7),
        Region::Queenside => files(0..=2),
        Region::Centre => box_(3, 4),
        Region::GreaterCentre => box_(2, 5),
        Region::Half { opponent } => {
            let c = side(*opponent);
            Bitboard::FULL.into_iter().filter(|s| rank(*s, c) <= 3).collect()
        }
        Region::KingZone { opponent } => {
            let k = p.board().king_of(side(*opponent)).ok_or("no king")?;
            (attacks::king_attacks(k) | Bitboard::from(k)).into_iter().collect()
        }
        Region::Line(a, z) => (attacks::between(sq(a), sq(z)) | Bitboard::from(sq(a)) | Bitboard::from(sq(z))).into_iter().collect(),
    })
}

/// Starting squares of `r` for `c`.
fn start(r: Role, c: Color) -> Bitboard {
    let w = match r {
        Role::Knight => Bitboard::from(Square::B1) | Bitboard::from(Square::G1),
        Role::Bishop => Bitboard::from(Square::C1) | Bitboard::from(Square::F1),
        Role::Rook => Bitboard::from(Square::A1) | Bitboard::from(Square::H1),
        Role::Queen => Bitboard::from(Square::D1),
        _ => Bitboard::EMPTY,
    };
    if c == Color::White { w } else { w.flip_vertical() }
}

fn slider_line(role: Role, a: Square, b: Square) -> bool {
    let diag = attacks::bishop_attacks(a, Bitboard::EMPTY).contains(b);
    let orth = attacks::rook_attacks(a, Bitboard::EMPTY).contains(b);
    match role { Role::Bishop => diag, Role::Rook => orth, Role::Queen => diag || orth, _ => false }
}

// --------------------------------------------------------------------------- engine

/// Score in centipawns for the side to move (mate = ±10000) and the principal variation in UCI,
/// from the `info` lines of one search.
pub(crate) fn parse_search(lines: &[String]) -> (i32, Vec<String>) {
    let (mut score, mut pv) = (0, vec![]);
    for l in lines {
        let t: Vec<&str> = l.split_whitespace().collect();
        let Some(i) = t.iter().position(|x| *x == "score") else { continue };
        if t.contains(&"upperbound") || t.contains(&"lowerbound") { continue; }
        let (Some(kind), Some(n)) = (t.get(i + 1), t.get(i + 2)) else { continue };
        let n: i32 = n.parse().unwrap_or(0);
        score = if *kind == "mate" { if n > 0 { 10000 } else { -10000 } } else { n };
        if let Some(j) = t.iter().position(|x| *x == "pv") { pv = t[j + 1..].iter().map(|s| s.to_string()).collect(); }
    }
    (score, pv)
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Engine {
    _child: Child,
    inp: ChildStdin,
    out: BufReader<ChildStdout>,
    nodes: u64,
    cache: HashMap<String, (i32, Vec<String>)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Engine {
    pub fn start(path: &str, nodes: u64) -> Option<Engine> {
        let mut child = Command::new(path).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().ok()?;
        let inp = child.stdin.take()?;
        let out = BufReader::new(child.stdout.take()?);
        let mut e = Engine { _child: child, inp, out, nodes, cache: HashMap::new() };
        e.send("uci");
        e.until("uciok");
        e.send("setoption name Threads value 1");
        e.send("isready");
        e.until("readyok");
        Some(e)
    }

    fn send(&mut self, s: &str) {
        writeln!(self.inp, "{s}").unwrap();
        self.inp.flush().unwrap();
    }

    fn until(&mut self, tag: &str) -> Vec<String> {
        let mut lines = vec![];
        loop {
            let mut l = String::new();
            if self.out.read_line(&mut l).unwrap() == 0 { return lines; }
            let done = l.starts_with(tag);
            lines.push(l.trim().to_string());
            if done { return lines; }
        }
    }

    /// Score in centipawns for the side to move (mate = ±10000) and the principal variation in UCI.
    pub fn analyse(&mut self, p: &Chess) -> (i32, Vec<String>) { self.analyse_nodes(p, self.nodes) }

    pub fn analyse_nodes(&mut self, p: &Chess, nodes: u64) -> (i32, Vec<String>) {
        if p.is_checkmate() { return (-10000, vec![]); }
        if p.is_stalemate() || p.is_insufficient_material() { return (0, vec![]); }
        let f = fen(p);
        let key = format!("{f} {nodes}");
        if let Some(r) = self.cache.get(&key) { return r.clone(); }
        // Clear the hash so a search does not depend on which positions came before it.
        self.send("ucinewgame");
        self.send("isready");
        self.until("readyok");
        self.send(&format!("position fen {f}"));
        self.send(&format!("go nodes {nodes}"));
        let r = parse_search(&self.until("bestmove"));
        self.cache.insert(key, r.clone());
        r
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Engine {
    fn drop(&mut self) { let _ = writeln!(self.inp, "quit"); }
}

/// In the browser the search runs in JavaScript, outside this call. A search not in `cache` is
/// recorded in `missing`, answered with a dummy, and the caller runs the search and evaluates again.
#[cfg(target_arch = "wasm32")]
pub struct Engine {
    pub nodes: u64,
    pub cache: HashMap<String, (i32, Vec<String>)>,
    /// The first search this evaluation lacked: "fen nodes".
    pub missing: Option<String>,
}

#[cfg(target_arch = "wasm32")]
impl Engine {
    pub fn analyse(&mut self, p: &Chess) -> (i32, Vec<String>) { self.analyse_nodes(p, self.nodes) }

    pub fn analyse_nodes(&mut self, p: &Chess, nodes: u64) -> (i32, Vec<String>) {
        if p.is_checkmate() { return (-10000, vec![]); }
        if p.is_stalemate() || p.is_insufficient_material() { return (0, vec![]); }
        let key = format!("{} {nodes}", fen(p));
        if let Some(r) = self.cache.get(&key) { return r.clone(); }
        self.missing.get_or_insert(key);
        (0, vec![])
    }
}

// --------------------------------------------------------------------------- evaluator

/// Where a reason is evaluated: the position, the one before the enclosing move or line, and whose reason it is.
#[derive(Clone)]
struct Cx {
    p: Chess,
    prev: Chess,
    owner: Color,
    /// `hist` from `start` (None: a pass) reaches `p`; `Tempo` counts the moves in it.
    start: Chess,
    hist: Vec<Option<Move>>,
    /// The position before the annotated move, or before m under `Instead(m, ..)`. `Material` counts from
    /// here, so what that move itself takes counts.
    base: Option<Chess>,
}

impl Cx {
    /// A position with no history behind it.
    fn bare(p: Chess, prev: Chess, owner: Color) -> Cx { Cx { start: p.clone(), hist: vec![], p, prev, owner, base: None } }
    /// After `line` from `self.p`, as played by `play` with the same `first`.
    fn at(&self, p: Chess, owner: Color, first: Option<Color>, line: &[Move_]) -> Cx {
        match moves(&self.p, first, line) {
            Some(ms) => Cx { p, prev: self.p.clone(), owner, start: self.start.clone(), hist: [self.hist.clone(), ms].concat(), base: self.base.clone() },
            None => Cx { base: self.base.clone(), ..Cx::bare(p, self.p.clone(), owner) },
        }
    }
}

pub struct Ev {
    pub(crate) eng: Option<Engine>,
    /// Position before the move: `Trade` names pieces by their squares here.
    root: Chess,
    /// Position after the played move: `Stronger` and `Weaker` compare with it.
    played: Chess,
    /// Nodes that the comment claims hold, with their verdicts. Nodes under `Not`, `Prevents` and the like are not claims.
    pub claims: Vec<(String, Verdict)>,
    quiet: usize,
}

/// Whether `r` is for (+1) or against (-1) its owner, or neither (0). Claims are owner assets unless a
/// connective hands them to the opponent (`Allows`, `Concedes`) or negates them (`Loses`, `Not`).
/// `Quality` is about whoever owns the piece: `mine(s)` says whether that is the owner.
fn polarity(r: &Reason, mine: &dyn Fn(Sq) -> Option<bool>) -> i32 {
    let polarity = |r: &Reason| polarity(r, mine);
    use crate::dsl_claude::Annotation as A;
    let same = |rs: &[Reason]| {
        let ps: Vec<i32> = rs.iter().map(|r| polarity(r)).collect();
        if ps.iter().all(|x| *x > 0) { 1 } else if ps.iter().all(|x| *x < 0) { -1 } else { 0 }
    };
    match r {
        And(rs) | Either(rs) => if rs.is_empty() { 0 } else { same(rs) },
        Allows(_, r) | Concedes(r) | Loses(r) | Not(r) => -polarity(r),
        Threatens(_, r) | After(_, r) | Prevents(r) | Removes(r) | Gains(r) | Only(r) | Instead(_, r)
        | Permanent(r) | Eventually(r) | More(r) | Degree(_, r) | Suppose(_, r) => polarity(r),
        Because(a, _) => polarity(a),
        Outweighs(..) | Faster(..) => 1,
        But(..) | Balances(..) | NoMove(_) => 0,
        Sacrifice(..) => -1,
        Rated(a) => match a {
            A::Good | A::Best | A::Brilliant | A::Stronger => 1,
            A::Dubious | A::Mistake | A::Blunder | A::Weaker => -1,
            A::Interesting => 0,
        },
        Material { get, give, .. } => (worth(get) - worth(give)).signum(),
        Quality(s, g) => {
            let good = if matches!(g, Grade::Good) { 1 } else { -1 };
            match mine(s) { Some(true) => good, Some(false) => -good, None => 0 }
        }
        Trade { .. } | Check | Imbalance | Closed | Options(_) | Draw | Perpetual | SameColour(_) => 0,
        _ => 1,
    }
}

/// `r` uses the claim `kind` somewhere inside.
fn mentions(r: &Reason, kind: &str) -> bool {
    format!("{r:?}").split(|c: char| !c.is_alphanumeric()).any(|w| w == kind)
}

fn name(r: &Reason) -> String { format!("{r:?}").split(['(', ' ', '{']).next().unwrap().to_string() }

impl Ev {
    pub fn new(eng: Option<Engine>) -> Ev {
        Ev { eng, root: Chess::default(), played: Chess::default(), claims: vec![], quiet: 0 }
    }

    /// Evaluate `reason` for the side that plays `mov` in `root`.
    pub fn run(&mut self, root: &Chess, mov: &Move, reason: &Reason) -> Verdict {
        self.root = root.clone();
        self.claims.clear();
        self.played = root.clone().play(mov.clone()).unwrap();
        let cx = Cx { p: self.played.clone(), prev: root.clone(), owner: root.turn(), start: root.clone(), hist: vec![Some(mov.clone())], base: Some(root.clone()) };
        self.eval(reason, &cx)
    }

    /// The highest `king_attack` and `king_count` over the engine's line from `p`, `KING_PLIES` long.
    fn king_line(&mut self, p: &Chess, owner: Color) -> Option<(i32, i32)> {
        let (_, pv) = self.eng.as_mut()?.analyse(p);
        let (mut q, mut best, mut margin) = (p.clone(), 0, i32::MIN);
        for k in 0..=KING_PLIES {
            best = best.max(king_attack(&q, owner));
            margin = margin.max(king_count(&q, owner));
            let Some(m) = pv.get(k).and_then(|u| u.parse::<UciMove>().ok()).and_then(|u| u.to_move(&q).ok()) else { break };
            q = q.play(m).unwrap();
        }
        Some((best, margin))
    }

    fn owner_score(&mut self, p: &Chess, owner: Color) -> Option<i32> {
        let (s, _) = self.eng.as_mut()?.analyse(p);
        Some(if p.turn() == owner { s } else { -s })
    }

    fn quietly(&mut self, f: impl FnOnce(&mut Ev) -> Verdict) -> Verdict {
        self.quiet += 1;
        let v = f(self);
        self.quiet -= 1;
        v
    }

    fn eval(&mut self, r: &Reason, cx: &Cx) -> Verdict {
        let v = self.node(r, cx);
        if self.quiet == 0 && !matches!(r, And(_)) {
            self.claims.push((name(r), v.clone()));
        }
        v
    }

    fn node(&mut self, r: &Reason, cx: &Cx) -> Verdict {
        let node = r;
        let (p, owner) = (&cx.p, cx.owner);
        let b = p.board();
        let mine = |s: Square| b.color_at(s) == Some(owner);
        let theirs = |s: Square| b.color_at(s) == Some(!owner);
        match r {
            // ---------------------------------------------------------------- combinators
            And(rs) if rs.is_empty() => fail("nothing to hold"),
            And(rs) => all(rs.iter().map(|r| self.eval(r, cx)).collect::<Vec<_>>()),
            // Each alternative alone is not claimed, so they are not recorded as claims.
            Either(rs) => self.quietly(|ev| any(rs.iter().map(|r| ev.eval(r, cx)).collect(), "none holds")),
            Threatens(line, r) => {
                // After a check the threat must stand whatever the reply; the check makes it real.
                if as_mover(p, owner).is_none() {
                    let starts = movers(p, owner);
                    if starts.is_empty() { return fail("mated"); }
                    return self.quietly(|ev| all(starts.iter().map(|q| match play(q, None, line) {
                        Err(e) => fail(format!("after {}: {e}", reply(p, q))),
                        Ok(z) => ev.eval(r, &Cx::bare(z, p.clone(), owner)),
                    }).collect::<Vec<_>>()));
                }
                match play(p, Some(owner), line) {
                    Err(e) => fail(e),
                    Ok(q) => {
                        let v = self.eval(r, &cx.at(q, owner, Some(owner), line));
                        if v.v == V::Holds { self.threat_is_real(p, owner, line).unwrap_or(v) } else { v }
                    }
                }
            }
            After(line, r) | Allows(line, r) => match play(p, None, line) {
                Err(e) => fail(e),
                Ok(q) => {
                    let who = if matches!(node, After(..)) { owner } else { !owner };
                    let v = self.eval(r, &cx.at(q, who, None, line));
                    if v.v == V::Holds { self.line_is_played(p, line).unwrap_or(v) } else { v }
                }
            },
            Removes(r) => self.quietly(|ev| ev.before_after(r, cx, !owner, false)),
            Prevents(r) => self.quietly(|ev| {
                let before = ev.reachable(r, &cx.prev, !owner);
                let now = ev.reachable(r, &cx.p, !owner);
                match (before.v, now.v) {
                    (V::Holds, V::Fails) => ok(),
                    (V::Fails, _) => fail(format!("could not get it before either: {}", before.why)),
                    (_, V::Holds) => fail(format!("still can: {}", now.why)),
                    _ => unknown(format!("{} / {}", before.why, now.why)),
                }
            }),
            Loses(r) => self.quietly(|ev| ev.before_after(r, cx, owner, false)),
            Gains(r) => self.quietly(|ev| ev.before_after(r, cx, owner, true)),
            Concedes(r) => self.quietly(|ev| ev.before_after(r, cx, !owner, true)),
            Only(r) => {
                let v = self.eval(r, cx);
                if v.v != V::Holds { return v; }
                // "Only" means the only move that achieves `r` without losing ground: an alternative
                // that also achieves it counts only if the engine rates it within ONLY_MARGIN.
                self.quietly(|ev| {
                    let played = ev.owner_score(&cx.p, owner);
                    let mut undecided = false;
                    for m in cx.prev.legal_moves() {
                        let q = cx.prev.clone().play(m.clone()).unwrap();
                        if q.board() == cx.p.board() { continue; }
                        let alt = ev.eval(r, &Cx::bare(q.clone(), cx.prev.clone(), owner));
                        match alt.v {
                            V::Holds => match (played, ev.owner_score(&q, owner)) {
                                (Some(a), Some(z)) if a - z < ONLY_MARGIN =>
                                    return fail(format!("{} also does it ({:+} vs {a:+})", San::from_move(&cx.prev, m), z)),
                                (Some(_), Some(_)) => {}
                                _ => undecided = true,
                            },
                            V::Unknown => undecided = true,
                            V::Fails => {}
                        }
                    }
                    if undecided { unknown("some alternatives undecided") } else { ok() }
                })
            }
            NoMove(r) => self.quietly(|ev| {
                let played = ev.owner_score(&cx.p, owner);
                let mut undecided = false;
                for m in cx.prev.legal_moves() {
                    let q = cx.prev.clone().play(m.clone()).unwrap();
                    match ev.eval(r, &Cx::bare(q.clone(), cx.prev.clone(), owner)).v {
                        V::Holds => match (played, ev.owner_score(&q, owner)) {
                            (Some(a), Some(z)) if a - z < ONLY_MARGIN =>
                                return fail(format!("{} does it ({:+} vs {a:+})", San::from_move(&cx.prev, m), z)),
                            (Some(_), Some(_)) => {}
                            _ => undecided = true,
                        },
                        V::Unknown => undecided = true,
                        V::Fails => {}
                    }
                }
                if undecided { unknown("some moves undecided") } else { ok() }
            }),
            Instead(m, r) => match play(&cx.prev, None, &[m]) {
                Err(e) => fail(e),
                Ok(q) => {
                    // The history with the last move replaced, when that move was played from `cx.prev`.
                    let mut hist = cx.hist.clone();
                    let replaced = matches!(hist.pop(), Some(Some(_))) && replay(&cx.start, &hist).is_some_and(|z| z.board() == cx.prev.board() && z.turn() == cx.prev.turn());
                    let alt = if replaced {
                        hist.push(Some(san(&cx.prev, m).unwrap()));
                        Cx { p: q, prev: cx.prev.clone(), owner, start: cx.start.clone(), hist, base: Some(cx.prev.clone()) }
                    } else { Cx { base: Some(cx.prev.clone()), ..Cx::bare(q, cx.prev.clone(), owner) } };
                    self.eval(r, &alt)
                }
            },
            Sacrifice(roles, _) if roles.is_empty() => fail("no material named"),
            Sacrifice(roles, r) => {
                let need: i32 = roles.iter().map(|r| val(*r)).sum();
                let lost = material(&cx.prev, owner) - material(p, owner);
                let hang = if p.turn() != owner { best_capture(p) } else { 0 };
                let s = check(lost + hang >= need, format!("gives up {} of {need}", lost + hang));
                all([s, self.eval(r, cx)])
            }
            Not(r) => not(self.quietly(|ev| ev.eval(r, cx))),
            Because(a, b) => {
                let v = all([self.eval(a, cx), self.eval(b, cx)]);
                if v.v != V::Holds { return v; }
                if !matches!(&**a, Rated(Annotation::Dubious | Annotation::Mistake | Annotation::Blunder)) {
                    return Verdict { v: V::Holds, why: "both hold; cause unchecked".into() };
                }
                self.quietly(|ev| ev.explains(b, cx))
            }
            But(a, b) => {
                let v = all([self.eval(a, cx), self.eval(b, cx)]);
                if v.v != V::Holds { return v; }
                let mine = |s: Sq| cx.p.board().color_at(sq(s)).map(|c| c == owner);
                match (polarity(a, &mine), polarity(b, &mine)) {
                    (x, y) if x * y < 0 => ok(),
                    (0, _) | (_, 0) => unknown("no contrast: a part is neither for nor against the owner"),
                    _ => fail("no contrast: both parts point the same way"),
                }
            }
            // a is the owner's, b the opponent's: they must point opposite ways for the owner (sign only).
            Outweighs(a, b2) | Balances(a, b2) => {
                let vb = self.eval(b2, &Cx { owner: !owner, ..cx.clone() });
                let v = all([self.eval(a, cx), vb]);
                if v.v != V::Holds { return v; }
                let mine = |s: Sq| cx.p.board().color_at(sq(s)).map(|c| c == owner);
                let theirs = |s: Sq| mine(s).map(|m| !m);
                let (pa, pb) = (polarity(a, &mine), -polarity(b2, &theirs));
                if pa == 0 || pb == 0 { return unknown("a part is neither for nor against the owner"); }
                check(pa * pb < 0, "both parts point the same way")
            }
            Permanent(r) => {
                let v = self.eval(r, cx);
                // Pawn structure cannot be undone: pawns do not move backward.
                let structural = matches!(&**r, Isolated(_) | Doubled(_) | Backward(_) | Hole(_) | Weakened(_));
                if v.v != V::Holds || structural { return v; }
                // Anything else must still hold `PERMANENT_PLIES` along the engine's line.
                match self.along(r, cx, PERMANENT_PLIES) {
                    None => unknown("no engine line"),
                    Some(w) if w.v == V::Holds => Verdict { v: V::Holds, why: format!("still holds after {PERMANENT_PLIES} plies") },
                    Some(w) => fail(format!("gone after {PERMANENT_PLIES} plies: {}", w.why)),
                }
            }
            More(_) => unknown("comparison not modelled"),
            Degree(_, r) => {
                let v = self.eval(r, cx);
                if v.v == V::Holds { Verdict { v: V::Holds, why: "holds; degree unchecked".into() } } else { v }
            }
            Eventually(r) => {
                if self.quietly(|ev| ev.eval(r, cx)).v == V::Holds { return fail("already holds"); }
                self.along(r, cx, EVENTUALLY_PLIES).unwrap_or_else(|| unknown("no engine line"))
            }
            Faster(a, b) => self.quietly(|ev| ev.faster(a, b, cx)),
            Suppose(edits, r) => {
                let mut setup = p.to_setup(EnPassantMode::Legal);
                for (s, c) in edits {
                    match c {
                        Some(c) => setup.board.set_piece_at(sq(s), Piece::from_char(*c).unwrap()),
                        None => setup.board.discard_piece_at(sq(s)),
                    }
                }
                match setup.position::<Chess>(CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights()) {
                    Err(_) => fail("supposed position is illegal"),
                    Ok(q) => self.eval(r, &Cx::bare(q, cx.prev.clone(), owner)),
                }
            }

            // ---------------------------------------------------------------- tier 1: board facts
            // Mated on the board, or the owner has a forced mate (the engine's mate score).
            Mate if p.is_checkmate() => check(p.turn() != owner, "the owner is mated"),
            Mate => match self.owner_score(p, owner) {
                None => unknown("no engine"),
                Some(s) => check(s >= MATE, "no mate"),
            },
            Check => check(p.is_check(), "no check"),
            Material { get, give, total } => {
                let named = worth(get) - worth(give);
                let diff = |q: &Chess| material(q, owner) - material(q, !owner);
                let Some(settled) = self.settle(p) else { return unknown("no engine") };
                if *total { let d = diff(&settled); return check(d == named, format!("material {d:+} once the exchanges settle")); }
                let before = cx.base.clone().unwrap_or_else(|| cx.prev.clone());
                let at = if std::env::var("MAT_AT").as_deref() == Ok("settled") { &settled } else { p };
                let off = |c: Color, rs: &[Role]| rs.iter().all(|r| count(at, c, *r) + rs.iter().filter(|x| *x == r).count() <= count(&before, c, *r));
                let (won, gave) = (off(!owner, get), off(owner, give));
                let net = diff(&settled) - diff(&before);
                check(won && gave && net.signum() == named.signum(), format!("won {won}, gave {gave}, net {net:+} once the exchanges settle"))
            }
            Trade { give, get } if give.is_empty() || get.is_empty() => fail("a side of the trade is empty"),
            Trade { give, get } => {
                let root = self.root.clone();
                let gone = |s: &Sq, c: Color| {
                    let r = root.board().role_at(sq(s)).unwrap();
                    count(p, c, r) < count(&root, c, r)
                };
                let owner_attacked = |s: &Sq| {
                    let r = root.board().role_at(sq(s)).unwrap();
                    (b.by_color(owner) & b.by_role(r)).into_iter().any(|t| attackers(p, t, !owner).any())
                };
                let empty = give.iter().chain(get).find(|s| root.board().piece_at(sq(s)).is_none());
                if let Some(s) = empty { return fail(format!("nothing on {s} before the move")); }
                all(get.iter().map(|s| check(gone(s, !owner), format!("{s} not taken")))
                    .chain(give.iter().map(|s| check(gone(s, owner) || owner_attacked(s), format!("{s} not given")))).collect::<Vec<_>>())
            }
            Attack { by, targets } => {
                if !mine(sq(by)) { return fail(format!("{by} is not the owner's")); }
                if targets.is_empty() { return fail("no targets"); }
                all(targets.iter().map(|t| {
                    if !theirs(sq(t)) { fail(format!("no opponent piece on {t}")) }
                    else { check(b.attacks_from(sq(by)).contains(sq(t)), format!("{by} does not attack {t}")) }
                }).collect::<Vec<_>>())
            }
            Defend { by, target } => {
                if !mine(sq(by)) { return fail(format!("{by} is not the owner's")); }
                check(b.attacks_from(sq(by)).contains(sq(target)), format!("{by} does not reach {target}"))
            }
            Pin { by, front, behind } => {
                let (a, f, k) = (sq(by), sq(front), sq(behind));
                let between = attacks::between(a, k) & b.occupied();
                let role = b.role_at(a);
                if !(mine(a) && theirs(f) && theirs(k) && role.is_some_and(|r| slider_line(r, a, k)) && between == Bitboard::from(f)) {
                    return fail(format!("no pin {by}-{front}-{behind}"));
                }
                // The king counts as most valuable. A pin: the piece behind is worth more. A skewer (or equal
                // values): the piece behind falls once the front one moves, being undefended or worth the attacker.
                let worth = |s: Square| match b.role_at(s) { Some(Role::King) => 100, Some(r) => val(r), None => 0 };
                let defended = (attackers(p, k, !owner) & !Bitboard::from(f)).any();
                check(worth(k) > worth(f) || !defended || worth(k) >= worth(a),
                      format!("{behind} is defended and worth less than {by}: nothing to win"))
            }
            XRay { by, through, target } => {
                let (a, t, z) = (sq(by), sq(through), sq(target));
                let between = attacks::between(a, z) & b.occupied();
                check(mine(a) && theirs(t) && b.role_at(a).is_some_and(|r| slider_line(r, a, z))
                    && between == Bitboard::from(t), format!("no x-ray {by}-{through}-{target}"))
            }
            Block { line: (a, z), by } => {
                check(b.piece_at(sq(by)).is_some() && attacks::between(sq(a), sq(z)).contains(sq(by)), format!("{by} does not block"))
            }
            Open((a, z)) => {
                let ray = attacks::between(sq(a), sq(z)) | Bitboard::from(sq(a)) | Bitboard::from(sq(z));
                check((ray & pawns(p, owner)).is_empty(), format!("own pawn on {a}-{z}"))
            }
            Control(r) => all(match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss }.into_iter().map(|s| {
                let (own, opp) = (attackers(p, s, owner), attackers(p, s, !owner));
                check(own.any() && ((own & b.pawns()).any() || own.count() + 1 >= opp.count()), format!("{s}: {} vs {}", own.count(), opp.count()))
            }).collect::<Vec<_>>()),
            // On a king: the attack measure along the engine's line. The count (`king_count`) is reported only.
            Pressure(r @ Region::KingZone { .. }) => {
                let them = if matches!(r, Region::KingZone { opponent: true }) { !owner } else { owner };
                if them == owner { return fail("pressure on the owner's own king"); }
                let Some((best, margin)) = self.king_line(p, owner) else { return unknown("no engine") };
                let why = format!("attack {best} (needs {KING_PRESSURE}), {margin:+} attackers over defenders");
                if best < KING_PRESSURE { fail(why) } else { Verdict { v: V::Holds, why } }
            }
            Pressure(r) => {
                let ss = match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss };
                let attackers = |p: &Chess, s: Square, c: Color| attackers(p, s, c) & !zone_king(r, p, owner);
                if !matches!(r, Region::Squares(_)) {
                    let aimed = |c: Color| ss.iter().filter(|s| attackers(p, **s, c).any()).count();
                    return check(aimed(owner) > aimed(!owner), format!("aims at {} squares, opponent at {}", aimed(owner), aimed(!owner)));
                }
                all(ss.into_iter().map(|s| check(attackers(p, s, owner).any(), format!("nothing on {s}"))).collect::<Vec<_>>())
            }
            Outnumber(r) => {
                let ss = match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss };
                let n = |c: Color, s: Square| (attackers(p, s, c) & !zone_king(r, p, owner)).count();
                if !matches!(r, Region::Squares(_)) {
                    let (own, opp) = (ss.iter().filter(|s| n(owner, **s) > n(!owner, **s)).count(), ss.iter().filter(|s| n(!owner, **s) > n(owner, **s)).count());
                    return check(own > opp, format!("outnumbers on {own} squares, outnumbered on {opp}"));
                }
                all(ss.into_iter().map(|s| check(n(owner, s) > n(!owner, s), format!("{s}: {} vs {}", n(owner, s), n(!owner, s)))).collect::<Vec<_>>())
            }
            Bound { piece, duties } => {
                if !theirs(sq(piece)) { return fail(format!("{piece} is not the opponent's")); }
                if duties.is_empty() { return fail("no duties"); }
                all(duties.iter().map(|d| {
                    let (own, opp) = (attackers(p, sq(d), owner).count(), attackers(p, sq(d), !owner).count());
                    check(b.attacks_from(sq(piece)).contains(sq(d)) && own >= opp, format!("{piece} not tied to {d}: {own} vs {opp}"))
                }).collect::<Vec<_>>())
            }
            Decoy { from, to } => {
                let lured = every_mover(p, !owner, |q| q.legal_moves().iter().any(|m| m.from() == Some(sq(from)) && m.to() == sq(to)));
                check(theirs(sq(from)) && mine(sq(to)) && lured, format!("{from} cannot be lured to {to}"))
            }
            Develop(ps) => {
                let undeveloped = |roles: &[Role]| roles.iter().any(|r| (b.by_color(owner) & b.by_role(*r) & start(*r, owner)).any());
                match ps {
                    Pieces::All => check(!undeveloped(&[Role::Knight, Role::Bishop]), "minor piece at home"),
                    Pieces::At(s) => {
                        let s = sq(s);
                        let r = b.role_at(s);
                        check(mine(s) && r.is_some_and(|r| !matches!(r, Role::Pawn | Role::King) && !start(r, owner).contains(s)), "not developed")
                    }
                    Pieces::Every(r) => check(!undeveloped(&[*r]), format!("{r:?} at home")),
                    Pieces::On(reg) => {
                        let ss: Bitboard = match region(reg, p, owner) { Err(e) => return fail(e), Ok(ss) => ss.into_iter().collect() };
                        let home = [Role::Knight, Role::Bishop, Role::Rook, Role::Queen].into_iter()
                            .any(|r| (b.by_color(owner) & b.by_role(r) & start(r, owner) & ss).any());
                        check(!home, "a piece there is at home")
                    }
                }
            }
            Castle => {
                let k = b.king_of(owner).unwrap();
                let castled = rank(k, owner) == 0 && matches!(k.file() as u32, 1 | 2 | 6);
                let can = every_mover(p, owner, |q| q.legal_moves().iter().any(|m| matches!(m, Move::Castle { .. })));
                check(castled || can, "cannot castle")
            }
            BishopPair => {
                let bs = b.bishops() & b.by_color(owner);
                check(bs.into_iter().any(|s| s.is_light()) && bs.into_iter().any(|s| s.is_dark()), "no bishop pair")
            }
            Passer(s) => {
                let s = sq(s);
                check(mine(s) && b.role_at(s) == Some(Role::Pawn) && passed(p, s, owner), format!("{s} not passed"))
            }
            Majority(r) => {
                let ss = match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss };
                // Listed squares name owner pawns; the wing spans their files.
                if matches!(r, Region::Squares(_)) && !ss.iter().all(|s| mine(*s) && b.role_at(*s) == Some(Role::Pawn)) {
                    return fail("not all owner pawns");
                }
                let files: Vec<u32> = ss.iter().map(|s| s.file() as u32).collect();
                let (lo, hi) = (*files.iter().min().unwrap(), *files.iter().max().unwrap());
                let wing = |c: Color| pawns(p, c).into_iter().filter(|s| (lo..=hi).contains(&(s.file() as u32))).count();
                check(wing(owner) > wing(!owner), format!("{} vs {}", wing(owner), wing(!owner)))
            }
            // The pawn on s, or the pawn that queens on s: it has promoted in the line, can promote now,
            // or promotes along the engine's line within `PROMOTE_PLIES`.
            Promote(s) => {
                let s = sq(s);
                let last = rank(s, owner) == Rank::Eighth as u32;
                let promoted = last && mine(s) && cx.hist.iter().flatten().any(|m| m.to() == s && m.promotion().is_some());
                let can = every_mover(p, owner, |q| q.legal_moves().iter().any(|m| m.promotion().is_some() && (m.from() == Some(s) || (last && m.to() == s))));
                // The pawn named, or for a queening square the owner's most advanced pawn on its file.
                let pawn = if mine(s) && b.role_at(s) == Some(Role::Pawn) { Some(s) } else if last {
                    (b.by_color(owner) & b.by_role(Role::Pawn)).into_iter().filter(|t| t.file() == s.file()).max_by_key(|t| rank(*t, owner))
                } else { None };
                if promoted || can { ok() }
                else if let Some(from) = pawn {
                    match self.promotes_along(p, from) {
                        None => unknown("no engine"),
                        Some(at) => check(at.is_some_and(|at| at == s || !last), format!("the pawn on {from} does not promote in the engine's line")),
                    }
                } else { fail(format!("no pawn promotes on {s}")) }
            }
            Enables(line) if line.is_empty() => fail("no moves"),
            Enables(line) => self.enables(p, owner, line),
            Options(ms) if ms.len() < 2 => fail("fewer than two options"),
            Options(ms) => all(ms.iter().map(|m| self.enables(p, owner, &[m])).collect::<Vec<_>>()),
            SameColour(ss) if ss.len() < 2 => fail("fewer than two squares"),
            SameColour(ss) => check(ss.iter().all(|s| sq(s).is_light() == sq(ss[0]).is_light()), "mixed colours"),

            // ---------------------------------------------------------------- tier 2: engine
            Win => match self.owner_score(p, owner) {
                None => unknown("no engine"),
                Some(s) => check(s >= WIN, format!("eval {s:+}")),
            },
            Perpetual => self.perpetual(cx, owner),
            Draw => match self.owner_score(p, owner) {
                None => unknown("no engine"),
                Some(s) => check(s.abs() <= DRAW, format!("eval {s:+}")),
            },
            // Needs the stuck side to pass repeatedly, not once.
            Zugzwang => self.zugzwang(p),
            Simplify => self.simplify(cx),
            Initiative => self.initiative(cx, owner),
            Tempo(free) => self.tempo(cx, free),

            // ---------------------------------------------------------------- tier 3: heuristics
            Isolated(reg) | Doubled(reg) | Backward(reg) | Hole(reg) | Weakened(reg) | Hanging(reg) => {
                let ss = match region(reg, p, owner) { Err(e) => return fail(e), Ok(ss) => ss };
                let named = !matches!(reg, Region::Squares(_));
                let them = !owner;
                let pawn = |s: Square| b.color_at(s) == Some(them) && b.role_at(s) == Some(Role::Pawn);
                let test = |s: Square| match r {
                    Isolated(_) => check(pawn(s) && isolated(p, s, them), format!("{s} is not an isolated pawn")),
                    Doubled(_) => check(pawn(s) && doubled(p, s, them), format!("{s} is not a doubled pawn")),
                    Backward(_) => check(pawn(s) && backward(p, s, them), format!("{s} is not a backward pawn")),
                    Hole(_) => check(hole(p, s, them), format!("{s} can still be guarded by a pawn")),
                    Weakened(_) => check(weakened(p, s, them), format!("{s} can still be guarded from both sides")),
                    _ => check(b.color_at(s) == Some(them) && b.role_at(s) != Some(Role::King) && would_hang(p, s), format!("{s} does not hang")),
                };
                // A named region: holes count only on the opponent's 3rd and 4th ranks, weakenings only on its pawns.
                let vs: Vec<_> = ss.into_iter()
                    .filter(|s| !named || !matches!(r, Hole(_)) || matches!(rank(*s, them), 2 | 3))
                    .filter(|s| !named || !matches!(r, Weakened(_)) || pawn(*s))
                    .map(test).collect();
                if named { any(vs, "none in the region") } else { all(vs) }
            }
            Better { pieces, than } if pieces.is_empty() || than.is_empty() => fail("a side of the comparison is empty"),
            Better { pieces, than } => {
                let worth = |ss: &Vec<Sq>| ss.iter().filter_map(|s| b.role_at(sq(s))).map(cp).sum::<i32>();
                let mob = |ss: &Vec<Sq>| ss.iter().map(|s| safe_mobility(p, sq(s))).sum::<usize>();
                let sum = |ev: &mut Ev, ss: &Vec<Sq>| ss.iter().map(|s| ev.relocated(p, sq(s))).sum::<Option<i32>>();
                if let (Some(a), Some(z)) = (sum(self, pieces), sum(self, than)) {
                    return check(a > z, format!("{a:+} vs {z:+} cp against their other safe squares"));
                }
                let (a, z) = (worth(pieces), worth(than));
                if a != z { check(a > z, format!("{a} vs {z} cp")) } else { check(mob(pieces) > mob(than), format!("mobility {} vs {}", mob(pieces), mob(than))) }
            }
            Quality(s, g) => self.quality(p, sq(s), g),
            Coordinate(ss) if ss.len() < 2 => fail("fewer than two pieces"),
            Coordinate(ss) => {
                let s: Vec<Square> = ss.iter().map(|s| sq(s)).collect();
                if !s.iter().all(|x| mine(*x)) { return fail("not all the owner's"); }
                let linked = |a: Square, z: Square| b.attacks_from(a).contains(z) || b.attacks_from(z).contains(a);
                let mut seen = vec![s[0]];
                while let Some(n) = s.iter().find(|x| !seen.contains(x) && seen.iter().any(|y| linked(**x, *y))) { seen.push(*n); }
                check(seen.len() == s.len(), "pieces not connected")
            }
            Space => {
                let half = |c: Color| Bitboard::FULL.into_iter().filter(|s| rank(*s, c) >= 4).filter(|s| attackers(p, *s, c).any()).count();
                let (a, z) = (half(owner), half(!owner));
                check(a > z, format!("{a} vs {z} squares in the far half"))
            }
            Closed => {
                let w = pawns(p, Color::White);
                let locked = w.into_iter().filter(|s| s.offset(8).is_some_and(|t| pawns(p, Color::Black).contains(t))).count();
                let open = (0..8).filter(|f| (b.pawns() & Bitboard::from_file(shakmaty::File::new(*f))).is_empty()).count();
                check(locked >= 2 && open <= 2, format!("{locked} locked, {open} open files"))
            }
            Imbalance => {
                let pieces = |c: Color| [Role::Knight, Role::Bishop, Role::Rook, Role::Queen].map(|r| count(p, c, r));
                let mirror: Bitboard = pawns(p, Color::White).into_iter().map(|s| s.flip_vertical()).collect();
                check(pieces(owner) != pieces(!owner) || mirror != pawns(p, Color::Black), "symmetric")
            }
            Practical => unknown("practical chances not modelled"),
            Rated(a) => self.rated(a, cx),
        }
    }

    /// `Simplify`: from `cx.prev`, through `cx.p` and `SIMPLIFY_PLIES` of the engine's line on, a piece
    /// (not a pawn or king) or a pawn of each side comes off the board. Whether that is good is not asked.
    fn simplify(&mut self, cx: &Cx) -> Verdict {
        let Some(end) = self.pv_end(&cx.p, SIMPLIFY_PLIES) else { return unknown("no engine") };
        let pieces = |q: &Chess| (q.board().occupied() & !q.board().pawns() & !q.board().kings()).count() as i64;
        let pawns = |q: &Chess, c: Color| pawns(q, c).count();
        let off = pieces(&cx.prev) - pieces(&end);
        let both = pawns(&end, Color::White) < pawns(&cx.prev, Color::White) && pawns(&end, Color::Black) < pawns(&cx.prev, Color::Black);
        check(off > 0 || both, format!("{off} pieces off within {SIMPLIFY_PLIES} plies of the engine's line"))
    }

    /// `Tempo(free)`: in the moves since the position before the annotated move, the opponent spends more
    /// moves than it has pieces on new squares, at least one per free move, and the free moves are the owner's.
    /// A castle moves one piece; a captured piece is not counted.
    fn tempo(&self, cx: &Cx, free: &[Move_]) -> Verdict {
        if free.is_empty() { return fail("names no move"); }
        let (owner, them) = (cx.owner, !cx.owner);
        // Each of the opponent's pieces by where it stands now, with where it started.
        let mut at: std::collections::HashMap<Square, Square> = cx.start.board().by_color(them).into_iter().map(|s| (s, s)).collect();
        let (mut spent, mut castles, mut mine) = (0, 0, vec![]);
        let mut q = cx.start.clone();
        for m in &cx.hist {
            let Some(m) = m else { q = match q.swap_turn() { Ok(z) => z, Err(_) => return unknown("history does not replay") }; continue };
            if q.turn() == them {
                spent += 1;
                if let Move::Castle { king, rook } = m {
                    let side = if rook.file() > king.file() { CastlingSide::KingSide } else { CastlingSide::QueenSide };
                    let (k, r) = (at.remove(king), at.remove(rook));
                    if let Some(k) = k { at.insert(side.king_to(them), k); }
                    if let Some(r) = r { at.insert(side.rook_to(them), r); }
                    castles += 1;
                } else if let Some(o) = at.remove(&m.from().unwrap()) {
                    at.insert(m.to(), o);
                }
            } else {
                mine.push(San::from_move(&q, m.clone()).to_string());
                let taken = if let Move::EnPassant { from, to } = m { Square::from_coords(to.file(), from.rank()) } else { m.to() };
                if m.is_capture() { at.remove(&taken); }
            }
            q = match q.play(m.clone()) { Ok(z) => z, Err(_) => return unknown("history does not replay") };
        }
        let moved = (at.iter().filter(|(now, from)| now != from).count() as i32 - castles).max(0);
        let wasted = spent - moved;
        let bare = |s: &str| s.trim_end_matches(['+', '#']).to_string();
        if let Some(m) = free.iter().find(|m| !mine.iter().any(|x| bare(x) == bare(m))) {
            return fail(format!("{m} is not a move of {} here", side(owner)));
        }
        let why = format!("{} spends {spent} {} for {moved} {} on new squares", side(them), if spent == 1 { "move" } else { "moves" }, if moved == 1 { "piece" } else { "pieces" });
        Verdict { v: if wasted >= free.len() as i32 { V::Holds } else { V::Fails }, why }
    }

    /// `Faster(a, b)`: along the engine's line from `cx.p`, `a` holds for the owner before `b` holds
    /// for the opponent. Neither within `EVENTUALLY_PLIES`: unknown.
    fn faster(&mut self, a: &Reason, b: &Reason, cx: &Cx) -> Verdict {
        let Some((_, pv)) = self.eng.as_mut().map(|e| e.analyse(&cx.p)) else { return unknown("no engine") };
        // The first ply at which each side's reason holds along the engine's line.
        let (mut q, mut prev) = (cx.p.clone(), cx.prev.clone());
        let (mut first, mut second) = (None, None);
        for ply in 0..=EVENTUALLY_PLIES {
            if first.is_none() && self.eval(a, &Cx::bare(q.clone(), prev.clone(), cx.owner)).v == V::Holds { first = Some(ply) }
            if second.is_none() && self.eval(b, &Cx::bare(q.clone(), prev.clone(), !cx.owner)).v == V::Holds { second = Some(ply) }
            if first.is_some() && second.is_some() { break }
            let Some(m) = pv.get(ply).and_then(|u| u.parse::<UciMove>().ok()).and_then(|u| u.to_move(&q).ok()) else { break };
            prev = q.clone();
            q = q.play(m).unwrap();
        }
        let at = |x: Option<usize>| x.map_or("not along the engine's line".to_string(), |n| format!("after {n} plies"));
        let why = format!("the owner's {}, the opponent's {}", at(first), at(second));
        match (first, second) {
            (Some(x), Some(y)) if x < y => Verdict { v: V::Holds, why },
            (Some(_), None) => Verdict { v: V::Holds, why },
            (None, None) => unknown(why),
            _ => fail(why),
        }
    }

    /// The side to move is in zugzwang: passing, once or at every turn, is worth `INACCURACY` more
    /// winning chances than its best move. Passing at every turn holds (a draw) when the opponent,
    /// moving alone, can reach no check, capture or promotion: it can only shuffle (`shuffles`).
    fn zugzwang(&mut self, p: &Chess) -> Verdict {
        let s = p.turn();
        let Some(best) = self.owner_score(p, s).map(chances) else { return unknown("no engine") };
        let Ok(passed) = p.clone().swap_turn() else { return fail("in check: cannot pass") };
        let Some(once) = self.owner_score(&passed, s).map(chances) else { return unknown("no engine") };
        let forever = if once < 0.0 { match shuffles(&passed) { Some(true) => 0.0, _ => once } } else { once };
        let pass = once.max(forever);
        let why = format!("passing {} {pass:+.2}, best move {best:+.2}", if forever > once { "at every turn" } else { "once" });
        check(pass - best > INACCURACY, why)
    }

    /// For `who`, the reason held before and does not hold now, or with `gained` the other way round.
    fn before_after(&mut self, r: &Reason, cx: &Cx, who: Color, gained: bool) -> Verdict {
        let before = self.eval(r, &Cx::bare(cx.prev.clone(), cx.prev.clone(), who));
        let now = self.eval(r, &Cx { owner: who, ..cx.clone() });
        let (from, to) = if gained { (V::Fails, V::Holds) } else { (V::Holds, V::Fails) };
        match (before.v, now.v) {
            (b, n) if b == from && n == to => ok(),
            (b, _) if b == to => fail(if gained { format!("already held before") } else { format!("did not hold before: {}", before.why) }),
            (_, n) if n == from => fail(if gained { format!("does not hold: {}", now.why) } else { "still holds".into() }),
            _ => unknown(format!("{} / {}", before.why, now.why)),
        }
    }

    /// `r` holds for `who` in `p`, or after some move of `who` from `p` (a pass first if `who` is not to move).
    fn reachable(&mut self, r: &Reason, p: &Chess, who: Color) -> Verdict {
        if self.eval(r, &Cx::bare(p.clone(), p.clone(), who)).v == V::Holds { return ok(); }
        let starts = movers(p, who);
        if starts.is_empty() { return fail("mated"); }
        // After a check: some move gets it whatever the reply.
        all(starts.iter().map(|q| {
            let mut undecided = false;
            for m in q.legal_moves() {
                let next = q.clone().play(m.clone()).unwrap();
                match self.eval(r, &Cx::bare(next, q.clone(), who)).v {
                    V::Holds => return Verdict { v: V::Holds, why: format!("{}", San::from_move(q, m)) },
                    V::Unknown => undecided = true,
                    V::Fails => {}
                }
            }
            if undecided { unknown("some moves undecided") } else { fail("no move gets it") }
        }).collect::<Vec<_>>())
    }

    /// What the piece on `s` adds where it stands: the eval for its owner minus the median eval with
    /// the same piece on each other empty square where it is safe (bishops keep their colour, pawns stay
    /// off the back ranks), searched at `RELOC_NODES`. Nothing is removed. None without an engine, for a
    /// king, or with fewer than `RELOC_MIN` such squares.
    fn relocated(&mut self, p: &Chess, s: Square) -> Option<i32> {
        let (c, role) = (p.board().color_at(s)?, p.board().role_at(s)?);
        if role == Role::King { return None; }
        let score = |ev: &mut Ev, q: &Chess| {
            let (x, _) = ev.eng.as_mut()?.analyse_nodes(q, RELOC_NODES);
            Some(if q.turn() == c { x } else { -x })
        };
        let base = score(self, p)?;
        let mut vs = vec![];
        for t in !p.board().occupied() {
            if role == Role::Bishop && t.is_light() != s.is_light() { continue; }
            if role == Role::Pawn && matches!(t.rank(), Rank::First | Rank::Eighth) { continue; }
            let mut setup = p.to_setup(EnPassantMode::Legal);
            let piece = setup.board.remove_piece_at(s)?;
            setup.board.set_piece_at(t, piece);
            setup.ep_square = None;
            let Ok(q) = setup.position::<Chess>(CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights()) else { continue };
            if q.is_check() { continue; }
            // Safe: not attacked by a cheaper piece, and defended if attacked at all.
            let (att, def) = (attackers(&q, t, !c), attackers(&q, t, c));
            if att.into_iter().any(|a| val(q.board().role_at(a).unwrap()) < val(role)) || (att.any() && def.is_empty()) { continue; }
            vs.push(score(self, &q)?);
        }
        if vs.len() < RELOC_MIN { return None; }
        vs.sort();
        Some(base - vs[vs.len() / 2])
    }

    /// Good: the piece does more where it stands than on its other safe squares. Bad: less.
    fn quality(&mut self, p: &Chess, s: Square, g: &Grade) -> Verdict {
        let Some(role) = p.board().role_at(s) else { return fail(format!("{s} empty")) };
        let Some(w) = self.relocated(p, s) else { return unknown("no engine, a king, or too few other squares") };
        let why = format!("{s}: {w:+} cp vs the {role:?} on its other safe squares");
        match g {
            Grade::Good => check(w > 0, why),
            Grade::Bad => check(w < 0, why),
        }
    }

    /// Follow the engine's line while it captures, checks or promotes: the position once the exchanges settle.
    fn settle(&mut self, p: &Chess) -> Option<Chess> {
        let (_, pv) = self.eng.as_mut()?.analyse_nodes(p, SETTLE_NODES);
        let mut q = p.clone();
        for u in pv {
            let Ok(m) = u.parse::<UciMove>().map_err(|_| ()).and_then(|u| u.to_move(&q).map_err(|_| ())) else { break };
            let next = q.clone().play(m.clone()).unwrap();
            if !m.is_capture() && m.promotion().is_none() && !next.is_check() { break; }
            q = next;
        }
        Some(q)
    }

    /// The owner can legally play `line` next.
    fn enables(&mut self, p: &Chess, owner: Color, line: &[Move_]) -> Verdict {
        let starts = movers(p, owner);
        if starts.is_empty() { return fail("mated"); }
        // After a check: playable whatever the reply.
        all(starts.into_iter().map(|q| match play(&q, None, line) { Ok(_) => ok(), Err(e) => fail(e) }).collect::<Vec<_>>())
    }

    /// `b` explains why the move from `cx.prev` to `cx.p` was bad: the material `b` names must net the mover a loss.
    fn explains(&mut self, b: &Reason, cx: &Cx) -> Verdict {
        let owner = cx.owner;
        let parts: Vec<&Reason> = match b { And(rs) => rs.iter().collect(), r => vec![r] };

        // Sign: the net material the parts name, for the owner (pawns), against the loss.
        let (mut v, mut material_named, mut other) = (0, false, false);
        for part in &parts {
            match part {
                Not(_) | Rated(_) => {}
                Allows(line, r) if mentions(r, "Mate") => { let _ = line; return ok(); }
                Allows(line, r) if mentions(r, "Material") => {
                    let Ok(end) = play(&cx.p, None, line) else { continue };
                    // Counted from before the move, so what the move itself takes counts.
                    let Some(n) = self.net(&cx.prev, &end, !owner) else { return unknown("no engine") };
                    v -= n;
                    material_named = true;
                }
                Instead(m, inner) if mentions(inner, "Material") => {
                    let (line, who): (&[Move_], Color) = match &**inner {
                        After(l, _) => (l, owner),
                        Allows(l, _) => (l, !owner),
                        _ => (&[], owner),
                    };
                    // Counted from before m, so what m itself takes counts.
                    let Ok(start) = play(&cx.prev, None, &[m]) else { continue };
                    let Ok(end) = play(&start, None, line) else { continue };
                    let Some(n) = self.net(&cx.prev, &end, who) else { return unknown("no engine") };
                    // The owner missed what it would have won, or would have lost only this.
                    if who == owner { v -= n } else { v += n }
                    material_named = true;
                }
                _ => other = true,
            }
        }
        if !material_named { return ok(); }
        let why = format!("the material it names nets the mover {v:+} pawns, not a loss");
        // A named gain for the mover cannot explain a loss, whatever else the reason says.
        if v < 0 { ok() } else if v > 0 || !other { fail(why) } else { unknown(why) }
    }

    /// The owner's annotated move (or m under `Instead(m, ..)`) and the position before it, when it is the last
    /// move played to reach `cx.p` (also under `Allows([], ..)`): line checks walk it as their first ply.
    /// None after the moves of an `After`/`Allows` line and on the before side of `Loses`.
    fn lead(cx: &Cx, owner: Color) -> Option<(Chess, Move)> {
        let b = cx.base.as_ref()?;
        let (last, rest) = cx.hist.split_last()?;
        let m = last.clone()?;
        let before = replay(&cx.start, rest)?;
        if before.board() != b.board() || before.turn() != b.turn() || before.turn() != owner { return None; }
        let after = before.clone().play(m.clone()).ok()?;
        if after.board() != cx.p.board() || after.turn() != cx.p.turn() { return None; }
        Some((before, m))
    }

    /// `Perpetual`: the engine's line from `p`, searched again at its end until `PERPETUAL_PLIES`: every owner
    /// move checks and a position repeats.
    fn perpetual(&mut self, cx: &Cx, owner: Color) -> Verdict {
        let p = &cx.p;
        let key = |q: &Chess| (q.board().clone(), q.turn());
        let (mut q, mut seen, mut plies) = (p.clone(), vec![key(p)], 0);
        // The annotated move is the line's first ply: it must check too, and a return to before it repeats.
        if let Some((before, m)) = Self::lead(cx, owner) {
            if !p.is_check() { return fail(format!("{} is no check", San::from_move(&before, m))); }
            seen.insert(0, key(&before));
            plies = 1;
        }
        while plies < PERPETUAL_PLIES {
            let Some((_, pv)) = self.eng.as_mut().map(|e| e.analyse(&q)) else { return unknown("no engine") };
            if pv.is_empty() { break; }
            for u in &pv {
                let Some(m) = u.parse::<UciMove>().ok().and_then(|u| u.to_move(&q).ok()) else { break };
                let next = q.clone().play(m.clone()).unwrap();
                if q.turn() == owner && !next.is_check() { return fail(format!("{} is no check", San::from_move(&q, m))); }
                q = next;
                plies += 1;
                if seen.contains(&key(&q)) { return ok(); }
                seen.push(key(&q));
                if plies >= PERPETUAL_PLIES { break; }
            }
        }
        fail(format!("no repetition within {PERPETUAL_PLIES} plies"))
    }

    /// `r`, with its squares following their pieces, after `plies` of the engine's line from `cx.p`.
    /// None without an engine line.
    fn along(&mut self, r: &Reason, cx: &Cx, plies: usize) -> Option<Verdict> {
        let p = &cx.p;
        let (_, pv) = self.eng.as_mut()?.analyse(p);
        let outer = TRACK.with(|t| *t.borrow());
        let mut map = outer.unwrap_or(std::array::from_fn(|i| Some(Square::new(i as u32))));
        let mut q = p.clone();
        for u in pv.iter().take(plies) {
            let Ok(m) = u.parse::<UciMove>().map_err(|_| ()).and_then(|u| u.to_move(&q).map_err(|_| ())) else { break };
            track(&mut map, &m);
            q = q.play(m).unwrap();
        }
        if q.board() == p.board() { return None; }
        TRACK.with(|t| *t.borrow_mut() = Some(map));
        GONE.set(false);
        let v = self.eval(r, &Cx::bare(q, p.clone(), cx.owner));
        let gone = GONE.replace(false);
        TRACK.with(|t| *t.borrow_mut() = outer);
        Some(if gone { fail("the piece is captured along the line") } else { v })
    }

    /// The pawn on `s` promotes within `PROMOTE_PLIES` of the engine's line from `p`.
    /// Where it promotes, if it does.
    fn promotes_along(&mut self, p: &Chess, s: Square) -> Option<Option<Square>> {
        let (_, pv) = self.eng.as_mut()?.analyse(p);
        let (mut q, mut at) = (p.clone(), s);
        for u in pv.iter().take(PROMOTE_PLIES) {
            let Some(m) = u.parse::<UciMove>().ok().and_then(|u| u.to_move(&q).ok()) else { break };
            if m.from() == Some(at) {
                if m.promotion().is_some() { return Some(Some(m.to())) }
                at = m.to();
            } else if m.to() == at { return Some(None) }
            q = q.play(m).unwrap();
        }
        Some(None)
    }

    /// The position after the first `n` moves of the engine's line from `p`.
    fn pv_end(&mut self, p: &Chess, n: usize) -> Option<Chess> {
        let (_, pv) = self.eng.as_mut()?.analyse(p);
        let mut q = p.clone();
        for u in pv.iter().take(n) {
            let Some(m) = u.parse::<UciMove>().ok().and_then(|u| u.to_move(&q).ok()) else { break };
            q = q.play(m).unwrap();
        }
        Some(q)
    }

    /// Net material `side` wins from `start` to `end` once the exchanges at `end` settle, in pawns.
    fn net(&mut self, start: &Chess, end: &Chess, side: Color) -> Option<i32> {
        let diff = |q: &Chess| material(q, side) - material(q, !side);
        Some(diff(&self.settle(end)?) - diff(start))
    }

    /// Fails when the threat's first move scores more than `THREAT_MARGIN` below the position as it stands.
    /// With the owner to move, that is its best move.
    fn threat_is_real(&mut self, p: &Chess, owner: Color, line: &[Move_]) -> Option<Verdict> {
        let first = *line.first()?;
        if first == "--" { return None; }
        let q = as_mover(p, owner)?;
        let now = self.owner_score(p, owner)?;
        let after = self.owner_score(&q.clone().play(san(&q, first).ok()?).ok()?, owner)?;
        (after < now - THREAT_MARGIN).then(|| fail(format!("{first} gains {:+} cp, no threat", after - now)))
    }

    /// Fails at the first move of `line` that loses more than `LINE_MARGIN` winning chances against
    /// its mover's best move: a line must be (nearly) the best play for both sides. Passes are skipped.
    fn line_is_played(&mut self, p: &Chess, line: &[Move_]) -> Option<Verdict> {
        let mut q = p.clone();
        for m in line {
            if *m == "--" { q = q.swap_turn().ok()?; continue; }
            let next = q.clone().play(san(&q, m).ok()?).ok()?;
            let (lost, top, best) = self.loss(&q, &next)?;
            let top = top.map_or("?".to_string(), |b| San::from_move(&q, b).to_string());
            let margin = LINE_MARGIN + (LINE_MARGIN_EDGE - LINE_MARGIN) * best.abs();
            if lost > margin {
                return Some(fail(format!("{m} is not the best play: {top} is {lost:.3} winning chances better (margin {margin:.2})")));
            }
            q = next;
        }
        None
    }

    /// Winning chances the move from `prev` to `next` loses against the engine's best move there, and
    /// that move. Both positions after a move are searched alike, so the best move itself loses 0.
    /// Also the best move's winning chances for the mover.
    fn loss(&mut self, prev: &Chess, next: &Chess) -> Option<(f64, Option<Move>, f64)> {
        let (s, pv) = self.eng.as_mut()?.analyse(prev);
        let Some(top) = pv.first().and_then(|u| u.parse::<UciMove>().ok()).and_then(|u| u.to_move(prev).ok()) else {
            return Some((0.0, None, chances(s)));
        };
        let best = prev.clone().play(top.clone()).ok()?;
        if best.board() == next.board() && best.turn() == next.turn() { return Some((0.0, Some(top), chances(s))); }
        let b = -self.eng.as_mut()?.analyse(&best).0;
        let n = -self.eng.as_mut()?.analyse(next).0;
        Some(((chances(b) - chances(n)).max(0.0), Some(top), chances(b)))
    }

    /// The owner's best score in `prev` over the moves that do not lead to `not`.
    fn best_other(&mut self, prev: &Chess, not: &Chess, owner: Color) -> Option<i32> {
        let mut best = None;
        for m in prev.legal_moves() {
            let q = prev.clone().play(m).unwrap();
            if q.board() == not.board() && q.turn() == not.turn() { continue; }
            let s = self.owner_score(&q, owner)?;
            best = Some(best.map_or(s, |b: i32| b.max(s)));
        }
        best
    }

    /// The move from `cx.prev` to `cx.p` against the engine's best there; `Stronger` and `Weaker` against the played move.
    fn rated(&mut self, a: &Annotation, cx: &Cx) -> Verdict {
        if self.eng.is_none() { return unknown("no engine"); }
        // Lichess's grading: the eval before the move against the eval after it. The annotator
        // (annotate/server.py) searches the same way, so both give the same grade.
        let Some((_, pv)) = self.eng.as_mut().map(|e| e.analyse(&cx.prev)) else { return unknown("no engine") };
        let top_move = pv.first().and_then(|u| u.parse::<UciMove>().ok()).and_then(|u| u.to_move(&cx.prev).ok());
        let top = top_move.as_ref().is_some_and(|m| cx.prev.clone().play(m.clone()).unwrap().board() == cx.p.board());
        let (Some(best), Some(now)) = (self.owner_score(&cx.prev, cx.owner), self.owner_score(&cx.p, cx.owner)) else { return unknown("no engine") };
        let gap = best - now;
        let lost = chances(best) - chances(now);
        let name = top_move.map_or("?".to_string(), |m| San::from_move(&cx.prev, m).to_string());
        let why = format!("{:+.2} → {:+.2}, {lost:.3} winning chances lost (best {name})", best as f64 / 100.0, now as f64 / 100.0);
        match a {
            Annotation::Best => check(top || gap <= BEST_MARGIN, why),
            Annotation::Good | Annotation::Dubious | Annotation::Mistake | Annotation::Blunder => {
                let (lo, hi, name) = match a {
                    Annotation::Good => (f64::NEG_INFINITY, INACCURACY, "a good move loses less than"),
                    Annotation::Dubious => (INACCURACY, MISTAKE, "an inaccuracy loses"),
                    Annotation::Mistake => (MISTAKE, BLUNDER, "a mistake loses"),
                    _ => (BLUNDER, f64::INFINITY, "a blunder loses"),
                };
                let (lo, hi) = (lo - GRADE_MARGIN, hi + GRADE_MARGIN);
                let range = if lo.is_infinite() { format!("{hi:.2}") } else if hi.is_infinite() { format!("{lo:.2} or more") } else { format!("{lo:.2}-{hi:.2}") };
                check(lo <= lost && lost < hi, format!("{why}; {name} {range}"))
            }
            Annotation::Brilliant => {
                // One of the owner's pieces (not a pawn or the king) hangs that did not hang there before the move.
                let loose = |q: &Chess| (q.board().by_color(cx.owner) & !q.board().pawns() & !q.board().kings())
                    .into_iter().filter(|s| would_hang(q, *s)).collect::<Bitboard>();
                let sac = (loose(&cx.p) & !loose(&cx.prev)).any();
                if !(top || gap <= BEST_MARGIN) { fail(format!("not the best move: {why}")) }
                else if !sac { fail("sacrifices no piece") }
                else if let Some(alt) = self.best_other(&cx.prev, &cx.p, cx.owner)
                    && chances(alt) >= BRILLIANT_WINNING { fail(format!("already winning without it ({alt:+} cp)")) }
                else if chances(now) < -BRILLIANT_BAD { fail(format!("bad after it ({now:+} cp)")) }
                else { ok() }
            }
            Annotation::Stronger | Annotation::Weaker => {
                let played = self.played.clone();
                let Some(z) = self.owner_score(&played, cx.owner) else { return unknown("no engine") };
                let d = now - z;
                check(if matches!(a, Annotation::Stronger) { d > 0 } else { d < 0 }, format!("{d:+} cp vs the played move"))
            }
            Annotation::Interesting => unknown("judgement not checked"),
        }
    }

    /// Along the whole engine line (the annotated move first, when it is the owner's) the owner's first forcing
    /// move comes before the opponent's, or from the owner's first forcing move on the owner plays more forcing
    /// moves than the opponent. Forcing: see `forcing`. A check that gains nothing is hollow when the reply is
    /// forcing; otherwise it counts only if the same side's next move is forcing. The annotated move must itself be
    /// forcing, not a hollow check, and followed by a forcing move of the owner's: a quiet move can't take the initiative.
    fn initiative(&mut self, cx: &Cx, owner: Color) -> Verdict {
        let lead = Self::lead(cx, owner);
        let Some(eng) = self.eng.as_mut() else { return unknown("no engine") };
        let (_, pv) = eng.analyse(&cx.p);
        // The move before the line's first ply, for recaptures.
        let (mut q, mut last) = match &lead {
            Some((b, _)) => (b.clone(), None),
            None => (cx.p.clone(), cx.hist.last().cloned().flatten().filter(|_| replay(&cx.start, &cx.hist).is_some_and(|z| z.board() == cx.p.board()))),
        };
        // Per ply: the owner's, forcing, check, gains something besides the check.
        let (mut own, mut f, mut chk, mut gain) = (vec![], vec![], vec![], vec![]);
        let line = lead.as_ref().map(|(_, m)| m.clone()).into_iter().map(Ok).chain(pv.into_iter().map(Err));
        for u in line {
            let m = match u { Ok(m) => m, Err(u) => match u.parse::<UciMove>().ok().and_then(|u| u.to_move(&q).ok()) { Some(m) => m, None => break } };
            let next = q.clone().play(m.clone()).unwrap();
            let rec = m.is_capture() && last.as_ref().is_some_and(|l: &Move| l.is_capture() && l.to() == m.to());
            own.push(q.turn() == owner);
            f.push(forcing(&q, &m, &next, rec));
            chk.push(next.is_check());
            gain.push((m.is_capture() && !rec) || hits(&q, &m, &next));
            last = Some(m);
            q = next;
        }
        let n = f.len();
        let raw = f.clone();
        for i in (0..n).rev() {
            if !chk[i] || gain[i] { continue; }
            if i + 1 < n && f[i + 1] { f[i] = false } else if i + 2 < n { f[i] = f[i + 2] }
        }
        if let Some((before, m)) = &lead {
            let s = San::from_move(before, m.clone());
            if !raw[0] { return fail(format!("{s} is quiet: a quiet move can't itself take the initiative")); }
            if chk[0] && !gain[0] && ((n > 1 && f[1]) || (n > 2 && !f[2])) { return fail(format!("{s} is a hollow check")); }
            if n > 2 && !f[2] { return fail(format!("{s} is hollow: the owner's next move is not forcing")); }
        }
        let Some(first) = (0..n).find(|&i| f[i]) else { return fail(format!("no forcing move in {n} plies")) };
        if own[first] { return Verdict { v: V::Holds, why: format!("owner forces first, ply {} of {n}", first + 1) }; }
        let Some(start) = (0..n).find(|&i| own[i] && f[i]) else { return fail(format!("opponent forces first, ply {} of {n}; the owner never forces", first + 1)) };
        let me = (start..n).filter(|&i| own[i] && f[i]).count();
        let them = (start..n).filter(|&i| !own[i] && f[i]).count();
        let why = format!("opponent forces first, ply {} of {n}; from the owner's first forcing move: {me} against {them}", first + 1);
        Verdict { v: if me > them { V::Holds } else { V::Fails }, why }
    }
}

/// Where the moved piece lands (the king's square for a castle).
fn dest(q: &Chess, m: &Move) -> Square {
    match m {
        Move::Castle { king, rook } => (if rook.file() > king.file() { CastlingSide::KingSide } else { CastlingSide::QueenSide }).king_to(q.turn()),
        _ => m.to(),
    }
}

/// The moved piece attacks an opponent non-king piece bigger than itself or undefended.
fn hits(q: &Chess, m: &Move, next: &Chess) -> bool {
    let me = q.turn();
    (next.board().attacks_from(dest(q, m)) & next.board().by_color(!me)).into_iter().any(|t| {
        let r = next.board().role_at(t).unwrap();
        r != Role::King && (val(r) > val(m.role()) || attackers(next, t, !me).is_empty())
    })
}

/// Forcing for `Initiative`: a check; a capture, unless it takes back on the square the last move took on; an attack
/// on a bigger or undefended non-king piece; aiming the piece at more of the opponent king's squares (its square and
/// the ring) than before; a pawn attacking a pawn.
fn forcing(q: &Chess, m: &Move, next: &Chess, recapture: bool) -> bool {
    if next.is_check() || (m.is_capture() && !recapture) || hits(q, m, next) { return true; }
    let me = q.turn();
    let to = dest(q, m);
    if let Some(k) = q.board().king_of(!me) {
        let ring = attacks::king_attacks(k) | Bitboard::from_square(k);
        let before = m.from().map_or(0, |s| (q.board().attacks_from(s) & ring).count());
        if (next.board().attacks_from(to) & ring).count() > before { return true; }
    }
    m.role() == Role::Pawn && (next.board().attacks_from(to) & next.board().pawns() & next.board().by_color(!me)).any()
}

// --------------------------------------------------------------------------- report

#[cfg(not(target_arch = "wasm32"))]
fn engine() -> Option<Engine> {
    let path = std::env::var("STOCKFISH").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../vendor/stockfish/stockfish-ubuntu-x86-64-avx2").to_string()
    });
    let nodes = std::env::var("NODES").ok().and_then(|n| n.parse().ok()).unwrap_or(300_000);
    Engine::start(&path, nodes)
}



fn mark(v: V) -> &'static str { match v { V::Holds => "holds", V::Fails => "FAILS", V::Unknown => "unknown" } }

#[cfg(not(target_arch = "wasm32"))]
/// Every example on its real move.
pub fn report() {
    let eng = engine();
    if eng.is_none() { eprintln!("no engine: tier-2 reasons will be unknown"); }
    let mut ev = Ev::new(eng);
    let mut per: HashMap<String, [usize; 3]> = HashMap::new();
    let mut top = [0usize; 3];
    let idx = |v: V| match v { V::Holds => 0, V::Fails => 1, V::Unknown => 2 };

    for (i, ex) in examples().iter().enumerate() {
        let root: Chess = Fen::from_ascii(ex.fen.as_bytes()).unwrap().into_position(CastlingMode::Standard).unwrap();
        let mv = san(&root, ex.mov).unwrap();
        let v = ev.run(&root, &mv, &ex.reason);
        if QUESTIONABLE.contains(&ex.fen) {
            println!("\n#{i} {} ... questionable, not counted: {} {}", ex.mov, mark(v.v), v.why);
            continue;
        }
        top[idx(v.v)] += 1;
        println!("\n#{i} {} ... {}: {}", ex.mov, mark(v.v), v.why);
        for (n, c) in &ev.claims {
            per.entry(n.clone()).or_default()[idx(c.v)] += 1;
            if c.v != V::Holds { println!("    {n}: {} {}", mark(c.v), c.why); }
        }
    }

    println!("\nWhole reasons: {} holds / {} fails / {} unknown", top[0], top[1], top[2]);
    println!("\n{:<12} {:>6} {:>6} {:>8}", "reason", "holds", "fails", "unknown");
    let mut names: Vec<_> = per.into_iter().collect();
    names.sort();
    for (n, c) in names {
        println!("{n:<12} {:>6} {:>6} {:>8}", c[0], c[1], c[2]);
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// JSON lines `{fen, mov, dsl}` on stdin → `{english, v, why, claims}` or `{error}` per line, as the browser shows them.
pub fn check_lines() {
    use serde_json::{Value, json};
    let mut ev = Ev::new(engine());
    let mark = |v: V| match v { V::Holds => "holds", V::Fails => "fails", V::Unknown => "unknown" };
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() { continue; }
        let out = (|| -> Result<Value, String> {
            let q: Value = serde_json::from_str(&line).map_err(|e| format!("input: {e}"))?;
            let (fen, mov, dsl) = (q["fen"].as_str().unwrap_or(""), q["mov"].as_str().unwrap_or(""), q["dsl"].as_str().unwrap_or(""));
            let root: Chess = Fen::from_ascii(fen.trim().as_bytes()).map_err(|e| format!("FEN: {e}"))?
                .into_position(CastlingMode::Standard).map_err(|e| format!("FEN: {e}"))?;
            let m: Move_ = Box::leak(mov.trim().to_string().into_boxed_str());
            let mv = san(&root, m).map_err(|_| format!("{mov} is not legal here"))?;
            let reason = crate::text::parse(dsl)?;
            let english = crate::render::render_reason(&root, m, &reason);
            let v = ev.run(&root, &mv, &reason);
            let claims: Vec<Value> = ev.claims.iter().map(|(n, c)| json!({ "name": n, "v": mark(c.v), "why": c.why })).collect();
            Ok(json!({ "english": english, "v": mark(v.v), "why": v.why, "claims": claims }))
        })().unwrap_or_else(|e| json!({ "error": e }));
        println!("{out}");
    }
}
