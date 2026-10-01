//! Evaluates a `Reason` against the board.
//!
//! Tier 1: board facts, shakmaty only. Tier 2: Stockfish over UCI (`STOCKFISH`, default the vendored
//! binary). Tier 3: heuristics ported from commentary_gen/board_tools.py and positional.py.
//!
//! Caveats:
//! - `Because(a, b)` holds when `a` and `b` both hold. That `b` causes `a` is not checked.
//! - `Outweighs(a, b)` holds when both hold. The weighing is not checked.
//! - `Quality` is the eval drop from removing the piece, minus the median drop for its kind (`baseline.tsv`); `Better` compares that drop.
//! - `Eventually(r)` holds when `r` fails now and holds after the engine's line, `EVENTUALLY_PLIES` ahead.
//! - `Only(r)` and `NoMove(r)` ignore moves that achieve `r` but score `ONLY_MARGIN` or more below the played move.
//! - `WinMaterial` follows the engine's line (`SETTLE_NODES`) until the exchanges settle. `Sacrifice` uses static exchange evaluation.
//! - `Prevents`, `Loses`, `Gains` and `Concedes` compare with the position before the enclosing move or line.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::uci::UciMove;
use shakmaty::{
    Bitboard, CastlingMode, Chess, Color, EnPassantMode, Move, Piece, Position, Rank, Role, Square, attacks,
};

use crate::dsl_claude::{Annotation, Grade, Move_, Pieces, QUESTIONABLE, Reason, Reason::*, Region, Sq, examples};

/// Centipawns an alternative may trail the played move and still count against `Only` and `NoMove`.
const ONLY_MARGIN: i32 = 30;

// Thresholds in centipawns, set leniently: a comment's claim should hold unless the board clearly disagrees.
const WIN: i32 = 200;
const DRAW: i32 = 100;
/// A threat's first move may score this far below the position as it stands.
const THREAT_MARGIN: i32 = 50;
/// `Rated(Best)`: the engine's move, or this close to it.
const BEST_MARGIN: i32 = 10;
/// `Rated`: Good within this of the best move; Dubious, Mistake and Blunder more than this below it.
const RATED_BAD: i32 = 50;

/// Search size for settling exchanges in `WinMaterial`.
const SETTLE_NODES: u64 = 1_000_000;

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

fn val(r: Role) -> i32 {
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

fn material(p: &Chess, c: Color) -> i32 { p.board().by_color(c).into_iter().map(|s| val(p.board().role_at(s).unwrap())).sum() }
fn count(p: &Chess, c: Color, r: Role) -> usize { (p.board().by_color(c) & p.board().by_role(r)).count() }
fn attackers(p: &Chess, s: Square, c: Color) -> Bitboard { p.board().attacks_to(s, c, p.board().occupied()) }
fn pawns(p: &Chess, c: Color) -> Bitboard { p.board().pawns() & p.board().by_color(c) }
fn fen(p: &Chess) -> String { Fen::from_position(p, EnPassantMode::Legal).to_string() }

/// Pass if `c` is not to move. None when that side is in check.
fn as_mover(p: &Chess, c: Color) -> Option<Chess> {
    if p.turn() == c { Some(p.clone()) } else { p.clone().swap_turn().ok() }
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
    as_mover(p, !c).is_some_and(|b| see_square(&b, s) > 0)
}

/// Best net gain for the side to move from any capture.
fn best_capture(p: &Chess) -> i32 { p.capture_moves().iter().map(|m| see(p, m)).max().unwrap_or(0).max(0) }

fn safe_mobility(p: &Chess, s: Square) -> usize {
    let Some(c) = p.board().color_at(s) else { return 0 };
    as_mover(p, c).map_or(0, |b| b.legal_moves().iter().filter(|m| m.from() == Some(s) && see(&b, m) >= 0).count())
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

/// The squares of `r` for `owner`. Fails for `KingWing` while that king is on the d- or e-file.
fn region(r: &Region, p: &Chess, owner: Color) -> Result<Vec<Square>, String> {
    let files = |fs: std::ops::RangeInclusive<u32>| Bitboard::FULL.into_iter().filter(|s| fs.contains(&(s.file() as u32))).collect::<Vec<_>>();
    let box_ = |lo: u32, hi: u32| Bitboard::FULL.into_iter().filter(|s| (lo..=hi).contains(&(s.file() as u32)) && (lo..=hi).contains(&(s.rank() as u32))).collect();
    let side = |opponent: bool| if opponent { !owner } else { owner };
    Ok(match r {
        Region::Squares(ss) => ss.iter().map(|s| sq(s)).collect(),
        Region::Kingside => files(5..=7),
        Region::Queenside => files(0..=2),
        Region::Centre => box_(3, 4),
        Region::GreaterCentre => box_(2, 5),
        Region::Half { opponent } => {
            let c = side(*opponent);
            Bitboard::FULL.into_iter().filter(|s| rank(*s, c) <= 3).collect()
        }
        Region::KingWing { opponent } => {
            let k = p.board().king_of(side(*opponent)).ok_or("no king")?;
            match k.file() as u32 {
                5..=7 => files(5..=7),
                0..=2 => files(0..=2),
                _ => return Err(format!("king on {k} is on neither wing")),
            }
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

pub struct Engine {
    _child: Child,
    inp: ChildStdin,
    out: BufReader<ChildStdout>,
    nodes: u64,
    cache: HashMap<String, (i32, Vec<String>)>,
}

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
        let (mut score, mut pv) = (0, vec![]);
        for l in self.until("bestmove") {
            let t: Vec<&str> = l.split_whitespace().collect();
            let Some(i) = t.iter().position(|x| *x == "score") else { continue };
            if t.contains(&"upperbound") || t.contains(&"lowerbound") { continue; }
            let n: i32 = t[i + 2].parse().unwrap_or(0);
            score = if t[i + 1] == "mate" { if n > 0 { 10000 } else { -10000 } } else { n };
            if let Some(j) = t.iter().position(|x| *x == "pv") { pv = t[j + 1..].iter().map(|s| s.to_string()).collect(); }
        }
        self.cache.insert(key, (score, pv.clone()));
        (score, pv)
    }
}

impl Drop for Engine {
    fn drop(&mut self) { let _ = writeln!(self.inp, "quit"); }
}

// --------------------------------------------------------------------------- evaluator

/// Where a reason is evaluated: the position, the one before the enclosing move or line, and whose reason it is.
#[derive(Clone)]
struct Cx {
    p: Chess,
    prev: Chess,
    owner: Color,
}

impl Cx {
    fn at(&self, p: Chess, owner: Color) -> Cx { Cx { p, prev: self.p.clone(), owner } }
}

pub struct Ev {
    eng: Option<Engine>,
    /// Position before the move: `Trade` names pieces by their squares here.
    root: Chess,
    /// Position after the played move: `Stronger` and `Weaker` compare with it.
    played: Chess,
    /// Nodes that the comment claims hold, with their verdicts. Nodes under `Not`, `Prevents` and the like are not claims.
    pub claims: Vec<(String, Verdict)>,
    quiet: usize,
    /// Median `worth` per role over corpus positions (see `baseline`), indexed by `Role as usize`.
    typical: [i32; 7],
}

fn name(r: &Reason) -> String { format!("{r:?}").split(['(', ' ', '{']).next().unwrap().to_string() }

impl Ev {
    pub fn new(eng: Option<Engine>) -> Ev {
        let mut typical = [0; 7];
        for r in Role::ALL { typical[r as usize] = cp(r); }
        if let Ok(t) = std::fs::read_to_string(BASELINE) {
            for l in t.lines() {
                let f: Vec<&str> = l.split('\t').collect();
                if let (Some(r), Some(Ok(m))) = (f.first().and_then(|c| c.chars().next()).and_then(Role::from_char), f.get(1).map(|m| m.parse())) {
                    typical[r as usize] = m;
                }
            }
        }
        Ev { eng, root: Chess::default(), played: Chess::default(), claims: vec![], quiet: 0, typical }
    }

    /// Evaluate `reason` for the side that plays `mov` in `root`.
    pub fn run(&mut self, root: &Chess, mov: &Move, reason: &Reason) -> Verdict {
        self.root = root.clone();
        self.claims.clear();
        self.played = root.clone().play(mov.clone()).unwrap();
        let cx = Cx { p: self.played.clone(), prev: root.clone(), owner: root.turn() };
        self.eval(reason, &cx)
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
        let (p, owner) = (&cx.p, cx.owner);
        let b = p.board();
        let mine = |s: Square| b.color_at(s) == Some(owner);
        let theirs = |s: Square| b.color_at(s) == Some(!owner);
        match r {
            // ---------------------------------------------------------------- combinators
            And(rs) => all(rs.iter().map(|r| self.eval(r, cx)).collect::<Vec<_>>()),
            // Each alternative alone is not claimed, so they are not recorded as claims.
            Either(rs) => self.quietly(|ev| any(rs.iter().map(|r| ev.eval(r, cx)).collect(), "none holds")),
            Threatens(line, r) => match play(p, Some(owner), line) {
                Err(e) => fail(e),
                Ok(q) => {
                    let v = self.eval(r, &cx.at(q, owner));
                    if v.v == V::Holds { self.threat_is_real(p, owner, line).unwrap_or(v) } else { v }
                }
            },
            After(line, r) => match play(p, None, line) {
                Err(e) => fail(e),
                Ok(q) => self.eval(r, &cx.at(q, owner)),
            },
            Allows(line, r) => match play(p, None, line) {
                Err(e) => fail(e),
                Ok(q) => self.eval(r, &cx.at(q, !owner)),
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
                        let alt = ev.eval(r, &Cx { p: q.clone(), prev: cx.prev.clone(), owner });
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
                    match ev.eval(r, &Cx { p: q.clone(), prev: cx.prev.clone(), owner }).v {
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
                Ok(q) => self.eval(r, &Cx { p: q, prev: cx.prev.clone(), owner }),
            },
            Sacrifice(roles, r) => {
                let need: i32 = roles.iter().map(|r| val(*r)).sum();
                let lost = material(&cx.prev, owner) - material(p, owner);
                let hang = if p.turn() != owner { best_capture(p) } else { 0 };
                let s = check(lost + hang >= need, format!("gives up {} of {need}", lost + hang));
                all([s, self.eval(r, cx)])
            }
            // Simplify is accepted until simplicity is checked, negated too.
            Not(r) if matches!(&**r, Simplify) => Verdict { v: V::Holds, why: "simpler unchecked".into() },
            Not(r) => not(self.quietly(|ev| ev.eval(r, cx))),
            Because(a, b) => {
                let v = all([self.eval(a, cx), self.eval(b, cx)]);
                if v.v == V::Holds { Verdict { v: V::Holds, why: "both hold; cause unchecked".into() } } else { v }
            }
            But(a, b) => {
                let v = all([self.eval(a, cx), self.eval(b, cx)]);
                if v.v == V::Holds { Verdict { v: V::Holds, why: "both hold; contrast unchecked".into() } } else { v }
            }
            Outweighs(a, b2) => {
                let vb = self.eval(b2, &Cx { owner: !owner, ..cx.clone() });
                let v = all([self.eval(a, cx), vb]);
                if v.v == V::Holds { Verdict { v: V::Holds, why: "both hold; weighing unchecked".into() } } else { v }
            }
            Permanent(r) => {
                let v = self.eval(r, cx);
                // Pawn structure cannot be undone: pawns do not move backward.
                let structural = matches!(&**r, Isolated(_) | Doubled(_) | Backward(_) | Hole(_) | Weakened(_));
                if v.v == V::Holds && !structural { unknown("permanence unchecked") } else { v }
            }
            Balances(a, b2) => {
                let vb = self.eval(b2, &Cx { owner: !owner, ..cx.clone() });
                let v = all([self.eval(a, cx), vb]);
                if v.v == V::Holds { Verdict { v: V::Holds, why: "both hold; balance unchecked".into() } } else { v }
            }
            More(_) => unknown("comparison not modelled"),
            Degree(_, r) => {
                let v = self.eval(r, cx);
                if v.v == V::Holds { Verdict { v: V::Holds, why: "holds; degree unchecked".into() } } else { v }
            }
            Eventually(r) => {
                if self.quietly(|ev| ev.eval(r, cx)).v == V::Holds { return fail("already holds"); }
                let Some(eng) = self.eng.as_mut() else { return unknown("no engine") };
                let (_, pv) = eng.analyse(p);
                let outer = TRACK.with(|t| *t.borrow());
                let mut map = outer.unwrap_or(std::array::from_fn(|i| Some(Square::new(i as u32))));
                let mut q = p.clone();
                for u in pv.iter().take(EVENTUALLY_PLIES) {
                    let Ok(m) = u.parse::<UciMove>().map_err(|_| ()).and_then(|u| u.to_move(&q).map_err(|_| ())) else { break };
                    track(&mut map, &m);
                    q = q.play(m).unwrap();
                }
                if q.board() == p.board() { return unknown("no engine line"); }
                TRACK.with(|t| *t.borrow_mut() = Some(map));
                GONE.set(false);
                let v = self.eval(r, &cx.at(q, owner));
                let gone = GONE.replace(false);
                TRACK.with(|t| *t.borrow_mut() = outer);
                if gone { fail("the piece is captured along the line") } else { v }
            }
            Faster(..) => unknown("race not modelled"),
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
                    Ok(q) => self.eval(r, &Cx { p: q, ..cx.clone() }),
                }
            }

            // ---------------------------------------------------------------- tier 1: board facts
            Mate => check(p.is_checkmate(), "no mate"),
            Check => check(p.is_check(), "no check"),
            Material(n) => {
                let d = material(p, owner) - material(p, !owner);
                check(d == *n as i32, format!("material {d:+}"))
            }
            WinMaterial(roles) => {
                let before = cx.prev.clone();
                let taken = roles.iter().all(|r| count(p, !owner, *r) < count(&before, !owner, *r));
                let diff = |q: &Chess| material(q, owner) - material(q, !owner);
                let Some(settled) = self.settle(p) else { return unknown("no engine") };
                let net = diff(&settled) - diff(&before);
                check(taken && net > 0, format!("taken {taken}, net {net:+} once the exchanges settle"))
            }
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
                check(mine(a) && theirs(f) && theirs(k) && role.is_some_and(|r| slider_line(r, a, k))
                    && between == Bitboard::from(f), format!("no pin {by}-{front}-{behind}"))
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
            Pressure(r) => {
                let ss = match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss };
                if !matches!(r, Region::Squares(_)) {
                    let aimed = |c: Color| ss.iter().filter(|s| attackers(p, **s, c).any()).count();
                    return check(aimed(owner) > aimed(!owner), format!("aims at {} squares, opponent at {}", aimed(owner), aimed(!owner)));
                }
                all(ss.into_iter().map(|s| check(attackers(p, s, owner).any(), format!("nothing on {s}"))).collect::<Vec<_>>())
            }
            Outnumber(r) => {
                let ss = match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss };
                let n = |c: Color, s: Square| attackers(p, s, c).count();
                if !matches!(r, Region::Squares(_)) {
                    let (own, opp) = (ss.iter().filter(|s| n(owner, **s) > n(!owner, **s)).count(), ss.iter().filter(|s| n(!owner, **s) > n(owner, **s)).count());
                    return check(own > opp, format!("outnumbers on {own} squares, outnumbered on {opp}"));
                }
                all(ss.into_iter().map(|s| check(n(owner, s) > n(!owner, s), format!("{s}: {} vs {}", n(owner, s), n(!owner, s)))).collect::<Vec<_>>())
            }
            Bound { piece, duties } => {
                if !theirs(sq(piece)) { return fail(format!("{piece} is not the opponent's")); }
                all(duties.iter().map(|d| {
                    let (own, opp) = (attackers(p, sq(d), owner).count(), attackers(p, sq(d), !owner).count());
                    check(b.attacks_from(sq(piece)).contains(sq(d)) && own >= opp, format!("{piece} not tied to {d}: {own} vs {opp}"))
                }).collect::<Vec<_>>())
            }
            Decoy { from, to } => {
                let lured = as_mover(p, !owner).is_some_and(|q| q.legal_moves().iter().any(|m| m.from() == Some(sq(from)) && m.to() == sq(to)));
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
                let can = as_mover(p, owner).is_some_and(|q| q.legal_moves().iter().any(|m| matches!(m, Move::Castle { .. })));
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
            Promote(s) => {
                let s = sq(s);
                let can = as_mover(p, owner).is_some_and(|q| q.legal_moves().iter().any(|m| m.from() == Some(s) && m.promotion().is_some()));
                check(mine(s) && rank(s, owner) == Rank::Seventh as u32 && can, format!("{s} cannot promote"))
            }
            Enables(line) => self.enables(p, owner, line),
            Options(ms) => all(ms.iter().map(|m| self.enables(p, owner, &[m])).collect::<Vec<_>>()),
            SameColour(ss) => check(ss.iter().all(|s| sq(s).is_light() == sq(ss[0]).is_light()), "mixed colours"),

            // ---------------------------------------------------------------- tier 2: engine
            Win => match self.owner_score(p, owner) {
                None => unknown("no engine"),
                Some(s) => check(s >= WIN, format!("eval {s:+}")),
            },
            Draw => match self.owner_score(p, owner) {
                None => unknown("no engine"),
                Some(s) => check(s.abs() <= DRAW, format!("eval {s:+}")),
            },
            // Needs the stuck side to pass repeatedly, not once.
            Zugzwang => unknown("repeated passing not modelled"),
            Simplify => Verdict { v: V::Holds, why: "simpler unchecked".into() },
            Initiative => self.initiative(p, owner),
            Tempo(_) => unknown("tempo count not modelled"),

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
            Better { pieces, than } => {
                let worth = |ss: &Vec<Sq>| ss.iter().filter_map(|s| b.role_at(sq(s))).map(cp).sum::<i32>();
                let mob = |ss: &Vec<Sq>| ss.iter().map(|s| safe_mobility(p, sq(s))).sum::<usize>();
                let sum = |ev: &mut Ev, ss: &Vec<Sq>| ss.iter().map(|s| ev.worth(p, sq(s))).sum::<Option<i32>>();
                if let (Some(a), Some(z)) = (sum(self, pieces), sum(self, than)) {
                    return check(a > z, format!("worth {a} vs {z} cp"));
                }
                let (a, z) = (worth(pieces), worth(than));
                if a != z { check(a > z, format!("{a} vs {z} cp")) } else { check(mob(pieces) > mob(than), format!("mobility {} vs {}", mob(pieces), mob(than))) }
            }
            Quality(s, g) => self.quality(p, sq(s), g),
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

    /// For `who`, the reason held before and does not hold now, or with `gained` the other way round.
    fn before_after(&mut self, r: &Reason, cx: &Cx, who: Color, gained: bool) -> Verdict {
        let before = self.eval(r, &Cx { p: cx.prev.clone(), prev: cx.prev.clone(), owner: who });
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
        if self.eval(r, &Cx { p: p.clone(), prev: p.clone(), owner: who }).v == V::Holds { return ok(); }
        let Some(q) = as_mover(p, who) else { return fail("cannot pass in check") };
        let mut undecided = false;
        for m in q.legal_moves() {
            let next = q.clone().play(m.clone()).unwrap();
            match self.eval(r, &Cx { p: next, prev: q.clone(), owner: who }).v {
                V::Holds => return Verdict { v: V::Holds, why: format!("{}", San::from_move(&q, m)) },
                V::Unknown => undecided = true,
                V::Fails => {}
            }
        }
        if undecided { unknown("some moves undecided") } else { fail("no move gets it") }
    }

    /// How much worse the position gets for the piece's owner without the piece.
    /// None without an engine, for a king, or when the board without it is illegal.
    fn worth(&mut self, p: &Chess, s: Square) -> Option<i32> {
        let c = p.board().color_at(s)?;
        if p.board().role_at(s)? == Role::King { return None; }
        let with = self.owner_score(p, c)?;
        let mut setup = p.to_setup(EnPassantMode::Legal);
        setup.board.discard_piece_at(s);
        let q = setup.position::<Chess>(CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights()).ok()?;
        Some(with - self.owner_score(&q, c)?)
    }

    /// Good: the piece is worth more than a typical piece of its kind. Bad: less.
    fn quality(&mut self, p: &Chess, s: Square, g: &Grade) -> Verdict {
        let Some(role) = p.board().role_at(s) else { return fail(format!("{s} empty")) };
        let Some(w) = self.worth(p, s) else { return unknown("no engine, or illegal without the piece") };
        let q = w - self.typical[role as usize];
        match g {
            Grade::Good => check(q > 0, format!("{s}: {q:+} cp vs a typical {role:?}")),
            Grade::Bad => check(q < 0, format!("{s}: {q:+} cp vs a typical {role:?}")),
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
        let Some(mut q) = as_mover(p, owner) else { return fail("owner in check, cannot pass") };
        for m in line {
            if *m == "--" { q = match q.swap_turn() { Ok(x) => x, Err(_) => return fail("cannot pass") }; continue; }
            let mv = match san(&q, m) { Ok(x) => x, Err(e) => return fail(e) };
            q = q.play(mv).unwrap();
        }
        ok()
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

    /// The move from `cx.prev` to `cx.p` against the engine's best there; `Stronger` and `Weaker` against the played move.
    fn rated(&mut self, a: &Annotation, cx: &Cx) -> Verdict {
        let Some(eng) = self.eng.as_mut() else { return unknown("no engine") };
        let (best, pv) = eng.analyse(&cx.prev);
        let best = if cx.prev.turn() == cx.owner { best } else { -best };
        let top = pv.first().and_then(|u| u.parse::<UciMove>().ok()).and_then(|u| u.to_move(&cx.prev).ok())
            .is_some_and(|m| cx.prev.clone().play(m).unwrap().board() == cx.p.board());
        let Some(now) = self.owner_score(&cx.p, cx.owner) else { return unknown("no engine") };
        let gap = best - now;
        let why = format!("{gap} cp below best");
        match a {
            Annotation::Best => check(top || gap <= BEST_MARGIN, why),
            Annotation::Good => check(gap <= RATED_BAD, why),
            Annotation::Dubious | Annotation::Mistake | Annotation::Blunder => check(gap > RATED_BAD, why),
            Annotation::Stronger | Annotation::Weaker => {
                let played = self.played.clone();
                let Some(z) = self.owner_score(&played, cx.owner) else { return unknown("no engine") };
                let d = now - z;
                check(if matches!(a, Annotation::Stronger) { d > 0 } else { d < 0 }, format!("{d:+} cp vs the played move"))
            }
            Annotation::Interesting | Annotation::Brilliant => unknown("judgement not checked"),
        }
    }

    /// Two of the owner's first three moves in the engine line are checks, captures or attacks on a bigger or loose piece.
    fn initiative(&mut self, p: &Chess, owner: Color) -> Verdict {
        let Some(eng) = self.eng.as_mut() else { return unknown("no engine") };
        let (_, pv) = eng.analyse(p);
        let mut q = p.clone();
        let (mut mine, mut forcing) = (0, 0);
        for u in pv {
            let Ok(m) = u.parse::<UciMove>().map_err(|_| ()).and_then(|u| u.to_move(&q).map_err(|_| ())) else { break };
            let next = q.clone().play(m.clone()).unwrap();
            if q.turn() == owner {
                let hits = next.board().attacks_from(m.to()) & next.board().by_color(!owner);
                let threat = hits.into_iter().any(|t| {
                    let r = next.board().role_at(t).unwrap();
                    r != Role::King && (val(r) > val(m.role()) || attackers(&next, t, !owner).is_empty())
                });
                if m.is_capture() || next.is_check() || threat { forcing += 1; }
                mine += 1;
                if mine == 3 { break; }
            }
            q = next;
        }
        check(forcing >= 2, format!("{forcing} of {mine} forcing"))
    }
}

// --------------------------------------------------------------------------- report

fn engine() -> Option<Engine> {
    let path = std::env::var("STOCKFISH").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../vendor/stockfish/stockfish-ubuntu-x86-64-avx2").to_string()
    });
    let nodes = std::env::var("NODES").ok().and_then(|n| n.parse().ok()).unwrap_or(300_000);
    Engine::start(&path, nodes)
}

const BASELINE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/baseline.tsv");

/// Median `worth` per role over every 8th position of the commentary training set, written to `baseline.tsv`.
pub fn baseline() {
    let data = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../commentary_gen/data/train.jsonl")).unwrap();
    let mut ev = Ev::new(Some(engine().expect("baseline needs the engine")));
    let mut worths: HashMap<Role, Vec<i32>> = HashMap::new();
    for l in data.lines().step_by(8) {
        let Some(f) = l.split("\"fen\": \"").nth(1).and_then(|r| r.split('"').next()) else { continue };
        let Ok(p) = Fen::from_ascii(f.as_bytes()).map_err(|_| ()).and_then(|f| f.into_position::<Chess>(CastlingMode::Standard).map_err(|_| ())) else { continue };
        if p.is_check() || p.is_game_over() { continue; }
        for s in p.board().occupied() {
            let Some(w) = ev.worth(&p, s) else { continue };
            if w.abs() < 5000 { worths.entry(p.board().role_at(s).unwrap()).or_default().push(w); }
        }
    }
    let mut out = String::new();
    for r in Role::ALL {
        let Some(ws) = worths.get_mut(&r) else { continue };
        ws.sort();
        out += &format!("{}\t{}\t{}\n", r.char(), ws[ws.len() / 2], ws.len());
    }
    std::fs::write(BASELINE, &out).unwrap();
    print!("{out}");
}

fn mark(v: V) -> &'static str { match v { V::Holds => "holds", V::Fails => "FAILS", V::Unknown => "unknown" } }

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
