//! Evaluates a `Reason` against the board.
//!
//! Tier 1: board facts, shakmaty only. Tier 2: Stockfish over UCI (`STOCKFISH`, default the vendored
//! binary). Tier 3: heuristics ported from commentary_gen/board_tools.py and positional.py.
//!
//! Caveats:
//! - `Because(a, b)` holds when `a` and `b` both hold. That `b` causes `a` is not checked.
//! - `Outweighs(a, b)` holds when both hold. The weighing is not checked.
//! - `WinMaterial` and `Sacrifice` use static exchange evaluation, not a search.
//! - `Prevents`, `Loses`, `Gains` and `Concedes` compare with the position before the enclosing move or line.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::uci::UciMove;
use shakmaty::{
    Bitboard, CastlingMode, Chess, Color, EnPassantMode, Move, Piece, Position, Rank, Role, Square, attacks,
};

use crate::dsl_claude::{Move_, Reason, Reason::*, Region, Sq, examples};

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

fn not(v: Verdict) -> Verdict {
    match v.v {
        V::Holds => fail(format!("not: {}", if v.why.is_empty() { "it holds" } else { &v.why })),
        V::Fails => ok(),
        V::Unknown => v,
    }
}

// --------------------------------------------------------------------------- board helpers

fn sq(s: Sq) -> Square { s.parse().unwrap() }

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

fn san(p: &Chess, m: Move_) -> Result<Move, String> {
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
    pub fn analyse(&mut self, p: &Chess) -> (i32, Vec<String>) {
        if p.is_checkmate() { return (-10000, vec![]); }
        if p.is_stalemate() || p.is_insufficient_material() { return (0, vec![]); }
        let f = fen(p);
        if let Some(r) = self.cache.get(&f) { return r.clone(); }
        self.send(&format!("position fen {f}"));
        self.send(&format!("go nodes {}", self.nodes));
        let (mut score, mut pv) = (0, vec![]);
        for l in self.until("bestmove") {
            let t: Vec<&str> = l.split_whitespace().collect();
            let Some(i) = t.iter().position(|x| *x == "score") else { continue };
            if t.contains(&"upperbound") || t.contains(&"lowerbound") { continue; }
            let n: i32 = t[i + 2].parse().unwrap_or(0);
            score = if t[i + 1] == "mate" { if n > 0 { 10000 } else { -10000 } } else { n };
            if let Some(j) = t.iter().position(|x| *x == "pv") { pv = t[j + 1..].iter().map(|s| s.to_string()).collect(); }
        }
        self.cache.insert(f, (score, pv.clone()));
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
    /// Nodes that the comment claims hold, with their verdicts. Nodes under `Not`, `Prevents` and the like are not claims.
    pub claims: Vec<(String, Verdict)>,
    quiet: usize,
}

fn name(r: &Reason) -> String { format!("{r:?}").split(['(', ' ', '{']).next().unwrap().to_string() }

impl Ev {
    pub fn new(eng: Option<Engine>) -> Ev { Ev { eng, root: Chess::default(), claims: vec![], quiet: 0 } }

    /// Evaluate `reason` for the side that plays `mov` in `root`.
    pub fn run(&mut self, root: &Chess, mov: &Move, reason: &Reason) -> Verdict {
        self.root = root.clone();
        self.claims.clear();
        let cx = Cx { p: root.clone().play(mov.clone()).unwrap(), prev: root.clone(), owner: root.turn() };
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
            Prevents(r) => self.quietly(|ev| ev.before_after(r, cx, !owner, false)),
            Loses(r) => self.quietly(|ev| ev.before_after(r, cx, owner, false)),
            Gains(r) => self.quietly(|ev| ev.before_after(r, cx, owner, true)),
            Concedes(r) => self.quietly(|ev| ev.before_after(r, cx, !owner, true)),
            Only(r) => {
                let v = self.eval(r, cx);
                if v.v != V::Holds { return v; }
                self.quietly(|ev| {
                    let mut vs = vec![];
                    for m in cx.prev.legal_moves() {
                        let q = cx.prev.clone().play(m.clone()).unwrap();
                        if q.board() == cx.p.board() { continue; }
                        let alt = ev.eval(r, &Cx { p: q, prev: cx.prev.clone(), owner });
                        if alt.v == V::Holds { return fail(format!("{} also does it", San::from_move(&cx.prev, m))); }
                        vs.push(alt);
                    }
                    if vs.iter().any(|x| x.v == V::Unknown) { unknown("some alternatives undecided") } else { ok() }
                })
            }
            NoMove(r) => self.quietly(|ev| {
                let mut undecided = false;
                for m in cx.prev.legal_moves() {
                    let q = cx.prev.clone().play(m.clone()).unwrap();
                    match ev.eval(r, &Cx { p: q, prev: cx.prev.clone(), owner }).v {
                        V::Holds => return fail(format!("{} does it", San::from_move(&cx.prev, m))),
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
            Not(r) => not(self.quietly(|ev| ev.eval(r, cx))),
            Because(a, b) => {
                let v = all([self.eval(a, cx), self.eval(b, cx)]);
                if v.v == V::Holds { Verdict { v: V::Holds, why: "both hold; cause unchecked".into() } } else { v }
            }
            Outweighs(a, b2) => {
                let vb = self.eval(b2, &Cx { owner: !owner, ..cx.clone() });
                let v = all([self.eval(a, cx), vb]);
                if v.v == V::Holds { Verdict { v: V::Holds, why: "both hold; weighing unchecked".into() } } else { v }
            }
            Permanent(r) => {
                let v = self.eval(r, cx);
                let structural = matches!(&**r, Weak(Region::Squares(ss)) if ss.iter().all(|s| {
                    let s = sq(s);
                    match b.piece_at(s) {
                        None => hole(p, s, !owner),
                        Some(pc) => pc.role == Role::Pawn && (isolated(p, s, !owner) || doubled(p, s, !owner) || backward(p, s, !owner)),
                    }
                }));
                if v.v == V::Holds && !structural { unknown("permanence unchecked") } else { v }
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
            Material(n) => {
                let d = material(p, owner) - material(p, !owner);
                check(d == *n as i32, format!("material {d:+}"))
            }
            WinMaterial(roles) => {
                let before = cx.prev.clone();
                let taken = roles.iter().all(|r| count(p, !owner, *r) < count(&before, !owner, *r));
                let diff = |q: &Chess| material(q, owner) - material(q, !owner);
                // Recapture on the squares where the opponent lost material, not captures elsewhere.
                let hit = before.board().by_color(!owner) & !b.by_color(!owner) & b.by_color(owner);
                let back = if p.turn() != owner { hit.into_iter().map(|s| see_square(p, s)).max().unwrap_or(0) } else { 0 };
                let net = diff(p) - diff(&before) - back;
                check(taken && net > 0, format!("taken {taken}, net {net:+}"))
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
                check(own.any() && ((own & b.pawns()).any() || own.count() >= opp.count()), format!("{s}: {} vs {}", own.count(), opp.count()))
            }).collect::<Vec<_>>()),
            Pressure(r) => all(match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss }.into_iter()
                .map(|s| check(attackers(p, s, owner).any(), format!("nothing on {s}"))).collect::<Vec<_>>()),
            Outnumber(r) => all(match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss }.into_iter().map(|s| {
                let (own, opp) = (attackers(p, s, owner).count(), attackers(p, s, !owner).count());
                check(own > opp, format!("{s}: {own} vs {opp}"))
            }).collect::<Vec<_>>()),
            Overload { piece, duties } => {
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
            Develop(s) => {
                let s = sq(s);
                check(mine(s) && !matches!(b.role_at(s), Some(Role::Pawn | Role::King)) && rank(s, owner) != 0, "not developed")
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
                Some(s) => check(s >= 300, format!("eval {s:+}")),
            },
            Draw => match self.owner_score(p, owner) {
                None => unknown("no engine"),
                Some(s) => check(s.abs() <= 50, format!("eval {s:+}")),
            },
            Zugzwang => {
                let Some(passed) = p.clone().swap_turn().ok() else { return fail("in check") };
                match (self.owner_score(p, p.turn()), self.owner_score(&passed, p.turn())) {
                    (Some(now), Some(pass)) => check(pass - now >= 100, format!("passing gains {:+}", pass - now)),
                    _ => unknown("no engine"),
                }
            }
            Simplify => match self.owner_score(p, owner) {
                Some(s) => check(s >= 100, format!("eval {s:+}")),
                None => check(material(p, owner) > material(p, !owner), "not ahead in material"),
            },
            Initiative => self.initiative(p, owner),
            Tempo(_) => unknown("tempo count not modelled"),

            // ---------------------------------------------------------------- tier 3: heuristics
            Weak(r) => all(match region(r, p, owner) { Err(e) => return fail(e), Ok(ss) => ss }.into_iter().map(|s| {
                let s2 = s;
                match b.piece_at(s2) {
                    None => check(hole(p, s2, !owner), format!("{s} can still be guarded by a pawn")),
                    Some(pc) if pc.color == owner => fail(format!("{s} is the owner's")),
                    Some(pc) if pc.role == Role::Pawn => check(isolated(p, s2, !owner) || doubled(p, s2, !owner)
                        || backward(p, s2, !owner) || would_hang(p, s2), format!("{s} is a sound pawn")),
                    Some(_) => check(attackers(p, s2, !owner).is_empty() || would_hang(p, s2), format!("{s} is defended")),
                }
            }).collect::<Vec<_>>()),
            Active(s) => {
                let s2 = sq(s);
                match b.role_at(s2) {
                    _ if !mine(s2) => fail(format!("{s} is not the owner's")),
                    Some(Role::Pawn) => check(passed(p, s2, owner) || rank(s2, owner) >= 4, format!("{s} pawn not advanced")),
                    Some(r) => {
                        let need = match r { Role::Knight => 3, Role::Bishop | Role::Rook => 4, Role::Queen => 6, _ => 3 };
                        let m = safe_mobility(p, s2);
                        check(m >= need, format!("{s}: {m} safe moves"))
                    }
                    None => fail(format!("{s} empty")),
                }
            }
            Better { pieces, than } => {
                let worth = |ss: &Vec<Sq>| ss.iter().filter_map(|s| b.role_at(sq(s))).map(cp).sum::<i32>();
                let mob = |ss: &Vec<Sq>| ss.iter().map(|s| safe_mobility(p, sq(s))).sum::<usize>();
                let (a, z) = (worth(pieces), worth(than));
                if !pieces.iter().all(|s| mine(sq(s))) || !than.iter().all(|s| theirs(sq(s))) { return fail("wrong owners"); }
                if a != z { check(a > z, format!("{a} vs {z} cp")) } else { check(mob(pieces) > mob(than), format!("mobility {} vs {}", mob(pieces), mob(than))) }
            }
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

    /// The owner can play `line` next without losing material on each own move.
    fn enables(&mut self, p: &Chess, owner: Color, line: &[Move_]) -> Verdict {
        let Some(mut q) = as_mover(p, owner) else { return fail("owner in check, cannot pass") };
        for m in line {
            if *m == "--" { q = match q.swap_turn() { Ok(x) => x, Err(_) => return fail("cannot pass") }; continue; }
            let mv = match san(&q, m) { Ok(x) => x, Err(e) => return fail(e) };
            if q.turn() == owner && see(&q, &mv) < 0 { return fail(format!("{m} loses material")); }
            q = q.play(mv).unwrap();
        }
        ok()
    }

    /// Holds unless the engine rates the threat's first move well below its own best.
    fn threat_is_real(&mut self, p: &Chess, owner: Color, line: &[Move_]) -> Option<Verdict> {
        let first = *line.first()?;
        if first == "--" { return None; }
        let q = as_mover(p, owner)?;
        let best = self.owner_score(&q, owner)?;
        let after = self.owner_score(&q.clone().play(san(&q, first).ok()?).ok()?, owner)?;
        (best - after > 100).then(|| unknown(format!("{first} is {} cp below best", best - after)))
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

fn mark(v: V) -> &'static str { match v { V::Holds => "holds", V::Fails => "FAILS", V::Unknown => "unknown" } }

/// Every example on its real move, then on up to three other legal moves as a baseline.
pub fn report() {
    let eng = engine();
    if eng.is_none() { eprintln!("no engine: tier-2 reasons will be unknown"); }
    let mut ev = Ev::new(eng);
    let mut real: HashMap<String, [usize; 3]> = HashMap::new();
    let mut base: HashMap<String, [usize; 3]> = HashMap::new();
    let (mut top_real, mut top_base) = ([0usize; 3], [0usize; 3]);
    let idx = |v: V| match v { V::Holds => 0, V::Fails => 1, V::Unknown => 2 };

    for (i, ex) in examples().iter().enumerate() {
        let root: Chess = Fen::from_ascii(ex.fen.as_bytes()).unwrap().into_position(CastlingMode::Standard).unwrap();
        let mv = san(&root, ex.mov).unwrap();
        let v = ev.run(&root, &mv, &ex.reason);
        top_real[idx(v.v)] += 1;
        println!("\n#{i} {} ... {}: {}", ex.mov, mark(v.v), v.why);
        for (n, c) in &ev.claims {
            real.entry(n.clone()).or_default()[idx(c.v)] += 1;
            if c.v != V::Holds { println!("    {n}: {} {}", mark(c.v), c.why); }
        }

        let alts: Vec<Move> = root.legal_moves().into_iter().filter(|m| *m != mv).collect();
        for k in 0..3.min(alts.len()) {
            let m = &alts[(i * 7 + k * 13) % alts.len()];
            let v = ev.run(&root, m, &ex.reason);
            top_base[idx(v.v)] += 1;
            for (n, c) in &ev.claims { base.entry(n.clone()).or_default()[idx(c.v)] += 1; }
        }
    }

    let rate = |c: &[usize; 3]| if c[0] + c[1] == 0 { "-".to_string() } else { format!("{:.0}%", 100.0 * c[0] as f64 / (c[0] + c[1]) as f64) };
    println!("\nWhole reasons: real move {} holds / {} fails / {} unknown; other moves {} / {} / {}",
        top_real[0], top_real[1], top_real[2], top_base[0], top_base[1], top_base[2]);
    println!("\n{:<12} {:>6} {:>6} {:>6} {:>10} {:>10}", "reason", "holds", "fails", "unknown", "real", "baseline");
    let mut names: Vec<_> = real.keys().chain(base.keys()).cloned().collect();
    names.sort();
    names.dedup();
    for n in names {
        let r = real.get(&n).copied().unwrap_or_default();
        let z = base.get(&n).copied().unwrap_or_default();
        println!("{n:<12} {:>6} {:>6} {:>6} {:>10} {:>10}", r[0], r[1], r[2], rate(&r), rate(&z));
    }
}
