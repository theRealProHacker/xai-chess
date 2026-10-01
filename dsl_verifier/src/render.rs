//! Renders a `Reason` as English. Deterministic: one phrasing per variant, and every node is rendered.
//!
//! A reason renders to parts: clauses (prefix, subject, verb phrase) or finished text. Clauses stay
//! open so that `Not`, `Removes`, `Degree` and the like can change their verb, and so that clauses
//! with one subject coordinate ("develops the knight on d7 and controls h5").
//!
//! Pieces are named where `eval.rs` evaluates the node: after the move by default, before it under
//! `Removes` and `Loses`, before the top-level move for `Trade`. A line that does not replay leaves
//! its subtree with bare squares; rendering never fails.

use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Piece, Position, Role, Square};

use crate::dsl_claude::{Annotation, Grade, Level, Move_, Pieces, QUESTIONABLE, Reason, Reason::*, Region, Sq, examples};
use crate::eval::san;

// --------------------------------------------------------------------------- grammar

#[derive(Clone, PartialEq)]
enum Subj {
    Side(Color),
    /// A noun phrase, whether it is plural, and the square of the piece it names.
    Np(String, bool, Option<Square>),
}

impl Subj {
    fn text(&self) -> String {
        match self { Subj::Side(c) => side(*c).into(), Subj::Np(s, ..) => s.clone() }
    }
    fn plural(&self) -> bool { matches!(self, Subj::Np(_, true, _)) }
}

#[derive(Clone)]
struct Vp {
    modal: Option<&'static str>,
    adv: Option<&'static str>,
    neg: bool,
    past: bool,
    verb: &'static str,
    rest: String,
    /// An action ("attacks", "wins") may follow a line as a participle; a state may not.
    action: bool,
    /// "have" as an auxiliary: "has not castled", not "does not have castled".
    perfect: bool,
}

fn vp(verb: &'static str, rest: impl Into<String>) -> Vp {
    Vp { modal: None, adv: None, neg: false, past: false, verb, rest: rest.into(), action: false, perfect: false }
}
fn can(verb: &'static str, rest: impl Into<String>) -> Vp { Vp { modal: Some("can"), ..vp(verb, rest) } }
fn act(verb: &'static str, rest: impl Into<String>) -> Vp { Vp { action: true, ..vp(verb, rest) } }
fn perfect(rest: impl Into<String>) -> Vp { Vp { perfect: true, ..vp("have", rest) } }

fn third(v: &str) -> String {
    match v {
        "be" => "is".into(),
        "have" => "has".into(),
        _ if v.ends_with(['s', 'x', 'z', 'o']) || v.ends_with("sh") || v.ends_with("ch") => format!("{v}es"),
        _ if v.ends_with('y') && !v[..v.len() - 1].ends_with(['a', 'e', 'o']) => format!("{}ies", &v[..v.len() - 1]),
        _ => format!("{v}s"),
    }
}

fn gerund(v: &str) -> String {
    match v {
        "be" => "being".into(),
        "win" | "pin" | "put" | "get" | "stop" | "plan" => format!("{v}{}ing", &v[v.len() - 1..]),
        "control" => "controlling".into(),
        _ if v.ends_with('e') && !v.ends_with("ee") => format!("{}ing", &v[..v.len() - 1]),
        _ => format!("{v}ing"),
    }
}

impl Vp {
    fn tail(&self) -> String { if self.rest.is_empty() { String::new() } else { format!(" {}", self.rest) } }

    fn finite(&self, plural: bool) -> String {
        // "still" goes before a negated verb: "still does not", "is still not", "still cannot".
        let still = self.neg && self.adv == Some("still");
        let adv = if still { String::new() } else { self.adv.map(|a| format!(" {a}")).unwrap_or_default() };
        let pre = if still { "still " } else { "" };
        let head = if let Some(m) = self.modal {
            let m = match (self.neg, m) { (true, "can") => "cannot".into(), (true, m) => format!("{m} not"), (false, m) => m.into() };
            format!("{pre}{m}{adv} {}", self.verb)
        } else if self.verb == "have" && self.perfect {
            format!("{}{}{adv}", if plural { "have" } else { "has" }, if still { " still not" } else if self.neg { " not" } else { "" })
        } else if self.verb == "be" {
            let be = match (self.past, plural) { (true, false) => "was", (true, true) => "were", (false, false) => "is", (false, true) => "are" };
            format!("{be}{}{adv}", if still { " still not" } else if self.neg { " not" } else { "" })
        } else if self.neg {
            format!("{pre}{} not{adv} {}", if plural { "do" } else { "does" }, self.verb)
        } else {
            format!("{}{}", self.adv.map(|a| format!("{a} ")).unwrap_or_default(), if plural { self.verb.into() } else { third(self.verb) })
        };
        format!("{head}{}", self.tail())
    }

    fn ing(&self) -> String {
        let neg = if self.neg { "not " } else { "" };
        let v = if self.modal == Some("can") { format!("being able to {}", self.verb) } else { gerund(self.verb) };
        match self.adv {
            Some(a) => format!("{neg}{a} {v}{}", self.tail()),
            None => format!("{neg}{v}{}", self.tail()),
        }
    }

    /// `Not`: "no longer" becomes "still", "now" becomes "still not".
    fn negate(&mut self) {
        match self.adv {
            Some("no longer") => self.adv = Some("still"),
            Some("now") => { self.adv = Some("still"); self.neg = true }
            _ => self.neg = !self.neg,
        }
    }
}

#[derive(Clone)]
struct Clause {
    prefix: Option<String>,
    subj: Subj,
    vp: Vp,
}

impl Clause {
    fn text(&self) -> String {
        let pre = self.prefix.as_ref().map(|p| format!("{p}, ")).unwrap_or_default();
        format!("{pre}{} {}", self.subj.text(), self.vp.finite(self.subj.plural()))
    }
}

enum Part {
    C(Clause),
    T(String),
}
use Part::{C, T};

fn clause(subj: Subj, vp: Vp) -> Part { C(Clause { prefix: None, subj, vp }) }

fn list(xs: &[String], conj: &str) -> String {
    match xs {
        [] => String::new(),
        [a] => a.clone(),
        [init @ .., last] => format!("{} {conj} {last}", init.join(", ")),
    }
}

/// Clauses with one subject share it ("develops X and controls h5"); groups take ", and" before the last.
fn join(parts: &[Part]) -> String {
    let mut groups = vec![];
    let mut i = 0;
    while i < parts.len() {
        match &parts[i] {
            T(t) => { groups.push(t.clone()); i += 1; }
            C(c) => {
                let mut vps = vec![c.vp.finite(c.subj.plural())];
                i += 1;
                while let Some(C(d)) = parts.get(i) {
                    if d.subj != c.subj || d.prefix.is_some() { break }
                    vps.push(d.vp.finite(d.subj.plural()));
                    i += 1;
                }
                let pre = c.prefix.as_ref().map(|p| format!("{p}, ")).unwrap_or_default();
                // "stops the threat of X, winning a pawn, and stops ..." keeps the second verb apart.
                let vps = if vps.len() > 1 && vps.iter().any(|v| v.contains(',')) {
                    format!("{}, and {}", vps[..vps.len() - 1].join(", "), vps[vps.len() - 1])
                } else {
                    list(&vps, "and")
                };
                groups.push(format!("{pre}{} {vps}", c.subj.text()));
            }
        }
    }
    match groups.as_slice() {
        [] => String::new(),
        [a] => a.clone(),
        [init @ .., last] => format!("{}, and {last}", init.join(", ")),
    }
}

fn single(parts: &[Part]) -> Option<&Clause> {
    match parts { [C(c)] => Some(c), _ => None }
}

/// Apply `f` to a lone clause; otherwise wrap the joined text.
fn modify(mut parts: Vec<Part>, f: impl FnOnce(&mut Clause), wrap: impl FnOnce(String) -> String) -> Vec<Part> {
    if let [C(c)] = parts.as_mut_slice() {
        f(c);
        return parts;
    }
    vec![T(wrap(join(&parts)))]
}

/// A prefix ("after 5.bxc3,") on a lone clause that has none; otherwise on the text.
fn prefixed(parts: Vec<Part>, pre: String) -> Vec<Part> {
    match parts.as_slice() {
        [C(c)] if c.prefix.is_none() => modify(parts, |c| c.prefix = Some(pre), |t| t),
        _ => vec![T(format!("{pre}, {}", join(&parts)))],
    }
}

// --------------------------------------------------------------------------- names

fn side(c: Color) -> &'static str { if c == Color::White { "White" } else { "Black" } }

fn role(r: Role) -> &'static str {
    match r { Role::Pawn => "pawn", Role::Knight => "knight", Role::Bishop => "bishop", Role::Rook => "rook", Role::Queen => "queen", Role::King => "king" }
}

/// For pin or skewer: the king counts as most valuable.
fn worth(r: Role) -> u32 {
    match r { Role::Pawn => 1, Role::Knight | Role::Bishop => 3, Role::Rook => 5, Role::Queen => 9, Role::King => 100 }
}

fn count_word(n: usize) -> String {
    ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"].get(n).map_or(n.to_string(), |w| w.to_string())
}

fn square(s: Sq) -> Square { s.parse().unwrap() }

/// "a pawn", "two pawns and a knight".
fn material(roles: &[Role]) -> String {
    let mut out = vec![];
    for r in Role::ALL.iter().rev() {
        let n = roles.iter().filter(|x| *x == r).count();
        if n == 1 { out.push(format!("a {}", role(*r))) } else if n > 1 { out.push(format!("{} {}s", count_word(n), role(*r))) }
    }
    list(&out, "and")
}

fn ordinal(r: u32) -> String { format!("{r}{}", match r { 1 => "st", 2 => "nd", 3 => "rd", _ => "th" }) }

/// The line from `a` to `b`, either order: "the c-file", "the c-file from c2 to c8", "the b2–h8 diagonal".
fn line_name(a: Sq, b: Sq) -> String {
    let (x, y) = (square(a), square(b));
    let whole = |u: u32, v: u32| u.min(v) == 0 && u.max(v) == 7;
    if x.file() == y.file() {
        let f = format!("the {}-file", x.file().char());
        if whole(x.rank() as u32, y.rank() as u32) { f } else { format!("{f} from {a} to {b}") }
    } else if x.rank() == y.rank() {
        let r = format!("the {} rank", ordinal(x.rank() as u32 + 1));
        if whole(x.file() as u32, y.file() as u32) { r } else { format!("{r} from {a} to {b}") }
    } else {
        format!("the {a}–{b} diagonal")
    }
}

/// The line as text, and the position and the last move's square after it (None if it does not replay).
/// Numbered from `p` when `numbered` and there is no pass; otherwise segments between passes are listed.
fn line(p: Option<&Chess>, owner: Option<Color>, moves: &[Move_], numbered: bool) -> (String, Option<Chess>, Option<Square>) {
    let mut q = p.and_then(|p| if owner.is_some_and(|c| c != p.turn()) { p.clone().swap_turn().ok() } else { Some(p.clone()) });
    let numbered = numbered && !moves.contains(&"--");
    let (mut segs, mut cur, mut first, mut last) = (vec![], vec![], true, None);
    for m in moves {
        if *m == "--" {
            if !cur.is_empty() { segs.push(std::mem::take(&mut cur).join(" ")) }
            q = q.and_then(|q| q.swap_turn().ok());
            continue;
        }
        cur.push(match &q {
            Some(q) if numbered => match (q.turn(), first) {
                (Color::White, _) => format!("{}.{m}", q.fullmoves()),
                (Color::Black, true) => format!("{}...{m}", q.fullmoves()),
                (Color::Black, false) => m.to_string(),
            },
            _ => m.to_string(),
        });
        first = false;
        q = q.and_then(|q| {
            let mv = san(&q, m).ok()?;
            last = Some(match mv { Move::Castle { king, rook } => if rook > king { king.offset(2)? } else { king.offset(-2)? }, _ => mv.to() });
            q.play(mv).ok()
        });
        if q.is_none() { last = None }
    }
    if !cur.is_empty() { segs.push(cur.join(" ")) }
    (list(&segs, "and"), q, last)
}

// --------------------------------------------------------------------------- context

#[derive(Clone)]
struct Rx {
    /// Where pieces are named; None: bare squares.
    p: Option<Chess>,
    prev: Option<Chess>,
    /// Before the top-level move: `Trade` names pieces here.
    root: Chess,
    owner: Color,
    /// Lines get move numbers; false under threats, plans and other hypotheticals.
    numbered: bool,
    /// The move `Rated` is about: the top move or an `Instead` alternative.
    mv: Option<String>,
    /// Under `Instead`: "was stronger".
    past: bool,
}

impl Rx {
    fn opp(&self) -> Color { !self.owner }
    fn flip(&self) -> Rx { Rx { owner: self.opp(), ..self.clone() } }
    fn hypothetical(&self) -> Rx { Rx { numbered: false, mv: None, ..self.clone() } }
    /// After a line: `prev` is where the line started.
    fn at(&self, p: Option<Chess>, owner: Color) -> Rx { Rx { p, prev: self.p.clone(), owner, mv: None, past: false, ..self.clone() } }
    /// Before the move, as `Removes` and `Loses` look at it.
    fn before(&self) -> Rx { Rx { p: self.prev.clone(), ..self.clone() } }
    fn side_c(&self, v: Vp) -> Part { clause(Subj::Side(self.owner), v) }

    fn role_at(&self, s: Sq) -> Option<Role> { self.p.as_ref()?.board().role_at(square(s)) }
    fn piece(&self, s: Sq) -> String { self.role_at(s).map_or(s.to_string(), |r| format!("the {} on {s}", role(r))) }
    /// "the pawns on c3 and c4" when all are one kind, else each by name.
    fn pieces(&self, ss: &[Sq]) -> String {
        let roles: Vec<Option<Role>> = ss.iter().map(|s| self.role_at(s)).collect();
        match roles.first() {
            Some(Some(r)) if ss.len() > 1 && roles.iter().all(|x| *x == Some(*r)) =>
                format!("the {}s on {}", role(*r), list(&ss.iter().map(|s| s.to_string()).collect::<Vec<_>>(), "and")),
            _ => list(&ss.iter().map(|s| self.piece(s)).collect::<Vec<_>>(), "and"),
        }
    }
    fn np(&self, ss: &[Sq]) -> Subj {
        Subj::Np(self.pieces(ss), ss.len() > 1, (ss.len() == 1).then(|| square(ss[0])))
    }
    fn the(&self, s: Sq) -> Subj { Subj::Np(self.piece(s), false, Some(square(s))) }
    fn who(&self, opponent: bool) -> Color { if opponent { self.opp() } else { self.owner } }

    fn region(&self, r: &Region) -> String {
        match r {
            Region::Squares(ss) => list(&ss.iter().map(|s| s.to_string()).collect::<Vec<_>>(), "and"),
            Region::Kingside => "the kingside".into(),
            Region::Queenside => "the queenside".into(),
            Region::Centre => "the centre".into(),
            Region::GreaterCentre => "the greater centre".into(),
            Region::Half { opponent } => format!("{}'s half", side(self.who(*opponent))),
            Region::KingWing { opponent } => format!("{}'s castled side", side(self.who(*opponent))),
            Region::KingZone { opponent } => format!("the squares around {}'s king", side(self.who(*opponent))),
            Region::Line(a, b) => line_name(a, b),
        }
    }
    fn region_at(&self, r: &Region) -> String {
        match r {
            Region::Centre | Region::GreaterCentre | Region::Half { .. } => format!("in {}", self.region(r)),
            Region::KingZone { opponent } => format!("around {}'s king", side(self.who(*opponent))),
            _ => format!("on {}", self.region(r)),
        }
    }

    /// `Isolated`, `Doubled`, `Backward`: listed pieces as subject; a named region belongs to the opponent.
    fn structure(&self, r: &Region, state: &'static str, noun: &str) -> Part {
        match r {
            Region::Squares(ss) => clause(self.np(ss), vp("be", state)),
            _ => clause(Subj::Side(self.opp()), vp("have", format!("{noun} {}", self.region_at(r)))),
        }
    }

    /// The line's consequence: a participle for one action by the owner (threats only) or by the piece
    /// that moved last; otherwise "after which ...".
    fn consequence(&self, r: &Reason, after: &Rx, last: Option<Square>, owner_ok: bool) -> String {
        if let After(l2, r2) = r {
            let (t, q, _) = line(after.p.as_ref(), None, l2, after.numbered);
            return format!(", followed by {t}, after which {}", join(&render(r2, &after.at(q, after.owner))));
        }
        if matches!(r, Mate) { return ", mate".into() }
        let parts = render(r, after);
        match single(&parts) {
            Some(c) if c.vp.action && c.prefix.is_none()
                && ((owner_ok && c.subj == Subj::Side(self.owner)) || matches!(c.subj, Subj::Np(_, false, Some(s)) if Some(s) == last)) =>
                format!(", {}", c.vp.ing()),
            _ => format!(", after which {}", join(&parts)),
        }
    }
}

// --------------------------------------------------------------------------- reasons

fn render(r: &Reason, rx: &Rx) -> Vec<Part> {
    let (owner, opp) = (rx.owner, rx.opp());
    let p = rx.p.as_ref();
    match r {
        // ---------------------------------------------------------------- combinators
        And(rs) => rs.iter().flat_map(|r| render(r, rx)).collect(),
        Either(rs) => {
            let xs: Vec<String> = rs.iter().map(|r| join(&render(r, rx))).collect();
            vec![T(format!("either {}", xs.join(", or ")))]
        }
        Threatens(l, r) => {
            let h = rx.hypothetical();
            let (t, q, last) = line(p, Some(owner), l, false);
            let verb = if l.contains(&"--") { "plan" } else { "threaten" };
            vec![rx.side_c(vp(verb, format!("{t}{}", rx.consequence(r, &h.at(q, owner), last, true))))]
        }
        After(l, r) => {
            let (t, q, _) = line(p, None, l, rx.numbered);
            if matches!(**r, Mate) { return vec![T(format!("{t}, mate"))] }
            prefixed(render(r, &rx.at(q, owner)), format!("after {t}"))
        }
        Allows(l, r) if l.is_empty() => render(r, &rx.flip()),
        Allows(l, r) => {
            let (t, q, last) = line(p, None, l, rx.numbered);
            vec![rx.side_c(vp("allow", format!("{t}{}", rx.consequence(r, &rx.at(q, opp), last, false))))]
        }
        Prevents(r) => {
            let inner = Rx { p: None, prev: None, ..rx.hypothetical().flip() };
            vec![rx.side_c(vp("stop", format!("{} from getting a position where {}", side(opp), join(&render(r, &inner)))))]
        }
        Removes(r) => match &**r {
            Threatens(l, t) => {
                let b = rx.before().flip();
                let (text, q, last) = line(b.p.as_ref(), Some(opp), l, false);
                let after = b.hypothetical().at(q, opp);
                vec![rx.side_c(vp("stop", format!("the threat of {text}{}", b.consequence(t, &after, last, true))))]
            }
            _ => no_longer(render(r, &rx.before().flip())),
        },
        Loses(r) => match &**r {
            Tempo(n) => vec![rx.side_c(vp("lose", tempi(*n)))],
            _ => no_longer(render(r, &rx.before())),
        },
        Gains(r) => now(render(r, rx)),
        Concedes(r) => now(render(r, &rx.flip())),
        Only(r) => {
            let inner = render(r, &rx.hypothetical());
            let Some(mv) = rx.mv.clone() else { return vec![T(format!("{}, and no other move does", join(&inner)))] };
            let rest = match single(&inner) {
                Some(c) if c.subj == Subj::Side(owner) && c.prefix.is_none() => format!("the only move that {}", c.vp.finite(false)),
                _ => format!("the only move after which {}", join(&inner)),
            };
            vec![clause(Subj::Np(mv, false, None), Vp { past: rx.past, ..vp("be", rest) })]
        }
        NoMove(r) => match &**r {
            Not(r) => vec![T(format!("after every {} move, {}", side(owner), join(&render(r, &rx.hypothetical()))))],
            _ => {
                let inner = render(r, &rx.hypothetical());
                vec![rx.side_c(vp("have", match single(&inner) {
                    Some(c) if c.subj == Subj::Side(owner) && c.prefix.is_none() => format!("no move that {}", c.vp.finite(false)),
                    _ => format!("no move after which {}", join(&inner)),
                }))]
            }
        },
        Instead(m, r) => {
            let (t, q, _) = line(rx.prev.as_ref(), None, &[m], rx.numbered);
            let alt = Rx { p: q, mv: Some(t.clone()), past: true, ..rx.clone() };
            match &**r {
                Because(a, b) if matches!(**a, Rated(_)) => vec![T(format!("{}: {}", join(&render(a, &alt)), join(&render(b, &alt))))],
                Rated(_) => render(r, &alt),
                _ => prefixed(render(r, &alt), format!("with {t} instead")),
            }
        }
        Sacrifice(roles, r) => vec![rx.side_c(vp("give", format!("up {} so that {}", material(roles), join(&render(r, rx)))))],
        Not(r) => match &**r {
            Either(rs) => {
                let xs: Vec<String> = rs.iter().map(|r| join(&render(r, rx))).collect();
                vec![T(format!("neither {}", xs.join(", nor ")))]
            }
            Closed => vec![clause(Subj::Np("the position".into(), false, None), vp("be", "open"))],
            _ => modify(render(r, rx), |c| c.vp.negate(), |t| format!("it is not the case that {t}")),
        },
        Because(a, b) => match &**a {
            Rated(_) => vec![T(format!("{}: {}", join(&render(a, rx)), join(&render(b, rx))))],
            _ => vec![T(format!("{}, because {}", join(&render(a, rx)), join(&render(b, rx))))],
        },
        But(a, b) => {
            let (a, b) = (render(a, rx), render(b, rx));
            vec![T(match (single(&a), single(&b)) {
                (Some(x), Some(y)) if x.subj == y.subj && y.prefix.is_none() => format!("{}, but {}", x.text(), y.vp.finite(y.subj.plural())),
                _ => format!("{}, but {}", join(&a), join(&b)),
            })]
        }
        Outweighs(a, b) => vec![T(format!("{}, which outweighs the fact that {}", join(&render(a, rx)), join(&render(b, &rx.flip()))))],
        Balances(a, b) => vec![T(format!("{}, which is balanced by the fact that {}", join(&render(a, rx)), join(&render(b, &rx.flip()))))],
        Permanent(r) => modify(render(r, rx), |c| c.vp.adv = Some("permanently"), |t| format!("permanently, {t}")),
        Eventually(r) => prefixed(render(r, rx), "in the long run".into()),
        More(r) => match &**r {
            Develop(_) => vec![rx.side_c(vp("be", "ahead in development"))],
            _ => vec![T(format!("{}, more than {}", join(&render(r, rx)), side(opp)))],
        },
        Degree(l, r) => {
            let adv = match l { Level::Slight => "slightly", Level::Clear => "clearly", Level::Decisive => "decisively" };
            modify(render(r, rx), |c| c.vp.adv = Some(adv), |t| format!("{adv}, {t}"))
        }
        Faster(a, b) => vec![T(format!("{} before {}", join(&render(a, rx)), join(&render(b, &rx.flip()))))],
        Suppose(edits, r) => {
            let piece = |c: char| Piece::from_char(c).unwrap();
            let named = |s: Sq| p.and_then(|p| p.board().piece_at(square(s)));
            let cond = match edits.as_slice() {
                [(a, None), (b, Some(c))] | [(b, Some(c)), (a, None)] if named(a) == Some(piece(*c)) =>
                    format!("if {} stood on {b}", rx.piece(a)),
                _ => format!("if {}", list(&edits.iter().map(|(s, c)| match c {
                    Some(c) => format!("a {} {} stood on {s}", side(piece(*c).color).to_lowercase(), role(piece(*c).role)),
                    None => format!("{s} were empty"),
                }).collect::<Vec<_>>(), "and")),
            };
            let q = p.and_then(|p| {
                let mut setup = p.to_setup(EnPassantMode::Legal);
                for (s, c) in edits {
                    match c {
                        Some(c) => setup.board.set_piece_at(square(s), piece(*c)),
                        None => { setup.board.discard_piece_at(square(s)); }
                    }
                }
                setup.position::<Chess>(CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights()).ok()
            });
            prefixed(render(r, &Rx { p: q, ..rx.hypothetical() }), cond)
        }

        // ---------------------------------------------------------------- atoms
        Rated(a) => {
            let what = match a {
                Annotation::Blunder => "a blunder",
                Annotation::Mistake => "a mistake",
                Annotation::Dubious => "dubious",
                Annotation::Interesting => "interesting",
                Annotation::Good => "good",
                Annotation::Brilliant => "brilliant",
                Annotation::Best => "best",
                Annotation::Stronger => "stronger",
                Annotation::Weaker => "weaker",
            };
            let mv = rx.mv.clone().unwrap_or_else(|| "this line".into());
            vec![clause(Subj::Np(mv, false, None), Vp { past: rx.past, ..vp("be", what) })]
        }
        Check | Mate | Zugzwang => {
            let to_move = p.map_or(opp, |p| p.turn());
            let state = match r { Check => "in check", Mate => "checkmated", _ => "in zugzwang" };
            vec![clause(Subj::Side(to_move), vp("be", state))]
        }
        WinMaterial(roles) => vec![rx.side_c(act("win", material(roles)))],
        Material(n) => vec![rx.side_c(vp("be", match n.signum() {
            0 => "level in material".into(),
            s => {
                let k = n.unsigned_abs() as usize;
                let amount = if k == 1 { "a pawn".into() } else { format!("{} pawns", count_word(k)) };
                format!("{amount} {}", if s > 0 { "ahead" } else { "behind" })
            }
        }))],
        Trade { give, get } => {
            let origin = |s: &Sq| rx.root.board().role_at(square(s)).map_or(s.to_string(), |r| format!("the {s}-{}", role(r)));
            let names = |ss: &[Sq]| list(&ss.iter().map(origin).collect::<Vec<_>>(), "and");
            vec![rx.side_c(act("trade", format!("{} for {}", names(give), names(get))))]
        }
        Simplify => vec![clause(Subj::Np("trading pieces".into(), false, None), vp("favour", side(owner)))],
        Attack { by, targets } => {
            let verb = if targets.len() > 1 { "fork" } else { "attack" };
            vec![clause(rx.the(by), act(verb, rx.pieces(targets)))]
        }
        Pin { by, front, behind } => {
            let skewer = matches!((rx.role_at(front), rx.role_at(behind)), (Some(f), Some(b)) if worth(f) > worth(b));
            let v = if skewer { act("skewer", format!("{} against {}", rx.piece(front), rx.piece(behind))) } else { act("pin", format!("{} to {}", rx.piece(front), rx.piece(behind))) };
            vec![clause(rx.the(by), v)]
        }
        XRay { by, through, target } => {
            let own = p.and_then(|p| p.board().color_at(square(target))) == Some(owner);
            vec![clause(rx.the(by), act(if own { "defend" } else { "x-ray" }, format!("{} through {}", rx.piece(target), rx.piece(through))))]
        }
        Defend { by, target } => vec![clause(rx.the(by), act("defend", rx.piece(target)))],
        Bound { piece, duties } => {
            let state = if duties.len() > 1 { "overloaded, defending" } else { "tied to defending" };
            vec![clause(rx.the(piece), vp("be", format!("{state} {}", rx.pieces(duties))))]
        }
        Decoy { from, to } => vec![clause(rx.the(from), can("be", format!("lured to {to}")))],
        Block { line: (a, b), by } => vec![clause(rx.the(by), vp("block", line_name(a, b)))],
        Open((a, b)) => vec![clause(Subj::Np(line_name(a, b), false, None), vp("be", format!("open for {}", side(owner))))],
        Control(r) => vec![rx.side_c(act("control", rx.region(r)))],
        Pressure(r) => {
            let on = match r { Region::Squares(ss) => rx.pieces(ss), _ => rx.region(r) };
            vec![rx.side_c(act("put", format!("pressure on {on}")))]
        }
        Outnumber(r) => vec![rx.side_c(vp("outnumber", format!("the defenders of {}", rx.region(r))))],
        Isolated(r) => vec![rx.structure(r, "isolated", "an isolated pawn")],
        Doubled(r) => vec![rx.structure(r, "doubled", "doubled pawns")],
        Backward(r) => vec![rx.structure(r, "backward", "a backward pawn")],
        Hanging(r) => vec![match r {
            Region::Squares(ss) => clause(rx.np(ss), vp("hang", "")),
            _ => clause(Subj::Np(format!("a {} piece {}", side(opp).to_lowercase(), rx.region_at(r)), false, None), vp("hang", "")),
        }],
        Hole(r) => vec![match r {
            Region::Squares(ss) => clause(Subj::Np(rx.region(r), ss.len() > 1, None), vp("be", if ss.len() > 1 { "holes" } else { "a hole" })),
            _ => clause(Subj::Side(opp), vp("have", format!("a hole {}", rx.region_at(r)))),
        }],
        Weakened(r) => vec![match r {
            // Pawns or squares: a piece standing on the square is not what is weakened.
            Region::Squares(ss) if ss.iter().all(|s| rx.role_at(s) == Some(Role::Pawn)) => clause(rx.np(ss), vp("be", "weakened")),
            Region::Squares(ss) => clause(Subj::Np(rx.region(r), ss.len() > 1, None), vp("be", "weakened")),
            _ => clause(Subj::Np(format!("a {} pawn {}", side(opp).to_lowercase(), rx.region_at(r)), false, None), vp("be", "weakened")),
        }],
        Develop(ps) => vec![rx.side_c(match ps {
            Pieces::All => perfect("developed its knights and bishops"),
            Pieces::At(s) => vp("develop", rx.piece(s)),
            Pieces::Every(r @ (Role::Queen | Role::King)) => vp("develop", format!("the {}", role(*r))),
            Pieces::Every(r) => vp("develop", format!("the {}s", role(*r))),
            Pieces::On(reg) => perfect(format!("developed its pieces {}", rx.region_at(reg))),
        })],
        Quality(s, g) => {
            let g = match g { Grade::Good => "good", Grade::Bad => "bad" };
            vec![clause(rx.the(s), vp("be", format!("a {g} {}", rx.role_at(s).map_or("piece", role))))]
        }
        Better { pieces, than } => vec![clause(rx.np(pieces), vp("be", format!("worth more than {}", rx.pieces(than))))],
        Coordinate(ps) => vec![clause(rx.np(ps), vp("work", "together"))],
        Castle => {
            let castled = p.and_then(|p| p.board().king_of(owner)).is_some_and(|k| {
                owner.relative_rank(k.rank()) == shakmaty::Rank::First && matches!(k.file() as u32, 1 | 2 | 6)
            });
            vec![rx.side_c(if castled { perfect("castled") } else { can("castle", "") })]
        }
        Tempo(n) => vec![rx.side_c(vp("be", if *n >= 0 { format!("{} ahead", tempi(*n)) } else { format!("{} behind", tempi(-n)) }))],
        Initiative => vec![rx.side_c(vp("have", "the initiative"))],
        Space => vec![rx.side_c(vp("have", "more space"))],
        BishopPair => vec![rx.side_c(vp("have", "the bishop pair"))],
        Practical => vec![rx.side_c(vp("make", format!("the game harder for {} in practice", side(opp))))],
        Imbalance => vec![rx.side_c(vp("unbalance", "the position"))],
        Closed => vec![clause(Subj::Np("the centre".into(), false, None), vp("be", "closed"))],
        Passer(s) => vec![clause(rx.the(s), vp("be", "a passed pawn"))],
        Promote(s) => vec![clause(rx.the(s), can("promote", ""))],
        Majority(r) => vec![match r {
            Region::Squares(_) => clause(Subj::Np(format!("{}'s pawns on {}", side(owner), rx.region(r)), true, None),
                vp("outnumber", format!("{}'s on those files", side(opp)))),
            _ => rx.side_c(vp("have", format!("a pawn majority {}", rx.region_at(r)))),
        }],
        Enables(l) => vec![rx.side_c(can("play", line(None, None, l, false).0))],
        Options(ms) => vec![rx.side_c(vp("keep", format!("the choice between {} open", list(&ms.iter().map(|m| m.to_string()).collect::<Vec<_>>(), "and"))))],
        Draw => vec![clause(Subj::Np("the position".into(), false, None), vp("be", "a draw"))],
        Win => vec![rx.side_c(vp("be", "winning"))],
        SameColour(ss) => {
            let shade = if square(ss[0]).is_light() { "light" } else { "dark" };
            vec![clause(rx.np(ss), vp("stand", format!("on {shade} squares")))]
        }
    }
}

fn tempi(n: i8) -> String { if n == 1 { "a tempo".into() } else { format!("{} tempi", count_word(n as usize)) } }

fn no_longer(parts: Vec<Part>) -> Vec<Part> {
    modify(parts, |c| c.vp.adv = Some("no longer"), |t| format!("it is no longer the case that {t}"))
}

fn now(parts: Vec<Part>) -> Vec<Part> { modify(parts, |c| c.vp.adv = Some("now"), |t| format!("now {t}")) }

/// Capitalised, unless it opens with a square ("e3 and g3 are now weakened").
fn sentence(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() > 1 && (b'a'..=b'h').contains(&b[0]) && (b'1'..=b'8').contains(&b[1]) { return format!("{s}."); }
    let mut c = s.chars();
    c.next().map_or(String::new(), |f| format!("{}{}.", f.to_uppercase(), c.as_str()))
}

/// The reason given for `mov` in `root`, as English. A top-level `And` gives one sentence per reason.
/// The move opens the text unless a rating or `Only` already names it.
pub fn render_reason(root: &Chess, mov: Move_, reason: &Reason) -> String {
    let (mv, p, _) = line(Some(root), None, &[mov], true);
    let rx = Rx { p, prev: Some(root.clone()), root: root.clone(), owner: root.turn(), numbered: true, mv: Some(mv.clone()), past: false };
    let top: Vec<&Reason> = match reason { And(rs) => rs.iter().collect(), r => vec![r] };
    let text = top.iter().map(|r| sentence(&join(&render(r, &rx)))).collect::<Vec<_>>().join(" ");
    let rated = |r: &Reason| matches!(r, Rated(_) | Only(_)) || matches!(r, Because(a, _) if matches!(**a, Rated(_)));
    if top.first().is_some_and(|r| rated(r)) { text } else { format!("{mv}: {text}") }
}

fn parse(fen: &str) -> Chess { Fen::from_ascii(fen.as_bytes()).unwrap().into_position(CastlingMode::Standard).unwrap() }

/// Every example: the comment, then the rendering.
pub fn report() {
    for (i, ex) in examples().iter().enumerate() {
        let q = if QUESTIONABLE.contains(&ex.fen) { " (questionable)" } else { "" };
        println!("#{i} {}{q}\n  comment: {}\n  render:  {}\n", ex.mov, ex.comment, render_reason(&parse(ex.fen), ex.mov, &ex.reason));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_fen(fen: &str) -> String {
        let ex = examples().into_iter().find(|e| e.fen == fen).unwrap();
        render_reason(&parse(ex.fen), ex.mov, &ex.reason)
    }

    /// Whole token: not inside a longer SAN or square ("g6" in "Qxg6+").
    fn has_token(text: &str, w: &str) -> bool {
        text.match_indices(w).any(|(i, _)| {
            let before = text[..i].chars().next_back();
            let after = text[i + w.len()..].chars().next();
            !before.is_some_and(|c| c.is_ascii_alphanumeric()) && !after.is_some_and(|c| c.is_ascii_alphanumeric())
        })
    }

    /// Moves and squares the reason names, except line ends that render as a whole file or rank.
    fn names(r: &Reason, out: &mut Vec<String>) {
        fn sqs(ss: &[Sq], out: &mut Vec<String>) { out.extend(ss.iter().map(|s| s.to_string())) }
        fn mvs(ms: &[Move_], out: &mut Vec<String>) { out.extend(ms.iter().filter(|m| **m != "--").map(|m| m.to_string())) }
        fn line(a: Sq, b: Sq, out: &mut Vec<String>) {
            let n = line_name(a, b);
            if n.contains(" from ") || n.ends_with("diagonal") { out.extend([a.to_string(), b.to_string()]) }
        }
        fn region(r: &Region, out: &mut Vec<String>) {
            match r { Region::Squares(ss) => sqs(ss, out), Region::Line(a, b) => line(a, b, out), _ => {} }
        }
        match r {
            And(rs) | Either(rs) => rs.iter().for_each(|r| names(r, out)),
            Threatens(l, r) | After(l, r) | Allows(l, r) => { mvs(l, out); names(r, out) }
            Instead(m, r) => { out.push(m.to_string()); names(r, out) }
            Prevents(r) | Removes(r) | Loses(r) | Gains(r) | Concedes(r) | Only(r) | NoMove(r) | Sacrifice(_, r) | Not(r)
            | Permanent(r) | Eventually(r) | More(r) | Degree(_, r) => names(r, out),
            Suppose(e, r) => { out.extend(e.iter().map(|(s, _)| s.to_string())); names(r, out) }
            Because(a, b) | But(a, b) | Outweighs(a, b) | Balances(a, b) | Faster(a, b) => { names(a, out); names(b, out) }
            Trade { give, get } => { sqs(give, out); sqs(get, out) }
            Attack { by, targets } => { out.push(by.to_string()); sqs(targets, out) }
            Pin { by, front, behind } => sqs(&[by, front, behind], out),
            XRay { by, through, target } => sqs(&[by, through, target], out),
            Defend { by, target } => sqs(&[by, target], out),
            Bound { piece, duties } => { out.push(piece.to_string()); sqs(duties, out) }
            Decoy { from, to } => sqs(&[from, to], out),
            Block { line: (a, b), by } => { line(a, b, out); out.push(by.to_string()) }
            Open((a, b)) => line(a, b, out),
            Control(g) | Pressure(g) | Outnumber(g) | Isolated(g) | Doubled(g) | Backward(g) | Hole(g) | Weakened(g)
            | Hanging(g) | Majority(g) => region(g, out),
            Develop(Pieces::At(s)) | Quality(s, _) | Passer(s) | Promote(s) => out.push(s.to_string()),
            Develop(Pieces::On(g)) => region(g, out),
            Better { pieces, than } => { sqs(pieces, out); sqs(than, out) }
            Coordinate(ss) | SameColour(ss) => sqs(ss, out),
            Enables(l) | Options(l) => mvs(l, out),
            _ => {}
        }
    }

    #[test]
    fn every_example_renders_and_names_everything() {
        let roles = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
        for ex in examples() {
            let text = render_reason(&parse(ex.fen), ex.mov, &ex.reason);
            for bad in ["Squares(", "Some(", "None", "{", "[", "  ", " ,", "::", "after which after", "not no longer", "can not", "1 pawns", "lured to the", "have developed", "have castled"]
                .into_iter().chain(roles) {
                assert!(!text.contains(bad), "{}: {bad:?} in {text}", ex.mov);
            }
            let (head, rest) = text.split_once(": ").unwrap_or(("", &text));
            assert!(head.is_empty() || !rest.starts_with(head), "{}: move named twice in {text}", ex.mov);
            for r in ["knight", "bishop", "rook", "queen", "king"] {
                assert!(!text.contains(&format!("the {r} on ")) || !text.split(&format!("the {r} on ")).skip(1).any(|t| t.get(2..).is_some_and(|t| t.starts_with(" is weakened"))),
                    "{}: a {r} is weakened in {text}", ex.mov);
            }
            for s in text.split(". ") {
                assert!(s.matches(':').count() <= 1, "{}: two colons in {s}", ex.mov);
            }
            for r in roles.map(|r| r.to_lowercase()) {
                for (i, _) in text.match_indices(&format!("the {r} on ")) {
                    let rest = &text[i + r.len() + 8..];
                    let listed = text[..i].ends_with("and ") || text[..i].ends_with(", ");
                    assert!(listed || !rest.get(2..7).is_some_and(|x| x == " are "), "{}: agreement in {text}", ex.mov);
                }
            }
            let mut want = vec![];
            names(&ex.reason, &mut want);
            for w in want {
                // A square may be named only by the move that lands on it ("allows 31...Be8, defending f7").
                let ok = if w.len() == 2 && w.as_bytes()[1].is_ascii_digit() { text.contains(&w) } else { has_token(&text, &w) };
                assert!(ok, "{}: {w} missing from {text}", ex.mov);
            }
        }
    }

    #[test]
    fn goldens() {
        assert_eq!(
            by_fen("rn1qk2r/ppp1b1p1/2b1Bn1p/8/3PPP2/2N5/PP4PP/R1BQK2R b KQkq - 0 11"),
            "11...Nxe4 is a blunder: Black allows 12.Qh5+ g6 13.Qxg6+ Kf8 14.Qf7#, mate. 11...Bd7 was stronger: Black controls h5. \
             11...Nbd7 was stronger: Black develops the knight on d7 and controls h5, and White is a tempo ahead."
        );
        assert_eq!(
            by_fen("4r1k1/2pq2Bp/1pNp2n1/1P4R1/8/1b2P1P1/6BP/R5K1 w - - 1 31"),
            "31.Bh6: Black can no longer play h6. The rook on g5 pins the knight on g6 to the king on g8. White plans h4, h5 and hxg6, winning a knight."
        );
        assert_eq!(
            by_fen("2kr3r/ppp3pp/2n2p2/2bqp3/6bB/3P1N2/PPP1BPPP/R2Q1RK1 w - - 2 11"),
            "11.Qd2 is a mistake: White develops the queen, but does not stop the threat of g5, attacking the bishop on h4. \
             11.h3 was best: the pawn on h3 attacks the bishop on g4, and White stops the threat of g5, attacking the bishop on h4."
        );
        assert_eq!(
            by_fen("r1bqk1nr/pppp1ppp/2n5/4p3/1bP5/2N3P1/PP1PPPBP/R1BQK1NR b KQkq - 2 4"),
            "4...Bxc3: After 5.bxc3, Black no longer has the bishop pair. Black trades the b4-bishop for the c3-knight. \
             The knight on c3 no longer defends d5. After 5.bxc3, the pawns on c3 and c4 are doubled."
        );
        assert_eq!(
            by_fen("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2"),
            "2.Bg5: White threatens Bxf6 exf6, after which White trades the c1-bishop for the f6-knight, and the pawns on f6 and f7 are doubled. \
             White allows 2...Ne4, after which the knight on e4 attacks the bishop on g5, and Black loses a tempo. \
             White allows 2...e6, after which the queen on d8 defends the knight on f6. White allows 2...c5, attacking the pawn on d4. \
             White allows 2...d5, after which Black controls c4 and e4."
        );
    }

    #[test]
    fn illegal_line_renders_bare() {
        let root = Chess::default();
        let r = And(vec![After(vec!["Qh5"], Box::new(Attack { by: "h5", targets: vec!["f7"] }))]);
        assert_eq!(render_reason(&root, "e4", &r), "1.e4: After 1...Qh5, h5 attacks f7.");
    }
}
