//! Board geometry shared by every feature: squares as (file, rank) pairs, 0..8 each.

use shakmaty::{Bitboard, Board, Color, File, Piece, Rank, Role, Square};

pub type Fr = (i32, i32);

pub fn on(f: i32, r: i32) -> bool { (0..8).contains(&f) && (0..8).contains(&r) }
pub fn sq(f: i32, r: i32) -> Square { Square::from_coords(File::new(f as u32), Rank::new(r as u32)) }
pub fn fr(s: Square) -> Fr { (s.file() as i32, s.rank() as i32) }
pub fn name(f: i32, r: i32) -> String { sq(f, r).to_string() }
pub fn names(xs: impl IntoIterator<Item = Square>) -> Vec<String> { xs.into_iter().map(|s| s.to_string()).collect() }

pub fn val(r: Role) -> i32 {
    match r { Role::Pawn => 1, Role::Knight | Role::Bishop => 3, Role::Rook => 5, Role::Queen => 9, Role::King => 0 }
}

/// Pawns move this way along the ranks.
pub fn fwd(c: Color) -> i32 { if c.is_white() { 1 } else { -1 } }
/// Rank counted from `c`'s own back rank: 0 is home, 7 is promotion.
pub fn rel(c: Color, r: i32) -> i32 { if c.is_white() { r } else { 7 - r } }
pub fn side(c: Color) -> &'static str { if c.is_white() { "white" } else { "black" } }
pub fn title(c: Color) -> &'static str { if c.is_white() { "White" } else { "Black" } }

pub const MINORS_AND_MAJORS: [Role; 4] = [Role::Queen, Role::Rook, Role::Bishop, Role::Knight];

pub struct Geo {
    pub b: Board,
}

impl Geo {
    pub fn piece(&self, f: i32, r: i32) -> Option<Piece> { if on(f, r) { self.b.piece_at(sq(f, r)) } else { None } }
    pub fn is(&self, f: i32, r: i32, c: Color, role: Role) -> bool { self.piece(f, r) == Some(Piece { color: c, role }) }
    pub fn pawn(&self, f: i32, r: i32, c: Color) -> bool { self.is(f, r, c, Role::Pawn) }

    pub fn pawns(&self, c: Color) -> Vec<Fr> { (self.b.pawns() & self.b.by_color(c)).into_iter().map(fr).collect() }
    /// Knights, bishops, rooks and queens of `c`.
    pub fn pieces(&self, c: Color) -> Vec<(Square, Role)> {
        let mut out = vec![];
        for role in MINORS_AND_MAJORS {
            for s in self.b.by_piece(Piece { color: c, role }) { out.push((s, role)); }
        }
        out
    }
    pub fn count(&self, c: Color, role: Role) -> usize { self.b.by_piece(Piece { color: c, role }).count() }
    pub fn king(&self, c: Color) -> Option<Square> { self.b.king_of(c) }

    pub fn attacks(&self, s: Square) -> Bitboard { self.b.attacks_from(s) }
    pub fn attackers(&self, s: Square, c: Color) -> Bitboard { self.b.attacks_to(s, c, self.b.occupied()) }
    pub fn pawn_attacked(&self, f: i32, r: i32, by: Color) -> bool {
        let back = -fwd(by);
        self.pawn(f - 1, r + back, by) || self.pawn(f + 1, r + back, by)
    }

    /// Attackers of a square per colour, counting batteries: a slider behind a friendly slider (or
    /// behind the pawn that attacks the square) on the same line also bears on it. The same rule
    /// as `attackersOf` in docs/space.html, so the two demos agree square for square.
    pub fn control(&self, f: i32, r: i32) -> [u32; 2] {
        let mut n = [0u32; 2];
        let ix = |c: Color| if c.is_white() { 0 } else { 1 };
        for df in [-1, 1] {
            if self.pawn(f + df, r - 1, Color::White) { n[0] += 1; }
            if self.pawn(f + df, r + 1, Color::Black) { n[1] += 1; }
        }
        for (df, dr) in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
            if let Some(p) = self.piece(f + df, r + dr) { if p.role == Role::Knight { n[ix(p.color)] += 1; } }
        }
        for df in -1..=1 { for dr in -1..=1 {
            if (df, dr) == (0, 0) { continue; }
            if let Some(p) = self.piece(f + df, r + dr) { if p.role == Role::King { n[ix(p.color)] += 1; } }
        }}
        for (dirs, straight) in [([(1, 0), (-1, 0), (0, 1), (0, -1)], true), ([(1, 1), (1, -1), (-1, 1), (-1, -1)], false)] {
            for (df, dr) in dirs {
                let (mut ff, mut rr, mut line, mut step) = (f + df, r + dr, None::<Color>, 1);
                while on(ff, rr) {
                    if let Some(p) = self.piece(ff, rr) {
                        if line.is_some_and(|l| l != p.color) { break; }
                        let slider = p.role == Role::Queen || p.role == if straight { Role::Rook } else { Role::Bishop };
                        if slider {
                            n[ix(p.color)] += 1;
                            line = Some(p.color);
                        } else if !straight && step == 1 && p.role == Role::Pawn && dr == -fwd(p.color) {
                            line = Some(p.color);
                        } else {
                            break;
                        }
                    }
                    ff += df; rr += dr; step += 1;
                }
            }
        }
        n
    }

    /// Who holds a square: the side with more attackers.
    pub fn holder(&self, f: i32, r: i32) -> Option<Color> {
        let [w, b] = self.control(f, r);
        if w > b { Some(Color::White) } else if b > w { Some(Color::Black) } else { None }
    }
}
