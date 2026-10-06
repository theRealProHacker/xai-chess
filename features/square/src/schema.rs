//! The per-square vector: named groups of bits, each a product of a few dimensions.

pub const TYPES: [&str; 6] = ["pawn", "knight", "bishop", "rook", "queen", "king"];
pub const DIRS: [(i32, i32); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
const DIR_NAMES: [&str; 8] = ["n", "ne", "e", "se", "s", "sw", "w", "nw"];
pub const PLIES: usize = 15;

#[derive(Clone, Copy)]
pub enum Dim {
    /// a1..h8
    Sq,
    /// pawn..king
    Type,
    /// white_pawn..black_king
    Piece,
    /// ply 1..15 of a line
    Ply,
    /// 0..14 moves
    Moves,
    Dir,
    /// distance pairs on one ray: d1 ≥ 1, d2 ≥ 1, d1 + d2 ≤ 7
    Pair,
    File,
    /// rank 1..8, from White's side
    Rank,
    Named(&'static [&'static str]),
}

impl Dim {
    pub fn size(self) -> usize {
        match self {
            Dim::Sq => 64, Dim::Type => 6, Dim::Piece => 12, Dim::Ply | Dim::Moves => PLIES, Dim::Dir => 8,
            Dim::Pair => 21, Dim::File | Dim::Rank => 8, Dim::Named(v) => v.len(),
        }
    }
    fn label(self, i: usize) -> String {
        match self {
            Dim::Sq => shakmaty::Square::new(i as u32).to_string(),
            Dim::Type => TYPES[i].into(),
            Dim::Piece => format!("{}_{}", if i < 6 { "white" } else { "black" }, TYPES[i % 6]),
            Dim::Ply => format!("ply{}", i + 1),
            Dim::Moves => format!("in{i}"),
            Dim::Dir => DIR_NAMES[i].into(),
            Dim::Pair => { let (a, b) = unpair(i); format!("{a}_{b}") }
            Dim::File => format!("{}", (b'a' + i as u8) as char),
            Dim::Rank => format!("rank{}", i + 1),
            Dim::Named(v) => v[i].into(),
        }
    }
}

pub fn pair(d1: usize, d2: usize) -> usize { (1..d1).map(|k| 7 - k).sum::<usize>() + d2 - 1 }
fn unpair(i: usize) -> (usize, usize) {
    for d1 in 1..=6 { for d2 in 1..=7 - d1 { if pair(d1, d2) == i { return (d1, d2); } } }
    unreachable!()
}

const ON: &[&str] = &["on"];
const DIAGONALS: &[&str] = &[
    "a8", "a7b8", "a6c8", "a5d8", "a4e8", "a3f8", "a2g8", "a1h8", "b1h7", "c1h6", "d1h5", "e1h4", "f1h3", "g1h2", "h1",
    "a1", "a2b1", "a3c1", "a4d1", "a5e1", "a6f1", "a7g1", "a8h1", "b8h2", "c8h3", "d8h4", "e8h5", "f8h6", "g8h7", "h8"];
const LINES: &[&str] = &["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7", "rank8",
    "file_a", "file_b", "file_c", "file_d", "file_e", "file_f", "file_g", "file_h"];

use Dim::*;

/// Every group in vector order. A bit's index is the group's offset plus its mixed-radix index.
pub const GROUPS: &[(&str, &[Dim])] = &[
    // tactics, from the engine's best line
    ("mate", &[Ply]),
    ("material_win", &[Ply]),
    ("xray", &[Type, Type, Dir, Pair]),
    ("check", &[Named(ON)]),
    ("threat.mate", &[Ply]),
    ("threat.material_win", &[Ply]),
    ("threat.xray", &[Type, Type, Dir, Pair]),
    ("threat.check", &[Named(ON)]),
    ("threat.move", &[Sq]),
    // material
    ("occupant", &[Piece]),
    ("material.up", &[Named(&["up1", "up2", "up3plus"])]),
    ("material.trade", &[Named(&features::material::KINDS)]),
    ("material.coordinating", &[Named(ON)]),
    ("bishop_pair", &[Sq]),
    ("unlike_bishops", &[Sq]),
    // stability
    ("stability.pawn_ever", &[Sq]),
    ("stability.pawn_direct", &[Sq]),
    ("stability.attacked_in", &[Moves, Type, Sq]),
    // control and mobility
    ("control.reach", &[Sq]),
    ("control.geometry", &[Sq]),
    ("mobility.abs", &[Sq]),
    ("mobility.rel", &[Sq]),
    ("route.abs", &[Sq, Sq]),
    ("route.rel", &[Sq, Sq]),
    ("blocked_by.own", &[Type, Sq]),
    ("blocked_by.enemy", &[Type, Sq]),
    // the square itself
    ("centralisation", &[Named(&["edge0", "edge1", "edge2", "edge3"])]),
    ("square.control", &[Named(&["equal", "white1", "white2", "white3plus", "black1", "black2", "black3plus"])]),
    ("pawn_cover.white", &[Named(&["none", "one", "two"])]),
    ("pawn_cover.black", &[Named(&["none", "one", "two"])]),
    ("pawn_cover.white_ever", &[Named(ON)]),
    ("pawn_cover.black_ever", &[Named(ON)]),
    ("weak", &[Named(&["white", "black"])]),
    ("strong.white", &[Named(&["one", "two"])]),
    ("strong.black", &[Named(&["one", "two"])]),
    ("open_file", &[File]),
    ("half_open_file", &[File]),
    ("long_diagonal", &[Named(&["a1h8", "h1a8"])]),
    // role
    ("role.defends", &[Type, Sq]),
    ("role.needed_for", &[Type, Sq]),
    ("role.attacks", &[Type, Sq]),
    ("role.restricts", &[Type, Sq]),
    ("role.needs", &[Type, Sq]),
    ("role.protected_by", &[Type, Sq]),
    ("blocks.line", &[Named(LINES)]),
    ("blocks.diagonal", &[Named(DIAGONALS)]),
    ("blocks.pawn", &[Sq]),
    ("pawn_shield", &[Sq, Dir]),
    // rook
    ("rook.open_file", &[File]),
    ("rook.seventh", &[Named(ON)]),
    ("rook.eighth", &[Named(ON)]),
    ("rook.battery", &[Named(&["rook", "queen"]), Sq]),
    ("rook.behind_passer", &[Sq]),
    ("rook.behind_hanging", &[Sq]),
    // pawn
    ("pawn.chain", &[Sq]),
    ("pawn.chain_pos", &[Named(&["front", "middle", "base"])]),
    ("pawn.chain_dir", &[Named(&["queenside", "centre", "kingside"])]),
    ("pawn.passed", &[Named(ON)]),
    ("pawn.hanging", &[Named(ON)]),
    ("pawn.isolated", &[Named(ON)]),
    ("pawn.doubled_with", &[Rank]),
    ("pawn.blocked_by", &[Named(&["ahead1", "ahead2", "ahead3", "ahead4", "ahead5", "ahead6"]), Type]),
    ("pawn.centre", &[Named(&features::centre::KINDS)]),
    ("pawn.rank4_6", &[Named(ON)]),
    ("pawn.rank7", &[Named(ON)]),
];

fn size(dims: &[Dim]) -> usize { dims.iter().map(|d| d.size()).product() }

pub fn offset(name: &str) -> (usize, &'static [Dim]) {
    let mut off = 0;
    for (n, dims) in GROUPS {
        if *n == name { return (off, dims); }
        off += size(dims);
    }
    panic!("no group {name}")
}

/// Bits per square.
pub fn len() -> usize { GROUPS.iter().map(|(_, d)| size(d)).sum() }

/// `group.label.label…` for a bit index.
pub fn label(mut i: usize) -> String {
    for (n, dims) in GROUPS {
        let s = size(dims);
        if i < s {
            let mut parts = vec![n.to_string()];
            let mut stride = s;
            for d in dims.iter() {
                stride /= d.size();
                parts.push(d.label(i / stride));
                i %= stride;
            }
            return parts.join(".");
        }
        i -= s;
    }
    panic!("bit out of range")
}

/// (name, offset, size) of every group.
pub fn groups() -> Vec<(&'static str, usize, usize)> {
    let mut off = 0;
    GROUPS.iter().map(|(n, d)| { let s = size(d); let o = off; off += s; (*n, o, s) }).collect()
}
