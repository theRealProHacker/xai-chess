//! Binary positional features from journey.md ("Features"), one module per section.
//!
//! Every feature is a yes/no about one side (`side` is "white" or "black") or about the whole
//! position (`side` is None). Counts and scores are one-hot: "three pawn islands" is its own
//! feature, so exactly one of a count family is on per side.

mod centre;
mod context;
mod geo;
mod initiative;
mod king;
mod material;
mod pawns;
mod pieces;
mod space;
#[cfg(target_arch = "wasm32")]
mod web;

use serde::Serialize;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Color, Position};

/// The researched cases shown on docs/features.html; `tests/examples.rs` holds the detector to them.
pub const EXAMPLES: &str = include_str!("../examples.json");

#[derive(Serialize, Debug, Clone)]
pub struct Feature {
    /// `family.value`, e.g. `pawns.islands.3`.
    pub id: String,
    pub side: Option<&'static str>,
    pub on: bool,
    /// Squares that make the feature true (or the closest it came), for highlighting.
    pub squares: Vec<String>,
    /// One line on why it is on or off.
    pub why: String,
}

#[derive(Serialize)]
pub struct Family {
    pub id: &'static str,
    pub title: &'static str,
    /// What the detector checks, in one or two sentences.
    pub rule: &'static str,
}

pub(crate) struct Out(pub Vec<Feature>);

impl Out {
    pub fn global(&mut self, id: impl Into<String>, on: bool, squares: Vec<String>, why: impl Into<String>) {
        self.0.push(Feature { id: id.into(), side: None, on, squares, why: why.into() });
    }
    pub fn side(&mut self, id: impl Into<String>, c: Color, on: bool, squares: Vec<String>, why: impl Into<String>) {
        self.0.push(Feature { id: id.into(), side: Some(geo::side(c)), on, squares, why: why.into() });
    }
    /// One-hot over `buckets`: the bucket `value` falls in is on, the rest off.
    pub fn one_hot(&mut self, family: &str, c: Option<Color>, buckets: &[&str], hit: usize, squares: Vec<String>, why: &str) {
        for (i, b) in buckets.iter().enumerate() {
            let on = i == hit;
            // every bucket carries the reason, so an off bucket still says what the rule saw
            let (sq, w) = (if on { squares.clone() } else { vec![] }, why.to_string());
            match c {
                Some(c) => self.side(format!("{family}.{b}"), c, on, sq, w),
                None => self.global(format!("{family}.{b}"), on, sq, w),
            }
        }
    }
}

pub fn position(fen: &str) -> Result<Chess, String> {
    Fen::from_ascii(fen.trim().as_bytes())
        .map_err(|e| format!("FEN: {e}"))?
        .into_position(CastlingMode::Standard)
        .map_err(|e| format!("FEN: {e}"))
}

/// Every feature of the position. `top` is the engine's principal variation (UCI) from the
/// position; without it the initiative family is left out.
pub fn extract(pos: &Chess, top: Option<&[String]>) -> Vec<Feature> {
    let g = geo::Geo { b: pos.board().clone() };
    let mut o = Out(vec![]);
    let phase = context::run(&g, pos, &mut o);
    material::run(&g, &mut o);
    space::local(&g, &mut o);
    king::run(&g, &mut o);
    pieces::run(&g, phase, &mut o);
    pawns::run(&g, &mut o);
    centre::run(&g, &mut o);
    space::run(&g, &mut o);
    space::pawn_advance(&g, &mut o);
    if let Some(top) = top { initiative::run(pos, top, &mut o); }
    o.0
}

pub fn extract_fen(fen: &str, top: Option<&[String]>) -> Result<Vec<Feature>, String> {
    Ok(extract(&position(fen)?, top))
}

/// The families in journey order, with the rule each detector applies.
pub fn families() -> Vec<Family> {
    let f = |id, title, rule| Family { id, title, rule };
    vec![
        f("context.phase", "Game phase",
          "As Lichess splits a game, counting queens, rooks and minor pieces: endgame at six or fewer; middlegame at ten or fewer, or once a back rank holds fewer than four pieces, or once many small areas of the board hold pieces of both colours; opening otherwise."),
        f("context.turn", "Side to move", "Read from the FEN."),
        f("material.balance", "Material balance",
          "Pawn 1, knight 3, bishop 3, rook 5, queen 9. Equal, or one side ahead by exactly 1, exactly 2, or 3 or more."),
        f("material.imbalance", "Unlike material",
          "Split both armies into like-for-like pairs and named trades, every piece used once: bishop against knight, rook against minor piece (the exchange), two minor pieces against a rook, two knights against a bishop, two rooks against a queen, three minor pieces against a queen, queen against rook and minor piece, or a minor piece against two or more pawns. Each trade in the split is on; one position can hold several. Where no split exists, the type is other."),
        f("material.bishop_pair", "Bishop pair", "A bishop on each square colour, while the opponent lacks one of the two."),
        f("material.opposite_bishops", "Unlike bishops", "Each side has exactly one bishop, and they run on different square colours."),
        f("local", "Local superiority",
          "In each sector (queenside a–c, centre d–e, kingside f–h) count the knights, bishops, rooks and queens that stand on its files or attack at least three of its squares in the opponent's half. A rook still on its back rank behind an own pawn does not count. The side with more is superior by the difference: 1, 2, or 3 or more."),
        f("king.castled", "King position", "Castled kingside: the king stands on g1, h1, g2 or h2 (g8, h8, g7, h7 for Black). Castled queenside: on the a- to c-file, first or second rank. Uncastled: anywhere else."),
        f("king.shield", "Pawn shield breached",
          "For a castled king. Kingside: breached when the g- or the h-file holds no own pawn, or when at most one of the pawns on f2, g2 and h2 (f7, g7, h7) is left. Queenside the same with the b- and a-file and a2, b2, c2."),
        f("king.attack", "Attackers and defenders",
          "On the king's sector, counted as for local superiority. Attackers: enemy knights, bishops, rooks and queens that stand on the sector's files, hit three of its squares in the king's half, or hit a square next to the king. Defenders: own pieces that stand on the sector's files or hit a square next to the king. A rook still on its back rank behind an own pawn counts only if it hits a square next to the king. The difference decides: more defenders, equal, or 1, 2, 3, or at least 4 more attackers."),
        f("pieces.out_of_play", "Piece out of play",
          "A knight, bishop, rook or queen that has at most two safe moves, hits no enemy piece or pawn, no square of the central sixteen (c3–f6) and no square next to the enemy king. Lines run through its own pieces other than pawns and the king, which can step aside. A rook behind an own advanced pawn that can step forward is in play. In the opening, pieces on the back rank are left to development."),
        f("pieces.knight_outpost", "Knight on an outpost",
          "A knight on its 4th, 5th or 6th rank, guarded by an own pawn, that no enemy pawn can come to attack: every enemy pawn on a neighbouring file ahead of it is blocked by a pawn on its way."),
        f("pieces.bishop_diagonal", "Bishop on an open diagonal",
          "A bishop that reaches at least five squares along one diagonal, at least three of them in the opponent's half."),
        f("pieces.bad_bishop", "Bad bishop", "A bishop with at least two own centre pawns (c- to f-file) on its square colour that are blocked by enemy pawns and standing ahead of it."),
        f("pieces.rook_file", "Rook on an open file", "Open: no pawn on the rook's file, and up the file the rook reaches a square on the opponent's second or back rank that no enemy piece or pawn attacks. Half-open: only enemy pawns on it, none of them guarded by a pawn."),
        f("pieces.rook_seventh", "Rook on the seventh", "A rook on the opponent's second rank."),
        f("pieces.rook_behind_passer", "Rook behind a passed pawn", "A rook on the file of its own passed pawn, behind it."),
        f("pieces.rook_blocked", "Rook behind blocked pawns", "A rook whose file holds an own pawn, in front of it, that an enemy pawn blocks or that is backward (its advance square hit by an enemy pawn and no own pawn beside or behind it to support it)."),
        f("pieces.queen", "Safe and active queen",
          "The queen hits at least five squares in the opponent's half, and no enemy pawn, minor piece or rook attacks it."),
        f("pieces.development", "Development lead",
          "Opening only. Count knights and bishops off their home squares, plus one for a castled king. The lead is the difference: equal, 1, 2, or 3 or more."),
        f("pieces.coordination", "Coordination",
          "Two proxies for the grade the journey asks for: rooks connected (two rooks on one rank or file with nothing between), and a loose piece (a knight, bishop, rook or queen nothing guards)."),
        f("pawns.islands", "Pawn islands", "Groups of own pawns on adjacent files: 1, 2, 3, or 4 or more."),
        f("pawns.doubled", "Doubled pawns", "Pawns beyond the first on a file, summed over files: 0, 1, 2, or 3 or more."),
        f("pawns.weak", "Weak pawns",
          "Isolated or backward pawns that no pawn guards, that the opponent can attack down a file free of his own pawns, and that cannot advance: the square in front is occupied or hit by an enemy pawn. Counted 0, 1, 2, 3+."),
        f("pawns.passed", "Passed pawns",
          "No enemy pawn in front on its own or a neighbouring file, and no own pawn in front. Counted 1, 2, 3+, plus whether one is guarded by a pawn."),
        f("centre.type", "Centre type",
          "Closed: on the d- and the e-file a white pawn stands directly against a black one. Mobile: one side has two or more centre pawns (c–f files, one on d or e) that can advance and the other has none on d or e. Open: two files from c to f without pawns. Static: one such file, or two half-open ones. Dynamic: anything else."),
        f("centre.control", "Centre control",
          "On d4, e4, d5 and e5, count each side's attackers, with a pawn standing on the square as one more. A side takes a full point for a square only it attacks, half a point for one both attack where it has more. The score is the difference, from 0 to 4 in half points."),
        f("pawns.majority", "Mobile pawn majority",
          "More pawns than the opponent on a sector (queenside a–c, centre d–e, kingside f–h), at least one of them free to advance."),
        f("space", "Space",
          "In each sector, count the squares in the opponent's half a side holds (it attacks them more often, batteries included, as on the Space demo). Two or more squares more than the opponent is more space."),
        f("space_pawns", "Space (pawn advance)",
          "Count each side's pawns on its 4th, 5th and 6th rank. The side with more leads by the difference: 1, 2, or 3 or more."),
        f("initiative", "Initiative",
          "The side with the initiative makes the other answer its threats. The side not to move has it when it threatens something now: a check, mate in one, or to win a piece (hit by something cheaper, or hit and unguarded). Otherwise the engine's best move decides: if it leaves such a threat and the best reply answers without a threat of its own, the side to move has the initiative; if the reply carries a threat, the initiative passes to the other side."),
    ]
}
