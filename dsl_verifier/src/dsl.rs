#![allow(dead_code)]

use shakmaty::{Chess, Role};

// Simple move
type Move_ = &'static str;

// A reason for a move, e.g. "this move mates in 2", or "this move threatens to win material".
enum Reason {
    And(Box<Reason>, Box<Reason>),
    Mate(Vec<Move_>),
    Threatens(Vec<Move_>, Box<Reason>),
    WinMaterial(Vec<Role>),
}

/// Reasoning about a move in a position
struct Reasoning {
    pos: Chess,
    mov: Move_,
    reason: Reason
}

fn simple_parse(fen: &str) -> Chess {
    use shakmaty::{CastlingMode, fen::Fen};
    Fen::from_ascii(fen.as_bytes())
        .unwrap()
        .into_position(CastlingMode::Standard)
        .unwrap()
}

// Plays a SAN line from `pos`, panicking on an illegal move.
// let mut play = |sans: &[&str]| {
//     let mut line = MoveList::new();
//     for san in sans {
//         let m = San::from_ascii(san.as_bytes()).unwrap().to_move(&pos).unwrap();
//         pos.play_unchecked(m);
//         line.push(m);
//     }
//     line
// };


#[test]
fn test() {
    // Fried Liver, 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5 d5 5.exd5 Nxd5 6.Nxf7 Kxf7 7.Qf3+ Kg8??
    let fen = "r1bq1bkr/ppp3pp/2n5/3np3/2B5/5Q2/PPPP1PPP/RNB1K2R w KQ - 2 8";
    let pos: Chess = simple_parse(fen);

    Reasoning {
        pos: pos.clone(),
        mov: "Qxd5+",
        reason: Reason::Mate(vec!["Qxd5", "Bxd5+", "Be6", "Bxe6#"]),
    };
}

