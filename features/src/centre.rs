//! 5. Centre type and centre control.

use shakmaty::Color;

use crate::Out;
use crate::geo::{Geo, fwd, name, on};

fn file_char(f: i32) -> char { (b'a' + f as u8) as char }

pub fn run(g: &Geo, o: &mut Out) {
    let has = |f: i32, c: Color| (0..8).any(|r| g.pawn(f, r, c));
    let (w, b) = (Color::White, Color::Black);

    // closed: on d and e a white pawn stands directly against a black one
    let rams: Vec<String> = [3, 4].iter().flat_map(|&f| (0..7).filter(move |&r| g.pawn(f, r, w) && g.pawn(f, r + 1, b))
        .flat_map(move |r| [name(f, r), name(f, r + 1)])).collect();
    let closed = [3, 4].iter().all(|&f| (0..7).any(|r| g.pawn(f, r, w) && g.pawn(f, r + 1, b)));

    // mobile: one side has free centre pawns, the other none on d or e
    let mobile = Color::ALL.into_iter().find(|&c| {
        let ps: Vec<_> = g.pawns(c).into_iter().filter(|p| (2..=5).contains(&p.0)).collect();
        ps.len() >= 2 && ps.iter().any(|p| p.0 == 3 || p.0 == 4)
            && ps.iter().all(|&(f, r)| on(f, r + fwd(c)) && g.piece(f, r + fwd(c)).is_none())
            && !has(3, c.other()) && !has(4, c.other())
    });

    let open: Vec<i32> = (2..=5).filter(|&f| !has(f, w) && !has(f, b)).collect();
    let half: Vec<i32> = (2..=5).filter(|&f| has(f, w) != has(f, b)).collect();
    let files = |v: &[i32]| v.iter().map(|&f| format!("{}-file", file_char(f))).collect::<Vec<_>>().join(", ");

    let (hit, sq, why) = if closed {
        (1, rams, "pawns locked against each other on the d- and e-files".to_string())
    } else if let Some(c) = mobile {
        let ps = g.pawns(c).into_iter().filter(|p| (2..=5).contains(&p.0)).map(|(f, r)| name(f, r)).collect::<Vec<_>>();
        (2, ps.clone(), format!("{} centre pawns {} can advance; no {} pawn on d or e",
            if c.is_white() { "White's" } else { "Black's" }, ps.join(" "), if c.is_white() { "black" } else { "white" }))
    } else if open.len() >= 2 {
        (0, vec![], format!("open: {}", files(&open)))
    } else if open.len() == 1 || half.len() >= 2 {
        (3, vec![], format!("{}{}", if open.is_empty() { String::new() } else { format!("open: {}; ", files(&open)) },
                            if half.is_empty() { String::new() } else { format!("half-open: {}", files(&half)) }))
    } else {
        (4, vec![], "pawns in contact, nothing fixed".to_string())
    };
    o.one_hot("centre.type", None, &["open", "closed", "mobile", "static", "dynamic"], hit, sq, &why);

    // control of d4 e4 d5 e5, in half points
    let mut score = [0i32; 2]; // in half points
    let mut parts = vec![];
    for (f, r) in [(3, 3), (4, 3), (3, 4), (4, 4)] {
        let mut n = g.control(f, r);
        if let Some(p) = g.piece(f, r) { if p.role == shakmaty::Role::Pawn { n[if p.color.is_white() { 0 } else { 1 }] += 1; } }
        let (s, who) = match (n[0], n[1]) {
            (a, 0) if a > 0 => (2, Some(0)),
            (0, b) if b > 0 => (2, Some(1)),
            (a, b) if a > b => (1, Some(0)),
            (a, b) if b > a => (1, Some(1)),
            _ => (0, None),
        };
        if let Some(i) = who { score[i] += s; }
        parts.push(format!("{} {}", name(f, r), match (s, who) { (2, Some(0)) => "White", (2, Some(1)) => "Black",
            (1, Some(0)) => "White ½", (1, Some(1)) => "Black ½", _ => "even" }));
    }
    let d = score[0] - score[1];
    let why = parts.join(", ");
    o.global("centre.control.equal", d == 0, vec![], why.clone());
    let buckets = ["plus0_5", "plus1", "plus1_5", "plus2", "plus2_5", "plus3", "plus3_5", "plus4"];
    for c in Color::ALL {
        let mine = if c.is_white() { d } else { -d };
        let hit = if mine > 0 { (mine - 1) as usize } else { 99 };
        o.one_hot("centre.control", Some(c), &buckets, hit, ["d4", "e4", "d5", "e5"].map(String::from).to_vec(), &why);
    }
}
