//! 5. Pawn structure: islands, doubled, weak and passed pawns, mobile majorities.
//!
//! isolated, passed and backward follow commentary_gen/board_tools.pawn_structure_data and
//! docs/space.html, so all three agree on which pawn is which.

use shakmaty::Color;

use crate::Out;
use crate::geo::{Fr, Geo, fwd, name, on, rel};

pub const SECTORS: [(&str, std::ops::RangeInclusive<i32>); 3] = [("queenside", 0..=2), ("centre", 3..=4), ("kingside", 5..=7)];

pub fn isolated(g: &Geo, c: Color, f: i32) -> bool {
    !g.pawns(c).iter().any(|&(pf, _)| (pf - f).abs() == 1)
}

/// No enemy pawn ahead on this or a neighbouring file, and no own pawn ahead on this file.
pub fn passed(g: &Geo, c: Color, f: i32, r: i32) -> bool {
    let mut rr = r + fwd(c);
    while on(0, rr) {
        if g.pawn(f, rr, c) { return false; }
        for ff in f - 1..=f + 1 { if g.pawn(ff, rr, c.other()) { return false; } }
        rr += fwd(c);
    }
    true
}

/// Not isolated, no own pawn beside or behind it on a neighbouring file, and its stop square is
/// hit by an enemy pawn.
pub fn backward(g: &Geo, c: Color, f: i32, r: i32) -> bool {
    if isolated(g, c, f) { return false; }
    let support = g.pawns(c).iter().any(|&(pf, pr)| (pf - f).abs() == 1 && rel(c, pr) <= rel(c, r));
    let stop = r + fwd(c);
    !support && on(f, stop) && g.pawn_attacked(f, stop, c.other())
}

fn islands(g: &Geo, c: Color) -> Vec<Vec<i32>> {
    let files: Vec<bool> = (0..8).map(|f| g.pawns(c).iter().any(|&(pf, _)| pf == f)).collect();
    let mut out: Vec<Vec<i32>> = vec![];
    for f in 0..8 {
        if !files[f as usize] { continue; }
        if f > 0 && files[f as usize - 1] { out.last_mut().unwrap().push(f); } else { out.push(vec![f]); }
    }
    out
}

pub fn weak(g: &Geo, c: Color) -> Vec<Fr> {
    let them = c.other();
    g.pawns(c).into_iter().filter(|&(f, r)| {
        let kind = isolated(g, c, f) || backward(g, c, f, r);
        let guarded = g.pawn_attacked(f, r, c);
        let mut rr = r + fwd(c);
        let mut half_open = true;
        while on(0, rr) { if g.pawn(f, rr, them) { half_open = false; } rr += fwd(c); }
        let stop = r + fwd(c);
        let stuck = !on(f, stop) || g.piece(f, stop).is_some() || g.pawn_attacked(f, stop, them);
        kind && !guarded && half_open && stuck
    }).collect()
}

fn sqs(v: &[Fr]) -> Vec<String> { v.iter().map(|&(f, r)| name(f, r)).collect() }

pub fn run(g: &Geo, o: &mut Out) {
    for c in Color::ALL {
        let ps = g.pawns(c);
        let isl = islands(g, c);
        if ps.is_empty() {
            o.one_hot("pawns.islands", Some(c), &["1", "2", "3", "4plus"], 99, vec![], "");
        } else {
            let desc = isl.iter().map(|i| { let a = (b'a' + i[0] as u8) as char; let z = (b'a' + *i.last().unwrap() as u8) as char;
                if a == z { a.to_string() } else { format!("{a}–{z}") } }).collect::<Vec<_>>().join(", ");
            o.one_hot("pawns.islands", Some(c), &["1", "2", "3", "4plus"], (isl.len() - 1).min(3), sqs(&ps),
                      &format!("pawns on {desc}"));
        }

        let doubled: Vec<Fr> = ps.iter().copied().filter(|&(f, _)| ps.iter().filter(|p| p.0 == f).count() > 1).collect();
        let extra: usize = (0..8).map(|f| ps.iter().filter(|p| p.0 == f).count().saturating_sub(1)).sum();
        o.one_hot("pawns.doubled", Some(c), &["0", "1", "2", "3plus"], extra.min(3), sqs(&doubled),
                  &if extra == 0 { "no two pawns on one file".to_string() } else { format!("{extra} extra pawn{} on a file: {}", if extra == 1 { "" } else { "s" }, sqs(&doubled).join(" ")) });

        let w = weak(g, c);
        o.one_hot("pawns.weak", Some(c), &["0", "1", "2", "3plus"], w.len().min(3), sqs(&w),
                  &if w.is_empty() { "no weak pawn".to_string() } else { format!("weak: {}", sqs(&w).join(" ")) });

        let pp: Vec<Fr> = ps.iter().copied().filter(|&(f, r)| passed(g, c, f, r)).collect();
        let hit = if pp.is_empty() { 99 } else { (pp.len() - 1).min(2) };
        o.one_hot("pawns.passed", Some(c), &["1", "2", "3plus"], hit, sqs(&pp), &if pp.is_empty() { "no passed pawn".to_string() } else { format!("passed: {}", sqs(&pp).join(" ")) });
        let prot: Vec<Fr> = pp.iter().copied().filter(|&(f, r)| g.pawn_attacked(f, r, c)).collect();
        o.side("pawns.passed.protected", c, !prot.is_empty(), sqs(&prot),
               if prot.is_empty() { String::new() } else { format!("pawn-guarded passer: {}", sqs(&prot).join(" ")) });

        for (sector, files) in SECTORS {
            let mine: Vec<Fr> = ps.iter().copied().filter(|p| files.contains(&p.0)).collect();
            let theirs = g.pawns(c.other()).iter().filter(|p| files.contains(&p.0)).count();
            let free = mine.iter().any(|&(f, r)| on(f, r + fwd(c)) && g.piece(f, r + fwd(c)).is_none());
            let on_ = mine.len() > theirs && free;
            o.side(format!("pawns.majority.{sector}"), c, on_, if on_ { sqs(&mine) } else { vec![] },
                   format!("{} against {} pawns{}", mine.len(), theirs, if mine.len() > theirs && !free { ", none can advance" } else { "" }));
        }
    }
}
