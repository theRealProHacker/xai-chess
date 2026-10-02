//! 2. Local superiority and 6. Space, both per sector.

use shakmaty::Color;

use crate::Out;
use crate::geo::{Geo, fr, name, names, rel, title};
use crate::pawns::SECTORS;

/// Pieces that stand on the sector's files or hit three of its squares in the opponent's half.
pub fn local(g: &Geo, o: &mut Out) {
    for (sector, files) in SECTORS {
        let forces = |c: Color| -> Vec<shakmaty::Square> {
            g.pieces(c).into_iter().map(|x| x.0).filter(|&s| {
                files.contains(&fr(s).0) || g.attacks(s).into_iter()
                    .filter(|&t| { let (f, r) = fr(t); files.contains(&f) && rel(c, r) >= 4 }).count() >= 3
            }).collect()
        };
        let (mut fw, mut fb) = (forces(Color::White), forces(Color::Black));
        fw.sort_by_key(|s| s.to_string()); fb.sort_by_key(|s| s.to_string());
        let d = fw.len() as i32 - fb.len() as i32;
        let why = format!("{sector}: White {} ({}), Black {} ({})", fw.len(), names(fw.clone()).join(" "), fb.len(), names(fb.clone()).join(" "));
        o.global(format!("local.{sector}.equal"), d == 0, vec![], why.clone());
        for c in Color::ALL {
            let mine = if c.is_white() { d } else { -d };
            let hit = if mine > 0 { (mine - 1).min(2) as usize } else { 99 };
            let sq = names(if c.is_white() { fw.clone() } else { fb.clone() });
            o.one_hot(&format!("local.{sector}"), Some(c), &["plus1", "plus2", "plus3plus"], hit, sq, &why);
        }
    }
}

pub fn run(g: &Geo, o: &mut Out) {
    for (sector, files) in SECTORS {
        let held = |c: Color| -> Vec<String> {
            files.clone().flat_map(|f| (0..8).map(move |r| (f, r)))
                .filter(|&(_, r)| rel(c, r) >= 4)
                .filter(|&(f, r)| g.holder(f, r) == Some(c)).map(|(f, r)| name(f, r)).collect()
        };
        let (hw, hb) = (held(Color::White), held(Color::Black));
        for c in Color::ALL {
            let (mine, theirs) = if c.is_white() { (&hw, &hb) } else { (&hb, &hw) };
            let on = mine.len() >= theirs.len() + 2;
            o.side(format!("space.{sector}"), c, on, if on { mine.clone() } else { vec![] },
                   format!("{sector}: {} holds {} squares in the opponent's half, {} holds {}", title(c), mine.len(),
                           title(c.other()), theirs.len()));
        }
    }
}
