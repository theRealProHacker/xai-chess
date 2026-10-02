//! 7. Initiative: the side to move's best moves all force the opponent's hand.

use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Position, Square};

use crate::Out;
use crate::geo::{Geo, val};

/// Enemy pieces `c` could win: hit by something cheaper, or hit and unguarded.
fn targets(pos: &Chess, c: Color) -> Vec<Square> {
    let g = Geo { b: pos.board().clone() };
    g.pieces(c.other()).into_iter().filter(|&(s, role)| {
        let att = g.attackers(s, c);
        att.any() && (att.into_iter().any(|a| g.b.role_at(a).is_some_and(|r| val(r) < val(role))) || g.attackers(s, c.other()).is_empty())
    }).map(|x| x.0).collect()
}

/// Can `c` mate in one if the opponent passes?
fn mate_threat(pos: &Chess, c: Color) -> Option<String> {
    let mut setup = pos.to_setup(EnPassantMode::Legal);
    setup.turn = c;
    setup.ep_square = None;
    let p: Chess = setup.position(CastlingMode::Standard).ok()?;
    p.legal_moves().into_iter().find(|m| p.clone().play(*m).is_ok_and(|q| q.is_checkmate()))
        .map(|m| shakmaty::san::San::from_move(&p, m).to_string())
}

/// Is the move a check, a capture, or a new threat? With the reason.
pub fn forcing(pos: &Chess, uci: &str) -> Result<(bool, String), String> {
    let m = uci.parse::<UciMove>().map_err(|e| format!("{uci}: {e}"))?.to_move(pos).map_err(|e| format!("{uci}: {e}"))?;
    let san = shakmaty::san::San::from_move(pos, m).to_string();
    let c = pos.turn();
    let after = pos.clone().play(m).map_err(|e| e.to_string())?;
    if after.is_check() { return Ok((true, format!("{san} checks"))); }
    if m.is_capture() { return Ok((true, format!("{san} captures"))); }
    let before = targets(pos, c);
    let new: Vec<_> = targets(&after, c).into_iter().filter(|s| !before.contains(s) || *s == m.to()).collect();
    if let Some(&s) = new.first() {
        let role = after.board().role_at(s).unwrap();
        return Ok((true, format!("{san} attacks the {} on {s}", format!("{role:?}").to_lowercase())));
    }
    if let Some(mate) = mate_threat(&after, c) { return Ok((true, format!("{san} threatens {mate}"))); }
    Ok((false, format!("{san} is quiet")))
}

pub fn run(pos: &Chess, top: &[String], o: &mut Out) {
    let c = pos.turn();
    let res: Vec<(bool, String)> = top.iter().filter_map(|u| forcing(pos, u).ok()).collect();
    let on = !res.is_empty() && res.iter().all(|r| r.0);
    let why = res.iter().map(|r| r.1.clone()).collect::<Vec<_>>().join("; ");
    o.side("initiative", c, on, vec![], why);
    o.side("initiative", c.other(), false, vec![], "not on move".to_string());
}
