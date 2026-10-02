//! 7. Initiative: one side's threats force the other to answer them.

use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Position, Role, Square};

use crate::Out;
use crate::geo::{Geo, val};

/// Enemy pieces `c` could win: hit by something cheaper, or hit and unguarded. The king only
/// counts against an unguarded piece.
fn targets(pos: &Chess, c: Color) -> Vec<Square> {
    let g = Geo { b: pos.board().clone() };
    g.pieces(c.other()).into_iter().filter(|&(s, role)| {
        let att = g.attackers(s, c);
        let guarded = g.attackers(s, c.other()).any();
        att.any() && (!guarded || att.into_iter().any(|a| g.b.role_at(a).is_some_and(|r| r != Role::King && val(r) < val(role))))
    }).map(|x| x.0).collect()
}

/// What `c` threatens in `pos` if it could move now: mate in one, or a piece it could win.
/// A check is a threat when `c` is the one giving it.
fn threat(pos: &Chess, c: Color) -> Option<String> {
    if pos.turn() != c && pos.is_check() { return Some("gives check".into()); }
    if let Some(&s) = targets(pos, c).first() {
        let role = pos.board().role_at(s).unwrap();
        return Some(format!("threatens the {} on {s}", format!("{role:?}").to_lowercase()));
    }
    mate_threat(pos, c).map(|m| format!("threatens {m}"))
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

/// The side not to move has the initiative when it threatens something now: the side to move
/// must answer. Otherwise `line` is the engine's principal variation: the side to move has the
/// initiative when its best move leaves a threat that the reply answers without a threat of its
/// own; a reply that does carry a threat hands the initiative to the opponent.
pub fn run(pos: &Chess, line: &[String], o: &mut Out) {
    let c = pos.turn();
    let title = |c: Color| if c.is_white() { "White" } else { "Black" };
    if let Some(t) = threat(pos, c.other()) {
        o.side("initiative", c.other(), true, vec![], format!("{} {t}; {} must answer", title(c.other()), title(c)));
        o.side("initiative", c, false, vec![], format!("{} must answer a threat", title(c)));
        return;
    }
    let (holder, why) = match walk(pos, line) {
        Err(e) => (None, e),
        Ok(None) => (None, "the best move leaves no threat".to_string()),
        Ok(Some(((m1, t1), None))) => (Some(c), format!("{m1}: {} {t1}", title(c))),
        Ok(Some(((m1, t1), Some((m2, t2))))) =>
            (Some(c.other()), format!("{m1}: {} {t1}, but {m2}: {} {t2}", title(c), title(c.other()))),
    };
    for side in [c, c.other()] {
        o.side("initiative", side, holder == Some(side), vec![], why.clone());
    }
}

type Step = (String, String);

/// The threat the best move leaves, and the threat the reply leaves in turn.
fn walk(pos: &Chess, line: &[String]) -> Result<Option<(Step, Option<Step>)>, String> {
    let Some(u1) = line.first() else { return Err("no engine move".into()) };
    let (san1, p1) = play(pos, u1)?;
    let Some(t1) = threat(&p1, pos.turn()) else { return Ok(None) };
    let reply = match line.get(1) {
        Some(u2) => { let (san2, p2) = play(&p1, u2)?; threat(&p2, p1.turn()).map(|t2| (san2, t2)) }
        None => None,
    };
    Ok(Some(((san1, t1), reply)))
}

fn play(pos: &Chess, uci: &str) -> Result<(String, Chess), String> {
    let m = uci.parse::<UciMove>().map_err(|e| format!("{uci}: {e}"))?.to_move(pos).map_err(|e| format!("{uci}: {e}"))?;
    let san = shakmaty::san::San::from_move(pos, m).to_string();
    Ok((san, pos.clone().play(m).map_err(|e| e.to_string())?))
}
