//! One hand-checked position per group: a bit that must be on, and a neighbour that must be off.

use square::{Lines, chess, extract, schema::label};

fn at(fen: &str, best: &[&str], null: &[&str], s: &str) -> Vec<String> {
    let pos = chess(fen).unwrap();
    let (b, n): (Vec<String>, Vec<String>) = (best.iter().map(|x| x.to_string()).collect(), null.iter().map(|x| x.to_string()).collect());
    let lines = Lines { best: (!b.is_empty()).then_some(&b[..]), null: (!n.is_empty()).then_some(&n[..]) };
    let sq: shakmaty::Square = s.parse().unwrap();
    extract(&pos, &lines)[sq as usize].iter().map(|&i| label(i)).collect()
}

fn on(fen: &str, s: &str) -> Vec<String> { at(fen, &[], &[], s) }

#[track_caller]
fn has(v: &[String], yes: &[&str], no: &[&str]) {
    for y in yes { assert!(v.iter().any(|x| x == y), "missing {y} in {v:?}"); }
    for n in no { assert!(!v.iter().any(|x| x == n), "unexpected {n} in {v:?}"); }
}

const SCHOLAR: &str = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4";

#[test]
fn mate_and_material_lines() {
    has(&at(SCHOLAR, &["h5f7"], &[], "f7"), &["mate.ply1", "material_win.ply1"], &["mate.ply2"]);
    has(&at(SCHOLAR, &["h5f7"], &[], "e1"), &[], &["mate.ply1"]);
    // a quiet line is neither
    has(&at(SCHOLAR, &["g1f3"], &[], "f3"), &[], &["mate.ply1", "material_win.ply1"]);
}

#[test]
fn threat_after_null_move() {
    let v = at(SCHOLAR, &[], &["f6h5"], "f6");
    has(&v, &["threat.material_win.ply1", "threat.move.h5"], &["threat.mate.ply1", "threat.check.on"]);
}

#[test]
fn xray_and_check() {
    has(&on("k7/8/8/8/N7/8/8/R3K3 w - - 0 1", "a1"), &["xray.knight.king.n.3_4"], &["check.on"]);
    has(&on("k7/8/8/8/8/8/8/R3K3 b - - 0 1", "a1"), &["check.on"], &[]);
}

#[test]
fn material_groups() {
    let fen = "4k3/p7/8/8/8/8/PP6/4K3 w - - 0 1";
    has(&on(fen, "a2"), &["occupant.white_pawn", "material.up.up1"], &["material.up.up2"]);
    has(&on(fen, "a7"), &["occupant.black_pawn"], &["material.up.up1"]);
    // knight and bishop against a rook: the side with two minors keeps its rook to coordinate
    let fen = "r3k2r/8/8/8/8/8/8/1NB1K2R w - - 0 1";
    has(&on(fen, "b1"), &["material.trade.two_minors_vs_rook", "material.up.up1"], &["material.coordinating.on"]);
    has(&on(fen, "h1"), &["material.coordinating.on"], &["material.trade.two_minors_vs_rook"]);
    has(&on(fen, "a8"), &["material.trade.two_minors_vs_rook"], &["material.coordinating.on"]);
    has(&on("2b1k3/8/8/8/8/8/8/2B1KB2 w - - 0 1", "c1"), &["bishop_pair.f1"], &[]);
    has(&on("2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1", "c1"), &["unlike_bishops.c8"], &["bishop_pair.c8"]);
}

#[test]
fn stability() {
    // c7 can come to c6; e7 is stopped by the e6 pawn
    let v = on("4k3/2p1p3/4P3/3N4/8/8/8/4K3 w - - 0 1", "d5");
    has(&v, &["stability.pawn_ever.c7", "stability.pawn_direct.c7"], &["stability.pawn_ever.e7", "stability.pawn_direct.e7"]);
    // the bishop gets to e4 in one move and hits d5 from there
    has(&on("4k3/8/8/3N4/8/8/8/1b2K3 w - - 0 1", "d5"), &["stability.attacked_in.in1.bishop.b1"], &["stability.attacked_in.in0.bishop.b1"]);
}

#[test]
fn control_and_mobility() {
    let v = on("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1", "a1");
    has(&v, &["control.reach.e1", "blocked_by.own.pawn.a2", "blocked_by.own.king.e1"], &["mobility.abs.e1", "control.reach.a3"]);
    // through the own rook on a2
    has(&on("4k3/8/8/8/8/8/R7/R3K3 w - - 0 1", "a1"), &["control.geometry.a8"], &["control.reach.a3"]);
    // past the own pawn's attack square, no further
    has(&on("4k3/8/8/8/8/8/3P4/2B1K3 w - - 0 1", "c1"), &["control.geometry.e3"], &["control.geometry.f4", "control.reach.e3"]);
    // f3 and h3 are hit by the g4 pawn
    let v = on("4k3/8/8/8/6p1/8/8/4K1N1 w - - 0 1", "g1");
    has(&v, &["mobility.abs.f3", "mobility.abs.h3", "mobility.rel.e2", "route.abs.f5.e2", "route.abs.f5.f3", "route.rel.f5.e2"],
        &["mobility.rel.f3", "route.abs.f5.h3", "route.rel.f5.f3"]);
}

#[test]
fn the_square() {
    let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let v = on(start, "e3");
    has(&v, &["centralisation.edge2", "square.control.white3plus", "pawn_cover.white.two", "pawn_shield.e7.n", "pawn_shield.e2.s", "pawn_shield.a7.nw"],
        &["weak.white", "pawn_shield.h6.ne"]);
    has(&on(start, "d4"), &["pawn_cover.white_ever.on", "long_diagonal.a1h8"], &["weak.white"]);
    let v = on("4k3/8/8/3N4/2P1P3/8/8/4K3 w - - 0 1", "d5");
    has(&v, &["pawn_cover.white.two", "strong.white.two", "weak.black"], &["strong.white.one"]);
    has(&on("4k3/p7/8/8/8/8/8/R3K3 w - - 0 1", "a1"), &["half_open_file.a"], &["open_file.a", "rook.open_file.a"]);
}

#[test]
fn files_and_rooks() {
    has(&on("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1"), &["open_file.a", "rook.open_file.a"], &["half_open_file.a"]);
    has(&on("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1"), &["open_file.a"], &["rook.open_file.a"]);
    let fen = "4k3/7R/8/8/P7/8/8/R2QK3 w - - 0 1";
    has(&on(fen, "a1"), &["rook.battery.queen.d1", "rook.behind_passer.a4"], &["rook.seventh.on"]);
    has(&on(fen, "h7"), &["rook.seventh.on"], &["rook.eighth.on"]);
    let fen = "4k3/pp3ppp/8/8/2PP4/8/P4PPP/2R1K3 w - - 0 1";
    has(&on(fen, "c1"), &["rook.behind_hanging.c4"], &[]);
    has(&on(fen, "c4"), &["pawn.hanging.on"], &["pawn.isolated.on"]);
    has(&on("4k3/pp3ppp/8/8/2PP4/8/PP3PPP/2R1K3 w - - 0 1", "c4"), &[], &["pawn.hanging.on"]);
}

#[test]
fn role() {
    let fen = "4k3/8/5n2/8/4P3/2N5/8/4K3 w - - 0 1";
    has(&on(fen, "c3"), &["role.defends.pawn.e4", "role.needed_for.pawn.e4"], &[]);
    has(&on(fen, "e4"), &["role.protected_by.knight.c3", "role.needs.knight.c3", "blocks.line.rank4", "blocks.diagonal.a8h1"], &[]);
    has(&on(fen, "f6"), &["role.attacks.pawn.e4"], &["role.defends.pawn.e4"]);
    has(&on("4k3/8/8/4N3/8/8/8/4K3 w - - 0 1", "e5"), &["role.restricts.king.d7", "role.restricts.king.f7"], &["role.restricts.king.e7"]);
    has(&on("4k3/8/8/4n3/4P3/8/8/4K3 w - - 0 1", "e5"), &["blocks.pawn.e4"], &[]);
    has(&on("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1"), &[], &["blocks.line.rank1", "blocks.line.file_a"]);
}

#[test]
fn pawns() {
    let fen = "4k3/8/8/4P3/3P4/2P5/8/4K3 w - - 0 1";
    has(&on(fen, "c3"), &["pawn.chain.d4", "pawn.chain.e5", "pawn.chain_pos.base", "pawn.chain_dir.kingside"], &["pawn.chain_pos.front"]);
    has(&on(fen, "d4"), &["pawn.chain_pos.middle"], &[]);
    has(&on(fen, "e5"), &["pawn.chain_pos.front", "pawn.rank4_6.on", "pawn.passed.on", "pawn.blocked_by.ahead3.king"], &[]);
    has(&on("4k3/8/8/8/4P3/8/4P3/4K3 w - - 0 1", "e2"), &["pawn.doubled_with.rank4"], &["pawn.chain_pos.base"]);
    has(&on("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", "a2"), &["pawn.isolated.on"], &["pawn.centre.open"]);
}

#[test]
fn doubled_ranks_are_absolute() {
    has(&on("4k3/4p3/8/4p3/8/8/8/4K3 w - - 0 1", "e7"), &["pawn.doubled_with.rank5"], &["pawn.doubled_with.rank4"]);
}
