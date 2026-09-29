#![allow(dead_code)]

use shakmaty::Role;

/// SAN move. In a line, "--" is a pass.
pub(crate) type Move_ = &'static str;
/// Square such as "e4". A piece is named by its square where the reason is evaluated.
pub(crate) type Sq = &'static str;

/// Squares named the way players name them. `opponent` is relative to the reason's owner.
/// Listed `Squares` must each satisfy the reason. A named region is quantified the way players speak:
/// `Control` every square, `Isolated`, `Doubled`, `Backward` and `Hanging` some such opponent pawn or piece,
/// `Hole` some hole on the opponent's 3rd or 4th rank,
/// `Pressure` more squares aimed at than the opponent aims at, `Outnumber` more squares outnumbering than outnumbered.
#[derive(Debug)]
pub(crate) enum Region {
    Squares(Vec<Sq>),
    /// The f-, g- and h-files. The d- and e-files are the centre, as in "central majority".
    Kingside,
    /// The a-, b- and c-files.
    Queenside,
    /// d4, e4, d5, e5.
    Centre,
    /// c3 to f6.
    GreaterCentre,
    /// Ranks 1-4 of the owner, or of the opponent.
    Half { opponent: bool },
    /// The wing that king stands on: "Black's castled side". Neither wing while it is on the d- or e-file.
    KingWing { opponent: bool },
    /// The king's square and its neighbours.
    KingZone { opponent: bool },
    /// A file, rank or diagonal from one square to the other, both included.
    Line(Sq, Sq),
}

/// How good a move is, on the annotation scale (?? ? ?! !? ! !!).
#[derive(Debug)]
pub(crate) enum Annotation {
    Blunder,
    Mistake,
    Dubious,
    Interesting,
    Good,
    Brilliant,
    Best,
    /// Under `Instead`: the alternative is better than the played move.
    Stronger,
    /// Under `Instead`: the alternative is worse than the played move.
    Weaker,
}

/// How good a piece is: what it adds beyond its material. `Degree` scales it.
#[derive(Debug)]
pub(crate) enum Grade {
    Good,
    Bad,
}

/// How strongly a reason holds.
#[derive(Debug)]
pub(crate) enum Level {
    Slight,
    Clear,
    Decisive,
}

/// Which of the owner's knights, bishops, rooks and queens a reason is about.
#[derive(Debug)]
pub(crate) enum Pieces {
    /// Development in general: the knights and bishops.
    All,
    /// The piece on this square.
    At(Sq),
    /// Every piece of this kind: "the queen", "the rooks".
    Every(Role),
    /// The pieces that start in this region: "the kingside".
    On(Region),
}

/// A reason belongs to a side: the mover at the top, the opponent inside `Allows` and `Prevents`.
/// It is evaluated after the move (or after the line that wraps it).
/// A line starts with the side to move; `Threatens` and `Enables` let the owner move first.
#[derive(Debug)]
pub(crate) enum Reason {
    And(Vec<Reason>),
    /// At least one holds ("either trades queens or loses the pawn").
    Either(Vec<Reason>),
    /// The owner plays `line` unopposed, then the reason holds. Also multi-move plans.
    Threatens(Vec<Move_>, Box<Reason>),
    /// After this continuation the reason holds (forced replies, traps, "if ... then").
    After(Vec<Move_>, Box<Reason>),
    /// After `line` (empty: at once) the opponent has the reason.
    Allows(Vec<Move_>, Box<Reason>),
    /// An opponent reason that held before the move no longer does (prophylaxis, parrying).
    Prevents(Box<Reason>),
    /// An own reason that held before the move no longer does.
    Loses(Box<Reason>),
    /// An own reason that did not hold before the move now does (discovered attack, clearance).
    Gains(Box<Reason>),
    /// An opponent reason that did not hold before the move now does ("weakens d3").
    Concedes(Box<Reason>),
    /// No other move achieves the reason.
    Only(Box<Reason>),
    /// No move at all achieves the reason ("I search for the move that prevents both, but there is none").
    NoMove(Box<Reason>),
    /// What the alternative move would have achieved.
    Instead(Move_, Box<Reason>),
    /// Material given up for the reason.
    Sacrifice(Vec<Role>, Box<Reason>),
    /// The reason does not hold ("not really weak", "no way to attack it").
    Not(Box<Reason>),
    /// The first reason holds because of the second.
    Because(Box<Reason>, Box<Reason>),
    /// Both hold; the second goes against what the first suggests ("develops, but does not prevent g5").
    But(Box<Reason>, Box<Reason>),
    /// The owner's first reason counts for more than the opponent's second.
    Outweighs(Box<Reason>, Box<Reason>),
    /// The reason cannot be undone ("permanently weak", "pawns don't move backward").
    Permanent(Box<Reason>),
    /// Does not hold now, holds later: at the end of the best line ("in the long run the bishop is strong").
    Eventually(Box<Reason>),
    /// The owner's first reason and the opponent's second cancel out ("the big centre is balanced by the bishop").
    Balances(Box<Reason>, Box<Reason>),
    /// The owner has more of the reason than the opponent: `More(Develop(All))` is "ahead in development".
    More(Box<Reason>),
    /// How strongly the reason holds: "a slight weakness", "much more space", "a decisive attack".
    Degree(Level, Box<Reason>),
    /// The owner achieves the first reason before the opponent achieves the second.
    Faster(Box<Reason>, Box<Reason>),
    /// The reason would hold with these squares changed (FEN letter, or `None` for empty).
    Suppose(Vec<(Sq, Option<char>)>, Box<Reason>),

    /// How good the move is; `Because` gives why. Under `Instead`: how good the alternative is.
    Rated(Annotation),
    /// The side to move is in check.
    Check,
    /// The side to move is checkmated.
    Mate,
    WinMaterial(Vec<Role>),
    /// The owner is this many pawns ahead in material (N, B = 3, R = 5, Q = 9; negative: behind).
    Material(i8),
    /// Pieces named by their squares before the move.
    Trade { give: Vec<Sq>, get: Vec<Sq> },
    /// Trading pieces in general favours the owner.
    Simplify,
    /// One target is an attack, several a fork; `by` need not be the moved piece (discovery).
    Attack { by: Sq, targets: Vec<Sq> },
    /// `front` is the only piece between `by` and `behind`: pin, skewer, x-ray.
    Pin { by: Sq, front: Sq, behind: Sq },
    /// `by` reaches `target` through `through`, the only piece between them, which is the opponent's.
    /// Own piece on `target`: x-ray defence; otherwise x-ray attack.
    XRay { by: Sq, through: Sq, target: Sq },
    Defend { by: Sq, target: Sq },
    /// The piece is tied to defending every duty; several duties make it overloaded.
    Bound { piece: Sq, duties: Vec<Sq> },
    /// The piece on `from` is lured, or deflected, to `to`.
    Decoy { from: Sq, to: Sq },
    /// `by` stands between the two squares of the line.
    Block { line: (Sq, Sq), by: Sq },
    /// The line holds no pawn of the owner.
    Open((Sq, Sq)),
    Control(Region),
    /// Pieces aimed at these squares, short of a threat.
    Pressure(Region),
    /// More owner pieces bear on these squares than opponent pieces defend them.
    Outnumber(Region),
    /// Opponent pawns with no pawn of their own side on the files beside them.
    Isolated(Region),
    /// Opponent pawns sharing a file with another of their own.
    Doubled(Region),
    /// Opponent pawns with no own pawn beside or behind them, whose stop square an owner pawn guards.
    Backward(Region),
    /// Squares, or opponent pawns, that no opponent pawn can ever guard.
    Hole(Region),
    /// Pawns or squares one of whose neighbouring files has no opponent pawn left that can ever guard them
    /// (g6 after h7-h6). Easier to attack, not necessarily attackable now. `Hole` is the case with none on either file.
    Weakened(Region),
    /// Opponent pieces or pawns the owner wins material by capturing.
    Hanging(Region),
    /// The pieces are off their starting squares.
    Develop(Pieces),
    /// How good the piece is, whoever owns it. `Quality(s, Bad)` is a bad piece.
    Quality(Sq, Grade),
    /// These pieces are worth more than those here (good vs bad bishop, B vs N, 3 minors vs Q).
    Better { pieces: Vec<Sq>, than: Vec<Sq> },
    /// The pieces work together.
    Coordinate(Vec<Sq>),
    Castle,
    /// The owner is this many tempi ahead (negative: behind). `Loses(Tempo(1))` wastes one.
    Tempo(i8),
    /// The owner makes the forcing moves and the opponent keeps answering.
    Initiative,
    /// The owner controls more squares in the opponent's half than the opponent does in the owner's.
    Space,
    /// Unbalances the position: sharper, riskier for both sides.
    Imbalance,
    /// The centre is locked by pawns, few lines are open. `Not(Closed)` is an open position.
    Closed,
    BishopPair,
    Passer(Sq),
    /// Owner pawns outnumbering the opponent's on the region's files.
    Majority(Region),
    Promote(Sq),
    /// The owner can legally play the line next. Whether it loses material is a separate claim (`Hanging`).
    Enables(Vec<Move_>),
    /// The owner keeps the choice between these moves open (waiting, flexibility, tension).
    Options(Vec<Move_>),
    Zugzwang,
    Draw,
    Win,
    /// A practical gain, not better on the board: harder for the opponent to play, or saves the owner's clock.
    Practical,
    /// The pieces or squares stand on one square colour (bad bishop; `Not` for opposite-coloured bishops).
    SameColour(Vec<Sq>),
}

/// A reason given for `mov` in the position `fen`.
pub(crate) struct Reasoning {
    pub(crate) fen: &'static str,
    pub(crate) mov: Move_,
    pub(crate) comment: &'static str,
    pub(crate) reason: Reason,
}

use Level::*;
use Pieces::*;
use Annotation::*;
use Reason::*;
use Region::*;
use Role::*;

fn threatens(line: Vec<Move_>, r: Reason) -> Reason { Threatens(line, Box::new(r)) }
fn after(line: Vec<Move_>, r: Reason) -> Reason { After(line, Box::new(r)) }
fn allows(line: Vec<Move_>, r: Reason) -> Reason { Allows(line, Box::new(r)) }
fn prevents(r: Reason) -> Reason { Prevents(Box::new(r)) }
fn loses(r: Reason) -> Reason { Loses(Box::new(r)) }
fn gains(r: Reason) -> Reason { Gains(Box::new(r)) }
fn concedes(r: Reason) -> Reason { Concedes(Box::new(r)) }
fn only(r: Reason) -> Reason { Only(Box::new(r)) }
fn no_move(r: Reason) -> Reason { NoMove(Box::new(r)) }
fn instead(m: Move_, r: Reason) -> Reason { Instead(m, Box::new(r)) }
fn sacrifice(roles: Vec<Role>, r: Reason) -> Reason { Sacrifice(roles, Box::new(r)) }
fn not(r: Reason) -> Reason { Not(Box::new(r)) }
fn faster(a: Reason, b: Reason) -> Reason { Faster(Box::new(a), Box::new(b)) }
fn because(a: Reason, b: Reason) -> Reason { Because(Box::new(a), Box::new(b)) }
fn but(a: Reason, b: Reason) -> Reason { But(Box::new(a), Box::new(b)) }
fn outweighs(a: Reason, b: Reason) -> Reason { Outweighs(Box::new(a), Box::new(b)) }
fn balances(a: Reason, b: Reason) -> Reason { Balances(Box::new(a), Box::new(b)) }
fn more(r: Reason) -> Reason { More(Box::new(r)) }
fn degree(l: Level, r: Reason) -> Reason { Degree(l, Box::new(r)) }
fn permanent(r: Reason) -> Reason { Permanent(Box::new(r)) }
fn eventually(r: Reason) -> Reason { Eventually(Box::new(r)) }
fn suppose(edits: Vec<(Sq, Option<char>)>, r: Reason) -> Reason { Suppose(edits, Box::new(r)) }

/// One example per reason, taken from commentary_gen/data (train + dev).
pub(crate) fn examples() -> Vec<Reasoning> {
    vec![
        // dev-0167: Allows, Instead, Rated
        Reasoning {
            fen: "rn1qk2r/ppp1b1p1/2b1Bn1p/8/3PPP2/2N5/PP4PP/R1BQK2R b KQkq - 0 11",
            mov: "Nxe4",
            comment: r#"Black blunders by accepting the pawn and mate is forced - 12.Qh5+ g6 13.Qxg6+Kf8 14.Qf7#. (11.Bxe4 12.Nxe4 also looks weak for Black even if not immediatelylosing the game). Stronger would have been either 11...Bd7 or 11...Nbd7 todevelop the Queen’s pieces while still keeping control of h5. AlthoughBlack would have been up a Knight for two Pawns, White still would haveretained a strong initiative with mobility, space, and control with a complexand interesting game ahead."#,
            reason: And(vec![
                because(Rated(Blunder), allows(vec!["Qh5+", "g6", "Qxg6+", "Kf8", "Qf7#"], Mate)),
                instead("Bd7", because(Rated(Stronger), Control(Squares(vec!["h5"])))),
                instead("Nbd7", because(Rated(Stronger), And(vec![Develop(At("d7")), Control(Squares(vec!["h5"])), allows(vec![], Tempo(1))]))),
            ]),
            // Before Rated: the consequences, but neither "blunders" nor "stronger".
            // reason: And(vec![
            //     allows(vec!["Qh5+", "g6", "Qxg6+", "Kf8", "Qf7#"], Mate),
            //     instead("Bd7", Control(Squares(vec!["h5"]))),
            //     instead("Nbd7", And(vec![Develop(At("d7")), Control(Squares(vec!["h5"])), allows(vec![], Tempo(1))])),
            // ]),
        },
        // train-1179: Loses, Sacrifice, Block
        Reasoning {
            fen: "r1bqkb1r/pp3ppp/2n2n2/3p4/3p4/5NP1/PP1NPPBP/R1BQ1RK1 b kq - 1 8",
            mov: "d3",
            comment: r#"The attempt to protect this pawn with 8...Bc5 would be nothing but a waste of time. The advantage of the move played in the text is that the immediate capture of the pawn will close the "d" file preventing White from attacking the weak Pd5 with the Rooks"#,
            reason: And(vec![
                instead("Bc5", loses(Tempo(1))),
                sacrifice(vec![Pawn], after(vec!["exd3"], And(vec![
                    Block { line: ("d1", "d5"), by: "d3" },
                    prevents(Pressure(Squares(vec!["d5"]))),
                ]))),
            ]),
        },
        // train-0058: Only, Weak
        Reasoning {
            fen: "r1br2k1/ppq1bppp/2p2n2/3P4/3QN3/1P2PB2/PB3PPP/R2R2K1 b - - 0 15",
            mov: "Rxd5",
            comment: r#"essentially forced, since after cxd5 the isolated queen pawn becomes a hugetarget and of course Nxd5 loses to the mate on g7."#,
            reason: And(vec![
                only(WinMaterial(vec![Pawn])),
                instead("cxd5", allows(vec![], Isolated(Squares(vec!["d5"])))),
                instead("Nxd5", allows(vec!["Qxg7#"], Mate)),
            ]),
        },
        // train-1113: Bound
        Reasoning {
            fen: "r2q1rk1/pp3ppp/1n6/3n4/3P4/1QP5/P2BB1PP/R4RK1 b - - 0 16",
            mov: "Rc8",
            comment: r#"Bogo attempts totie up White's Queenside forces by attacking the weakened c3-pawn. Since thepawn is now attacked twice (by Black's d5-Knight and Rook), White's Queen andd2-Bishop are tied to its defense."#,
            reason: And(vec![
                Weakened(Squares(vec!["c3"])),
                Attack { by: "c8", targets: vec!["c3"] },
                Attack { by: "d5", targets: vec!["c3"] },
                Bound { piece: "b3", duties: vec!["c3"] },
                Bound { piece: "d2", duties: vec!["c3"] },
            ]),
        },
        // train-1245: Defend, Block, Control
        Reasoning {
            fen: "r1r3k1/3nqppp/2pbpn2/pp5b/2P1P3/PP3N1P/1BQNBPP1/R3R1K1 b - - 0 15",
            mov: "e5",
            comment: r#"The pawn is overprotected andshuts out the Bb2 from the game, as well as effectively controlling d4. Itscounterpart on e4 is weaker and can be pressured more effectively."#,
            reason: And(vec![
                Defend { by: "d6", target: "e5" },
                Defend { by: "d7", target: "e5" },
                Block { line: ("b2", "h8"), by: "e5" },
                Control(Squares(vec!["d4"])),
                Weakened(Squares(vec!["e4"])),
            ]),
        },
        // train-0701: Attack (fork), Prevents(Quality)
        Reasoning {
            fen: "r1b2rk1/1p3p1p/p2b1p2/3B4/3P4/1P2q3/PB1N2PP/R2Q3K w - - 0 18",
            mov: "Ne4",
            comment: r#"a strong follow-up, forking d6 and f6. In calculating it, I also noticedthe fact that the Black queen has very few squares left. My opponent now playsthe "obvious move", removing the Bd6 from threat and protecting f6, whichhowever loses."#,
            reason: And(vec![
                Attack { by: "e4", targets: vec!["d6", "f6"] },
                prevents(Quality("e3", Grade::Good)),
            ]),
        },
        // train-1277: Pin, Enables, Threatens a plan
        Reasoning {
            fen: "4r1k1/2pq2Bp/1pNp2n1/1P4R1/8/1b2P1P1/6BP/R5K1 w - - 1 31",
            mov: "Bh6",
            comment: r#"Bh6 prevents movement of the enemy h pawn and pins the knight. White has sacrificed the queen for a minor piece and a rook. Future plans involve playing h4 and h5 to win the pinned knight."#,
            reason: And(vec![
                prevents(Enables(vec!["h6"])),
                Pin { by: "g5", front: "g6", behind: "g8" },
                threatens(vec!["h4", "--", "h5", "--", "hxg6"], WinMaterial(vec![Knight])),
            ]),
        },
        // train-0643: Pressure
        Reasoning {
            fen: "2r2rk1/pp3ppp/2n1p3/b3P3/3P3q/2NQBP1P/PP3P1K/2R2R2 b - - 2 18",
            mov: "Rcd8",
            comment: r#"Interestingly, it was only whenconsidering this move, where the rook takes the queen's place on d8, that I spotthe Nxe5 threat. The move increases pressure on the backward pawn d4 andre-establishes the pin."#,
            reason: And(vec![
                Pressure(Squares(vec!["d4"])),
                Backward(Squares(vec!["d4"])),
                Pin { by: "d8", front: "d4", behind: "d3" },
                threatens(vec!["Nxe5"], WinMaterial(vec![Pawn])),
            ]),
        },
        // train-1010: Passer
        Reasoning {
            fen: "r2q1rk1/p3ppbp/1p4p1/3P4/4P3/8/P3QPPP/1RBR2K1 w - - 2 18",
            mov: "Ba3",
            comment: r#"Preparing White's two main ideas : the creation of a passed pawn with e4-e5 and d5-d6, and the control of the c-file with Rb1-c1-c6. Black cannot prevent both."#,
            reason: And(vec![
                threatens(vec!["e5", "--", "d6", "exd6", "exd6"], Passer("d6")),
                threatens(vec!["Rbc1", "--", "Rc6"], Control(Line("c2", "c8"))),
            ]),
            // Before Region:
            // threatens(vec!["Rbc1", "--", "Rc6"], Control(Squares(vec!["c2", "c3", "c4", "c5", "c6", "c7", "c8"]))),
        },
        // train-0528: Promote
        Reasoning {
            fen: "6rk/4R1pp/2p2n2/2p5/6b1/2N3P1/PP1p1P1P/R5K1 b - - 0 24",
            mov: "Nd5",
            comment: r#"the onlymove for Black. Now the pawn threatens to queen, since Black is in a position toremove the Nc3 and White's Re7 cannot get to the d-file to defend."#,
            reason: only(And(vec![
                threatens(vec!["Nxc3"], Promote("d2")),
                Block { line: ("d7", "d2"), by: "d5" },
            ])),
        },
        // train-1408: Zugzwang
        Reasoning {
            fen: "8/8/5p2/3nk2K/6P1/5R2/8/8 w - - 1 74",
            mov: "Kg6",
            comment: r#"1-0 Black resigns. Black now has no way to defend f6. Even if MVL tries to hold on for example with 74... Ke6, White can simply make a waiting move with 75. Rf2. Black is then forced to play 75... Ne7+, since Black's king loses the opposition and this is the only knight move that doesn't immediately lose the f6 pawn, but then after 76. Kg7 Nd5 White plays 77. Rf5! and Black is in zugzwang: no matter what move he makes, the f6 pawn will be lost. Which is why MVL resigned in this position."#,
            reason: And(vec![
                Pressure(Squares(vec!["f6"])),
                after(vec!["Ke6", "Rf2", "Ne7+", "Kg7", "Nd5", "Rf5"], Zugzwang),
            ]),
        },
        // train-0915: Draw, Quality
        Reasoning {
            fen: "8/6pk/7p/8/8/1R2KP2/7r/8 w - - 0 41",
            mov: "Rb7",
            comment: r#"despite Black's passed h-pawn, this should be adraw, since White's rook is ideally placed to harass Black's king and pressurethe pawns from the side."#,
            reason: And(vec![Draw, Quality("b7", Grade::Good), Pressure(Squares(vec!["g7"])), Pin { by: "b7", front: "g7", behind: "h7" }]),
        },
        // train-0021: Win
        Reasoning {
            fen: "4B2k/7p/p2bP1p1/1p1P4/1P2K1P1/P6P/8/8 w - - 1 40",
            mov: "Bc6",
            comment: r#"1-0 White wins. Now white can easily win the light squared pawns. This is the idea that allows white to win, there are many similar endgame setups with opposite colored bishops where it is simply a draw."#,
            reason: And(vec![Win, Pressure(Squares(vec!["b5"]))]),
        },
        // train-1104: Practical
        Reasoning {
            fen: "7r/3b2k1/2pp2p1/pp2p1Np/4P3/1P1PR1P1/P1P2P1P/6K1 w - - 2 31",
            mov: "Rf3",
            comment: r#"The idea being that I'm nowthreatening 32.Rf7+ Of course, he can do something, especially with 31...Be8 But, there would be a lot of pressure on the seventh rank which wouldbe almost overwhelming to black. He can, however, do something about it,but the entire plan was to put him under pressure. The idea being is thatwhen under pressure, you tend to make mistakes which is what gives theopponent a bigger advantage."#,
            reason: And(vec![
                threatens(vec!["Rf7+"], Attack { by: "f7", targets: vec!["g7", "d7"] }),
                allows(vec!["Be8"], Defend { by: "e8", target: "f7" }),
                Pressure(Squares(vec!["f7"])),
                Practical,
            ]),
        },
        // train-0260: Trade, BishopPair
        Reasoning {
            fen: "r1bqk1nr/pppp1ppp/2n5/4p3/1bP5/2N3P1/PP1PPPBP/R1BQK1NR b KQkq - 2 4",
            mov: "Bxc3",
            comment: r#"Black gives up the two bishops without any pressure. Inreturn, the benefit is removing a good knight from the board that helps controld5 and doubling White's pawns. I'm still happy as White here."#,
            reason: And(vec![
                after(vec!["bxc3"], loses(BishopPair)),
                Trade { give: vec!["b4"], get: vec!["c3"] },
                prevents(Defend { by: "c3", target: "d5" }),
                after(vec!["bxc3"], Doubled(Squares(vec!["c3", "c4"]))),
            ]),
        },
        // train-1100: Decoy, Enables
        Reasoning {
            fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            mov: "f4",
            comment: r#"This is the King's Gambit Position. White plays f4, a bold attack on black's e pawn, planning on diverting it from the center. It gambits the f pawn and weakens white's kingside in order to get a positional advantage and a lead in development."#,
            reason: And(vec![
                sacrifice(vec![Pawn], And(vec![
                    Attack { by: "f4", targets: vec!["e5"] },
                    Decoy { from: "e5", to: "f4" },
                    Enables(vec!["d4"]),
                    Tempo(1),
                ])),
                concedes(Weakened(Squares(vec!["e3", "g3"]))),
            ]),
        },
        // train-1024: Develop, Tempo
        Reasoning {
            fen: "rnbqk2r/ppp2ppp/3b4/2P5/3Q2n1/8/PP2PPPP/RNB1KBNR b KQkq - 0 6",
            mov: "Nc6",
            comment: r#"Black develops his N with tempo to c6, countering the threat on d6 with his on threat on the White Q.Black is ready to castle. White is 4 tempi behind before he can castle. White now loses a further tempo because he has to move his Q to safety. The best move is Qc4, from where you could develop your light square B to Be6, gaining another tempo on the Q."#,
            reason: And(vec![
                Develop(At("c6")),
                Attack { by: "c6", targets: vec!["d4"] },
                Tempo(5),
                after(vec!["Qc4", "Be6"], And(vec![Develop(At("e6")), Attack { by: "e6", targets: vec!["c4"] }, Tempo(6)])),
            ]),
        },
        // train-0629: Castle
        Reasoning {
            fen: "r3k1nr/ppp2ppp/2n5/3q4/1b1P4/2N2B2/PP3PPP/R1BQK2R b KQkq - 0 9",
            mov: "Qc4",
            comment: r#"Although black is not winning that pawn but they are still putting a lot of pressure and preventing white from castling meaning that king E1 is stuck in the center. Black is also threatening takes for bishop C3 and knight D4"#,
            reason: And(vec![
                Pressure(Squares(vec!["d4"])),
                prevents(Castle),
                threatens(vec!["Nxd4"], WinMaterial(vec![Pawn])),
            ]),
        },
        // train-1246: Quality
        Reasoning {
            fen: "1rbr2k1/2Rn1ppp/1P1q4/p2p4/P2B4/4P3/4B1PP/3Q1RK1 w - - 1 28",
            mov: "Bd3",
            comment: r#"I was pleased to find this move, which isquiet but effective. The bishop is centralized and now threatens action on thekingside against Black's weakly defended king. Black's pieces are too tied up onthe queenside to be able to defend against White's sudden threats."#,
            reason: And(vec![Quality("d3", Grade::Good), Pressure(Squares(vec!["h7"]))]),
        },
        // train-1313: Control
        Reasoning {
            fen: "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
            mov: "e4",
            comment: r#"White has good control of the center, so you must respond to this. White threatens to play e5, so how do you control the e5 square with one of your pawns?"#,
            reason: And(vec![
                Control(Squares(vec!["d5", "e5", "f5"])),
                threatens(vec!["e5"], Attack { by: "e5", targets: vec!["f6"] }),
            ]),
        },
        // train-0375: Open
        Reasoning {
            fen: "qr4k1/3n2pp/b1p1p3/3pPpn1/2PP4/N1B1P3/4B1PP/2Q1R1K1 w - - 5 22",
            mov: "Qc2",
            comment: r#"Tartakower plans on opening the c-file with cxd5. He advances the Queen in without preparation for playing Rc1, doubling his forces on the soon-to-be-opened without file."#,
            reason: And(vec![
                Enables(vec!["Rc1"]),
                threatens(vec!["cxd5"], Open(("c1", "c8"))),
            ]),
        },
        // train-0409: Trade
        Reasoning {
            fen: "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
            mov: "Bg5",
            comment: r#"In this position, white prepares to trade knight for bishop inflicting doubled pawns upon Black in the process. Black has tons of ways to counter this. 2... Ne4 is the most common reply. Black does break one of the opening rules (do not move a piece twice) but it attacks white's bishop. 2... e6 also avoids doubled pawns since the queen can recapture if White plays Bxf6. d5 and c5 can also be played to attack the center."#,
            reason: And(vec![
                threatens(vec!["Bxf6", "exf6"], And(vec![Trade { give: vec!["c1"], get: vec!["f6"] }, Doubled(Squares(vec!["f6", "f7"]))])),
                allows(vec!["Ne4"], And(vec![Attack { by: "e4", targets: vec!["g5"] }, loses(Tempo(1))])),
                allows(vec!["e6"], Defend { by: "d8", target: "f6" }),
                allows(vec!["c5"], Attack { by: "c5", targets: vec!["d4"] }),
                allows(vec!["d5"], Control(Squares(vec!["c4", "e4"]))),
            ]),
        },
        // train-1601: Not
        Reasoning {
            fen: "2r3k1/p3npp1/1qp1p2p/3R4/3P4/8/PPQ1NPPP/3R2K1 b - - 0 24",
            mov: "cxd5",
            comment: r#"Black has now the open file and his left side Pawn position is very solid, while White has a weak d-Pawn. The apparently weak Black a Pawn is not actually weak because White has no way to attack it."#,
            reason: And(vec![
                Open(("c8", "c1")),
                Isolated(Squares(vec!["d4"])),
                but(allows(vec![], Isolated(Squares(vec!["a7"]))), not(allows(vec![], Hanging(Squares(vec!["a7"]))))),
            ]),
        },
        // train-1234: Majority, Better
        Reasoning {
            fen: "r4rk1/3n3p/3p1qp1/2pP1p2/p1P1P3/8/PPB1Q1PP/3R1RK1 b - - 0 24",
            mov: "f4",
            comment: r#"Black's pawn sacrifice has brought him a whole string of positional advantages 1) His 3-2 majority on the Kingside is now extremely active 2) He controls the vital e5 square, and cna plant a Knight there as a blockader 3) The hapless White Bishop is no match at all for the Black Knight"#,
            reason: And(vec![
                Majority(Squares(vec!["f4", "g6", "h7"])),
                Control(Squares(vec!["e5"])),
                threatens(vec!["Ne5"], prevents(Enables(vec!["e5"]))),
                Better { pieces: vec!["d7"], than: vec!["c2"] },
            ]),
        },
        // train-0775: Simplify
        Reasoning {
            fen: "1r3bk1/1prR1p1p/6p1/1pP5/8/P4BP1/3R1PKP/8 b - - 0 37",
            mov: "Rxd7",
            comment: r#"Due to the weakness of the doubled b-pawns, it's to Yates' advantage totry to get as many pieces off of the board as possible (so that there will befewer White pieces to attack the weak pawns)."#,
            reason: And(vec![
                Trade { give: vec!["c7"], get: vec!["d7"] },
                Simplify,
                prevents(Pressure(Squares(vec!["b5", "b7"]))),
            ]),
        },
        // train-0574: Faster
        Reasoning {
            fen: "2krr3/ppqn1pp1/2p1p2b/8/2NPQP2/2P5/PP2B2P/R3KR2 b Q - 0 22",
            mov: "f6",
            comment: r#"here I focused on the plan of ripping open the e-file with the ...e5 pawnadvance, but this is far too slow. White in response should simply castleimmediately, which takes the sting out of Black's idea."#,
            reason: And(vec![
                Enables(vec!["e5"]),
                not(faster(Open(("e8", "e1")), Castle)),
                allows(vec!["O-O-O"], Castle),
            ]),
        },
        // train-0732: Outnumber
        Reasoning {
            fen: "3r3k/6pp/2q2p2/ppp5/7P/P4N2/BP4P1/4R1BK b - - 0 33",
            mov: "h5",
            comment: r#"White makes room for his king, and Black does the same. This was the kind of chance that White was waiting for, because he now has chances of a king's side attack. Fron his point fo view the disparity of four pieces against two is at an optimum: had he not excanged some pieces the ratio would have been favourable, whilst if he were now to exchange rooks he would hardly have a sufficient attacking force."#,
            reason: And(vec![
                Enables(vec!["Kh7"]),
                allows(vec![], not(Simplify)),
            ]),
        },
        // dev-0051: Options
        Reasoning {
            fen: "rnbqkbnr/pp1p1ppp/4p3/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4",
            mov: "Nc6",
            comment: r#"Namedafter Mark Taimanov. Black develops the knight to a natural square andkeeps his options open regarding the placement of his other pieces. Oneof the ideas of this system is to develop the bishop to b4 or c5. Whitecan prevent this by 5.Nb5 d6. Taimanov's idea was to play 5...a6 (preventingNb5) followed by ...Nge7 and ...Nxd4. Then when White recaptures with thequeen, Black can attack it with ...Nc6, gaining time. A more popular setupinvolves ...Qc7, ...a6 and ...Nf6: this is often called the Paulsen Variation."#,
            reason: And(vec![
                Develop(At("c6")),
                Options(vec!["Bb4", "Bc5"]),
                allows(vec!["Nb5", "d6"], prevents(Enables(vec!["Bc5"]))),
                threatens(vec!["a6"], after(vec!["Nb5"], Hanging(Squares(vec!["b5"])))),
                threatens(vec!["a6", "--", "Nge7", "--", "Nxd4", "Qxd4", "Nc6"], And(vec![
                    Attack { by: "c6", targets: vec!["d4"] },
                    Tempo(1),
                ])),
            ]),
        },
        // train-0837: Suppose
        Reasoning {
            fen: "8/2p2ppk/p2p3p/6r1/2nPp3/B1PqP3/P4PPP/2Q1R2K w - - 17 32",
            mov: "Bb4",
            comment: r#"The threat was 32 ... Ra5. Bb4 wouldbe more effective if the Black rook were still on g6, since then the bishopcould only be driven off by 32 ... a5, when 33 a4!? axb4 34 cxb4 wouldgive White at least the hope of counter play in exchange for his piece. Now, however, 32 ... c5 33 dxc5 dxc5 34 Ba3 drives the bishop back andopens the d file for Black's queen and rook while allowing Black to maintainhis queenside bind. White prevents the rook from reaching a5 or b5, butthat is more than offset by the increased mobility of the Black queen andBlack's control of the d file."#,
            reason: And(vec![
                after(vec!["Ra5"], Hanging(Squares(vec!["a5"]))),
                suppose(vec![("g5", None), ("g6", Some('r'))],
                    after(vec!["a5", "a4", "axb4", "cxb4"], sacrifice(vec![Bishop], Tempo(1)))),
                allows(vec!["c5", "dxc5", "dxc5", "Ba3"], And(vec![
                    Open(("d8", "d1")),
                    Quality("d3", Grade::Good),
                    Quality("c4", Grade::Good),
                ])),
            ]),
        },
        // train-0565: Coordinate
        Reasoning {
            fen: "r1bqk2r/pp3pp1/2pbpn1p/8/3P3Q/3B1N2/PPP2PPP/R1B1K2R b KQkq - 2 11",
            mov: "Ke7",
            comment: r#"This quite paradoxical king move to the center of the board was found by me back in 1988, while I was preparing for the aforementioned game against Kasparov. I was afraid of the queen shift to the kingside. The exchange of queens seemed to me to be a dull idea. It took me a long time before I found the correct decision. And so this important novelty remained a secret for five years! The idea of the king move is that black unexpectedly harmonizes the placement of his pieces, which were a bit out of sync only a move ago. But now the threat is g7-g5-g4, winning a piece. If white wants to maintain the opening advantage, he must act decisively."#,
            reason: And(vec![
                prevents(Pressure(Squares(vec!["g7"]))),
                Coordinate(vec!["e7", "d8", "h8"]),
                threatens(vec!["g5", "--", "g4"], WinMaterial(vec![Knight])),
            ]),
        },
        // train-1687: Space, Because, Outweighs
        Reasoning {
            fen: "r3k2r/pppqbppp/3p1n2/8/3pP3/2N5/PPP2PPP/R1BQR1K1 w kq - 0 10",
            mov: "Qxd4",
            comment: r#"White now has a clear advantage in space because the pawn on e4 controls more space than the pawn on d6 and because of the powerfully placed queen."#,
            reason: because(Space, And(vec![
                outweighs(Control(Squares(vec!["d5", "f5"])), Control(Squares(vec!["c5", "e5"]))),
                Quality("d4", Grade::Good),
            ])),
        },
        // train-0927: Initiative
        Reasoning {
            fen: "r3r1k1/pppqnpp1/1b3nbp/1P1p4/3P2P1/2PBNN1P/P4P2/R1BQR1K1 w - - 1 18",
            mov: "Ne5",
            comment: r#"White has advanced pawns and the structure is a bit weak. Therefore the initiative must be maintained. Notice that the advance of the h-pawn has weakened g6."#,
            reason: And(vec![
                because(Initiative, allows(vec![], Weakened(Kingside))),
                Weakened(Squares(vec!["g6"])),
            ]),
        },
        // train-0946: Permanent, Imbalance
        Reasoning {
            fen: "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
            mov: "f5",
            comment: r#"An aggressive opening for Black that leads to sharp, unbalanced positions. Black is playing to win not unlike the Sicilian but instead of c5 you are committing the riskier f5 pawn which weakens f7.Black gets to control e4 but has permanently weakened the King's position via f7."#,
            reason: And(vec![
                Imbalance,
                Control(Squares(vec!["e4"])),
                allows(vec![], permanent(Hole(Squares(vec!["f7"])))),
            ]),
        },
        // train-1178: Outweighs
        Reasoning {
            fen: "r2q1rk1/p3bppp/1np1p3/3n2B1/3P4/1BN5/PP2QPPP/3R1RK1 w - - 0 15",
            mov: "Bc1",
            comment: r#"Black was wrong to allow himself to be saddled with the isolated c-pawn which will prove weaker than the White d-pawn. White, with his greater freedom of movement, will now have chances of a Kingside attack. He already threatens Rd3 and Black finds it essential to bring reinforcements across to the King'swing"#,
            reason: And(vec![
                outweighs(Isolated(Squares(vec!["c6"])), Isolated(Squares(vec!["d4"]))),
                Space,
                threatens(vec!["Rd3", "--", "Rh3"], Pressure(Squares(vec!["h7"]))),
            ]),
        },
        // train-0725: Trade and Better between groups
        Reasoning {
            fen: "r1b1k2r/pp3ppp/2p5/2b1n3/P3P3/2N5/1P2BPPP/R1B1KR2 b Qkq - 1 13",
            mov: "Ng4",
            comment: r#"Black intends to trade on f2, reasoning that in the material swap (gaining R+Pfor N+B) will be good, as Black's second rook can operate on the central files.This is both bad counting in general terms (normally two pieces are worth morethan rook and pawn) and a misjudgment of this particular position."#,
            reason: And(vec![
                threatens(vec!["Nxf2", "Rxf2", "Bxf2+", "Kxf2"], And(vec![
                    Trade { give: vec!["e5", "c5"], get: vec!["f1", "f2"] },
                    Quality("h8", Grade::Good),
                ])),
                Better { pieces: vec!["g4", "c5"], than: vec!["f1", "f2"] },
            ]),
        },
        // train-1693: NoMove
        Reasoning {
            fen: "2k4r/pp6/3r1pp1/4pn1p/3PQ3/P1P2R1P/1q4PK/4R3 b - - 0 35",
            mov: "Qxa3",
            comment: r#"Uh oh, white is threatening my pawn at e5 (36. Pxe5 Pxe5?37. Qxe5 and white gains a pawn, forks both rooks, and forces black toget back on defense). But worse, he's threatening 36. Rb1!, skewering theblack queen against the b7 pawn. After 36. ... Qxa3, 37. Qxb7 Kmoves,38. Qb8 Kmoves, 39. Qxh8 and now white is right back in the game, materialis even, and black's kingside is going to crumble as the white queen attacksit from the rear. I search around for the move that prevents both eventualities,but there is none. So I take the a3 pawn, extending my material lead toa knight and 2 pawns, knowing I will lose a pawn in the center and haveto run back on defense."#,
            reason: And(vec![
                no_move(prevents(And(vec![
                    threatens(vec!["dxe5", "fxe5", "Qxe5"], WinMaterial(vec![Pawn])),
                    threatens(vec!["Rb1"], Pin { by: "b1", front: "b2", behind: "b7" }),
                ]))),
                WinMaterial(vec![Pawn]),
                allows(vec!["dxe5", "fxe5", "Qxe5"], And(vec![
                    WinMaterial(vec![Pawn]),
                    Attack { by: "e5", targets: vec!["d6", "h8"] },
                    Initiative,
                ])),
            ]),
        },
        // train-0035: SameColour
        Reasoning {
            fen: "rn1qkbnr/pp2pppp/2p3b1/8/3P4/6N1/PPP2PPP/R1BQKBNR w KQkq - 3 6",
            mov: "h4",
            comment: r#"In this position, as in the Advance Variation, White finds it most profitable to provoke a weakening in Black's position by threatening Bg6. It also allows White, if she chooses, to virtually force a bishop trade by Ng1-f3 and Bf1-d3. Black's bishop is very good and Black's light-square pawns make White's bishop mediocre, so a trade is considered best."#,
            reason: And(vec![
                threatens(vec!["h5"], Attack { by: "h5", targets: vec!["g6"] }),
                Enables(vec!["Nf3", "--", "Bd3"]),
                because(
                    threatens(vec!["Nf3", "--", "Bd3", "--", "Bxg6"], Trade { give: vec!["f1"], get: vec!["g6"] }),
                    And(vec![
                        degree(Clear, Quality("g6", Grade::Good)),
                        because(Quality("f1", Grade::Bad), SameColour(vec!["f1", "b7", "c6", "f7"])),
                    ]),
                ),
            ]),
        },
        // train-0415: Material
        Reasoning {
            fen: "1k4r1/1p3pb1/p1p1b2p/2pqP3/5Q2/1P1P1N1P/P1P1R1P1/5RK1 w - - 0 26",
            mov: "Qc4",
            comment: r#"There I was, expectingthreat after threat on the kingside, justifying 24. ... f4?! and blackplays 25. ... a6 instead. Now I don't even know what to do. I had expectedto have to defend, but it looks like I can attack instead. How about tradingqueens? Black has no move that doesn't lose the pawn on c4, and he's alreadydown a pawn and the exchange, so both options are no good. Lose more material,or exchange the big guns. Sounds win-win to me."#,
            reason: And(vec![
                Attack { by: "c4", targets: vec!["d5", "c5"] },
                after(vec!["Qxc4", "dxc4"], because(
                    And(vec![Trade { give: vec!["f4"], get: vec!["d5"] }, Simplify]),
                    Material(3),
                )),
                // Every Black move trades queens, leaves the trade on offer, or loses material.
                allows(vec![], no_move(not(Either(vec![
                    Trade { give: vec!["d5"], get: vec!["f4"] },
                    allows(vec![], Attack { by: "c4", targets: vec!["d5"] }),
                    allows(vec![], Hanging(Half { opponent: true })),
                    allows(vec![], Hanging(Half { opponent: false })),
                ])))),
            ]),
        },
        // train-1699: Closed
        Reasoning {
            fen: "r4rk1/pb2qppp/1p1p1b2/2pPp3/2PBP3/1P3NP1/P1Q2PBP/R2R2K1 b - - 0 16",
            mov: "cxd4",
            comment: r#"Here, Fritz creator Frans Morsch was worried. With this sort of closed position, the program cannot capitalize on its phenomenal tactical abilities and Kramnik has the opportunity to implement his superior strategic knowledge. The black pawn on d4 is strong. Not only because it's passed (with all the locked pawns around it, it's hard to imagine it advancing to queen), but also because it's solidly placed in the heart of White's camp, and hinders the White pieces."#,
            reason: And(vec![
                allows(vec![], because(Practical, Closed)),
                because(Quality("d4", Grade::Good), And(vec![
                    Passer("d4"),
                    Control(Squares(vec!["c3", "e3"])),
                ])),
                because(not(Promote("d4")), Closed),
            ]),
        },
        // train-0452: Gains
        Reasoning {
            fen: "3r2k1/4b1pp/p7/1p2q3/2nrP3/5P1N/PPR1Q1P1/2R2NK1 b - - 2 27",
            mov: "Bc5",
            comment: r#"Black threatens to win the White Queen by way of a discoveredcheck: 28...Rd2+ (the Rook move uncovers the Bishop check, forcing the WhiteKing to move) 29.Kh1 Rxe2 30.Rxe2 (winning the Queen for a Rook)."#,
            reason: threatens(vec!["Rd2+"], And(vec![
                gains(Attack { by: "c5", targets: vec!["g1"] }),
                after(vec!["Kh1", "Rxe2", "Rxe2"], WinMaterial(vec![Queen])),
            ])),
            // Before Gains: the check, but not that the rook move uncovers it.
            // reason: threatens(vec!["Rd2+"], And(vec![
            //     Attack { by: "c5", targets: vec!["g1"] },
            //     after(vec!["Kh1", "Rxe2", "Rxe2"], WinMaterial(vec![Queen])),
            // ])),
        },
        // train-0121: Concedes
        Reasoning {
            fen: "r4r1k/pppqn1b1/3pn2p/1P1Nppp1/P1P5/BQ1P1NPb/4PP1P/1R2R1KB w - - 0 17",
            mov: "e3",
            comment: r#"this weakens d3, which is important in a number of variations, but alsostrengthens f4, prevents the Black knight from entering d4, and offers thechance of opening the e-file for White if the e-pawn is exchanged."#,
            reason: And(vec![
                concedes(Hole(Squares(vec!["d3"]))),
                Control(Squares(vec!["f4"])),
                after(vec!["Nd4"], Hanging(Squares(vec!["d4"]))),
            ]),
            // Before Concedes: d3 is weak now, not that e3 made it so.
            // reason: And(vec![
            //     allows(vec![], Weak(Squares(vec!["d3"]))),
            //     Control(Squares(vec!["f4"])),
            //     prevents(Enables(vec!["Nd4"])),
            // ]),
        },
        // train-1210: XRay
        Reasoning {
            fen: "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            mov: "O-O",
            comment: r#"Rio Gambit. White's strongest, most common response. This defends white's king, bringing the rook to a more central file and preparing to bring it to the e-file to defend the e4 pawn and line up on black's king. This temporarily gambits the e4 pawn, allowing black to capture with Nxe4, but white can capture the e5 pawn in response with Re1, forcing the e4 knight to move while x-ray attacking the e5 pawn. This allows Nxe5, even if black threatens to win the white-squared bishop with Nd6, as after Nxe5, white threatens Nxc6, winning black's queen if black captures the bishop with Nxb5."#,
            reason: sacrifice(vec![Pawn], after(vec!["Nxe4", "Re1"], And(vec![
                Attack { by: "e1", targets: vec!["e4"] },
                XRay { by: "e1", through: "e4", target: "e5" },
            ]))),
            // Before XRay: right geometry, but claims the knight cannot move, while the point is that it must.
            // reason: sacrifice(vec![Pawn], after(vec!["Nxe4", "Re1"], Pin { by: "e1", front: "e4", behind: "e5" })),
        },
        // train-0535: Region (Queenside, Kingside)
        Reasoning {
            fen: "6k1/pp5p/6p1/2Pp1p2/8/1P2b2P/P4PP1/6K1 w - - 0 28",
            mov: "fxe3",
            comment: r#"Timeto evaluate this endgame. It's quite complex, actually. Both sides haveisolated pawns. The kings are preparing to advance to the center. Blackhas a pawn majority on the kingside, while white has one on the queenside. The queenside majority gives white dangerous chances, though. The majorityhas a lust to quickly expand with b4 and a4, with moves like b5 and thenc6 coming in, probably creating a strong passed pawn. This may not happen,but it is a threat. It at least keeps the black king somewhat tied downto this area to prevent a passed pawn from becoming a deadly threat."#,
            reason: And(vec![
                outweighs(Majority(Queenside), Majority(Kingside)),
                threatens(vec!["b4", "--", "b5", "--", "c6", "bxc6", "bxc6"], Passer("c6")),
            ]),
            // Before Region: the wing is guessed from the files of the listed pawns.
            // reason: And(vec![
            //     outweighs(Majority(Squares(vec!["a2", "b3", "c5"])), Majority(Squares(vec!["f5", "g6", "h7"]))),
            //     threatens(vec!["b4", "--", "b5", "--", "c6", "bxc6", "bxc6"], Passer("c6")),
            // ]),
        },
        // train-0924: Region (Centre)
        Reasoning {
            fen: "3r2k1/2r2pq1/p2pb1p1/1p4Np/2nPPQ2/1B5P/PP3RP1/3R2K1 w - - 3 34",
            mov: "Kh1",
            comment: r#"White's initiative has now brough him a decisive positional advantage. He controls the centre and the half-open f-file, as well as enjoying the greater freedom of movement. What follows now is not so much a matter of initiative as direct attack, for which Kh1 was a fine preparation (preventing Qd4+)"#,
            reason: And(vec![Control(Centre), Open(("f1", "f8"))]),
            // Before Region:
            // reason: And(vec![Control(Squares(vec!["d4", "e4", "d5", "e5"])), Open(("f1", "f8"))]),
        },
        // dev-0005: Region (Kingside) under Weak and Pressure
        Reasoning {
            fen: "r1bqk1nr/2p1bpp1/ppn1p2p/3pP3/3P4/P1N1BP2/1PP3PP/R2QKBNR w KQkq - 0 8",
            mov: "Bd3",
            comment: r#"Taking control of another very good diagonal, and especially preventingBlack's Rook from leaving its original post. At this point, White's developmentis almost complete, with a good control of the center squares. Black isa bit cramped up, most of his pieces still being in their home squares.His DSB on e7 is very nice, but his kingside is very weak at the moment,and will be the target of my attack."#,
            reason: And(vec![not(allows(vec![], Develop(On(Kingside)))), Pressure(Kingside)]),
            // Read as structure, which fails: no hole or weak pawn on Black's kingside.
            // reason: And(vec![Weak(Kingside), Pressure(Kingside)]),
            // Before Region:
            // reason: And(vec![Weak(Squares(vec!["f7", "g7", "h6"])), Pressure(Squares(vec!["h7"]))]),
        },
        // dev-0236: But
        Reasoning {
            fen: "2kr3r/ppp3pp/2n2p2/2bqp3/6bB/3P1N2/PPP1BPPP/R2Q1RK1 w - - 2 11",
            mov: "Qd2",
            comment: r#"Although Qd2 develops the queen, it does not prevent the threat of g5. Playing h3 would have removed the threat of black's light-squared bishop on the knight and remove the threat of g5. Mistake. h3 was best."#,
            reason: And(vec![
                because(Rated(Mistake), but(
                    Develop(Every(Queen)),
                    not(prevents(threatens(vec!["g5"], Attack { by: "g5", targets: vec!["h4"] }))),
                )),
                instead("h3", because(Rated(Best), And(vec![
                    Attack { by: "h3", targets: vec!["g4"] },
                    prevents(threatens(vec!["g5"], Attack { by: "g5", targets: vec!["h4"] })),
                ]))),
            ]),
            // Before But: both halves, not that the second undercuts the first.
            // reason: And(vec![
            //     because(Rated(Mistake), And(vec![
            //         Develop(Every(Queen)),
            //         not(prevents(threatens(vec!["g5"], Attack { by: "g5", targets: vec!["h4"] }))),
            //     ])),
            //     instead("h3", ...),
            // ]),
        },
        // dev-0273: More, Degree
        Reasoning {
            fen: "rnbqk2r/pp2b1pp/2n2p2/2p1p3/2Pp4/1P1PPN2/PB1NBPPP/R2Q1RK1 b kq - 6 9",
            mov: "O-O",
            comment: r#"Notice that even though White has developed more pieces,Black’s pieces have much more mobility: White controls only four squares in Black’s territory, while Black controls seven in White’s territory. (Plus, Black controls the d4 square four times.) That is what a space advantage can do for your pieces!"#,
            reason: but(allows(vec![], more(Develop(All))), degree(Clear, Space)),
            // Before More and Degree: "developed more" as a tempo count, "much more" dropped.
            // reason: but(allows(vec![], Tempo(1)), Space),
        },
        // dev-0124: Balances
        Reasoning {
            fen: "r1bq1rk1/pp2pp1p/6p1/2pPb3/4P3/2P5/P3BPPP/1RBQK2R w K - 0 12",
            mov: "c4",
            comment: r#"Consolidates the centre but invites black to put pressure onc3 via Qa4+. I plan to give Harpov control of the queen side in exchangefor greater development and freedom on the king side. On the other handblack has managed to create significant weakness on the dark squares. Hisdeveloped bishop is far superior. So white's big centre is balanced byblack's big bishop and dark square control."#,
            reason: balances(Majority(Centre), And(vec![
                degree(Clear, Better { pieces: vec!["e5"], than: vec!["c1"] }),
                Control(Squares(vec!["d4"])),
            ])),
            // Before Balances: both sides' assets, but not that they cancel out.
            // reason: And(vec![Majority(Centre), allows(vec![], And(vec![
            //     Better { pieces: vec!["e5"], than: vec!["c1"] },
            //     Control(Squares(vec!["d4"])),
            // ]))]),
        },
        // train-0330: Check
        Reasoning {
            fen: "3r4/4kpp1/pNR2np1/8/P5P1/7P/5P2/6K1 w - - 5 37",
            mov: "Rc7+",
            comment: r#"Yates would have advanced his Rook to the seventh rank anyway, but doing it without with check is even better! Now he keeps the initiative (since the King is without forced to avoid check)."#,
            reason: And(vec![Quality("c7", Grade::Good), because(Initiative, Check)]),
            // Before Check: the check as an attack on the king's square.
            // reason: And(vec![Quality("c7", Grade::Good), because(Initiative, Attack { by: "c7", targets: vec!["e7"] })]),
        },
    ]
}

/// Plays every line of every example and checks the board facts that need no engine.
#[test]
fn examples_are_legal() {
    use shakmaty::{CastlingMode, Chess, Color, Position, Square, attacks, fen::Fen, san::San};

    fn play(mut p: Chess, owner: Option<Color>, line: &[Move_]) -> Chess {
        if owner.is_some_and(|c| c != p.turn()) {
            p = p.swap_turn().unwrap();
        }
        for m in line {
            p = if *m == "--" {
                p.swap_turn().unwrap()
            } else {
                let mv = San::from_ascii(m.as_bytes()).unwrap().to_move(&p);
                p.play(mv.unwrap_or_else(|_| panic!("illegal {m}"))).unwrap()
            };
        }
        p
    }
    fn sq(s: Sq) -> Square { s.parse().unwrap() }
    fn hits(p: &Chess, by: Sq, t: Sq) -> bool { p.board().attacks_from(sq(by)).contains(sq(t)) }
    fn check(r: &Reason, p: &Chess, owner: Color, root: &Chess) {
        let occ = |s: Sq| assert!(p.board().piece_at(sq(s)).is_some(), "empty {s}");
        match r {
            And(rs) => rs.iter().for_each(|r| check(r, p, owner, root)),
            Threatens(l, r) => check(r, &play(p.clone(), Some(owner), l), owner, root),
            After(l, r) => check(r, &play(p.clone(), None, l), owner, root),
            Allows(l, r) => check(r, &play(p.clone(), None, l), !owner, root),
            Only(r) | Sacrifice(_, r) | Permanent(r) | Gains(r) => check(r, p, owner, root),
            Concedes(r) => check(r, p, !owner, root),
            Because(a, b) | But(a, b) => { check(a, p, owner, root); check(b, p, owner, root) }
            Outweighs(a, b) | Balances(a, b) => { check(a, p, owner, root); check(b, p, !owner, root) }
            Degree(_, r) => check(r, p, owner, root),
            Check => assert!(p.is_check()),
            Instead(m, r) => check(r, &play(root.clone(), None, &[m]), owner, root),
            Mate => assert!(p.is_checkmate()),
            Enables(l) => { play(p.clone(), Some(owner), l); }
            Attack { by, targets } => targets.iter().for_each(|t| { occ(t); assert!(hits(p, by, t), "{by} x {t}") }),
            Defend { by, target } => assert!(hits(p, by, target), "{by} defends {target}"),
            Bound { piece, duties } => duties.iter().for_each(|d| assert!(hits(p, piece, d))),
            Pin { by, front, behind } => {
                let between = attacks::between(sq(by), sq(behind)) & p.board().occupied();
                assert!(attacks::aligned(sq(by), sq(front), sq(behind)) && between.count() == 1 && between.contains(sq(front)));
            }
            XRay { by, through, target } => {
                let between = attacks::between(sq(by), sq(target)) & p.board().occupied();
                assert!(attacks::aligned(sq(by), sq(through), sq(target)) && between.count() == 1 && between.contains(sq(through)));
                assert_eq!(p.board().color_at(sq(through)), Some(!owner), "{through} is not the opponent's");
            }
            Block { line: (a, b), by } => { occ(by); assert!(attacks::between(sq(a), sq(b)).contains(sq(by))) }
            Develop(At(s)) | Quality(s, _) => occ(s),
            Majority(Squares(ps)) => ps.iter().for_each(|s| assert_eq!(p.board().role_at(sq(s)), Some(Pawn))),
            Options(ms) => ms.iter().for_each(|m| { play(p.clone(), Some(owner), &[m]); }),
            Trade { give, get } => give.iter().chain(get).for_each(|s| assert!(root.board().piece_at(sq(s)).is_some(), "empty {s}")),
            Better { pieces, than } => pieces.iter().chain(than).for_each(|s| occ(s)),
            Coordinate(ps) => ps.iter().for_each(|s| occ(s)),
            Material(n) => {
                let v = |c: Color| { let b = p.board(); (b.pawns() & b.by_color(c)).count() + 3 * ((b.knights() | b.bishops()) & b.by_color(c)).count() + 5 * (b.rooks() & b.by_color(c)).count() + 9 * (b.queens() & b.by_color(c)).count() };
                assert_eq!(v(owner) as i8 - v(!owner) as i8, *n, "material");
            }
            SameColour(ss) => assert!(ss.iter().all(|s| sq(s).is_light() == sq(ss[0]).is_light()), "colours {ss:?}"),
            Passer(s) | Promote(s) => assert_eq!(p.board().role_at(sq(s)), Some(Pawn)),
            _ => {}
        }
    }
    for ex in examples() {
        let root: Chess = Fen::from_ascii(ex.fen.as_bytes()).unwrap().into_position(CastlingMode::Standard).unwrap();
        let p = play(root.clone(), None, &[ex.mov]);
        check(&ex.reason, &p, root.turn(), &root);
    }
}
