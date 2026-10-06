//! One FEN per line on stdin, optionally followed by tab-separated engine lines (UCI,
//! space-separated): the best line, then the line after a null move. One JSON line out per FEN:
//! for each square, the bits that are on.
//!
//!   square [--engine PATH] [--nodes N] [--indices] [--schema]
//!
//!   --engine   run Stockfish for both lines (1 thread, fixed nodes, so runs repeat)
//!   --nodes    nodes per search, default 1000000
//!   --indices  bit indices instead of names
//!   --schema   print the groups (name, offset, size), every bit's label and the vector length, then exit

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use shakmaty::fen::Fen;
use shakmaty::{EnPassantMode, Square};
use square::schema;

struct Engine { _child: Child, inp: ChildStdin, out: BufReader<ChildStdout>, nodes: u64 }

impl Engine {
    fn new(path: &str, nodes: u64) -> Engine {
        let mut child = Command::new(path).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        let inp = child.stdin.take().unwrap();
        let out = BufReader::new(child.stdout.take().unwrap());
        let mut e = Engine { _child: child, inp, out, nodes };
        e.send("uci\nsetoption name Threads value 1\nsetoption name Hash value 64\nisready");
        e.until("readyok");
        e
    }
    fn send(&mut self, s: &str) { writeln!(self.inp, "{s}").unwrap(); self.inp.flush().unwrap(); }
    fn until(&mut self, prefix: &str) -> Vec<String> {
        let mut lines = vec![];
        loop {
            let mut l = String::new();
            if self.out.read_line(&mut l).unwrap() == 0 { panic!("engine closed"); }
            let done = l.starts_with(prefix);
            lines.push(l);
            if done { return lines; }
        }
    }
    /// The principal variation of a fixed-node search.
    fn pv(&mut self, fen: &str) -> Vec<String> {
        self.send("ucinewgame");
        self.send(&format!("position fen {fen}\ngo nodes {}", self.nodes));
        let lines = self.until("bestmove");
        lines.iter().rev().find_map(|l| l.split_once(" pv ").map(|(_, pv)| pv.split_whitespace().map(String::from).collect()))
            .unwrap_or_default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    if flag("--schema") {
        let groups: Vec<_> = schema::groups().into_iter().map(|(n, o, s)| serde_json::json!({ "group": n, "offset": o, "size": s })).collect();
        println!("{}", serde_json::json!({ "bits_per_square": schema::len(), "groups": groups,
            "labels": (0..schema::len()).map(schema::label).collect::<Vec<_>>() }));
        return;
    }
    let nodes = value("--nodes").map(|n| n.parse().unwrap()).unwrap_or(1_000_000);
    let mut engine = value("--engine").map(|p| Engine::new(&p, nodes));
    let indices = flag("--indices");

    let out = std::io::stdout();
    let mut out = out.lock();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        let mut parts = line.split('\t');
        let fen = parts.next().unwrap().trim();
        let split = |s: Option<&str>| s.map(|s| s.split_whitespace().map(String::from).collect::<Vec<_>>());
        let (mut best, mut passed) = (split(parts.next()), split(parts.next()));
        let pos = match square::chess(fen) {
            Ok(p) => p,
            Err(e) => { writeln!(out, "{}", serde_json::json!({ "fen": fen, "error": e })).unwrap(); continue; }
        };
        if let Some(e) = engine.as_mut() {
            best = Some(e.pv(fen));
            passed = square::null_position(&pos).map(|n| e.pv(&Fen::from_position(&n, EnPassantMode::Legal).to_string()));
        }
        let bits = square::extract(&pos, &square::Lines { best: best.as_deref(), null: passed.as_deref() });
        let mut squares = serde_json::Map::new();
        for s in Square::ALL {
            let v = &bits[s as usize];
            if v.is_empty() { continue; }
            squares.insert(s.to_string(), if indices { serde_json::json!(v) } else { serde_json::json!(v.iter().map(|&i| schema::label(i)).collect::<Vec<_>>()) });
        }
        writeln!(out, "{}", serde_json::json!({ "fen": fen, "best": best, "null": passed, "squares": squares })).unwrap();
    }
}
