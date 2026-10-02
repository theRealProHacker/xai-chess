//! One FEN per line on stdin, optionally followed by a tab and the engine's principal variation
//! (UCI, space-separated). One JSON line out per FEN: the features that are on.
//!
//!   features [--all]     --all also prints the features that are off

use std::io::{BufRead, Write};

fn main() {
    let all = std::env::args().any(|a| a == "--all");
    let out = std::io::stdout();
    let mut out = out.lock();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        let (fen, top) = match line.split_once('\t') {
            Some((f, t)) => (f, Some(t.split_whitespace().map(String::from).collect::<Vec<_>>())),
            None => (line.as_str(), None),
        };
        let v = match features::extract_fen(fen, top.as_deref()) {
            Ok(fs) => serde_json::json!({ "fen": fen, "features": fs.into_iter().filter(|f| all || f.on).collect::<Vec<_>>() }),
            Err(e) => serde_json::json!({ "fen": fen, "error": e }),
        };
        writeln!(out, "{v}").unwrap();
    }
}
