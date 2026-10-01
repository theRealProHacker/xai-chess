//! The browser API: JSON in and out. The engine runs in JavaScript; see `eval::Engine`.

use std::cell::RefCell;
use std::collections::HashMap;

use serde_json::{Value, json};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};
use wasm_bindgen::prelude::*;

use crate::dsl_claude::{Move_, QUESTIONABLE, examples as all_examples};
use crate::eval::{Engine, Ev, V, parse_search, san};
use crate::render::render_reason;
use crate::text::{parse, show};

/// Search size, as in the native `NODES` default.
const NODES: u64 = 300_000;

thread_local! {
    static CACHE: RefCell<HashMap<String, (i32, Vec<String>)>> = RefCell::new(HashMap::new());
    static PANIC: RefCell<String> = const { RefCell::new(String::new()) };
}

fn hook() {
    std::panic::set_hook(Box::new(|i| PANIC.with(|p| *p.borrow_mut() = i.to_string())));
}

/// The message of the panic that trapped the last call.
#[wasm_bindgen]
pub fn last_panic() -> String { PANIC.with(|p| p.borrow().clone()) }

#[wasm_bindgen]
pub fn examples() -> String {
    hook();
    let xs: Vec<Value> = all_examples()
        .iter()
        .map(|e| json!({ "fen": e.fen, "mov": e.mov, "comment": e.comment, "dsl": show(&e.reason), "questionable": QUESTIONABLE.contains(&e.fen) }))
        .collect();
    Value::Array(xs).to_string()
}

fn leak(s: &str) -> Move_ { Box::leak(s.trim().to_string().into_boxed_str()) }

fn position(fen: &str, mov: &str) -> Result<Chess, String> {
    let root: Chess = Fen::from_ascii(fen.trim().as_bytes())
        .map_err(|e| format!("FEN: {e}"))?
        .into_position(CastlingMode::Standard)
        .map_err(|e| format!("FEN: {e}"))?;
    san(&root, leak(mov)).map_err(|_| format!("{} is not legal here", mov.trim()))?;
    Ok(root)
}

#[wasm_bindgen]
pub fn render(fen: &str, mov: &str, dsl: &str) -> String {
    hook();
    let r = position(fen, mov).and_then(|root| Ok(render_reason(&root, leak(mov), &parse(dsl)?)));
    match r { Ok(t) => json!({ "text": t }), Err(e) => json!({ "error": e }) }.to_string()
}

fn mark(v: V) -> &'static str { match v { V::Holds => "holds", V::Fails => "fails", V::Unknown => "unknown" } }

/// The verdict, or `{"need": "fen nodes"}`: run that search, pass it to `searched`, and call again.
#[wasm_bindgen]
pub fn evaluate(fen: &str, mov: &str, dsl: &str) -> String {
    hook();
    let (root, reason) = match position(fen, mov).and_then(|root| Ok((root, parse(dsl)?))) {
        Ok(x) => x,
        Err(e) => return json!({ "error": e }).to_string(),
    };
    let mv = san(&root, leak(mov)).unwrap();
    let cache = CACHE.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut ev = Ev::new(Some(Engine { nodes: NODES, cache, missing: None }));
    let v = ev.run(&root, &mv, &reason);
    let eng = ev.eng.take().unwrap();
    CACHE.with(|c| *c.borrow_mut() = eng.cache);
    if let Some(k) = eng.missing { return json!({ "need": k }).to_string(); }
    let claims: Vec<Value> = ev.claims.iter().map(|(n, c)| json!({ "name": n, "v": mark(c.v), "why": c.why })).collect();
    json!({ "v": mark(v.v), "why": v.why, "claims": claims }).to_string()
}

/// The `info` lines of the search `evaluate` asked for.
#[wasm_bindgen]
pub fn searched(key: &str, lines: &str) {
    let ls: Vec<String> = lines.lines().map(str::to_string).collect();
    CACHE.with(|c| c.borrow_mut().insert(key.to_string(), parse_search(&ls)));
}
