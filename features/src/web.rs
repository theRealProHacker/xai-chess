//! The browser API: JSON in and out.

use std::cell::RefCell;

use serde_json::json;
use wasm_bindgen::prelude::*;

thread_local! {
    static PANIC: RefCell<String> = const { RefCell::new(String::new()) };
}

fn hook() {
    std::panic::set_hook(Box::new(|i| PANIC.with(|p| *p.borrow_mut() = i.to_string())));
}

/// The message of the panic that trapped the last call.
#[wasm_bindgen]
pub fn last_panic() -> String { PANIC.with(|p| p.borrow().clone()) }

#[wasm_bindgen]
pub fn families() -> String { hook(); serde_json::to_string(&crate::families()).unwrap() }

#[wasm_bindgen]
pub fn examples() -> String { crate::EXAMPLES.to_string() }

/// Every feature, on and off. `top` is the engine's best moves, space-separated UCI, or empty.
#[wasm_bindgen]
pub fn extract(fen: &str, top: &str) -> String {
    hook();
    let top: Vec<String> = top.split_whitespace().map(String::from).collect();
    match crate::extract_fen(fen, (!top.is_empty()).then_some(&top[..])) {
        Ok(fs) => json!({ "features": fs }),
        Err(e) => json!({ "error": e }),
    }.to_string()
}
