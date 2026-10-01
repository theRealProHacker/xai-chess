mod dsl_claude;
mod eval;
mod render;
mod text;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(not(target_arch = "wasm32"))]
pub use eval::{baseline, report};
#[cfg(not(target_arch = "wasm32"))]
pub use render::report as render_report;
