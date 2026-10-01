mod dsl;
mod dsl_claude;
mod eval;
mod render;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("baseline") => eval::baseline(),
        Some("render") => render::report(),
        _ => eval::report(),
    }
}
