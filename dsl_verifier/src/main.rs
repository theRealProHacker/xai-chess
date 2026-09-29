mod dsl;
mod dsl_claude;
mod eval;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("baseline") => eval::baseline(),
        _ => eval::report(),
    }
}
