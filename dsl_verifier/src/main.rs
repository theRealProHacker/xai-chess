mod dsl;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("baseline") => dsl_verifier::baseline(),
        Some("render") => dsl_verifier::render_report(),
        _ => dsl_verifier::report(),
    }
}
