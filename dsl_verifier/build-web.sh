#!/usr/bin/env bash
# Builds the renderer and evaluator for docs/render.html into docs/vendor/dsl/.
# Needs: rustup target add wasm32-unknown-unknown; cargo install wasm-bindgen-cli --version 0.2.104
set -euo pipefail
cd "$(dirname "$0")"
cargo build --lib --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir ../docs/vendor/dsl \
  target/wasm32-unknown-unknown/release/dsl_verifier.wasm
