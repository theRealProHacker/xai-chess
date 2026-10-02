#!/usr/bin/env bash
# Builds the feature extractor for docs/features.html into docs/vendor/features/.
# Needs: rustup target add wasm32-unknown-unknown; cargo install wasm-bindgen-cli --version 0.2.104
set -euo pipefail
cd "$(dirname "$0")"
cargo build --lib --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir ../docs/vendor/features \
  target/wasm32-unknown-unknown/release/features.wasm
