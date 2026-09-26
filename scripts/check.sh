#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 tools/quality.py format --base "${1:-origin/master}"
python3 -m unittest discover -s tools -p 'test_*.py'
cargo test --workspace --locked
cargo run -p bml-tests --example contract_audit --locked
python3 tools/quality.py trace
python3 tools/quality.py lint
