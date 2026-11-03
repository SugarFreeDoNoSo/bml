# Repository guidance

## Boundaries and contracts

- This is a Cargo workspace without a root package. The planned `bml-core` does not exist yet. The `bml` facade reexports the library APIs; the symbolic/tensor implementations live in `bml-domain` alongside the legacy AST/SoA/transformer. Do not conflate their numerical contracts.
- `bml-domain` has no dependencies and denies unsafe code. Native tensor kernels use IEEE arithmetic; `symbolic` evaluation and tensor `apply_basis` enforce real-domain/range checks. A closed symbolic term is not necessarily evaluable.
- New GGUF work belongs in `bml-gguf`; `bml-parser` is the compatibility adapter preserving historical imports and `io::Result`. Keep the reader/codecs independent of compiler/runtime. The `tensor` feature alone adds the optional domain dependency.
- GGML axis zero is contiguous. `GgufTensorView::to_tensor_f32()` reverses dimensions, not flat values: GGML `[3, 2]` becomes row-major `[2, 3]`. Decoding allocates; mapped files must not be modified or truncated while borrowed.
- Cross-crate tests under root `tests/` are explicitly registered in `crates/tests/Cargo.toml` as package `bml-tests`; they are not unregistered tests or a separate root package.

## Verification

- Use the toolchain pinned in `rust-toolchain.toml` (including rustfmt, Clippy and LLVM tools), and `--locked` for Cargo verification. `just setup` updates stable rather than installing that pinned toolchain.
- Canonical local/CI gate: `./scripts/check.sh origin/master` (or pass the actual base ref). Requires Python 3 and the base ref locally. Order: changed-file formatting → Python tooling tests → workspace tests → historical contract audit → traceability → Clippy ratchet.
- Formatting is intentionally limited to changed Rust files. Use `rustfmt --edition 2021 --config skip_children=true path/to/file.rs`; avoid unrelated workspace-wide reformatting.
- Run `python3 tools/quality.py lint`, not `just clippy`, to match CI: existing diagnostics are ratcheted against `quality/clippy-baseline.json`, rather than all warnings being denied. Do not regenerate baselines to make a change pass; coverage floors also require reviewed changes.
- Useful focused checks (append `-- test_name` to a test command to filter):

  ```sh
  cargo test -p bml-domain --test symbolic_contracts --locked
  cargo test -p bml-domain --test tensor_contracts --locked
  cargo test -p bml-domain --doc --locked
  cargo test -p bml-compiler --test contracts --locked
  cargo test -p bml-tests --test gguf_tensor_bridge --locked
  cargo run -p bml-tests --example contract_audit --locked
  ```

- The historical audit must report zero discrepancies; older documentation describing expected failures is historical, not permission to weaken it.
- For GGUF changes, test both `cargo test -p bml-gguf --no-default-features --locked` and `cargo test -p bml-gguf --all-features --locked`. Workspace tests alone do not verify the feature-free boundary: `bml-tests` enables `tensor` through its dev-dependency.
- `just test-compiler` skips proptests; `just test-fast`/`test-ci` are not the full workspace gate. `just test-loom` runs the actual append-only and pending-queue Loom models with `--locked`; the separate `thread_concurrency` system-thread tests do not establish Loom interleaving coverage.
- Ordinary tests need no downloaded model, ptrace or perf. Opt-in instrumentation must use an exact ignored test, not blanket `--ignored` (which also selects helper child tests). Commands and prerequisites: `docs/QUALITY.md`; real-model tests use `BML_TEST_GGUF`.
- Traceability uses `quality/contracts.feature` and `quality/traceability.json`, keyed by Cargo package and test target. If renaming mapped tests, update the mapping; check with `python3 tools/quality.py trace`.

## Further context

- `docs/QUALITY.md` covers coverage/mutation commands, optional instrumentation and API limits; `docs/TESTING.md` covers independent numerical oracles and fixture conventions. Keep performance projections separate from validated inference/tokens-per-second claims.
- `docs/architecture/TENSOR_LIBRARY.md` describes the migration target, not just the implemented architecture. Current reconstruction evidence is under `quality/evidence/`; lost historical runs are not validation of current code.
