# Local disposition of the 47 historical survivors

All 47 `MissedMutant` records now have a local disposition: **38 detected by
contracts, 9 equivalent/redundant under the current contracts, 0 undisposed**.
The two historical timeouts have separate local detection evidence in
`directory-eof-timeout-replay.json` and `rpn-preflight-timeout-replay.json`.

This is not global mutation approval. The historical report still has
`end_time=null`; equivalence exclusions are not preapproved by the quality gate.
The report and its original categories have not been rewritten.

## Provenance

* Base commit: `7126d221e8b1a5e6581fc30be91c360413e06f79`.
* Historical report: `target/spec-mutations-2026-11-03/mutants.out/outcomes.json`,
  SHA-256 `c02868d540b005352913f19ad1afd028047ff84cac5fb02137e9c1f2ca4aa9df`.
* Complete per-mutant ledger, including prior detecting evidence and exact
  historical names/diffs: `target/quality/survivor-dispositions-ledger.json`,
  SHA-256 `b5d20bd18a2932218e39469121ae79803d52ae3030209406dc6cb17266487712`.
* The earlier 18 detections remain supported by their individual committed
  evidence reports. This follow-up adds 20 detections and six equivalence
  arguments. Three OR/XOR equivalences were already documented separately.

Mutations were applied independently in isolated source snapshots. Each build
directory was initially fresh; subsequent mutations rebuilt at the same source
path. Original implementations were tested before and after mutations. Production
sources were not mutated. The distributed reader overlay preserves the prior
pending implementation and adds three regression tests.

Common environment: pinned Rust 1.98.1, `--locked`, `CARGO_INCREMENTAL=0`,
`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, and
`TMPDIR=$REPO/target/tool-install-tmp`. Main replay snapshot/build:
`target/delivery-snapshots/pending-survivors-replay` and
`target/pending-survivors-build`. Distributed replay uses
`target/delivery-snapshots/distributed-survivors-replay` and
`target/distributed-survivors-build`.

## Twenty additional detections

Historical line numbers identify the recorded diffs; current source locations
may differ. Replay indexes refer to `target/quality/pending-survivors.json`.

| Index | Historical location | Detecting contract |
| --- | --- | --- |
| 0 | cli/main.rs:135:9 | Existing artifact-marker admission test |
| 1 | cli/main.rs:219:29 | Existing persisted fragment-size CLI test |
| 2 | cli/main.rs:413:47 | Compiled generation outgrows a three-token prompt, emitting five tokens without EOS |
| 3 | cli/main.rs:501:21 | Last requested native token succeeds without evaluating an unrequested overflowing next position |
| 4 | cli/main.rs:884:34 | Same terminal-token contract through the RPN CLI |
| 7 | compiler/distributed.rs:382:58 | Valid shard above 1 MiB remains admissible |
| 8 | compiler/distributed.rs:383:50 | Valid header above 4 KiB remains admissible |
| 9 | compiler/distributed.rs:470:44 | 65,537 fully encoded tensor records exceed the independent count ceiling |
| 10 | compiler/distributed.rs:470:96 | Exactly 65,536 valid tensor records are admitted |
| 11 | compiler/distributed.rs:596:20 | Every opcode decodes at its exact wire width; each truncated prefix fails |
| 12 | compiler/gguf_compiler.rs:561:13 | Independent zero head/embedding/context metadata rejection, mapped and decoded |
| 13 | compiler/gguf_compiler.rs:1247:32 | Finite gate/up projections overflow in SwiGLU; no backend receives the nonfinite activation, and cache position is not committed |
| 14 | compiler/gguf_compiler.rs:1272:74 | Four-component head uses angles position and position/4; F32 cached keys match an independent trigonometric oracle |
| 15 | compiler/kv_cache.rs:45:52 | Default cache admits 1 MiB and rejects payloads exceeding 512 MiB before allocation |
| 6 | compiler/causal_artifact.rs:218:20 | A 3 MiB GGUF token expands past 16 MiB in JSON; native publication is rejected |
| 16 | compiler/symbolic_artifact.rs:100:36 | Zero-layer symbolic plan retains optional final normalization |
| 20 | compiler/symbolic_artifact.rs:198:38 | Exactly 65,536 generated fragments load; a larger compilation is rejected |
| 21 | compiler/symbolic_artifact.rs:236:20 | JSON expansion is rejected before symbolic publication |
| 17 | compiler/symbolic_artifact.rs:112:28 | Legacy oversized context is rejected before materializing its prefix plan |
| 22 | compiler/symbolic_artifact.rs:358:24 | Aggregate stored bytes admit exactly 64 MiB and reject one more byte |

The initial package-wide replay detected indexes 0 and 1 with existing tests.
The other 18 passed their previous suites and failed after the new contracts.
Failures are test assertions, not compile failures. The distributed original
passes 102 tests after additions; the numerical original passes 26 focused tests;
generation passes both new tests. The final expanded compiler suite passes 193
tests with the original and each of the six equivalent/redundant substitutions.

The preflight mutant still returns the same budget error, but allocates
17,648,686 bytes for the rejected plan. The allocation contract permits a generous
8 MiB for parsing/validation of this tiny fixture and detects that resource
regression. The aggregate-byte fixture isolates container admission using empty
fragment records; it is not evidence of executable model semantics. Both tests
exercise production entry points and unchanged production limits.

An initial draft of the numerical test failed on the original because a residual
overflow was rejected earlier as `non-finite layer output`. That failed attempt
is retained in `numeric-baseline-new.log` and `numeric-survivor-replays.json`.
The corrected test uses SwiGLU overflow to exercise projection input admission;
the v2 baseline passes. No implementation was changed to fit the test.

## Verification records

Each JSON record below contains commands, exit codes, log paths and SHA-256
hashes. All referenced log hashes were rechecked. Baseline/restored runs exit 0;
detecting mutant runs exit 101. The original source bytes were verified after
restoration. Logs are retained under `target/quality/`.

| Record under target/quality/ | SHA-256 |
| --- | --- |
| pending-survivors-replays.json | 7992c9653d7ef8d2e446e691c4ba33194149cb164ddb19cc28bce5adab1b593a |
| distributed-survivor-replays.json | 96d8d4105505161b85fa84573ae0afe6877c02cac956e25dc43fef15cfe8ba38 |
| numeric-survivor-replays-v2.json | 4943218472a5b416b3d96d1010d27b44a66ef764c1bc51f75afde326fee21607 |
| generation-survivor-replays.json | 8037b995eb153fae14b68e6099d9db478d7435888846eec8abcd04f32399455e |
| manifest-budget-replays.json | 8195b7fdd6f43e223ac409bd0d67feabd1279be31b9f8165b12ea1e242f00322 |
| artifact-plan-replays.json | 35a451c73f23376dd1e095aff65d370d5d221becfe9beabdaacec1534643a377 |
| artifact-resource-replays.json | 98d5d4fc1b01278935fd8374dd1cecfd7e52291afbc7e17c8203b0b4395f1d90 |
| compiler-equivalence-replays.json | f3ebaeb7234d3e82f7f9d7b39e7fb91bfcb61371a280f70850d64922549fcd8f |
| survivor-delivery-inputs.json | 8859dde8c7052bd5e26d3bb8f82df6e8e08211038dfff5a9b4f85f07b4069471 |

Commands: `cargo test -p bml-cli --locked`,
`cargo test -p bml-compiler --locked`; focused variants additionally select
`--test generation_budget_contracts`, `--lib --test distributed_loading_contracts`,
`--test causal_contracts --test inference_contracts`, or
`--test artifact_manifest_budget_contracts --test artifact_preflight_allocation_contract`.
Equivalence proofs: `compiler-survivor-equivalence.md` and
`exact-or-xor-equivalence.md`. Passing runs alone do not establish equivalence.

## Historical Unviable audit

962 of the 1,079 historical `Unviable` logs contain the same E0505 error at
`crates/domain/tests/bml_matrix_boundaries.rs:87:54`: moving `two` while a closure
borrows it. This also occurs when the mutant is in another crate. These records
do not establish intrinsic mutant non-compilability. No particular race or root
cause is asserted from this diagnostic alone.

Full affected-name/log/hash inventory:
`target/quality/global-unviable-test-error-audit.json`, SHA-256
`fe4b2c3c55e4f9c222fc42a67f2472e6d818f354135f707d94f65df98bcf0462`.
The old report remains an incomplete historical result. A replacement campaign
must use a frozen source tree and pass the original workspace baseline first.

## Delivery gate

`./scripts/check.sh HEAD` passed in the isolated delivery snapshot, with HEAD
fixed at the recorded base and the nine source/test overlays identified by
`survivor-delivery-inputs.json`. Results: **656 Rust tests, 63 Python tests,
0 historical discrepancies, traceability 28/28, 0 new Clippy diagnostics**
(69 current versus 192 baseline). Changed-file formatting passed. The tested
overlay hashes match the delivered working files.

Log: `target/quality/survivor-delivery-gate.log`, SHA-256
`b2eab8b1d1dfe8c757bbf564dd6ae3c352239481138c8e1fbe88fdcc1dd30c85`.
This gate covers this delivery snapshot. The larger pending worktree and a new
complete mutation campaign require their own verification.
