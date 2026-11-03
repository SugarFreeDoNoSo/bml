# Provenance

The shared algorithm is adapted from `crossbeam-queue 0.3.13`,
`src/array_queue.rs`, SHA-256
`d3c421b83cb0ad6cc01ef6a249a87060f1e5753a5238cb6242feff98cade906d`.
The original MIT and Apache-2.0 licenses are preserved alongside this file.

Production and Loom compile exactly the same `src/array_queue.rs`. Index
arithmetic, CAS operations, fences and memory orderings are unchanged. Adaptations
are import paths, scoped UnsafeCell access, exclusive atomic access helpers,
formatting, and documentation examples using this crate. `adaptation.diff` records
all changes against the pinned upstream file.

The public queue always uses native synchronization, including with `loom`
enabled. The private model namespace substitutes Loom atomics and UnsafeCell;
backoff yields through Loom. Unsafe slot access stays in this dependency, so
`bml-runtime` keeps its `deny(unsafe_code)` boundary.

Run `cargo test -p bml-queue --all-features --locked`. Six small models exercise
admission, concurrent consumption, publication before producer completion, slot
reuse, and destruction. The native API and eleven documentation examples also
run. The full upstream API is retained, but these models cover the subset used by
`PendingWork`; they do not exhaustively verify force_push or exclusive iteration.
See `quality/evidence/pending-work-loom.json` in the workspace for executed
checks and isolated memory-order sensitivity replays.

This covers the pending-work ArrayQueue only. It is not instrumentation of
Crossbeam deque, epoch reclamation, remote delivery, or work-stealing liveness.
