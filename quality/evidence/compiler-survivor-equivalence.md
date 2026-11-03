# Six compiler survivor dispositions

These are equivalence/redundancy arguments for the current public entry points,
not detected mutants. They depend on the inspected builders and validation paths;
changes to those dependencies require reconsidering the dispositions. They do not
claim equal instruction counts or allocation traces.

Base: `7126d221e8b1a5e6581fc30be91c360413e06f79`. The initial isolated replay ran
`cargo test -p bml-compiler --locked` on the original, each independent mutation,
and the restored original: 178 tests passed in each of these six mutated runs.
The historical source-line expressions were taken from the recorded diffs, not
inferred from nearby operators. In particular, the projection mutation affects
the **u32 index ceiling**, not the caller's node budget.

Replay records: `target/quality/pending-survivors-replays.json`, SHA-256
`7992c9653d7ef8d2e446e691c4ba33194149cb164ddb19cc28bce5adab1b593a`.
Original/mutant/restored source and log hashes are recorded there. Source snapshots
are under `target/delivery-snapshots/pending-survivors-replay`. The report at
`target/spec-mutations-2026-11-03/mutants.out/outcomes.json` remains unchanged
(SHA-256 `c02868d540b005352913f19ad1afd028047ff84cac5fb02137e9c1f2ca4aa9df`).

## 1. Empty projection guard (replay 5)

`causal_artifact.rs:143:29`, `columns == 0 || rows == 0` → `&&`.

Every caller of `required_projections` supplies a compiler created through the
mapped GGUF path. For transformer layers, `Model::from_gguf` validates nonzero
configuration dimensions and the required tensor shapes. For zero-layer models,
the only projection is the explicit output head or tied embedding: mapped model
admission requires `[n_embd, vocab.len()]`, nonzero `n_embd`, a valid nonempty
vocabulary, and a decodable first row. Thus every projection reaching this guard
has both dimensions positive. Neither predicate can fire. Dimension maps are not
publicly mutable. This does not justify removing the defensive guard.

## 2. Preliminary program minimum (replay 18)

`symbolic_artifact.rs:115:20`, `minimum > program_limit` → `>=`.

Only equality differs. For shared-output plans, `minimum=context>0`. A layered
plan contains at least one Dot and one Softmax key for each prefix width, hence
at least `2*context` programs. When `context==program_limit`, the original also
returns `symbolic AOT program budget exceeded` at the complete-plan count check.
Compilation clamps the limit to 4096; loading uses 4096, so the differing work
is bounded. The mutant rejects earlier, without weakening the allocation bound.
For legacy scalar loading the limit is always 4096 and equality is impossible:
`90*91/2=4095`, `91*92/2=4186`. Zero-layer plans skip this branch.

The separate `/`→`%` mutation is **not equivalent**: it bypasses the early bound
for arbitrarily large legacy contexts. The allocation contract detects it.

## 3. Remaining program counter (replay 19)

`symbolic_artifact.rs:194:32`, `remaining.max_programs -= graphs.len()` → `+=`.

Compilation always requests shared outputs. Its complete required-key preflight
has already established `keys.len() <= limits.max_programs <= 4096`. Every key
produces exactly one graph: buffered Norm/Softmax/Rope, or scalar
Dot/Swiglu/Add/Scale. Before each successful compile the original remaining count
is at least one; raising it instead cannot change any admission or emitted code.
This counter is used only for program-count admission, not construction choices.
The changed counter stays at most 8192, so it cannot overflow supported usize.
Other budgets and error paths are unaffected. The defensive decrement remains.

## 4. Constructed node-count assertion (replay 23)

`symbolic_kernels.rs:243:27`, `soa.len()!=nodes || soa.len()>max_nodes` → `&&`.

The earlier admission has established `nodes <= max_nodes` and `nodes <= u32::MAX`.
For fixed-width operations, `required_nodes` runs the same deterministic builder
with the same parameters; capacity only reserves storage. For nonempty Norm and
Softmax, construction is affine in width: per-input nodes, exactly width-1 sum
additions, a fixed normalization/reciprocal block, and per-output nodes. Arithmetic
identities allocate a fixed number of nodes and do not branch on numeric values.
Consequently the measured width-1/2 increment is exact for every admitted width,
and the real builder produces `soa.len()==nodes`. Both operands of the assertion
are false. Dot returns through its separate row compiler before this assertion.

## 5. Constructed program-basis assertion (replay 24)

`symbolic_kernels.rs:388:43`, nonempty pool `||` forbidden opcode → `&&`.

`finish` has only two call sites. Dot's row builder uses external constants and
never appends owned values. Every other path clears each program's owned pool
before calling `finish`. Builders use only One, Var, Const and Bml nodes, with
stack bookkeeping from the linearizers and StoreResult for buffered outputs.
They do not construct Zero, native arithmetic, indexed reads or loops. Every
emitted opcode is therefore accepted by `is_symbolic_opcode`; both original
predicates are false. Persisted bytecode does not enter this builder helper:
artifact loading has its own independent checks and malicious-opcode contracts.

## 6. Row's u32 ceiling (replay 25)

`symbolic_projection.rs:49:35`, `nodes > u32::MAX` → `>=`.

Count the current append-only identities: exp2=2 nodes, log2=6, sub=9,
add=21, mul=35, mul_signed=197. A row of width w therefore has
`w*(197+2)+(w-1)*21 = 220*w-21` nodes. The two comparisons differ only at
`nodes=4,294,967,295`, but `(4,294,967,295+21) % 220 = 156`, so no integer width
reaches that count. Checked arithmetic rejects overflow before the comparison.
The caller's independent inclusive `max_nodes` comparison is unchanged.

The size probe independently measured product=197 and add=21 (plus several
compiled opcode counts). Log `target/quality/symbolic-size-probe.log`, SHA-256
`56837ec4125ee4e7e14945d80c62fe1e0803eb606944ac8fadcdaeee74974d4b`.
Measurements supplement the structural proof; finite samples alone are not it.

## Expanded-suite replay

After integrating the additional compiler contracts, the original, each of the
six substitutions, and the restored original all pass **193 tests**. These are
eight independent executions, not caught mutants. Records:
`target/quality/compiler-equivalence-replays.json`, SHA-256
`f3ebaeb7234d3e82f7f9d7b39e7fb91bfcb61371a280f70850d64922549fcd8f`.

## Source identities

| File | SHA-256 |
| --- | --- |
| crates/compiler/src/causal_artifact.rs | e233e88cff1f92d29d53e0aae1e2baf9a5d5dd89e7abfe9390e7da2075ce014d |
| crates/compiler/src/symbolic_artifact.rs | bfef57d09defe4385d53f7c9096d9b3ec3cfc5ad4a1182296ef821b9ebd239ce |
| crates/compiler/src/symbolic_kernels.rs | 1d85b37f6b2d0c28c011fa98dac0fc2b7ffa9d77c878761441d9ca2a5c99c368 |
| crates/compiler/src/symbolic_projection.rs | b04b6fced0e52faf828efc72623d138a340e4fb4202eee9de37ba92f0c7b0210 |
| crates/domain/src/transformer.rs | 3262e23322739f5c6e795c43a3652b467da1083a1eb864303ac6d616a90a56bd |
| crates/domain/src/transformer/operations.rs | d5789f282379d04bba9aacb47051a662d1fbff004712707a9ff16442f069f777 |
