# Exact rational OR/XOR survivors

## Scope and provenance

Replay base: `d9f72a32bc9a71f9f7364ecded381e8bf15c2b6f`.
Isolated snapshot: `target/delivery-snapshots/exact-equivalent-replay`.
Production source was not edited. Each mutation was applied independently to
`crates/domain/src/number/exact.rs`, within the named function's brace-delimited
body, then the original bytes were restored and checked against Git HEAD.

Historical report: `target/spec-mutations-2026-11-03/mutants.out/outcomes.json`,
SHA-256 `c02868d540b005352913f19ad1afd028047ff84cac5fb02137e9c1f2ca4aa9df`.
It remains unchanged, with `end_time=null`; this is not global campaign approval.

## Equivalence argument

For unsigned integers, `a | b == a ^ b` when `a & b == 0`.

* `ExactRational::from_f64`: `(1u64 << 52) | fraction`.
  `fraction = bits & ((1u64 << 52) - 1)` has only bits 0–51 set;
  the other operand has only bit 52 set. This holds for every input reaching
  the expression, independently of sign and exponent.
* `shift_right`: `words[i] >> rest | next << (64 - rest)`.
  The branch requires `rest != 0`, and `rest = shift % 64`, hence 1–63.
  The right-shift operand occupies bits 0 through 63-rest; the left-shift
  operand occupies bits 64-rest through 63. Missing next words are zero.
  The operands never overlap, including when earlier words have been updated.
* `divide_exact`: `(*word << 1) | carry`.
  Each outer iteration initializes carry using `& 1`. Each inner iteration
  assigns the next carry from the old word shifted right by 63. Thus induction
  gives carry in {0,1} at every execution; the shifted word has bit zero clear.

These three substitutions are equivalent, not caught mutants. No artificial
test or production rewrite was introduced to change their mutation score.

## Replays

Command for each variant: `cargo test -p bml-domain --locked`.
Environment: `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_INCREMENTAL=0`, `CARGO_TARGET_DIR=$REPO/target/exact-equivalent-build`,
`TMPDIR=$REPO/target/tool-install-tmp`. The initially fresh build directory and
snapshot path were reused for subsequent rebuilds. All five runs exited zero,
with 167 passing tests per run (including doctests).

Logs are under `target/quality/exact-equivalent-<variant>.log`:

| Variant | Source SHA-256 | Log SHA-256 |
| --- | --- | --- |
| baseline | da29637bbc49161ca1e684f87bd0e1f971af89e9ed9ff8eda3f2d7af024a73bf | b84430d9601f9165c2fe200ec7b724d0d1051b4c85744148b64ae792ec58bc72 |
| significand | 7a673ab2d4da226d92a8b92b20828fbde4e42b912854b1c8d7c10911360a47ff | 79e0248c3cb18d5cf11b73ba6ac79aea7a5d83f0c026d1da44e4275928d3e7ad |
| shift | f8e6d6ab346e43280639f70f25f9713f4ea74b0f395130e7094f13fd4bed4484 | f69087674fe42cfdf22ac4242f7cb128b06a4e159755b6528f26a903aeb73649 |
| division | ee4e83ee1aa595b126234ca632633cd86a95469aefde5c76c4e4f4b5d66320a1 | 490ee39b8113376c02a5173308b88912bbc934b73af6b06c03673686ef0c62d2 |
| restored | da29637bbc49161ca1e684f87bd0e1f971af89e9ed9ff8eda3f2d7af024a73bf | 6b881e7f766ccc7b2801036ed330a35543e1774d7ff384e66457d4a5ecaf9009 |

Machine-readable replay records: `target/quality/exact-equivalent-results.json`.
Two earlier preparation attempts stopped at uniqueness assertions before any
Cargo execution; they are not test results.

Disposition ledger: of 47 historical survivors, 18 have local contractual
detection and these three have equivalence arguments; 26 remain undisposed.
The two historical timeouts have separate local detection evidence. Historical
outcomes remain historical; no complete replacement campaign is claimed.
