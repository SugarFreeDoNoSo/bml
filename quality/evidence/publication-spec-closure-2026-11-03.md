# Publication task closure — 2026-11-03

## Scope and authoritative state

The user explicitly requested committing and pushing ready local changes, verifying
the remote branch and reporting unpublished work. The publication comprised:

- `ca8d030ceaebc17cfb36a8e1c3575e9238a3f257` — implementation, contracts, tests and development tooling.
- `1ef5f5d39a1d7ad6b3a0065f98164d1121044d4b` — documentation, specifications, archives and scoped evidence.

A fresh `git ls-remote origin refs/heads/master` at
`2026-11-03T21:06:36+00:00` returned
`1ef5f5d39a1d7ad6b3a0065f98164d1121044d4b`. Local `HEAD` and `origin/master` resolve to the same commit.
The publication itself was completed before these task-status corrections;
these corrections are not claimed to be already published.

`git merge-base --is-ancestor` returned zero for each previously recorded
GGUF-weight delivery commit (`d03d4a2`, `999f6df`, `858d8fc`, `bec670b`,
`dffba3c`, `fa2dceb`, `1c984ee`, `29ca9c4`, `bb1f9ff`) and for
`ca8d030` against the verified remote commit. The corresponding code is
therefore included in the published history.

## Task decisions

| Change and task | Current disposition | Reason |
| --- | --- | --- |
| `fix-placeholders` 9.3 | Complete | The task requires commit and push; both are verified. No archive was performed by this audit. |
| `bml-inference-pipeline` 10.3 | Complete | Implementation and evidence were published. The independent requirements 8.6 (symbolic real-model range/scalability) and 9.6 (PMU/L1i) remain open. |
| `bml-exact-numbers-and-matrices` 9.3 | Partially complete; checkbox remains open | Publication is complete, but this task also requires archival after all contracts. ARM/NEON execution requirements 7.3–7.4 are still open. |
| `gguf-to-bml-weights` 5.3 | Partially complete; checkbox remains open | Delivery commits and documentation are published. Final contract review and archival remain separate work. The crate's mutation result does not certify the workspace gate. |
| `bml-1-0-release` 6.3 | Partially complete; checkbox remains open | Ready implementation and evidence were published with authorization. Final release delivery still depends on the unapproved global gate in 6.2. |

No release, hardware, numerical or global mutation requirement was marked
complete merely because its documentation or implementation was pushed.
The goal of finishing all specifications remains unfulfilled.

## Verification provenance

The publication evidence file is
`quality/evidence/publish-ready-2026-11-03.json` (SHA-256
`065805cf18e10486df11a7f617ce342381e9b7839db9b85cc68c9f8f6a6473a8`).
Its recorded terminal exit codes are zero, and this audit recomputed the
following log hashes successfully. These are the completed publication checks,
not new build/test runs performed by this audit.

| Check | Log | Verified SHA-256 | Recorded exit |
| --- | --- | --- | --- |
| canonical | `target/quality/publish-ready-worktree-gate.log` | `a52be8b25aa6fb4c5161a72e5fc495fb99e6c94944f5ace0b89aad856149a071` | 0 |
| features | `target/quality/publish-ready-features.log` | `45503513a748b003dc83d0e13e810f3040f4318b86d5c5176e8f2aa949a21e69` | 0 |
| openspec | `target/quality/publish-ready-openspec.log` | `c057d40ba75ba779a12f9596c1d64c0deb0b80ab7d41bfc6a8077c73b0fda44c` | 0 |

After the task-status edits, the installed OpenSpec 1.13.1 CLI ran
`openspec validate --all --strict`: 11 passed, zero failed, exit 0.
`git diff --check` over the five edited task files also passed.
No tests, builds, commits, pushes or archive commands were run by this audit.

## Unpublished local material at the publication boundary

The original remote-verification record in
`target/quality/publication-remote-verification.json` reports only
`.serena/.gitignore` and `.serena/project.yml` as untracked working-tree files.
`target/` and `sandbox/.local/` remain ignored, including model/build files,
user notebook edits and the experimental bounded-RMS diagnostic. This describes
the verified publication boundary, not future changes made while development
continues.
