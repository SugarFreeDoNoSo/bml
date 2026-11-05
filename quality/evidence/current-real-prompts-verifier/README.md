# Portable verification of completed Hello/Hi diagnostics

This verifier is prepared for source `bbb89735100e2e4ec6486f3b7029e48977944f3c` and measured binary `f65902f207d85ff0cb0a3ccb148604d1f177199c7de8fdaead0bb0ffbd931c2f`. It verifies saved bytes and original build provenance. It does not run Cargo, execute the binary, load a model into an inference runtime or make network requests.

Run it only after the serial controller has saved **both** full cases as completed:

```sh
python3 quality/evidence/current-real-prompts-verifier/verify_real_prompts.py \
  /home/dev/Work/bml \
  /home/dev/Work/bml/target/quality/real-prompts-bbb89735100e \
  > completed-prompts-verification.json
```

Check exit status. Pending, failed, partial or native-only results return nonzero and emit no success JSON. The verifier requires Hello `[1,15043]` and Hi `[1,6324]`, each with two independent native/symbolic cache positions and four complete vectors. It does not infer a pass from the published launch receipt or native preflight.

The repository argument must contain the Git object for the pinned revision, plus the published `quality/evidence/current-real-prompts-build-native.{bundle.json,tar.gz}`. The archive and bundle digests are literal pins in the verifier. The results directory must contain `full-cases.json`, `full-hello{.log,.exit}`, `full-hi{.log,.exit}`, and both output folders. Each output folder must contain exactly `protocol.json`, `summary.json`, `position-{0,1}.json`, and `{native,symbolic}-{0,1}.f64le`.

The immutable measured binary, pinned GGUF and the 13-file AOT directory must also remain available. Their original locations are taken from authenticated receipts. Copies may be supplied with `--binary`, `--model` and `--artifact`; their hashes remain mandatory. The result directory and repository may be relocated, preserving recorded metadata and relative file names. The original source worktree and Cargo target are not required: read-only Git blob verification reconstructs the captured inputs.

| Facet | Required evidence and check |
|---|---|
| Original compilation | Pinned archive bytes, exact original Cargo command, JSON stdout/stderr hashes, successful build-finished, driver and six numerical libraries recorded with `fresh:false`, measured binary hash. |
| Source snapshot | All 796 captured before/after input records agree and match blobs in pinned Git tree `81da51727c6598fc70a863f4177b71eb2baf352f`; the driver and its three support modules are included. This authenticates that revision, not arbitrary later HEAD changes. |
| Model and AOT | GGUF SHA256 `da3087…`; all 13 regular AOT files, exact sizes/hashes; manifest SHA256 `0b6e9b…`, version4, context2, source hash and bounded profile. |
| Predeclared prompts | Full protocol must match the authenticated native preflight protocol with only its mode changed to full. Fixed text, Rust-encoded IDs, source/binary, profile, limits and capacity therefore remain fixed. |
| Native preflight | Preserved as native-only, `bml_executed:false`, required cache positions1/2. It cannot satisfy full execution. |
| Complete execution | Both controller cases have exit0, exact new ignored test command, correct output paths and ordered timestamps; saved log contains its passing harness and both position records. Failure files and partial output folders are rejected. |
| Vectors | Four per prompt, exactly 32,000 finite little-endian f64 values each; all raw SHA256 values retained. No fixed predicted output token is imposed. |
| Statistics | Recompute full-vector maximum and sequential mean absolute error, plus scaled RMS in the driver's arithmetic order. Saved statistics must match exactly. Argmax follows Rust `total_cmp`, including signed zero and the last equal maximum. |
| Cache and BML families | Native and symbolic cache positions1/2, exact protocol hash, actual ten executed family names. Neither a compile-only artifact nor a native-only summary qualifies. |
| Interpretation | No global numeric tolerance, native/BML argmax equality requirement, throughput claim or task-completion declaration is added. Statistics equality checks report integrity, not numerical parity. Per-operation contracts and other model/prompt obligations remain separate. |

Small synthetic fixtures test complete saved vectors, tampered statistics with rehashed JSON/logs, cache state, signed-zero argmax, scaled RMS, missing/partial files, wrong commands and native-only substitution. The positive fixture deliberately has maximum error0.25, demonstrating that no retrospective parity threshold is introduced:

```sh
python3 -m unittest discover \
  -s quality/evidence/current-real-prompts-verifier -p 'test_*.py' -v
```

The original build/archive and 796 Git inputs were authenticated during preparation, without executing the model. A live pending-run check was correctly rejected. Both full real results remain pending; synthetic fixtures and input authentication are not full-model validation.
