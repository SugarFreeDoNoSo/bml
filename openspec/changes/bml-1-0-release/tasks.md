## 1. Contract and references
- [x] 1.1 Confirm full 1.0 scope and first target with the user.
- [x] 1.2 Record supported Llama profile and independent reference revision.
- [x] 1.3 Record model identity, fixed prompts and parity tolerances before validation.

## 2. Import and tokenizer
- [x] 2.1 Validate model dimensions, tensor roles, dtypes and finite resources.
- [x] 2.2 Implement SentencePiece BPE, scores/types, byte fallback and BOS/EOS.
- [x] 2.3 Verify tokenizer against pinned llama.cpp, including Unicode/whitespace.

## 3. Model execution
- [x] 3.1 Implement full causal Llama blocks and final logits with independent tests.
- [x] 3.2 Verify KV/prefill/decode equivalence, context bounds and failed-request rollback.
- [x] 3.3 Preserve quantized resources and expose execution/memory plans.
- [x] 3.4 Match full real-model logits and greedy sequences against the reference.

## 4. Resources and AOT
- [x] 4.1 Implement normalized resource schema, checked ranges and SHA-256 integrity.
- [x] 4.2 Generate and link model-specific native executable with embedded resources.
- [x] 4.3 Test malformed/corrupt/truncated resources and unsupported configurations.
- [x] 4.4 Match native standalone output against the validated model/reference.
- [x] 4.5 Run in an isolated environment without model, BML, toolchain or network.

## 5. Release evidence
- [x] 5.1 Verify numeric facade, feature matrix and independent GGUF boundaries.
- [x] 5.2 Record measured prefill/decode latency, RAM and executable size.
- [x] 5.3 Update entrypoint documentation, version and supported-profile limitations.
- [x] 5.4 Review public errors/APIs, resource budgets and legacy compatibility.

## 6. Cierre
- [x] 6.1 `openspec validate bml-1-0-release` passes.
- [ ] 6.2 Canonical workspace gate, feature checks and applicable coverage/mutation pass.
- [ ] 6.3 Commit the reviewable release work with reproducible evidence; publish only when authorized.
