## ADDED Requirements

### Requirement: Validated dense Llama execution
The implementation SHALL validate the supported dense Llama configuration,
tokenizer and tensors before execution, preserve causal KV semantics and
reject unsupported variants or exhausted context with explicit errors.

#### Scenario: Independent parity
- **WHEN** the same admitted model, prompt and greedy policy are executed
- **THEN** token IDs match the pinned reference and all finite logits satisfy
  the declared numerical budget over prefill and multiple decode positions.

#### Scenario: Context exhausted
- **WHEN** an input would exceed the session context
- **THEN** it is rejected without overwriting existing KV history.

#### Scenario: Quantized resource access
- **WHEN** quantized projection weights are loaded
- **THEN** their stored representation remains compressed and decoding is bounded.

#### Scenario: Mixed Q4_0 and Q6_K resources
- **WHEN** an admitted Llama GGUF contains Q4_0 layers and a Q6_K output projection
- **THEN** both formats are decoded with the GGML block layout, Q6_K rows contain complete 256-element blocks, and export/reload preserves compressed bytes and generated logits.
- **AND** Q6_K decoding matches the pinned GGML reference, including signed scales and finite IEEE edge cases; unsupported formats still fail explicitly.
