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
