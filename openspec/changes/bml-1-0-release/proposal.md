## Why

The user selected a complete BML 1.0: the numeric library, correct Llama
inference checked against llama.cpp, and autonomous AOT executables. The
existing library has contracts, but the historical inference implementations
contain placeholders and the existing standalone artifacts only run synthetic
matrix kernels. Those implementations cannot establish release readiness.

## What Changes

- Preserve the numeric, tensor and independent GGUF APIs and their contracts.
- Add a separate CPU inference layer with a strict dense Llama profile,
  SentencePiece BPE, bounded KV state and explicit unsupported-model errors.
- Keep compressed weight resources; decode individual rows for CPU kernels.
- Export a versioned normalized resource container and generate a native
  executable bound to its model and execution plan, with embedded resources.
- Verify real tokens/logits against a pinned independent reference and run
  the resulting executable without the source model or build tools.
- Record reproducible correctness and performance evidence before versioning.

## Capabilities

### New Capabilities
- `bml-release-contract`: release scope and evidence requirements.
- `llama-profile`: validated dense Llama CPU execution and tokenization.
- `standalone-artifact`: normalized resources and autonomous native execution.

### Modified Capabilities
None. Historical experimental APIs remain distinct; their old completion
checkboxes do not serve as validation for this release.

## Impact

New `bml-inference` layer and command-line tools, independent of legacy
`bml-compiler`/`bml-runtime`. No inference dependency is added to `bml-domain`
or to the default numeric facade. Linux x86_64 GNU CPU is the first target.
GPU, other architectures, Gemma4, complex arithmetic and advanced distributed
scheduling remain outside the first supported release profile.
