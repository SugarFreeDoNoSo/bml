## Context

The latest roadmap prioritizes the numeric library and separately specifies
R0–R4 for correct standalone Llama execution. On 2026-11-01 the user explicitly
included both in BML 1.0. The implementation must satisfy these acceptance
criteria rather than rename the existing experimental release.

## Goals / Non-Goals

- G1: stable numeric contracts and optional tensor/GGUF integration.
- G2: independently validated dense Llama inference, first on Linux/amd64 CPU.
- G3: self-contained model executable with declared OS/ABI dependencies.
- G4: reproducible tests and measured latency/memory; no required speedup.
- NG1: GPU, Gemma4, other OS/targets, complex/trigonometric synthesis or a
  distributed production serving platform in this initial profile.

## Decisions

- D1: implement inference in `bml-inference`, not by combining incompatible
  historical execution paths. A model borrows immutable compressed tensors;
  each session owns bounded F32 KV state and checks context before mutation.
- D2: admit dense Llama with RMSNorm, full-head unscaled RoPE, causal GQA,
  SiLU/SwiGLU, residuals and tied or separate output weights. Reject unknown
  variants, missing tensors and invalid shapes. The plan is explicit.
- D3: reuse standalone GGUF codecs and decode at most one row for projection;
  never expand all quantized weights into an F32 pool.
- D4: serialize normalized configuration, tokenizer and tensor descriptors,
  align raw tensor segments, and checksum metadata/resources with SHA-256.
  Append the payload and versioned footer to the native ELF. Runtime maps
  its own executable; it does not parse a hidden GGUF or invoke llama.cpp.
- D5: reference revision is llama.cpp
  `785322d17f2d51e5185cf996a1f4e46d87640841`; first candidate is
  `ggml-org/tiny-llamas` revision `99dd1a73db5a37100bd4ae633f4cfce6560e1567`,
  `stories15M.gguf`, SHA-256
  `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`.
  Reference uses CPU, one thread, F32 KV, no flash attention and greedy
  selection. Initially require finite full-vocabulary logits satisfying
  `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)` for F32 weights.
  Quantized comparison needs a separately declared error budget; failures
  must not be hidden by relaxing the F32 threshold after observation.
- D6: synthetic unit fixtures do not establish model parity. Use fixed
  prompts, multiple decode positions, tokenizer edge cases and an isolated
  execution that only receives the binary and its declared system libraries.
- D7: fixed generation prompts are `Once upon a time`, `Hello`,
  `  café\n世界 😀`, and `The little girl`, each with four greedy tokens and
  context 128. Additional tokenizer cases cover empty input, spaces,
  newlines, Unicode and literal control-token spellings. The gate compares
  all vocabulary logits at prefill and every generated step, text and bytes.

## Risks / Trade-offs

- R1: reference CPU reductions differ; inspect the first divergent operation
  before accepting numerical differences.
- R2: a generic interpreter hidden in an executable would not establish the
  planned AOT contract. Generated source must bind the selected model/plan
  and link native BML kernels, without external model/runtime installation.
- R3: mmap requires immutable backing files. Hashes detect corruption but
  are not signatures; hostile executable provenance is not established.
- R4: scalar CPU correctness may be slower than llama.cpp. Measure actual
  inference after correctness, and report results without extrapolating TPS.
- R5: model license and redistribution rights remain attached to resources;
  keep validation artifacts local until an explicitly authorized release.
