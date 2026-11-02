## ADDED Requirements

### Requirement: Autonomous model executable
The compiler SHALL produce a native Linux x86_64 executable containing the
validated plan, tokenizer and normalized model resources, linked to BML's
own CPU execution kernels with declared system dependencies.

#### Scenario: Clean execution
- **WHEN** only the executable and declared system libraries are available
- **THEN** inference succeeds without a GGUF, BML installation, compiler,
  Python, llama.cpp or network and reproduces the validated outputs.

#### Scenario: Corrupt resources
- **WHEN** embedded resource hashes, ranges, alignment or format are invalid
- **THEN** startup reports an error before executing model operations.
