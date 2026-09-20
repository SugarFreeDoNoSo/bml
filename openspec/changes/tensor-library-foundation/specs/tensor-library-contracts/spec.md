## ADDED Requirements

### Requirement: Tensor data is separate from symbolic terms
The library SHALL store tensor values in validated buffers, borrowed views or quantized blocks rather than requiring one pure symbolic tree per element.

#### Scenario: Load a GGUF weight tensor
- **WHEN** a supported tensor is loaded
- **THEN** its dimensions, dtype, block layout and exact byte range are preserved or converted explicitly

### Requirement: GGUF reads have precise bounds
The reader SHALL respect general.alignment, validate dimensions and checked byte ranges, and reject truncated supported blocks.

#### Scenario: Read custom alignment
- **WHEN** a valid F32 tensor uses alignment 64
- **THEN** the reader returns its known values rather than padding bytes

#### Scenario: Decode Q4_0
- **WHEN** a complete Q4_0 block is decoded
- **THEN** low nibbles populate the first 16 values and high nibbles the following 16 values, with the block scale applied

### Requirement: Compiled plans preserve context and values
A compiled plan SHALL preserve constant pools and the identity of reused values, and SHALL retain its operator basis and numerical policy.

#### Scenario: Reuse a non-top value
- **WHEN** Op(Op(x,y),x) is compiled for x=1 and y=4 under BML
- **THEN** execution returns 1 instead of duplicating the current stack top
