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

### Requirement: Tensor shapes and storage are validated
The library SHALL represent rank-zero tensors with one element, any zero dimension with zero elements, and reject overflowing nonempty dimension products. Owned and borrowed buffers SHALL have exactly the validated element count. Coordinates SHALL be bounds checked before checked offset arithmetic.

#### Scenario: Empty shape with huge preceding dimensions
- **WHEN** a shape is [usize::MAX, 2, 0]
- **THEN** it is empty with canonical zero strides and an attempted coordinate fails bounds validation without overflowing

#### Scenario: Reshape borrowed and owned values
- **WHEN** a reshape preserves the element count
- **THEN** it preserves the value buffer and order, and a borrowed view cannot outlive its buffer
- **AND** a failed owned reshape leaves the original tensor unchanged

### Requirement: CPU kernels preserve shape and dtype contracts
The library SHALL provide contiguous f32/f64 add, mul, rank-one dot and rank-two matmul reference kernels. Elementwise shapes SHALL match exactly and matmul inner dimensions SHALL agree. Native arithmetic SHALL use the original dtype with left-to-right reductions and no fused multiply-add. Empty reductions SHALL return positive zero.

#### Scenario: Asymmetric rectangular product
- **WHEN** a row-major 2x3 tensor is multiplied by a 3x4 tensor
- **THEN** the output is 2x4 with independently verified values and neither operand is transposed

#### Scenario: Unsupported layout or broadcasting
- **WHEN** a view imports noncanonical strides or an elementwise operation receives different shapes
- **THEN** it returns an explicit layout or shape error

### Requirement: Tensor basis evaluation has an explicit numeric policy
The library SHALL apply an explicitly selected EML/BML primitive elementwise using the checked f64 evaluator, returning the original input dtype. For f32 results it SHALL reject overflow and nonzero-to-zero rounding while allowing exact zero and nonzero subnormals. Failure SHALL identify the first invalid flat index and return no partial tensor.

#### Scenario: Narrow a BML result to f32
- **WHEN** BML(-149,1) is requested for f32 tensors
- **THEN** the result is the smallest positive f32 subnormal
- **AND** BML(-150,1) fails with an output-range error at the failing element
