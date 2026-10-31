Feature: Reliable tensors, scalar compilation and GGUF decoding
  @C01
  Scenario: Reuse a nonadjacent shared node
    Given BML(BML(x,y),x) with x=1 and y=4
    When the DAG is compiled and executed
    Then the result is 1 and each shared value retains its identity
  @C02
  Scenario: Preserve folded constants
    Given a folded BML(1,1)
    When compiled, fragmented, serialized and restored
    Then reference and runtime evaluation return 2
  @C03
  Scenario: Multiply positive operands smaller than one
    Given operands including 0.5 and 2
    When expanded into BML
    Then multiplication agrees with the native oracle in the tested domain
  @C04
  Scenario: Honor alignment and validate ranges
    Given GGUF files with valid or malformed shapes and byte ranges
    When opened
    Then supported valid tensors have exact slices and malformed ranges fail
  @C05
  Scenario: Decode quantized values in GGML order
    Given Q4_0 and Q8_0 blocks with independently known values
    When decoded
    Then low Q4 nibbles precede high nibbles and Q8 values are signed
  @C06
  Scenario: Reject truncated blocks
    Given incomplete Q4_0 or Q8_0 data
    When opened or decoded
    Then an error is returned without zero padding
  @C07
  Scenario: Validate shapes and borrowed storage
    Given scalars, empty shapes, huge dimensions and exact or mismatched buffers
    When tensors and views are constructed, indexed or reshaped
    Then dimensions and values agree, views borrow storage and invalid inputs fail
  @C08
  Scenario: Compute native CPU reference kernels
    Given f32 and f64 tensors including asymmetric rectangular matrices
    When add, mul, dot or matmul is requested
    Then results match independent answers and incompatible shapes fail
  @C09
  Scenario: Apply an explicit checked operator basis
    Given tensors of identical shape and an EML or BML basis
    When the primitive is applied elementwise
    Then dtype and shape are preserved or the first invalid flat index is reported
  @C10
  Scenario: Decode standalone storage formats exactly
    Given independent F32/F16/Q4_0/Q8_0 byte fixtures and malformed lengths
    When decoded without compiler or runtime dependencies
    Then values and block ordering match independent oracles or a typed error is returned
  @C11
  Scenario: Read standalone GGUF with typed failures
    Given valid and malformed GGUF headers, metadata, shapes and payloads
    When the independent reader opens and decodes them
    Then endian, version, structural, missing-tensor and unsupported-dtype errors are distinct
  @C12
  Scenario: Borrow exact GGUF metadata and encoded storage together
    Given a reader with distinct named tensors and aligned padding
    When a tensor view is requested
    Then its descriptor and exact bytes borrow the same reader and unsupported layouts fail
  @C13
  Scenario: Convert GGML storage explicitly into row-major CPU tensors
    Given an asymmetric 2 by 3 matrix stored with GGML dimensions 3 by 2
    When a workspace consumer enables the tensor feature and multiplies it
    Then the owned tensor keeps flat order and the rectangular result matches an independent answer
  @C14
  Scenario: Reject unsupported architectures at the GGUF execution boundary
    Given Gemma4 or unknown architecture metadata without a tokenizer or weights
    When any experimental GGUF compilation or inference entry point is called
    Then it returns an explicit architecture error while metadata inspection remains available
  @C15
  Scenario: Preserve the contiguous GGUF input axis during experimental inference
    Given rectangular F32 embeddings or Q4_0 token blocks with independent known values
    When embeddings and tied output projections are evaluated without transformer layers
    Then token vectors and logits match manual answers without transposing the file layout

  @C16
  Scenario: Encode a pure BML number as an arbitrary-size integer
    Given prefix codes with zero terminals and ordered binary operators
    When integer words are parsed or combined without a stored bit length
    Then complete terms round-trip and malformed codes or excessive constructions fail

  @C17
  Scenario: Expand scalar operations through BML identities
    Given encoded real operands and a bit budget
    When arithmetic and base-two transcendental rules construct new terms
    Then independent numeric answers agree in the documented domains and evaluation failures remain explicit

  @C18
  Scenario: Synthesize finite floats into checked pure terms
    Given finite floats and explicit size, search and error limits
    When numerical conversion constructs an expression without a constant pool
    Then accepted terms satisfy the tolerance and unsupported inputs return typed errors
