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
