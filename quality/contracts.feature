Feature: Reliable tensors, scalar compilation and GGUF decoding
  @C52
  Scenario: Execute historical native inference with causal GQA and preserved residuals
    Given a validated Llama fixture with two query heads, one KV head and independently known uniform-attention logits
    When a prefix is evaluated at once or incrementally
    Then logits agree, token order matters and normalization leaves the residual state intact
    And invalid IDs, incompatible caches and exhausted context fail before committing history
    And arithmetic failures preserve the committed prefix and permit a safe retry
  @C51
  Scenario: Emit generated token bytes and stop the historical CLI at EOS
    Given a projection-only fixture that generates two UTF-8 byte pieces followed by EOS
    When bml-cli runs the fixture with a larger token limit
    Then stdout contains the complete character without unknown-token placeholders and generation stops at EOS
  @C50
  Scenario: Tokenize historical inference prompts with the validated GGUF policy
    Given scored SentencePiece BPE metadata with BOS and byte fallback tokens
    When the historical vocabulary encodes prompts or emits token pieces
    Then merges use scores, BOS follows metadata, and case, repeated spaces and Unicode bytes are preserved
    And missing BPE metadata and invalid token IDs are rejected explicitly
  @C49
  Scenario: Apply native Llama activation and final normalization in historical inference
    Given signed closed-form SwiGLU inputs and a separate output head with nondefault RMSNorm epsilon
    When historical cached and uncached inference compute logits
    Then SwiGLU uses sigmoid(gate) and final weighted RMSNorm precedes the output projection exactly once
    And malformed normalization weights or epsilon are rejected before execution
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
    Given independent F32/F16/Q4_0/Q8_0/Q6_K byte fixtures and malformed lengths
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

  @C19
  Scenario: Preserve SentencePiece scores, spaces and bytes
    Given a validated vocabulary with scores, token types and byte fallback
    When text is encoded and token pieces are decoded
    Then token boundaries, whitespace, Unicode bytes and special-token policy are preserved

  @C20
  Scenario: Execute complete dense Llama blocks against independent answers
    Given signed synthetic weights, grouped attention and a declared RoPE configuration
    When a causal model evaluates multiple token positions
    Then residuals, normalization, projections and logits agree with independent oracles

  @C21
  Scenario: Bound session memory and commit only complete tokens
    Given a session with explicit context and numerical memory limits
    When a batch is invalid or a generated native layer sequence fails
    Then the request fails without committing incomplete KV history

  @C22
  Scenario: Validate normalized resources before autonomous inference
    Given compressed tensors with a versioned manifest and SHA-256 identities
    When resources or an ELF appendix are opened
    Then malformed metadata, ranges, plans and payloads fail before model execution

  @C23
  Scenario: Publish model resources without partial or overwritten destinations
    Given an export destination that may already exist or be written concurrently
    When a model resource is published
    Then exactly one complete output can be created and failed exports leave no partial destination

  @C24
  Scenario: Reject ambiguous GGUF names
    Given repeated metadata keys or tensor names in GGUF versions two or three
    When the reader imports the file
    Then the second conflicting entry is rejected with a typed structural error

  @C25
  Scenario: Execute the native plan only with its bound embedded resources
    Given an ELF with a model appendix and a compiled resource identity and layer plan
    When the native entrypoint processes a prompt
    Then it agrees with the checked session or rejects a mismatched hash or plan before evaluation

@C26
Scenario: Preserve borrowed compressed weights during model execution
Given Q4_0 embedding rows and signed Q8_0 output rows
When a model evaluates selected rows
Then compressed storage remains borrowed and decoded results match independent answers
@C27
Scenario: Preserve compressed resources through normalized export
Given a quantized GGUF model
When normalized resources are exported and reopened
Then encoded tensor lengths, dtypes, dimensions and model identity remain intact
@C28
Scenario: Bound retained diagnostic logits before generation
Given a request for complete logits exceeding the diagnostic payload budget
When the command-line entrypoint validates the request
Then it rejects the request before expensive generation

  @C29
  Scenario: Stop generation at EOS in both execution paths
    Given a model whose unique maximum logit selects EOS before the requested token count
    When the CLI and embedded native executable generate tokens with complete logits
    Then each emits one EOS token, no text bytes, and retains the final decode logits

  @C30
  Scenario: Preserve canonical BML tree streams and reject truncating exports
    Given leaf and multiword BML trees with explicit resource budgets
    When fixed-width codes or versioned streams are exported and imported
    Then boundaries, counts, useful bits, depth and padding are checked

  @C31
  Scenario: Parse exact rational targets independently of floating evaluation
    Given decimal strings and finite IEEE values including subnormals
    When exact targets are parsed with bounded integer arithmetic
    Then reduced rationals agree with independent oracles and decimal 0.1 differs from IEEE 0.1

  @C32
  Scenario: Verify exact derivations without floating-point acceptance
    Given a BML tree, an exact target and a replayable derivation
    When the derivation is verified under resource limits
    Then invalid proofs are rejected and exact 2049 remains valid despite direct evaluation overflow

  @C33
  Scenario: Preserve certified arithmetic domains and transactional assignment
    Given certified signed operands including zero and mixed numeric values
    When checked arithmetic or fallible assignment is requested
    Then signs are proved, zero divisors fail and failed assignments preserve both operands

  @C34
  Scenario: Recover conservative exact proofs before operating on raw BML trees
    Given serialized BML trees without retained certificates
    When checked operators or transactional assignment are used
    Then recognized rules are replayed, unjustified domains fail and structural methods retain their contracts

  @C35
  Scenario: Execute bounded symbolic matrix operations without implicit floats
    Given row-major BML matrices with checked shapes and prepared numeric scalars
    When standard or generalized products reduce from left to right
    Then independent results, empty reductions, localized failures and transactional replacement satisfy their contracts

  @C36
  Scenario: Compare exact synthesis strategies with explicit bounded evidence
    Given an exact target and candidate, factorization and cache budgets
    When certified candidates are ranked by expanded operators, depth and code
    Then attempts and failures are reported without claiming global minimality or inferring numeric stability

  @C37
  Scenario: Simplify certified rational expressions before explicitly rounded evaluation
    Given a verified BML derivation with overflowing direct intermediates
    When simplified evaluation rounds the exact rational with ties to even
    Then its proof remains independent of rounding and range/resource failures are explicit

  @C38
  Scenario: Check bounded symbolic properties against independent oracles
    Given exhaustively generated small prefix codes and signed integer operands
    When grammar, streams and checked arithmetic are exercised
    Then invalid codes fail and successful results match independent grammar and integer answers

  @C39
  Scenario: Compare structural BML codes with safely selected SIMD kernels
    Given canonical multipalabra codes and explicit batch budgets
    When scalar or available forced vector backends compare codes
    Then unsigned ordering, tails, first preflight errors and unchanged inputs agree exactly without numerical evaluation claims

  @C40
  Scenario: Execute causal inference in fixed layer stages
    Given independent bounded sessions with fixed nonempty layer ranges
    When staged hidden states process several tokens and interleaved requests
    Then logits equal the local route and invalid positions or numerical failures do not commit partial stages

  @C41
  Scenario: Execute causal layer stages over bounded TCP frames
    Given two dedicated connections with independent causal KV sessions
    When token hidden states traverse both workers in order
    Then logits match local inference and duplicate positions and oversized frames are rejected

  @C42
  Scenario: Launch a bounded causal layer worker as a process
    Given a local validated GGUF and an explicit nonempty layer range
    When the worker admits a dedicated TCP session
    Then token logits equal local inference and invalid limits fail before serving

  @C43
  Scenario: Coordinate an ordered remote inference pipeline
    Given dedicated stages covering every validated model layer exactly once
    When prompt tokens execute or a later stage fails after an earlier commit
    Then successful logits match local inference and partial failure invalidates the whole pipeline

  @C44
  Scenario: Indexed reads require a real result buffer
    Given indexed reads at top level and both supported loop depths
    When evaluated by the bufferless program and fragment interpreters
    Then the result is NaN rather than a fabricated zero
    And the runtime with a populated result buffer returns its stored value

  @C45
  Scenario: Invalid result-buffer addresses cannot alias another cell
    Given a hot loop with two populated result slots
    When an indexed read overflows u32 or a store exceeds its slot
    Then the invalid read returns NaN and every existing result remains unchanged

  @C46
  Scenario: Graph preflight bounds execution and preserves complete context
    Given fragmented bytecode with constants, inputs, nested loops and result slots
    When validated against instruction, step and value budgets
    Then malformed references, stack effects and loop bounds are rejected before execution
    And valid execution preserves context across fragments without partial writes on setup failure

  @C47
  Scenario: Graph RPC validates framing and complete context
    Given a versioned graph request containing inputs, result slots and compiled bytecode
    When encoded, received and executed under explicit resource limits
    Then exact IEEE values are preserved and malformed requests or replies fail
    And oversized graph headers are rejected before reading or allocating their bodies

  @C48
  Scenario: Completed graph results publish into fresh append-only slots
    Given a completed RPC result with a scalar and result buffer
    When append-only storage has insufficient capacity
    Then no positions are reserved and the unchanged result is available for local publication retry
    And successful publication preserves every IEEE bit in a new stable range

  @C53
  Scenario: RPN projections preserve GGUF rows and share the causal attention contract
    Given signed rectangular weights and a causal grouped-query attention fixture
    When projections execute as row-bounded RPN dot products
    Then invalid weight ranges are rejected and outputs match independent dot products
    And complete-prefix and incremental logits match the closed-form causal oracle

  @C54
  Scenario: Legacy graph directories decode within explicit file and collection budgets
    Given independently encoded versioned headers and fragment files
    When loaded with byte, operation, constant and fragment limits
    Then every opcode and IEEE constant bit pattern is preserved
    And truncation, malformed lengths, unknown versions or opcodes and trailing bytes are rejected
    And admission budgets are inclusive and cumulative across all files

  @C55
  Scenario: Historical inference retains encoded weights without changing projection arithmetic
    Given Q4 embeddings and a Q6 head with independent GGML golden values
    When matrices remain mapped and each contiguous row is decoded on demand
    Then projections match the independent answers and fully expanded storage
    And no matrix weights are retained in the decoded normalization pool

  @C56
  Scenario: Compiled causal artifacts execute stored programs with verified source bindings
    Given serialized row graphs and persisted configuration and tokenizer metadata
    When loaded against the SHA-256-identified external GGUF source
    Then causal GQA logits match the independent oracle and incremental execution
    And saved bytecode is executed rather than regenerated from dimensions
    And invalid identities, bindings, stack effects and loop budgets are rejected

  @C57
  Scenario: Compile and run commands generate text from a persisted causal artifact
    Given a byte-token GGUF compiled into a directory and relocated weight file
    When the CLI loads its saved programs and verified metadata
    Then it emits complete UTF-8 token pieces, stops at EOS and respects context capacity

  @C58
  Scenario: The HTTP server executes compiled causal artifacts for JSON and SSE
    Given a persisted byte-token artifact with hash-verified relocated weights
    When real HTTP completion and streaming requests reach the server executable
    Then both return the same complete UTF-8 text without echoing the prompt
    And SSE has exactly one finish event and one DONE marker
    And a deliberately changed stored program changes the HTTP result

  @C59
  Scenario: Signed BML expansions avoid infinite intermediates on bounded inputs
    Given signed half-integer inputs between minus one and one
    When the transformer builds products using only One, Var and Bml nodes
    Then every intermediate is finite with positive logarithm arguments
    And products match the independent fractional oracle
    And negation and positive scaling preserve signed values on the tested grid
    And range failures and invalid scaling domains remain observable

  @C60
  Scenario: Transformer operations expand into the extended BML grammar
    Given signed variables and constant weights with valid operation shapes
    When matmul, RMSNorm, RoPE, softmax and SwiGLU sub-DAGs are constructed
    Then bounded examples match independent mathematical answers with finite real intermediates
    And every RoPE pair and both SwiGLU branches contribute to the outputs
    And invalid shapes, references and epsilon fail before graph mutation

  @C61
  Scenario: Transformer sub-DAGs survive serialization without native arithmetic
    Given the five transformer operations built by BMLTransformer
    When each output is linearized, fragmented, serialized, reloaded and executed
    Then outputs match independent answers using BML arithmetic and stack operations only
    And the bytecode contains no native addition or multiplication fallback

  @C62
  Scenario: External graph constants have explicit bounded execution contexts
    Given an empty-pool graph with declared external constant references
    When validated and executed with a caller-owned constant slice
    Then references and the full value budget are checked inclusively
    And changed constants affect repeated executions without being retained
    And wrong context sizes or mixed constant modes fail before buffer writes

  @C63
  Scenario: Symbolic projections read GGUF rows without embedding their weights
    Given a mapped GGUF and a node-budgeted BML-only row template
    When the serialized template executes with externally supplied decoded row constants
    Then projection outputs match an independent fixture oracle
    And the serialized program contains no weight values or native arithmetic fallback
    And node budgets and size overflow are rejected before row construction

  @C64
  Scenario: A complete bounded causal model executes BML-only activation arithmetic
    Given a mapped Llama fixture with GQA, rotary positions and nonzero signed MLP weights
    When prefill and incremental decode execute budgeted symbolic operation programs
    Then logits match an independent full-model oracle and native inference within tolerance
    And dot products, RMSNorm, RoPE, softmax, SwiGLU, attention scaling and residuals all execute
    And budget or numerical failures preserve committed KV positions and allow retry
    And the profile reports numerical failures instead of silently using native arithmetic

  @C65
  Scenario: Symbolic AOT artifacts execute a complete stored operation plan
    Given a hash-bound mapped source and all BML kernels for a declared context
    When loaded with a zero construction-node budget
    Then prefill and decode match the verified lazy profile without recompilation
    And changing a stored residual program changes the independently predicted logits
    And source, literal, plan, bytecode, context and resource mismatches are rejected
    And failed compilation does not publish a manifest or overwrite existing files

  @C66
  Scenario: The CLI compiles and runs symbolic AOT artifacts
    Given an explicit symbolic compile request and context bound
    When a separate run command loads the artifact with relocated identical weights
    Then UTF-8 token bytes and EOS are emitted using the persisted profile
    And generation respects both the requested and compiled context limits
    And missing compile context or ambiguous artifact profiles are rejected

  @C67
  Scenario: HTTP and SSE execute the persisted symbolic profile within compiled context
    Given a symbolic AOT artifact and relocated hash-identical GGUF weights
    When the server handles completion and streaming requests
    Then both emit complete UTF-8 text using the stored programs
    And full context produces an empty completion while oversized prompts fail explicitly
    And SSE has exactly one terminal event and DONE marker, including errors
    And changed stored bytecode changes HTTP output while ambiguous profiles fail startup

  @C68
  Scenario: Legacy distributed shards restore configuration through bounded parsing
    Given a shard and its sibling distributed header
    When the loader decodes the persisted files
    Then model configuration and all opcode encodings survive roundtrip
    And every truncated prefix is rejected without a panic
    And invalid versions, counts, UTF-8, ranges and trailing bytes are rejected
    And missing headers and oversized files fail before count-driven allocation

  @C69
  Scenario: Multi-output RPN shares the reachable DAG and preserves scalar arithmetic
    Given ordered roots and externally bound output offsets
    When one persisted fragmented program writes all requested outputs
    Then each shared BML node is emitted once with its original operand order
    And outputs match scalar lowering bit for bit with repeated roots and external weights
    And legacy IEEE intermediates and a balanced administrative result are preserved
    And malformed references and overflowing output indices are rejected

  @C70
  Scenario: Symbolic vector templates use linear stored work under explicit budgets
    Given RMSNorm, softmax and RoPE built with the same BML identities
    When their roots share one output-buffer program
    Then outputs match the scalar templates bit for bit
    And RMSNorm and softmax bytecode grows linearly with width
    And node, program and opcode budgets remain enforced at inclusive boundaries

  @C71
  Scenario: Symbolic AOT v2 reads shared outputs and preserves v1 compatibility
    Given genuine scalar-output v1 and buffered-output v2 artifacts for the same source
    When loaded with zero construction budget and executed for prefill and decode
    Then their output bits agree and their persisted profile identifiers remain distinct
    And missing or nonfinite outputs fail without advancing committed KV positions
    And valid modified stored writes change the independently predicted logits

  @C72
  Scenario: Public GGUF compiler entry points produce complete symbolic causal plans
    Given a mapped GQA fixture with rotary positions and nonzero signed MLP weights
    When either GGUF compiler entry point constructs and publishes its bounded AOT plan
    Then prefill and decode match the independent full-model oracle without recompilation
    And every transformer operation uses BML bytecode and externally bound weights
    And explicit context, hardware fragment size and construction budgets are enforced
    And publication rejects a nonempty destination without overwriting its manifest

  @C73
  Scenario: Complex symbolic interpretation declares its branch and evaluation budget
    Given explicit EML or BML basis and a fixed logarithm branch
    When finite complex operands or a pure tree are evaluated
    Then signed branch cuts and nonprincipal branches match independent analytical answers
    And extreme logarithm magnitudes avoid spurious overflow or underflow
    And invalid operands, logarithm zero and exhausted node or depth budgets are rejected

  @C74
  Scenario: Bounded symbolic synthesis reports numerical success or budget exhaustion
    Given a finite target, explicit interpretation, tolerance and search budgets
    When ordered pure trees are enumerated by increasing size
    Then small known targets yield the independently expected syntax and value
    And invalid candidates consume attempts and exhausted searches retain explicit status
    And zero budgets and unsupported request limits cannot bypass the search bounds
