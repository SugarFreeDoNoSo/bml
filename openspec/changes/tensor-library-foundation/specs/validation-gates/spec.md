## ADDED Requirements

### Requirement: Correctness tests have independent expectations
Tests SHALL compare observable results against independent expected values and explicit numerical tolerances. Missing optional fixtures SHALL be visible rather than silently counted as successful coverage.

#### Scenario: Finite result contract
- **WHEN** a finite arithmetic reference is compared with a compiled result
- **THEN** an unexpected NaN or infinity fails the test before applying tolerances

### Requirement: Performance comparisons execute equivalent work
Benchmarks SHALL validate results before timing, retain required execution context and distinguish measured operation throughput from extrapolation.

#### Scenario: Compare inference throughput
- **WHEN** tokens per second are compared with another inference engine
- **THEN** both execute matching model workloads and quality checks rather than converting scalar operation counts into tokens

### Requirement: Instrumentation claims match instrumentation
Loom claims SHALL use instrumented synchronization. Instruction-cache claims SHALL measure machine code or instruction counters separately from bytecode data. Allocation claims SHALL observe allocations in the stated execution region.

#### Scenario: Unsupported tracing environment
- **WHEN** an instrumented job lacks ptrace capability
- **THEN** it reports an explicit unavailable capability and ordinary deterministic correctness tests remain separately runnable
