## ADDED Requirements

### Requirement: BML 1.0 evidence
The release SHALL preserve the numeric library contracts and provide verified
CPU Llama inference and a standalone executable for the declared target.
Historical synthetic throughput SHALL NOT be presented as real inference TPS.

#### Scenario: Release readiness
- **WHEN** BML is designated 1.0
- **THEN** real-model correctness, standalone execution and quality evidence
  exist for the delivered revision, and unvalidated targets remain explicit.

#### Scenario: Optional application dependencies
- **WHEN** a consumer enables only the numeric facade
- **THEN** inference, networking and compiler dependencies are not introduced.
