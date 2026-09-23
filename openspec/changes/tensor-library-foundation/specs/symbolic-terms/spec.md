## ADDED Requirements

### Requirement: Closed and open terms are distinct
The library SHALL represent closed terms using only One and ordered binary Op nodes. Open terms SHALL additionally permit named variables and embedded closed terms.

#### Scenario: Inject and close a term
- **WHEN** a closed term is embedded into an open term and collapsed
- **THEN** the exact original structure is returned without numerical evaluation

### Requirement: Binding is partial structural substitution
Binding SHALL substitute only names present in the closed-term environment, preserve operand order and leave the original term unchanged.

#### Scenario: Bind an incomplete environment
- **WHEN** Op(x,y) is bound with an environment containing only x
- **THEN** y remains free and collapse reports an unbound variable

#### Scenario: Repeat a closed substitution
- **WHEN** the same environment is bound twice
- **THEN** the result equals binding it once

### Requirement: Interpretation is explicit and checked
The finite-real evaluator SHALL take an explicit EML or BML basis and return a typed error for non-finite input, invalid log domain, overflow or exponential underflow to zero.

#### Scenario: Evaluate one shared syntax under two bases
- **WHEN** Op(One,One) is evaluated under EML and BML
- **THEN** the results approximate e and equal 2 respectively

#### Scenario: Closed term has invalid numerical domain
- **WHEN** a closed term has a non-positive logarithmic argument
- **THEN** structural closure succeeds and finite-real evaluation reports a domain error
