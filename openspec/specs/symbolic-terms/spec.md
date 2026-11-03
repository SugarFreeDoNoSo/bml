# symbolic-terms Specification

## Purpose
Define pure closed/open symbolic terms, structural substitution and explicitly selected real or complex interpretations. Bound numerical evaluation and synthesis without equating structural closure, numerical tolerance and exact symbolic identity.

## Requirements

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

### Requirement: Complex evaluation has an explicit branch and resource budget
The library SHALL provide a separate finite-complex EML/BML interpretation with an explicit fixed logarithm branch, a visited-node budget and a root-inclusive depth budget. Its branch SHALL use atan2(im,re) plus a caller-supplied integer number of full turns. It SHALL preserve the existing real evaluator's policy.

#### Scenario: Evaluate across the logarithm cut
- **WHEN** the operator receives a negative real logarithm operand with positive or negative imaginary zero
- **THEN** the principal branch uses +pi or -pi respectively and an explicit nonzero branch adds its declared full turns
- **AND** BML divides both components of the complex logarithm, including the branch increment, by ln(2)

#### Scenario: Exhaust the evaluation budget
- **WHEN** the tree requires one more node or one more depth level than the supplied bound
- **THEN** evaluation returns the corresponding budget error and preserves the borrowed tree

#### Scenario: Reject invalid complex arithmetic
- **WHEN** an operand is nonfinite, the logarithm operand is zero, the exponential overflows or its magnitude rounds to zero
- **THEN** evaluation returns a typed numerical error without a partial value

### Requirement: Symbolic synthesis is a bounded numerical search
The library SHALL enumerate small pure One/Op trees under explicit node and candidate budgets, an explicit real or complex interpretation and a finite nonnegative absolute tolerance. It SHALL declare its maximum tree size, reject larger requests and count invalid numerical candidates as attempts. Results SHALL distinguish tolerance satisfaction from budget exhaustion; neither floating-point agreement nor exhaustion SHALL imply a symbolic equality or unrepresentability proof.

#### Scenario: Find a bounded candidate
- **WHEN** the requested finite target is reached within the selected numerical tolerance and budgets
- **THEN** the result includes the pure tree, evaluated value, error, node count and tolerance-satisfied status under that exact interpretation

#### Scenario: Exhaust a search
- **WHEN** no candidate satisfies the tolerance before a supplied limit
- **THEN** the result identifies the exhausted limit and may return the best candidate without labeling it a successful synthesis
