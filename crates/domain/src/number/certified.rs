//! Replayable exact derivations. Numerical evaluation is never a proof step.
//!
//! ```
//! use bml_domain::number::{Bml, ExactSynthesisOptions};
//! let value = Bml::from_decimal_str_exact("12.375")?;
//! assert_eq!(value.target().numerator_words_le(), &[99]);
//! assert_eq!(value.target().denominator_words_le(), &[8]);
//! value.derivation().verify(value.term(), value.target(), ExactSynthesisOptions::default())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
use super::{
    exact, Bml, BuildError, ExactInputError, ExactInputLimits, ExactRational, Rules,
    DEFAULT_MAX_BITS,
};
use std::{collections::BTreeMap, fmt};

/// Budgets for exact synthesis and certificate replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactSynthesisOptions {
    /// Input and intermediate rational budgets.
    pub rational: ExactInputLimits,
    /// Maximum expanded bits in each intermediate BML tree.
    pub max_bits: usize,
    /// Maximum derivation steps (also bounds the synthesis memo table).
    pub max_steps: usize,
    /// Sum of expanded tree bits retained during certificate replay.
    pub max_total_bits: usize,
    /// Maximum recursive integer decomposition depth during synthesis.
    pub max_recursion: usize,
}
impl Default for ExactSynthesisOptions {
    fn default() -> Self {
        Self {
            rational: ExactInputLimits::default(),
            max_bits: DEFAULT_MAX_BITS,
            max_steps: 4096,
            max_total_bits: 16 * DEFAULT_MAX_BITS,
            max_recursion: 128,
        }
    }
}

/// One exact algebraic rule, referring only to earlier steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExactStep {
    /// Generator one.
    One,
    /// Canonical `B(1,B(B(1,1),1)) = 0`.
    Zero,
    /// `B(x,1) = 2^x`, for an integer rational exponent.
    Exp2(usize),
    /// `B(x,B(y,1)) = 2^x-y`, without materializing `2^y` as a rational.
    /// This identity holds for every real y, since `2^y > 0`.
    PowerDifference(usize, usize),
    /// Real addition using the proven BML expansion.
    Add(usize, usize),
    /// Real subtraction using the proven BML expansion.
    Sub(usize, usize),
    /// Real negation using the proven BML expansion.
    Neg(usize),
    /// Multiplication using the positive-operand rule; signs must be explicit.
    PositiveMul(usize, usize),
    /// Division using the positive-operand rule; signs must be explicit.
    PositiveDiv(usize, usize),
}

/// Untrusted proof data, replayed against a claimed exact target and term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactDerivation {
    /// Ordered proof program; forward references are invalid.
    pub steps: Vec<ExactStep>,
    /// Index of the result in the proof program.
    pub root: usize,
}

/// Construction, mathematical domain and proof errors are separate from f64 evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExactError {
    /// Exact rational parsing or arithmetic failed.
    Rational(ExactInputError),
    /// Expanded tree construction exceeded its budget.
    Build(BuildError),
    /// A proof or search resource budget was exhausted.
    Limit,
    /// A derivation references unavailable steps or disagrees with its claim.
    InvalidProof,
    /// A rule's mathematical domain was not justified.
    DomainNotJustified,
}
impl fmt::Display for ExactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rational(e) => write!(f, "exact rational: {e}"),
            Self::Build(e) => write!(f, "exact construction: {e}"),
            Self::Limit => f.write_str("exact derivation resource limit exceeded"),
            Self::InvalidProof => f.write_str("invalid exact derivation"),
            Self::DomainNotJustified => f.write_str("exact rule domain not justified"),
        }
    }
}
impl std::error::Error for ExactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rational(e) => Some(e),
            Self::Build(e) => Some(e),
            _ => None,
        }
    }
}
impl From<ExactInputError> for ExactError {
    fn from(e: ExactInputError) -> Self {
        Self::Rational(e)
    }
}
impl From<BuildError> for ExactError {
    fn from(e: BuildError) -> Self {
        Self::Build(e)
    }
}

/// A term accompanied by its checked mathematical target and derivation.
///
/// Its `to_f64` may still overflow or lose accuracy. A certificate establishes
/// a real identity, not a guarantee about direct floating-point execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertifiedBml {
    term: Bml,
    target: ExactRational,
    derivation: ExactDerivation,
}
impl CertifiedBml {
    /// Fully expanded term, with no external constant pool.
    pub fn term(&self) -> &Bml {
        &self.term
    }
    /// Exact reduced rational value proved by the derivation.
    pub fn target(&self) -> &ExactRational {
        &self.target
    }
    /// Replayable proof; cloning and modifying it does not modify this certificate.
    pub fn derivation(&self) -> &ExactDerivation {
        &self.derivation
    }
    /// Consume the proof wrapper to retain only the structural term.
    pub fn into_term(self) -> Bml {
        self.term
    }
}

impl ExactDerivation {
    /// Replay all steps, check the claimed target and exact tree structure.
    /// No tolerance, float evaluation, or caller-provided sign assertion is used.
    pub fn verify(
        &self,
        term: &Bml,
        target: &ExactRational,
        options: ExactSynthesisOptions,
    ) -> Result<CertifiedBml, ExactError> {
        let (actual_term, actual_target) = self.replay(options)?;
        if &actual_term != term || &actual_target != target {
            return Err(ExactError::InvalidProof);
        }
        Ok(CertifiedBml {
            term: actual_term,
            target: actual_target,
            derivation: self.clone(),
        })
    }

    /// Materialize a derivation and its exact rational value under explicit budgets.
    pub fn certify(&self, options: ExactSynthesisOptions) -> Result<CertifiedBml, ExactError> {
        let (term, target) = self.replay(options)?;
        Ok(CertifiedBml {
            term,
            target,
            derivation: self.clone(),
        })
    }

    fn replay(&self, options: ExactSynthesisOptions) -> Result<(Bml, ExactRational), ExactError> {
        if self.steps.len() > options.max_steps {
            return Err(ExactError::Limit);
        }
        if self.root >= self.steps.len() {
            return Err(ExactError::InvalidProof);
        }
        let mut results: Vec<(Bml, ExactRational)> = Vec::new();
        results
            .try_reserve_exact(self.steps.len())
            .map_err(|_| ExactError::Limit)?;
        let mut total = 0usize;
        for step in &self.steps {
            let rules = Rules {
                max_bits: options.max_bits.min(options.max_total_bits - total),
            };
            let get = |i: usize| results.get(i).ok_or(ExactError::InvalidProof);
            let pair = match *step {
                ExactStep::One => (Bml::one(), ExactRational::from_i64(1)),
                ExactStep::Zero => {
                    let one = Bml::one();
                    let two = one.op_with_limit(&one, rules.max_bits)?;
                    let four = two.op_with_limit(&one, rules.max_bits)?;
                    (
                        one.op_with_limit(&four, rules.max_bits)?,
                        ExactRational::from_i64(0),
                    )
                }
                ExactStep::Exp2(a) => {
                    let (term, value) = get(a)?;
                    let value = value.checked_exp2(options.rational)?;
                    (rules.exp2(term)?, value)
                }
                ExactStep::PowerDifference(a, b) => {
                    let (left, x) = get(a)?;
                    let (right, y) = get(b)?;
                    let value = x
                        .checked_exp2(options.rational)?
                        .checked_sub(y, options.rational)?;
                    (
                        left.op_with_limit(&rules.exp2(right)?, rules.max_bits)?,
                        value,
                    )
                }
                ExactStep::Neg(a) => {
                    let (term, value) = get(a)?;
                    (rules.neg(term)?, value.negated())
                }
                ExactStep::Add(a, b)
                | ExactStep::Sub(a, b)
                | ExactStep::PositiveMul(a, b)
                | ExactStep::PositiveDiv(a, b) => {
                    let (left, x) = get(a)?;
                    let (right, y) = get(b)?;
                    match *step {
                        ExactStep::Add(..) => {
                            (rules.add(left, right)?, x.checked_add(y, options.rational)?)
                        }
                        ExactStep::Sub(..) => {
                            (rules.sub(left, right)?, x.checked_sub(y, options.rational)?)
                        }
                        _ => {
                            if x.is_negative() || x.is_zero() || y.is_negative() || y.is_zero() {
                                return Err(ExactError::DomainNotJustified);
                            }
                            if matches!(step, ExactStep::PositiveMul(..)) {
                                (rules.mul(left, right)?, x.checked_mul(y, options.rational)?)
                            } else {
                                (rules.div(left, right)?, x.checked_div(y, options.rational)?)
                            }
                        }
                    }
                }
            };
            if pair.0.bit_len() > options.max_bits
                || exact::bit_len(pair.1.numerator_words_le()) > options.rational.max_bits
                || exact::bit_len(pair.1.denominator_words_le()) > options.rational.max_bits
            {
                return Err(ExactError::Limit);
            }
            total = total
                .checked_add(pair.0.bit_len())
                .ok_or(ExactError::Limit)?;
            if total > options.max_total_bits {
                return Err(ExactError::Limit);
            }
            results.push(pair);
        }
        Ok(results.swap_remove(self.root))
    }
}

struct Synthesizer {
    options: ExactSynthesisOptions,
    steps: Vec<ExactStep>,
    integers: BTreeMap<Vec<u64>, usize>,
}
impl Synthesizer {
    fn push(&mut self, step: ExactStep) -> Result<usize, ExactError> {
        if self.steps.len() >= self.options.max_steps {
            return Err(ExactError::Limit);
        }
        self.steps.try_reserve(1).map_err(|_| ExactError::Limit)?;
        self.steps.push(step);
        Ok(self.steps.len() - 1)
    }
    fn integer(&mut self, words: &[u64], depth: usize) -> Result<usize, ExactError> {
        if depth > self.options.max_recursion
            || exact::bit_len(words) > self.options.rational.max_bits
        {
            return Err(ExactError::Limit);
        }
        if let Some(index) = self.integers.get(words) {
            return Ok(*index);
        }
        let step = match words {
            [] => ExactStep::Zero,
            [1] => ExactStep::One,
            _ => {
                let bits = exact::bit_len(words);
                let ones: u32 = words.iter().map(|word| word.count_ones()).sum();
                if ones == 1 {
                    let exponent = bits - 1;
                    let exponent_words = if exponent == 0 {
                        vec![]
                    } else {
                        vec![u64::try_from(exponent).map_err(|_| ExactError::Limit)?]
                    };
                    ExactStep::Exp2(self.integer(&exponent_words, depth + 1)?)
                } else {
                    let mut remainder =
                        exact::shift_left(&[1], bits, self.options.rational.max_bits)?;
                    exact::subtract(&mut remainder, words);
                    let exponent = self.integer(
                        &[u64::try_from(bits).map_err(|_| ExactError::Limit)?],
                        depth + 1,
                    )?;
                    let remainder = self.integer(&remainder, depth + 1)?;
                    ExactStep::PowerDifference(exponent, remainder)
                }
            }
        };
        let index = self.push(step)?;
        self.integers.insert(words.to_vec(), index);
        Ok(index)
    }
}

impl Bml {
    /// Synthesize a certified exact decimal target with default budgets.
    /// Unlike `from_f64`, success never depends on a numerical tolerance.
    pub fn from_decimal_str_exact(text: &str) -> Result<CertifiedBml, ExactError> {
        Self::from_decimal_str_exact_with_options(text, ExactSynthesisOptions::default())
    }
    /// Exact decimal synthesis with explicit input, proof and expanded-tree budgets.
    pub fn from_decimal_str_exact_with_options(
        text: &str,
        options: ExactSynthesisOptions,
    ) -> Result<CertifiedBml, ExactError> {
        let target = ExactRational::from_decimal_str(text, options.rational)?;
        Self::from_rational_exact(&target, options)
    }
    /// Synthesize the exact stored IEEE value, not its shortest decimal spelling.
    pub fn from_f64_exact(value: f64) -> Result<CertifiedBml, ExactError> {
        Self::from_f64_exact_with_options(value, ExactSynthesisOptions::default())
    }
    /// Exact IEEE synthesis with explicit budgets; even a valid subnormal may fail.
    pub fn from_f64_exact_with_options(
        value: f64,
        options: ExactSynthesisOptions,
    ) -> Result<CertifiedBml, ExactError> {
        let target = ExactRational::from_f64(value, options.rational)?;
        Self::from_rational_exact(&target, options)
    }
    /// Construct a certified rational using memoized power/near-power decomposition.
    /// This is a bounded construction, not a claim of minimum expression size.
    pub fn from_rational_exact(
        target: &ExactRational,
        options: ExactSynthesisOptions,
    ) -> Result<CertifiedBml, ExactError> {
        let mut synth = Synthesizer {
            options,
            steps: Vec::new(),
            integers: BTreeMap::new(),
        };
        let mut root = synth.integer(target.numerator_words_le(), 0)?;
        if target.denominator_words_le() != [1] {
            let bottom = synth.integer(target.denominator_words_le(), 0)?;
            root = synth.push(ExactStep::PositiveDiv(root, bottom))?;
        }
        if target.is_negative() {
            root = synth.push(ExactStep::Neg(root))?;
        }
        let proof = ExactDerivation {
            steps: synth.steps,
            root,
        };
        let result = proof.certify(options)?;
        if &result.target != target {
            return Err(ExactError::InvalidProof);
        }
        Ok(result)
    }
}
