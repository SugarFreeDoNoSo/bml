//! Bounded synthesis of finite floats into pure terms.

use super::{Bml, BuildError, Rules, DEFAULT_MAX_BITS};
use crate::symbolic::EvalError;
use std::{collections::HashMap, fmt};

/// Limits and acceptance criteria for numerical synthesis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthesisOptions {
    /// Maximum bits in every intermediate and the final expression.
    pub max_bits: usize,
    /// Largest denominator considered before using the exact dyadic input.
    pub max_denominator: u32,
    /// Allowed absolute error (added to the relative allowance).
    pub absolute_tolerance: f64,
    /// Allowed relative error multiplied by the magnitude of the input.
    pub relative_tolerance: f64,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_bits: DEFAULT_MAX_BITS,
            max_denominator: 1024,
            absolute_tolerance: 0.0,
            relative_tolerance: 1e-12,
        }
    }
}

/// A float could not be synthesized and evaluated within the supplied limits.
#[derive(Clone, Debug, PartialEq)]
pub enum ConversionError {
    /// NaN and infinities are not finite real inputs.
    NonFiniteInput,
    /// Limits must be positive and tolerances finite and nonnegative.
    InvalidOptions,
    /// Structural construction exceeded its budget.
    Build(BuildError),
    /// The expression encounters a domain or range error during evaluation.
    Evaluation(EvalError),
    /// Evaluation completed but did not meet the requested error allowance.
    Precision {
        /// Original input.
        requested: f64,
        /// Evaluated approximation.
        actual: f64,
        /// Absolute error allowance.
        allowed_error: f64,
    },
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput => f.write_str("BML synthesis requires a finite input"),
            Self::InvalidOptions => f.write_str("invalid BML synthesis limits or tolerances"),
            Self::Build(error) => write!(f, "BML synthesis: {error}"),
            Self::Evaluation(error) => write!(f, "synthesized BML cannot be evaluated: {error}"),
            Self::Precision {
                requested,
                actual,
                allowed_error,
            } => write!(
                f,
                "BML approximation {actual} of {requested} exceeds error allowance {allowed_error}"
            ),
        }
    }
}

impl std::error::Error for ConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Build(error) => Some(error),
            Self::Evaluation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BuildError> for ConversionError {
    fn from(error: BuildError) -> Self {
        Self::Build(error)
    }
}

impl Bml {
    /// Synthesize a finite float, checking the default `1e-12` relative tolerance.
    ///
    /// This is a partial numerical conversion, distinct from parsing an integer
    /// *encoding* with [`Self::from_u64`]. It stores only the resulting bits.
    /// Negative zero is normalized to zero. Large, tiny or poorly conditioned
    /// inputs may fail despite having a mathematical representation.
    pub fn from_f64(value: f64) -> Result<Self, ConversionError> {
        Self::from_f64_with_options(value, SynthesisOptions::default())
    }

    /// Synthesize with explicit size, search and precision limits.
    ///
    /// First try a rational with a small denominator, then, if none fits the
    /// allowance, construct the IEEE input's exact dyadic rational. The result
    /// must also pass checked numerical evaluation and the error allowance.
    /// This algorithm is not an exhaustive search over all equivalent trees.
    pub fn from_f64_with_options(
        value: f64,
        options: SynthesisOptions,
    ) -> Result<Self, ConversionError> {
        if !value.is_finite() {
            return Err(ConversionError::NonFiniteInput);
        }
        if options.max_bits == 0
            || options.max_denominator == 0
            || !options.absolute_tolerance.is_finite()
            || options.absolute_tolerance < 0.0
            || !options.relative_tolerance.is_finite()
            || options.relative_tolerance < 0.0
        {
            return Err(ConversionError::InvalidOptions);
        }
        let allowance = options.absolute_tolerance + options.relative_tolerance * value.abs();
        if !allowance.is_finite() {
            return Err(ConversionError::InvalidOptions);
        }
        let mut synthesis = Synthesis {
            rules: Rules {
                max_bits: options.max_bits,
            },
            integers: HashMap::new(),
        };
        let mut term = synthesis.magnitude(value.abs(), options.max_denominator, allowance)?;
        if value < 0.0 {
            term = synthesis.rules.neg(&term)?;
        }
        let actual = term.to_f64().map_err(ConversionError::Evaluation)?;
        if (actual - value).abs() > allowance {
            return Err(ConversionError::Precision {
                requested: value,
                actual,
                allowed_error: allowance,
            });
        }
        Ok(term)
    }
}

struct Synthesis {
    rules: Rules,
    integers: HashMap<u64, Bml>,
}

impl Synthesis {
    // Construct n using n = 2^k - r. Recursion decreases n, and callers
    // supply at most 53-bit unsigned integers. No literal numerical nodes.
    fn integer(&mut self, n: u64) -> Result<Bml, BuildError> {
        if let Some(term) = self.integers.get(&n) {
            return Ok(term.clone());
        }
        let term = match n {
            0 => self.rules.log2(&Bml::one())?,
            1 => Bml::one(),
            _ if n.is_power_of_two() => {
                let exponent = self.integer(u64::from(n.trailing_zeros()))?;
                self.rules.exp2(&exponent)?
            }
            _ => {
                let k = 64 - n.leading_zeros();
                let left = self.integer(u64::from(k))?;
                let remainder = self.integer((1u64 << k) - n)?;
                let right = self.rules.exp2(&remainder)?;
                left.op_with_limit(&right, self.rules.max_bits)?
            }
        };
        self.integers.insert(n, term.clone());
        Ok(term)
    }

    fn magnitude(
        &mut self,
        value: f64,
        max_denominator: u32,
        allowance: f64,
    ) -> Result<Bml, BuildError> {
        if value == 0.0 {
            return self.integer(0);
        }
        // Restrict this search to exactly representable integer numerators.
        for denominator in 1..=max_denominator {
            let numerator = (value * f64::from(denominator)).round();
            if numerator > (1u64 << 53) as f64 {
                break;
            }
            if numerator > 0.0 && (numerator / f64::from(denominator) - value).abs() <= allowance {
                let top = self.integer(numerator as u64)?;
                if denominator == 1 {
                    return Ok(top);
                }
                let bottom = self.integer(u64::from(denominator))?;
                return self.rules.div(&top, &bottom);
            }
        }
        let bits = value.to_bits();
        let biased_exponent = ((bits >> 52) & 0x7ff) as i32;
        let mut mantissa = bits & ((1u64 << 52) - 1);
        let mut exponent = if biased_exponent == 0 {
            -1074
        } else {
            mantissa |= 1u64 << 52;
            biased_exponent - 1023 - 52
        };
        let trailing = mantissa.trailing_zeros();
        mantissa >>= trailing;
        exponent += trailing as i32;
        let top = self.integer(mantissa)?;
        if exponent == 0 {
            return Ok(top);
        }
        let mut power = self.integer(u64::from(exponent.unsigned_abs()))?;
        if exponent < 0 {
            power = self.rules.neg(&power)?;
        }
        if mantissa != 1 {
            power = self.rules.add(&self.rules.log2(&top)?, &power)?;
        }
        self.rules.exp2(&power)
    }
}
