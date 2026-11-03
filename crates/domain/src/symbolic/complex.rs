//! Finite complex interpretation with an explicit logarithm branch.
//!
//! This is a separate numerical policy from the real evaluator. It rejects
//! log(0), exponential underflow to zero and nonfinite intermediates. Ordinary
//! rounding, component underflow and cancellation are permitted; it does not
//! implement extended-complex infinities or certify an error bound.
use super::{Basis, ClosedTerm};
use std::fmt;

/// Rectangular complex value. Public fields can contain invalid inputs; every
/// evaluation entry point checks finiteness before doing arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex64 {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}
impl Complex64 {
    /// Embed a real value with a positive-zero imaginary component.
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    /// Whether both components are finite.
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

/// A fixed logarithm branch applied at every operator in an evaluation.
///
/// The argument is `atan2(im, re) + 2*pi*turns`. At the negative real axis,
/// positive/negative imaginary zero selects +pi/-pi respectively. Branch zero
/// is principal with this signed-cut convention. This does not track analytic
/// continuation along a path. For BML, the entire complex logarithm is divided
/// by ln(2), including the branch increment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogBranch {
    /// Number of full turns added to the principal argument.
    pub turns: i32,
}

/// Limits on visited tree occurrences and root-inclusive depth.
/// Zero permits no evaluation, including the generator `One`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationBudget {
    /// Maximum number of visited occurrences (shared-looking boxes count twice).
    pub max_nodes: usize,
    /// Maximum root-inclusive depth.
    pub max_depth: usize,
}

/// Explicit failures of the bounded finite-complex evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComplexEvalError {
    /// An operand contains NaN or infinity.
    NonFiniteInput,
    /// The logarithm's operand is zero on both axes.
    LogZero,
    /// An intermediate or final value is outside finite f64 range.
    Overflow,
    /// The exponential magnitude rounded to zero.
    Underflow,
    /// The number of visited tree occurrences exceeds the supplied budget.
    NodeBudget,
    /// The root-inclusive tree depth exceeds the supplied budget.
    DepthBudget,
}
impl fmt::Display for ComplexEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonFiniteInput => "non-finite complex operand",
            Self::LogZero => "complex logarithm of zero",
            Self::Overflow => "complex floating-point overflow",
            Self::Underflow => "complex exponential underflow to zero",
            Self::NodeBudget => "complex evaluation node budget exceeded",
            Self::DepthBudget => "complex evaluation depth budget exceeded",
        })
    }
}
impl std::error::Error for ComplexEvalError {}

impl Basis {
    /// Evaluate `base^x - log_base(y)` using the caller's logarithm branch.
    pub fn apply_complex(
        self,
        x: Complex64,
        y: Complex64,
        branch: LogBranch,
    ) -> Result<Complex64, ComplexEvalError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(ComplexEvalError::NonFiniteInput);
        }
        let large = y.re.abs().max(y.im.abs());
        if large == 0.0 {
            return Err(ComplexEvalError::LogZero);
        }
        // ln(hypot(re, im)) without overflowing hypot or squaring a tiny value.
        let ratio = y.re.abs().min(y.im.abs()) / large;
        let log_radius = large.ln() + 0.5 * (ratio * ratio).ln_1p();
        let log_base = match self {
            Self::Eml => 1.0,
            Self::Bml => std::f64::consts::LN_2,
        };
        let magnitude = match self {
            Self::Eml => x.re.exp(),
            Self::Bml => x.re.exp2(),
        };
        let angle = x.im * log_base;
        if !magnitude.is_finite() || !angle.is_finite() {
            return Err(ComplexEvalError::Overflow);
        }
        if magnitude == 0.0 {
            return Err(ComplexEvalError::Underflow);
        }
        let argument = y.im.atan2(y.re) + std::f64::consts::TAU * f64::from(branch.turns);
        let (sin, cos) = angle.sin_cos();
        let result = Complex64 {
            re: magnitude * cos - log_radius / log_base,
            im: magnitude * sin - argument / log_base,
        };
        if result.is_finite() {
            Ok(result)
        } else {
            Err(ComplexEvalError::Overflow)
        }
    }
}

impl ClosedTerm {
    /// Evaluate iteratively, left child first, under explicit work/depth limits.
    /// The borrowed term is never changed. These limits govern evaluation, not
    /// construction, cloning or destruction of the caller's Box-based AST.
    pub fn evaluate_complex(
        &self,
        basis: Basis,
        branch: LogBranch,
        budget: EvaluationBudget,
    ) -> Result<Complex64, ComplexEvalError> {
        let mut pending = vec![(self, 1usize, false)];
        let mut values = Vec::new();
        let mut visited = 0usize;
        while let Some((term, depth, expanded)) = pending.pop() {
            if expanded {
                let right = values.pop().expect("evaluated right child");
                let left = values.pop().expect("evaluated left child");
                values.push(basis.apply_complex(left, right, branch)?);
                continue;
            }
            if visited == budget.max_nodes {
                return Err(ComplexEvalError::NodeBudget);
            }
            if depth > budget.max_depth {
                return Err(ComplexEvalError::DepthBudget);
            }
            visited += 1;
            match term {
                Self::One => values.push(Complex64::real(1.0)),
                Self::Op(left, right) => {
                    let next = depth.checked_add(1).ok_or(ComplexEvalError::DepthBudget)?;
                    pending.push((term, depth, true));
                    pending.push((right, next, false));
                    pending.push((left, next, false));
                }
            }
        }
        Ok(values.pop().expect("evaluated root"))
    }
}
