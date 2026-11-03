//! Deterministic, budgeted enumeration of small pure symbolic trees.
//!
//! A candidate meets a numerical tolerance under one explicit interpretation;
//! it is not a proof of symbolic equality, exact synthesis, or representability.
//! Terms are searched in increasing node count, then prefix grammar order
//! (`One` before `Op`). Invalid numerical candidates consume an attempt too.
//!
//! ```
//! use bml_domain::symbolic::{Basis, complex::Complex64, synthesis::{
//!     synthesize, Interpretation, StopReason, SynthesisBudget,
//! }};
//! let result = synthesize(Complex64::real(2.0), Interpretation::Real(Basis::Bml),
//!     0.0, SynthesisBudget { max_nodes: 3, max_candidates: 2 })?;
//! assert_eq!(result.stop, StopReason::ToleranceMet);
//! assert_eq!(result.best.unwrap().nodes, 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
use super::{
    complex::{Complex64, EvaluationBudget, LogBranch},
    Basis, ClosedTerm,
};
use std::fmt;

/// Hard size bound for this small-formula, Box-based enumerator. It bounds
/// construction/destruction depth as well as the scratch space per attempt.
pub const MAX_TERM_NODES: usize = 255;

/// Numerical policy used to evaluate every complete candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpretation {
    /// Existing strict finite-real interpretation; the target must be real.
    Real(Basis),
    /// Finite-complex interpretation with one fixed logarithm branch.
    Complex {
        /// Natural or binary basis.
        basis: Basis,
        /// Explicit branch; no implicit principal-branch fallback.
        branch: LogBranch,
    },
}

/// Independent bounds on candidate size and total evaluated candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynthesisBudget {
    /// Maximum nodes in one tree, at most [`MAX_TERM_NODES`].
    pub max_nodes: usize,
    /// Maximum complete candidates attempted, including numerical failures.
    pub max_candidates: usize,
}

/// Best finite candidate found under the selected numerical policy.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// Pure tree containing only `One` and ordered `Op` nodes.
    pub term: ClosedTerm,
    /// Evaluated rectangular value under the requested policy.
    pub value: Complex64,
    /// Euclidean absolute error; may be infinite for very distant finite values.
    pub error: f64,
    /// Tree occurrence count, including generators and operators.
    pub nodes: usize,
}

/// Why enumeration stopped. Exhaustion is not a claim of unrepresentability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The reported best candidate meets the supplied absolute tolerance.
    ToleranceMet,
    /// No more candidates may be attempted under the supplied work budget.
    CandidateBudget,
    /// All grammar trees up to the supplied node bound were attempted.
    NodeBudget,
}

/// Result of a bounded search, including unsuccessful searches.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// Best finite candidate, even if its error exceeds the requested tolerance.
    /// None means no finite candidate was evaluated.
    pub best: Option<Candidate>,
    /// Number of complete candidates attempted, including invalid evaluations.
    pub attempted: usize,
    /// Explicit termination status; inspect this before using the candidate.
    pub stop: StopReason,
}

/// Invalid search requests are rejected before enumerating any tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthesisError {
    /// Target is nonfinite, or has a nonzero imaginary part in real mode.
    Target,
    /// Absolute tolerance is negative or nonfinite.
    Tolerance,
    /// The node budget exceeds the small-formula implementation's hard limit.
    NodeLimit,
}
impl fmt::Display for SynthesisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Target => "invalid symbolic synthesis target",
            Self::Tolerance => "invalid symbolic synthesis tolerance",
            Self::NodeLimit => "symbolic synthesis node limit exceeds 255",
        })
    }
}
impl std::error::Error for SynthesisError {}

/// Search bounded pure trees. Zero budgets return an exhausted result; they do
/// not evaluate even `One`. Cost per attempt and live scratch are bounded by
/// [`MAX_TERM_NODES`]; previous candidate trees are discarded, except the best.
pub fn synthesize(
    target: Complex64,
    interpretation: Interpretation,
    tolerance: f64,
    budget: SynthesisBudget,
) -> Result<SearchResult, SynthesisError> {
    if !target.is_finite() || matches!(interpretation, Interpretation::Real(_)) && target.im != 0.0
    {
        return Err(SynthesisError::Target);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(SynthesisError::Tolerance);
    }
    if budget.max_nodes > MAX_TERM_NODES {
        return Err(SynthesisError::NodeLimit);
    }
    let mut result = SearchResult {
        best: None,
        attempted: 0,
        stop: StopReason::NodeBudget,
    };
    for size in (1..=budget.max_nodes).step_by(2) {
        // Depth-first enumeration of valid binary prefix codes of exactly size.
        // Feasibility checks avoid exploring prefixes that cannot close there.
        let mut pending = vec![(Vec::<bool>::new(), 1usize)];
        while let Some((prefix, slots)) = pending.pop() {
            if result.attempted == budget.max_candidates {
                result.stop = StopReason::CandidateBudget;
                return Ok(result);
            }
            if slots == 0 {
                let mut stack = Vec::new();
                for &op in prefix.iter().rev() {
                    let term = if op {
                        let left = stack.pop().expect("complete prefix left child");
                        let right = stack.pop().expect("complete prefix right child");
                        ClosedTerm::op(left, right)
                    } else {
                        ClosedTerm::One
                    };
                    stack.push(term);
                }
                let term = stack.pop().expect("nonempty prefix");
                result.attempted += 1;
                let value = match interpretation {
                    Interpretation::Real(basis) => {
                        term.evaluate_real(basis).ok().map(Complex64::real)
                    }
                    Interpretation::Complex { basis, branch } => term
                        .evaluate_complex(
                            basis,
                            branch,
                            EvaluationBudget {
                                max_nodes: size,
                                max_depth: size,
                            },
                        )
                        .ok(),
                };
                if let Some(value) = value {
                    let error = (value.re - target.re).hypot(value.im - target.im);
                    if result.best.as_ref().is_none_or(|best| error < best.error) {
                        result.best = Some(Candidate {
                            term,
                            value,
                            error,
                            nodes: size,
                        });
                    }
                    if error <= tolerance {
                        result.stop = StopReason::ToleranceMet;
                        return Ok(result);
                    }
                }
                continue;
            }
            let remaining = size - prefix.len() - 1;
            // Push Op first so One is visited first by the LIFO traversal.
            if slots < remaining {
                let mut next = prefix.clone();
                next.push(true);
                pending.push((next, slots + 1));
            }
            if (slots == 1 && remaining == 0) || (slots > 1 && slots - 1 <= remaining) {
                let mut next = prefix;
                next.push(false);
                pending.push((next, slots - 1));
            }
        }
    }
    Ok(result)
}
