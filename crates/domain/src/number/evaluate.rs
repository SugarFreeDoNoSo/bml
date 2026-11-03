//! Explicit certificate-based simplification followed by one binary64 rounding.
use super::{Bml, CertifiedBml, ExactError, ExactSynthesisOptions, RationalFloatError};
use std::fmt;

/// Simplified evaluation cannot replace proof failures with approximate success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimplifiedEvalError {
    /// Recognition or proof replay failed under its configured budgets.
    Proof(ExactError),
    /// Correctly rounded conversion exceeded its numeric/resource limits.
    Rounding(RationalFloatError),
}
impl fmt::Display for SimplifiedEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proof(e) => write!(f, "simplification proof: {e}"),
            Self::Rounding(e) => write!(f, "simplified evaluation: {e}"),
        }
    }
}
impl std::error::Error for SimplifiedEvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Proof(e) => Some(e),
            Self::Rounding(e) => Some(e),
        }
    }
}
impl CertifiedBml {
    /// Replay the proof and round its rational target once, nearest/ties-to-even.
    /// This explicitly avoids direct exp2/log2 intermediates; `term().to_f64()`
    /// retains its direct numerical contract and can still overflow.
    pub fn to_f64_simplified(
        &self,
        options: ExactSynthesisOptions,
    ) -> Result<f64, SimplifiedEvalError> {
        let verified = self
            .derivation()
            .verify(self.term(), self.target(), options)
            .map_err(SimplifiedEvalError::Proof)?;
        verified
            .target()
            .to_f64_rounded(options.rational)
            .map_err(SimplifiedEvalError::Rounding)
    }
}
impl Bml {
    /// Recognize/replay exact identities, then round the proven rational once.
    /// Unsupported real expressions return an explicit proof error; they do not
    /// fall back to approximate sign/domain checks.
    pub fn to_f64_simplified(
        &self,
        options: ExactSynthesisOptions,
    ) -> Result<f64, SimplifiedEvalError> {
        let verified = self
            .certify_exact(options)
            .map_err(SimplifiedEvalError::Proof)?;
        verified
            .target()
            .to_f64_rounded(options.rational)
            .map_err(SimplifiedEvalError::Rounding)
    }
}
