//! Pure BML terms stored as a compact binary prefix code.
//!
//! `0` denotes the terminal one and `1` introduces an ordered binary
//! application. Bits are read from most to least significant.
//!
//! ```
//! use bml_domain::number::Bml;
//! let a = Bml::from_f64(1.5)?;
//! let b = Bml::from_u64(4)?; // binary 100: the expression BML(1, 1) = 2
//! let sum = a.add(&b)?;
//! assert!((sum.to_f64()? - 3.5).abs() < 1e-10);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod certified;
mod checked;
mod convert;
mod encoding;
mod evaluate;
mod exact;
mod matrix;
mod operations;
mod recognize;
mod search;
mod stream;

pub use certified::{CertifiedBml, ExactDerivation, ExactError, ExactStep, ExactSynthesisOptions};
pub use checked::ExactOperand;
pub use convert::{ConversionError, SynthesisOptions};
pub use encoding::{BuildError, EncodingError, DEFAULT_MAX_BITS};
pub use evaluate::SimplifiedEvalError;
pub use exact::{ExactInputError, ExactInputLimits, ExactRational, RationalFloatError};
pub use matrix::{BmlMatrix, MatrixError, MatrixLimits, ReductionInit};
pub use operations::Rules;
pub use search::{
    CandidateOutcome, CandidateRecord, DirectEvaluationMeasurement, ExactSearchOptions,
    ExactSearchReport, ExactSearchResult, ExactSearcher, ExactStrategy, SearchTermination,
    TreeCost,
};
pub use stream::{StreamError, StreamLimits};

/// A closed BML term, represented by the words of its canonical integer.
///
/// Equality and hashing compare structure, not evaluated numerical values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Bml {
    words: Vec<u64>,
}
