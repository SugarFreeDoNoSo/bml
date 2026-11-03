//! Scalar BML terms and operations, with opt-in tensor and GGUF integrations.
//!
//! A [`Bml`] stores a pure prefix code. Arithmetic methods construct terms;
//! [`Bml::to_f64`] evaluates them under checked finite-real semantics.
//! Float synthesis is bounded and may return an error.

pub use bml_domain::number::{
    Bml, BmlMatrix, BuildError, CandidateOutcome, CandidateRecord, CertifiedBml, ConversionError,
    DirectEvaluationMeasurement, EncodingError, ExactDerivation, ExactError, ExactInputError,
    ExactInputLimits, ExactOperand, ExactRational, ExactSearchOptions, ExactSearchReport,
    ExactSearchResult, ExactSearcher, ExactStep, ExactStrategy, ExactSynthesisOptions, MatrixError,
    MatrixLimits, RationalFloatError, ReductionInit, Rules, SearchTermination, SimplifiedEvalError,
    StreamError, StreamLimits, SynthesisOptions, TreeCost, DEFAULT_MAX_BITS,
};
pub use bml_domain::symbolic::EvalError;

/// Native tensor types and kernels.
#[cfg(feature = "tensor")]
pub use bml_domain::tensor;

/// Opt-in experimental native tensor kernels with explicit scalar fallback.
#[cfg(feature = "tensor-simd")]
pub use bml_kernels::tensor as tensor_simd;

/// GGUF reader and codecs; enable `gguf-tensor` for the tensor bridge.
#[cfg(feature = "gguf")]
pub use bml_gguf as gguf;

/// Experimental structural code comparison kernels; no numerical SIMD claim.
#[cfg(feature = "code-simd")]
pub use bml_kernels as code_simd;
