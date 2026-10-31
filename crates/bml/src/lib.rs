//! Scalar BML terms and operations, with opt-in tensor and GGUF integrations.
//!
//! A [`Bml`] stores a pure prefix code. Arithmetic methods construct terms;
//! [`Bml::to_f64`] evaluates them under checked finite-real semantics.
//! Float synthesis is bounded and may return an error.

pub use bml_domain::number::{
    Bml, BuildError, ConversionError, EncodingError, Rules, SynthesisOptions, DEFAULT_MAX_BITS,
};
pub use bml_domain::symbolic::EvalError;

/// Native tensor types and kernels.
#[cfg(feature = "tensor")]
pub use bml_domain::tensor;

/// GGUF reader and codecs; enable `gguf-tensor` for the tensor bridge.
#[cfg(feature = "gguf")]
pub use bml_gguf as gguf;
