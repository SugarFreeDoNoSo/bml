//! CPU Llama inference with explicit model validation and bounded context.
//!
//! The numeric BML library remains independent of this optional application
//! layer. Quantized resources stay compressed until individual rows are used.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod driver;
pub mod model;
pub mod remote;
pub mod resources;
pub mod tokenizer;

/// Typed model metadata shared by GGUF and autonomous weight containers.
pub trait ModelMetadata {
    /// Read a typed metadata value without converting its representation.
    fn get_metadata(&self, key: &str) -> Option<&bml_gguf::GgufMetadataValue>;
    /// Enumerate all metadata to reject unsupported architecture variants.
    fn metadata(&self) -> &std::collections::HashMap<String, bml_gguf::GgufMetadataValue>;
    /// Declared model architecture.
    fn architecture(&self) -> Option<&str> {
        match self.get_metadata("general.architecture") {
            Some(bml_gguf::GgufMetadataValue::String(s)) => Some(s),
            _ => None,
        }
    }
}
impl ModelMetadata for bml_gguf::GgufReader {
    fn get_metadata(&self, key: &str) -> Option<&bml_gguf::GgufMetadataValue> {
        self.get_metadata(key)
    }
    fn metadata(&self) -> &std::collections::HashMap<String, bml_gguf::GgufMetadataValue> {
        self.metadata()
    }
}
impl ModelMetadata for std::collections::HashMap<String, bml_gguf::GgufMetadataValue> {
    fn get_metadata(&self, key: &str) -> Option<&bml_gguf::GgufMetadataValue> {
        self.get(key)
    }
    fn metadata(&self) -> &Self {
        self
    }
}

/// An invalid model, unsupported profile, resource or execution request.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self(value.to_string())
    }
}

impl From<bml_gguf::GgufError> for Error {
    fn from(value: bml_gguf::GgufError) -> Self {
        Self(value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self(value.to_string())
    }
}

/// Result of model validation, resource loading or CPU execution.
pub type Result<T> = std::result::Result<T, Error>;

/// A borrowed compressed tensor in GGML order (axis zero is contiguous).
#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    /// Canonical GGUF tensor name.
    pub name: String,
    /// Dimensions in GGML order.
    pub dims: Vec<u64>,
    /// Stored element or block encoding.
    pub dtype: bml_gguf::GgufDataType,
    /// Exact stored bytes, borrowed from immutable model storage.
    pub data: &'a [u8],
}
