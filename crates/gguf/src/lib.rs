//! Standalone GGUF reader and F32/F16/Q4_0/Q8_0 codecs.
//!
//! The reader owns a read-only mmap; borrowed bytes cannot outlive it. The file
//! must remain unchanged and untruncated while mapped. Mapping avoids a full
//! userspace copy, not physical memory usage. Decoding allocates explicitly.
//! No compiler, runtime, tokenizer or model dependency is required.
//! The optional `tensor` feature adds explicit conversion to the CPU tensor
//! types currently in `bml-domain`, without changing GGML element order.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod codecs;
mod error;
pub mod gguf;
pub mod mmap;
mod view;

pub use codecs::decode_f32;
pub use error::GgufError;
pub use gguf::{
    create_gguf_with_metadata, GgufDataType, GgufHeader, GgufMetadataValue, GgufReader,
    GgufTensorInfo, GGUF_MAGIC, GGUF_SUPPORTED_VERSION,
};
pub use mmap::MmapFile;
pub use view::GgufTensorView;
