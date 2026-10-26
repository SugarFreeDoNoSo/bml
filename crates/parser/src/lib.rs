//! # bml-parser
//!
//! Adaptador de compatibilidad para `bml-gguf`. Conserva los imports
//! anteriores y `GgufParser::open -> std::io::Result`. Para código nuevo,
//! `bml_gguf::GgufReader` expone errores tipados y codecs independientes.
//!
//! # Zero-Copy
//!
//! Los tensores se referencian desde el disco directo al espacio de
//! memoria de Rust vía mmap, sin copiar el archivo completo a un buffer.
//! Las páginas sí usan RAM; el archivo debe permanecer sin modificaciones.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod gguf;
pub mod mmap;

pub use gguf::{
    create_gguf_with_metadata, GgufDataType, GgufHeader, GgufMetadataValue, GgufParser,
    GgufTensorInfo, GGUF_MAGIC, GGUF_SUPPORTED_VERSION,
};
pub use mmap::MmapFile;
