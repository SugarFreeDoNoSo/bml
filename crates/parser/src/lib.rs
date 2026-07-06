//! # bml-parser
//!
//! Ingesta de archivos GGUF (GPT-Generated Unified Format) mediante
//! mapeo directo a memoria (`memmap2`) sin copias a RAM. Decodifica
//! cabeceras mágicas, metadatos y referencias a tensores.
//!
//! # Zero-Copy
//!
//! Los tensores se referencian desde el disco directo al espacio de
//! memoria de Rust vía mmap. No hay syscalls `read` de los tensores a
//! buffers userspace; solo `mmap`.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod gguf;
pub mod mmap;

pub use gguf::{
    create_gguf_with_metadata, GgufDataType, GgufHeader, GgufMetadataValue, GgufParser,
    GgufTensorInfo, GGUF_MAGIC, GGUF_SUPPORTED_VERSION,
};
pub use mmap::MmapFile;
