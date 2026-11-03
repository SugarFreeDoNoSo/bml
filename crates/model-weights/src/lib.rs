//! Public contracts for the BMLGGUF v1 symbolic-weight container.
//!
//! Defines types, budgets, typed diagnostics and bounded file reading. Opening
//! validates the catalog; each block read verifies its hash and BMLS structure.
//! The transactional writer replays certificates and verifies the complete file
//! before no-clobber publication. The optional `gguf-import` feature provides
//! `convert::from_gguf` and typed GGUF error causes without introducing
//! a reverse dependency from the standalone GGUF reader.
#![deny(unsafe_code)]

#[cfg(feature = "gguf-import")]
pub mod convert;
mod error;
#[cfg(feature = "gguf-import")]
mod gguf_preflight;
mod limits;
mod memory;
mod metadata;
mod metadata_encode;
mod reader;
mod types;
mod writer;

pub use bml_domain::number::{Bml, CertifiedBml, ExactError, ExactSynthesisOptions, StreamError};
pub use error::*;
pub use limits::*;
pub use reader::WeightReader;
pub use types::*;
pub use writer::WeightWriter;

/// Fixed BMLGGUF v1 header length in bytes.
pub const HEADER_BYTES: u64 = 256;
/// Fixed block-index record length in bytes.
pub const BLOCK_RECORD_BYTES: u64 = 80;
/// Container magic, independent of filename extension.
pub const MAGIC: [u8; 8] = *b"BMLGGUF\0";
