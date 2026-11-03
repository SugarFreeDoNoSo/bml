use crate::{Bml, ConversionLimits, Error, UnsupportedKind};
use sha2::{Digest as _, Sha256};
use std::{fmt, path::PathBuf, time::Duration};

/// A tensor index; existence is checked by each consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub u64);
/// A block index; existence is checked by each consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u64);
/// SHA-256 bytes, independent of textual presentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Digest(pub [u8; 32]);
impl Digest {
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }
}
impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Original GGML encoding; output weights are always BML trees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceDtype {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q6K,
}
impl From<SourceDtype> for u32 {
    fn from(value: SourceDtype) -> Self {
        match value {
            SourceDtype::F32 => 0,
            SourceDtype::F16 => 1,
            SourceDtype::Q4_0 => 2,
            SourceDtype::Q8_0 => 8,
            SourceDtype::Q6K => 14,
        }
    }
}
impl TryFrom<u32> for SourceDtype {
    type Error = Error;
    fn try_from(value: u32) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            14 => Ok(Self::Q6K),
            _ => Err(Error::Unsupported {
                kind: UnsupportedKind::SourceDtype,
                value: value.into(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub id: TensorId,
    pub name: String,
    pub ggml_dimensions: Vec<u64>,
    pub source_dtype: SourceDtype,
    pub element_count: u64,
    pub first_block: BlockId,
    pub block_count: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockInfo {
    pub id: BlockId,
    pub tensor_id: TensorId,
    pub first_element: u64,
    pub term_count: u64,
    pub data_offset: u64,
    pub data_bytes: u64,
    pub useful_bits: u64,
    pub digest: Digest,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedBlock {
    pub id: BlockId,
    pub tensor_id: TensorId,
    pub first_element: u64,
    pub terms: Vec<Bml>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightLocation {
    pub tensor_id: TensorId,
    pub tensor_name: String,
    pub flat_index: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileSummary {
    pub version: u16,
    pub basis: u8,
    pub file_bytes: u64,
    pub metadata_bytes: u64,
    pub tensors_bytes: u64,
    pub blocks_bytes: u64,
    pub payload_bytes: u64,
    pub tensor_count: u64,
    pub block_count: u64,
    pub source_sha256: Digest,
    pub catalog_sha256: Digest,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolIdentity {
    pub name: String,
    pub version: String,
    pub revision: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub schema: u32,
    pub profile: String,
    pub source_gguf_version: u32,
    pub converter: ToolIdentity,
    pub codec: ToolIdentity,
    pub limits: ConversionLimits,
    pub gguf: Vec<GgufMetadataEntry>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufMetadataEntry {
    pub key: String,
    pub value: MetadataValue,
}
/// IEEE metadata is stored as bits, preserving signed zero and NaN payloads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Bool(bool),
    String(String),
    F32Bits(u32),
    F64Bits(u64),
    Array(MetadataArray),
}
/// Homogeneous arrays preserve their element type even when empty; no nesting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    String(Vec<String>),
    F32Bits(Vec<u32>),
    F64Bits(Vec<u64>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationReport {
    pub blocks_checked: u64,
    pub terms_checked: u64,
    pub bytes_checked: u64,
    pub all_payload_blocks_checked: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteReport {
    pub path: PathBuf,
    pub summary: FileSummary,
    pub output_sha256: Digest,
    pub terms_written: u64,
    pub peak_temporary_bytes: u64,
}
/// Accounted memory is not an operating-system RSS measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionReport {
    pub write: WriteReport,
    pub source_sha256: Digest,
    pub source_bytes: u64,
    pub weights_synthesized: u64,
    pub weights_reused: u64,
    pub synthesis_steps: u64,
    pub tree_bytes: u64,
    pub tree_bits: u64,
    pub peak_resident_bytes: u64,
    pub peak_cache_bytes: u64,
    pub read_decode_time: Duration,
    pub synthesis_certification_time: Duration,
    pub write_verify_time: Duration,
}
