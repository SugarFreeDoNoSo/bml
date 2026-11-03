use crate::{BlockId, Digest, ExactError, StreamError, TensorId, WeightLocation};
use std::{fmt, path::PathBuf};

/// Stable operation and failure categories. More variants may be added later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IoOperation {
    Open,
    Read,
    Seek,
    CreateTemporary,
    Write,
    Flush,
    Publish,
    Cleanup,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedKind {
    Version,
    Endian,
    Basis,
    Flags,
    SourceDtype,
    GgufVersion,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatKind {
    Magic,
    HeaderLength,
    Reserved,
    FileLength,
    SectionLayout,
    TensorShape,
    DuplicateTensor,
    BlockCoverage,
    BlockHeader,
    ArithmeticOverflow,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegrityScope {
    Header,
    Catalog,
    Block(BlockId),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Resource {
    MaxFileBytes,
    MaxMetadataBytes,
    MaxMetadataEntries,
    MaxMetadataArrayElements,
    MaxNameBytes,
    MaxStringBytes,
    MaxTensors,
    MaxBlocks,
    MaxIndexBytes,
    MaxBlockTerms,
    MaxBlockBytes,
    MaxWeightBits,
    MaxWeightDepth,
    MaxTotalTerms,
    MaxSynthesisStepsPerWeight,
    MaxSynthesisStepsTotal,
    MaxOutputBytes,
    MaxResidentBytes,
    MaxCacheBytes,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndexKind {
    Tensor,
    Block,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WriterOperation {
    BeginTensor,
    WriteBlock,
    EndTensor,
    Finish,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WriterState {
    Ready,
    TensorOpen,
    Aborted,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetadataReason {
    InvalidJson,
    DuplicateKey,
    UnknownField,
    MissingField,
    InvalidType,
    InvalidValue,
    InvalidUtf8,
    UnsupportedProfile,
    NestedArray,
}

/// Typed diagnostic with original causes and an exact weight location when known.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Io {
        operation: IoOperation,
        path: PathBuf,
        source: std::io::Error,
    },
    Unsupported {
        kind: UnsupportedKind,
        value: u64,
    },
    Format {
        kind: FormatKind,
        offset: Option<u64>,
        tensor: Option<TensorId>,
        block: Option<BlockId>,
    },
    Integrity {
        scope: IntegrityScope,
        expected: Digest,
        actual: Digest,
    },
    Limit {
        resource: Resource,
        limit: u64,
        required: Option<u64>,
        location: Option<WeightLocation>,
    },
    Index {
        kind: IndexKind,
        index: u64,
        count: u64,
    },
    State {
        operation: WriterOperation,
        state: WriterState,
    },
    DestinationExists {
        path: PathBuf,
    },
    SourceDestinationConflict {
        source: PathBuf,
        destination: PathBuf,
    },
    Metadata {
        key: Option<String>,
        reason: MetadataReason,
    },
    NonFiniteWeight {
        location: WeightLocation,
        decoded_f32_bits: u32,
    },
    Synthesis {
        location: WeightLocation,
        source: ExactError,
    },
    Stream {
        block: BlockId,
        source: StreamError,
    },
    #[cfg(feature = "gguf-import")]
    Gguf {
        tensor: Option<String>,
        source: bml_gguf::GgufError,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(f, "{operation:?} {}: {source}", path.display()),
            Self::Unsupported { kind, value } => write!(f, "unsupported {kind:?}: {value}"),
            Self::Format {
                kind,
                offset,
                tensor,
                block,
            } => write!(
                f,
                "invalid {kind:?} at byte {offset:?}, tensor {tensor:?}, block {block:?}"
            ),
            Self::Integrity {
                scope,
                expected,
                actual,
            } => write!(
                f,
                "integrity failure for {scope:?}: expected {expected}, got {actual}"
            ),
            Self::Limit {
                resource,
                limit,
                required,
                location,
            } => write!(
                f,
                "{resource:?} limit {limit}, required {required:?}, weight {location:?}"
            ),
            Self::Index { kind, index, count } => {
                write!(f, "{kind:?} index {index} outside count {count}")
            }
            Self::State { operation, state } => {
                write!(f, "cannot {operation:?} while writer is {state:?}")
            }
            Self::DestinationExists { path } => {
                write!(f, "destination already exists: {}", path.display())
            }
            Self::SourceDestinationConflict {
                source,
                destination,
            } => write!(
                f,
                "source {} conflicts with destination {}",
                source.display(),
                destination.display()
            ),
            Self::Metadata { key, reason } => write!(f, "metadata {key:?}: {reason:?}"),
            Self::NonFiniteWeight {
                location,
                decoded_f32_bits,
            } => write!(f, "nonfinite weight {location:?}: 0x{decoded_f32_bits:08x}"),
            Self::Synthesis { location, source } => write!(f, "weight {location:?}: {source}"),
            Self::Stream { block, source } => write!(f, "block {block:?}: {source}"),
            #[cfg(feature = "gguf-import")]
            Self::Gguf { tensor, source } => write!(f, "GGUF tensor {tensor:?}: {source}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Synthesis { source, .. } => Some(source),
            Self::Stream { source, .. } => Some(source),
            #[cfg(feature = "gguf-import")]
            Self::Gguf { source, .. } => Some(source),
            _ => None,
        }
    }
}
