//! Errors at the GGUF file and codec boundaries.
use crate::GgufDataType;
use std::{fmt, io};

/// A failed read or conversion; callers need not parse diagnostic strings.
#[derive(Debug)]
pub enum GgufError {
    /// Opening or mapping the file failed.
    Io(io::Error),
    /// The file is not GGUF.
    InvalidMagic(u32),
    /// The version is not supported.
    UnsupportedVersion(u32),
    /// Only little-endian GGUF is currently supported.
    UnsupportedEndianness,
    /// A required byte range was not present.
    Truncated {
        /// Start of the required range.
        offset: usize,
        /// Required byte count.
        needed: usize,
        /// Available byte count.
        available: usize,
    },
    /// Metadata could not be decoded.
    InvalidMetadata {
        /// Start of the metadata entry.
        offset: usize,
        /// Detailed diagnostic from the metadata decoder.
        reason: String,
    },
    /// A tensor descriptor could not be decoded.
    InvalidTensorInfo {
        /// Start of the descriptor.
        offset: usize,
        /// Detailed descriptor diagnostic.
        reason: String,
    },
    /// general.alignment must be a nonzero U32 multiple of eight.
    InvalidAlignment,
    /// A tensor offset does not meet the declared alignment.
    UnalignedTensor {
        /// Tensor name.
        name: String,
        /// Offset relative to the data section.
        offset: u64,
        /// Required alignment in bytes.
        alignment: usize,
    },
    /// The tensor's shape, block layout or byte range is invalid.
    InvalidTensorRange {
        /// Tensor name.
        name: String,
    },
    /// No tensor has the requested name.
    TensorNotFound(String),
    /// No f32 decoder exists for this storage format.
    UnsupportedDataType(GgufDataType),
    /// Quantized data requires complete blocks.
    InvalidBlockElements {
        /// Storage format.
        data_type: GgufDataType,
        /// Requested number of elements.
        elements: usize,
        /// Required elements per block.
        block_elements: usize,
    },
    /// Checked size arithmetic overflowed.
    SizeOverflow,
    /// Encoded bytes must have exactly the expected length.
    InvalidDataLength {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },
    /// Allocation of the output buffer failed.
    Allocation,
    /// Conversion to the optional row-major CPU tensor representation failed.
    #[cfg(feature = "tensor")]
    Tensor(bml_domain::tensor::TensorError),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GGUF error: {self:?}")
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            #[cfg(feature = "tensor")]
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for GgufError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<GgufError> for io::Error {
    fn from(error: GgufError) -> Self {
        match error {
            GgufError::Io(error) => error,
            other => Self::new(io::ErrorKind::InvalidData, other),
        }
    }
}

#[cfg(feature = "tensor")]
impl From<bml_domain::tensor::TensorError> for GgufError {
    fn from(error: bml_domain::tensor::TensorError) -> Self {
        Self::Tensor(error)
    }
}
