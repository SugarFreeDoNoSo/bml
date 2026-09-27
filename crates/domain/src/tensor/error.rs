use crate::symbolic::EvalError;
use std::fmt;

/// Structural, allocation or checked numeric failure of a tensor operation.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorError {
    /// The shape's element count exceeds usize.
    SizeOverflow,
    /// A buffer or reshape has the wrong element count.
    LengthMismatch {
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// Broadcasting is not implicit, even for shapes of equal element count.
    ShapeMismatch {
        /// Left dimensions.
        left: Vec<usize>,
        /// Right dimensions.
        right: Vec<usize>,
    },
    /// A kernel, stride list or index has the wrong number of axes.
    RankMismatch {
        /// Required rank.
        expected: usize,
        /// Supplied rank.
        actual: usize,
    },
    /// A coordinate reaches or exceeds its axis length.
    IndexOutOfBounds {
        /// Axis position.
        axis: usize,
        /// Supplied coordinate.
        index: usize,
        /// Axis length.
        size: usize,
    },
    /// The supplied strides are not the canonical contiguous layout.
    NonContiguous,
    /// Matmul requires matching inner dimensions.
    InnerDimensionMismatch {
        /// Columns of the left matrix.
        left: usize,
        /// Rows of the right matrix.
        right: usize,
    },
    /// The output buffer cannot be reserved (including byte-capacity overflow).
    AllocationFailed,
    /// Checked EML/BML evaluation failed at a flat row-major index.
    Evaluation {
        /// Position of the first invalid pair.
        index: usize,
        /// Scalar evaluation error.
        source: EvalError,
    },
    /// A finite f64 basis result overflows f32 or rounds a nonzero value to zero.
    OutputRange {
        /// Position of the first unrepresentable result.
        index: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => f.write_str("tensor element count exceeds usize"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "expected {expected} values, got {actual}")
            }
            Self::ShapeMismatch { left, right } => {
                write!(f, "incompatible shapes: {left:?} and {right:?}")
            }
            Self::RankMismatch { expected, actual } => {
                write!(f, "expected rank {expected}, got {actual}")
            }
            Self::IndexOutOfBounds { axis, index, size } => {
                write!(f, "index {index} outside axis {axis} of length {size}")
            }
            Self::NonContiguous => f.write_str("canonical contiguous strides required"),
            Self::InnerDimensionMismatch { left, right } => {
                write!(f, "matmul inner dimensions differ: {left} and {right}")
            }
            Self::AllocationFailed => f.write_str("cannot reserve tensor output"),
            Self::Evaluation { index, source } => {
                write!(f, "basis evaluation at element {index}: {source}")
            }
            Self::OutputRange { index } => write!(
                f,
                "basis result at element {index} is outside the output dtype range"
            ),
        }
    }
}

impl std::error::Error for TensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation { source, .. } => Some(source),
            _ => None,
        }
    }
}
