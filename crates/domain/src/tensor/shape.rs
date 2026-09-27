use super::{check_rank, TensorError};

/// Validated row-major shape. Rank zero is a scalar with one element.
/// Any zero dimension makes the tensor empty, even if other dimensions are huge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
    strides: Vec<usize>,
    len: usize,
}

impl Shape {
    /// Validates the dimension product with checked arithmetic.
    /// Empty shapes use all-zero strides: there is no addressable element.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let dims = dims.into();
        let mut strides = vec![0; dims.len()];
        let mut len = 0usize;
        if !dims.contains(&0) {
            len = 1;
            for axis in (0..dims.len()).rev() {
                strides[axis] = len;
                len = len
                    .checked_mul(dims[axis])
                    .ok_or(TensorError::SizeOverflow)?;
            }
        }
        Ok(Self { dims, strides, len })
    }

    /// Axis lengths, from slowest to fastest varying.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Canonical strides in elements, not bytes. All zero for empty tensors.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Number of axes; zero for a scalar.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Number of values, including one for a scalar.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether at least one axis has length zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Converts coordinates to a flat index after checking every axis.
    pub fn offset(&self, indices: &[usize]) -> Result<usize, TensorError> {
        check_rank(indices.len(), self.rank())?;
        for (axis, &index) in indices.iter().enumerate() {
            let size = self.dims[axis];
            if index >= size {
                return Err(TensorError::IndexOutOfBounds { axis, index, size });
            }
        }
        // Validate all axes first: an empty tensor can have huge preceding axes.
        indices
            .iter()
            .zip(&self.dims)
            .try_fold(0usize, |offset, (&index, &size)| {
                offset
                    .checked_mul(size)
                    .and_then(|value| value.checked_add(index))
                    .ok_or(TensorError::SizeOverflow)
            })
    }
}
