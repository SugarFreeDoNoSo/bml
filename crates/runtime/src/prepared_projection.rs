//! Exact symbolic projections over persisted BML weights.
//!
//! This path consumes BMLS trees directly, without GGUF, weight synthesis or
//! floating-point evaluation. It is separate from the legacy IEEE RPN profile.
//! Loading establishes structural validity; applying a projection additionally
//! requires the matrix engine to recognize/justify its scalar expressions.
use bml_domain::number::{BmlMatrix, MatrixError, MatrixLimits};
use bml_model_weights::{BlockId, Error, TensorId, WeightReader};
use std::fmt;

/// A failure before returning a complete prepared matrix or projection result.
#[derive(Debug)]
#[non_exhaustive]
pub enum ProjectionError {
    /// Container validation or block integrity failure.
    Storage(Error),
    /// Matrix shape, resource or localized scalar failure.
    Matrix(MatrixError),
    /// Dense projection weights must have exactly two GGML dimensions.
    Rank {
        /// Tensor requested by the caller.
        tensor: TensorId,
        /// Rank present in the container.
        actual: usize,
    },
}
impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage(e) => write!(f, "prepared weight storage: {e}"),
            Self::Matrix(e) => write!(f, "prepared projection: {e}"),
            Self::Rank { tensor, actual } => write!(
                f,
                "projection tensor {tensor:?} has rank {actual}, expected 2"
            ),
        }
    }
}
impl std::error::Error for ProjectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(e) => Some(e),
            Self::Matrix(e) => Some(e),
            Self::Rank { .. } => None,
        }
    }
}
impl From<Error> for ProjectionError {
    fn from(e: Error) -> Self {
        Self::Storage(e)
    }
}
impl From<MatrixError> for ProjectionError {
    fn from(e: MatrixError) -> Self {
        Self::Matrix(e)
    }
}

/// Reusable row-major projection matrix containing the original persisted trees.
///
/// Weights are owned: once loaded, the container handle and source file can be
/// discarded. Matrix budgets explicitly bound materialization of this tensor.
pub struct PreparedProjection {
    tensor: TensorId,
    name: String,
    weights: BmlMatrix,
}
impl PreparedProjection {
    /// Load a GGML `[columns, rows]` tensor as a row-major `[rows, columns]`
    /// matrix without permuting its flat sequence or synthesizing any weights.
    /// Block ranges and aggregate bits are checked before reserving the matrix.
    pub fn load(
        reader: &mut WeightReader,
        tensor: TensorId,
        limits: MatrixLimits,
    ) -> Result<Self, ProjectionError> {
        let info = reader.tensor(tensor)?.clone();
        if info.ggml_dimensions.len() != 2 {
            return Err(ProjectionError::Rank {
                tensor,
                actual: info.ggml_dimensions.len(),
            });
        }
        let columns = usize::try_from(info.ggml_dimensions[0]).map_err(|_| MatrixError::Limit)?;
        let rows = usize::try_from(info.ggml_dimensions[1]).map_err(|_| MatrixError::Limit)?;
        let count = rows.checked_mul(columns).ok_or(MatrixError::Limit)?;
        if count > limits.max_elements {
            return Err(MatrixError::Limit.into());
        }
        let end = info
            .first_block
            .0
            .checked_add(info.block_count)
            .ok_or(MatrixError::Limit)?;
        let mut total_bits = 0usize;
        for id in info.first_block.0..end {
            let block = reader.block_info(BlockId(id))?;
            total_bits = total_bits
                .checked_add(usize::try_from(block.useful_bits).map_err(|_| MatrixError::Limit)?)
                .ok_or(MatrixError::Limit)?;
            if total_bits > limits.max_total_bits {
                return Err(MatrixError::Limit.into());
            }
        }
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(count)
            .map_err(|_| MatrixError::Limit)?;
        for id in info.first_block.0..end {
            let block = reader.read_block(BlockId(id))?;
            if block
                .terms
                .iter()
                .any(|term| term.bit_len() > limits.scalar.max_bits)
            {
                return Err(MatrixError::Limit.into());
            }
            terms.extend(block.terms);
        }
        Ok(Self {
            tensor,
            name: info.name,
            weights: BmlMatrix::new(rows, columns, terms, limits)?,
        })
    }
    /// Identity of the tensor in its source container.
    pub fn tensor_id(&self) -> TensorId {
        self.tensor
    }
    /// Original GGUF tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Immutable prepared trees; access does not perform evaluation or synthesis.
    pub fn weights(&self) -> &BmlMatrix {
        &self.weights
    }
    /// Apply `weights × activations`, where activations have shape
    /// `[input_columns, batch]`. Produces symbolic `[output_rows, batch]` terms.
    ///
    /// Arithmetic and certificate recognition are bounded by `limits`; failures
    /// preserve the weights and return no partial output or floating fallback.
    pub fn apply(
        &self,
        activations: &BmlMatrix,
        limits: MatrixLimits,
    ) -> Result<BmlMatrix, ProjectionError> {
        Ok(self.weights.matmul_with_limits(activations, limits)?)
    }
}
