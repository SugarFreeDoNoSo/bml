use super::{check_rank, Shape, Tensor, TensorError, TensorView};
use crate::symbolic::Basis;
use std::ops::{Add, Mul};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// CPU kernel element types: f32 and f64. Sealed to fix numeric semantics.
pub trait Scalar: sealed::Sealed + Copy + Add<Output = Self> + Mul<Output = Self> {
    /// Additive identity for empty reductions.
    const ZERO: Self;
    /// Exact widening conversion for checked basis evaluation.
    fn to_f64(self) -> f64;
    /// Converts a finite basis result; rejects overflow and nonzero-to-zero rounding.
    fn from_basis_result(value: f64) -> Option<Self>;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    fn to_f64(self) -> f64 {
        self
    }
    fn from_basis_result(value: f64) -> Option<Self> {
        value.is_finite().then_some(value)
    }
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_basis_result(value: f64) -> Option<Self> {
        let narrowed = value as f32;
        (narrowed.is_finite() && (narrowed != 0.0 || value == 0.0)).then_some(narrowed)
    }
}

/// Elementwise addition with identical shapes. Uses ordinary IEEE arithmetic.
pub fn add<T: Scalar>(
    left: &TensorView<'_, T>,
    right: &TensorView<'_, T>,
) -> Result<Tensor<T>, TensorError> {
    zip_map(left, right, |_, x, y| Ok(x + y))
}

/// Elementwise product with identical shapes. Uses ordinary IEEE arithmetic.
pub fn mul<T: Scalar>(
    left: &TensorView<'_, T>,
    right: &TensorView<'_, T>,
) -> Result<Tensor<T>, TensorError> {
    zip_map(left, right, |_, x, y| Ok(x * y))
}

/// Rank-one dot product, accumulated from left to right in the element type.
/// Empty vectors return positive zero; multiplication is not fused with addition.
pub fn dot<T: Scalar>(
    left: &TensorView<'_, T>,
    right: &TensorView<'_, T>,
) -> Result<T, TensorError> {
    check_rank(left.shape.rank(), 1)?;
    check_rank(right.shape.rank(), 1)?;
    same_shape(left, right)?;
    Ok(left
        .data
        .iter()
        .zip(right.data)
        .fold(T::ZERO, |sum, (&x, &y)| sum + x * y))
}

/// Two-dimensional row-major (m,k) × (k,n) -> (m,n), with no transposition.
/// Each cell reduces k in order using the element type's IEEE arithmetic.
/// An empty inner dimension produces zeros; output size/allocation is checked.
pub fn matmul<T: Scalar>(
    left: &TensorView<'_, T>,
    right: &TensorView<'_, T>,
) -> Result<Tensor<T>, TensorError> {
    check_rank(left.shape.rank(), 2)?;
    check_rank(right.shape.rank(), 2)?;
    let [m, k] = [left.shape.dims()[0], left.shape.dims()[1]];
    let [rhs_k, n] = [right.shape.dims()[0], right.shape.dims()[1]];
    if k != rhs_k {
        return Err(TensorError::InnerDimensionMismatch {
            left: k,
            right: rhs_k,
        });
    }
    let shape = Shape::new([m, n])?;
    let mut data = output_buffer(shape.len())?;
    data.resize(shape.len(), T::ZERO);
    for (flat, cell) in data.iter_mut().enumerate() {
        let row = flat / n;
        let col = flat % n;
        for inner in 0..k {
            *cell = *cell + left.data[row * k + inner] * right.data[inner * n + col];
        }
    }
    Tensor::from_vec(shape, data)
}

/// Checked EML/BML on identical shapes, preserving the input dtype.
/// Evaluates using Basis::apply_real in f64, including for f32 inputs, then
/// rounds once to the output dtype. Reports the first invalid flat index.
/// This applies the primitive operator; it does not rewrite add/mul/matmul.
pub fn apply_basis<T: Scalar>(
    left: &TensorView<'_, T>,
    right: &TensorView<'_, T>,
    basis: Basis,
) -> Result<Tensor<T>, TensorError> {
    zip_map(left, right, |index, x, y| {
        let value = basis
            .apply_real(x.to_f64(), y.to_f64())
            .map_err(|source| TensorError::Evaluation { index, source })?;
        T::from_basis_result(value).ok_or(TensorError::OutputRange { index })
    })
}

fn same_shape<T>(left: &TensorView<'_, T>, right: &TensorView<'_, T>) -> Result<(), TensorError> {
    if left.shape != right.shape {
        return Err(TensorError::ShapeMismatch {
            left: left.shape.dims().to_vec(),
            right: right.shape.dims().to_vec(),
        });
    }
    Ok(())
}

fn output_buffer<T>(len: usize) -> Result<Vec<T>, TensorError> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| TensorError::AllocationFailed)?;
    Ok(data)
}

fn zip_map<T: Scalar>(
    left: &TensorView<'_, T>,
    right: &TensorView<'_, T>,
    mut f: impl FnMut(usize, T, T) -> Result<T, TensorError>,
) -> Result<Tensor<T>, TensorError> {
    same_shape(left, right)?;
    let mut data = output_buffer(left.shape.len())?;
    for (index, (&x, &y)) in left.data.iter().zip(right.data).enumerate() {
        data.push(f(index, x, y)?);
    }
    Tensor::from_vec(left.shape.clone(), data)
}
