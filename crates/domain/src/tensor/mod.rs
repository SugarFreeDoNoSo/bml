//! Contiguous row-major tensors and deterministic CPU reference kernels.
//!
//! Shape metadata is validated independently of values. Views borrow the value
//! buffer (only metadata is cloned). Native kernels use the element type's IEEE
//! arithmetic; [`apply_basis`] explicitly selects checked EML/BML evaluation.
//! Broadcasting, strided computation and implicit dtype conversion are absent.
//!
//! ```
//! use bml_domain::tensor::{matmul, Shape, Tensor};
//! let a = Tensor::from_vec(Shape::new([2, 3])?, vec![1., 2., 3., 4., 5., 6.])?;
//! let b = Tensor::from_vec(Shape::new([3, 1])?, vec![2., -1., 0.5])?;
//! let c = matmul(&a.view(), &b.view())?;
//! assert_eq!(c.shape().dims(), &[2, 1]);
//! assert_eq!(c.as_slice(), &[1.5, 6.0]);
//! # Ok::<(), bml_domain::tensor::TensorError>(())
//! ```

mod error;
mod kernels;
mod shape;

pub use error::TensorError;
pub use kernels::{add, apply_basis, dot, matmul, mul, Scalar};
pub use shape::Shape;

/// An owned buffer whose length exactly matches its immutable shape metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    shape: Shape,
    data: Vec<T>,
}

impl<T> Tensor<T> {
    /// Takes ownership without copying values; rejects a mismatched length.
    pub fn from_vec(shape: Shape, data: Vec<T>) -> Result<Self, TensorError> {
        check_len(&shape, data.len())?;
        Ok(Self { shape, data })
    }

    /// Validated row-major dimensions and strides.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Values in row-major order (last axis varies fastest).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable values; the slice cannot resize the buffer or invalidate shape.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Releases shape metadata and returns the original owned buffer.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Borrows values; clones only the shape metadata.
    pub fn view(&self) -> TensorView<'_, T> {
        TensorView {
            shape: self.shape.clone(),
            data: &self.data,
        }
    }

    /// Changes dimensions without reallocating values. Failure leaves self intact.
    pub fn reshape(&mut self, shape: Shape) -> Result<(), TensorError> {
        check_len(&shape, self.data.len())?;
        self.shape = shape;
        Ok(())
    }
}

/// Immutable borrowed values with owned metadata; cannot outlive the buffer.
///
/// ```compile_fail
/// use bml_domain::tensor::{Shape, Tensor};
/// let view = {
///     let tensor = Tensor::from_vec(Shape::new([1]).unwrap(), vec![1.0]).unwrap();
///     tensor.view()
/// };
/// println!("{:?}", view.as_slice());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TensorView<'a, T> {
    shape: Shape,
    data: &'a [T],
}

impl<'a, T> TensorView<'a, T> {
    /// Borrows an exact contiguous slice, without copying values.
    pub fn from_slice(shape: Shape, data: &'a [T]) -> Result<Self, TensorError> {
        check_len(&shape, data.len())?;
        Ok(Self { shape, data })
    }

    /// Imports only canonical row-major strides, in elements. Other layouts
    /// return `NonContiguous`, including alternative singleton/empty strides.
    /// Callers must slice to the tensor's exact range before constructing a view.
    pub fn from_strided(
        shape: Shape,
        data: &'a [T],
        strides: &[usize],
    ) -> Result<Self, TensorError> {
        check_rank(strides.len(), shape.rank())?;
        if strides != shape.strides() {
            return Err(TensorError::NonContiguous);
        }
        Self::from_slice(shape, data)
    }

    /// Validated row-major dimensions and strides.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Borrows the values for the original buffer lifetime.
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }

    /// Returns an element or a typed rank/bounds error, never wraps an index.
    pub fn get(&self, indices: &[usize]) -> Result<&'a T, TensorError> {
        Ok(&self.data[self.shape.offset(indices)?])
    }

    /// Returns new metadata over the same values; the original view is unchanged.
    pub fn reshape(&self, shape: Shape) -> Result<Self, TensorError> {
        Self::from_slice(shape, self.data)
    }
}

fn check_len(shape: &Shape, actual: usize) -> Result<(), TensorError> {
    if shape.len() != actual {
        return Err(TensorError::LengthMismatch {
            expected: shape.len(),
            actual,
        });
    }
    Ok(())
}

fn check_rank(actual: usize, expected: usize) -> Result<(), TensorError> {
    if actual != expected {
        return Err(TensorError::RankMismatch { expected, actual });
    }
    Ok(())
}
