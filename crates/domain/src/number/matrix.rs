//! Row-major symbolic matrices. Values remain expanded BML trees, not floats.
use super::{Bml, CertifiedBml, ExactError, ExactOperand, ExactSynthesisOptions};
use std::{
    fmt,
    ops::{Add, Index, Mul},
};

/// Explicit matrix and operation budgets, including reduction intermediates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixLimits {
    /// Maximum elements in each input or output matrix.
    pub max_elements: usize,
    /// Maximum sum of expanded tree bits in each matrix.
    pub max_total_bits: usize,
    /// Cumulative bits of produced combination/accumulation results per operation.
    pub max_work_bits: usize,
    /// Per-scalar proof, rational and tree budgets.
    pub scalar: ExactSynthesisOptions,
}
impl Default for MatrixLimits {
    fn default() -> Self {
        Self {
            max_elements: 65_536,
            max_total_bits: 16_777_216,
            max_work_bits: 134_217_728,
            scalar: ExactSynthesisOptions::default(),
        }
    }
}

/// Shape, resource, index or localized scalar failure. No partial matrix is returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// Shape product overflow or incompatible element count/shapes.
    Shape,
    /// Matrix or cumulative work budget exhausted.
    Limit,
    /// A result or reduction intermediate exceeded a matrix budget.
    Budget {
        /// Output row whose construction exhausted the budget.
        row: usize,
        /// Output column whose construction exhausted the budget.
        column: usize,
        /// Inner reduction index, if failure occurred during reduction.
        inner: Option<usize>,
    },
    /// Checked access or replacement used an invalid index.
    Index {
        /// Requested row.
        row: usize,
        /// Requested column.
        column: usize,
    },
    /// A scalar input, output or reduction step could not be justified/built.
    Scalar {
        /// Row of the failing input or output.
        row: usize,
        /// Column of the failing input or output.
        column: usize,
        /// Inner reduction index, or None for preparation/elementwise work.
        inner: Option<usize>,
        /// Original scalar error.
        source: ExactError,
    },
    /// A generalized empty reduction has no explicit identity policy.
    EmptyReduction,
    /// Conversion of a scalar before entering the matrix kernel failed.
    ScalarPreparation(ExactError),
}
impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget { row, column, inner } => {
                write!(f, "matrix budget at ({row},{column}), reduction {inner:?}")
            }
            Self::Shape => f.write_str("incompatible or overflowing symbolic matrix shape"),
            Self::Limit => f.write_str("symbolic matrix resource budget exceeded"),
            Self::Index { row, column } => {
                write!(f, "matrix index ({row},{column}) is out of range")
            }
            Self::Scalar {
                row,
                column,
                inner,
                source,
            } => write!(
                f,
                "symbolic matrix ({row},{column}), reduction {inner:?}: {source}"
            ),
            Self::EmptyReduction => {
                f.write_str("empty generalized reduction requires an explicit value")
            }
            Self::ScalarPreparation(e) => write!(f, "matrix scalar preparation: {e}"),
        }
    }
}
impl std::error::Error for MatrixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scalar { source, .. } | Self::ScalarPreparation(source) => Some(source),
            _ => None,
        }
    }
}

/// Initialization of a left-to-right generalized product reduction.
#[derive(Clone, Copy, Debug)]
pub enum ReductionInit<'a> {
    /// Start with the first combined pair; the explicit fallback applies only
    /// to an empty inner dimension. None makes that case an error.
    First {
        /// Explicit value for empty reductions.
        empty: Option<&'a Bml>,
    },
    /// Start every reduction with this value, including empty reductions.
    Value(&'a Bml),
}

/// Contiguous row-major Bml headers; each tree owns a separate word allocation.
/// No IndexMut is exposed: use transactional `replace` to preserve bit budgets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BmlMatrix {
    rows: usize,
    columns: usize,
    values: Vec<Bml>,
    limits: MatrixLimits,
}

fn count(rows: usize, columns: usize, limits: MatrixLimits) -> Result<usize, MatrixError> {
    let count = rows.checked_mul(columns).ok_or(MatrixError::Shape)?;
    if count > limits.max_elements {
        return Err(MatrixError::Limit);
    }
    Ok(count)
}
fn output_buffer<T>(len: usize) -> Result<Vec<T>, MatrixError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| MatrixError::Limit)?;
    Ok(values)
}
fn scalar_error(
    row: usize,
    column: usize,
    inner: Option<usize>,
    source: ExactError,
) -> MatrixError {
    MatrixError::Scalar {
        row,
        column,
        inner,
        source,
    }
}
fn charge(value: &Bml, work: &mut usize, limits: MatrixLimits) -> Result<(), MatrixError> {
    if value.bit_len() > limits.scalar.max_bits {
        return Err(MatrixError::Limit);
    }
    *work = work
        .checked_add(value.bit_len())
        .ok_or(MatrixError::Limit)?;
    if *work > limits.max_work_bits {
        return Err(MatrixError::Limit);
    }
    Ok(())
}

impl BmlMatrix {
    /// Construct without synthesizing or numerically evaluating the supplied trees.
    pub fn new(
        rows: usize,
        columns: usize,
        values: Vec<Bml>,
        limits: MatrixLimits,
    ) -> Result<Self, MatrixError> {
        if count(rows, columns, limits)? != values.len() {
            return Err(MatrixError::Shape);
        }
        let result = Self {
            rows,
            columns,
            values,
            limits,
        };
        result.validate(limits)?;
        Ok(result)
    }
    /// Matrix dimensions in row-major order.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
    /// Prepared structural trees in row-major order, without mutable access.
    pub fn as_slice(&self) -> &[Bml] {
        &self.values
    }
    /// Configured invariant and default operation limits.
    pub fn limits(&self) -> MatrixLimits {
        self.limits
    }
    /// Checked row/column access. Unlike Index, invalid indices return errors.
    pub fn get(&self, row: usize, column: usize) -> Result<&Bml, MatrixError> {
        if row >= self.rows || column >= self.columns {
            return Err(MatrixError::Index { row, column });
        }
        Ok(&self.values[row * self.columns + column])
    }
    /// Replace an element only if shape and total/per-element budgets remain valid.
    pub fn replace(&mut self, row: usize, column: usize, value: Bml) -> Result<Bml, MatrixError> {
        let old = self.get(row, column)?;
        let total = self
            .total_bits()?
            .checked_sub(old.bit_len())
            .and_then(|n| n.checked_add(value.bit_len()))
            .ok_or(MatrixError::Limit)?;
        if value.bit_len() > self.limits.scalar.max_bits || total > self.limits.max_total_bits {
            return Err(MatrixError::Limit);
        }
        Ok(std::mem::replace(
            &mut self.values[row * self.columns + column],
            value,
        ))
    }
    /// Logical tree bits, excluding Vec headers/capacities and proof metadata.
    pub fn total_bits(&self) -> Result<usize, MatrixError> {
        self.values.iter().try_fold(0usize, |sum, value| {
            sum.checked_add(value.bit_len()).ok_or(MatrixError::Limit)
        })
    }
    fn validate(&self, limits: MatrixLimits) -> Result<(), MatrixError> {
        if count(self.rows, self.columns, limits)? != self.values.len() {
            return Err(MatrixError::Shape);
        }
        if self.total_bits()? > limits.max_total_bits
            || self
                .values
                .iter()
                .any(|v| v.bit_len() > limits.scalar.max_bits)
        {
            return Err(MatrixError::Limit);
        }
        Ok(())
    }
    fn prepare(&self, limits: MatrixLimits) -> Result<Vec<CertifiedBml>, MatrixError> {
        self.validate(self.limits)?;
        self.validate(limits)?;
        let mut prepared = output_buffer(self.values.len())?;
        for (i, value) in self.values.iter().enumerate() {
            prepared.push(
                value
                    .certify_exact(limits.scalar)
                    .map_err(|e| scalar_error(i / self.columns, i % self.columns, None, e))?,
            );
        }
        Ok(prepared)
    }
    fn elementwise(
        &self,
        rhs: &Self,
        limits: MatrixLimits,
        multiply: bool,
    ) -> Result<Self, MatrixError> {
        if self.shape() != rhs.shape() {
            return Err(MatrixError::Shape);
        }
        let a = self.prepare(limits)?;
        let b = rhs.prepare(limits)?;
        let mut values = output_buffer(a.len())?;
        let mut work = 0;
        let mut output_bits = 0usize;
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            let result = if multiply {
                a.checked_mul(b, limits.scalar)
            } else {
                a.checked_add(b, limits.scalar)
            };
            let value = result
                .map_err(|e| scalar_error(i / self.columns, i % self.columns, None, e))?
                .into_term();
            charge(&value, &mut work, limits).map_err(|_| MatrixError::Budget {
                row: i / self.columns,
                column: i % self.columns,
                inner: None,
            })?;
            retain(
                &mut values,
                value,
                &mut output_bits,
                limits,
                i / self.columns,
                i % self.columns,
            )?;
        }
        Self::new(self.rows, self.columns, values, limits)
    }
    /// Elementwise sum with explicit matrix and scalar limits.
    pub fn add_with_limits(&self, rhs: &Self, limits: MatrixLimits) -> Result<Self, MatrixError> {
        self.elementwise(rhs, limits, false)
    }
    /// Elementwise multiplication, distinct from matrix multiplication.
    pub fn hadamard(&self, rhs: &Self) -> Result<Self, MatrixError> {
        self.elementwise(rhs, self.limits, true)
    }
    /// Elementwise multiplication with explicit budgets.
    pub fn hadamard_with_limits(
        &self,
        rhs: &Self,
        limits: MatrixLimits,
    ) -> Result<Self, MatrixError> {
        self.elementwise(rhs, limits, true)
    }
    /// Prepare a numeric or certified scalar once, then add it to each element.
    pub fn add_scalar<T: ExactOperand>(
        &self,
        scalar: T,
        limits: MatrixLimits,
    ) -> Result<Self, MatrixError> {
        self.scalar_map(scalar, limits, false)
    }
    /// Prepare a numeric or certified scalar once, then multiply each element.
    pub fn scale<T: ExactOperand>(
        &self,
        scalar: T,
        limits: MatrixLimits,
    ) -> Result<Self, MatrixError> {
        self.scalar_map(scalar, limits, true)
    }
    fn scalar_map<T: ExactOperand>(
        &self,
        scalar: T,
        limits: MatrixLimits,
        multiply: bool,
    ) -> Result<Self, MatrixError> {
        self.validate(limits)?;
        let scalar = scalar
            .into_certified(limits.scalar)
            .map_err(MatrixError::ScalarPreparation)?;
        let prepared = self.prepare(limits)?;
        let mut values = output_buffer(prepared.len())?;
        let mut work = 0;
        let mut output_bits = 0usize;
        for (i, value) in prepared.iter().enumerate() {
            let result = if multiply {
                value.checked_mul(&scalar, limits.scalar)
            } else {
                value.checked_add(&scalar, limits.scalar)
            };
            let value = result
                .map_err(|e| scalar_error(i / self.columns, i % self.columns, None, e))?
                .into_term();
            charge(&value, &mut work, limits).map_err(|_| MatrixError::Budget {
                row: i / self.columns,
                column: i % self.columns,
                inner: None,
            })?;
            retain(
                &mut values,
                value,
                &mut output_bits,
                limits,
                i / self.columns,
                i % self.columns,
            )?;
        }
        Self::new(self.rows, self.columns, values, limits)
    }
    /// Standard product: first product starts each nonempty reduction, followed
    /// by left-to-right additions. Empty reductions yield canonical symbolic zero.
    pub fn matmul_with_limits(
        &self,
        rhs: &Self,
        limits: MatrixLimits,
    ) -> Result<Self, MatrixError> {
        if self.columns != rhs.rows {
            return Err(MatrixError::Shape);
        }
        let len = count(self.rows, rhs.columns, limits)?;
        self.validate(limits)?;
        rhs.validate(limits)?;
        if len == 0 {
            return Self::new(self.rows, rhs.columns, vec![], limits);
        }
        let a = self.prepare(limits)?;
        let b = rhs.prepare(limits)?;
        let mut values = output_buffer(len)?;
        let mut work = 0;
        let mut output_bits = 0usize;
        for row in 0..self.rows {
            for column in 0..rhs.columns {
                let mut accumulator: Option<CertifiedBml> = None;
                for inner in 0..self.columns {
                    let product = a[row * self.columns + inner]
                        .checked_mul(&b[inner * rhs.columns + column], limits.scalar)
                        .map_err(|e| scalar_error(row, column, Some(inner), e))?;
                    charge(product.term(), &mut work, limits).map_err(|_| MatrixError::Budget {
                        row,
                        column,
                        inner: Some(inner),
                    })?;
                    accumulator = Some(match accumulator {
                        None => product,
                        Some(previous) => {
                            let sum = previous
                                .checked_add(&product, limits.scalar)
                                .map_err(|e| scalar_error(row, column, Some(inner), e))?;
                            charge(sum.term(), &mut work, limits).map_err(|_| {
                                MatrixError::Budget {
                                    row,
                                    column,
                                    inner: Some(inner),
                                }
                            })?;
                            sum
                        }
                    });
                }
                let value = match accumulator {
                    Some(value) => value.into_term(),
                    None => {
                        let zero = Bml::from_u64(88).map_err(|_| MatrixError::Shape)?;
                        charge(&zero, &mut work, limits).map_err(|_| MatrixError::Budget {
                            row,
                            column,
                            inner: None,
                        })?;
                        zero
                    }
                };
                retain(&mut values, value, &mut output_bits, limits, row, column)?;
            }
        }
        Self::new(self.rows, rhs.columns, values, limits)
    }
    /// Generalized left-to-right product. Callbacks own their mathematical domain
    /// policy; returned trees and all reduction intermediates remain budgeted.
    /// This API never assumes associativity or an identity for primitive B.
    pub fn generalized_product<C, A>(
        &self,
        rhs: &Self,
        init: ReductionInit<'_>,
        limits: MatrixLimits,
        mut combine: C,
        mut accumulate: A,
    ) -> Result<Self, MatrixError>
    where
        C: FnMut(&Bml, &Bml) -> Result<Bml, ExactError>,
        A: FnMut(&Bml, &Bml) -> Result<Bml, ExactError>,
    {
        if self.columns != rhs.rows {
            return Err(MatrixError::Shape);
        }
        self.validate(self.limits)?;
        rhs.validate(rhs.limits)?;
        self.validate(limits)?;
        rhs.validate(limits)?;
        let len = count(self.rows, rhs.columns, limits)?;
        if self.columns == 0 && matches!(init, ReductionInit::First { empty: None }) {
            return Err(MatrixError::EmptyReduction);
        }
        if len == 0 {
            return Self::new(self.rows, rhs.columns, vec![], limits);
        }
        let mut values = output_buffer(len)?;
        let mut work = 0;
        let mut output_bits = 0usize;
        for row in 0..self.rows {
            for column in 0..rhs.columns {
                let initial = match init {
                    ReductionInit::Value(value) => Some(value),
                    ReductionInit::First { empty } if self.columns == 0 => empty,
                    _ => None,
                };
                let mut accumulator = if let Some(value) = initial {
                    charge(value, &mut work, limits).map_err(|_| MatrixError::Budget {
                        row,
                        column,
                        inner: None,
                    })?;
                    Some(value.clone())
                } else {
                    None
                };
                for inner in 0..self.columns {
                    let product = combine(&self[(row, inner)], &rhs[(inner, column)])
                        .map_err(|e| scalar_error(row, column, Some(inner), e))?;
                    charge(&product, &mut work, limits).map_err(|_| MatrixError::Budget {
                        row,
                        column,
                        inner: Some(inner),
                    })?;
                    accumulator = Some(match accumulator {
                        None => product,
                        Some(previous) => {
                            let sum = accumulate(&previous, &product)
                                .map_err(|e| scalar_error(row, column, Some(inner), e))?;
                            charge(&sum, &mut work, limits).map_err(|_| MatrixError::Budget {
                                row,
                                column,
                                inner: Some(inner),
                            })?;
                            sum
                        }
                    });
                }
                retain(
                    &mut values,
                    accumulator.ok_or(MatrixError::EmptyReduction)?,
                    &mut output_bits,
                    limits,
                    row,
                    column,
                )?;
            }
        }
        Self::new(self.rows, rhs.columns, values, limits)
    }
}
impl Index<(usize, usize)> for BmlMatrix {
    type Output = Bml;
    /// Panics for an invalid index; use get for checked access.
    fn index(&self, (row, column): (usize, usize)) -> &Bml {
        self.get(row, column)
            .expect("symbolic matrix index out of bounds")
    }
}
impl Add<&BmlMatrix> for &BmlMatrix {
    type Output = Result<BmlMatrix, MatrixError>;
    fn add(self, rhs: &BmlMatrix) -> Self::Output {
        self.add_with_limits(rhs, self.limits)
    }
}
impl Mul<&BmlMatrix> for &BmlMatrix {
    type Output = Result<BmlMatrix, MatrixError>;
    fn mul(self, rhs: &BmlMatrix) -> Self::Output {
        self.matmul_with_limits(rhs, self.limits)
    }
}

fn retain(
    values: &mut Vec<Bml>,
    value: Bml,
    total: &mut usize,
    limits: MatrixLimits,
    row: usize,
    column: usize,
) -> Result<(), MatrixError> {
    *total = total
        .checked_add(value.bit_len())
        .ok_or(MatrixError::Budget {
            row,
            column,
            inner: None,
        })?;
    if *total > limits.max_total_bits {
        return Err(MatrixError::Budget {
            row,
            column,
            inner: None,
        });
    }
    values.push(value);
    Ok(())
}

macro_rules! owned_matrix_operator {
    ($trait:ident,$method:ident) => {
        impl $trait<BmlMatrix> for BmlMatrix {
            type Output = Result<BmlMatrix, MatrixError>;
            fn $method(self, rhs: BmlMatrix) -> Self::Output {
                $trait::$method(&self, &rhs)
            }
        }
        impl $trait<&BmlMatrix> for BmlMatrix {
            type Output = Result<BmlMatrix, MatrixError>;
            fn $method(self, rhs: &BmlMatrix) -> Self::Output {
                $trait::$method(&self, rhs)
            }
        }
        impl $trait<BmlMatrix> for &BmlMatrix {
            type Output = Result<BmlMatrix, MatrixError>;
            fn $method(self, rhs: BmlMatrix) -> Self::Output {
                $trait::$method(self, &rhs)
            }
        }
    };
}
owned_matrix_operator!(Add, add);
owned_matrix_operator!(Mul, mul);
