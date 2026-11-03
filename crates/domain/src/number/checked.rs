//! Arithmetic that preserves and replays exact operand derivations.
//!
//! ```
//! use bml_domain::number::{Bml, ExactSynthesisOptions};
//! let a = Bml::from_decimal_str_exact("-2")?;
//! let b = Bml::from_decimal_str_exact("3")?;
//! let mut product = (&a * &b)?;
//! assert!(product.target().is_negative());
//! product.try_add_assign(&b, ExactSynthesisOptions::default())?;
//! let sum = (&product + 4i32)?; // numeric four, not prefix code four
//! assert_eq!(sum.target().numerator_words_le(), &[1]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
use super::{
    Bml, CertifiedBml, ExactDerivation, ExactError, ExactInputError, ExactRational, ExactStep,
    ExactSynthesisOptions,
};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Sub,
    Mul,
    Div,
}

impl ExactStep {
    fn relocated(&self, offset: usize) -> Result<Self, ExactError> {
        let index = |i: usize| i.checked_add(offset).ok_or(ExactError::Limit);
        Ok(match *self {
            Self::One => Self::One,
            Self::Zero => Self::Zero,
            Self::Exp2(a) => Self::Exp2(index(a)?),
            Self::PowerDifference(a, b) => Self::PowerDifference(index(a)?, index(b)?),
            Self::Add(a, b) => Self::Add(index(a)?, index(b)?),
            Self::Sub(a, b) => Self::Sub(index(a)?, index(b)?),
            Self::Neg(a) => Self::Neg(index(a)?),
            Self::PositiveMul(a, b) => Self::PositiveMul(index(a)?, index(b)?),
            Self::PositiveDiv(a, b) => Self::PositiveDiv(index(a)?, index(b)?),
        })
    }
}

fn append(
    steps: &mut Vec<ExactStep>,
    step: ExactStep,
    options: ExactSynthesisOptions,
) -> Result<usize, ExactError> {
    if steps.len() >= options.max_steps {
        return Err(ExactError::Limit);
    }
    steps.try_reserve(1).map_err(|_| ExactError::Limit)?;
    steps.push(step);
    Ok(steps.len() - 1)
}

impl CertifiedBml {
    fn binary(
        &self,
        rhs: &Self,
        operation: Operation,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        if matches!(operation, Operation::Div) && rhs.target().is_zero() {
            return Err(ExactInputError::DivisionByZero.into());
        }
        let count = self
            .derivation()
            .steps
            .len()
            .checked_add(rhs.derivation().steps.len())
            .ok_or(ExactError::Limit)?;
        if count >= options.max_steps {
            return Err(ExactError::Limit);
        }
        let mut steps = Vec::new();
        steps
            .try_reserve_exact(count)
            .map_err(|_| ExactError::Limit)?;
        steps.extend_from_slice(&self.derivation().steps);
        let offset = steps.len();
        for step in &rhs.derivation().steps {
            steps.push(step.relocated(offset)?);
        }
        let mut a = self.derivation().root;
        let mut b = rhs
            .derivation()
            .root
            .checked_add(offset)
            .ok_or(ExactError::Limit)?;
        let step = match operation {
            Operation::Add => ExactStep::Add(a, b),
            Operation::Sub => ExactStep::Sub(a, b),
            Operation::Mul | Operation::Div => {
                // A proven zero numerator is valid for both operations. Division
                // by zero was rejected above, before this simplification.
                if self.target().is_zero()
                    || (matches!(operation, Operation::Mul) && rhs.target().is_zero())
                {
                    ExactStep::Zero
                } else {
                    if self.target().is_negative() {
                        a = append(&mut steps, ExactStep::Neg(a), options)?;
                    }
                    if rhs.target().is_negative() {
                        b = append(&mut steps, ExactStep::Neg(b), options)?;
                    }
                    let positive = match operation {
                        Operation::Mul => ExactStep::PositiveMul(a, b),
                        _ => ExactStep::PositiveDiv(a, b),
                    };
                    if self.target().is_negative() != rhs.target().is_negative() {
                        let positive = append(&mut steps, positive, options)?;
                        ExactStep::Neg(positive)
                    } else {
                        positive
                    }
                }
            }
        };
        let root = append(&mut steps, step, options)?;
        ExactDerivation { steps, root }.certify(options)
    }

    /// Exact addition with explicit construction and proof budgets.
    pub fn checked_add(
        &self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        self.binary(rhs, Operation::Add, options)
    }
    /// Exact subtraction with explicit construction and proof budgets.
    pub fn checked_sub(
        &self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        self.binary(rhs, Operation::Sub, options)
    }
    /// Exact multiplication for either sign and zero, justified by operand proofs.
    pub fn checked_mul(
        &self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        self.binary(rhs, Operation::Mul, options)
    }
    /// Exact division for either sign, rejecting a proved zero divisor.
    pub fn checked_div(
        &self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        self.binary(rhs, Operation::Div, options)
    }
    /// Exact negation, without inferring signs from floats.
    pub fn checked_neg(&self, options: ExactSynthesisOptions) -> Result<Self, ExactError> {
        if self.derivation().steps.len() >= options.max_steps {
            return Err(ExactError::Limit);
        }
        let mut proof = self.derivation().clone();
        proof.root = append(&mut proof.steps, ExactStep::Neg(proof.root), options)?;
        proof.certify(options)
    }
    /// Replace the target only after successful complete exact addition.
    pub fn try_add_assign(
        &mut self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<(), ExactError> {
        let result = self.checked_add(rhs, options)?;
        *self = result;
        Ok(())
    }
    /// Replace the target only after successful complete exact multiplication.
    pub fn try_mul_assign(
        &mut self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<(), ExactError> {
        let result = self.checked_mul(rhs, options)?;
        *self = result;
        Ok(())
    }
}

/// Numeric inputs accepted by certified arithmetic. Integer operands mean their
/// mathematical value; floats mean their exact IEEE rational, never BML codes.
pub trait ExactOperand {
    /// Convert once before applying the operation, propagating synthesis failures.
    fn into_certified(self, options: ExactSynthesisOptions) -> Result<CertifiedBml, ExactError>;
}
impl ExactOperand for CertifiedBml {
    fn into_certified(self, _: ExactSynthesisOptions) -> Result<CertifiedBml, ExactError> {
        Ok(self)
    }
}
impl ExactOperand for &CertifiedBml {
    fn into_certified(self, options: ExactSynthesisOptions) -> Result<CertifiedBml, ExactError> {
        self.derivation()
            .verify(self.term(), self.target(), options)
    }
}
impl ExactOperand for f64 {
    fn into_certified(self, options: ExactSynthesisOptions) -> Result<CertifiedBml, ExactError> {
        Bml::from_f64_exact_with_options(self, options)
    }
}
macro_rules! integer_operand {
    ($($ty:ty),* $(,)?) => { $(
        impl ExactOperand for $ty {
            fn into_certified(self, options: ExactSynthesisOptions) -> Result<CertifiedBml, ExactError> {
                let target = ExactRational::from_decimal_str(&self.to_string(), options.rational)?;
                Bml::from_rational_exact(&target, options)
            }
        }
    )* };
}
integer_operand!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! arithmetic {
    ($trait:ident, $method:ident, $checked:ident) => {
        impl<T: ExactOperand> $trait<T> for &CertifiedBml {
            type Output = Result<CertifiedBml, ExactError>;
            fn $method(self, rhs: T) -> Self::Output {
                let options = ExactSynthesisOptions::default();
                self.$checked(&rhs.into_certified(options)?, options)
            }
        }
        impl<T: ExactOperand> $trait<T> for CertifiedBml {
            type Output = Result<CertifiedBml, ExactError>;
            fn $method(self, rhs: T) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}
arithmetic!(Add, add, checked_add);
arithmetic!(Sub, sub, checked_sub);
arithmetic!(Mul, mul, checked_mul);
arithmetic!(Div, div, checked_div);
impl Neg for &CertifiedBml {
    type Output = Result<CertifiedBml, ExactError>;
    fn neg(self) -> Self::Output {
        self.checked_neg(ExactSynthesisOptions::default())
    }
}
impl Neg for CertifiedBml {
    type Output = Result<CertifiedBml, ExactError>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

macro_rules! reverse_numeric {
    ($($ty:ty),* $(,)?) => { $(
        impl Add<&CertifiedBml> for $ty {
            type Output = Result<CertifiedBml, ExactError>;
            fn add(self, rhs: &CertifiedBml) -> Self::Output { self.into_certified(ExactSynthesisOptions::default())? + rhs }
        }
        impl Sub<&CertifiedBml> for $ty {
            type Output = Result<CertifiedBml, ExactError>;
            fn sub(self, rhs: &CertifiedBml) -> Self::Output { self.into_certified(ExactSynthesisOptions::default())? - rhs }
        }
        impl Mul<&CertifiedBml> for $ty {
            type Output = Result<CertifiedBml, ExactError>;
            fn mul(self, rhs: &CertifiedBml) -> Self::Output { self.into_certified(ExactSynthesisOptions::default())? * rhs }
        }
        impl Div<&CertifiedBml> for $ty {
            type Output = Result<CertifiedBml, ExactError>;
            fn div(self, rhs: &CertifiedBml) -> Self::Output { self.into_certified(ExactSynthesisOptions::default())? / rhs }
        }
    )* };
}
reverse_numeric!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f64);

impl ExactOperand for Bml {
    fn into_certified(self, options: ExactSynthesisOptions) -> Result<CertifiedBml, ExactError> {
        self.certify_exact(options)
    }
}
impl ExactOperand for &Bml {
    fn into_certified(self, options: ExactSynthesisOptions) -> Result<CertifiedBml, ExactError> {
        self.certify_exact(options)
    }
}

impl Bml {
    /// Checked real addition after conservative recovery of operand proofs.
    pub fn checked_add(
        &self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        Ok(self
            .certify_exact(options)?
            .checked_add(&rhs.certify_exact(options)?, options)?
            .into_term())
    }
    /// Checked real subtraction after conservative recovery of operand proofs.
    pub fn checked_sub(
        &self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        Ok(self
            .certify_exact(options)?
            .checked_sub(&rhs.certify_exact(options)?, options)?
            .into_term())
    }
    /// Checked real multiplication including negative operands and zero.
    pub fn checked_mul(
        &self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        Ok(self
            .certify_exact(options)?
            .checked_mul(&rhs.certify_exact(options)?, options)?
            .into_term())
    }
    /// Checked real division, rejecting proved zero or unjustified operand domains.
    pub fn checked_div(
        &self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<Self, ExactError> {
        Ok(self
            .certify_exact(options)?
            .checked_div(&rhs.certify_exact(options)?, options)?
            .into_term())
    }
    /// Checked negation; structural `neg()` retains its historical contract.
    pub fn checked_neg(&self, options: ExactSynthesisOptions) -> Result<Self, ExactError> {
        Ok(self
            .certify_exact(options)?
            .checked_neg(options)?
            .into_term())
    }
    /// Transactional checked addition. Failure preserves both encoded operands.
    pub fn try_add_assign(
        &mut self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<(), ExactError> {
        let result = self.checked_add(rhs, options)?;
        *self = result;
        Ok(())
    }
    /// Transactional checked multiplication. Failure preserves both operands.
    pub fn try_mul_assign(
        &mut self,
        rhs: &Self,
        options: ExactSynthesisOptions,
    ) -> Result<(), ExactError> {
        let result = self.checked_mul(rhs, options)?;
        *self = result;
        Ok(())
    }
}

macro_rules! tree_arithmetic {
    ($trait:ident,$method:ident,$checked:ident) => {
        impl<T: ExactOperand> $trait<T> for &Bml {
            type Output = Result<Bml, ExactError>;
            fn $method(self, rhs: T) -> Self::Output {
                let options = ExactSynthesisOptions::default();
                Ok(self
                    .certify_exact(options)?
                    .$checked(&rhs.into_certified(options)?, options)?
                    .into_term())
            }
        }
        impl<T: ExactOperand> $trait<T> for Bml {
            type Output = Result<Bml, ExactError>;
            fn $method(self, rhs: T) -> Self::Output {
                $trait::$method(&self, rhs)
            }
        }
    };
}
tree_arithmetic!(Add, add, checked_add);
tree_arithmetic!(Sub, sub, checked_sub);
tree_arithmetic!(Mul, mul, checked_mul);
tree_arithmetic!(Div, div, checked_div);
impl Neg for &Bml {
    type Output = Result<Bml, ExactError>;
    fn neg(self) -> Self::Output {
        self.checked_neg(ExactSynthesisOptions::default())
    }
}
impl Neg for Bml {
    type Output = Result<Bml, ExactError>;
    fn neg(self) -> Self::Output {
        self.checked_neg(ExactSynthesisOptions::default())
    }
}
macro_rules! reverse_tree_numeric {
    ($($ty:ty),* $(,)?) => { $(
        impl Add<&Bml> for $ty {
            type Output=Result<Bml,ExactError>;
            fn add(self,rhs:&Bml)->Self::Output {
                let options=ExactSynthesisOptions::default();
                Ok(self.into_certified(options)?.checked_add(&rhs.certify_exact(options)?,options)?.into_term())
            }
        }
        impl Sub<&Bml> for $ty {
            type Output=Result<Bml,ExactError>;
            fn sub(self,rhs:&Bml)->Self::Output {
                let options=ExactSynthesisOptions::default();
                Ok(self.into_certified(options)?.checked_sub(&rhs.certify_exact(options)?,options)?.into_term())
            }
        }
        impl Mul<&Bml> for $ty {
            type Output=Result<Bml,ExactError>;
            fn mul(self,rhs:&Bml)->Self::Output {
                let options=ExactSynthesisOptions::default();
                Ok(self.into_certified(options)?.checked_mul(&rhs.certify_exact(options)?,options)?.into_term())
            }
        }
        impl Div<&Bml> for $ty {
            type Output=Result<Bml,ExactError>;
            fn div(self,rhs:&Bml)->Self::Output {
                let options=ExactSynthesisOptions::default();
                Ok(self.into_certified(options)?.checked_div(&rhs.certify_exact(options)?,options)?.into_term())
            }
        }
    )* };
}
reverse_tree_numeric!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f64);
