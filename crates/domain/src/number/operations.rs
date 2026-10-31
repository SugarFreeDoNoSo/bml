//! Structural expansion of arithmetic into the BML basis.
//!
//! Rules do not evaluate their inputs. Their domains describe mathematical
//! identities; the finite-real evaluator can additionally fail on intermediate
//! overflow, underflow or invalid logarithms, or lose significant precision.

use super::{Bml, BuildError, DEFAULT_MAX_BITS};

/// A budget for each intermediate and result produced by the expansion rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    /// Maximum number of bits in each constructed term.
    pub max_bits: usize,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            max_bits: DEFAULT_MAX_BITS,
        }
    }
}

impl Rules {
    fn op(&self, left: &Bml, right: &Bml) -> Result<Bml, BuildError> {
        left.op_with_limit(right, self.max_bits)
    }

    /// Build `2^x`.
    pub fn exp2(&self, x: &Bml) -> Result<Bml, BuildError> {
        self.op(x, &Bml::one())
    }

    /// Build `log2(x)`, for positive `x`.
    pub fn log2(&self, x: &Bml) -> Result<Bml, BuildError> {
        let inner = self.op(&Bml::one(), x)?;
        let reciprocal = self.exp2(&inner)?;
        self.op(&Bml::one(), &reciprocal)
    }

    // x - y, assuming x is strictly positive.
    fn sub_positive(&self, x: &Bml, y: &Bml) -> Result<Bml, BuildError> {
        self.op(&self.log2(x)?, &self.exp2(y)?)
    }

    /// Build `x + y` over the reals.
    pub fn add(&self, x: &Bml, y: &Bml) -> Result<Bml, BuildError> {
        let p = self.exp2(x)?;
        let q = self.op(x, &p)?; // 2^x - x is strictly positive.
        let difference = self.sub_positive(&q, y)?;
        self.sub_positive(&p, &difference)
    }

    /// Build `-x` over the reals.
    pub fn neg(&self, x: &Bml) -> Result<Bml, BuildError> {
        let p = self.exp2(x)?;
        let q = self.op(x, &p)?;
        self.sub_positive(&q, &p)
    }

    /// Build `x - y` over the reals.
    pub fn sub(&self, x: &Bml, y: &Bml) -> Result<Bml, BuildError> {
        let p = self.exp2(x)?;
        let q = self.op(x, &p)?;
        self.sub_positive(&p, &self.add(&q, y)?)
    }

    /// Build `x * y` for strictly positive operands.
    pub fn mul(&self, x: &Bml, y: &Bml) -> Result<Bml, BuildError> {
        self.exp2(&self.add(&self.log2(x)?, &self.log2(y)?)?)
    }

    /// Build `x / y` for strictly positive operands.
    pub fn div(&self, x: &Bml, y: &Bml) -> Result<Bml, BuildError> {
        self.exp2(&self.sub(&self.log2(x)?, &self.log2(y)?)?)
    }

    /// Build `1 / x` for strictly positive `x`.
    pub fn recip(&self, x: &Bml) -> Result<Bml, BuildError> {
        self.exp2(&self.neg(&self.log2(x)?)?)
    }

    // Multiply any real x by positive y, without taking log2(x).
    // p = 2^x > 0 and q = 2^x - x > 0, so x*y = p*y - q*y.
    fn scale(&self, x: &Bml, y: &Bml) -> Result<Bml, BuildError> {
        let p = self.exp2(x)?;
        let q = self.op(x, &p)?;
        self.sub_positive(&self.mul(&p, y)?, &self.mul(&q, y)?)
    }

    /// Build `x^y` for strictly positive `x` and `y`.
    pub fn pow(&self, x: &Bml, y: &Bml) -> Result<Bml, BuildError> {
        self.exp2(&self.scale(&self.log2(x)?, y)?)
    }

    /// Build `sqrt(x)` for strictly positive `x`.
    pub fn sqrt(&self, x: &Bml) -> Result<Bml, BuildError> {
        let half = self.exp2(&self.neg(&Bml::one())?)?;
        self.exp2(&self.scale(&self.log2(x)?, &half)?)
    }
}

impl Bml {
    /// Build `2^self` using the default bit limit.
    pub fn exp2(&self) -> Result<Self, BuildError> {
        Rules::default().exp2(self)
    }
    /// Build `log2(self)` for positive values, using the default bit limit.
    pub fn log2(&self) -> Result<Self, BuildError> {
        Rules::default().log2(self)
    }
    /// Build the real sum using the default bit limit.
    pub fn add(&self, rhs: &Self) -> Result<Self, BuildError> {
        Rules::default().add(self, rhs)
    }
    /// Build the real difference using the default bit limit.
    pub fn sub(&self, rhs: &Self) -> Result<Self, BuildError> {
        Rules::default().sub(self, rhs)
    }
    /// Build the real negation using the default bit limit.
    pub fn neg(&self) -> Result<Self, BuildError> {
        Rules::default().neg(self)
    }
    /// Build the product of positive values using the default bit limit.
    pub fn mul(&self, rhs: &Self) -> Result<Self, BuildError> {
        Rules::default().mul(self, rhs)
    }
    /// Build the quotient of positive values using the default bit limit.
    pub fn div(&self, rhs: &Self) -> Result<Self, BuildError> {
        Rules::default().div(self, rhs)
    }
    /// Build the reciprocal of a positive value using the default bit limit.
    pub fn recip(&self) -> Result<Self, BuildError> {
        Rules::default().recip(self)
    }
    /// Build the power for positive `self` and `rhs` using the default bit limit.
    pub fn pow(&self, rhs: &Self) -> Result<Self, BuildError> {
        Rules::default().pow(self, rhs)
    }
    /// Build the square root for positive `self` using the default bit limit.
    pub fn sqrt(&self) -> Result<Self, BuildError> {
        Rules::default().sqrt(self)
    }
}
