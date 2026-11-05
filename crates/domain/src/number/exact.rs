//! Exact input targets. Parsing does not imply successful BML synthesis.
use std::{cmp::Ordering, fmt};

/// Bounds for exact rational input parsing, independent of term synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactInputLimits {
    /// All decimal digits, including leading and trailing zeros.
    pub max_digits: usize,
    /// Maximum bits in either integer component, before reduction as well.
    pub max_bits: usize,
    /// Maximum subtraction steps in binary GCD reduction.
    pub max_gcd_steps: usize,
}
impl Default for ExactInputLimits {
    fn default() -> Self {
        Self {
            max_digits: 1024,
            max_bits: 4096,
            max_gcd_steps: 16384,
        }
    }
}

/// Exact input conversion failed before term synthesis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExactInputError {
    /// This rational operation requires a proven integer exponent.
    NonIntegralExponent,
    /// Exact rational arithmetic proved that the divisor is zero.
    DivisionByZero,
    /// Input does not match `[+-]?[0-9]+(\.[0-9]+)?`.
    Syntax,
    /// IEEE NaN or infinity is not a real rational input.
    NonFinite,
    /// Input length, integer size or reduction work exceeds its budget.
    Limit,
}
impl fmt::Display for ExactInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonIntegralExponent => "exact rational exponent is not an integer",
            Self::DivisionByZero => "exact rational division by zero",
            Self::Syntax => "invalid exact decimal syntax",
            Self::NonFinite => "exact input must be finite",
            Self::Limit => "exact input resource budget exhausted",
        })
    }
}
impl std::error::Error for ExactInputError {}

/// Reduced signed rational with canonical little-endian magnitude words.
///
/// Zero is nonnegative, has an empty numerator, and denominator one. The fields
/// are private so callers cannot forge a reduction or a zero denominator.
/// This is a mathematical target, not a certificate about a BML tree.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExactRational {
    negative: bool,
    numerator: Vec<u64>,
    denominator: Vec<u64>,
}
impl ExactRational {
    /// Whether the nonzero rational is negative.
    pub fn is_negative(&self) -> bool {
        self.negative
    }
    /// Canonical magnitude of the numerator; empty means zero.
    pub fn numerator_words_le(&self) -> &[u64] {
        &self.numerator
    }
    /// Canonical positive denominator.
    pub fn denominator_words_le(&self) -> &[u64] {
        &self.denominator
    }

    /// Parse a decimal without floating arithmetic or exponential notation.
    pub fn from_decimal_str(text: &str, limits: ExactInputLimits) -> Result<Self, ExactInputError> {
        // Check a byte bound before scanning an arbitrarily long input.
        if text.len().saturating_sub(2) > limits.max_digits {
            return Err(ExactInputError::Limit);
        }
        let (negative, body) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        let mut point = false;
        let mut whole = 0usize;
        let mut fraction = 0usize;
        let mut digits = 0usize;
        for byte in body.bytes() {
            if byte == b'.' && !point && whole > 0 {
                point = true;
                continue;
            }
            if !byte.is_ascii_digit() {
                return Err(ExactInputError::Syntax);
            }
            digits += 1;
            if digits > limits.max_digits {
                return Err(ExactInputError::Limit);
            }
            if point {
                fraction += 1;
            } else {
                whole += 1;
            }
        }
        if whole == 0 || (point && fraction == 0) {
            return Err(ExactInputError::Syntax);
        }
        let mut numerator = Vec::new();
        for digit in body.bytes().filter(u8::is_ascii_digit) {
            mul_add_small(&mut numerator, 10, (digit - b'0') as u64, limits.max_bits)?;
        }
        let mut denominator = vec![1];
        check_bits(&denominator, limits.max_bits)?;
        for _ in 0..fraction {
            mul_add_small(&mut denominator, 10, 0, limits.max_bits)?;
        }
        Self::reduce(negative, numerator, denominator, limits)
    }

    /// Decode exactly the stored IEEE value, including all subnormals.
    /// Both signs of zero become the same mathematical zero.
    pub fn from_f64(value: f64, limits: ExactInputLimits) -> Result<Self, ExactInputError> {
        let bits = value.to_bits();
        let exponent = (bits >> 52) & 0x7ff;
        if exponent == 0x7ff {
            return Err(ExactInputError::NonFinite);
        }
        let fraction = bits & ((1u64 << 52) - 1);
        let mut significand = if exponent == 0 {
            fraction
        } else {
            (1u64 << 52) | fraction
        };
        if significand == 0 {
            check_bits(&[1], limits.max_bits)?;
            return Ok(Self {
                negative: false,
                numerator: vec![],
                denominator: vec![1],
            });
        }
        let mut power = if exponent == 0 {
            -1074
        } else {
            exponent as i32 - 1023 - 52
        };
        let trailing = significand.trailing_zeros();
        significand >>= trailing;
        power += trailing as i32;
        let mut numerator = vec![significand];
        let mut denominator = vec![1];
        if power >= 0 {
            numerator = shift_left(&numerator, power as usize, limits.max_bits)?;
        } else {
            denominator = shift_left(&denominator, (-power) as usize, limits.max_bits)?;
        }
        check_bits(&numerator, limits.max_bits)?;
        check_bits(&denominator, limits.max_bits)?;
        Ok(Self {
            negative: bits >> 63 != 0,
            numerator,
            denominator,
        })
    }

    fn reduce(
        negative: bool,
        numerator: Vec<u64>,
        denominator: Vec<u64>,
        limits: ExactInputLimits,
    ) -> Result<Self, ExactInputError> {
        if numerator.is_empty() {
            return Ok(Self {
                negative: false,
                numerator,
                denominator: vec![1],
            });
        }
        let divisor = gcd(
            numerator.clone(),
            denominator.clone(),
            limits.max_gcd_steps,
            limits.max_bits,
        )?;
        Ok(Self {
            negative,
            numerator: divide_exact(&numerator, &divisor),
            denominator: divide_exact(&denominator, &divisor),
        })
    }
}

fn normalize(words: &mut Vec<u64>) {
    let keep = words
        .iter()
        .rposition(|word| *word != 0)
        .map_or(0, |index| index + 1);
    words.truncate(keep);
}
pub(super) fn bit_len(words: &[u64]) -> usize {
    words.last().map_or(0, |last| {
        (words.len() - 1) * 64 + 64 - last.leading_zeros() as usize
    })
}
fn check_bits(words: &[u64], max: usize) -> Result<(), ExactInputError> {
    if bit_len(words) > max {
        Err(ExactInputError::Limit)
    } else {
        Ok(())
    }
}
fn mul_add_small(
    words: &mut Vec<u64>,
    multiplier: u64,
    add: u64,
    max: usize,
) -> Result<(), ExactInputError> {
    let mut carry = add as u128;
    for word in words.iter_mut() {
        let next = (*word as u128) * multiplier as u128 + carry;
        *word = next as u64;
        carry = next >> 64;
    }
    if carry != 0 {
        words.try_reserve(1).map_err(|_| ExactInputError::Limit)?;
        words.push(carry as u64);
    }
    check_bits(words, max)
}
pub(super) fn shift_left(
    words: &[u64],
    shift: usize,
    max: usize,
) -> Result<Vec<u64>, ExactInputError> {
    let bits = bit_len(words)
        .checked_add(shift)
        .ok_or(ExactInputError::Limit)?;
    if bits > max {
        return Err(ExactInputError::Limit);
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(bits.div_ceil(64))
        .map_err(|_| ExactInputError::Limit)?;
    result.resize(bits.div_ceil(64), 0);
    for (i, word) in words.iter().enumerate() {
        let offset = i + shift / 64;
        result[offset] |= word << (shift % 64);
        if !shift.is_multiple_of(64) && offset + 1 < result.len() {
            result[offset + 1] |= word >> (64 - shift % 64);
        }
    }
    normalize(&mut result);
    Ok(result)
}
fn trailing(words: &[u64]) -> usize {
    words
        .iter()
        .position(|w| *w != 0)
        .map_or(0, |i| i * 64 + words[i].trailing_zeros() as usize)
}
fn shift_right(words: &mut Vec<u64>, shift: usize) {
    words.drain(..shift / 64);
    let rest = shift % 64;
    if rest != 0 {
        for i in 0..words.len() {
            words[i] = words[i] >> rest | words.get(i + 1).copied().unwrap_or(0) << (64 - rest);
        }
    }
    normalize(words);
}
fn compare(a: &[u64], b: &[u64]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}
pub(super) fn subtract(a: &mut Vec<u64>, b: &[u64]) {
    let mut borrow = false;
    for (i, word) in a.iter_mut().enumerate() {
        let (x, one) = word.overflowing_sub(b.get(i).copied().unwrap_or(0));
        let (y, two) = x.overflowing_sub(u64::from(borrow));
        *word = y;
        borrow = one || two;
    }
    debug_assert!(!borrow);
    normalize(a);
}
fn gcd(
    mut a: Vec<u64>,
    mut b: Vec<u64>,
    steps: usize,
    max: usize,
) -> Result<Vec<u64>, ExactInputError> {
    let common = trailing(&a).min(trailing(&b));
    let shift = trailing(&a);
    shift_right(&mut a, shift);
    let shift = trailing(&b);
    shift_right(&mut b, shift);
    for _ in 0..steps {
        match compare(&a, &b) {
            Ordering::Equal => return shift_left(&a, common, max),
            Ordering::Greater => {
                subtract(&mut a, &b);
                let shift = trailing(&a);
                shift_right(&mut a, shift);
            }
            Ordering::Less => {
                subtract(&mut b, &a);
                let shift = trailing(&b);
                shift_right(&mut b, shift);
            }
        }
    }
    Err(ExactInputError::Limit)
}
// Binary long division is bounded by the input bit budget; divisor is a GCD.
fn divide_exact(value: &[u64], divisor: &[u64]) -> Vec<u64> {
    let mut quotient = vec![0u64; value.len()];
    let mut remainder = Vec::with_capacity(divisor.len() + 1);
    for bit in (0..bit_len(value)).rev() {
        let mut carry = (value[bit / 64] >> (bit % 64)) & 1;
        for word in &mut remainder {
            let next = *word >> 63;
            *word = (*word << 1) | carry;
            carry = next;
        }
        if carry != 0 {
            remainder.push(carry);
        }
        if compare(&remainder, divisor) != Ordering::Less {
            subtract(&mut remainder, divisor);
            quotient[bit / 64] |= 1u64 << (bit % 64);
        }
    }
    debug_assert!(remainder.is_empty());
    normalize(&mut quotient);
    quotient
}

impl ExactRational {
    /// Exact integer target, distinct from parsing an integer as a BML code.
    pub fn from_i64(value: i64) -> Self {
        Self {
            negative: value < 0,
            numerator: if value == 0 {
                vec![]
            } else {
                vec![value.unsigned_abs()]
            },
            denominator: vec![1],
        }
    }

    /// Whether this target is mathematically zero.
    pub fn is_zero(&self) -> bool {
        self.numerator.is_empty()
    }

    /// Exact negation, preserving canonical zero.
    pub fn negated(&self) -> Self {
        let mut result = self.clone();
        if !result.is_zero() {
            result.negative = !result.negative;
        }
        result
    }

    /// Add exact rational targets within an intermediate integer budget.
    pub fn checked_add(
        &self,
        rhs: &Self,
        limits: ExactInputLimits,
    ) -> Result<Self, ExactInputError> {
        self.check_limit(limits)?;
        rhs.check_limit(limits)?;
        let left = multiply(&self.numerator, &rhs.denominator, limits.max_bits)?;
        let right = multiply(&rhs.numerator, &self.denominator, limits.max_bits)?;
        let denominator = multiply(&self.denominator, &rhs.denominator, limits.max_bits)?;
        let (negative, numerator) = if self.negative == rhs.negative {
            (self.negative, add(&left, &right, limits.max_bits)?)
        } else if compare(&left, &right) == Ordering::Less {
            let mut result = right;
            subtract(&mut result, &left);
            (rhs.negative, result)
        } else {
            let mut result = left;
            subtract(&mut result, &right);
            (self.negative, result)
        };
        Self::reduce(negative, numerator, denominator, limits)
    }

    /// Subtract exact rational targets within an intermediate integer budget.
    pub fn checked_sub(
        &self,
        rhs: &Self,
        limits: ExactInputLimits,
    ) -> Result<Self, ExactInputError> {
        self.checked_add(&rhs.negated(), limits)
    }

    /// Multiply exact rational targets. Budgets apply before reduction.
    pub fn checked_mul(
        &self,
        rhs: &Self,
        limits: ExactInputLimits,
    ) -> Result<Self, ExactInputError> {
        self.check_limit(limits)?;
        rhs.check_limit(limits)?;
        let numerator = multiply(&self.numerator, &rhs.numerator, limits.max_bits)?;
        let denominator = multiply(&self.denominator, &rhs.denominator, limits.max_bits)?;
        Self::reduce(
            self.negative != rhs.negative,
            numerator,
            denominator,
            limits,
        )
    }

    /// Divide exact targets; a zero divisor is a proven domain error.
    pub fn checked_div(
        &self,
        rhs: &Self,
        limits: ExactInputLimits,
    ) -> Result<Self, ExactInputError> {
        if rhs.is_zero() {
            return Err(ExactInputError::DivisionByZero);
        }
        self.check_limit(limits)?;
        rhs.check_limit(limits)?;
        let numerator = multiply(&self.numerator, &rhs.denominator, limits.max_bits)?;
        let denominator = multiply(&self.denominator, &rhs.numerator, limits.max_bits)?;
        Self::reduce(
            self.negative != rhs.negative,
            numerator,
            denominator,
            limits,
        )
    }

    fn check_limit(&self, limits: ExactInputLimits) -> Result<(), ExactInputError> {
        check_bits(&self.numerator, limits.max_bits)?;
        check_bits(&self.denominator, limits.max_bits)
    }
}

fn add(a: &[u64], b: &[u64], max: usize) -> Result<Vec<u64>, ExactInputError> {
    let size = a.len().max(b.len());
    let mut result = Vec::new();
    result
        .try_reserve_exact(size.checked_add(1).ok_or(ExactInputError::Limit)?)
        .map_err(|_| ExactInputError::Limit)?;
    let mut carry = 0u128;
    for i in 0..size {
        let next =
            a.get(i).copied().unwrap_or(0) as u128 + b.get(i).copied().unwrap_or(0) as u128 + carry;
        result.push(next as u64);
        carry = next >> 64;
    }
    if carry != 0 {
        result.push(carry as u64);
    }
    check_bits(&result, max)?;
    Ok(result)
}

fn multiply(a: &[u64], b: &[u64], max: usize) -> Result<Vec<u64>, ExactInputError> {
    if a.is_empty() || b.is_empty() {
        return Ok(vec![]);
    }
    // A nonzero product needs at least bits(a)+bits(b)-1 bits.
    let bound = bit_len(a)
        .checked_add(bit_len(b))
        .ok_or(ExactInputError::Limit)?;
    if bound - 1 > max {
        return Err(ExactInputError::Limit);
    }
    let size = a.len().checked_add(b.len()).ok_or(ExactInputError::Limit)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(size)
        .map_err(|_| ExactInputError::Limit)?;
    result.resize(size, 0u64);
    for (i, x) in a.iter().enumerate() {
        let mut carry = 0u128;
        for (j, y) in b.iter().enumerate() {
            let next = (*x as u128) * (*y as u128) + result[i + j] as u128 + carry;
            result[i + j] = next as u64;
            carry = next >> 64;
        }
        result[i + b.len()] = carry as u64;
    }
    normalize(&mut result);
    check_bits(&result, max)?;
    Ok(result)
}

impl ExactRational {
    pub(super) fn natural(words: Vec<u64>) -> Self {
        Self {
            negative: false,
            numerator: words,
            denominator: vec![1],
        }
    }

    /// Exact `2^self` when the exponent is an integer within the bit budget.
    /// This intentionally does not claim rational closure for fractional powers.
    pub fn checked_exp2(&self, limits: ExactInputLimits) -> Result<Self, ExactInputError> {
        self.check_limit(limits)?;
        if self.denominator != [1] {
            return Err(ExactInputError::NonIntegralExponent);
        }
        let exponent = match self.numerator.as_slice() {
            [] => 0,
            [word] => usize::try_from(*word).map_err(|_| ExactInputError::Limit)?,
            _ => return Err(ExactInputError::Limit),
        };
        let power = shift_left(&[1], exponent, limits.max_bits)?;
        Ok(if self.negative {
            Self {
                negative: false,
                numerator: vec![1],
                denominator: power,
            }
        } else {
            Self::natural(power)
        })
    }
}

/// Failure to round an exact rational into a finite nonzero f64.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RationalFloatError {
    /// Input or shifted integer intermediates exceed the explicit bit budget.
    Limit,
    /// Round-to-nearest would produce infinity.
    Overflow,
    /// A nonzero rational rounds to zero.
    Underflow,
}
impl fmt::Display for RationalFloatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limit => "exact rational rounding budget exhausted",
            Self::Overflow => "exact rational rounds beyond finite f64",
            Self::Underflow => "nonzero exact rational rounds to zero",
        })
    }
}
impl std::error::Error for RationalFloatError {}

impl ExactRational {
    /// Round once to binary64, nearest with ties to even, using integer division.
    /// Subnormal results are supported. Nonzero-to-zero underflow and infinite
    /// overflow are errors; mathematical zero returns positive zero.
    /// `max_bits` bounds input and shifted integer intermediates. Work is bounded
    /// by their bit lengths; no approximate equality or floating arithmetic is used.
    pub fn to_f64_rounded(&self, limits: ExactInputLimits) -> Result<f64, RationalFloatError> {
        self.check_limit(limits)
            .map_err(|_| RationalFloatError::Limit)?;
        if self.is_zero() {
            return Ok(0.0);
        }
        let nbits =
            i64::try_from(bit_len(&self.numerator)).map_err(|_| RationalFloatError::Limit)?;
        let dbits =
            i64::try_from(bit_len(&self.denominator)).map_err(|_| RationalFloatError::Limit)?;
        let mut exponent = nbits - dbits;
        if exponent > 1024 {
            return Err(RationalFloatError::Overflow);
        }
        if exponent < -1075 {
            return Err(RationalFloatError::Underflow);
        }
        let shift = |words: &[u64], bits: usize| {
            shift_left(words, bits, limits.max_bits).map_err(|_| RationalFloatError::Limit)
        };
        let ordering = if exponent >= 0 {
            compare(
                &self.numerator,
                &shift(&self.denominator, exponent as usize)?,
            )
        } else {
            compare(
                &shift(&self.numerator, (-exponent) as usize)?,
                &self.denominator,
            )
        };
        if ordering == Ordering::Less {
            exponent -= 1;
        }
        if exponent > 1023 {
            return Err(RationalFloatError::Overflow);
        }
        if exponent < -1075 {
            return Err(RationalFloatError::Underflow);
        }
        let normal = exponent >= -1022;
        let scaling = if normal { 52 - exponent } else { 1074 };
        let (numerator, denominator) = if scaling >= 0 {
            (
                shift(&self.numerator, scaling as usize)?,
                self.denominator.clone(),
            )
        } else {
            (
                self.numerator.clone(),
                shift(&self.denominator, (-scaling) as usize)?,
            )
        };
        let mut significand = rounded_quotient(&numerator, &denominator)?;
        let magnitude = if normal {
            if significand == 1u64 << 53 {
                significand >>= 1;
                exponent += 1;
            }
            if exponent > 1023 {
                return Err(RationalFloatError::Overflow);
            }
            ((exponent + 1023) as u64) << 52 | (significand - (1u64 << 52))
        } else {
            if significand == 0 {
                return Err(RationalFloatError::Underflow);
            }
            // Rounding can carry into the smallest normal value (bit 52).
            significand
        };
        Ok(f64::from_bits(magnitude | (u64::from(self.negative) << 63)))
    }
}

fn rounded_quotient(value: &[u64], divisor: &[u64]) -> Result<u64, RationalFloatError> {
    let mut quotient = 0u64;
    let mut remainder = Vec::new();
    remainder
        .try_reserve_exact(
            divisor
                .len()
                .checked_add(1)
                .ok_or(RationalFloatError::Limit)?,
        )
        .map_err(|_| RationalFloatError::Limit)?;
    for bit in (0..bit_len(value)).rev() {
        let mut carry = (value[bit / 64] >> (bit % 64)) & 1;
        for word in &mut remainder {
            let next = *word >> 63;
            *word = (*word << 1) | carry;
            carry = next;
        }
        if carry != 0 {
            remainder.push(carry);
        }
        if compare(&remainder, divisor) != Ordering::Less {
            subtract(&mut remainder, divisor);
            if bit >= 64 {
                return Err(RationalFloatError::Limit);
            }
            quotient |= 1u64 << bit;
        }
    }
    // Compare 2*remainder with divisor without allocating an extra high bit.
    let mut complement = divisor.to_vec();
    subtract(&mut complement, &remainder);
    let ordering = compare(&remainder, &complement);
    if ordering == Ordering::Greater || (ordering == Ordering::Equal && quotient & 1 != 0) {
        quotient = quotient.checked_add(1).ok_or(RationalFloatError::Limit)?;
    }
    Ok(quotient)
}
