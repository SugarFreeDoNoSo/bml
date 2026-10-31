use super::Bml;
use crate::symbolic::{Basis, EvalError};
use std::fmt;

/// Default maximum number of encoded bits produced by an operation.
pub const DEFAULT_MAX_BITS: usize = 1_048_576;

/// Invalid prefix encoding of a BML term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodingError {
    /// No binary digits were supplied.
    Empty,
    /// A character other than `0` or `1` occurred at this byte index.
    InvalidDigit {
        /// Byte offset of the invalid character.
        index: usize,
    },
    /// Leading zeroes are not a canonical integer representation.
    LeadingZero,
    /// The input ends before all operands have been supplied.
    Truncated,
    /// More bits follow the end of the encoded term.
    TrailingBits,
    /// The input is too large to represent on this platform.
    SizeOverflow,
}

/// Failure to construct a term within its configured size bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The resulting encoding would exceed the permitted bit count.
    BitLimit {
        /// Number of bits required by this construction.
        required: usize,
        /// Maximum permitted number of bits.
        limit: usize,
    },
    /// The resulting length cannot be represented on this platform.
    SizeOverflow,
}

impl fmt::Display for EncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("empty binary encoding"),
            Self::InvalidDigit { index } => write!(f, "invalid binary digit at byte {index}"),
            Self::LeadingZero => f.write_str("noncanonical leading zero"),
            Self::Truncated => f.write_str("incomplete BML term"),
            Self::TrailingBits => f.write_str("bits after complete BML term"),
            Self::SizeOverflow => f.write_str("encoded size overflow"),
        }
    }
}

impl std::error::Error for EncodingError {}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitLimit { required, limit } => {
                write!(f, "term requires {required} bits (limit {limit})")
            }
            Self::SizeOverflow => f.write_str("encoded size overflow"),
        }
    }
}

impl std::error::Error for BuildError {}

impl Bml {
    /// The terminal whose mathematical value is one.
    pub fn one() -> Self {
        Self { words: Vec::new() }
    }

    /// Parse a canonical integer encoding, validating the complete term.
    pub fn from_u64(value: u64) -> Result<Self, EncodingError> {
        Self::from_words_le(&[value])
    }

    /// Parse little-endian words as an integer, ignoring high zero words.
    pub fn from_words_le(words: &[u64]) -> Result<Self, EncodingError> {
        let len = words
            .iter()
            .rposition(|&word| word != 0)
            .map_or(0, |i| i + 1);
        let value = Self {
            words: words[..len].to_vec(),
        };
        value.validate()?;
        Ok(value)
    }

    /// Return the integer encoding when it fits in a single word.
    pub fn to_u64(&self) -> Option<u64> {
        match self.words.as_slice() {
            [] => Some(0),
            [word] => Some(*word),
            _ => None,
        }
    }

    /// Return canonical little-endian words; zero is the empty slice.
    pub fn to_words_le(&self) -> &[u64] {
        &self.words
    }

    /// Parse a binary prefix encoding without leading zeroes.
    pub fn from_binary(bits: &str) -> Result<Self, EncodingError> {
        if bits.is_empty() {
            return Err(EncodingError::Empty);
        }
        for (index, byte) in bits.bytes().enumerate() {
            if byte != b'0' && byte != b'1' {
                return Err(EncodingError::InvalidDigit { index });
            }
        }
        if bits.len() > 1 && bits.as_bytes()[0] == b'0' {
            return Err(EncodingError::LeadingZero);
        }
        let word_count = bits
            .len()
            .checked_add(63)
            .ok_or(EncodingError::SizeOverflow)?
            / 64;
        let mut words = vec![0; word_count];
        for (index, byte) in bits.bytes().rev().enumerate() {
            if byte == b'1' {
                words[index / 64] |= 1u64 << (index % 64);
            }
        }
        Self::from_words_le(&words)
    }

    /// Return the canonical binary prefix code.
    pub fn to_binary(&self) -> String {
        (0..self.bit_len())
            .rev()
            .map(|i| if self.bit(i) { '1' } else { '0' })
            .collect()
    }

    /// Number of encoded bits, including the zero terminal.
    pub fn bit_len(&self) -> usize {
        match self.words.last() {
            None => 1,
            Some(word) => (self.words.len() - 1) * 64 + (64 - word.leading_zeros() as usize),
        }
    }

    pub(crate) fn bit(&self, index: usize) -> bool {
        self.words
            .get(index / 64)
            .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
    }

    fn validate(&self) -> Result<(), EncodingError> {
        let mut slots = 1usize;
        for index in (0..self.bit_len()).rev() {
            if slots == 0 {
                return Err(EncodingError::TrailingBits);
            }
            slots -= 1;
            if self.bit(index) {
                slots = slots.checked_add(2).ok_or(EncodingError::SizeOverflow)?;
            }
        }
        if slots == 0 {
            Ok(())
        } else {
            Err(EncodingError::Truncated)
        }
    }

    /// Construct an ordered BML application, using the default size limit.
    pub fn op(&self, rhs: &Self) -> Result<Self, BuildError> {
        self.op_with_limit(rhs, DEFAULT_MAX_BITS)
    }

    /// Construct an ordered BML application with a maximum encoded bit count.
    pub fn op_with_limit(&self, rhs: &Self, max_bits: usize) -> Result<Self, BuildError> {
        let left_len = self.bit_len();
        let right_len = rhs.bit_len();
        let len = left_len
            .checked_add(right_len)
            .and_then(|n| n.checked_add(1))
            .ok_or(BuildError::SizeOverflow)?;
        if len > max_bits {
            return Err(BuildError::BitLimit {
                required: len,
                limit: max_bits,
            });
        }
        let count = len.checked_add(63).ok_or(BuildError::SizeOverflow)? / 64;
        let mut words = vec![0u64; count];
        words[(len - 1) / 64] |= 1u64 << ((len - 1) % 64);
        for i in 0..right_len {
            if rhs.bit(i) {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        for i in 0..left_len {
            if self.bit(i) {
                let pos = i + right_len;
                words[pos / 64] |= 1u64 << (pos % 64);
            }
        }
        Ok(Self { words })
    }

    /// Evaluate the term under the checked real BML interpretation.
    ///
    /// Construction is structural: intermediate domain or range errors can
    /// occur even for a well-formed term.
    pub fn to_f64(&self) -> Result<f64, EvalError> {
        let mut stack = Vec::new();
        for index in 0..self.bit_len() {
            if self.bit(index) {
                // Validated prefix codes always have two operands here.
                let left = stack.pop().expect("valid BML left operand");
                let right = stack.pop().expect("valid BML right operand");
                stack.push(Basis::Bml.apply_real(left, right)?);
            } else {
                stack.push(1.0);
            }
        }
        Ok(stack[0])
    }
}

impl fmt::Display for Bml {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_binary())
    }
}
