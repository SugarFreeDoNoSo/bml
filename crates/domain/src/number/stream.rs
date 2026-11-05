//! Version 1: `BMLS`, version u8=1, basis u8=2 (BML), two reserved zero
//! bytes, then count and useful bit length as little-endian u64, then MSB-first
//! packed prefix codes. Low padding bits must be zero; no trailing bytes.
use super::{Bml, DEFAULT_MAX_BITS};
use std::fmt;

/// Resource limits checked before allocating decoded terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamLimits {
    /// Total useful bits across all terms.
    pub max_bits: usize,
    /// Number of terms, including single leaves.
    pub max_terms: usize,
    /// Maximum nesting in each tree (a leaf has depth zero).
    pub max_depth: usize,
}

impl Default for StreamLimits {
    fn default() -> Self {
        Self {
            max_bits: DEFAULT_MAX_BITS,
            max_terms: 65_536,
            max_depth: 4096,
        }
    }
}

/// Malformed or over-budget stream, or a fixed-word export that would truncate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamError {
    /// Invalid magic, reserved fields, version, or interpretation base.
    Header,
    /// Byte count, tree count, or useful bit count is inconsistent.
    Length,
    /// Nonzero low padding bits.
    Padding,
    /// A configured resource limit or platform size was exceeded.
    Limit,
    /// The integer encoding does not fit the requested word width.
    WordWidth,
}
impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Header => "unsupported BML stream header",
            Self::Length => "inconsistent BML stream length or tree count",
            Self::Padding => "noncanonical BML stream padding",
            Self::Limit => "BML stream resource limit exceeded",
            Self::WordWidth => "BML encoding exceeds fixed word width",
        })
    }
}
impl std::error::Error for StreamError {}

impl Bml {
    /// Export the code without truncation; this is not the numerical value.
    pub fn try_to_u64(&self) -> Result<u64, StreamError> {
        self.to_u64().ok_or(StreamError::WordWidth)
    }

    /// Export the code without truncation; at most 63 operators fit.
    pub fn try_to_u128(&self) -> Result<u128, StreamError> {
        if self.words.len() > 2 {
            return Err(StreamError::WordWidth);
        }
        Ok(self.words.first().copied().unwrap_or(0) as u128
            | ((self.words.get(1).copied().unwrap_or(0) as u128) << 64))
    }

    /// Number of operators in the fully expanded tree, including duplicates.
    pub fn operator_count(&self) -> usize {
        (self.bit_len() - 1) / 2
    }

    /// Serialize trees with explicit interpretation, lengths and resource limits.
    pub fn encode_stream(values: &[Self], limits: StreamLimits) -> Result<Vec<u8>, StreamError> {
        if values.len() > limits.max_terms {
            return Err(StreamError::Limit);
        }
        let bits = values.iter().try_fold(0usize, |n, v| {
            n.checked_add(v.bit_len()).ok_or(StreamError::Limit)
        })?;
        if bits > limits.max_bits {
            return Err(StreamError::Limit);
        }
        // Validate depth before allocating the output, including for imported trees.
        for value in values {
            check_depth(
                (0..value.bit_len()).rev().map(|i| value.bit(i)),
                limits.max_depth,
            )?;
        }
        let size = bits.checked_add(7).ok_or(StreamError::Limit)? / 8;
        let total = size.checked_add(24).ok_or(StreamError::Limit)?;
        let mut out = Vec::new();
        out.try_reserve_exact(total)
            .map_err(|_| StreamError::Limit)?;
        out.extend_from_slice(b"BMLS\x01\x02\0\0");
        out.extend_from_slice(
            &u64::try_from(values.len())
                .map_err(|_| StreamError::Limit)?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u64::try_from(bits)
                .map_err(|_| StreamError::Limit)?
                .to_le_bytes(),
        );
        out.resize(total, 0);
        let mut cursor = 0;
        for value in values {
            for i in (0..value.bit_len()).rev() {
                if value.bit(i) {
                    out[24 + cursor / 8] |= 1 << (7 - cursor % 8);
                }
                cursor += 1;
            }
        }
        Ok(out)
    }

    /// Decode a complete canonical stream, rejecting malformed data before
    /// allocating the result. This validates structure, not real domains.
    pub fn decode_stream(bytes: &[u8], limits: StreamLimits) -> Result<Vec<Self>, StreamError> {
        if bytes.len() < 24 {
            return Err(StreamError::Length);
        }
        if &bytes[..8] != b"BMLS\x01\x02\0\0" {
            return Err(StreamError::Header);
        }
        let count = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap()))
            .map_err(|_| StreamError::Limit)?;
        let bits = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
            .map_err(|_| StreamError::Limit)?;
        if count > limits.max_terms || bits > limits.max_bits {
            return Err(StreamError::Limit);
        }
        let size = bits.checked_add(7).ok_or(StreamError::Limit)? / 8;
        if size.checked_add(24) != Some(bytes.len()) || count > bits {
            return Err(StreamError::Length);
        }
        if bits % 8 != 0 && bytes[bytes.len() - 1] & ((1 << (8 - bits % 8)) - 1) != 0 {
            return Err(StreamError::Padding);
        }
        let bit = |i: usize| bytes[24 + i / 8] & (1 << (7 - i % 8)) != 0;
        // First pass checks every tree and nesting before output allocation.
        let mut cursor = 0;
        for _ in 0..count {
            let consumed = check_depth((cursor..bits).map(bit), limits.max_depth)?;
            cursor += consumed;
        }
        if cursor != bits {
            return Err(StreamError::Length);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| StreamError::Limit)?;
        cursor = 0;
        for _ in 0..count {
            let start = cursor;
            let mut slots = 1usize;
            // The first pass proved this tree closes within the remaining
            // bits. Walk that finite range independently of the output cursor.
            for index in start..bits {
                slots -= 1;
                if bit(index) {
                    slots += 2;
                }
                cursor = index + 1;
                if slots == 0 {
                    break;
                }
            }
            let length = cursor - start;
            let word_count = length.div_ceil(64);
            let mut words = Vec::new();
            words
                .try_reserve_exact(word_count)
                .map_err(|_| StreamError::Limit)?;
            words.resize(word_count, 0);
            for i in 0..length {
                if bit(start + i) {
                    let pos = length - 1 - i;
                    words[pos / 64] |= 1 << (pos % 64);
                }
            }
            if length == 1 {
                words.clear();
            }
            values.push(Self { words });
        }
        Ok(values)
    }
}

// Pending right-child depths represent the traversal stack, bounded by depth.
fn check_depth(bits: impl Iterator<Item = bool>, max_depth: usize) -> Result<usize, StreamError> {
    let mut pending = Vec::new();
    let mut depth = 0usize;
    for (i, op) in bits.enumerate() {
        if op {
            depth = depth.checked_add(1).ok_or(StreamError::Limit)?;
            if depth > max_depth {
                return Err(StreamError::Limit);
            }
            pending.try_reserve(1).map_err(|_| StreamError::Limit)?;
            pending.push(depth);
        } else if let Some(next) = pending.pop() {
            depth = next;
        } else {
            return Ok(i + 1);
        }
    }
    Err(StreamError::Length)
}
