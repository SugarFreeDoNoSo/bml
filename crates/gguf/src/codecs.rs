//! Explicit little-endian tensor decoding, independent of inference.

use crate::{GgufDataType, GgufError};

/// Decode exactly `elements` stored values into a new `f32` buffer.
///
/// Supports F32, F16, Q4_0 and Q8_0. Quantized data must contain complete
/// 32-element blocks; Q4_0 stores the low 16 nibbles before the high 16.
/// The input must have exactly the expected length: neither trailing bytes
/// nor truncated blocks are accepted. Empty input with zero elements is valid.
///
/// This is an explicit allocation and, for quantized types, dequantization.
/// F32 values retain their bits. F16 values widen exactly, including signed
/// zero, infinities and NaN sign/payload bits. Quantized multiplication follows
/// ordinary IEEE `f32` arithmetic, including non-finite scales.
///
/// # Errors
///
/// Returns a typed error for unsupported types, incomplete block element
/// counts, length arithmetic overflow, an incorrect byte length or allocation
/// failure. No partial output is returned.
pub fn decode_f32(
    data_type: GgufDataType,
    data: &[u8],
    elements: usize,
) -> Result<Vec<f32>, GgufError> {
    let (block_elements, block_bytes) = match data_type {
        GgufDataType::F32 => (1, 4),
        GgufDataType::F16 => (1, 2),
        GgufDataType::Q4_0 => (32, 18),
        GgufDataType::Q8_0 => (32, 34),
        _ => return Err(GgufError::UnsupportedDataType(data_type)),
    };
    if !elements.is_multiple_of(block_elements) {
        return Err(GgufError::InvalidBlockElements {
            data_type,
            elements,
            block_elements,
        });
    }
    let expected = (elements / block_elements)
        .checked_mul(block_bytes)
        .ok_or(GgufError::SizeOverflow)?;
    if data.len() != expected {
        return Err(GgufError::InvalidDataLength {
            expected,
            actual: data.len(),
        });
    }

    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| GgufError::Allocation)?;
    match data_type {
        GgufDataType::F32 => {
            values.extend(
                data.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
        }
        GgufDataType::F16 => {
            values.extend(
                data.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            );
        }
        GgufDataType::Q4_0 => {
            for block in data.as_chunks::<18>().0 {
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for shift in [0, 4] {
                    values.extend(
                        block[2..]
                            .iter()
                            .map(|byte| (((byte >> shift) & 15) as i32 - 8) as f32 * scale),
                    );
                }
            }
        }
        GgufDataType::Q8_0 => {
            for block in data.as_chunks::<34>().0 {
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                values.extend(block[2..].iter().map(|&q| q as i8 as f32 * scale));
            }
        }
        _ => unreachable!("the data type was validated before allocation"),
    }
    Ok(values)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let magnitude = match exponent {
        // All binary16 subnormals widen exactly; this multiplication cannot
        // underflow f32 and also handles zero without losing its sign.
        0 => ((mantissa as f32) * f32::from_bits(0x3380_0000)).to_bits(),
        31 => 0x7f80_0000 + (mantissa << 13),
        _ => ((u32::from(exponent) + 112) << 23) + (mantissa << 13),
    };
    // Sign, exponent and mantissa occupy disjoint fields; adding them is exact.
    f32::from_bits(sign + magnitude)
}
