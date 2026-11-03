//! Experimental SIMD comparison of canonical integer tree encodings.
//!
//! The default API compares representation, not mathematical values.
//! The opt-in `tensor-simd` feature separately provides native IEEE f64 add/mul.
//! Neither API approximates exp2/log2 or rewrites symbolic trees.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
use bml_domain::number::Bml;
use std::{cmp::Ordering, fmt};

/// Opt-in native IEEE tensor kernels, separate from structural code comparison.
#[cfg(feature = "tensor-simd")]
pub mod tensor;

/// Requested backend. Forced unavailable backends return an error before work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable default until workload-specific performance evidence warrants promotion.
    Auto,
    /// Reference high-to-low unsigned word comparison.
    Scalar,
    /// x86/x86-64 AVX2 equality probes, four words at a time.
    Avx2,
    /// AArch64 NEON equality probes, two words at a time.
    Neon,
}
impl Backend {
    /// Runtime/target availability; Auto and Scalar are always supported.
    pub fn available(self) -> bool {
        match self {
            Self::Auto | Self::Scalar => true,
            Self::Avx2 => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    std::is_x86_feature_detected!("avx2")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            }
            Self::Neon => {
                #[cfg(target_arch = "aarch64")]
                {
                    std::arch::is_aarch64_feature_detected!("neon")
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    false
                }
            }
        }
    }
    fn resolve(self) -> Result<Self, CompareError> {
        if !self.available() {
            return Err(CompareError::BackendUnavailable(self));
        }
        Ok(if self == Self::Auto {
            Self::Scalar
        } else {
            self
        })
    }
}
/// Batch bounds checked before allocating any output or executing a vector kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompareLimits {
    /// Maximum number of input pairs.
    pub max_pairs: usize,
    /// Maximum logical bits in both inputs combined, counting repeated trees.
    pub max_total_bits: usize,
}
impl Default for CompareLimits {
    fn default() -> Self {
        Self {
            max_pairs: 65_536,
            max_total_bits: 16_777_216,
        }
    }
}
/// Preflight failures never return partial comparisons.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompareError {
    /// Left and right batch lengths differ.
    Length,
    /// Input pair count exceeds its budget.
    PairLimit,
    /// First input pair at which the cumulative bit budget is exceeded.
    BitLimit {
        /// Zero-based pair index.
        index: usize,
    },
    /// The forced instruction set is unavailable on this target/CPU.
    BackendUnavailable(Backend),
    /// Output allocation could not be reserved.
    Allocation,
}
impl fmt::Display for CompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length => f.write_str("code comparison batch lengths differ"),
            Self::PairLimit => f.write_str("code comparison pair limit exceeded"),
            Self::BitLimit { index } => write!(f, "code comparison bit limit at pair {index}"),
            Self::BackendUnavailable(backend) => {
                write!(f, "code comparison backend {backend:?} unavailable")
            }
            Self::Allocation => f.write_str("code comparison output allocation failed"),
        }
    }
}
impl std::error::Error for CompareError {}
/// Complete ordered comparison results and the backend actually used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comparisons {
    /// Backend selected after capability checks.
    pub backend: Backend,
    /// Unsigned integer code ordering for each original input pair.
    pub orderings: Vec<Ordering>,
}

/// Compare canonical integer codes in input order; unequal lengths are ordered
/// by bit length, equal lengths lexicographically from the most significant word.
/// Domain-invalid real terms are allowed: structural code comparison does not
/// certify their real interpretation. Inputs are already validated Bml grammar.
/// Preflight order: lengths, pair budget, first bit-budget failure, backend.
pub fn compare_codes(
    left: &[Bml],
    right: &[Bml],
    backend: Backend,
    limits: CompareLimits,
) -> Result<Comparisons, CompareError> {
    if left.len() != right.len() {
        return Err(CompareError::Length);
    }
    if left.len() > limits.max_pairs {
        return Err(CompareError::PairLimit);
    }
    let mut bits = 0usize;
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        bits = bits
            .checked_add(a.bit_len())
            .and_then(|n| n.checked_add(b.bit_len()))
            .ok_or(CompareError::BitLimit { index })?;
        if bits > limits.max_total_bits {
            return Err(CompareError::BitLimit { index });
        }
    }
    let backend = backend.resolve()?;
    let mut orderings = Vec::new();
    orderings
        .try_reserve_exact(left.len())
        .map_err(|_| CompareError::Allocation)?;
    for (a, b) in left.iter().zip(right) {
        let order = a.bit_len().cmp(&b.bit_len());
        orderings.push(if order != Ordering::Equal {
            order
        } else {
            dispatch(a.to_words_le(), b.to_words_le(), backend)
        });
    }
    Ok(Comparisons { backend, orderings })
}
fn scalar(a: &[u64], b: &[u64]) -> Ordering {
    for i in (0..a.len()).rev() {
        let order = a[i].cmp(&b[i]);
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}
fn dispatch(a: &[u64], b: &[u64], backend: Backend) -> Ordering {
    debug_assert_eq!(a.len(), b.len());
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2 => {
            // SAFETY: resolve checked AVX2; equal lengths and immutable live slices
            // are established by canonical bit lengths. Kernel checks each load.
            unsafe { avx2(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // SAFETY: resolve checked NEON; the kernel bounds every unaligned load.
            unsafe { neon(a, b) }
        }
        _ => scalar(a, b),
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2(a: &[u64], b: &[u64]) -> Ordering {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut end = a.len();
    while end >= 4 {
        let start = end - 4;
        // SAFETY: start..end lies within both equal-length slices, and loadu
        // requires no alignment. No read crosses a slice end; no pointers escape.
        let equal = unsafe {
            let x = _mm256_loadu_si256(a.as_ptr().add(start).cast());
            let y = _mm256_loadu_si256(b.as_ptr().add(start).cast());
            _mm256_movemask_epi8(_mm256_cmpeq_epi64(x, y)) == -1
        };
        if !equal {
            return scalar(&a[start..end], &b[start..end]);
        }
        end = start;
    }
    scalar(&a[..end], &b[..end])
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(a: &[u64], b: &[u64]) -> Ordering {
    use std::arch::aarch64::*;
    let mut end = a.len();
    while end >= 2 {
        let start = end - 2;
        // SAFETY: both live slices contain these two words; vld1q permits
        // unaligned addresses. Capability was checked before dispatch.
        let equal = unsafe {
            let mask = vceqq_u64(
                vld1q_u64(a.as_ptr().add(start)),
                vld1q_u64(b.as_ptr().add(start)),
            );
            vgetq_lane_u64::<0>(mask) == u64::MAX && vgetq_lane_u64::<1>(mask) == u64::MAX
        };
        if !equal {
            return scalar(&a[start..end], &b[start..end]);
        }
        end = start;
    }
    scalar(&a[..end], &b[..end])
}
