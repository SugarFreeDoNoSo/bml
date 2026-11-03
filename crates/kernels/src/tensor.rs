//! Contiguous f64 add/mul under the native tensor IEEE contract.
//! No reassociation, reduction, fused arithmetic or exp/log approximation.
use bml_domain::tensor::{self, Tensor, TensorError, TensorView};

/// Explicit execution policy; the reference remains the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Policy {
    /// Use the domain reference implementation on every CPU.
    #[default]
    Reference,
    /// Use AVX2 if present, otherwise fall back to the domain reference.
    PreferAvx2,
}
/// Backend actually used, including fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutedBackend {
    /// Portable domain reference kernel.
    Reference,
    /// AVX2 four-lane arithmetic with a scalar tail.
    Avx2,
}
/// Complete output and the selected backend. Inputs are never modified.
#[derive(Debug)]
pub struct Output {
    /// Result with the original input shape.
    pub tensor: Tensor<f64>,
    /// Observed backend selection, not merely the requested preference.
    pub backend: ExecutedBackend,
}
fn select(policy: Policy, avx2: bool) -> ExecutedBackend {
    if policy == Policy::PreferAvx2 && avx2 {
        ExecutedBackend::Avx2
    } else {
        ExecutedBackend::Reference
    }
}
/// Elementwise addition; mismatched shapes fail before allocating output.
pub fn add(
    left: &TensorView<'_, f64>,
    right: &TensorView<'_, f64>,
    policy: Policy,
) -> Result<Output, TensorError> {
    apply(left, right, policy, false)
}
/// Elementwise multiplication; no broadcasting or numerical-domain restriction.
pub fn mul(
    left: &TensorView<'_, f64>,
    right: &TensorView<'_, f64>,
    policy: Policy,
) -> Result<Output, TensorError> {
    apply(left, right, policy, true)
}
fn apply(
    left: &TensorView<'_, f64>,
    right: &TensorView<'_, f64>,
    policy: Policy,
    multiply: bool,
) -> Result<Output, TensorError> {
    if left.shape() != right.shape() {
        return Err(TensorError::ShapeMismatch {
            left: left.shape().dims().to_vec(),
            right: right.shape().dims().to_vec(),
        });
    }
    let backend = select(policy, super::Backend::Avx2.available());
    if backend == ExecutedBackend::Reference {
        let tensor = if multiply {
            tensor::mul(left, right)?
        } else {
            tensor::add(left, right)?
        };
        return Ok(Output { tensor, backend });
    }
    let a = left.as_slice();
    let mut values = Vec::new();
    values
        .try_reserve_exact(a.len())
        .map_err(|_| TensorError::AllocationFailed)?;
    values.resize(a.len(), 0.0);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: availability was checked above; validated views have equal shape
    // and length. Output is a separate allocation with exactly that length.
    unsafe {
        avx2(a, right.as_slice(), &mut values, multiply)
    };
    Ok(Output {
        tensor: Tensor::from_vec(left.shape().clone(), values)?,
        backend,
    })
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2(a: &[f64], b: &[f64], output: &mut [f64], multiply: bool) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut i = 0;
    while a.len() - i >= 4 {
        // SAFETY: all three slices have equal lengths, i..i+4 is in bounds,
        // loadu/storeu require no alignment, and output cannot alias inputs.
        unsafe {
            let x = _mm256_loadu_pd(a.as_ptr().add(i));
            let y = _mm256_loadu_pd(b.as_ptr().add(i));
            let value = if multiply {
                _mm256_mul_pd(x, y)
            } else {
                _mm256_add_pd(x, y)
            };
            _mm256_storeu_pd(output.as_mut_ptr().add(i), value);
        }
        i += 4;
    }
    for j in i..a.len() {
        output[j] = if multiply { a[j] * b[j] } else { a[j] + b[j] };
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unavailable_avx2_explicitly_falls_back() {
        assert_eq!(
            select(Policy::PreferAvx2, false),
            ExecutedBackend::Reference
        );
        assert_eq!(select(Policy::Reference, true), ExecutedBackend::Reference);
        assert_eq!(select(Policy::PreferAvx2, true), ExecutedBackend::Avx2);
    }
}
