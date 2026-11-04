use bml_domain::number::Bml;
use bml_kernels::{compare_codes, Backend, CompareError, CompareLimits};
use std::cmp::Ordering;
fn chain(operators: usize, right: bool) -> Bml {
    let mut value = Bml::one();
    for _ in 0..operators {
        value = if right {
            Bml::one().op(&value).unwrap()
        } else {
            value.op(&Bml::one()).unwrap()
        };
    }
    value
}
fn available() -> Vec<Backend> {
    #[cfg(target_arch = "aarch64")]
    assert!(Backend::Neon.available(), "AArch64 must exercise NEON");
    let backends: Vec<_> = [Backend::Scalar, Backend::Avx2, Backend::Neon]
        .into_iter()
        .filter(|b| b.available())
        .collect();
    eprintln!("comparison backends exercised: {backends:?}");
    backends
}
#[test]
fn vector_backends_match_independent_binary_string_order_including_tails() {
    let limits = CompareLimits::default();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for operators in [
        0, 1, 2, 3, 30, 31, 32, 63, 64, 95, 96, 127, 128, 159, 160, 255, 256, 511,
    ] {
        let a = chain(operators, false);
        let b = chain(operators, true);
        left.extend([a.clone(), a.clone(), b.clone(), a.clone()]);
        right.extend([a.clone(), b, a, chain(operators + 1, false)]);
    }
    let expected: Vec<_> = left
        .iter()
        .zip(&right)
        .map(|(a, b)| {
            let a = a.to_binary();
            let b = b.to_binary();
            a.len().cmp(&b.len()).then_with(|| a.cmp(&b))
        })
        .collect();
    for backend in available() {
        let results = compare_codes(&left, &right, backend, limits).unwrap();
        assert_eq!(results.backend, backend);
        assert_eq!(results.orderings, expected, "{backend:?}");
    }
    assert_eq!(
        compare_codes(&left, &right, Backend::Auto, limits)
            .unwrap()
            .backend,
        Backend::Scalar
    );
}
#[test]
fn every_word_difference_position_and_unsigned_order_are_preserved() {
    let limits = CompareLimits::default();
    // Move a small unequal subtree through a chain to exercise high/low words,
    // chunk boundaries, and scalar tails without forging invalid prefix codes.
    for depth in 0..260 {
        let mut a = Bml::from_u64(24).unwrap();
        let mut b = Bml::from_u64(20).unwrap();
        for _ in 0..depth {
            a = a.op(&Bml::one()).unwrap();
            b = b.op(&Bml::one()).unwrap();
        }
        let expected = a.to_binary().cmp(&b.to_binary());
        assert_eq!(expected, Ordering::Greater);
        for backend in available() {
            assert_eq!(
                compare_codes(
                    std::slice::from_ref(&a),
                    std::slice::from_ref(&b),
                    backend,
                    limits
                )
                .unwrap()
                .orderings,
                vec![expected]
            );
        }
    }
}
#[test]
fn structural_order_does_not_evaluate_signs_or_domains() {
    let limits = CompareLimits::default();
    let one = Bml::one();
    let zero = Bml::from_u64(88).unwrap();
    let invalid = one.op(&zero).unwrap();
    assert!(invalid.to_f64().is_err());
    let negative = Bml::from_decimal_str_exact("-1").unwrap().into_term();
    let values = vec![one, zero, negative, invalid];
    for backend in available() {
        assert_eq!(
            compare_codes(&values, &values, backend, limits)
                .unwrap()
                .orderings,
            vec![Ordering::Equal; 4]
        );
    }
    assert!(Bml::from_binary("1000").is_err());
}
#[test]
fn preflight_preserves_first_error_and_never_modifies_inputs() {
    let values = vec![Bml::one(); 3];
    let before = values.clone();
    let limits = CompareLimits::default();
    for backend in [Backend::Auto, Backend::Scalar, Backend::Avx2, Backend::Neon] {
        assert_eq!(
            compare_codes(&values, &values[..2], backend, limits),
            Err(CompareError::Length)
        );
        assert_eq!(
            compare_codes(
                &values,
                &values,
                backend,
                CompareLimits {
                    max_pairs: 2,
                    ..limits
                }
            ),
            Err(CompareError::PairLimit)
        );
        assert_eq!(
            compare_codes(
                &values,
                &values,
                backend,
                CompareLimits {
                    max_total_bits: 3,
                    ..limits
                }
            ),
            Err(CompareError::BitLimit { index: 1 })
        );
        if !backend.available() {
            assert_eq!(
                compare_codes(&values, &values, backend, limits),
                Err(CompareError::BackendUnavailable(backend))
            );
        }
    }
    for backend in available() {
        assert!(compare_codes(&[], &[], backend, limits)
            .unwrap()
            .orderings
            .is_empty());
    }
    assert_eq!(values, before);
}
