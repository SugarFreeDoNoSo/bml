use bml_domain::number::{
    Bml, CertifiedBml, ExactError, ExactInputError, ExactRational, ExactSynthesisOptions,
};

fn exact(n: i64) -> CertifiedBml {
    Bml::from_decimal_str_exact(&n.to_string()).unwrap()
}

#[test]
fn certified_arithmetic_covers_signs_zero_and_division() {
    let limits = ExactSynthesisOptions::default();
    for a in -2..=2 {
        for b in -2..=2 {
            let left = exact(a);
            let right = exact(b);
            let before = (left.clone(), right.clone());
            let sum = (&left + &right).unwrap();
            let difference = (&left - &right).unwrap();
            let product = (&left * &right).unwrap();
            assert_eq!(sum.target(), &ExactRational::from_i64(a + b));
            assert_eq!(difference.target(), &ExactRational::from_i64(a - b));
            assert_eq!(product.target(), &ExactRational::from_i64(a * b));
            for result in [&sum, &difference, &product] {
                result
                    .derivation()
                    .verify(result.term(), result.target(), limits)
                    .unwrap();
            }
            if b == 0 {
                assert_eq!(
                    &left / &right,
                    Err(ExactError::Rational(ExactInputError::DivisionByZero))
                );
            } else {
                let result = (&left / &right).unwrap();
                let expected = ExactRational::from_i64(a)
                    .checked_div(&ExactRational::from_i64(b), limits.rational)
                    .unwrap();
                assert_eq!(result.target(), &expected);
            }
            assert_eq!((left, right), before);
        }
    }
    assert_eq!((-&exact(3)).unwrap().target(), &ExactRational::from_i64(-3));
    assert_eq!((-exact(-3)).unwrap().target(), &ExactRational::from_i64(3));
}

#[test]
fn assignments_commit_only_complete_results() {
    let options = ExactSynthesisOptions::default();
    let mut value = exact(-2);
    let rhs = exact(3);
    let original = value.clone();
    let original_rhs = rhs.clone();
    let small = ExactSynthesisOptions {
        max_bits: 10,
        ..options
    };
    assert!(value.try_add_assign(&rhs, small).is_err());
    assert_eq!(value, original);
    assert!(value.try_mul_assign(&rhs, small).is_err());
    assert_eq!(value, original);
    value.try_add_assign(&rhs, options).unwrap();
    assert_eq!(value.target(), &ExactRational::from_i64(1));
    value.try_mul_assign(&rhs, options).unwrap();
    assert_eq!(value.target(), &ExactRational::from_i64(3));
    assert_eq!(rhs, original_rhs);
}

#[test]
fn mixed_operands_are_values_and_preserve_order() {
    let one = exact(1);
    for result in [
        (&one + 4i32).unwrap(),
        (4u64 + &one).unwrap(),
        (one.clone() + 4i128).unwrap(),
    ] {
        assert_eq!(result.target(), &ExactRational::from_i64(5));
    }
    assert_eq!(Bml::from_u64(4).unwrap().to_f64().unwrap(), 2.0);
    assert_eq!((4i64 - &one).unwrap().target(), &ExactRational::from_i64(3));
    assert_eq!(
        (&one - 4i64).unwrap().target(),
        &ExactRational::from_i64(-3)
    );
    assert_eq!((4u32 * &one).unwrap().target(), &ExactRational::from_i64(4));
    assert_eq!((4u16 / &one).unwrap().target(), &ExactRational::from_i64(4));
    let expected = ExactRational::from_f64(0.1, ExactSynthesisOptions::default().rational).unwrap();
    let result = (&exact(0) + 0.1f64).unwrap();
    assert_eq!(result.target(), &expected);
    assert_eq!(
        (&one + f64::INFINITY),
        Err(ExactError::Rational(ExactInputError::NonFinite))
    );
}

#[test]
fn checked_chains_preserve_certificates_and_enforce_proof_budget() {
    let options = ExactSynthesisOptions::default();
    let a = exact(1);
    let b = exact(2);
    let sum = (&a + &b).unwrap();
    let result = (&sum * &b).unwrap();
    assert_eq!(result.target(), &ExactRational::from_i64(6));
    let before = result.clone();
    let tiny = ExactSynthesisOptions {
        max_steps: 2,
        ..options
    };
    assert_eq!(result.checked_add(&b, tiny), Err(ExactError::Limit));
    assert_eq!(result.checked_neg(tiny), Err(ExactError::Limit));
    assert_eq!(result, before);
}
