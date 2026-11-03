use bml_domain::number::{Bml, ExactError, ExactSynthesisOptions};

#[test]
fn recover_exact_proofs_from_serialized_trees() {
    let options = ExactSynthesisOptions::default();
    for text in [
        "0", "1", "2", "3", "4", "15", "-3", "12.375", "0.0010", "-0.125", "0.1", "2049",
    ] {
        let original = Bml::from_decimal_str_exact(text).unwrap();
        let decoded = Bml::from_words_le(original.term().to_words_le()).unwrap();
        let recovered = decoded
            .certify_exact(options)
            .unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(recovered.term(), original.term());
        assert_eq!(recovered.target(), original.target());
    }
}

#[test]
fn recover_chained_expansions_with_signed_operands() {
    let options = ExactSynthesisOptions::default();
    let a = Bml::from_decimal_str_exact("-2").unwrap();
    let b = Bml::from_decimal_str_exact("3").unwrap();
    for result in [
        (&a + &b).unwrap(),
        (&a - &b).unwrap(),
        (&a * &b).unwrap(),
        (&a / &b).unwrap(),
        (-&a).unwrap(),
    ] {
        let recovered = result.term().certify_exact(options).unwrap();
        assert_eq!(recovered.target(), result.target());
    }
}

#[test]
fn unsupported_domains_and_budgets_do_not_gain_certificates() {
    let options = ExactSynthesisOptions::default();
    let zero = Bml::from_decimal_str_exact("0").unwrap().into_term();
    let invalid = Bml::one().op(&zero).unwrap();
    assert_eq!(
        invalid.certify_exact(options),
        Err(ExactError::DomainNotJustified)
    );
    let half = Bml::from_decimal_str_exact("0.5").unwrap().into_term();
    // sqrt(2) is real but outside the rational certificate rules.
    let irrational = half.exp2().unwrap();
    assert!(irrational.to_f64().unwrap().is_finite());
    assert_eq!(
        irrational.certify_exact(options),
        Err(ExactError::DomainNotJustified)
    );
    for tiny in [
        ExactSynthesisOptions {
            max_steps: 0,
            ..options
        },
        ExactSynthesisOptions {
            max_bits: 1,
            ..options
        },
        ExactSynthesisOptions {
            max_recursion: 0,
            ..options
        },
    ] {
        assert!(Bml::from_u64(24).unwrap().certify_exact(tiny).is_err());
    }
}

#[test]
fn raw_tree_operators_preserve_order_and_support_chaining() {
    use bml_domain::number::ExactRational;
    let options = ExactSynthesisOptions::default();
    let a = Bml::from_decimal_str_exact("-2").unwrap().into_term();
    let b = Bml::from_decimal_str_exact("3").unwrap().into_term();
    let originals = (a.clone(), b.clone());
    let check = |value: Bml, n: i64| {
        assert_eq!(
            value.certify_exact(options).unwrap().target(),
            &ExactRational::from_i64(n)
        )
    };
    check((&a + &b).unwrap(), 1);
    check((a.clone() + &b).unwrap(), 1);
    check((&a + b.clone()).unwrap(), 1);
    check((a.clone() + b.clone()).unwrap(), 1);
    check((&a - &b).unwrap(), -5);
    check((&a * &b).unwrap(), -6);
    check((-&a).unwrap(), 2);
    check((-a.clone()).unwrap(), 2);
    check((&a + 4i32).unwrap(), 2);
    check((4u64 + &a).unwrap(), 2);
    check((4i64 - &a).unwrap(), 6);
    check((4u32 * &a).unwrap(), -8);
    check((4i16 / &a).unwrap(), -2);
    let sum = (&a + &b).unwrap();
    check((&sum * &b).unwrap(), 3);
    assert_eq!((a, b), originals);
}

#[test]
fn raw_tree_assignment_failures_are_transactional() {
    use bml_domain::number::ExactInputError;
    let options = ExactSynthesisOptions::default();
    let mut a = Bml::from_u64(4).unwrap();
    let b = Bml::from_u64(24).unwrap();
    let before = a.clone();
    let small = ExactSynthesisOptions {
        max_bits: 5,
        ..options
    };
    assert!(a.try_add_assign(&b, small).is_err());
    assert_eq!(a, before);
    assert!(a.try_mul_assign(&b, small).is_err());
    assert_eq!(a, before);
    let zero = Bml::from_u64(88).unwrap();
    assert_eq!(
        &a / &zero,
        Err(ExactError::Rational(ExactInputError::DivisionByZero))
    );
    let invalid = Bml::one().op(&zero).unwrap();
    assert_eq!(
        a.try_add_assign(&invalid, options),
        Err(ExactError::DomainNotJustified)
    );
    assert_eq!(a, before);
    assert_eq!(
        a.try_mul_assign(&invalid, options),
        Err(ExactError::DomainNotJustified)
    );
    assert_eq!(a, before);
    a.try_add_assign(&Bml::one(), options).unwrap();
    assert_eq!(
        a.certify_exact(options)
            .unwrap()
            .target()
            .numerator_words_le(),
        &[3]
    );
    a.try_mul_assign(&Bml::one(), options).unwrap();
    assert_eq!(
        a.certify_exact(options)
            .unwrap()
            .target()
            .numerator_words_le(),
        &[3]
    );
    assert_eq!(b.to_u64(), Some(24));
}

#[test]
fn structural_methods_remain_distinct_from_checked_operators() {
    let options = ExactSynthesisOptions::default();
    let one = Bml::one();
    let zero = Bml::from_u64(88).unwrap();
    let invalid = one.op(&zero).unwrap();
    // Structural methods do not gain implicit domain checks.
    assert!(invalid.add(&one).is_ok());
    assert_eq!(&invalid + &one, Err(ExactError::DomainNotJustified));
    let sum = (&one + &one).unwrap();
    assert_ne!(sum, one.op(&one).unwrap());
    assert_eq!(
        sum.certify_exact(options)
            .unwrap()
            .target()
            .numerator_words_le(),
        &[2]
    );
    assert!((&one + f64::NAN).is_err());
}
