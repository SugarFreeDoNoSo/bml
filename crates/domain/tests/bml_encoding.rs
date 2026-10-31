use bml_domain::number::{Bml, BuildError, EncodingError};

#[test]
fn prefix_codes_preserve_operand_order_and_evaluate() {
    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let four = two.op(&one).unwrap();
    let another_one = one.op(&two).unwrap();
    for (value, text, integer, numeric) in [
        (&one, "0", 0, 1.0),
        (&two, "100", 4, 2.0),
        (&four, "11000", 24, 4.0),
        (&another_one, "10100", 20, 1.0),
    ] {
        assert_eq!(value.to_binary(), text);
        assert_eq!(value.to_u64(), Some(integer));
        assert_eq!(&Bml::from_binary(text).unwrap(), value);
        assert_eq!(&Bml::from_u64(integer).unwrap(), value);
        assert_eq!(value.to_f64(), Ok(numeric));
    }
}

#[test]
fn rejects_invalid_codes() {
    assert_eq!(Bml::from_binary(""), Err(EncodingError::Empty));
    assert_eq!(
        Bml::from_binary("10x"),
        Err(EncodingError::InvalidDigit { index: 2 })
    );
    assert_eq!(Bml::from_binary("00"), Err(EncodingError::LeadingZero));
    assert_eq!(Bml::from_binary("1"), Err(EncodingError::Truncated));
    assert_eq!(Bml::from_binary("1000"), Err(EncodingError::TrailingBits));
    assert_eq!(Bml::from_u64(3), Err(EncodingError::Truncated));
}

#[test]
fn large_codes_round_trip_across_words_and_normalize() {
    let one = Bml::one();
    let mut term = one.clone();
    for _ in 0..1000 {
        term = term.op(&one).unwrap();
    }
    assert_eq!(term.bit_len(), 2001);
    assert_eq!(term.to_u64(), None);
    assert_eq!(Bml::from_binary(&term.to_binary()).unwrap(), term);
    let mut words = term.to_words_le().to_vec();
    words.extend([0, 0]);
    assert_eq!(Bml::from_words_le(&words).unwrap(), term);
    assert_eq!(Bml::from_words_le(&[0, 0]).unwrap(), one);
}

#[test]
fn construction_enforces_the_requested_bit_limit() {
    let one = Bml::one();
    assert_eq!(
        one.op_with_limit(&one, 2),
        Err(BuildError::BitLimit {
            required: 3,
            limit: 2
        })
    );
    assert_eq!(one.op_with_limit(&one, 3).unwrap().to_binary(), "100");
}

#[test]
fn valid_structure_can_fail_real_evaluation() {
    use bml_domain::symbolic::EvalError;

    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let four = two.op(&one).unwrap();
    let zero = one.op(&four).unwrap();
    assert_eq!(zero.to_f64(), Ok(0.0));
    assert!(
        matches!(one.op(&zero).unwrap().to_f64(), Err(EvalError::LogDomain { value }) if value == 0.0)
    );
}
