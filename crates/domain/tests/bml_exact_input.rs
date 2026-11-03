use bml_domain::number::{ExactInputError, ExactInputLimits, ExactRational};

fn small(value: &ExactRational) -> (bool, u128, u128) {
    let decode = |words: &[u64]| {
        assert!(words.len() <= 2);
        words
            .iter()
            .rev()
            .fold(0u128, |n, w| (n << 64) | *w as u128)
    };
    (
        value.is_negative(),
        decode(value.numerator_words_le()),
        decode(value.denominator_words_le()),
    )
}

#[test]
fn decimal_targets_are_reduced_without_float_rounding() {
    let limits = ExactInputLimits::default();
    for (text, negative, n, d) in [
        ("12.375", false, 99, 8),
        ("0.0010", false, 1, 1000),
        ("-0.125", true, 1, 8),
        ("-0.000", false, 0, 1),
        ("+00012.3750", false, 99, 8),
        ("0.1", false, 1, 10),
        ("18446744073709551616", false, 1u128 << 64, 1),
        (
            "340282366920938463463374607431768211455",
            false,
            u128::MAX,
            1,
        ),
    ] {
        assert_eq!(
            small(&ExactRational::from_decimal_str(text, limits).unwrap()),
            (negative, n, d)
        );
    }
    let decimal = ExactRational::from_decimal_str("0.1", limits).unwrap();
    let float = ExactRational::from_f64(0.1, limits).unwrap();
    assert_ne!(decimal, float);
    assert_eq!(small(&float), (false, 3602879701896397, 36028797018963968));
}

#[test]
fn ieee_fields_cover_subnormals_normals_and_zero() {
    let limits = ExactInputLimits::default();
    for sign in [0, 1u64 << 63] {
        let smallest = ExactRational::from_f64(f64::from_bits(sign | 1), limits).unwrap();
        assert_eq!(smallest.is_negative(), sign != 0);
        assert_eq!(smallest.numerator_words_le(), &[1]);
        assert_eq!(smallest.denominator_words_le().len(), 17);
        assert!(smallest.denominator_words_le()[..16]
            .iter()
            .all(|w| *w == 0));
        assert_eq!(smallest.denominator_words_le()[16], 1u64 << 50);
        let zero = ExactRational::from_f64(f64::from_bits(sign), limits).unwrap();
        assert_eq!(small(&zero), (false, 0, 1));
    }
    let min_normal = ExactRational::from_f64(f64::MIN_POSITIVE, limits).unwrap();
    assert_eq!(min_normal.numerator_words_le(), &[1]);
    assert_eq!(min_normal.denominator_words_le().len(), 16);
    assert_eq!(min_normal.denominator_words_le()[15], 1u64 << 62);
    let max = ExactRational::from_f64(f64::MAX, limits).unwrap();
    assert_eq!(max.denominator_words_le(), &[1]);
    assert_eq!(max.numerator_words_le().len(), 16);
    assert!(max.numerator_words_le()[..15].iter().all(|w| *w == 0));
    assert_eq!(max.numerator_words_le()[15], u64::MAX << 11);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            ExactRational::from_f64(value, limits),
            Err(ExactInputError::NonFinite)
        );
    }
}

#[test]
fn grammar_and_each_resource_budget_are_enforced() {
    let limits = ExactInputLimits::default();
    for input in [
        "", " ", " 1", "1 ", "1e-3", "12,375", ".5", "1.", "+", "--1", "1..0", "１２", "1_000",
    ] {
        assert_eq!(
            ExactRational::from_decimal_str(input, limits),
            Err(ExactInputError::Syntax),
            "{input}"
        );
    }
    assert_eq!(
        ExactRational::from_decimal_str(
            "0.0010",
            ExactInputLimits {
                max_digits: 4,
                ..limits
            }
        ),
        Err(ExactInputError::Limit)
    );
    assert_eq!(
        ExactRational::from_decimal_str(
            "256",
            ExactInputLimits {
                max_bits: 8,
                ..limits
            }
        ),
        Err(ExactInputError::Limit)
    );
    // The unreduced denominator exceeds the budget, despite a zero target.
    assert_eq!(
        ExactRational::from_decimal_str(
            "0.000",
            ExactInputLimits {
                max_bits: 8,
                ..limits
            }
        ),
        Err(ExactInputError::Limit)
    );
    assert_eq!(
        ExactRational::from_decimal_str(
            "12.375",
            ExactInputLimits {
                max_gcd_steps: 0,
                ..limits
            }
        ),
        Err(ExactInputError::Limit)
    );
    assert_eq!(
        ExactRational::from_f64(
            f64::from_bits(1),
            ExactInputLimits {
                max_bits: 1074,
                ..limits
            }
        ),
        Err(ExactInputError::Limit)
    );
    assert!(ExactRational::from_f64(
        f64::from_bits(1),
        ExactInputLimits {
            max_bits: 1075,
            ..limits
        }
    )
    .is_ok());
}

#[test]
fn generated_decimals_match_independent_euclidean_u128_oracle() {
    let limits = ExactInputLimits::default();
    for n in (0..2000u128).chain([u64::MAX as u128, 1u128 << 64, (1u128 << 100) + 77]) {
        for digits in 0..=6 {
            let denominator = 10u128.pow(digits);
            let text = if digits == 0 {
                n.to_string()
            } else {
                format!(
                    "{}.{:0width$}",
                    n / denominator,
                    n % denominator,
                    width = digits as usize
                )
            };
            let (mut a, mut b) = (n, denominator);
            while b != 0 {
                (a, b) = (b, a % b);
            }
            assert_eq!(
                small(&ExactRational::from_decimal_str(&text, limits).unwrap()),
                (false, n / a, denominator / a),
                "{text}"
            );
        }
    }
}

#[test]
fn large_decimal_reduction_crosses_many_word_boundaries() {
    let limits = ExactInputLimits::default();
    let text = format!("1.{}", "0".repeat(300));
    assert_eq!(
        small(&ExactRational::from_decimal_str(&text, limits).unwrap()),
        (false, 1, 1)
    );
    let text = format!("0.{}125", "0".repeat(297));
    let rational = ExactRational::from_decimal_str(&text, limits).unwrap();
    assert_eq!(rational.numerator_words_le(), &[1]);
    // 10^300 / 125 = 8 * 10^297, calculated independently by decimal parsing.
    let expected =
        ExactRational::from_decimal_str(&format!("8{}", "0".repeat(297)), limits).unwrap();
    assert_eq!(
        rational.denominator_words_le(),
        expected.numerator_words_le()
    );
}

#[test]
fn signed_decimal_byte_preflight_accepts_the_exact_digit_budget() {
    let limits = ExactInputLimits {
        max_digits: 2,
        ..ExactInputLimits::default()
    };
    for text in ["+1.0", "-1.0"] {
        let parsed = ExactRational::from_decimal_str(text, limits).unwrap();
        assert_eq!(
            parsed,
            ExactRational::from_i64(if text.starts_with('-') { -1 } else { 1 })
        );
    }
    for text in ["+1.00", "-1.00", "+invalid"] {
        assert_eq!(
            ExactRational::from_decimal_str(text, limits),
            Err(ExactInputError::Limit)
        );
    }
    assert_eq!(
        ExactRational::from_decimal_str("+1.x", limits),
        Err(ExactInputError::Syntax)
    );
}

#[test]
fn input_failures_have_readable_diagnostics_without_fabricated_causes() {
    use std::error::Error;
    for error in [
        ExactInputError::NonIntegralExponent,
        ExactInputError::DivisionByZero,
        ExactInputError::Syntax,
        ExactInputError::NonFinite,
        ExactInputError::Limit,
    ] {
        assert!(!error.to_string().trim().is_empty());
        assert!(error.source().is_none());
    }
}

#[test]
fn decimal_syntax_and_digit_limit_have_stable_failure_precedence() {
    let limits = ExactInputLimits {
        max_digits: 1,
        ..ExactInputLimits::default()
    };
    // Both fit the coarse byte preflight. A leading point is invalid before
    // scanning any digits, whereas a valid prefix can exhaust the digit budget.
    assert_eq!(
        ExactRational::from_decimal_str(".12", limits),
        Err(ExactInputError::Syntax)
    );
    assert_eq!(
        ExactRational::from_decimal_str("12", limits),
        Err(ExactInputError::Limit)
    );
    assert_eq!(
        ExactRational::from_decimal_str("1", limits).unwrap(),
        ExactRational::from_i64(1)
    );
    // Leading zeros count toward input work even when reduction yields zero.
    assert_eq!(
        ExactRational::from_decimal_str("00", limits),
        Err(ExactInputError::Limit)
    );
}
