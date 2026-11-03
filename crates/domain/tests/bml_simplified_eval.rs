use bml_domain::number::{
    Bml, ExactError, ExactInputLimits, ExactRational, ExactSynthesisOptions, RationalFloatError,
    SimplifiedEvalError,
};

#[test]
fn proven_simplification_avoids_overflow_without_changing_direct_evaluation() {
    let options = ExactSynthesisOptions::default();
    let value = Bml::from_decimal_str_exact("2049").unwrap();
    assert!(value.term().to_f64().is_err());
    assert_eq!(value.to_f64_simplified(options), Ok(2049.0));
    assert_eq!(value.term().to_f64_simplified(options), Ok(2049.0));
    assert!(value.term().to_f64().is_err());
    let decimal = Bml::from_decimal_str_exact("0.1").unwrap();
    assert_eq!(
        decimal.to_f64_simplified(options).unwrap().to_bits(),
        0.1f64.to_bits()
    );
    let small = ExactSynthesisOptions {
        max_steps: 0,
        ..options
    };
    assert!(matches!(
        value.to_f64_simplified(small),
        Err(SimplifiedEvalError::Proof(ExactError::Limit))
    ));
}

#[test]
fn round_trip_finite_ieee_values_including_subnormals() {
    let limits = ExactInputLimits::default();
    let fixed = [
        0,
        1,
        2,
        (1u64 << 52) - 1,
        1u64 << 52,
        f64::MAX.to_bits(),
        1u64 << 63,
        (1u64 << 63) | 1,
        1f64.to_bits(),
        (-1f64).to_bits(),
    ];
    let mut state = 0x1234_5678_9abc_def0u64;
    for bits in fixed.into_iter().chain((0..2048).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state
    })) {
        let original = f64::from_bits(bits);
        if !original.is_finite() {
            continue;
        }
        let target = ExactRational::from_f64(original, limits).unwrap();
        let actual = target.to_f64_rounded(limits).unwrap();
        assert_eq!(
            actual.to_bits(),
            if original == 0.0 { 0 } else { bits },
            "bits {bits:x}"
        );
    }
}

#[test]
fn ties_use_even_significands_and_decimal_oracle_agrees() {
    let limits = ExactInputLimits::default();
    for (text, expected) in [
        (
            "1.00000000000000011102230246251565404236316680908203125",
            1f64.to_bits(),
        ),
        (
            "1.00000000000000033306690738754696212708950042724609375",
            1f64.to_bits() + 2,
        ),
        (
            "-1.00000000000000011102230246251565404236316680908203125",
            (-1f64).to_bits(),
        ),
        ("0.1", 0.1f64.to_bits()),
    ] {
        let target = ExactRational::from_decimal_str(text, limits).unwrap();
        assert_eq!(target.to_f64_rounded(limits).unwrap().to_bits(), expected);
        assert_eq!(text.parse::<f64>().unwrap().to_bits(), expected);
    }
}

#[test]
fn independent_hardware_division_oracle_for_exactly_representable_integer_inputs() {
    let limits = ExactInputLimits::default();
    let mut state = 7u64;
    for _ in 0..2000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let a = (state >> 40) as i64;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b = ((state >> 40) + 1) as i64;
        let target = ExactRational::from_i64(a)
            .checked_div(&ExactRational::from_i64(b), limits)
            .unwrap();
        assert_eq!(
            target.to_f64_rounded(limits).unwrap().to_bits(),
            ((a as f64) / (b as f64)).to_bits()
        );
    }
}

#[test]
fn normal_subnormal_overflow_and_underflow_boundaries_are_explicit() {
    let limits = ExactInputLimits::default();
    let pow = |e| ExactRational::from_i64(e).checked_exp2(limits).unwrap();
    let half_min = pow(-1075);
    assert_eq!(
        half_min.to_f64_rounded(limits),
        Err(RationalFloatError::Underflow)
    );
    let three_halves = half_min
        .checked_mul(&ExactRational::from_i64(3), limits)
        .unwrap();
    assert_eq!(three_halves.to_f64_rounded(limits).unwrap().to_bits(), 2);
    let boundary = pow(-1022).checked_sub(&half_min, limits).unwrap();
    assert_eq!(
        boundary.to_f64_rounded(limits).unwrap().to_bits(),
        f64::MIN_POSITIVE.to_bits()
    );
    let maximum = ExactRational::from_f64(f64::MAX, limits).unwrap();
    let overflow_tie = maximum.checked_add(&pow(970), limits).unwrap();
    assert_eq!(
        overflow_tie.to_f64_rounded(limits),
        Err(RationalFloatError::Overflow)
    );
    assert_eq!(
        overflow_tie
            .checked_sub(&ExactRational::from_i64(1), limits)
            .unwrap()
            .to_f64_rounded(limits),
        Ok(f64::MAX)
    );
    assert_eq!(
        pow(1024).to_f64_rounded(limits),
        Err(RationalFloatError::Overflow)
    );
    assert_eq!(
        pow(-1076).to_f64_rounded(limits),
        Err(RationalFloatError::Underflow)
    );
    assert_eq!(
        ExactRational::from_i64(1).to_f64_rounded(ExactInputLimits {
            max_bits: 52,
            ..limits
        }),
        Err(RationalFloatError::Limit)
    );
}

#[test]
fn simplified_failures_preserve_diagnostic_and_typed_error_chain() {
    use std::error::Error;
    let certificate = Bml::from_decimal_str_exact("1").unwrap();
    let error = certificate
        .to_f64_simplified(ExactSynthesisOptions {
            max_steps: 0,
            ..ExactSynthesisOptions::default()
        })
        .unwrap_err();
    assert!(!error.to_string().trim().is_empty());
    assert_eq!(
        error.source().unwrap().downcast_ref::<ExactError>(),
        Some(&ExactError::Limit)
    );
    for cause in [
        RationalFloatError::Limit,
        RationalFloatError::Overflow,
        RationalFloatError::Underflow,
    ] {
        let wrapper = SimplifiedEvalError::Rounding(cause.clone());
        assert!(!wrapper.to_string().trim().is_empty());
        assert_eq!(
            wrapper
                .source()
                .unwrap()
                .downcast_ref::<RationalFloatError>(),
            Some(&cause)
        );
    }
}

#[test]
fn rounding_carries_from_the_last_significand_bit_into_the_exponent() {
    let limits = ExactInputLimits::default();
    let half_ulp = ExactRational::from_i64(-53).checked_exp2(limits).unwrap();
    let tie = ExactRational::from_i64(2)
        .checked_sub(&half_ulp, limits)
        .unwrap();
    assert_eq!(
        tie.to_f64_rounded(limits).unwrap().to_bits(),
        2.0_f64.to_bits()
    );
    let below = tie
        .checked_sub(
            &ExactRational::from_i64(-54).checked_exp2(limits).unwrap(),
            limits,
        )
        .unwrap();
    assert_eq!(
        below.to_f64_rounded(limits).unwrap().to_bits(),
        2.0_f64.to_bits() - 1
    );
}

#[test]
fn rounding_estimated_exponent_boundaries_do_not_reject_finite_results() {
    let limits = ExactInputLimits::default();
    let pow = |n| ExactRational::from_i64(n).checked_exp2(limits).unwrap();
    let three = ExactRational::from_i64(3);
    let large = pow(1025).checked_div(&three, limits).unwrap();
    assert_eq!(
        large.to_f64_rounded(limits).unwrap().to_bits(),
        0x7fe5555555555555
    );
    let tiny = pow(-1076).checked_mul(&three, limits).unwrap();
    assert_eq!(tiny.to_f64_rounded(limits).unwrap().to_bits(), 1);
    for error in [
        RationalFloatError::Limit,
        RationalFloatError::Overflow,
        RationalFloatError::Underflow,
    ] {
        assert!(!error.to_string().trim().is_empty());
    }
}
