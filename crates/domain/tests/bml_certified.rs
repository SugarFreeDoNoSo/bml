use bml_domain::number::{
    Bml, ExactDerivation, ExactError, ExactInputError, ExactInputLimits, ExactRational, ExactStep,
    ExactSynthesisOptions,
};

#[test]
fn synthesis_depth_budget_is_inclusive_for_literals_and_decomposition() {
    // Literal 0/1 require no recursive decomposition. 2 = 2^1 and
    // 4 = 2^(2^1) require respectively one and two recursive edges.
    // 3 exercises near-power exponents; 81 also exercises the deeper remainder
    // path after memoized exponents have already been constructed.
    for (value, required_depth) in [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (81, 5)] {
        let target = ExactRational::from_i64(value);
        let at_limit = ExactSynthesisOptions {
            max_recursion: required_depth,
            ..ExactSynthesisOptions::default()
        };
        let certificate = Bml::from_rational_exact(&target, at_limit).unwrap();
        assert_eq!(certificate.target(), &target);
        certificate
            .derivation()
            .verify(certificate.term(), &target, at_limit)
            .unwrap();
        if required_depth > 0 {
            assert_eq!(
                Bml::from_rational_exact(
                    &target,
                    ExactSynthesisOptions {
                        max_recursion: required_depth - 1,
                        ..at_limit
                    }
                ),
                Err(ExactError::Limit)
            );
        }
    }
}

#[test]
fn synthesis_checks_the_supplied_rational_budget_before_decomposing() {
    let options = ExactSynthesisOptions {
        rational: ExactInputLimits {
            max_bits: 1,
            ..ExactInputLimits::default()
        },
        ..ExactSynthesisOptions::default()
    };
    let one = ExactRational::from_i64(1);
    assert_eq!(
        Bml::from_rational_exact(&one, options).unwrap().target(),
        &one
    );
    // The rational was constructed independently with a larger budget.
    // Synthesis must enforce its own input limit before attempting arithmetic.
    assert_eq!(
        Bml::from_rational_exact(&ExactRational::from_i64(4), options),
        Err(ExactError::Limit)
    );
}

#[test]
fn exact_conversion_proves_targets_without_float_acceptance() {
    let limits = ExactInputLimits::default();
    for text in [
        "0", "1", "2", "3", "4", "15", "-3", "12.375", "0.0010", "-0.125", "-0.000", "0.1",
    ] {
        let result = Bml::from_decimal_str_exact(text).unwrap();
        assert_eq!(
            result.target(),
            &ExactRational::from_decimal_str(text, limits).unwrap()
        );
        result
            .derivation()
            .verify(
                result.term(),
                result.target(),
                ExactSynthesisOptions::default(),
            )
            .unwrap();
    }
    let decimal = Bml::from_decimal_str_exact("0.1").unwrap();
    let float = Bml::from_f64_exact(0.1).unwrap();
    assert_ne!(decimal.target(), float.target());
    assert_eq!(
        float.target(),
        &ExactRational::from_f64(0.1, limits).unwrap()
    );
}

#[test]
fn small_naturals_match_independent_integer_tree_interpreter() {
    // Integer-only exp2 and exact log2, with no use of production rules or floats.
    fn interpret(bits: &str) -> i128 {
        let mut stack = Vec::new();
        for bit in bits.bytes().rev() {
            if bit == b'0' {
                stack.push(1i128);
            } else {
                let x = stack.pop().unwrap();
                let y = stack.pop().unwrap();
                assert!((0..127).contains(&x) && y > 0);
                let y = y as u128;
                assert!(y.is_power_of_two());
                stack.push((1i128 << x) - y.trailing_zeros() as i128);
            }
        }
        assert_eq!(stack.len(), 1);
        stack[0]
    }
    for n in 0..100 {
        let result = Bml::from_decimal_str_exact(&n.to_string()).unwrap();
        assert_eq!(interpret(&result.term().to_binary()), n);
    }
    let fifteen = Bml::from_decimal_str_exact("15").unwrap();
    assert_eq!(fifteen.term().operator_count(), 4);
    assert_eq!(fifteen.term().bit_len(), 9);
}

#[test]
fn small_exact_result_can_have_overflowing_direct_evaluation() {
    let result = Bml::from_decimal_str_exact("2049").unwrap();
    assert_eq!(result.target(), &ExactRational::from_i64(2049));
    assert!(result.term().to_f64().is_err());
    let twelve = Bml::from_decimal_str_exact("12").unwrap().into_term();
    let remainder = Bml::from_decimal_str_exact("2047").unwrap().into_term();
    assert_eq!(
        result.term(),
        &twelve.op(&remainder.exp2().unwrap()).unwrap()
    );
    result
        .derivation()
        .verify(
            result.term(),
            result.target(),
            ExactSynthesisOptions::default(),
        )
        .unwrap();
}

#[test]
fn proof_replay_rejects_false_claims_forward_references_and_domains() {
    let options = ExactSynthesisOptions::default();
    let one = Bml::from_decimal_str_exact("1").unwrap();
    assert_eq!(
        one.derivation()
            .verify(one.term(), &ExactRational::from_i64(2), options),
        Err(ExactError::InvalidProof)
    );
    assert_eq!(
        one.derivation()
            .verify(&Bml::from_u64(4).unwrap(), one.target(), options),
        Err(ExactError::InvalidProof)
    );
    for proof in [
        ExactDerivation {
            steps: vec![],
            root: 0,
        },
        ExactDerivation {
            steps: vec![ExactStep::Exp2(0)],
            root: 0,
        },
        ExactDerivation {
            steps: vec![ExactStep::One],
            root: 1,
        },
    ] {
        assert_eq!(proof.certify(options), Err(ExactError::InvalidProof));
    }
    let negative_product = ExactDerivation {
        steps: vec![
            ExactStep::One,
            ExactStep::Neg(0),
            ExactStep::PositiveMul(0, 1),
        ],
        root: 2,
    };
    assert_eq!(
        negative_product.certify(options),
        Err(ExactError::DomainNotJustified)
    );
}

#[test]
fn exact_synthesis_limits_do_not_turn_into_approximate_success() {
    let options = ExactSynthesisOptions::default();
    for limited in [
        ExactSynthesisOptions {
            max_steps: 0,
            ..options
        },
        ExactSynthesisOptions {
            max_bits: 0,
            ..options
        },
        ExactSynthesisOptions {
            max_total_bits: 0,
            ..options
        },
        ExactSynthesisOptions {
            max_recursion: 0,
            ..options
        },
    ] {
        assert!(Bml::from_decimal_str_exact_with_options("15", limited).is_err());
    }
    for sign in [0, 1u64 << 63] {
        let value = f64::from_bits(sign | 1);
        let target = ExactRational::from_f64(value, options.rational).unwrap();
        match Bml::from_f64_exact(value) {
            Ok(result) => assert_eq!(result.target(), &target),
            Err(
                ExactError::Limit
                | ExactError::Build(_)
                | ExactError::Rational(ExactInputError::Limit),
            ) => (),
            other => panic!("unexpected subnormal result: {other:?}"),
        }
    }
    assert_eq!(
        Bml::from_f64_exact(-0.0).unwrap(),
        Bml::from_f64_exact(0.0).unwrap()
    );
    assert_eq!(
        Bml::from_f64_exact(f64::NAN),
        Err(ExactError::Rational(ExactInputError::NonFinite))
    );
}

#[test]
fn rational_arithmetic_matches_signed_fraction_oracle() {
    let limits = ExactInputLimits::default();
    for a in -12i64..=12 {
        for b in -12i64..=12 {
            let x = ExactRational::from_i64(a)
                .checked_div(&ExactRational::from_i64(8), limits)
                .unwrap();
            let y = ExactRational::from_i64(b)
                .checked_div(&ExactRational::from_i64(5), limits)
                .unwrap();
            let expected = |n, d| {
                ExactRational::from_i64(n)
                    .checked_div(&ExactRational::from_i64(d), limits)
                    .unwrap()
            };
            assert_eq!(
                x.checked_add(&y, limits).unwrap(),
                expected(5 * a + 8 * b, 40)
            );
            assert_eq!(
                x.checked_sub(&y, limits).unwrap(),
                expected(5 * a - 8 * b, 40)
            );
            assert_eq!(x.checked_mul(&y, limits).unwrap(), expected(a * b, 40));
            if b == 0 {
                assert_eq!(
                    x.checked_div(&y, limits),
                    Err(ExactInputError::DivisionByZero)
                );
            } else {
                assert_eq!(x.checked_div(&y, limits).unwrap(), expected(5 * a, 8 * b));
            }
        }
    }
    assert_eq!(
        ExactRational::from_i64(i64::MIN).numerator_words_le(),
        &[1u64 << 63]
    );
}

#[test]
fn replay_exercises_arithmetic_rules_and_intermediate_limits() {
    let options = ExactSynthesisOptions::default();
    for (step, expected) in [
        (ExactStep::Add(0, 1), 3),
        (ExactStep::Sub(0, 1), -1),
        (ExactStep::PositiveMul(0, 1), 2),
        (ExactStep::PositiveDiv(1, 0), 2),
    ] {
        let proof = ExactDerivation {
            steps: vec![ExactStep::One, ExactStep::Exp2(0), step],
            root: 2,
        };
        assert_eq!(
            proof.certify(options).unwrap().target(),
            &ExactRational::from_i64(expected)
        );
        assert!(proof
            .certify(ExactSynthesisOptions {
                max_total_bits: 4,
                ..options
            })
            .is_err());
    }
    let zero_divisor = ExactDerivation {
        steps: vec![
            ExactStep::One,
            ExactStep::Zero,
            ExactStep::PositiveDiv(0, 1),
        ],
        root: 2,
    };
    assert_eq!(
        zero_divisor.certify(options),
        Err(ExactError::DomainNotJustified)
    );
}

#[test]
fn rational_multiword_carries_borrows_and_limits() {
    let limits = ExactInputLimits::default();
    let parse = |s| ExactRational::from_decimal_str(s, limits).unwrap();
    let a = parse("18446744073709551615");
    let b = parse("18446744073709551616");
    assert_eq!(
        a.checked_add(&ExactRational::from_i64(1), limits).unwrap(),
        b
    );
    assert_eq!(
        b.checked_sub(&a, limits).unwrap(),
        ExactRational::from_i64(1)
    );
    let square = a.checked_mul(&a, limits).unwrap();
    assert_eq!(square, parse("340282366920938463426481119284349108225"));
    assert_eq!(square.checked_div(&a, limits).unwrap(), a);
    let power = ExactRational::from_i64(200).checked_exp2(limits).unwrap();
    let expected = ExactRational::from_i64(400).checked_exp2(limits).unwrap();
    assert_eq!(power.checked_mul(&power, limits).unwrap(), expected);
    assert_eq!(expected.checked_div(&power, limits).unwrap(), power);
    let tiny = ExactInputLimits {
        max_bits: 64,
        ..limits
    };
    assert_eq!(a.checked_mul(&a, tiny), Err(ExactInputError::Limit));
    assert_eq!(
        a.checked_add(&ExactRational::from_i64(1), tiny),
        Err(ExactInputError::Limit)
    );
    assert_eq!(
        parse("0.5").checked_exp2(limits),
        Err(ExactInputError::NonIntegralExponent)
    );
    assert_eq!(
        ExactRational::from_i64(-3).checked_exp2(limits).unwrap(),
        parse("0.125")
    );
}

#[test]
fn exact_failure_diagnostics_are_nonempty_and_keep_typed_causes() {
    use bml_domain::number::BuildError;
    use std::error::Error;
    let invalid_decimal = Bml::from_decimal_str_exact("1e3").unwrap_err();
    assert!(!invalid_decimal.to_string().trim().is_empty());
    assert_eq!(
        invalid_decimal
            .source()
            .unwrap()
            .downcast_ref::<ExactInputError>(),
        Some(&ExactInputError::Syntax)
    );
    let non_finite = Bml::from_f64_exact(f64::INFINITY).unwrap_err();
    assert_eq!(
        non_finite
            .source()
            .unwrap()
            .downcast_ref::<ExactInputError>(),
        Some(&ExactInputError::NonFinite)
    );
    for error in [
        ExactError::Limit,
        ExactError::InvalidProof,
        ExactError::DomainNotJustified,
    ] {
        assert!(!error.to_string().trim().is_empty());
        assert!(error.source().is_none());
    }
    // A construction cause must stay inspectable without parsing its message.
    let build = Bml::one().op_with_limit(&Bml::one(), 1).unwrap_err();
    let wrapped = ExactError::from(build.clone());
    assert!(!wrapped.to_string().trim().is_empty());
    assert_eq!(
        wrapped.source().unwrap().downcast_ref::<BuildError>(),
        Some(&build)
    );
}

#[test]
fn replay_accepts_exact_inclusive_budgets_and_rejects_before_oversized_build() {
    use bml_domain::number::BuildError;
    // One uses one bit; B(One, One) uses three. The retained proof is four
    // bits in two steps, derived from grammar rather than measured by the API.
    let proof = ExactDerivation {
        steps: vec![ExactStep::One, ExactStep::Exp2(0)],
        root: 1,
    };
    let exact = ExactSynthesisOptions {
        max_steps: 2,
        max_bits: 3,
        max_total_bits: 4,
        ..ExactSynthesisOptions::default()
    };
    let certified = proof.certify(exact).unwrap();
    assert_eq!(certified.target(), &ExactRational::from_i64(2));
    assert_eq!(certified.term(), &Bml::one().op(&Bml::one()).unwrap());
    assert_eq!(
        proof.certify(ExactSynthesisOptions {
            max_steps: 1,
            ..exact
        }),
        Err(ExactError::Limit)
    );
    assert_eq!(
        proof.certify(ExactSynthesisOptions {
            max_total_bits: 3,
            ..exact
        }),
        Err(ExactError::Build(BuildError::BitLimit {
            required: 3,
            limit: 2
        }))
    );
    // A maximum caller budget does not overflow when some bits are already used.
    assert_eq!(
        proof
            .certify(ExactSynthesisOptions {
                max_total_bits: usize::MAX,
                ..exact
            })
            .unwrap(),
        certified
    );
}

#[test]
fn positive_rules_reject_each_nonpositive_operand() {
    for invalid in [1, 2] {
        for (left, right) in [(invalid, 0), (0, invalid)] {
            for operation in [
                ExactStep::PositiveMul(left, right),
                ExactStep::PositiveDiv(left, right),
            ] {
                let proof = ExactDerivation {
                    steps: vec![
                        ExactStep::One,
                        ExactStep::Zero,
                        ExactStep::Neg(0),
                        operation,
                    ],
                    root: 3,
                };
                assert_eq!(
                    proof.certify(ExactSynthesisOptions::default()),
                    Err(ExactError::DomainNotJustified)
                );
            }
        }
    }
}

#[test]
fn replay_checks_structural_and_rational_budgets_independently() {
    let one = ExactDerivation {
        steps: vec![ExactStep::One],
        root: 0,
    };
    assert_eq!(
        one.certify(ExactSynthesisOptions {
            max_bits: 0,
            ..ExactSynthesisOptions::default()
        }),
        Err(ExactError::Limit)
    );
    // Zero has a zero-bit numerator and a one-bit denominator. Its literal
    // proof step must still enforce the denominator budget.
    let zero = ExactDerivation {
        steps: vec![ExactStep::Zero],
        root: 0,
    };
    let no_rational_bits = ExactSynthesisOptions {
        rational: ExactInputLimits {
            max_bits: 0,
            ..ExactInputLimits::default()
        },
        ..ExactSynthesisOptions::default()
    };
    assert_eq!(zero.certify(no_rational_bits), Err(ExactError::Limit));
    assert_eq!(one.certify(no_rational_bits), Err(ExactError::Limit));
    let inclusive = ExactSynthesisOptions {
        max_bits: 1,
        rational: ExactInputLimits {
            max_bits: 1,
            ..ExactInputLimits::default()
        },
        ..ExactSynthesisOptions::default()
    };
    assert_eq!(
        one.certify(inclusive).unwrap().target(),
        &ExactRational::from_i64(1)
    );
}

#[test]
fn rational_long_carry_chains_and_input_budgets_are_enforced() {
    let limits = ExactInputLimits::default();
    let parse = |text| ExactRational::from_decimal_str(text, limits).unwrap();
    let all_ones = parse("340282366920938463463374607431768211455");
    let one = ExactRational::from_i64(1);
    assert_eq!(
        all_ones.checked_add(&one, limits).unwrap(),
        parse("340282366920938463463374607431768211456")
    );
    assert_eq!(
        all_ones.checked_mul(&all_ones, limits).unwrap(),
        parse("115792089237316195423570985008687907852589419931798687112530834793049593217025")
    );
    let exact_boundary = ExactInputLimits {
        max_bits: 128,
        ..limits
    };
    assert_eq!(
        all_ones.checked_mul(&one, exact_boundary).unwrap(),
        all_ones
    );
    let too_small = ExactInputLimits {
        max_bits: 127,
        ..limits
    };
    assert_eq!(
        all_ones.checked_sub(&all_ones, too_small),
        Err(ExactInputError::Limit)
    );
    assert_eq!(
        all_ones.checked_div(&all_ones, too_small),
        Err(ExactInputError::Limit)
    );
    assert_eq!(
        all_ones.checked_mul(&ExactRational::from_i64(0), too_small),
        Err(ExactInputError::Limit)
    );
    assert_eq!(
        ExactRational::from_i64(0).checked_exp2(limits).unwrap(),
        one
    );
}
