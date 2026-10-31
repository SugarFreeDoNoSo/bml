use bml_domain::number::{Bml, BuildError, Rules};
use bml_domain::symbolic::EvalError;

fn approx(term: Bml, expected: f64) {
    let actual = term.to_f64().unwrap();
    assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
}

#[test]
fn arithmetic_rules_construct_and_evaluate_symbolic_terms() {
    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let four = two.op(&one).unwrap();
    let zero = one.op(&four).unwrap();
    approx(one.exp2().unwrap(), 2.0);
    approx(four.log2().unwrap(), 2.0);
    approx(one.add(&two).unwrap(), 3.0);
    approx(zero.sub(&one).unwrap(), -1.0);
    approx(two.neg().unwrap(), -2.0);
    approx(two.mul(&two).unwrap(), 4.0);
    approx(four.div(&two).unwrap(), 2.0);
    approx(two.recip().unwrap(), 0.5);
    approx(two.pow(&two).unwrap(), 4.0);
    approx(four.sqrt().unwrap(), 2.0);
}

#[test]
fn roots_and_powers_include_one_and_fractional_bases() {
    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let four = two.op(&one).unwrap();
    let zero = one.op(&four).unwrap();
    let minus_one = zero.op(&four).unwrap();
    let half = minus_one.op(&one).unwrap();
    let quarter = two.neg().unwrap().exp2().unwrap();

    for (term, value) in [
        (&quarter, 0.25_f64),
        (&half, 0.5),
        (&one, 1.0),
        (&two, 2.0),
        (&four, 4.0),
    ] {
        approx(term.sqrt().unwrap(), value.sqrt());
        approx(term.pow(&half).unwrap(), value.powf(0.5));
        approx(term.pow(&two).unwrap(), value.powi(2));
    }
    approx(half.mul(&two).unwrap(), 1.0);
    approx(half.div(&two).unwrap(), 0.25);
    approx(half.recip().unwrap(), 2.0);
}

#[test]
fn signed_sums_and_differences_match_native_arithmetic() {
    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let four = two.op(&one).unwrap();
    let zero = one.op(&four).unwrap();
    let minus_one = zero.op(&four).unwrap();
    let half = minus_one.op(&one).unwrap();
    let minus_half = half.neg().unwrap();
    let cases = [
        (&minus_one, -1.0),
        (&minus_half, -0.5),
        (&zero, 0.0),
        (&half, 0.5),
        (&one, 1.0),
        (&two, 2.0),
    ];
    for (left, x) in cases {
        for (right, y) in cases {
            approx(left.add(right).unwrap(), x + y);
            approx(left.sub(right).unwrap(), x - y);
        }
    }
}

#[test]
fn valid_expansions_report_numeric_range_failures() {
    let mut term = Bml::one();
    for _ in 0..4 {
        term = term.exp2().unwrap();
    }
    assert_eq!(term.to_f64().unwrap(), 65536.0);
    let overflow = term.exp2().unwrap();
    assert_eq!(overflow.to_f64(), Err(EvalError::Overflow));
}

#[test]
fn construction_does_not_hide_evaluation_domain_errors() {
    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let four = two.op(&one).unwrap();
    let zero = one.op(&four).unwrap();
    let expression = zero.mul(&one).unwrap();
    assert!(matches!(expression.to_f64(), Err(EvalError::LogDomain { value }) if value == 0.0));
}

#[test]
fn rule_budget_applies_to_every_intermediate() {
    let one = Bml::one();
    let small = Rules { max_bits: 2 };
    assert_eq!(
        small.exp2(&one),
        Err(BuildError::BitLimit {
            required: 3,
            limit: 2
        })
    );
    assert!(matches!(
        small.add(&one, &one),
        Err(BuildError::BitLimit { .. })
    ));
    assert_eq!(Rules { max_bits: 3 }.exp2(&one).unwrap().to_binary(), "100");
}
