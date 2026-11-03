use bml_domain::symbolic::{
    complex::{Complex64, ComplexEvalError as Error, EvaluationBudget, LogBranch},
    Basis, ClosedTerm,
};
use std::f64::consts::{LN_2, PI, TAU};

fn close(actual: Complex64, expected: Complex64) {
    assert!(actual.is_finite());
    for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
        assert!(
            (a - e).abs() <= 2e-12 * e.abs().max(1.0),
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn complex_bases_branches_and_signed_cuts_have_independent_answers() {
    for basis in [Basis::Eml, Basis::Bml] {
        let scale = if basis == Basis::Eml { 1.0 } else { LN_2 };
        // exp(i*pi) = -1; log(2i) = ln(2) + i*pi/2.
        let x = Complex64 {
            re: 0.0,
            im: PI / scale,
        };
        let y = Complex64 { re: 0.0, im: 2.0 };
        for turns in [-2, 0, 3] {
            close(
                basis.apply_complex(x, y, LogBranch { turns }).unwrap(),
                Complex64 {
                    re: -1.0 - LN_2 / scale,
                    im: -(PI / 2.0 + TAU * f64::from(turns)) / scale,
                },
            );
        }
        for (zero, argument) in [(0.0, PI), (-0.0, -PI)] {
            close(
                basis
                    .apply_complex(
                        Complex64::real(0.0),
                        Complex64 { re: -1.0, im: zero },
                        LogBranch { turns: 0 },
                    )
                    .unwrap(),
                Complex64 {
                    re: 1.0,
                    im: -argument / scale,
                },
            );
        }
        for (x, y) in [(0.0, 1.0), (3.0, 4.0), (-1.0, 0.25)] {
            close(
                basis
                    .apply_complex(
                        Complex64::real(x),
                        Complex64::real(y),
                        LogBranch { turns: 0 },
                    )
                    .unwrap(),
                Complex64::real(basis.apply_real(x, y).unwrap()),
            );
        }
    }
}

#[test]
fn complex_range_checks_handle_extreme_finite_logarithms_and_bad_operands() {
    let branch = LogBranch { turns: 0 };
    for basis in [Basis::Eml, Basis::Bml] {
        let scale = if basis == Basis::Eml { 1.0 } else { LN_2 };
        for component in [f64::MAX, f64::from_bits(1)] {
            close(
                basis
                    .apply_complex(
                        Complex64::real(0.0),
                        Complex64 {
                            re: component,
                            im: component,
                        },
                        branch,
                    )
                    .unwrap(),
                Complex64 {
                    re: 1.0 - (component.ln() + LN_2 / 2.0) / scale,
                    im: -PI / (4.0 * scale),
                },
            );
        }
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            for bad in [
                Complex64 { re: value, im: 0.0 },
                Complex64 { re: 0.0, im: value },
            ] {
                assert_eq!(
                    basis.apply_complex(bad, Complex64::real(1.0), branch),
                    Err(Error::NonFiniteInput)
                );
                assert_eq!(
                    basis.apply_complex(Complex64::real(1.0), bad, branch),
                    Err(Error::NonFiniteInput)
                );
            }
        }
        for zero in [0.0, -0.0] {
            assert_eq!(
                basis.apply_complex(
                    Complex64::real(0.0),
                    Complex64 { re: zero, im: -0.0 },
                    branch
                ),
                Err(Error::LogZero)
            );
        }
        assert_eq!(
            basis.apply_complex(Complex64::real(2048.0), Complex64::real(1.0), branch),
            Err(Error::Overflow)
        );
        assert_eq!(
            basis.apply_complex(Complex64::real(-2048.0), Complex64::real(1.0), branch),
            Err(Error::Underflow)
        );
    }
}

#[test]
fn complex_terms_obey_exact_node_depth_budgets_and_preserve_the_tree() {
    let two = ClosedTerm::op(ClosedTerm::One, ClosedTerm::One);
    let four = ClosedTerm::op(two, ClosedTerm::One);
    let sixteen = ClosedTerm::op(four, ClosedTerm::One);
    let negative_two = ClosedTerm::op(ClosedTerm::One, sixteen);
    let term = ClosedTerm::op(ClosedTerm::One, negative_two);
    let before = term.clone();
    let branch = LogBranch { turns: 0 };
    let budget = EvaluationBudget {
        max_nodes: 11,
        max_depth: 6,
    };
    close(
        term.evaluate_complex(Basis::Bml, branch, budget).unwrap(),
        Complex64 {
            re: 1.0,
            im: -PI / LN_2,
        },
    );
    assert!(term.evaluate_real(Basis::Bml).is_err());
    assert_eq!(
        term.evaluate_complex(
            Basis::Bml,
            branch,
            EvaluationBudget {
                max_nodes: 10,
                ..budget
            }
        ),
        Err(Error::NodeBudget)
    );
    assert_eq!(
        term.evaluate_complex(
            Basis::Bml,
            branch,
            EvaluationBudget {
                max_depth: 5,
                ..budget
            }
        ),
        Err(Error::DepthBudget)
    );
    for limit in [
        EvaluationBudget {
            max_nodes: 0,
            max_depth: 1,
        },
        EvaluationBudget {
            max_nodes: 1,
            max_depth: 0,
        },
    ] {
        assert!(ClosedTerm::One
            .evaluate_complex(Basis::Bml, branch, limit)
            .is_err());
    }
    assert_eq!(term, before);
    let zero = ClosedTerm::op(
        ClosedTerm::One,
        ClosedTerm::op(
            ClosedTerm::op(ClosedTerm::One, ClosedTerm::One),
            ClosedTerm::One,
        ),
    );
    let invalid = ClosedTerm::op(ClosedTerm::One, zero);
    assert_eq!(
        invalid.evaluate_complex(
            Basis::Bml,
            branch,
            EvaluationBudget {
                max_nodes: 9,
                max_depth: 5
            }
        ),
        Err(Error::LogZero)
    );
    let pair = ClosedTerm::op(ClosedTerm::One, ClosedTerm::One);
    close(
        pair.evaluate_complex(
            Basis::Bml,
            LogBranch { turns: 1 },
            EvaluationBudget {
                max_nodes: 3,
                max_depth: 2,
            },
        )
        .unwrap(),
        Complex64 {
            re: 2.0,
            im: -TAU / LN_2,
        },
    );
}
