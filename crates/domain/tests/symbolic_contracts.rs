use bml_domain::symbolic::{Basis, ClosedTerm, EvalError, OpenTerm, UnboundVariable};
use std::collections::{BTreeSet, HashMap};

fn var(name: &str) -> OpenTerm {
    OpenTerm::Var(name.to_owned())
}

fn pair() -> ClosedTerm {
    ClosedTerm::op(ClosedTerm::One, ClosedTerm::One)
}

#[test]
fn partial_bind_preserves_unknown_variables_and_original() {
    let original = OpenTerm::op(var("x"), var("y"));
    let env = HashMap::from([("x".to_owned(), ClosedTerm::One)]);
    assert_eq!(
        original.bind(&env),
        OpenTerm::op(ClosedTerm::One.into(), var("y"))
    );
    assert_eq!(original.free_variables(), BTreeSet::from(["x", "y"]));
}

#[test]
fn complete_binding_closes_without_evaluating() {
    let original = OpenTerm::op(var("x"), var("y"));
    let env = HashMap::from([("x".to_owned(), ClosedTerm::One), ("y".to_owned(), pair())]);
    let closed = original.bind(&env).collapse().unwrap();
    assert_eq!(closed, ClosedTerm::op(ClosedTerm::One, pair()));
    assert_eq!(closed.evaluate_real(Basis::Bml), Ok(1.0));
}

#[test]
fn collapse_reports_the_first_unbound_name() {
    assert_eq!(
        OpenTerm::op(var("z"), var("a")).collapse(),
        Err(UnboundVariable {
            name: "z".to_owned()
        })
    );
}

#[test]
fn free_variables_are_unique_and_sorted() {
    let term = OpenTerm::op(var("z"), OpenTerm::op(var("a"), var("z")));
    assert_eq!(term.free_variables(), BTreeSet::from(["a", "z"]));
}

#[test]
fn injection_and_collapse_preserve_closed_structure() {
    let closed = ClosedTerm::op(pair(), ClosedTerm::One);
    assert_eq!(OpenTerm::from(closed.clone()).collapse(), Ok(closed));
}

#[test]
fn operator_order_and_parenthesization_are_preserved() {
    let left = ClosedTerm::op(pair(), ClosedTerm::One);
    let right = ClosedTerm::op(ClosedTerm::One, pair());
    assert_ne!(left, right);
    assert_eq!(left.evaluate_real(Basis::Bml), Ok(4.0));
    assert_eq!(right.evaluate_real(Basis::Bml), Ok(1.0));
}

#[test]
fn basis_is_explicit_and_changes_interpretation() {
    let term = pair();
    assert_eq!(term.evaluate_real(Basis::Bml), Ok(2.0));
    let eml = term.evaluate_real(Basis::Eml).unwrap();
    assert!((eml - std::f64::consts::E).abs() < 1e-14);
}

#[test]
fn primitive_has_independent_known_answers() {
    assert_eq!(Basis::Bml.apply_real(3.0, 4.0), Ok(6.0));
    assert_eq!(Basis::Bml.apply_real(-1.0, 0.25), Ok(2.5));
    assert_eq!(Basis::Eml.apply_real(0.0, 1.0), Ok(1.0));
}

#[test]
fn closed_does_not_imply_numerically_valid() {
    // In base 2: Op(One, Op(pair, One)) = 2 - log2(4) = 0.
    let zero = ClosedTerm::op(ClosedTerm::One, ClosedTerm::op(pair(), ClosedTerm::One));
    let invalid = OpenTerm::op(ClosedTerm::One.into(), zero.into())
        .collapse()
        .unwrap();
    assert_eq!(
        invalid.evaluate_real(Basis::Bml),
        Err(EvalError::LogDomain { value: 0.0 })
    );
}

#[test]
fn real_domain_rejects_zero_signed_zero_and_negatives() {
    for basis in [Basis::Eml, Basis::Bml] {
        for y in [0.0, -0.0, -1.0] {
            assert!(matches!(
                basis.apply_real(1.0, y),
                Err(EvalError::LogDomain { .. })
            ));
        }
    }
}

#[test]
fn non_finite_inputs_are_errors() {
    for basis in [Basis::Eml, Basis::Bml] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(basis.apply_real(value, 1.0), Err(EvalError::NonFiniteInput));
            assert_eq!(basis.apply_real(1.0, value), Err(EvalError::NonFiniteInput));
        }
    }
}

#[test]
fn range_errors_are_distinct() {
    for basis in [Basis::Eml, Basis::Bml] {
        assert_eq!(basis.apply_real(2048.0, 1.0), Err(EvalError::Overflow));
        assert_eq!(basis.apply_real(-2048.0, 1.0), Err(EvalError::Underflow));
    }
}

#[test]
fn overflow_in_a_tree_is_propagated() {
    let mut term = pair();
    for _ in 0..5 {
        term = ClosedTerm::op(term, ClosedTerm::One);
    }
    assert_eq!(term.evaluate_real(Basis::Bml), Err(EvalError::Overflow));
}

#[test]
fn binding_laws_hold_for_a_bounded_family_of_terms() {
    let leaves = [ClosedTerm::One.into(), var("x"), var("y")];
    let mut terms = leaves.to_vec();
    for left in &leaves {
        for right in &leaves {
            terms.push(OpenTerm::op(left.clone(), right.clone()));
        }
    }
    let x = HashMap::from([("x".to_owned(), pair())]);
    let y = HashMap::from([("y".to_owned(), ClosedTerm::One)]);
    let both: HashMap<_, _> = x.clone().into_iter().chain(y.clone()).collect();
    for term in terms {
        assert_eq!(term.bind(&HashMap::new()), term);
        let bound = term.bind(&x);
        assert_eq!(bound.bind(&x), bound);
        let expected: BTreeSet<_> = term
            .free_variables()
            .into_iter()
            .filter(|n| *n != "x")
            .collect();
        assert_eq!(bound.free_variables(), expected);
        assert_eq!(bound.bind(&y), term.bind(&both));
        assert_eq!(bound.bind(&y), term.bind(&y).bind(&x));
        assert!(term.bind(&both).collapse().is_ok());
    }
}

#[test]
fn an_already_bound_variable_is_not_rebound() {
    let first = HashMap::from([("x".to_owned(), ClosedTerm::One)]);
    let second = HashMap::from([("x".to_owned(), pair())]);
    assert_eq!(
        var("x").bind(&first).bind(&second).collapse(),
        Ok(ClosedTerm::One)
    );
}
