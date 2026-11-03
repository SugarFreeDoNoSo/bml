use bml_domain::number::{
    Bml, BmlMatrix, ExactError, ExactSynthesisOptions, MatrixError, MatrixLimits, ReductionInit,
};
use std::error::Error;

fn limits(elements: usize, total: usize, scalar: usize) -> MatrixLimits {
    MatrixLimits {
        max_elements: elements,
        max_total_bits: total,
        scalar: ExactSynthesisOptions {
            max_bits: scalar,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn construction_limits_are_inclusive_and_independent() {
    // Codes 0 and 4 occupy one and three bits, independently of their values.
    let one = Bml::one();
    let two = Bml::from_u64(4).unwrap();
    let exact = limits(2, 4, 3);
    let matrix = BmlMatrix::new(1, 2, vec![one.clone(), two.clone()], exact).unwrap();
    assert_eq!(matrix.limits(), exact);
    assert_eq!(matrix.total_bits(), Ok(4));
    for tight in [limits(1, 4, 3), limits(2, 3, 3), limits(2, 4, 2)] {
        assert_eq!(
            BmlMatrix::new(1, 2, vec![one.clone(), two.clone()], tight),
            Err(MatrixError::Limit)
        );
    }
    // Exceeding a per-element cap must fail even when the sum cap permits it.
    assert_eq!(
        BmlMatrix::new(1, 1, vec![two], limits(1, 100, 1)),
        Err(MatrixError::Limit)
    );
}

#[test]
fn replacement_checks_each_budget_and_updates_the_requested_cell() {
    let one = Bml::one();
    let two = Bml::from_u64(4).unwrap();
    let four = Bml::from_u64(24).unwrap();
    let mut matrix = BmlMatrix::new(2, 3, vec![one.clone(); 6], limits(6, 8, 3)).unwrap();
    assert_eq!(matrix.replace(1, 2, two.clone()), Ok(one.clone()));
    assert_eq!(matrix.total_bits(), Ok(8));
    assert_eq!(
        matrix.as_slice(),
        &[
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            one.clone(),
            two.clone()
        ]
    );
    let before = matrix.clone();
    assert_eq!(matrix.replace(0, 0, two.clone()), Err(MatrixError::Limit));
    assert_eq!(matrix, before);
    // Here total fits, so only the individual scalar cap rejects four.
    let mut scalar_limited = BmlMatrix::new(1, 1, vec![one], limits(1, 100, 3)).unwrap();
    let before = scalar_limited.clone();
    assert_eq!(scalar_limited.replace(0, 0, four), Err(MatrixError::Limit));
    assert_eq!(scalar_limited, before);
    assert!(scalar_limited.replace(0, 0, two).is_ok());
}

#[test]
fn generalized_results_obey_inclusive_scalar_work_and_output_caps() {
    let matrix = BmlMatrix::new(1, 1, vec![Bml::one()], MatrixLimits::default()).unwrap();
    let two = Bml::from_u64(4).unwrap();
    let exact = MatrixLimits {
        max_work_bits: 3,
        ..limits(1, 3, 3)
    };
    let product = |budget| {
        matrix.generalized_product(
            &matrix,
            ReductionInit::First { empty: None },
            budget,
            |_, _| Ok(two.clone()),
            |_, _| unreachable!("one inner element needs no accumulation"),
        )
    };
    assert_eq!(
        product(exact).unwrap().as_slice(),
        std::slice::from_ref(&two)
    );
    for tight in [
        MatrixLimits {
            max_work_bits: 2,
            ..exact
        },
        MatrixLimits {
            max_total_bits: 2,
            ..exact
        },
        MatrixLimits {
            scalar: ExactSynthesisOptions {
                max_bits: 2,
                ..exact.scalar
            },
            ..exact
        },
    ] {
        assert!(matches!(
            product(tight),
            Err(MatrixError::Budget {
                row: 0,
                column: 0,
                ..
            })
        ));
    }
}

#[test]
fn first_term_fallback_is_used_only_for_empty_reductions() {
    let defaults = MatrixLimits::default();
    let one = Bml::one();
    let fallback = Bml::from_u64(24).unwrap();
    let matrix = BmlMatrix::new(1, 1, vec![one.clone()], defaults).unwrap();
    let product = matrix
        .generalized_product(
            &matrix,
            ReductionInit::First {
                empty: Some(&fallback),
            },
            defaults,
            |a, _| Ok(a.clone()),
            |_, _| unreachable!(),
        )
        .unwrap();
    assert_eq!(product.as_slice(), &[one]);
    let left = BmlMatrix::new(1, 0, vec![], defaults).unwrap();
    let right = BmlMatrix::new(0, 1, vec![], defaults).unwrap();
    let product = left
        .generalized_product(
            &right,
            ReductionInit::First {
                empty: Some(&fallback),
            },
            defaults,
            |_, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .unwrap();
    assert_eq!(product.as_slice(), &[fallback]);
}

#[test]
fn matrix_diagnostics_preserve_typed_scalar_causes() {
    for error in [
        MatrixError::Shape,
        MatrixError::Limit,
        MatrixError::Budget {
            row: 1,
            column: 2,
            inner: Some(3),
        },
        MatrixError::Index { row: 1, column: 2 },
        MatrixError::EmptyReduction,
    ] {
        assert!(!error.to_string().is_empty());
        assert!(error.source().is_none());
    }
    for error in [
        MatrixError::Scalar {
            row: 1,
            column: 2,
            inner: None,
            source: ExactError::DomainNotJustified,
        },
        MatrixError::ScalarPreparation(ExactError::DomainNotJustified),
    ] {
        assert!(!error.to_string().is_empty());
        assert_eq!(
            error.source().unwrap().downcast_ref::<ExactError>(),
            Some(&ExactError::DomainNotJustified)
        );
    }
}
