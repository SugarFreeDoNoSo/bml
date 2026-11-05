use bml_domain::number::{
    Bml, BmlMatrix, CertifiedBml, ExactError, ExactOperand, ExactRational, ExactSynthesisOptions,
    MatrixError, MatrixLimits, ReductionInit,
};
fn term(n: i64) -> Bml {
    Bml::from_decimal_str_exact(&n.to_string())
        .unwrap()
        .into_term()
}
fn matrix(rows: usize, columns: usize, values: &[i64]) -> BmlMatrix {
    BmlMatrix::new(
        rows,
        columns,
        values.iter().map(|n| term(*n)).collect(),
        MatrixLimits::default(),
    )
    .unwrap()
}
fn targets(matrix: &BmlMatrix) -> Vec<ExactRational> {
    matrix
        .as_slice()
        .iter()
        .map(|v| {
            v.certify_exact(ExactSynthesisOptions::default())
                .unwrap()
                .target()
                .clone()
        })
        .collect()
}

#[test]
fn shapes_access_and_transactional_replacement() {
    let limits = MatrixLimits::default();
    assert_eq!(
        BmlMatrix::new(usize::MAX, 2, vec![], limits),
        Err(MatrixError::Shape)
    );
    assert_eq!(
        BmlMatrix::new(2, 3, vec![Bml::one()], limits),
        Err(MatrixError::Shape)
    );
    let mut a = matrix(2, 3, &[1, 2, 3, 4, 0, -1]);
    assert_eq!(a.shape(), (2, 3));
    assert_eq!(a[(1, 0)], term(4));
    assert_eq!(a.get(2, 0), Err(MatrixError::Index { row: 2, column: 0 }));
    let before = a.clone();
    assert!(a.replace(0, 3, term(1)).is_err());
    assert_eq!(a, before);
    assert_eq!(a.replace(1, 0, term(2)).unwrap(), term(4));
    let tight = MatrixLimits {
        max_total_bits: 1,
        ..limits
    };
    let mut small = BmlMatrix::new(1, 1, vec![Bml::one()], tight).unwrap();
    assert_eq!(small.replace(0, 0, term(2)), Err(MatrixError::Limit));
    assert_eq!(small[(0, 0)], Bml::one());
}

#[test]
fn elementwise_operations_use_values_not_codes() {
    let a = matrix(1, 2, &[2, -1]);
    let b = matrix(1, 2, &[2, 0]);
    let original = (a.clone(), b.clone());
    let sum = (&a + &b).unwrap();
    assert_eq!(
        targets(&sum),
        vec![ExactRational::from_i64(4), ExactRational::from_i64(-1)]
    );
    assert_ne!(sum[(0, 0)].to_u64(), Some(8));
    assert_eq!(
        targets(&a.hadamard(&b).unwrap()),
        vec![ExactRational::from_i64(4), ExactRational::from_i64(0)]
    );
    assert_eq!((&a + &matrix(2, 1, &[1, 1])), Err(MatrixError::Shape));
    assert_eq!((a, b), original);
}

#[test]
fn rectangular_product_matches_independent_integer_oracle() {
    let aa = [1, 2, 1, 2, 1, 2];
    let bb = [1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2];
    let a = matrix(2, 3, &aa);
    let b = matrix(3, 4, &bb);
    let product = (&a * &b).unwrap();
    assert_eq!(product.shape(), (2, 4));
    let mut expected = Vec::new();
    for r in 0..2 {
        for c in 0..4 {
            let mut sum = 0i64;
            for k in 0..3 {
                sum += aa[r * 3 + k] * bb[k * 4 + c];
            }
            expected.push(ExactRational::from_i64(sum));
        }
    }
    assert_eq!(targets(&product), expected);
    let p0 = (&a[(0, 0)] * &b[(0, 0)]).unwrap();
    let p1 = (&a[(0, 1)] * &b[(1, 0)]).unwrap();
    let p2 = (&a[(0, 2)] * &b[(2, 0)]).unwrap();
    assert_eq!(product[(0, 0)], (&(&p0 + &p1).unwrap() + &p2).unwrap());
}

#[test]
fn empty_products_use_symbolic_zero_and_bound_zero_size_shapes() {
    let limits = MatrixLimits::default();
    let a = matrix(2, 0, &[]);
    let b = matrix(0, 3, &[]);
    let product = (&a * &b).unwrap();
    assert_eq!(product.shape(), (2, 3));
    assert!(product.as_slice().iter().all(|v| v.to_u64() == Some(88)));
    let huge = BmlMatrix::new(usize::MAX, 0, vec![], limits).unwrap();
    let empty = matrix(0, 0, &[]);
    assert_eq!((&huge * &empty).unwrap().shape(), (usize::MAX, 0));
    let limited = MatrixLimits {
        max_work_bits: 6,
        ..limits
    };
    assert_eq!(
        a.matmul_with_limits(&b, limited),
        Err(MatrixError::Budget {
            row: 0,
            column: 0,
            inner: None
        })
    );
}

#[test]
fn generalized_product_preserves_nonassociative_order_and_empty_policy() {
    let limits = MatrixLimits::default();
    let a = matrix(1, 3, &[1, 2, 4]);
    let b = matrix(3, 1, &[1, 1, 1]);
    let product = a
        .generalized_product(
            &b,
            ReductionInit::First { empty: None },
            limits,
            |a, _| Ok(a.clone()),
            |a, b| Ok(a.op(b)?),
        )
        .unwrap();
    assert_eq!(
        product[(0, 0)],
        term(1).op(&term(2)).unwrap().op(&term(4)).unwrap()
    );
    assert_ne!(
        product[(0, 0)],
        term(1).op(&term(2).op(&term(4)).unwrap()).unwrap()
    );
    let empty_a = matrix(2, 0, &[]);
    let empty_b = matrix(0, 1, &[]);
    assert_eq!(
        empty_a.generalized_product(
            &empty_b,
            ReductionInit::First { empty: None },
            limits,
            |a, _| Ok(a.clone()),
            |a, b| Ok(a.op(b)?)
        ),
        Err(MatrixError::EmptyReduction)
    );
    let seed = term(2);
    let result = empty_a
        .generalized_product(
            &empty_b,
            ReductionInit::Value(&seed),
            limits,
            |_, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .unwrap();
    assert!(result.as_slice().iter().all(|v| *v == seed));
    let seeded = a
        .generalized_product(
            &b,
            ReductionInit::Value(&seed),
            limits,
            |a, _| Ok(a.clone()),
            |a, b| Ok(a.op(b)?),
        )
        .unwrap();
    assert_eq!(
        seeded[(0, 0)],
        seed.op(&term(1))
            .unwrap()
            .op(&term(2))
            .unwrap()
            .op(&term(4))
            .unwrap()
    );
}

#[test]
fn failures_are_localized_and_do_not_mutate_inputs() {
    let limits = MatrixLimits::default();
    let a = matrix(1, 2, &[1, 2]);
    let b = matrix(2, 1, &[2, 1]);
    let before = (a.clone(), b.clone());
    let tiny = MatrixLimits {
        max_work_bits: 1,
        ..limits
    };
    assert_eq!(
        a.matmul_with_limits(&b, tiny),
        Err(MatrixError::Budget {
            row: 0,
            column: 0,
            inner: Some(0)
        })
    );
    let invalid = Bml::one().op(&term(0)).unwrap();
    let bad = BmlMatrix::new(1, 2, vec![Bml::one(), invalid], limits).unwrap();
    assert!(matches!(
        &bad + &a,
        Err(MatrixError::Scalar {
            row: 0,
            column: 1,
            inner: None,
            source: ExactError::DomainNotJustified
        })
    ));
    assert_eq!((a, b), before);
}

#[test]
fn numeric_scalar_is_prepared_once_before_the_kernel() {
    use std::cell::Cell;
    struct Counted<'a>(&'a Cell<usize>);
    impl ExactOperand for Counted<'_> {
        fn into_certified(
            self,
            options: ExactSynthesisOptions,
        ) -> Result<CertifiedBml, ExactError> {
            self.0.set(self.0.get() + 1);
            Bml::from_decimal_str_exact_with_options("2", options)
        }
    }
    let calls = Cell::new(0);
    let a = matrix(2, 2, &[1, 2, 1, 2]);
    let limits = MatrixLimits::default();
    let result = a.scale(Counted(&calls), limits).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(
        targets(&result),
        [2, 4, 2, 4]
            .into_iter()
            .map(ExactRational::from_i64)
            .collect::<Vec<_>>()
    );
    let added = a.add_scalar(4i32, limits).unwrap();
    assert_eq!(
        targets(&added),
        [5, 6, 5, 6]
            .into_iter()
            .map(ExactRational::from_i64)
            .collect::<Vec<_>>()
    );
}

#[test]
fn scalar_output_budget_reports_the_last_row_major_position_transactionally() {
    let input = matrix(2, 3, &[1; 6]);
    let before = input.clone();
    // Measure an independently constructed scalar result, rather than assume
    // a particular expansion length or use the matrix's accounting itself.
    let scalar_sum = (&Bml::one() + &Bml::one()).unwrap();
    let total_output_bits = 6 * scalar_sum.bit_len();
    let tight = MatrixLimits {
        max_total_bits: total_output_bits - 1,
        ..MatrixLimits::default()
    };
    assert_eq!(
        input.add_scalar(1i32, tight),
        Err(MatrixError::Budget {
            row: 1,
            column: 2,
            inner: None,
        })
    );
    assert_eq!(input, before);

    let exact = MatrixLimits {
        max_total_bits: total_output_bits,
        ..tight
    };
    let output = input.add_scalar(1i32, exact).unwrap();
    assert_eq!(output.shape(), (2, 3));
    assert_eq!(output.total_bits().unwrap(), total_output_bits);
    assert_eq!(targets(&output), vec![ExactRational::from_i64(2); 6]);
    assert_eq!(input, before);
}

#[test]
fn signed_matmul_and_output_budget_preserve_inputs() {
    let limits = MatrixLimits::default();
    let a = matrix(1, 3, &[-1, 0, 1]);
    let b = matrix(3, 1, &[1, 2, -1]);
    assert_eq!(
        targets(&(a.clone() * b.clone()).unwrap()),
        vec![ExactRational::from_i64(-2)]
    );
    assert_eq!(
        targets(&(a.clone() * &b).unwrap()),
        vec![ExactRational::from_i64(-2)]
    );
    assert_eq!(
        targets(&(&a * b.clone()).unwrap()),
        vec![ExactRational::from_i64(-2)]
    );
    let one = matrix(1, 1, &[1]);
    let tight = MatrixLimits {
        max_total_bits: 1,
        ..limits
    };
    assert_eq!(
        one.add_with_limits(&one, tight),
        Err(MatrixError::Budget {
            row: 0,
            column: 0,
            inner: None
        })
    );
    assert_eq!(one.as_slice(), &[Bml::one()]);
    assert!(matches!(
        one.scale(f64::NAN, limits),
        Err(MatrixError::ScalarPreparation(_))
    ));
}
