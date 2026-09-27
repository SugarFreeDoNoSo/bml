use bml_domain::symbolic::{Basis, EvalError};
use bml_domain::tensor::{
    add, apply_basis, dot, matmul, mul, Scalar, Shape, Tensor, TensorError, TensorView,
};
use std::error::Error;

fn tensor<T>(dims: impl Into<Vec<usize>>, values: Vec<T>) -> Tensor<T> {
    Tensor::from_vec(Shape::new(dims).unwrap(), values).unwrap()
}

#[test]
fn shape_scalar_empty_and_overflow_contracts() {
    let scalar = Shape::new([]).unwrap();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.len(), 1);
    assert!(!scalar.is_empty());
    assert_eq!(scalar.strides(), &[]);
    assert_eq!(scalar.offset(&[]), Ok(0));
    for dims in [
        vec![0],
        vec![0, usize::MAX, 2],
        vec![usize::MAX, 2, 0],
        vec![2, 0, 3],
    ] {
        let empty = Shape::new(dims.clone()).unwrap();
        assert_eq!(empty.dims(), dims);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.strides(), vec![0; dims.len()]);
    }
    assert_eq!(Shape::new([usize::MAX, 2]), Err(TensorError::SizeOverflow));
    assert_eq!(Shape::new([2, usize::MAX]), Err(TensorError::SizeOverflow));
    let large = Shape::new([usize::MAX]).unwrap();
    assert_eq!(large.len(), usize::MAX);
    assert_eq!(large.offset(&[usize::MAX - 1]), Ok(usize::MAX - 1));
}

#[test]
fn row_major_coordinates_and_bounds_are_exact() {
    let shape = Shape::new([2, 3, 4]).unwrap();
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.len(), 24);
    assert_eq!(shape.strides(), &[12, 4, 1]);
    let mut expected = 0;
    for x in 0..2 {
        for y in 0..3 {
            for z in 0..4 {
                assert_eq!(shape.offset(&[x, y, z]), Ok(expected));
                expected += 1;
            }
        }
    }
    assert_eq!(
        shape.offset(&[0, 0]),
        Err(TensorError::RankMismatch {
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(
        shape.offset(&[0, 0, 0, 0]),
        Err(TensorError::RankMismatch {
            expected: 3,
            actual: 4
        })
    );
    for (indices, axis, index, size) in [
        ([2, 0, 0], 0, 2, 2),
        ([0, 3, 0], 1, 3, 3),
        ([0, 0, usize::MAX], 2, usize::MAX, 4),
    ] {
        assert_eq!(
            shape.offset(&indices),
            Err(TensorError::IndexOutOfBounds { axis, index, size })
        );
    }
    let empty = Shape::new([usize::MAX, 2, 0]).unwrap();
    assert_eq!(
        empty.offset(&[usize::MAX - 1, 1, 0]),
        Err(TensorError::IndexOutOfBounds {
            axis: 2,
            index: 0,
            size: 0
        })
    );
}

#[test]
fn owned_and_borrowed_buffers_require_exact_lengths() {
    for actual in [0, 3, 5] {
        let shape = Shape::new([2, 2]).unwrap();
        let values = vec![0.0; actual];
        let expected = TensorError::LengthMismatch {
            expected: 4,
            actual,
        };
        assert_eq!(
            Tensor::from_vec(shape.clone(), values.clone()),
            Err(expected.clone())
        );
        assert_eq!(TensorView::from_slice(shape, &values), Err(expected));
    }
    assert_eq!(
        Tensor::<f64>::from_vec(Shape::new([]).unwrap(), vec![]),
        Err(TensorError::LengthMismatch {
            expected: 1,
            actual: 0
        })
    );
    assert!(tensor([2, 0], Vec::<f64>::new()).as_slice().is_empty());
}

#[test]
fn views_and_reshape_preserve_ownership_and_values() {
    let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ptr = values.as_ptr();
    let mut owned = tensor([2, 3], values);
    assert_eq!(owned.as_slice().as_ptr(), ptr);
    {
        let view = owned.view();
        assert_eq!(view.shape().dims(), &[2, 3]);
        assert_eq!(view.as_slice().as_ptr(), ptr);
        assert_eq!(view.get(&[1, 2]), Ok(&6.0));
        assert_eq!(
            view.get(&[2, 0]),
            Err(TensorError::IndexOutOfBounds {
                axis: 0,
                index: 2,
                size: 2
            })
        );
        let reshaped = view.reshape(Shape::new([3, 2]).unwrap()).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(reshaped.as_slice().as_ptr(), ptr);
        assert_eq!(reshaped.get(&[2, 0]), Ok(&5.0));
        assert_eq!(view.shape().dims(), &[2, 3]);
        assert_eq!(
            view.reshape(Shape::new([7]).unwrap()),
            Err(TensorError::LengthMismatch {
                expected: 7,
                actual: 6
            })
        );
    }
    assert_eq!(
        owned.reshape(Shape::new([5]).unwrap()),
        Err(TensorError::LengthMismatch {
            expected: 5,
            actual: 6
        })
    );
    assert_eq!(owned.shape().dims(), &[2, 3]);
    owned.reshape(Shape::new([6]).unwrap()).unwrap();
    assert_eq!(owned.shape().dims(), &[6]);
    owned.as_mut_slice()[1] = -2.0;
    let values = owned.into_vec();
    assert_eq!(values.as_ptr(), ptr);
    assert_eq!(values, [1.0, -2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn external_views_are_exact_slices_and_reject_strides() {
    let buffer = [-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0];
    let shape = Shape::new([2, 3]).unwrap();
    let data = &buffer[1..7];
    let view = TensorView::from_strided(shape.clone(), data, &[3, 1]).unwrap();
    assert_eq!(view.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(view.as_slice().as_ptr(), data.as_ptr());
    assert_eq!(view.get(&[0, 1]), Ok(&2.0));
    assert_eq!(
        TensorView::from_strided(shape.clone(), data, &[1, 2]),
        Err(TensorError::NonContiguous)
    );
    assert_eq!(
        TensorView::from_strided(shape.clone(), data, &[3]),
        Err(TensorError::RankMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        TensorView::from_strided(shape, &buffer, &[3, 1]),
        Err(TensorError::LengthMismatch {
            expected: 6,
            actual: 8
        })
    );
    let scalar = TensorView::from_slice(Shape::new([]).unwrap(), &buffer[1..2]).unwrap();
    assert_eq!(scalar.get(&[]), Ok(&1.0));
    assert_eq!(
        scalar.get(&[0]),
        Err(TensorError::RankMismatch {
            expected: 0,
            actual: 1
        })
    );
}

#[test]
fn f32_native_kernels_have_known_signed_fractional_results() {
    let a = tensor([4], vec![-2.0_f32, 0.0, 0.5, 3.0]);
    let b = tensor([4], vec![3.0_f32, -4.0, 2.0, 0.25]);
    assert_eq!(
        add(&a.view(), &b.view()).unwrap().as_slice(),
        &[1.0, -4.0, 2.5, 3.25]
    );
    assert_eq!(
        mul(&a.view(), &b.view()).unwrap().as_slice(),
        &[-6.0, -0.0, 1.0, 0.75]
    );
    assert_eq!(dot(&a.view(), &b.view()), Ok(-4.25));
    assert_eq!(a.as_slice(), &[-2.0, 0.0, 0.5, 3.0]);
    assert_eq!(b.as_slice(), &[3.0, -4.0, 2.0, 0.25]);
}

#[test]
fn scalar_and_empty_elementwise_results_preserve_shapes() {
    let a = tensor([], vec![0.5_f64]);
    let b = tensor([], vec![2.0_f64]);
    let sum = add(&a.view(), &b.view()).unwrap();
    assert_eq!(sum.shape().dims(), &[]);
    assert_eq!(sum.as_slice(), &[2.5]);
    assert_eq!(mul(&a.view(), &b.view()).unwrap().as_slice(), &[1.0]);
    let empty = tensor([2, 0, 3], Vec::<f32>::new());
    for result in [
        add(&empty.view(), &empty.view()),
        mul(&empty.view(), &empty.view()),
        apply_basis(&empty.view(), &empty.view(), Basis::Eml),
    ] {
        let result = result.unwrap();
        assert_eq!(result.shape().dims(), &[2, 0, 3]);
        assert!(result.as_slice().is_empty());
    }
    let empty_vector = tensor([0], Vec::<f64>::new());
    assert_eq!(
        dot(&empty_vector.view(), &empty_vector.view())
            .unwrap()
            .to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn kernels_reject_broadcast_rank_and_inner_dimension_mismatches() {
    let a = tensor([2, 3], vec![1.0_f64; 6]);
    let b = tensor([3, 2], vec![1.0_f64; 6]);
    let mismatch = TensorError::ShapeMismatch {
        left: vec![2, 3],
        right: vec![3, 2],
    };
    assert_eq!(add(&a.view(), &b.view()), Err(mismatch.clone()));
    assert_eq!(mul(&a.view(), &b.view()), Err(mismatch.clone()));
    assert_eq!(apply_basis(&a.view(), &b.view(), Basis::Bml), Err(mismatch));
    let short = tensor([1, 3], vec![1.0; 3]);
    assert!(matches!(
        add(&a.view(), &short.view()),
        Err(TensorError::ShapeMismatch { .. })
    ));
    let v = tensor([6], vec![1.0; 6]);
    for result in [dot(&a.view(), &v.view()), dot(&v.view(), &a.view())] {
        assert_eq!(
            result,
            Err(TensorError::RankMismatch {
                expected: 1,
                actual: 2
            })
        );
    }
    let shorter = tensor([2], vec![1.0; 2]);
    assert_eq!(
        dot(&v.view(), &shorter.view()),
        Err(TensorError::ShapeMismatch {
            left: vec![6],
            right: vec![2]
        })
    );
    for result in [matmul(&a.view(), &v.view()), matmul(&v.view(), &a.view())] {
        assert_eq!(
            result,
            Err(TensorError::RankMismatch {
                expected: 2,
                actual: 1
            })
        );
    }
    assert_eq!(
        matmul(&a.view(), &a.view()),
        Err(TensorError::InnerDimensionMismatch { left: 3, right: 2 })
    );
}

#[test]
fn rectangular_matmul_has_independent_known_answers() {
    let a = tensor([2, 3], vec![1.0_f64, -2.0, 0.5, 0.0, 3.0, -1.0]);
    let b = tensor(
        [3, 4],
        vec![
            2.0, 0.0, -1.0, 4.0, 1.0, -3.0, 2.0, 0.5, -2.0, 1.0, 0.0, 3.0,
        ],
    );
    let c = matmul(&a.view(), &b.view()).unwrap();
    assert_eq!(c.shape().dims(), &[2, 4]);
    assert_eq!(c.as_slice(), &[-1.0, 6.5, -5.0, 4.5, 5.0, -10.0, 6.0, -1.5]);
    let af = tensor([2, 3], a.as_slice().iter().map(|x| *x as f32).collect());
    let bf = tensor([3, 4], b.as_slice().iter().map(|x| *x as f32).collect());
    assert_eq!(
        matmul(&af.view(), &bf.view()).unwrap().as_slice(),
        &[-1.0_f32, 6.5, -5.0, 4.5, 5.0, -10.0, 6.0, -1.5]
    );
}

#[test]
fn small_matrices_match_integer_oracle_across_shapes() {
    // Independent integer arithmetic; each output is a row/column iterator dot.
    for m in 1..=4 {
        for k in 1..=5 {
            for n in 1..=4 {
                let a: Vec<i64> = (0..m * k).map(|x| (x as i64 * 7 % 13) - 6).collect();
                let b: Vec<i64> = (0..k * n).map(|x| (x as i64 * 3 % 11) - 5).collect();
                let mut expected = Vec::new();
                for row in a.chunks(k) {
                    for col in 0..n {
                        expected.push(
                            row.iter()
                                .zip(b.iter().skip(col).step_by(n))
                                .map(|(x, y)| x * y)
                                .sum::<i64>() as f64,
                        );
                    }
                }
                let left = tensor([m, k], a.iter().map(|x| *x as f64).collect());
                let right = tensor([k, n], b.iter().map(|x| *x as f64).collect());
                let actual = matmul(&left.view(), &right.view()).unwrap();
                assert_eq!(actual.shape().dims(), &[m, n]);
                assert_eq!(actual.as_slice(), expected);
            }
        }
    }
}

#[test]
fn empty_matmul_checks_output_shape_and_capacity() {
    for (m, k, n) in [(2, 0, 3), (0, 2, 3), (2, 3, 0), (0, 0, 0)] {
        let a = tensor([m, k], vec![1.0_f64; m * k]);
        let b = tensor([k, n], vec![2.0_f64; k * n]);
        let c = matmul(&a.view(), &b.view()).unwrap();
        assert_eq!(c.shape().dims(), &[m, n]);
        assert_eq!(c.as_slice(), vec![0.0; m * n]);
    }
    let a = tensor([usize::MAX, 0], Vec::<f64>::new());
    let b = tensor([0, 2], Vec::<f64>::new());
    assert_eq!(matmul(&a.view(), &b.view()), Err(TensorError::SizeOverflow));
    let b = tensor([0, 1], Vec::<f64>::new());
    assert_eq!(
        matmul(&a.view(), &b.view()),
        Err(TensorError::AllocationFailed)
    );
}

#[test]
fn reductions_use_dtype_and_documented_order() {
    let f = tensor([3], vec![16_777_216.0_f32, 1.0, -16_777_216.0]);
    let ones = tensor([3], vec![1.0_f32; 3]);
    assert_eq!(dot(&f.view(), &ones.view()), Ok(0.0));
    let m = f.view().reshape(Shape::new([1, 3]).unwrap()).unwrap();
    let col = ones.view().reshape(Shape::new([3, 1]).unwrap()).unwrap();
    assert_eq!(matmul(&m, &col).unwrap().as_slice(), &[0.0]);
    let d = tensor([3], vec![16_777_216.0_f64, 1.0, -16_777_216.0]);
    assert_eq!(dot(&d.view(), &tensor([3], vec![1.0; 3]).view()), Ok(1.0));
    let max = tensor([1], vec![f64::MAX]);
    assert_eq!(
        add(&max.view(), &max.view()).unwrap().as_slice(),
        &[f64::INFINITY]
    );
    let nan = tensor([1], vec![f32::NAN]);
    assert!(mul(&nan.view(), &nan.view()).unwrap().as_slice()[0].is_nan());
}

#[test]
fn basis_tensor_evaluation_uses_explicit_base_and_preserves_dtype() {
    let x = tensor([2, 2], vec![0.0_f64, 1.0, 2.0, -1.0]);
    let y = tensor([2, 2], vec![1.0_f64, 2.0, 4.0, 0.5]);
    let result = apply_basis(&x.view(), &y.view(), Basis::Bml).unwrap();
    assert_eq!(result.shape().dims(), &[2, 2]);
    assert_eq!(result.as_slice(), &[1.0, 1.0, 2.0, 1.5]);
    let eml = apply_basis(&x.view(), &y.view(), Basis::Eml).unwrap();
    for (actual, expected) in
        eml.as_slice()
            .iter()
            .zip([1.0, 2.025134647899099, 6.00276173781076, 1.0610266217313877])
    {
        assert!((actual - expected).abs() < 1e-14);
    }
    let xf = tensor([2], vec![1.0_f32, -1.0]);
    let yf = tensor([2], vec![1.0_f32, 2.0]);
    let rf: Tensor<f32> = apply_basis(&xf.view(), &yf.view(), Basis::Bml).unwrap();
    assert_eq!(rf.as_slice(), &[2.0_f32, -0.5]);
    assert_eq!(
        apply_basis(
            &tensor([], vec![1.0_f32]).view(),
            &tensor([], vec![1.0_f32]).view(),
            Basis::Eml
        )
        .unwrap()
        .as_slice(),
        &[std::f32::consts::E]
    );
}

#[test]
fn basis_failures_identify_first_invalid_element() {
    let x = tensor([3], vec![0.0_f64, 1.0, f64::NAN]);
    for invalid in [0.0, -1.0, -0.0] {
        let y = tensor([3], vec![1.0, invalid, 0.0]);
        assert_eq!(
            apply_basis(&x.view(), &y.view(), Basis::Bml),
            Err(TensorError::Evaluation {
                index: 1,
                source: EvalError::LogDomain { value: invalid }
            })
        );
    }
    for (value, source) in [
        (f64::INFINITY, EvalError::NonFiniteInput),
        (f64::NAN, EvalError::NonFiniteInput),
        (1024.0, EvalError::Overflow),
        (-1075.0, EvalError::Underflow),
    ] {
        let x = tensor([2], vec![0.0_f64, value]);
        let y = tensor([2], vec![1.0_f64; 2]);
        assert_eq!(
            apply_basis(&x.view(), &y.view(), Basis::Bml),
            Err(TensorError::Evaluation { index: 1, source })
        );
    }
    let a = tensor([1], vec![1.0_f32]);
    let b = tensor([1], vec![f32::NAN]);
    assert_eq!(
        apply_basis(&a.view(), &b.view(), Basis::Eml),
        Err(TensorError::Evaluation {
            index: 0,
            source: EvalError::NonFiniteInput
        })
    );
}

#[test]
fn basis_f32_range_checks_preserve_zero_and_subnormals() {
    for x in [128.0_f32, -150.0] {
        let a = tensor([2], vec![1.0_f32, x]);
        let b = tensor([2], vec![1.0_f32; 2]);
        assert_eq!(
            apply_basis(&a.view(), &b.view(), Basis::Bml),
            Err(TensorError::OutputRange { index: 1 })
        );
    }
    let a = tensor([3], vec![0.0_f32, -149.0, 127.0]);
    let b = tensor([3], vec![2.0_f32, 1.0, 1.0]);
    let result = apply_basis(&a.view(), &b.view(), Basis::Bml).unwrap();
    assert_eq!(
        result.as_slice(),
        &[0.0, f32::from_bits(1), 2.0_f32.powi(127)]
    );
    assert_eq!(
        <f32 as Scalar>::from_basis_result(-0.0).unwrap().to_bits(),
        (-0.0_f32).to_bits()
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(<f64 as Scalar>::from_basis_result(value), None);
        assert_eq!(<f32 as Scalar>::from_basis_result(value), None);
    }
    assert_eq!(<f32 as Scalar>::from_basis_result(-f64::MIN_POSITIVE), None);
    assert_eq!(
        <f32 as Scalar>::from_basis_result(-f64::from(f32::MAX)),
        Some(-f32::MAX)
    );
    assert_eq!(
        <f64 as Scalar>::from_basis_result(f64::from_bits(1)),
        Some(f64::from_bits(1))
    );
}

#[test]
fn tensor_errors_are_actionable_and_preserve_scalar_sources() {
    let cases = [
        (
            TensorError::SizeOverflow,
            "tensor element count exceeds usize",
        ),
        (
            TensorError::LengthMismatch {
                expected: 6,
                actual: 5,
            },
            "expected 6 values, got 5",
        ),
        (
            TensorError::ShapeMismatch {
                left: vec![2, 3],
                right: vec![3, 2],
            },
            "incompatible shapes: [2, 3] and [3, 2]",
        ),
        (
            TensorError::RankMismatch {
                expected: 2,
                actual: 1,
            },
            "expected rank 2, got 1",
        ),
        (
            TensorError::IndexOutOfBounds {
                axis: 1,
                index: 3,
                size: 3,
            },
            "index 3 outside axis 1 of length 3",
        ),
        (
            TensorError::NonContiguous,
            "canonical contiguous strides required",
        ),
        (
            TensorError::InnerDimensionMismatch { left: 3, right: 2 },
            "matmul inner dimensions differ: 3 and 2",
        ),
        (
            TensorError::AllocationFailed,
            "cannot reserve tensor output",
        ),
        (
            TensorError::OutputRange { index: 4 },
            "basis result at element 4 is outside the output dtype range",
        ),
    ];
    for (error, expected) in cases {
        assert_eq!(error.to_string(), expected);
        assert!(error.source().is_none());
    }
    let error = TensorError::Evaluation {
        index: 2,
        source: EvalError::Overflow,
    };
    assert_eq!(
        error.to_string(),
        "basis evaluation at element 2: floating-point overflow"
    );
    assert_eq!(
        error.source().unwrap().downcast_ref::<EvalError>(),
        Some(&EvalError::Overflow)
    );
}
