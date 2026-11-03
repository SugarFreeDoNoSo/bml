#![cfg(feature = "tensor-simd")]
use bml_domain::tensor::{Shape, Tensor, TensorError, TensorView};
use bml_kernels::tensor::{add, mul, ExecutedBackend, Policy};

#[test]
fn vector_lanes_tails_and_ieee_edges_match_scalar_operations() {
    let special = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut seed = 42u64;
    for length in 0..70 {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..length {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            a.push(if i < special.len() {
                special[i]
            } else {
                f64::from_bits(seed)
            });
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            b.push(if i < special.len() {
                special[special.len() - 1 - i]
            } else {
                f64::from_bits(seed)
            });
        }
        let shape = Shape::new(vec![length]).unwrap();
        let left = Tensor::from_vec(shape.clone(), a.clone()).unwrap();
        let right = Tensor::from_vec(shape, b.clone()).unwrap();
        for policy in [Policy::Reference, Policy::PreferAvx2] {
            for multiply in [false, true] {
                let out = if multiply {
                    mul(&left.view(), &right.view(), policy)
                } else {
                    add(&left.view(), &right.view(), policy)
                }
                .unwrap();
                assert_eq!(out.tensor.shape(), left.shape());
                let expected_backend =
                    if policy == Policy::PreferAvx2 && bml_kernels::Backend::Avx2.available() {
                        ExecutedBackend::Avx2
                    } else {
                        ExecutedBackend::Reference
                    };
                assert_eq!(out.backend, expected_backend);
                for (i, actual) in out.tensor.as_slice().iter().enumerate() {
                    let expected = if multiply { a[i] * b[i] } else { a[i] + b[i] };
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "length={length} lane={i}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn shape_contracts_scalar_empty_mismatch_and_unaligned_views() {
    for dims in [vec![], vec![0, 3], vec![2, 3]] {
        let shape = Shape::new(dims).unwrap();
        let left = Tensor::from_vec(shape.clone(), vec![2.0; shape.len()]).unwrap();
        let right = Tensor::from_vec(shape.clone(), vec![3.0; shape.len()]).unwrap();
        assert_eq!(
            add(&left.view(), &right.view(), Policy::PreferAvx2)
                .unwrap()
                .tensor
                .as_slice(),
            vec![5.0; shape.len()]
        );
    }
    let storage = [99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 99.0];
    let view = TensorView::from_slice(Shape::new(vec![5]).unwrap(), &storage[1..6]).unwrap();
    assert_eq!(
        mul(&view, &view, Policy::PreferAvx2)
            .unwrap()
            .tensor
            .as_slice(),
        &[1.0, 4.0, 9.0, 16.0, 25.0]
    );
    let a = Tensor::from_vec(Shape::new(vec![2, 3]).unwrap(), vec![1.0; 6]).unwrap();
    let b = Tensor::from_vec(Shape::new(vec![3, 2]).unwrap(), vec![1.0; 6]).unwrap();
    for policy in [Policy::Reference, Policy::PreferAvx2] {
        assert!(matches!(
            add(&a.view(), &b.view(), policy),
            Err(TensorError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            mul(&a.view(), &b.view(), policy),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }
}

#[test]
fn signed_zero_subnormal_overflow_and_nan_cartesian_pairs() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let a: Vec<_> = values
        .iter()
        .flat_map(|x| values.iter().map(move |_| *x))
        .collect();
    let b: Vec<_> = values.iter().flat_map(|_| values).collect();
    let shape = Shape::new(vec![a.len()]).unwrap();
    let left = TensorView::from_slice(shape.clone(), &a).unwrap();
    let right = TensorView::from_slice(shape, &b).unwrap();
    for multiply in [false, true] {
        let output = if multiply {
            mul(&left, &right, Policy::PreferAvx2)
        } else {
            add(&left, &right, Policy::PreferAvx2)
        }
        .unwrap();
        for (i, actual) in output.tensor.as_slice().iter().enumerate() {
            let expected = if multiply { a[i] * b[i] } else { a[i] + b[i] };
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "pair={i}");
            }
        }
    }
}
