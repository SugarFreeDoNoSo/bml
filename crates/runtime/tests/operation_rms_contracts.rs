use bml_compiler::{
    op_fragments::{compile_rmsnorm_fragment, compile_rmsnorm_fragment_checked},
    symbolic_kernels::RmsBounds,
    Fragment, RpnOp,
};
use bml_domain::EvalContext;
use bml_runtime::{buffer::ResultBuffer, HotLoop, Runtime};

fn reference(input: &[f64], weights: &[f64], epsilon: f64) -> Vec<f64> {
    let root = (input.iter().map(|x| x * x).sum::<f64>() / input.len() as f64 + epsilon).sqrt();
    input
        .iter()
        .zip(weights)
        .map(|(x, w)| x * w / root)
        .collect()
}

#[test]
fn complete_rms_fragments_match_independent_normalization_and_preserve_storage() {
    let cases = [
        (vec![3.0, 4.0], vec![2.0, -0.5], 3.5),
        (vec![0.0, -0.0, 0.0], vec![1.0, -2.0, 0.0], 1e-5),
        (
            vec![-1.0, 0.5, 2.0, -3.0],
            vec![-2.0, 0.0, 1.25, -0.5],
            0.125,
        ),
        (vec![1.0], vec![4.0], 1e-5),
    ];
    for (inputs, weights, epsilon) in cases {
        let n = inputs.len();
        let size = n + 3;
        let input_base = 1;
        let weight_base = size as u32 + 2;
        let compiled = compile_rmsnorm_fragment_checked(
            "norm",
            input_base,
            2,
            weight_base,
            n as u32,
            epsilon,
            RmsBounds {
                input: 16.0,
                weight: 16.0,
            },
        )
        .unwrap();
        compiled.validate_values(&inputs, &weights).unwrap();
        let op = compiled.operation();
        assert_eq!(op.const_pool.as_deref(), Some(compiled.constants()));
        assert_eq!(compiled.constants(), compiled.config().fixed_literals());
        assert!(op
            .fragment
            .ops
            .iter()
            .all(|op| !matches!(op, RpnOp::FMul | RpnOp::FAdd)));
        let mut buffer = ResultBuffer::new(4, size);
        for slot in 0..4 {
            buffer.slot_mut(slot).fill(77.0);
        }
        for (i, &x) in inputs.iter().enumerate() {
            buffer.write_indexed(input_base, i as u32, x);
        }
        for (i, &w) in weights.iter().enumerate() {
            buffer.write_indexed(weight_base, i as u32, w);
        }
        let before = buffer.data().to_vec();
        let expected = reference(&inputs, &weights, epsilon);
        let mut hot = HotLoop::with_capacity(op.fragment.ops.len() + 1);
        hot.execute_fragment(&Fragment {
            ops: vec![RpnOp::One],
        });
        let parent = EvalContext::new(&[], &[-999.0]);
        let context = op.context(&parent);
        for _ in 0..3 {
            hot.execute_fragment_full(&op.fragment, &context, &mut buffer);
            assert_eq!(hot.stack_depth(), 1);
            for (&actual, &expected) in buffer.slot(2).iter().zip(&expected) {
                assert!(
                    actual.is_finite() && (actual - expected).abs() < 2e-7,
                    "actual={actual}, expected={expected}"
                );
            }
        }
        assert_eq!(hot.stack_pop(), Some(1.0));
        for (i, (&before, &after)) in before.iter().zip(buffer.data()).enumerate() {
            if !(2 * size..2 * size + n).contains(&i) {
                assert_eq!(after.to_bits(), before.to_bits());
            }
        }
        if inputs == [3.0, 4.0] {
            assert!((buffer.read(2, 0) - 1.5).abs() < 2e-7);
            assert!((buffer.read(2, 1) + 0.5).abs() < 2e-7);
        }
    }
}

#[test]
fn complete_rms_supports_large_widths_with_constant_code_and_stack() {
    let mut code_sizes = Vec::new();
    for n in [1, 32, 288, 2048] {
        let inputs: Vec<_> = (0..n).map(|i| ((i * 7 % 19) as f64 - 9.0) / 4.0).collect();
        let weights: Vec<_> = (0..n).map(|i| ((i * 11 % 13) as f64 - 6.0) / 4.0).collect();
        let compiled = compile_rmsnorm_fragment_checked(
            "norm",
            0,
            2,
            n as u32,
            n as u32,
            1e-5,
            RmsBounds {
                input: 4.0,
                weight: 4.0,
            },
        )
        .unwrap();
        compiled.validate_values(&inputs, &weights).unwrap();
        let operation = compiled.operation();
        code_sizes.push(operation.fragment.ops.len());
        let mut buffer = ResultBuffer::new(3, n);
        buffer.slot_mut(0).copy_from_slice(&inputs);
        buffer.slot_mut(1).copy_from_slice(&weights);
        let expected = reference(&inputs, &weights, 1e-5);
        let mut hot = HotLoop::with_capacity(operation.fragment.ops.len());
        let capacity = hot.stack_capacity();
        let context = operation.context(&EvalContext::new(&[], &[]));
        hot.execute_fragment_full(&operation.fragment, &context, &mut buffer);
        assert_eq!(hot.stack_depth(), 0);
        assert_eq!(hot.stack_capacity(), capacity);
        for (&actual, &expected) in buffer.slot(2).iter().zip(&expected) {
            assert!(actual.is_finite() && (actual - expected).abs() < 2e-7);
        }
    }
    assert!(code_sizes.iter().all(|size| *size == code_sizes[0]));
    assert!(code_sizes[0] * std::mem::size_of::<RpnOp>() < 32768);
}

#[test]
fn complete_rms_rejects_bad_declarations_shapes_and_addresses() {
    let bounds = RmsBounds {
        input: 4.0,
        weight: 4.0,
    };
    for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(compile_rmsnorm_fragment_checked("bad", 0, 2, 2, 2, epsilon, bounds).is_err());
    }
    for (input, weight, width) in [(u32::MAX, 0, 2), (0, u32::MAX, 2), (0, 0, 0), (0, 0, 65026)] {
        assert!(
            compile_rmsnorm_fragment_checked("bad", input, 2, weight, width, 1e-5, bounds).is_err()
        );
    }
    let checked = compile_rmsnorm_fragment_checked("norm", 0, 2, 2, 2, 1e-5, bounds).unwrap();
    for (inputs, weights) in [
        (vec![1.0], vec![1.0, 1.0]),
        (vec![1.0, 1.0], vec![1.0]),
        (vec![4.01, 1.0], vec![1.0, 1.0]),
        (vec![1.0, 1.0], vec![1.0, -4.01]),
        (vec![f64::NAN, 0.0], vec![1.0, 1.0]),
        (vec![0.0, 1.0], vec![f64::INFINITY, 1.0]),
    ] {
        assert!(checked.validate_values(&inputs, &weights).is_err());
    }
}

#[test]
fn default_rms_runtime_binds_owned_constants_and_zero_width_is_noop() {
    let mut buffer = ResultBuffer::new(4, 4);
    buffer.slot_mut(0).copy_from_slice(&[3.0, -4.0, 0.0, 1.0]);
    buffer.slot_mut(1).copy_from_slice(&[2.0, -0.5, 1.0, 0.0]);
    let expected = reference(buffer.slot(0), buffer.slot(1), 1e-5);
    let operation = compile_rmsnorm_fragment("default", 0, 2, 4, 4);
    let parent = EvalContext::new(&[], &[f64::NAN]);
    let mut runtime = Runtime::new(2048, 0);
    runtime.execute_fragments_sequential(&[operation], &parent, &mut buffer);
    for (&actual, &expected) in buffer.slot(2).iter().zip(&expected) {
        assert!(actual.is_finite() && (actual - expected).abs() < 2e-7);
    }
    let before = buffer.data().to_vec();
    runtime.execute_fragments_sequential(
        &[compile_rmsnorm_fragment(
            "empty",
            u32::MAX,
            u32::MAX,
            u32::MAX,
            0,
        )],
        &parent,
        &mut buffer,
    );
    assert_eq!(buffer.data(), before);
    assert_eq!(runtime.stack_depth(), 0);
    // The historical constructor declares dimension-dependent weight headroom.
    let large = compile_rmsnorm_fragment("wide", 0, 2, 2048, 2048);
    assert!(large.const_pool.is_some());
}

#[test]
fn independent_rms_fragments_use_distinct_owned_epsilon_on_parallel_workers() {
    let bounds = RmsBounds {
        input: 16.0,
        weight: 16.0,
    };
    let first = compile_rmsnorm_fragment_checked("first", 0, 3, 4, 2, 3.5, bounds).unwrap();
    let second = compile_rmsnorm_fragment_checked("second", 2, 4, 4, 2, 0.5, bounds).unwrap();
    let a = [3.0, 4.0];
    let b = [-1.0, 2.0];
    let weights = [2.0, -0.5];
    let expected_a = reference(&a, &weights, 3.5);
    let expected_b = reference(&b, &weights, 0.5);
    first.validate_values(&a, &weights).unwrap();
    second.validate_values(&b, &weights).unwrap();
    let fragments = [first.into_operation(), second.into_operation()];
    for cores in [1, 2, 8] {
        let mut buffer = ResultBuffer::new(5, 2);
        buffer.slot_mut(0).copy_from_slice(&a);
        buffer.slot_mut(1).copy_from_slice(&b);
        buffer.slot_mut(2).copy_from_slice(&weights);
        let context = EvalContext::new(&[], &[f64::NAN]);
        Runtime::new(2048, 0).execute_with_cores(&fragments, &context, &mut buffer, cores);
        for (slot, expected) in [(3, &expected_a), (4, &expected_b)] {
            for (&actual, &expected) in buffer.slot(slot).iter().zip(expected) {
                assert!(actual.is_finite() && (actual - expected).abs() < 2e-7);
            }
        }
    }
}
