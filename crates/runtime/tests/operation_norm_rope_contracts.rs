use bml_compiler::bml_ops::{
    bml_rmsnorm_full_program, bml_rmsnorm_program, bml_rmsnorm_scale_program_checked,
    bml_rope_program, bml_rope_vector_program, RmsScaleBounds, RopeVectorConfig,
};
use bml_compiler::{symbolic_kernels::RmsBounds, Fragment, RpnOp, RpnProgram};
use bml_domain::EvalContext;
use bml_runtime::{buffer::ResultBuffer, HotLoop};

#[test]
fn compact_rope_retains_original_bounded_dag_bits_and_fits_one_fragment() {
    use bml_domain::{transformer::BoundedOpConfig, BMLTransformer};
    let input = [-3.0, 0.25, 0.0, -0.0, -1.5, 2.25, 4.0, -2.0];
    for bound in [4.0, 16.0] {
        let mut builder = BMLTransformer::new();
        let inputs = [builder.var(0), builder.var(1)];
        let coefficients = [builder.var(2), builder.var(3)];
        let roots = builder
            .operation_bounded(
                &inputs,
                &coefficients,
                &BoundedOpConfig::rope(bound).unwrap(),
            )
            .unwrap();
        let raw = builder.into_soa();
        let original: Vec<_> = roots
            .iter()
            .map(|&root| bml_compiler::linearize(&raw, root))
            .collect();
        for position in [0, 1, 127] {
            let config = RopeVectorConfig {
                heads: 2,
                head_dim: 4,
                position,
                frequency_base: 37.0,
                input_bound: bound,
            };
            let compiled = bml_rope_vector_program(0, 1, config).unwrap();
            compiled.validate_values(&input).unwrap();
            assert!(compiled.program().ops.len() * std::mem::size_of::<RpnOp>() <= 32768);
            let mut buffer = ResultBuffer::new(2, 8);
            buffer.slot_mut(0).copy_from_slice(&input);
            let context = EvalContext::new(&[], &compiled.program().const_pool);
            execute(compiled.program(), &context, &mut buffer);
            for pair in 0..4 {
                let angle = position as f64 * 37.0_f64.powf(-2.0 * (pair % 2) as f64 / 4.0);
                let (sin, cos) = angle.sin_cos();
                let values = [input[2 * pair], input[2 * pair + 1], cos, sin];
                for (component, scalar) in original.iter().enumerate() {
                    let expected =
                        scalar.evaluate_with_ctx(&EvalContext::new(&values, &scalar.const_pool));
                    assert_eq!(
                        buffer.read(1, (2 * pair + component) as u32).to_bits(),
                        expected.to_bits()
                    );
                }
            }
        }
    }
}

fn execute(program: &RpnProgram, context: &EvalContext, buffer: &mut ResultBuffer) {
    let mut hot = HotLoop::with_capacity(program.ops.len() + 64);
    hot.execute_fragment(&Fragment {
        ops: vec![RpnOp::One],
    });
    let fragment = Fragment {
        ops: program.ops.clone(),
    };
    for _ in 0..2 {
        hot.execute_fragment_full(&fragment, context, buffer);
        assert_eq!(hot.stack_depth(), 1, "temporary stack values leaked");
    }
    assert_eq!(hot.stack_pop(), Some(1.0));
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() < 1e-8,
            "element {index}: {actual} != {expected}"
        );
    }
}

fn norm_buffer(input: &[f64], weight: &[f64]) -> ResultBuffer {
    let mut buffer = ResultBuffer::new(4, 16);
    for slot in 0..4 {
        buffer.slot_mut(slot).fill(-999.0);
    }
    for (index, (&x, &w)) in input.iter().zip(weight).enumerate() {
        buffer.write_indexed(3, index as u32, x);
        buffer.write_indexed(20, index as u32, w);
    }
    buffer
}

fn unrelated_cells(before: &[f64], after: &[f64], output_start: usize, width: usize) {
    for (index, (&before, &after)) in before.iter().zip(after).enumerate() {
        if !(output_start..output_start + width).contains(&index) {
            assert_eq!(before.to_bits(), after.to_bits(), "unrelated cell {index}");
        }
    }
}

#[test]
fn legacy_rms_scale_writes_each_signed_output_using_precomputed_factor() {
    let input = [3.0, 4.0, -3.0, 0.0, -0.0];
    let weight = [2.0, -0.5, -2.0, 5.0, -5.0];
    let mut buffer = norm_buffer(&input, &weight);
    let before = buffer.data().to_vec();
    let program = bml_rmsnorm_program(3, 20, 5, 2, 3);
    assert!(program.const_pool.is_empty());
    assert_eq!(
        program
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::FMul))
            .count(),
        2
    );
    assert!(!program
        .ops
        .iter()
        .any(|op| matches!(op, RpnOp::Bml | RpnOp::FAdd)));
    execute(
        &program,
        &EvalContext::new(&[], &[f64::NAN, -17.0, 0.25]),
        &mut buffer,
    );
    close(&buffer.slot(3)[..5], &[1.5, -0.5, 1.5, 0.0, 0.0]);
    unrelated_cells(&before, buffer.data(), 48, 5);
    let empty = bml_rmsnorm_program(3, 20, 0, 2, 3);
    let before_empty = buffer.data().to_vec();
    execute(&empty, &EvalContext::new(&[], &[]), &mut buffer);
    assert_eq!(buffer.data(), before_empty);
}

#[test]
fn legacy_rms_scale_retains_native_ieee_range_and_signed_zero() {
    let input = [1000.0, -1000.0, -0.0, f64::INFINITY];
    let weight = [3.0, 0.5, 1.0, 0.0];
    let mut buffer = norm_buffer(&input, &weight);
    let program = bml_rmsnorm_program(3, 20, 4, 0, 3);
    execute(&program, &EvalContext::new(&[], &[2.0]), &mut buffer);
    assert_eq!(buffer.slot(3)[0], 6000.0);
    assert_eq!(buffer.slot(3)[1], -1000.0);
    assert_eq!(buffer.slot(3)[2].to_bits(), (-0.0_f64).to_bits());
    assert!(buffer.slot(3)[3].is_nan());
}

#[test]
fn configured_rms_scale_supports_inverse_above_one_and_checks_operands() {
    let bounds = RmsScaleBounds {
        input: 0.5,
        reciprocal: 4.0,
        weight: 8.0,
    };
    let input = [0.5, -0.25, 0.0];
    let weight = [-8.0, -4.0, 8.0];
    let compiled = bml_rmsnorm_scale_program_checked(3, 20, 3, 4.0, 3, bounds).unwrap();
    compiled.validate_values(&input, &weight).unwrap();
    let program = compiled.program();
    let mut buffer = norm_buffer(&input, &weight);
    execute(
        program,
        &EvalContext::new(&[], &program.const_pool),
        &mut buffer,
    );
    close(&buffer.slot(3)[..3], &[-16.0, 4.0, 0.0]);
    assert!(compiled.validate_values(&[0.6, 0.0, 0.0], &weight).is_err());
    assert!(compiled
        .validate_values(&input, &[1.0, f64::NAN, 1.0])
        .is_err());
    assert!(compiled.validate_values(&[], &[]).is_err());
    for reciprocal in [0.0, -1.0, 4.1, f64::INFINITY, f64::NAN] {
        assert!(bml_rmsnorm_scale_program_checked(3, 20, 3, reciprocal, 3, bounds).is_err());
    }
    assert!(bml_rmsnorm_scale_program_checked(u32::MAX, 0, 2, 1.0, 3, bounds).is_err());
    assert!(bml_rmsnorm_scale_program_checked(
        0,
        0,
        2,
        1.0,
        3,
        RmsScaleBounds {
            input: 256.0,
            reciprocal: 256.0,
            weight: 256.0
        }
    )
    .is_err());
}

#[test]
fn complete_rms_program_uses_shared_mean_epsilon_root_and_nonunit_weights() {
    let compiled = bml_rmsnorm_full_program(
        3,
        20,
        2,
        3.5,
        3,
        RmsBounds {
            input: 16.0,
            weight: 16.0,
        },
    )
    .unwrap();
    compiled.validate_values(&[3.0, 4.0], &[2.0, -0.5]).unwrap();
    let program = compiled.program();
    let mut buffer = norm_buffer(&[3.0, 4.0], &[2.0, -0.5]);
    execute(
        program,
        &EvalContext::new(&[], &program.const_pool),
        &mut buffer,
    );
    close(&buffer.slot(3)[..2], &[1.5, -0.5]);
    assert!(bml_rmsnorm_full_program(
        3,
        20,
        2,
        0.0,
        3,
        RmsBounds {
            input: 16.0,
            weight: 16.0
        }
    )
    .is_err());
}

#[test]
fn legacy_rope_reads_context_coordinates_and_writes_both_signed_components() {
    let program = bml_rope_program(3, 7, 2, 5, 2);
    assert!(program.const_pool.is_empty());
    assert!(!program
        .ops
        .iter()
        .any(|op| matches!(op, RpnOp::FMul | RpnOp::FAdd)));
    for (x, y, cosine, sine, expected) in [
        (3.0, 4.0, 0.0, 1.0, [-4.0, 3.0]),
        (-3.0, 4.0, 1.0, 0.0, [-3.0, 4.0]),
        (0.0, -2.0, 0.6, 0.8, [1.6, -1.2]),
        (-0.0, 0.0, -1.0, 0.0, [0.0, 0.0]),
    ] {
        let mut inputs = [f64::NAN; 8];
        inputs[3] = x;
        inputs[7] = y;
        let mut constants = [f64::NAN; 6];
        constants[2] = cosine;
        constants[5] = sine;
        let mut buffer = ResultBuffer::new(4, 8);
        for slot in 0..4 {
            buffer.slot_mut(slot).fill(-999.0);
        }
        let before = buffer.data().to_vec();
        execute(
            &program,
            &EvalContext::new(&inputs, &constants),
            &mut buffer,
        );
        close(&buffer.slot(2)[..2], &expected);
        unrelated_cells(&before, buffer.data(), 16, 2);
    }
}

fn run_rope(config: RopeVectorConfig, input: &[f64]) -> Vec<f64> {
    let compiled = bml_rope_vector_program(5, 2, config).unwrap();
    compiled.validate_values(input).unwrap();
    let size = input.len() + 8;
    let mut buffer = ResultBuffer::new(3, size);
    for slot in 0..3 {
        buffer.slot_mut(slot).fill(-999.0);
    }
    for (index, &value) in input.iter().enumerate() {
        buffer.write_indexed(5, index as u32, value);
    }
    let before = buffer.data().to_vec();
    let program = compiled.program();
    execute(
        program,
        &EvalContext::new(&[], &program.const_pool),
        &mut buffer,
    );
    unrelated_cells(&before, buffer.data(), 2 * size, input.len());
    buffer.slot(2)[..input.len()].to_vec()
}

#[test]
fn vector_rope_uses_distinct_frequencies_and_restarts_them_for_each_head() {
    let config = RopeVectorConfig {
        heads: 2,
        head_dim: 4,
        position: 3,
        frequency_base: 10000.0,
        input_bound: 16.0,
    };
    let input = [1.0, -2.0, 3.0, 4.0, -1.0, 2.0, -3.0, -4.0];
    let expected: Vec<_> = input
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(pair, p)| {
            let angle: f64 = if pair % 2 == 0 { 3.0 } else { 0.03 };
            [
                p[0] * angle.cos() - p[1] * angle.sin(),
                p[0] * angle.sin() + p[1] * angle.cos(),
            ]
        })
        .collect();
    close(&run_rope(config, &input), &expected);
}

#[test]
fn vector_rope_position_zero_covers_all_1024_pairs_with_constant_bytecode() {
    let config = RopeVectorConfig {
        heads: 1,
        head_dim: 2048,
        position: 0,
        frequency_base: 10000.0,
        input_bound: 16.0,
    };
    let input: Vec<_> = (0..2048).map(|index| (index % 31) as f64 - 15.0).collect();
    close(&run_rope(config, &input), &input);
    let large = bml_rope_vector_program(5, 2, config).unwrap();
    let small = bml_rope_vector_program(
        5,
        2,
        RopeVectorConfig {
            head_dim: 2,
            ..config
        },
    )
    .unwrap();
    assert_eq!(large.program().ops.len(), small.program().ops.len());
}

#[test]
fn vector_rope_rejects_bad_shapes_addresses_and_nonfinite_or_excessive_inputs() {
    let config = RopeVectorConfig {
        heads: 1,
        head_dim: 2,
        position: 2,
        frequency_base: 10000.0,
        input_bound: 2.5,
    };
    let compiled = bml_rope_vector_program(5, 2, config).unwrap();
    compiled.validate_values(&[-2.5, 2.5]).unwrap();
    assert!(compiled.validate_values(&[2.6, 0.0]).is_err());
    assert!(compiled.validate_values(&[f64::NAN, 0.0]).is_err());
    assert!(compiled.validate_values(&[]).is_err());
    for bad in [
        RopeVectorConfig { heads: 0, ..config },
        RopeVectorConfig {
            head_dim: 3,
            ..config
        },
        RopeVectorConfig {
            heads: u32::MAX,
            ..config
        },
        RopeVectorConfig {
            head_dim: 65538,
            ..config
        },
        RopeVectorConfig {
            frequency_base: 0.0,
            ..config
        },
        RopeVectorConfig {
            frequency_base: f64::NAN,
            ..config
        },
        RopeVectorConfig {
            input_bound: 241.0,
            ..config
        },
        RopeVectorConfig {
            input_bound: f64::NAN,
            ..config
        },
    ] {
        assert!(bml_rope_vector_program(5, 2, bad).is_err());
    }
    assert!(bml_rope_vector_program(u32::MAX, 2, config).is_err());
}
