//! Compact v4 RMS preserves the numerical formula and bindings of v3.
use bml_compiler::{
    symbolic_bounded::{compile_operation, BoundedProfile},
    symbolic_kernels::{
        compile_operation_with_rms_bounds, CompiledKernel, KernelLimits, Operation, OutputLayout,
        RmsBounds,
    },
    BmlGraph, RpnOp,
};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
};

fn profile() -> BoundedProfile {
    BoundedProfile {
        rms: RmsBounds {
            input: 16.0,
            weight: 8.0,
        },
        ..BoundedProfile::default()
    }
}
fn graph(kernel: &CompiledKernel, limits: GraphLimits) -> Result<ValidatedGraph, GraphError> {
    assert_eq!(kernel.graphs.len(), 1);
    let stored = &kernel.graphs[0];
    ValidatedGraph::new_with_external_constants(
        BmlGraph::deserialize(&stored.serialize(), stored.threshold).unwrap(),
        kernel.signature.inputs,
        kernel.signature.fixed.len() + kernel.signature.external,
        1,
        kernel.width,
        limits,
    )
}
fn graph_limits() -> GraphLimits {
    GraphLimits {
        max_ops: 10_000_000,
        max_steps: 20_000_000,
        max_values: 10_000_000,
    }
}
fn execute(kernel: &CompiledKernel, input: &[f64], weights: &[f64], indexed: bool) -> Vec<f64> {
    let validated = graph(kernel, graph_limits()).unwrap();
    let mut constants = kernel.signature.fixed.clone();
    constants.extend_from_slice(weights);
    let mut buffer = ResultBuffer::from_vec(1, kernel.width, vec![f64::NAN; kernel.width]).unwrap();
    let result = if indexed {
        validated.execute_indexed_symbolic_with_constants(input, &constants, &mut buffer)
    } else {
        validated.execute_symbolic_with_constants(input, &constants, &mut buffer)
    };
    assert_eq!(result.unwrap(), 1.0);
    buffer.into_vec()
}
fn ops(kernel: &CompiledKernel) -> Vec<RpnOp> {
    kernel
        .graphs
        .iter()
        .flat_map(|g| &g.fragments)
        .flat_map(|f| &f.ops)
        .copied()
        .collect()
}

#[test]
fn compact_rms_preserves_v3_bits_and_independent_oracle_at_model_widths() {
    let epsilon = 1e-5;
    for width in [1, 288, 2048] {
        let current = compile_operation(
            Operation::Norm,
            width,
            epsilon,
            profile(),
            KernelLimits::default(),
        )
        .unwrap();
        let historical = compile_operation_with_rms_bounds(
            Operation::Norm,
            width,
            epsilon,
            OutputLayout::Buffer,
            Some(profile().rms),
            KernelLimits {
                max_nodes: 5_000_000,
                max_ops: 10_000_000,
                max_output_ops: 10_000_000,
                ..KernelLimits::default()
            },
        )
        .unwrap();
        assert_eq!(current.layout, OutputLayout::IndexedBuffer);
        assert_eq!(current.signature.inputs, historical.signature.inputs);
        assert_eq!(current.signature.external, historical.signature.external);
        assert_eq!(current.signature.outputs, historical.signature.outputs);
        assert_eq!(current.signature.fixed, historical.signature.fixed[..17]);
        assert_eq!(
            profile()
                .signature(Operation::Norm, width, epsilon, OutputLayout::Buffer)
                .unwrap(),
            historical.signature
        );
        assert!(historical.graphs.iter().all(|g| g.const_pool.is_empty()));
        assert!(!ops(&historical).iter().any(|op| matches!(
            op,
            RpnOp::Loop { .. }
                | RpnOp::ContextVarIndexed { .. }
                | RpnOp::ContextConstIndexed { .. }
        )));
        let inputs: Vec<_> = (0..width)
            .map(|i| ((i * 17 % 31) as f64 - 15.0) / 4.0)
            .collect();
        let weights: Vec<_> = (0..width)
            .map(|i| ((i * 13 % 29) as f64 - 14.0) / 2.0)
            .collect();
        profile()
            .validate_values(Operation::Norm, width, epsilon, &inputs, &weights)
            .unwrap();
        let actual = execute(&current, &inputs, &weights, true);
        let old = execute(&historical, &inputs, &weights, false);
        let divisor = (inputs.iter().map(|x| x * x).sum::<f64>() / width as f64 + epsilon).sqrt();
        for (index, ((&a, &b), (&x, &w))) in actual
            .iter()
            .zip(&old)
            .zip(inputs.iter().zip(&weights))
            .enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "width {width}, output {index}");
            let expected = x / divisor * w;
            assert!(
                (a - expected).abs() < 2e-7,
                "width {width}, output {index}: {a} != {expected}"
            );
        }
    }
}

#[test]
fn compact_rms_storage_and_peak_stack_are_constant_with_exact_linear_step_budgets() {
    let mut measurements = Vec::new();
    for width in [1, 2, 288, 2048] {
        let kernel = compile_operation(
            Operation::Norm,
            width,
            1e-5,
            profile(),
            KernelLimits::default(),
        )
        .unwrap();
        assert_eq!(kernel.graphs[0].fragments.len(), 1);
        let instructions = ops(&kernel);
        assert!(instructions.iter().all(|op| !matches!(
            op,
            RpnOp::FAdd | RpnOp::FMul | RpnOp::Zero | RpnOp::VarIndexed { .. }
        )));
        assert_eq!(
            instructions
                .iter()
                .filter(|op| matches!(op, RpnOp::Loop { .. }))
                .count(),
            2
        );
        assert!(instructions
            .iter()
            .any(|op| matches!(op, RpnOp::ContextConstIndexed { base } if *base == 17)));
        assert_eq!(kernel.signature.fixed.len(), 17);
        let validated = graph(&kernel, graph_limits()).unwrap();
        let stack = validated.stack_capacity();
        let steps = validated.execution_steps();
        let values = kernel.signature.inputs
            + kernel.signature.fixed.len()
            + kernel.signature.external
            + width;
        let exact = GraphLimits {
            max_ops: instructions.len(),
            max_steps: steps,
            max_values: values + stack,
        };
        assert!(graph(&kernel, exact).is_ok());
        assert!(matches!(
            graph(
                &kernel,
                GraphLimits {
                    max_steps: steps - 1,
                    ..exact
                }
            ),
            Err(GraphError::Limit)
        ));
        assert!(matches!(
            graph(
                &kernel,
                GraphLimits {
                    max_values: values + stack - 1,
                    ..exact
                }
            ),
            Err(GraphError::Limit)
        ));
        measurements.push((width, instructions.len(), stack, steps));
    }
    let slope = measurements[1].3 - measurements[0].3;
    for &(width, stored, stack, steps) in &measurements {
        assert_eq!(stored, measurements[0].1);
        assert_eq!(stack, measurements[0].2);
        assert_eq!(steps, measurements[0].3 + (width - 1) * slope);
    }
    println!("compact RMS (width, stored ops, peak stack, execution steps): {measurements:?}");
}

#[test]
fn compact_rms_compilation_preserves_node_opcode_program_and_atomic_loop_limits() {
    let width = 2048;
    let limits = KernelLimits::default();
    let kernel = compile_operation(Operation::Norm, width, 1e-5, profile(), limits).unwrap();
    let count = ops(&kernel).len();
    let bytes = count * std::mem::size_of::<RpnOp>();
    for restricted in [
        KernelLimits {
            max_nodes: 0,
            ..limits
        },
        KernelLimits {
            max_nodes: 100,
            ..limits
        },
        KernelLimits {
            max_programs: 0,
            ..limits
        },
        KernelLimits {
            max_ops: count - 1,
            ..limits
        },
        KernelLimits {
            max_output_ops: count - 1,
            ..limits
        },
        KernelLimits {
            fragment_bytes: bytes - 1,
            ..limits
        },
    ] {
        assert!(compile_operation(Operation::Norm, width, 1e-5, profile(), restricted).is_err());
    }
    let exact = KernelLimits {
        max_ops: count,
        max_output_ops: count,
        fragment_bytes: bytes,
        max_programs: 1,
        ..limits
    };
    assert!(compile_operation(Operation::Norm, width, 1e-5, profile(), exact).is_ok());
    assert!(
        compile_operation(
            Operation::Norm,
            width,
            1e-5,
            BoundedProfile::default(),
            limits
        )
        .is_err(),
        "width2048/weight16 is outside the declared normalized-product domain"
    );
    assert!(compile_operation(Operation::Norm, 0, 1e-5, profile(), limits).is_err());
}

#[test]
fn compact_rms_keeps_checked_failure_and_dynamic_context_semantics() {
    let kernel = compile_operation(
        Operation::Norm,
        288,
        1e-5,
        profile(),
        KernelLimits::default(),
    )
    .unwrap();
    for inputs in [
        vec![0.0; 288],
        (0..288)
            .map(|i| if i % 2 == 0 { -16.0 } else { 16.0 })
            .collect(),
    ] {
        let weights = vec![-8.0; 288];
        profile()
            .validate_values(Operation::Norm, 288, 1e-5, &inputs, &weights)
            .unwrap();
        let actual = execute(&kernel, &inputs, &weights, true);
        let divisor = (inputs.iter().map(|x| x * x).sum::<f64>() / 288.0 + 1e-5).sqrt();
        for (&a, &x) in actual.iter().zip(&inputs) {
            assert!((a - x / divisor * -8.0).abs() < 2e-7);
        }
    }
    let mut constants = kernel.signature.fixed.clone();
    constants.extend(vec![1.0; 288]);
    let validated = graph(&kernel, graph_limits()).unwrap();
    let mut buffer = ResultBuffer::new(1, 288);
    let invalid = vec![f64::INFINITY; 288];
    assert!(matches!(
        validated.execute_indexed_symbolic_with_constants(&invalid, &constants, &mut buffer),
        Err(GraphError::Numerical)
    ));
    // The reciprocal width is required positive by the original BML formula.
    // Corrupting a finite literal must fail in the checked arithmetic path.
    constants[5] = 0.0;
    assert!(matches!(
        validated.execute_indexed_symbolic_with_constants(&vec![1.0; 288], &constants, &mut buffer),
        Err(GraphError::Numerical)
    ));
}

#[test]
fn compact_rms_large_input_bound_matches_independent_normalization() {
    let width = 2048;
    let epsilon = f64::from(1e-5_f32);
    let profile = BoundedProfile {
        rms: RmsBounds {
            input: 160.0,
            weight: 1.0,
        },
        ..BoundedProfile::default()
    };
    let kernel = compile_operation(
        Operation::Norm,
        width,
        epsilon,
        profile,
        KernelLimits::default(),
    )
    .unwrap();
    let weights: Vec<_> = (0..width).map(|i| ((i % 9) as f64 - 4.0) / 4.0).collect();
    for case in 0..4 {
        let input: Vec<_> = (0..width)
            .map(|i| match case {
                0 => 0.0,
                1 => ((i % 17) as f64 - 8.0) / 800.0,
                2 => {
                    if i == 624 {
                        -141.791_535_630_283_93
                    } else {
                        0.0
                    }
                }
                _ => ((i % 17) as f64 - 8.0) * 20.0,
            })
            .collect();
        profile
            .validate_values(Operation::Norm, width, epsilon, &input, &weights)
            .unwrap();
        let actual = execute(&kernel, &input, &weights, true);
        let divisor = (input.iter().map(|x| x * x).sum::<f64>() / width as f64 + epsilon).sqrt();
        let mut max_error = 0.0_f64;
        for (index, ((&observed, &x), &weight)) in
            actual.iter().zip(&input).zip(&weights).enumerate()
        {
            let expected = x / divisor * weight;
            max_error = max_error.max((observed - expected).abs());
            assert!(
                (observed - expected).abs() < 2e-7,
                "case={case} index={index}: {observed} != {expected}"
            );
        }
        eprintln!("case={case} max_abs_error={max_error}");
    }
}
