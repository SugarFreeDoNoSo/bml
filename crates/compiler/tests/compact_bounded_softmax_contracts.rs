//! Compact shifted softmax retains the previous scalar identities and sum order.
use bml_compiler::{
    fragment_program,
    rpn::linearize_outputs,
    symbolic_bounded::{compile_operation, BoundedProfile},
    symbolic_kernels::{CompiledKernel, KernelLimits, Operation, OutputLayout},
    BmlGraph, RpnOp,
};
use bml_domain::{transformer::BoundedOpConfig, BMLTransformer};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
};

fn graph_limits() -> GraphLimits {
    GraphLimits {
        max_ops: 10_000_000,
        max_steps: 20_000_000,
        max_values: 10_000_000,
    }
}
fn validate(kernel: &CompiledKernel, limits: GraphLimits) -> Result<ValidatedGraph, GraphError> {
    let graph = &kernel.graphs[0];
    ValidatedGraph::new_with_external_constants(
        BmlGraph::deserialize(&graph.serialize(), graph.threshold).unwrap(),
        kernel.signature.inputs,
        kernel.signature.fixed.len(),
        1,
        kernel.width,
        limits,
    )
}
fn execute(kernel: &CompiledKernel, scores: &[f64], indexed: bool) -> Vec<f64> {
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut inputs = scores.to_vec();
    inputs.push(maximum);
    BoundedProfile::default()
        .validate_values(Operation::Softmax, scores.len(), 1e-5, &inputs, &[])
        .unwrap();
    let graph = validate(kernel, graph_limits()).unwrap();
    let mut buffer = ResultBuffer::from_vec(1, scores.len(), vec![f64::NAN; scores.len()]).unwrap();
    let result = if indexed {
        graph.execute_indexed_symbolic_with_constants(&inputs, &kernel.signature.fixed, &mut buffer)
    } else {
        graph.execute_symbolic_with_constants(&inputs, &kernel.signature.fixed, &mut buffer)
    };
    assert_eq!(result.unwrap(), 1.0);
    buffer.into_vec()
}
fn historical(width: usize) -> CompiledKernel {
    let profile = BoundedProfile::default();
    let config = BoundedOpConfig::softmax(width, profile.score).unwrap();
    let mut builder = BMLTransformer::new();
    let input: Vec<_> = (0..width + 1).map(|i| builder.var(i as u32)).collect();
    let roots = builder.operation_bounded(&input, &[], &config).unwrap();
    let mut dag = builder.into_soa();
    let signature = profile
        .signature(Operation::Softmax, width, 1e-5, OutputLayout::Buffer)
        .unwrap();
    assert_eq!(dag.const_pool, signature.fixed[..10]);
    dag.const_pool = signature.fixed.clone();
    let mut program = linearize_outputs(&dag, &roots, 0, 10).unwrap();
    program.const_pool.clear();
    CompiledKernel {
        operation: Operation::Softmax,
        width,
        signature,
        layout: OutputLayout::Buffer,
        graphs: vec![fragment_program(
            &program,
            KernelLimits::default().fragment_bytes,
        )],
    }
}
fn stored(kernel: &CompiledKernel) -> usize {
    kernel
        .graphs
        .iter()
        .flat_map(|graph| &graph.fragments)
        .map(|f| f.ops.len())
        .sum()
}

#[test]
fn compact_softmax_matches_historical_bits_and_independent_stable_oracle() {
    for width in [1, 288, 2048] {
        let current = compile_operation(
            Operation::Softmax,
            width,
            1e-5,
            BoundedProfile::default(),
            KernelLimits::default(),
        )
        .unwrap();
        assert_eq!(current.layout, OutputLayout::IndexedBuffer);
        let old = historical(width);
        assert_eq!(current.signature.fixed, old.signature.fixed[..10]);
        let scores: Vec<_> = (0..width)
            .map(|i| -16.0 + (i * 11 % 33) as f64 / 2.0)
            .collect();
        let actual = execute(&current, &scores, true);
        let previous = execute(&old, &scores, false);
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<_> = scores.iter().map(|x| (x - maximum).exp()).collect();
        let total = exps.iter().sum::<f64>();
        for (index, ((&a, &b), &exp)) in actual.iter().zip(&previous).zip(&exps).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "width {width}, output {index}");
            assert!(a.is_finite() && (a - exp / total).abs() < 2e-7);
        }
        assert!((actual.iter().sum::<f64>() - 1.0).abs() < 2e-7);
    }
    // An entirely negative vector must use its actual maximum, not the bound.
    let kernel = compile_operation(
        Operation::Softmax,
        3,
        1e-5,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let actual = execute(&kernel, &[-16.0, -15.0, -14.0], true);
    assert!(actual[0] < actual[1] && actual[1] < actual[2]);
    assert_eq!(
        execute(&kernel, &[-16.0; 3], true),
        execute(&historical(3), &[-16.0; 3], false)
    );
}

#[test]
fn compact_softmax_has_constant_code_literals_stack_and_linear_execution_work() {
    let mut measurements = Vec::new();
    for width in [1, 2, 288, 2048] {
        let kernel = compile_operation(
            Operation::Softmax,
            width,
            1e-5,
            BoundedProfile::default(),
            KernelLimits::default(),
        )
        .unwrap();
        assert_eq!(kernel.signature.fixed.len(), 10);
        assert_eq!(kernel.signature.inputs, width + 1);
        assert_eq!(kernel.graphs[0].fragments.len(), 1);
        let ops = &kernel.graphs[0].fragments[0].ops;
        assert!(ops.iter().all(|op| !matches!(
            op,
            RpnOp::FAdd | RpnOp::FMul | RpnOp::Zero | RpnOp::VarIndexed { .. }
        )));
        assert!(
            ops.iter()
                .any(|op| matches!(op, RpnOp::Loop { count: 0, .. }))
                == (width == 1)
        );
        let graph = validate(&kernel, graph_limits()).unwrap();
        let steps = graph.execution_steps();
        let stack = graph.stack_capacity();
        let values = kernel.signature.inputs + kernel.signature.fixed.len() + width;
        let exact = GraphLimits {
            max_ops: ops.len(),
            max_steps: steps,
            max_values: values + stack,
        };
        assert!(validate(&kernel, exact).is_ok());
        assert!(matches!(
            validate(
                &kernel,
                GraphLimits {
                    max_steps: steps - 1,
                    ..exact
                }
            ),
            Err(GraphError::Limit)
        ));
        assert!(matches!(
            validate(
                &kernel,
                GraphLimits {
                    max_values: values + stack - 1,
                    ..exact
                }
            ),
            Err(GraphError::Limit)
        ));
        measurements.push((width, ops.len(), stack, steps));
    }
    let slope = measurements[1].3 - measurements[0].3;
    for &(width, ops, stack, steps) in &measurements {
        assert_eq!(ops, measurements[0].1);
        // Width one skips the reduction loop, so its actual peak may be lower.
        // All nonempty reductions share one width-independent stack bound.
        assert!(stack <= measurements[1].2);
        if width > 1 {
            assert_eq!(stack, measurements[1].2);
        }
        assert_eq!(steps, measurements[0].3 + (width - 1) * slope);
    }
    println!("compact softmax (width, stored ops, peak stack, execution steps): {measurements:?}");
}

#[test]
fn compact_softmax_rejects_budget_exhaustion_and_invalid_checked_contexts() {
    let profile = BoundedProfile::default();
    let width = 288;
    let limits = KernelLimits::default();
    let kernel = compile_operation(Operation::Softmax, width, 1e-5, profile, limits).unwrap();
    let ops = stored(&kernel);
    let bytes = ops * std::mem::size_of::<RpnOp>();
    for restricted in [
        KernelLimits {
            max_nodes: 1,
            ..limits
        },
        KernelLimits {
            max_programs: 0,
            ..limits
        },
        KernelLimits {
            max_ops: ops - 1,
            ..limits
        },
        KernelLimits {
            max_output_ops: ops - 1,
            ..limits
        },
        KernelLimits {
            fragment_bytes: bytes - 1,
            ..limits
        },
    ] {
        assert!(compile_operation(Operation::Softmax, width, 1e-5, profile, restricted).is_err());
    }
    assert!(compile_operation(
        Operation::Softmax,
        width,
        1e-5,
        profile,
        KernelLimits {
            max_ops: ops,
            max_output_ops: ops,
            fragment_bytes: bytes,
            max_programs: 1,
            ..limits
        }
    )
    .is_ok());
    assert!(compile_operation(Operation::Softmax, 0, 1e-5, profile, limits).is_err());
    let mut invalid = vec![-1.0; width + 1];
    invalid[width] = 0.0;
    assert!(profile
        .validate_values(Operation::Softmax, width, 1e-5, &invalid, &[])
        .is_err());
    invalid[0] = f64::NAN;
    let graph = validate(&kernel, graph_limits()).unwrap();
    let mut buffer = ResultBuffer::new(1, width);
    assert!(matches!(
        graph.execute_indexed_symbolic_with_constants(
            &invalid,
            &kernel.signature.fixed,
            &mut buffer
        ),
        Err(GraphError::Numerical)
    ));
    let mut constants = kernel.signature.fixed.clone();
    constants[5] = 0.0; // The positive reciprocal operand bound cannot be zero.
    assert!(matches!(
        graph.execute_indexed_symbolic_with_constants(
            &vec![-1.0; width + 1],
            &constants,
            &mut buffer
        ),
        Err(GraphError::Numerical)
    ));
}
