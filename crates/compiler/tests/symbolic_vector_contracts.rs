use bml_compiler::{
    symbolic_kernels::{
        compile_operation, compile_operation_with_layout, CompiledKernel, KernelLimits, Operation,
        OutputLayout,
    },
    BmlGraph, RpnOp,
};
use bml_domain::BMLTransformer;
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
};

fn evaluate(kernel: CompiledKernel, inputs: &[f64], weights: &[f64]) -> Vec<f64> {
    assert_eq!(inputs.len(), kernel.signature.inputs);
    assert_eq!(weights.len(), kernel.signature.external);
    let mut constants = kernel.signature.fixed.clone();
    constants.extend_from_slice(weights);
    let (slots, size) = match kernel.layout {
        OutputLayout::Scalars => (0, 0),
        OutputLayout::Buffer => (1, kernel.signature.outputs),
    };
    let mut buffer = ResultBuffer::new(slots, size);
    if slots != 0 {
        buffer.slot_mut(0).fill(f64::NAN);
    }
    let mut scalars = Vec::new();
    for graph in kernel.graphs {
        assert!(graph.const_pool.is_empty());
        let serialized = graph.serialize();
        let stored = BmlGraph::deserialize(&serialized, graph.threshold).unwrap();
        let graph = ValidatedGraph::new_with_external_constants(
            stored,
            inputs.len(),
            constants.len(),
            slots,
            size,
            GraphLimits::default(),
        )
        .unwrap();
        scalars.push(
            graph
                .execute_with_constants(inputs, &constants, &mut buffer)
                .unwrap(),
        );
    }
    match kernel.layout {
        OutputLayout::Scalars => scalars,
        OutputLayout::Buffer => buffer.into_vec(),
    }
}

fn opcodes(kernel: &CompiledKernel) -> usize {
    kernel
        .graphs
        .iter()
        .flat_map(|graph| &graph.fragments)
        .map(|fragment| fragment.ops.len())
        .sum()
}

#[test]
fn persisted_vector_outputs_match_each_legacy_scalar_bit_for_bit() {
    for operation in [Operation::Norm, Operation::Softmax, Operation::Rope] {
        let widths: &[usize] = if operation == Operation::Rope {
            &[2]
        } else {
            &[1, 2, 4, 8]
        };
        for &width in widths {
            let limits = KernelLimits {
                fragment_bytes: 256,
                ..Default::default()
            };
            let vector = compile_operation(operation, width, 0.5, limits).unwrap();
            let scalar =
                compile_operation_with_layout(operation, width, 0.5, OutputLayout::Scalars, limits)
                    .unwrap();
            assert_eq!(vector.layout, OutputLayout::Buffer);
            assert_eq!(vector.graphs.len(), 1);
            assert_eq!(vector.signature.outputs, width);
            assert_eq!(scalar.graphs.len(), width);
            let prefix = operation.signature(width, 0.5).unwrap().fixed.len();
            assert_eq!(
                vector.signature.fixed[prefix..],
                (0..width).map(|i| i as f64).collect::<Vec<_>>()
            );
            let stores = vector.graphs[0]
                .fragments
                .iter()
                .flat_map(|fragment| &fragment.ops)
                .filter(|op| matches!(op, RpnOp::StoreResult { slot: 0 }))
                .count();
            assert_eq!(stores, width);
            let weights = match operation {
                Operation::Norm => (0..width)
                    .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
                    .collect(),
                Operation::Rope => vec![0.0, 1.0],
                _ => Vec::new(),
            };
            let input: Vec<_> = (0..width)
                .map(|i| if i % 2 == 0 { 0.125 } else { -0.25 })
                .collect();
            let vector = evaluate(vector, &input, &weights);
            let scalar = evaluate(scalar, &input, &weights);
            assert!(vector.iter().chain(&scalar).all(|value| value.is_finite()));
            assert_eq!(
                vector
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                scalar
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "operation={operation:?}, width={width}"
            );
        }
    }
}

#[test]
fn vector_program_storage_grows_linearly_with_operation_width() {
    for operation in [Operation::Norm, Operation::Softmax] {
        let mut counts = Vec::new();
        for width in [8, 16, 32] {
            let vector = compile_operation(operation, width, 0.5, KernelLimits::default()).unwrap();
            assert_eq!(vector.graphs.len(), 1);
            counts.push(opcodes(&vector));
            eprintln!(
                "SYMBOLIC_VECTOR_SIZE operation={} width={width} programs=1 opcodes={}",
                operation.name(),
                opcodes(&vector)
            );
        }
        // Doubling width adds the same per-element expansion, rather than
        // repeating the shared reduction for every output.
        assert_eq!(counts[2] - counts[1], 2 * (counts[1] - counts[0]));
        let scalar = compile_operation_with_layout(
            operation,
            32,
            0.5,
            OutputLayout::Scalars,
            KernelLimits::default(),
        )
        .unwrap();
        assert!(counts[2] < opcodes(&scalar));
        eprintln!(
            "SYMBOLIC_SCALAR_SIZE operation={} width=32 programs=32 opcodes={}",
            operation.name(),
            opcodes(&scalar)
        );
        // A shared program still obeys the per-program opcode cap. Increase
        // only that explicit cap to measure a wider vector under the unchanged
        // node/total-opcode budgets; the former node estimate rejected width128.
        let wide = compile_operation(
            operation,
            128,
            0.5,
            KernelLimits {
                max_output_ops: 1_000_000,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(wide.graphs.len(), 1);
        assert_eq!(opcodes(&wide) - counts[2], 6 * (counts[2] - counts[1]));
        eprintln!(
            "SYMBOLIC_VECTOR_SIZE operation={} width=128 programs=1 opcodes={}",
            operation.name(),
            opcodes(&wide)
        );
    }
}

#[test]
fn vector_compilation_respects_exact_node_and_persisted_program_budgets() {
    for operation in [Operation::Norm, Operation::Softmax] {
        let width = 8;
        let mut builder = BMLTransformer::new();
        let input: Vec<_> = (0..width).map(|i| builder.var(i as u32)).collect();
        if operation == Operation::Norm {
            let weights: Vec<_> = (0..width)
                .map(|i| builder.external_constant(i as u32))
                .collect();
            builder.rms_norm(&input, &weights, 0.5).unwrap();
        } else {
            builder.softmax(&input).unwrap();
        }
        let nodes = builder.soa().len();
        let limits = KernelLimits {
            max_nodes: nodes,
            max_programs: 1,
            ..Default::default()
        };
        let compiled = compile_operation(operation, width, 0.5, limits).unwrap();
        let count = opcodes(&compiled);
        assert!(compile_operation(
            operation,
            width,
            0.5,
            KernelLimits {
                max_nodes: nodes - 1,
                ..limits
            }
        )
        .is_err());
        assert!(compile_operation(
            operation,
            width,
            0.5,
            KernelLimits {
                max_programs: 0,
                ..limits
            }
        )
        .is_err());
        assert!(compile_operation_with_layout(
            operation,
            width,
            0.5,
            OutputLayout::Scalars,
            limits
        )
        .is_err());
        assert!(compile_operation(
            operation,
            width,
            0.5,
            KernelLimits {
                max_ops: count,
                max_output_ops: count,
                ..limits
            }
        )
        .is_ok());
        for budget in [
            KernelLimits {
                max_ops: count - 1,
                ..limits
            },
            KernelLimits {
                max_output_ops: count - 1,
                ..limits
            },
        ] {
            assert!(compile_operation(operation, width, 0.5, budget).is_err());
        }
        assert!(compile_operation(operation, usize::MAX, 0.5, limits).is_err());
    }
}
