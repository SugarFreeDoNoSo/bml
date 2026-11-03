use bml_compiler::{
    fragment_program, gguf_compiler::InferenceCompiler, symbolic_projection::compile_row, BmlGraph,
    RpnOp,
};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
};
#[path = "support/causal_fixture.rs"]
mod causal_fixture;

#[test]
fn symbolic_row_templates_execute_mapped_gguf_weights_without_serializing_them() {
    let source = causal_fixture::fixture(false);
    let model = InferenceCompiler::open_mapped(&source.0).unwrap();
    assert!(model.mapped_bytes() > 0);
    let compiled = compile_row(4, 10_000).unwrap();
    assert_eq!(compiled.columns, 4);
    assert!(compiled.program.const_pool.is_empty());
    assert!(compiled.program.ops.iter().all(|op| matches!(
        op,
        RpnOp::One
            | RpnOp::Var(_)
            | RpnOp::Const(_)
            | RpnOp::Bml
            | RpnOp::Pick { .. }
            | RpnOp::Drop
            | RpnOp::Dup
            | RpnOp::Swap
    )));
    for index in 0..4 {
        assert!(compiled.program.ops.contains(&RpnOp::Var(index)));
        assert!(compiled.program.ops.contains(&RpnOp::Const(index)));
    }
    let graph = fragment_program(&compiled.program, 256);
    let bytes = graph.serialize();
    let stored = BmlGraph::deserialize(&bytes, 256).unwrap();
    assert!(stored.const_pool.is_empty());
    assert!(matches!(
        ValidatedGraph::new(stored.clone(), 4, 0, 0, GraphLimits::default()),
        Err(GraphError::Reference)
    ));
    let executable =
        ValidatedGraph::new_with_external_constants(stored, 4, 4, 0, 0, GraphLimits::default())
            .unwrap();
    for input in [[0.25, -0.5, 0.0, 0.125], [-0.25, 0.5, -0.125, 0.0]] {
        let output = model
            .project_rows(&input, "blk.0.attn_v.weight", |x, row| {
                // Quantized/F32 source rows are decoded on demand. Only this small
                // per-execution f64 conversion is owned; the graph stores no row.
                let constants: Vec<_> = row.iter().map(|&w| f64::from(w)).collect();
                executable
                    .execute_with_constants(x, &constants, &mut ResultBuffer::new(0, 0))
                    .map_err(|e| e.to_string())
            })
            .unwrap();
        // Fixture V selects the first two embedding coordinates independently
        // of this compiler and of the native projection implementation.
        assert_eq!(output.len(), 2);
        for (actual, expected) in output.iter().zip(&input[..2]) {
            assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
        }
    }
    assert_eq!(graph.serialize(), bytes);
}

#[test]
fn row_node_budget_is_exact_and_rejects_overflow_before_building() {
    for columns in [1, 2, 4, 7] {
        let row = compile_row(columns, 100_000).unwrap();
        let exact = compile_row(columns, row.nodes).unwrap();
        assert_eq!(exact.nodes, row.nodes);
        assert_eq!(exact.program.ops, row.program.ops);
        assert!(compile_row(columns, row.nodes - 1).is_err());
    }
    assert!(compile_row(0, usize::MAX).is_err());
    assert!(compile_row(1, 0).is_err());
    assert!(compile_row(usize::MAX, usize::MAX).is_err());
    assert!(compile_row(u32::MAX as usize, usize::MAX).is_err());
}
