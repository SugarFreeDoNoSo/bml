//! The five operation sub-DAGs must execute as persisted BML-only arithmetic,
//! not merely agree in a bespoke DAG interpreter or native-kernel fallback.
use bml_compiler::{fragment_program, linearize, BmlGraph, RpnOp};
use bml_domain::{BMLTransformer, NodeId};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
};

fn persisted_outputs(t: BMLTransformer, roots: &[NodeId], inputs: &[f64], expected: &[f64]) {
    assert_eq!(roots.len(), expected.len());
    let soa = t.into_soa();
    for (&root, &expected) in roots.iter().zip(expected) {
        let program = linearize(&soa, root);
        assert!(program.ops.iter().any(|op| matches!(op, RpnOp::Bml)));
        assert!(
            program.ops.iter().all(|op| matches!(
                op,
                RpnOp::One
                    | RpnOp::Const(_)
                    | RpnOp::Var(_)
                    | RpnOp::Bml
                    | RpnOp::Dup
                    | RpnOp::Pick { .. }
                    | RpnOp::Drop
                    | RpnOp::Swap
            )),
            "no native arithmetic or opaque operation is permitted"
        );
        let graph = fragment_program(&program, 256);
        assert!(graph.fragments.len() > 1);
        let bytes = graph.serialize();
        let stored = BmlGraph::deserialize(&bytes, 256).unwrap();
        assert_eq!(stored.const_pool, soa.const_pool);
        let validated =
            ValidatedGraph::new(stored, inputs.len(), 0, 0, GraphLimits::default()).unwrap();
        let value = validated
            .execute(inputs, &mut ResultBuffer::new(0, 0))
            .unwrap();
        assert!((value - expected).abs() < 2e-7, "{value} != {expected}");
    }
}

#[test]
fn all_transformer_operations_execute_after_bml_only_serialization() {
    let mut t = BMLTransformer::new();
    let x: Vec<_> = (0..4).map(|i| t.var(i)).collect();
    let w: Vec<_> = [0.5, -0.25, 0.0, 0.5]
        .iter()
        .map(|&v| t.constant(v))
        .collect();
    let roots = t.matmul(&x, &w, 2, 2, 2).unwrap();
    persisted_outputs(
        t,
        &roots,
        &[0.5, -0.5, -0.25, 0.5],
        &[0.25, -0.375, -0.125, 0.3125],
    );

    let mut t = BMLTransformer::new();
    let x = [t.var(0), t.var(1)];
    let weights = [t.constant(0.5), t.constant(-0.5)];
    let roots = t.rms_norm(&x, &weights, 0.5).unwrap();
    persisted_outputs(t, &roots, &[0.25, -0.25], &[1.0 / 6.0, 1.0 / 6.0]);

    let mut t = BMLTransformer::new();
    let x = [t.var(0), t.var(1), t.var(2), t.var(3)];
    let cos = [t.constant(0.0), t.constant(-1.0)];
    let sin = [t.constant(1.0), t.constant(0.0)];
    let roots = t.rope(&x, &cos, &sin).unwrap();
    persisted_outputs(
        t,
        &roots,
        &[0.25, -0.5, 0.5, -0.25],
        &[0.5, 0.25, -0.5, 0.25],
    );

    let mut t = BMLTransformer::new();
    let x = [t.var(0), t.var(1)];
    let roots = t.softmax(&x).unwrap();
    let z = (-0.25_f64).exp() + 0.25_f64.exp();
    persisted_outputs(
        t,
        &roots,
        &[-0.25, 0.25],
        &[(-0.25_f64).exp() / z, 0.25_f64.exp() / z],
    );

    let mut t = BMLTransformer::new();
    let x = [t.var(0), t.var(1), t.var(2), t.var(3)];
    let roots = t.swiglu(&x[..2], &x[2..]).unwrap();
    persisted_outputs(
        t,
        &roots,
        &[-0.25, 0.25, 0.5, -0.25],
        &[
            -0.125 / (1.0 + 0.25_f64.exp()),
            -0.0625 / (1.0 + (-0.25_f64).exp()),
        ],
    );
}
