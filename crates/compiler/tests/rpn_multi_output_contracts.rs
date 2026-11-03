//! Shared multi-output lowering must preserve stored scalar DAG evaluation.
use bml_compiler::{fragment_program, linearize, rpn::linearize_outputs, BmlGraph, RpnOp};
use bml_domain::{BMLTransformer, EvalContext, NodeId, NodeKind, NodeSoA};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
};

fn execute_outputs(
    soa: &NodeSoA,
    roots: &[NodeId],
    inputs: &[f64],
    constants: &[f64],
    output_constants_start: u32,
) -> Vec<f64> {
    let mut program = linearize_outputs(soa, roots, 0, output_constants_start).unwrap();
    assert_eq!(program.const_pool, soa.const_pool);
    assert!(program.ops.iter().all(|op| matches!(
        op,
        RpnOp::One
            | RpnOp::Zero
            | RpnOp::Var(_)
            | RpnOp::Const(_)
            | RpnOp::Bml
            | RpnOp::Pick { .. }
            | RpnOp::Drop
            | RpnOp::StoreResult { slot: 0 }
    )));
    program.const_pool.clear();
    // Split the stack-producing code, stores and cleanup across fragments.
    let threshold = 2 * std::mem::size_of::<RpnOp>();
    let graph = fragment_program(&program, threshold);
    assert!(graph.fragments.len() > 1);
    let loaded = BmlGraph::deserialize(&graph.serialize(), threshold).unwrap();
    let validated = ValidatedGraph::new_with_external_constants(
        loaded,
        inputs.len(),
        constants.len(),
        1,
        roots.len(),
        GraphLimits::default(),
    )
    .unwrap();
    let mut buffer = ResultBuffer::from_vec(1, roots.len(), vec![f64::NAN; roots.len()]).unwrap();
    assert_eq!(
        validated
            .execute_with_constants(inputs, constants, &mut buffer)
            .unwrap()
            .to_bits(),
        1.0_f64.to_bits()
    );
    let actual = buffer.into_vec();
    for (&root, &actual) in roots.iter().zip(&actual) {
        let expected = linearize(soa, root).evaluate_with_ctx(&EvalContext::new(inputs, constants));
        assert_eq!(actual.to_bits(), expected.to_bits(), "root={root}");
    }
    actual
}

fn reachable_bml_count(soa: &NodeSoA, roots: &[NodeId]) -> usize {
    let mut visited = std::collections::BTreeSet::new();
    let mut pending = roots.to_vec();
    let mut count = 0;
    while let Some(node) = pending.pop() {
        if visited.insert(node) && soa.kinds[node as usize] == NodeKind::Bml {
            count += 1;
            pending.push(soa.lefts[node as usize]);
            pending.push(soa.rights[node as usize]);
        }
    }
    count
}

#[test]
fn persisted_shared_outputs_preserve_external_weights_and_repeated_roots() {
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let weight = t.external_constant(0);
    let shared = t.bml(x, weight);
    let one = t.one();
    let exp = t.bml(shared, one);
    let complement = t.bml(one, shared);
    let roots = [shared, exp, shared, complement, one];
    // This unrelated node must not be emitted or require an external input.
    let missing_input = t.var(100);
    t.exp2(missing_input);
    let soa = t.into_soa();
    let program = linearize_outputs(&soa, &roots, 0, 1).unwrap();
    let count = program.ops.iter().filter(|op| **op == RpnOp::Bml).count();
    assert_eq!(count, reachable_bml_count(&soa, &roots));
    assert_eq!(count, 3);
    assert_eq!(
        program
            .ops
            .iter()
            .filter(|op| **op == RpnOp::Var(0))
            .count(),
        1
    );
    assert_eq!(
        program
            .ops
            .iter()
            .filter(|op| **op == RpnOp::Const(0))
            .count(),
        1
    );
    for (input, weight) in [(1.0, 2.0), (3.0, 4.0), (0.5, 0.5)] {
        let constants = [weight, 0.0, 1.0, 2.0, 3.0, 4.0];
        let output = execute_outputs(&soa, &roots, &[input], &constants, 1);
        assert_eq!(output[0].to_bits(), output[2].to_bits());
        if input == 1.0 {
            assert_eq!(output, [1.0, 2.0, 1.0, 2.0, 1.0]);
        }
    }
}

#[test]
fn vector_reductions_share_nodes_and_keep_scalar_rounding() {
    for width in [2, 4, 8] {
        let mut t = BMLTransformer::new();
        let x: Vec<_> = (0..width).map(|i| t.var(i)).collect();
        let roots = t.softmax(&x).unwrap();
        let soa = t.into_soa();
        let output_start = soa.const_pool.len() as u32;
        let mut constants = soa.const_pool.clone();
        constants.extend((0..width).map(f64::from));
        let inputs: Vec<_> = (0..width).map(|i| f64::from(i) / 64.0 - 0.125).collect();
        let outputs = execute_outputs(&soa, &roots, &inputs, &constants, output_start);
        assert!(outputs.iter().all(|value| value.is_finite()));
        let denominator: f64 = inputs.iter().map(|value| value.exp()).sum();
        for (&input, &output) in inputs.iter().zip(&outputs) {
            assert!((output - input.exp() / denominator).abs() < 1e-10);
        }
        let shared = linearize_outputs(&soa, &roots, 0, output_start).unwrap();
        let shared_bml = shared.ops.iter().filter(|op| **op == RpnOp::Bml).count();
        let scalar_bml: usize = roots
            .iter()
            .map(|&root| {
                linearize(&soa, root)
                    .ops
                    .iter()
                    .filter(|op| **op == RpnOp::Bml)
                    .count()
            })
            .sum();
        assert_eq!(shared_bml, reachable_bml_count(&soa, &roots));
        assert!(shared_bml < scalar_bml);
    }
}

#[test]
fn shared_outputs_retain_legacy_ieee_infinite_intermediates() {
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let negative = t.neg(x);
    let one = t.one();
    let zero = t.log2(one);
    let negative_infinity = t.log2(zero);
    let roots = [negative, negative_infinity, negative, zero];
    let soa = t.into_soa();
    for value in [-100.0, -2.0, 0.0, 2.0, 100.0] {
        let actual = execute_outputs(&soa, &roots, &[value], &[0.0, 1.0, 2.0, 3.0], 0);
        assert_eq!(actual[0], -value);
        assert_eq!(actual[1], f64::NEG_INFINITY);
        assert_eq!(actual[3], 0.0);
    }
}

#[test]
fn multi_output_preflight_rejects_invalid_roots_children_and_index_overflow() {
    let mut t = BMLTransformer::new();
    let one = t.one();
    let root = t.exp2(one);
    let soa = t.into_soa();
    assert!(linearize_outputs(&soa, &[], 0, 0).is_err());
    assert!(linearize_outputs(&soa, &[u32::MAX], 0, 0).is_err());
    assert!(linearize_outputs(&soa, &[root, root], 0, u32::MAX).is_err());
    let edge = linearize_outputs(&soa, &[root], 3, u32::MAX).unwrap();
    assert!(edge.ops.contains(&RpnOp::Const(u32::MAX)));
    assert!(edge.ops.contains(&RpnOp::StoreResult { slot: 3 }));
    for child in [root, u32::MAX] {
        let mut bad = soa.clone();
        bad.lefts[root as usize] = child;
        assert!(linearize_outputs(&bad, &[root], 0, 0).is_err());
        let mut bad = soa.clone();
        bad.rights[root as usize] = child;
        assert!(linearize_outputs(&bad, &[root], 0, 0).is_err());
    }
    let mut bad = soa.clone();
    bad.lefts.pop();
    assert!(linearize_outputs(&bad, &[root], 0, 0).is_err());
    let mut bad = soa;
    bad.rights.pop();
    assert!(linearize_outputs(&bad, &[root], 0, 0).is_err());
}
