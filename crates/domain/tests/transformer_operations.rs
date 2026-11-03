use bml_domain::{transformer::OperationBuildError, BMLTransformer, NodeId, NodeKind};

// Independent topological evaluation avoids exponential recursive revisits of
// shared sub-DAGs. Reject invalid real intermediates, not just invalid outputs.
fn evaluate(t: &BMLTransformer, roots: &[NodeId], inputs: &[f64]) -> Result<Vec<f64>, String> {
    let mut values = Vec::<f64>::new();
    for id in 0..t.soa().len() as NodeId {
        let node = t.soa().get(id);
        let value = match node.kind {
            NodeKind::One => 1.0,
            NodeKind::Const(i) => t.soa().const_pool[i as usize],
            NodeKind::Var(i) => inputs[i as usize],
            NodeKind::Bml => {
                let left = values[node.left.unwrap() as usize];
                let right = values[node.right.unwrap() as usize];
                let power = left.exp2();
                if right <= 0.0 || power == 0.0 || !power.is_finite() {
                    return Err(format!(
                        "node {id}: BML({left}, {right}) outside finite real range"
                    ));
                }
                power - right.log2()
            }
            NodeKind::Zero => panic!("builders should use the declared One/Var/Const/BML basis"),
        };
        if !value.is_finite() {
            return Err(format!("node {id}: nonfinite value"));
        }
        values.push(value);
    }
    Ok(roots.iter().map(|&id| values[id as usize]).collect())
}
fn variables(t: &mut BMLTransformer, count: u32) -> Vec<NodeId> {
    (0..count).map(|id| t.var(id)).collect()
}
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 2e-7, "actual {a}, expected {e}");
    }
}

#[test]
fn rectangular_signed_matmul_uses_all_rows_columns_and_inner_terms() {
    let mut t = BMLTransformer::new();
    let left = variables(&mut t, 6);
    let weights = [0.5, -0.25, 0.0, 0.5, -0.5, 0.25];
    let right: Vec<_> = weights.iter().map(|&x| t.constant(x)).collect();
    let roots = t.matmul(&left, &right, 2, 3, 2).unwrap();
    close(
        &evaluate(&t, &roots, &[0.5, -0.5, 0.25, -0.25, 0.5, 0.0]).unwrap(),
        &[0.125, -0.3125, -0.125, 0.3125],
    );
    let empty = t.matmul(&[], &[], 2, 0, 3).unwrap();
    close(
        &evaluate(&t, &empty, &[0.5, -0.5, 0.25, -0.25, 0.5, 0.0]).unwrap(),
        &[0.0; 6],
    );
}

#[test]
fn rms_norm_matches_closed_form_with_fractional_variance_and_signed_weights() {
    let mut t = BMLTransformer::new();
    let input = variables(&mut t, 2);
    let weights = [t.constant(0.5), t.constant(-0.5)];
    let roots = t.rms_norm(&input, &weights, 0.5).unwrap();
    // mean(x²)+epsilon = 1/16+1/2 = 9/16, sqrt = 3/4.
    close(
        &evaluate(&t, &roots, &[0.25, -0.25]).unwrap(),
        &[1.0 / 6.0, 1.0 / 6.0],
    );
    close(&evaluate(&t, &roots, &[0.0, 0.0]).unwrap(), &[0.0, 0.0]);
}

#[test]
fn rope_rotates_every_adjacent_pair_with_distinct_signed_angles() {
    let mut t = BMLTransformer::new();
    let input = variables(&mut t, 6);
    let cos = [t.constant(0.0), t.constant(-1.0), t.constant(0.6)];
    let sin = [t.constant(1.0), t.constant(0.0), t.constant(-0.8)];
    let roots = t.rope(&input, &cos, &sin).unwrap();
    close(
        &evaluate(&t, &roots, &[0.25, -0.5, 0.5, -0.25, 0.5, 0.25]).unwrap(),
        &[0.5, 0.25, -0.5, 0.25, 0.5, -0.25],
    );
}

#[test]
fn softmax_uses_natural_exponentials_and_swiglu_uses_both_branches() {
    let mut t = BMLTransformer::new();
    let input = variables(&mut t, 3);
    let roots = t.softmax(&input).unwrap();
    let values = [-0.25_f64, 0.0, 0.25];
    let denominator: f64 = values.iter().map(|x| x.exp()).sum();
    let expected: Vec<_> = values.iter().map(|x| x.exp() / denominator).collect();
    close(&evaluate(&t, &roots, &values).unwrap(), &expected);
    close(&evaluate(&t, &roots, &[0.0; 3]).unwrap(), &[1.0 / 3.0; 3]);
    assert!(evaluate(&t, &roots, &[1024.0, 0.0, 0.0]).is_err());

    let mut t = BMLTransformer::new();
    let input = variables(&mut t, 6);
    let roots = t.swiglu(&input[..3], &input[3..]).unwrap();
    let values = [-0.25_f64, 0.0, 0.25, 0.5, -0.5, -0.25];
    let expected: Vec<_> = (0..3)
        .map(|i| values[i] / (1.0 + (-values[i]).exp()) * values[i + 3])
        .collect();
    close(&evaluate(&t, &roots, &values).unwrap(), &expected);
}

#[test]
fn malformed_operations_fail_before_mutating_the_graph() {
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let before = t.soa().len();
    assert_eq!(
        t.matmul(&[x], &[x], usize::MAX, 2, 1),
        Err(OperationBuildError::SizeOverflow)
    );
    assert_eq!(
        t.matmul(&[], &[], usize::MAX, 0, 1),
        Err(OperationBuildError::Allocation)
    );
    assert_eq!(
        t.matmul(&[x], &[x], 2, 1, 1),
        Err(OperationBuildError::Shape)
    );
    assert_eq!(
        t.matmul(&[u32::MAX], &[x], 1, 1, 1),
        Err(OperationBuildError::InvalidNode(u32::MAX))
    );
    assert_eq!(t.rms_norm(&[], &[], 1.0), Err(OperationBuildError::Shape));
    assert_eq!(t.rms_norm(&[x], &[], 1.0), Err(OperationBuildError::Shape));
    for eps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            t.rms_norm(&[x], &[x], eps),
            Err(OperationBuildError::InvalidEpsilon)
        );
    }
    assert_eq!(
        t.rms_norm(&[x], &[u32::MAX], 1.0),
        Err(OperationBuildError::InvalidNode(u32::MAX))
    );
    assert_eq!(t.rope(&[x], &[], &[]), Err(OperationBuildError::Shape));
    assert_eq!(t.rope(&[x, x], &[x], &[]), Err(OperationBuildError::Shape));
    assert_eq!(
        t.rope(&[x, x], &[u32::MAX], &[x]),
        Err(OperationBuildError::InvalidNode(u32::MAX))
    );
    assert_eq!(t.softmax(&[]), Err(OperationBuildError::Shape));
    assert_eq!(
        t.softmax(&[u32::MAX]),
        Err(OperationBuildError::InvalidNode(u32::MAX))
    );
    assert_eq!(t.swiglu(&[x], &[]), Err(OperationBuildError::Shape));
    assert_eq!(
        t.swiglu(&[x], &[u32::MAX]),
        Err(OperationBuildError::InvalidNode(u32::MAX))
    );
    assert!(t.matmul(&[], &[], usize::MAX, 0, 0).unwrap().is_empty());
    assert_eq!(t.soa().len(), before);
    assert!(t.soa().const_pool.is_empty());
}
