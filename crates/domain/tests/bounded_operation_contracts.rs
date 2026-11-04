use bml_domain::{
    transformer::{BoundedBuildError, BoundedOpConfig},
    BMLTransformer, NodeId, NodeKind,
};

// Independent traversal deliberately checks even unused BML nodes: a finite
// final root must not hide exponential underflow/overflow or a bad logarithm.
fn evaluate(
    builder: &BMLTransformer,
    roots: &[NodeId],
    inputs: &[f64],
    external: &[f64],
) -> Result<Vec<f64>, String> {
    let mut constants = builder.soa().const_pool.clone();
    constants.extend_from_slice(external);
    let mut values: Vec<f64> = Vec::with_capacity(builder.soa().len());
    for index in 0..builder.soa().len() {
        let node = builder.soa().get(index as NodeId);
        let value = match node.kind {
            NodeKind::One => 1.0,
            NodeKind::Var(id) => inputs[id as usize],
            NodeKind::Const(id) => constants[id as usize],
            NodeKind::Zero => return Err("unexpected Zero outside basis".into()),
            NodeKind::Bml => {
                let left = values[node.left.unwrap() as usize];
                let right = values[node.right.unwrap() as usize];
                let exponential = left.exp2();
                if !left.is_finite()
                    || !right.is_finite()
                    || right <= 0.0
                    || !exponential.is_finite()
                    || exponential == 0.0
                {
                    return Err(format!("node {index}: invalid BML({left},{right})"));
                }
                exponential - right.log2()
            }
        };
        if !value.is_finite() {
            return Err(format!("node {index}: nonfinite result"));
        }
        values.push(value);
    }
    Ok(roots.iter().map(|&root| values[root as usize]).collect())
}

fn build(config: &BoundedOpConfig) -> (BMLTransformer, Vec<NodeId>) {
    let mut builder = BMLTransformer::new();
    let inputs: Vec<_> = (0..config.inputs())
        .map(|i| builder.var(i as u32))
        .collect();
    let fixed = config.fixed_literals();
    let external: Vec<_> = (0..config.external())
        .map(|i| builder.external_constant((fixed.len() + i) as u32))
        .collect();
    let roots = builder
        .operation_bounded(&inputs, &external, config)
        .unwrap();
    assert_eq!(builder.soa().const_pool, fixed);
    assert_eq!(roots.len(), config.outputs());
    (builder, roots)
}

fn close(
    config: &BoundedOpConfig,
    input: &[f64],
    external: &[f64],
    expected: &[f64],
    tolerance: f64,
) {
    config.validate_values(input, external).unwrap();
    let (builder, roots) = build(config);
    let actual = evaluate(&builder, &roots, input, external).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < tolerance,
            "element {index}: {actual} != {expected}; input={input:?}, external={external:?}"
        );
    }
}

#[test]
fn bounded_products_sums_and_dot_steps_match_independent_signed_arithmetic() {
    for (left, right) in [
        (128.0, 2.0),
        (16.0, 16.0),
        (1.0, 16.0),
        (256.0, 1.0),
        (240.0, 2.0),
        (0.01, 0.5),
    ] {
        let config = BoundedOpConfig::product(left, right).unwrap();
        for x in [-left, 0.0, left, -left / 7.0] {
            for y in [-right, 0.0, right, right / 3.0] {
                close(&config, &[x, y], &[], &[x * y], 2e-8);
            }
        }
    }
    let sum = BoundedOpConfig::add(32.0, 32.0).unwrap();
    for x in [-32.0, -12.0, 0.0, 12.0, 32.0] {
        for y in [-32.0, 0.0, 32.0] {
            close(&sum, &[x, y], &[], &[x + y], 2e-10);
        }
    }
    let step = BoundedOpConfig::dot_step(128.0, 2.0).unwrap();
    for accumulator in [-128.0, 0.0, 127.0] {
        for (x, y) in [(-128.0, -2.0), (128.0, -2.0), (0.0, 0.0), (1.5, -0.75)] {
            close(
                &step,
                &[accumulator, x, y],
                &[],
                &[accumulator + x * y],
                2e-8,
            );
        }
    }
    // Its finite-real range is checked dynamically, not promised for arbitrary
    // negative accumulated values accepted by the finite/<256 declaration.
    let (builder, roots) = build(&step);
    assert!(step.validate_values(&[-1e6, 0.0, 0.0], &[]).is_ok());
    assert!(evaluate(&builder, &roots, &[-1e6, 0.0, 0.0], &[]).is_err());
}

#[test]
fn bounded_rope_handles_real_bos_failure_and_rotation_boundaries() {
    let config = BoundedOpConfig::rope(16.0).unwrap();
    for (x, y) in [
        (2.3903015418765046, 1.1409290220926804),
        (4.0, 1.0),
        (-16.0, -16.0),
        (-16.0, 16.0),
        (16.0, 16.0),
        (0.0, 0.0),
    ] {
        for angle in [0.0_f64, 0.5, 1.0, 2.0, 3.0] {
            let (sine, cosine) = angle.sin_cos();
            close(
                &config,
                &[x, y],
                &[cosine, sine],
                &[x * cosine - y * sine, x * sine + y * cosine],
                2e-8,
            );
        }
    }
    let mut legacy = BMLTransformer::new();
    let x = legacy.var(0);
    let y = legacy.var(1);
    let cosine = legacy.var(2);
    let sine = legacy.var(3);
    let roots = legacy.rope(&[x, y], &[cosine], &[sine]).unwrap();
    assert!(evaluate(
        &legacy,
        &roots,
        &[2.3903015418765046, 1.1409290220926804, 1.0, 0.0],
        &[]
    )
    .is_err());
}

#[test]
fn bounded_swiglu_covers_gate_signs_extremes_and_zero() {
    let config = BoundedOpConfig::swiglu(16.0, 16.0).unwrap();
    for gate in [-16.0_f64, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 16.0] {
        for up in [-16.0, -4.0, 0.0, 4.0, 16.0] {
            close(
                &config,
                &[gate, up],
                &[],
                &[gate / (1.0 + (-gate).exp()) * up],
                3e-8,
            );
        }
    }
}

#[test]
fn bounded_swiglu_extended_range_checks_every_node_and_matches_silu() {
    for (gate_bound, up_bound) in [
        (16.000_000_000_000_004, 16.0),
        (17.2, 16.0),
        (18.0, 16.0),
        (18.0, 26.0),
    ] {
        let config = BoundedOpConfig::swiglu(gate_bound, up_bound).unwrap();
        let (builder, roots) = build(&config);
        for step in 0..=3600 {
            let gate = gate_bound * (f64::from(step) / 1800.0 - 1.0);
            for up in [-up_bound, -1.0, 0.0, 1.0, up_bound] {
                config.validate_values(&[gate, up], &[]).unwrap();
                let result = evaluate(&builder, &roots, &[gate, up], &[])
                    .unwrap_or_else(|e| panic!("gate={gate} up={up}: {e}"))[0];
                let expected = gate / (1.0 + (-gate).exp()) * up;
                assert!(
                    (result - expected).abs() < 3e-8,
                    "gate={gate} up={up}: {result} != {expected}"
                );
            }
        }
        assert!(config
            .validate_values(&[gate_bound + 0.01, 1.0], &[])
            .is_err());
    }
    assert!(BoundedOpConfig::swiglu(18.0, 27.0).is_err());
}

#[test]
fn bounded_softmax_uses_selected_maximum_for_negative_extremes_and_permutations() {
    for scores in [
        vec![-16.0_f64, -16.0],
        vec![-8.0, -8.0],
        vec![0.0],
        vec![-16.0, 16.0, 0.0, -8.0],
        vec![16.0, 0.0, -8.0, -16.0],
        vec![-8.0, -16.0, 16.0, 0.0],
        vec![0.0; 32],
        (0..128).map(|i| [-16.0, -8.0, 16.0, 1.5][i % 4]).collect(),
    ] {
        let config = BoundedOpConfig::softmax(scores.len(), 16.0).unwrap();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials: Vec<_> = scores.iter().map(|value| (value - maximum).exp()).collect();
        let denominator: f64 = exponentials.iter().sum();
        let expected: Vec<_> = exponentials
            .iter()
            .map(|value| value / denominator)
            .collect();
        let mut input = scores;
        input.push(maximum);
        close(&config, &input, &[], &expected, 2e-9);
    }
}

#[test]
fn bounded_operation_parameters_and_contexts_are_enforced_without_builder_mutation() {
    for bad in [0.0, -1.0, 1.0 / 512.0, 257.0, f64::NAN, f64::INFINITY] {
        assert!(BoundedOpConfig::product(bad, 1.0).is_err());
        assert!(BoundedOpConfig::product(1.0, bad).is_err());
        assert!(BoundedOpConfig::add(bad, 1.0).is_err());
        assert!(BoundedOpConfig::rope(bad).is_err());
        assert!(BoundedOpConfig::swiglu(bad, 1.0).is_err());
        assert!(BoundedOpConfig::softmax(2, bad).is_err());
    }
    assert!(BoundedOpConfig::product(32.0, 16.0).is_err());
    assert!(BoundedOpConfig::add(256.0, 256.0).is_err());
    assert!(BoundedOpConfig::rope(241.0).is_err());
    assert!(BoundedOpConfig::swiglu(18.0001, 1.0).is_err());
    assert!(BoundedOpConfig::softmax(2, 17.0).is_err());
    for width in [0, 65537, usize::MAX] {
        assert!(BoundedOpConfig::softmax(width, 16.0).is_err());
    }
    let config = BoundedOpConfig::rope(16.0).unwrap();
    assert_eq!(
        config.validate_values(&[0.0], &[1.0, 0.0]),
        Err(BoundedBuildError::Shape)
    );
    assert_eq!(
        config.validate_values(&[0.0, 0.0], &[1.0]),
        Err(BoundedBuildError::Shape)
    );
    for bad in [16.0 + 1e-9, -16.0 - 1e-9, f64::NAN, f64::INFINITY] {
        assert_eq!(
            config.validate_values(&[0.0, bad], &[1.0, 0.0]),
            Err(BoundedBuildError::Input(1))
        );
    }
    assert_eq!(
        config.validate_values(&[0.0, 0.0], &[0.0, 1.1]),
        Err(BoundedBuildError::Weight(1))
    );
    let softmax = BoundedOpConfig::softmax(2, 16.0).unwrap();
    assert_eq!(
        softmax.validate_values(&[-8.0, -16.0, 16.0], &[]),
        Err(BoundedBuildError::Input(2))
    );
    assert!(softmax.validate_values(&[-8.0, -16.0, -8.0], &[]).is_ok());
    let dot = BoundedOpConfig::dot_step(128.0, 2.0).unwrap();
    assert_eq!(
        dot.validate_values(&[256.0, 1.0, 1.0], &[]),
        Err(BoundedBuildError::Input(0))
    );
    assert_eq!(
        dot.validate_values(&[0.0, 129.0, 1.0], &[]),
        Err(BoundedBuildError::Input(1))
    );
    assert_eq!(
        dot.validate_values(&[0.0, 1.0, 2.1], &[]),
        Err(BoundedBuildError::Input(2))
    );
    let mut builder = BMLTransformer::new();
    let node = builder.var(0);
    let before = builder.soa().clone();
    assert_eq!(
        builder.operation_bounded(&[node], &[node, node], &config),
        Err(BoundedBuildError::Shape)
    );
    assert_eq!(
        builder.operation_bounded(&[node, u32::MAX], &[node, node], &config),
        Err(BoundedBuildError::InvalidNode(u32::MAX))
    );
    assert_eq!(
        builder.operation_bounded(&[node, node], &[u32::MAX, node], &config),
        Err(BoundedBuildError::InvalidNode(u32::MAX))
    );
    assert_eq!(builder.soa().len(), before.len());
    assert_eq!(builder.soa().const_pool, before.const_pool);
}

#[test]
fn bounded_softmax_literals_and_graph_growth_are_independent_of_observed_values() {
    let counts: Vec<_> = [1, 2, 3, 32]
        .iter()
        .map(|&width| {
            let config = BoundedOpConfig::softmax(width, 16.0).unwrap();
            assert_eq!(
                config.fixed_literals(),
                BoundedOpConfig::softmax(1, 16.0).unwrap().fixed_literals()
            );
            let (builder, _) = build(&config);
            builder.soa().len()
        })
        .collect();
    assert_eq!(counts[2] - counts[1], counts[1] - counts[0]);
    assert_eq!(counts[3], counts[0] + 31 * (counts[1] - counts[0]));
}
