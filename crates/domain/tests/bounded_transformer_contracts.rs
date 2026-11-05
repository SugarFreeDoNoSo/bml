use bml_domain::{
    transformer::{BoundedBuildError, BoundedRmsConfig},
    BMLTransformer, NodeId, NodeKind,
};

// Independent topological evaluator: validate every node, including intermediate
// exponentials that could underflow even if the final subtraction is finite.
fn evaluate(
    builder: &BMLTransformer,
    roots: &[NodeId],
    input: &[f64],
    external_weights: &[f64],
) -> Result<Vec<f64>, String> {
    let mut constants = builder.soa().const_pool.clone();
    constants.extend_from_slice(external_weights);
    let mut values = Vec::<f64>::with_capacity(builder.soa().len());
    for id in 0..builder.soa().len() as NodeId {
        let node = builder.soa().get(id);
        let value = match node.kind {
            NodeKind::One => 1.0,
            NodeKind::Var(index) => input[index as usize],
            NodeKind::Const(index) => constants[index as usize],
            NodeKind::Zero => return Err("unexpected Zero outside the declared basis".into()),
            NodeKind::Bml => {
                let left = values[node.left.unwrap() as usize];
                let right = values[node.right.unwrap() as usize];
                let power = left.exp2();
                if right <= 0.0 || !power.is_finite() || power == 0.0 {
                    return Err(format!(
                        "node {id}: BML({left}, {right}) outside finite reals"
                    ));
                }
                power - right.log2()
            }
        };
        if !value.is_finite() {
            return Err(format!("node {id}: nonfinite value"));
        }
        values.push(value);
    }
    Ok(roots.iter().map(|&root| values[root as usize]).collect())
}

fn build(config: &BoundedRmsConfig) -> (BMLTransformer, Vec<NodeId>) {
    let mut builder = BMLTransformer::new();
    let input: Vec<_> = (0..config.width()).map(|i| builder.var(i as u32)).collect();
    let fixed = config.fixed_literals();
    let weights: Vec<_> = (0..config.width())
        .map(|i| builder.external_constant((fixed.len() + i) as u32))
        .collect();
    let roots = builder.rms_norm_bounded(&input, &weights, config).unwrap();
    assert_eq!(builder.soa().const_pool, fixed);
    (builder, roots)
}

fn check_case(config: &BoundedRmsConfig, input: &[f64], weights: &[f64], tolerance: f64) {
    config.validate_values(input, weights).unwrap();
    let (builder, roots) = build(config);
    let actual = evaluate(&builder, &roots, input, weights).unwrap();
    // Native reference is independent of all BML arithmetic identities.
    let sum_of_squares: f64 = input.iter().map(|x| x * x).sum();
    let denominator = (sum_of_squares / input.len() as f64 + config.epsilon()).sqrt();
    for (index, ((result, x), weight)) in actual.iter().zip(input).zip(weights).enumerate() {
        let expected = x * weight / denominator;
        assert!(
            (result - expected).abs() < tolerance,
            "width {}, element {index}: actual {result}, expected {expected}",
            input.len()
        );
    }
}

#[test]
fn rms_bound_getters_reconstruct_the_same_admitted_domain() {
    let config = BoundedRmsConfig::new(3, 0.5, 8.0, 0.5).unwrap();
    let input = [8.0, -4.0, 0.0];
    let weights = [0.5, -0.25, 0.125];
    config.validate_values(&input, &weights).unwrap();
    // A consumer can persist the public declarations and reconstruct a
    // configuration admitting the same signed, nonuniform data.
    let restored = BoundedRmsConfig::new(
        config.width(),
        config.epsilon(),
        config.input_bound(),
        config.weight_bound(),
    )
    .unwrap();
    restored.validate_values(&input, &weights).unwrap();
    assert_eq!(restored.fixed_literals(), config.fixed_literals());
    check_case(&config, &input, &weights, 2e-6);
    check_case(&restored, &input, &weights, 2e-6);

    let bound = config.input_bound();
    config
        .validate_values(&[bound, -bound, 0.0], &weights)
        .unwrap();
    assert_eq!(
        config.validate_values(&[bound.next_up(), 0.0, 0.0], &weights),
        Err(BoundedBuildError::Input(0))
    );
    assert_eq!(
        config.validate_values(&[-bound.next_up(), 0.0, 0.0], &weights),
        Err(BoundedBuildError::Input(0))
    );
}

#[test]
fn bounded_rms_executes_weight_four_that_overflows_the_legacy_expansion() {
    let config = BoundedRmsConfig::new(1, 1e-5, 16.0, 16.0).unwrap();
    check_case(&config, &[1.0], &[4.0], 1e-8);
    check_case(&config, &[0.0], &[4.0], 1e-7);
    check_case(&config, &[0.01], &[-4.0], 1e-8);

    let mut old = BMLTransformer::new();
    let x = old.var(0);
    let w = old.var(1);
    let roots = old.rms_norm(&[x], &[w], 1e-5).unwrap();
    assert!(evaluate(&old, &roots, &[1.0, 4.0], &[]).is_err());
}

#[test]
fn bounded_rms_matches_independent_oracle_across_signs_zeros_widths_and_epsilon() {
    for width in [1, 2, 4, 32, 288] {
        for epsilon in [1e-5, 0.001, 0.25] {
            let config = BoundedRmsConfig::new(width, epsilon, 16.0, 16.0).unwrap();
            let input: Vec<_> = (0..width)
                .map(|i| [-16.0, 16.0, 0.0, 0.1, -0.03, 4.0, -7.0][i % 7])
                .collect();
            let weights: Vec<_> = (0..width)
                .map(|i| [16.0, -16.0, 0.0, 4.0, -0.25][i % 5])
                .collect();
            check_case(&config, &input, &weights, 2e-6);
            check_case(&config, &vec![0.0; width], &weights, 2e-6);
        }
    }
}

#[test]
fn bounded_rms_uses_correlation_for_sparse_vectors_and_accepts_safe_extreme_bounds() {
    let config = BoundedRmsConfig::new(288, 1e-5, 16.0, 16.0).unwrap();
    let mut input = vec![0.0; 288];
    input[167] = -16.0;
    let mut weights = vec![16.0; 288];
    weights[167] = -16.0;
    check_case(&config, &input, &weights, 2e-6);

    let large = BoundedRmsConfig::new(1, 256.0 * 256.0, 256.0, 240.0).unwrap();
    check_case(&large, &[-256.0], &[240.0], 2e-6);
    let small = BoundedRmsConfig::new(4, 1e-5, 1.0 / 256.0, 1.0 / 256.0).unwrap();
    check_case(
        &small,
        &[0.0, 1.0 / 256.0, -1.0 / 256.0, 0.001],
        &[1.0 / 256.0; 4],
        2e-6,
    );
}

#[test]
fn bounded_rms_rejects_parameters_and_operand_declarations_before_execution() {
    for width in [0, 255 * 255 + 1, usize::MAX] {
        assert_eq!(
            BoundedRmsConfig::new(width, 1e-5, 16.0, 16.0),
            Err(BoundedBuildError::Parameter("width"))
        );
    }
    for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            BoundedRmsConfig::new(1, epsilon, 16.0, 16.0),
            Err(BoundedBuildError::Parameter("epsilon"))
        );
    }
    for bound in [0.0, -1.0, 1.0 / 512.0, 257.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            BoundedRmsConfig::new(1, 1e-5, bound, 16.0),
            Err(BoundedBuildError::Parameter("input_bound"))
        );
        assert_eq!(
            BoundedRmsConfig::new(1, 1e-5, 16.0, bound),
            Err(BoundedBuildError::Parameter("weight_bound"))
        );
    }
    assert_eq!(
        BoundedRmsConfig::new(288, 1e-5, 16.0, 32.0),
        Err(BoundedBuildError::Parameter("normalized_weight_product"))
    );
    for epsilon in [f64::MIN_POSITIVE, 257.0] {
        assert_eq!(
            BoundedRmsConfig::new(1, epsilon, 16.0, 16.0),
            Err(BoundedBuildError::Parameter("scaled_epsilon"))
        );
    }
    let config = BoundedRmsConfig::new(2, 1e-5, 16.0, 16.0).unwrap();
    assert_eq!(
        config.validate_values(&[1.0], &[1.0, 1.0]),
        Err(BoundedBuildError::Shape)
    );
    assert_eq!(
        config.validate_values(&[1.0, 1.0], &[1.0]),
        Err(BoundedBuildError::Shape)
    );
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        16.0 + 1e-9,
        -16.0 - 1e-9,
    ] {
        assert_eq!(
            config.validate_values(&[0.0, value], &[0.0; 2]),
            Err(BoundedBuildError::Input(1))
        );
        assert_eq!(
            config.validate_values(&[0.0; 2], &[0.0, value]),
            Err(BoundedBuildError::Weight(1))
        );
    }
    assert!(config
        .validate_values(&[-16.0, 16.0], &[16.0, -16.0])
        .is_ok());
}

#[test]
fn bounded_rms_validates_structure_without_mutation_and_has_affine_node_growth() {
    let config = BoundedRmsConfig::new(1, 1e-5, 16.0, 16.0).unwrap();
    let mut builder = BMLTransformer::new();
    let x = builder.var(0);
    let before = builder.soa().clone();
    assert_eq!(
        builder.rms_norm_bounded(&[], &[x], &config),
        Err(BoundedBuildError::Shape)
    );
    assert_eq!(
        builder.rms_norm_bounded(&[x], &[], &config),
        Err(BoundedBuildError::Shape)
    );
    assert_eq!(
        builder.rms_norm_bounded(&[u32::MAX], &[x], &config),
        Err(BoundedBuildError::InvalidNode(u32::MAX))
    );
    assert_eq!(
        builder.rms_norm_bounded(&[x], &[u32::MAX], &config),
        Err(BoundedBuildError::InvalidNode(u32::MAX))
    );
    assert_eq!(builder.soa().len(), before.len());
    assert_eq!(builder.soa().const_pool, before.const_pool);

    let nodes: Vec<_> = [1, 2, 3, 32]
        .map(|width| {
            build(&BoundedRmsConfig::new(width, 1e-5, 16.0, 16.0).unwrap())
                .0
                .soa()
                .len()
        })
        .into();
    assert_eq!(nodes[2] - nodes[1], nodes[1] - nodes[0]);
    assert_eq!(nodes[3], nodes[0] + 31 * (nodes[1] - nodes[0]));
}
