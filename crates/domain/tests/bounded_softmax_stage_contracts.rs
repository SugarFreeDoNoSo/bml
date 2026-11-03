use bml_domain::{
    transformer::{BoundedBuildError, BoundedOpConfig},
    BMLTransformer, NodeId, NodeKind,
};

#[derive(Clone, Copy, Debug)]
enum Stage {
    Exp,
    Reduce,
    Inverse,
    Output,
}

impl Stage {
    fn arity(self) -> usize {
        match self {
            Self::Exp => 2,
            Self::Reduce | Self::Output => 3,
            Self::Inverse => 1,
        }
    }

    fn append(
        self,
        builder: &mut BMLTransformer,
        inputs: &[NodeId],
        config: &BoundedOpConfig,
    ) -> Result<NodeId, BoundedBuildError> {
        match self {
            Self::Exp => builder.softmax_bounded_exp(inputs, config),
            Self::Reduce => builder.softmax_bounded_reduce(inputs, config),
            Self::Inverse => builder.softmax_bounded_inverse(inputs, config),
            Self::Output => builder.softmax_bounded_output(inputs, config),
        }
    }

    fn build(self, config: &BoundedOpConfig) -> (BMLTransformer, NodeId) {
        let mut builder = BMLTransformer::new();
        let inputs: Vec<_> = (0..self.arity())
            .map(|index| builder.var(index as u32))
            .collect();
        let root = self.append(&mut builder, &inputs, config).unwrap();
        assert_eq!(builder.soa().const_pool, config.fixed_literals());
        (builder, root)
    }
}

// Independent topological evaluation: every BML must have a positive log
// operand and a finite, nonzero exponential, not only a finite final result.
fn checked(builder: &BMLTransformer, roots: &[NodeId], inputs: &[f64]) -> Vec<f64> {
    let mut values = Vec::<f64>::with_capacity(builder.soa().len());
    for id in 0..builder.soa().len() as NodeId {
        let node = builder.soa().get(id);
        let value = match node.kind {
            NodeKind::One => 1.0,
            NodeKind::Zero => panic!("unexpected zero outside the historical bounded basis"),
            NodeKind::Var(index) => inputs[index as usize],
            NodeKind::Const(index) => builder.soa().const_pool[index as usize],
            NodeKind::Bml => {
                let left = values[node.left.unwrap() as usize];
                let right = values[node.right.unwrap() as usize];
                let power = left.exp2();
                assert!(
                    right > 0.0 && power.is_finite() && power > 0.0,
                    "node {id}: BML({left}, {right}) is outside finite reals"
                );
                power - right.log2()
            }
        };
        assert!(value.is_finite(), "node {id} is not finite");
        values.push(value);
    }
    roots.iter().map(|&root| values[root as usize]).collect()
}

fn compare_chained(scores: &[f64], bound: f64) {
    let config = BoundedOpConfig::softmax(scores.len(), bound).unwrap();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut inputs = scores.to_vec();
    inputs.push(maximum);
    config.validate_values(&inputs, &[]).unwrap();
    let mut original = BMLTransformer::new();
    let variables: Vec<_> = (0..inputs.len())
        .map(|index| original.var(index as u32))
        .collect();
    let roots = original
        .operation_bounded(&variables, &[], &config)
        .unwrap();
    let expected_bits = checked(&original, &roots, &inputs);
    let (exp, exp_root) = Stage::Exp.build(&config);
    let (reduce, reduce_root) = Stage::Reduce.build(&config);
    let (inverse, inverse_root) = Stage::Inverse.build(&config);
    let (output, output_root) = Stage::Output.build(&config);
    // Preserve the initial positive term and the exact sequential sum order.
    let mut total = checked(&exp, &[exp_root], &[scores[0], maximum])[0];
    for &score in &scores[1..] {
        total = checked(&reduce, &[reduce_root], &[total, score, maximum])[0];
    }
    let inverse = checked(&inverse, &[inverse_root], &[total])[0];
    let native_exp: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let native_sum: f64 = native_exp.iter().sum();
    let mut probability_sum = 0.0;
    for (index, &score) in scores.iter().enumerate() {
        let actual = checked(&output, &[output_root], &[score, maximum, inverse])[0];
        assert_eq!(
            actual.to_bits(),
            expected_bits[index].to_bits(),
            "width={}, index={index}, bound={bound}",
            scores.len()
        );
        let oracle = native_exp[index] / native_sum;
        assert!(
            (actual - oracle).abs() <= 2e-9,
            "width={}, index={index}: actual={actual}, oracle={oracle}",
            scores.len()
        );
        probability_sum += actual;
    }
    assert!((probability_sum - 1.0).abs() <= 2e-9);
}

#[test]
fn bounded_softmax_stages_preserve_bits_for_negative_extremes_and_permutations() {
    for scores in [
        vec![-16.0],
        vec![16.0],
        vec![0.0],
        vec![-0.0, 0.0],
        vec![-16.0, -16.0],
        vec![-8.0, -8.0],
        vec![-16.0, 16.0, 0.0, -8.0],
        vec![16.0, 0.0, -8.0, -16.0],
        vec![-8.0, -16.0, 16.0, 0.0],
        vec![-16.0; 128],
        vec![16.0; 128],
        vec![0.0; 128],
    ] {
        compare_chained(&scores, 16.0);
    }
}

#[test]
fn bounded_softmax_stages_match_oracle_across_widths_and_valid_score_bounds() {
    for bound in [1.0 / 256.0, 0.5, 16.0] {
        for width in [1, 2, 3, 32, 128, 512] {
            let scores: Vec<_> = (0..width)
                .map(|i| [-bound, -bound / 2.0, bound, 0.0, bound / 4.0][i % 5])
                .collect();
            compare_chained(&scores, bound);
            compare_chained(&vec![-bound; width], bound);
        }
    }
}

#[test]
fn bounded_softmax_stages_keep_ten_literals_and_constant_node_counts() {
    for stage in [Stage::Exp, Stage::Reduce, Stage::Inverse, Stage::Output] {
        let counts: Vec<_> = [1, 2, 32, 128, 2048, 65536]
            .iter()
            .map(|&width| {
                let config = BoundedOpConfig::softmax(width, 16.0).unwrap();
                let (builder, _) = stage.build(&config);
                assert_eq!(builder.soa().const_pool.len(), 10);
                assert!(builder.soa().len() < 16384);
                builder.soa().len()
            })
            .collect();
        assert!(counts.iter().all(|count| *count == counts[0]));
        println!("{stage:?}: {} nodes", counts[0]);
    }
}

#[test]
fn bounded_softmax_stage_configuration_and_structure_errors_do_not_mutate() {
    let config = BoundedOpConfig::softmax(32, 16.0).unwrap();
    for stage in [Stage::Exp, Stage::Reduce, Stage::Inverse, Stage::Output] {
        let mut builder = BMLTransformer::new();
        let input = builder.var(0);
        builder.constant(123.0);
        let before = builder.soa().clone();
        let operands = vec![input; stage.arity()];
        for other in [
            BoundedOpConfig::product(1.0, 1.0).unwrap(),
            BoundedOpConfig::add(1.0, 1.0).unwrap(),
            BoundedOpConfig::dot_step(1.0, 1.0).unwrap(),
            BoundedOpConfig::rope(1.0).unwrap(),
            BoundedOpConfig::swiglu(1.0, 1.0).unwrap(),
        ] {
            assert_eq!(
                stage.append(&mut builder, &operands, &other),
                Err(BoundedBuildError::Parameter("softmax_stage"))
            );
        }
        for length in [0, stage.arity() - 1, stage.arity() + 1] {
            assert_eq!(
                stage.append(&mut builder, &vec![input; length], &config),
                Err(BoundedBuildError::Shape)
            );
        }
        for index in 0..stage.arity() {
            let mut invalid = operands.clone();
            invalid[index] = u32::MAX;
            assert_eq!(
                stage.append(&mut builder, &invalid, &config),
                Err(BoundedBuildError::InvalidNode(u32::MAX))
            );
        }
        assert_eq!(builder.soa().kinds, before.kinds);
        assert_eq!(builder.soa().lefts, before.lefts);
        assert_eq!(builder.soa().rights, before.rights);
        assert_eq!(builder.soa().const_pool, before.const_pool);
        stage.append(&mut builder, &operands, &config).unwrap();
        assert_eq!(builder.soa().const_pool[0], 123.0);
        assert_eq!(builder.soa().const_pool[1..], config.fixed_literals());
    }
}

#[test]
fn bounded_softmax_stages_retain_the_complete_vector_maximum_contract() {
    let config = BoundedOpConfig::softmax(3, 16.0).unwrap();
    assert!(config
        .validate_values(&[-16.0, 16.0, 0.0, 16.0], &[])
        .is_ok());
    for bad_maximum in [-16.0, 0.0, 15.999, f64::NAN, f64::INFINITY] {
        assert_eq!(
            config.validate_values(&[-16.0, 16.0, 0.0, bad_maximum], &[]),
            Err(BoundedBuildError::Input(3))
        );
    }
    for invalid_score in [-16.0001, 16.0001, f64::NAN, f64::INFINITY] {
        assert_eq!(
            config.validate_values(&[-16.0, invalid_score, 0.0, 16.0], &[]),
            Err(BoundedBuildError::Input(1))
        );
    }
}

#[test]
fn cached_exponential_sum_and_normalization_preserve_recomputed_stage_bits() {
    let mut sizes = Vec::new();
    for width in [1, 4, 288, 2048] {
        let config = BoundedOpConfig::softmax(width, 16.0).unwrap();
        let scores: Vec<_> = (0..width)
            .map(|i| -16.0 + (i * 7 % 29) as f64 / 2.0)
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (exp, exp_root) = Stage::Exp.build(&config);
        let (reduce, reduce_root) = Stage::Reduce.build(&config);
        let (inverse, inverse_root) = Stage::Inverse.build(&config);
        let (output, output_root) = Stage::Output.build(&config);
        let mut sum = BMLTransformer::new();
        let sum_vars = [sum.var(0), sum.var(1)];
        let sum_root = sum.softmax_bounded_sum(&sum_vars, &config).unwrap();
        let mut normalize = BMLTransformer::new();
        let norm_vars = [normalize.var(0), normalize.var(1)];
        let norm_root = normalize
            .softmax_bounded_normalize(&norm_vars, &config)
            .unwrap();
        assert_eq!(sum.soa().const_pool, config.fixed_literals());
        assert_eq!(normalize.soa().const_pool, config.fixed_literals());
        sizes.push((sum.soa().len(), normalize.soa().len()));
        let exponentials: Vec<_> = scores
            .iter()
            .map(|&score| checked(&exp, &[exp_root], &[score, maximum])[0])
            .collect();
        let mut total = exponentials[0];
        for (&score, &exponential) in scores[1..].iter().zip(&exponentials[1..]) {
            let expected = checked(&reduce, &[reduce_root], &[total, score, maximum])[0];
            total = checked(&sum, &[sum_root], &[total, exponential])[0];
            assert_eq!(total.to_bits(), expected.to_bits());
        }
        let reciprocal = checked(&inverse, &[inverse_root], &[total])[0];
        let denominator = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .sum::<f64>();
        for (&score, &exponential) in scores.iter().zip(&exponentials) {
            let expected = checked(&output, &[output_root], &[score, maximum, reciprocal])[0];
            let actual = checked(&normalize, &[norm_root], &[exponential, reciprocal])[0];
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert!((actual - (score - maximum).exp() / denominator).abs() < 2e-9);
        }
    }
    assert!(sizes.iter().all(|&size| size == sizes[0]));
}

#[test]
fn cached_softmax_stage_validation_precedes_any_builder_mutation() {
    type Append =
        fn(&mut BMLTransformer, &[NodeId], &BoundedOpConfig) -> Result<NodeId, BoundedBuildError>;
    let stages: [Append; 2] = [
        BMLTransformer::softmax_bounded_sum,
        BMLTransformer::softmax_bounded_normalize,
    ];
    let config = BoundedOpConfig::softmax(4, 16.0).unwrap();
    let other = BoundedOpConfig::product(1.0, 1.0).unwrap();
    for stage in stages {
        let mut builder = BMLTransformer::new();
        let variables = [builder.var(0), builder.var(1)];
        let nodes = builder.soa().len();
        assert!(matches!(
            stage(&mut builder, &variables[..1], &config),
            Err(BoundedBuildError::Shape)
        ));
        assert!(matches!(
            stage(&mut builder, &[variables[0], u32::MAX], &config),
            Err(BoundedBuildError::InvalidNode(_))
        ));
        assert!(matches!(
            stage(&mut builder, &variables, &other),
            Err(BoundedBuildError::Parameter("softmax_stage"))
        ));
        assert_eq!(builder.soa().len(), nodes);
        assert!(builder.soa().const_pool.is_empty());
    }
}
