use bml_domain::{
    transformer::{BoundedBuildError, BoundedRmsConfig},
    BMLTransformer, NodeId, NodeKind,
};

#[derive(Clone, Copy, Debug)]
enum Stage {
    Reduce,
    Prepare,
    Output,
}

impl Stage {
    fn arity(self) -> usize {
        match self {
            Self::Reduce => 2,
            Self::Prepare => 1,
            Self::Output => 4,
        }
    }

    fn append(
        self,
        builder: &mut BMLTransformer,
        operands: &[NodeId],
        config: &BoundedRmsConfig,
    ) -> Result<Vec<NodeId>, BoundedBuildError> {
        match self {
            Self::Reduce => builder
                .rms_norm_bounded_reduce(operands, config)
                .map(|root| vec![root]),
            Self::Prepare => builder
                .rms_norm_bounded_prepare(operands, config)
                .map(Vec::from),
            Self::Output => builder
                .rms_norm_bounded_output(operands, config)
                .map(|root| vec![root]),
        }
    }

    fn build(self, config: &BoundedRmsConfig) -> (BMLTransformer, Vec<NodeId>) {
        let mut builder = BMLTransformer::new();
        let operands: Vec<_> = (0..self.arity())
            .map(|index| {
                if matches!(self, Self::Output) && index == 1 {
                    // Learned weights stay external to the fixed literal prefix.
                    builder.external_constant(17)
                } else {
                    builder.var(index as u32)
                }
            })
            .collect();
        let roots = self.append(&mut builder, &operands, config).unwrap();
        assert_eq!(builder.soa().const_pool, config.fixed_literals());
        (builder, roots)
    }
}

// Independent evaluation of every appended node, including intermediates not
// retained as outputs. BML exp2 underflow and nonpositive log2 inputs fail.
fn checked(
    builder: &BMLTransformer,
    roots: &[NodeId],
    inputs: &[f64],
    external_weights: &[f64],
) -> Vec<f64> {
    let mut constants = builder.soa().const_pool.clone();
    constants.extend_from_slice(external_weights);
    let mut values = Vec::<f64>::with_capacity(builder.soa().len());
    for id in 0..builder.soa().len() as NodeId {
        let node = builder.soa().get(id);
        let value = match node.kind {
            NodeKind::One => 1.0,
            NodeKind::Zero => panic!("unexpected zero outside the historical bounded basis"),
            NodeKind::Var(index) => inputs[index as usize],
            NodeKind::Const(index) => constants[index as usize],
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

fn compare_chained(config: &BoundedRmsConfig, inputs: &[f64], weights: &[f64]) {
    config.validate_values(inputs, weights).unwrap();
    let mut original = BMLTransformer::new();
    let input_nodes: Vec<_> = (0..config.width())
        .map(|index| original.var(index as u32))
        .collect();
    let weight_nodes: Vec<_> = (0..config.width())
        .map(|index| original.external_constant(17 + index as u32))
        .collect();
    let roots = original
        .rms_norm_bounded(&input_nodes, &weight_nodes, config)
        .unwrap();
    let original_values = checked(&original, &roots, inputs, weights);
    let (reduce, reduce_roots) = Stage::Reduce.build(config);
    let (prepare, prepare_roots) = Stage::Prepare.build(config);
    let (output, output_roots) = Stage::Output.build(config);
    let mut total = config.fixed_literals()[4];
    for &input in inputs {
        total = checked(&reduce, &reduce_roots, &[total, input], &[])[0];
    }
    let prepared = checked(&prepare, &prepare_roots, &[total], &[]);
    let denominator =
        (inputs.iter().map(|x| x * x).sum::<f64>() / inputs.len() as f64 + config.epsilon()).sqrt();
    for (index, (&input, &weight)) in inputs.iter().zip(weights).enumerate() {
        let actual = checked(
            &output,
            &output_roots,
            &[input, weight, prepared[0], prepared[1]],
            &[weight],
        )[0];
        assert_eq!(
            actual.to_bits(),
            original_values[index].to_bits(),
            "width={}, index={index}, epsilon={}",
            config.width(),
            config.epsilon()
        );
        let expected = input * weight / denominator;
        assert!(
            (actual - expected).abs() <= 2e-6,
            "width={}, index={index}: actual={actual}, oracle={expected}",
            config.width()
        );
    }
}

#[test]
fn bounded_rms_stages_match_original_bits_and_independent_oracle() {
    for width in [1, 2, 4, 32, 288, 768] {
        for epsilon in [1e-5, 0.001, 0.25] {
            let config = BoundedRmsConfig::new(width, epsilon, 16.0, 16.0).unwrap();
            let inputs: Vec<_> = (0..width)
                .map(|i| [-16.0, 16.0, 0.0, -0.0, 0.1, -0.03, 4.0, -7.0][i % 8])
                .collect();
            let weights: Vec<_> = (0..width)
                .map(|i| [16.0, -16.0, 0.0, 4.0, -0.25][i % 5])
                .collect();
            compare_chained(&config, &inputs, &weights);
            compare_chained(&config, &vec![0.0; width], &weights);
        }
    }
}

#[test]
fn bounded_rms_stages_cover_sparse_inputs_and_extreme_valid_bounds() {
    let config = BoundedRmsConfig::new(288, 1e-5, 16.0, 16.0).unwrap();
    let mut inputs = vec![0.0; 288];
    inputs[167] = -16.0;
    let weights = vec![-16.0; 288];
    compare_chained(&config, &inputs, &weights);
    compare_chained(
        &BoundedRmsConfig::new(1, 256.0 * 256.0, 256.0, 240.0).unwrap(),
        &[-256.0],
        &[240.0],
    );
    compare_chained(
        &BoundedRmsConfig::new(4, 1e-5, 1.0 / 256.0, 1.0 / 256.0).unwrap(),
        &[0.0, 1.0 / 256.0, -1.0 / 256.0, 0.001],
        &[1.0 / 256.0; 4],
    );
}

#[test]
fn bounded_rms_stages_have_constant_size_and_exact_literal_prefix() {
    for stage in [Stage::Reduce, Stage::Prepare, Stage::Output] {
        let counts: Vec<_> = [1, 2, 32, 288, 768, 65025]
            .iter()
            .map(|&width| {
                let config = BoundedRmsConfig::new(width, 1e-5, 16.0, 1.0).unwrap();
                let (builder, _) = stage.build(&config);
                assert_eq!(builder.soa().const_pool.len(), 17);
                assert!(builder.soa().len() < 16384);
                builder.soa().len()
            })
            .collect();
        assert!(counts.iter().all(|count| *count == counts[0]));
        println!("{stage:?}: {} nodes", counts[0]);
    }
}

#[test]
fn bounded_rms_stage_errors_do_not_mutate_existing_nodes_or_literals() {
    let config = BoundedRmsConfig::new(32, 1e-5, 16.0, 16.0).unwrap();
    for stage in [Stage::Reduce, Stage::Prepare, Stage::Output] {
        let mut builder = BMLTransformer::new();
        let input = builder.var(0);
        builder.constant(123.0);
        let before = builder.soa().clone();
        for length in [0, stage.arity() - 1, stage.arity() + 1] {
            assert_eq!(
                stage.append(&mut builder, &vec![input; length], &config),
                Err(BoundedBuildError::Shape)
            );
        }
        for index in 0..stage.arity() {
            let mut operands = vec![input; stage.arity()];
            operands[index] = u32::MAX;
            assert_eq!(
                stage.append(&mut builder, &operands, &config),
                Err(BoundedBuildError::InvalidNode(u32::MAX))
            );
        }
        assert_eq!(builder.soa().kinds, before.kinds);
        assert_eq!(builder.soa().lefts, before.lefts);
        assert_eq!(builder.soa().rights, before.rights);
        assert_eq!(builder.soa().const_pool, before.const_pool);
        stage
            .append(&mut builder, &vec![input; stage.arity()], &config)
            .unwrap();
        assert_eq!(builder.soa().const_pool[0], 123.0);
        assert_eq!(builder.soa().const_pool[1..], config.fixed_literals());
    }
}
