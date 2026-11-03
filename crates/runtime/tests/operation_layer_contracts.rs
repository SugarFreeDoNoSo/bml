use bml_compiler::{
    op_fragments::{
        compile_layer_fragments, compile_layer_fragments_checked, CompiledLayerFragments,
        LayerConfig, LayerLayout, LayerWeights,
    },
    symbolic_bounded::BoundedProfile,
    BmlGraph, RpnOp,
};
use bml_domain::EvalContext;
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
    Runtime,
};

#[derive(Clone)]
struct Weights {
    norm: Vec<f64>,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    o: Vec<f64>,
    ffn_norm: Vec<f64>,
    gate: Vec<f64>,
    up: Vec<f64>,
    down: Vec<f64>,
}

fn weights(n: usize, ffn: usize) -> Weights {
    let matrix = |rows, cols, seed| {
        (0..rows * cols)
            .map(|i| ((i * seed + 3) % 17) as f64 / 32.0 - 0.22)
            .collect()
    };
    Weights {
        norm: (0..n)
            .map(|i| if i % 2 == 0 { 1.1 } else { -0.8 })
            .collect(),
        q: matrix(n, n, 7),
        k: matrix(n, n, 5),
        v: matrix(n, n, 3),
        o: matrix(n, n, 11),
        ffn_norm: (0..n)
            .map(|i| if i % 2 == 0 { -0.9 } else { 1.3 })
            .collect(),
        gate: matrix(ffn, n, 13),
        up: matrix(ffn, n, 2),
        down: matrix(n, ffn, 9),
    }
}

fn matmul(x: &[f64], w: &[f64]) -> Vec<f64> {
    w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, x)| w * x).sum())
        .collect()
}

fn norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let root = (x.iter().map(|x| x * x).sum::<f64>() / x.len() as f64 + eps).sqrt();
    x.iter().zip(w).map(|(x, w)| x * w / root).collect()
}

fn rotate(mut x: Vec<f64>, position: u32, base: f64) -> Vec<f64> {
    let n = x.len();
    for pair in 0..n / 2 {
        let angle = position as f64 * base.powf(-2.0 * pair as f64 / n as f64);
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (x[2 * pair], x[2 * pair + 1]);
        x[2 * pair] = a * cos - b * sin;
        x[2 * pair + 1] = a * sin + b * cos;
    }
    x
}

// Independent dense Llama equations: no compiler/domain helper or bytecode.
fn reference(
    input: &[f64],
    w: &Weights,
    config: LayerConfig,
    keys: &mut Vec<f64>,
    values: &mut Vec<f64>,
) -> Vec<f64> {
    let n = input.len();
    let normalized = norm(input, &w.norm, config.epsilon);
    let q = rotate(
        matmul(&normalized, &w.q),
        config.position,
        config.frequency_base,
    );
    keys.extend(rotate(
        matmul(&normalized, &w.k),
        config.position,
        config.frequency_base,
    ));
    values.extend(matmul(&normalized, &w.v));
    let mut scores: Vec<f64> = keys
        .chunks_exact(n)
        .map(|k| k.iter().zip(&q).map(|(k, q)| k * q).sum::<f64>() / (n as f64).sqrt())
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for score in &mut scores {
        *score = (*score - maximum).exp();
    }
    let total: f64 = scores.iter().sum();
    let mut attention = vec![0.0; n];
    for (row, score) in values.chunks_exact(n).zip(scores) {
        for (out, v) in attention.iter_mut().zip(row) {
            *out += v * (score / total);
        }
    }
    let projected = matmul(&attention, &w.o);
    let residual: Vec<_> = input.iter().zip(projected).map(|(x, y)| x + y).collect();
    let normalized = norm(&residual, &w.ffn_norm, config.epsilon);
    let gate = matmul(&normalized, &w.gate);
    let up = matmul(&normalized, &w.up);
    let hidden: Vec<_> = gate
        .iter()
        .zip(up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    residual
        .iter()
        .zip(matmul(&hidden, &w.down))
        .map(|(x, y)| x + y)
        .collect()
}

fn addresses(w: &Weights, start: u32) -> (LayerWeights, Vec<(u32, &[f64])>) {
    let mut cursor = start;
    let arrays = [
        &w.norm,
        &w.q,
        &w.k,
        &w.v,
        &w.o,
        &w.ffn_norm,
        &w.gate,
        &w.up,
        &w.down,
    ];
    let reads: Vec<_> = arrays
        .into_iter()
        .map(|values| {
            let base = cursor;
            cursor += values.len() as u32 + 1; // Sentinels between learned tensors.
            (base, values.as_slice())
        })
        .collect();
    let bases = LayerWeights {
        norm: reads[0].0,
        query: reads[1].0,
        key: reads[2].0,
        value: reads[3].0,
        output: reads[4].0,
        ffn_norm: reads[5].0,
        gate: reads[6].0,
        up: reads[7].0,
        down: reads[8].0,
    };
    (bases, reads)
}

fn config(position: u32, ffn: u32) -> LayerConfig {
    LayerConfig {
        embedding: 4,
        feed_forward: ffn,
        position,
        epsilon: 0.125,
        frequency_base: 37.0,
        bounds: BoundedProfile::default(),
    }
}

fn layout(size: u32) -> LayerLayout {
    LayerLayout {
        slot_size: size,
        input_base: 2,
        output_slot: 18,
        first_scratch_slot: 1,
        key_cache_slot: 16,
        value_cache_slot: 17,
    }
}

fn fixture(
    config: LayerConfig,
    layout: LayerLayout,
    w: &Weights,
) -> (CompiledLayerFragments, ResultBuffer) {
    let (weights, reads) = addresses(w, 20 * layout.slot_size + 1);
    let plan = compile_layer_fragments_checked(3, layout, weights, config).unwrap();
    let mut buffer = ResultBuffer::new(plan.required_slots(), layout.slot_size as usize);
    for slot in 0..buffer.n_slots() {
        buffer.slot_mut(slot).fill(77.0);
    }
    for (base, values) in reads {
        for (i, &v) in values.iter().enumerate() {
            buffer.write_indexed(base, i as u32, v);
        }
    }
    (plan, buffer)
}

fn checked(plan: &CompiledLayerFragments, buffer: &mut ResultBuffer) {
    for operation in plan.operations() {
        let mut fragment = operation.fragment.clone();
        fragment.ops.push(RpnOp::One);
        let constants = operation.const_pool.as_ref().unwrap();
        let graph = BmlGraph {
            fragments: vec![fragment],
            const_pool: vec![],
            threshold: 32768,
        };
        let graph = ValidatedGraph::new_with_external_constants(
            graph,
            0,
            constants.len(),
            buffer.n_slots(),
            buffer.slot_size(),
            GraphLimits {
                max_ops: 100_000,
                max_steps: 20_000_000,
                max_values: 1_000_000,
            },
        )
        .unwrap_or_else(|e| panic!("{}: {e}", operation.meta.name));
        assert_eq!(
            graph
                .execute_operation_with_constants(&[], constants, buffer)
                .unwrap_or_else(|e| panic!("{}: {e}", operation.meta.name)),
            1.0
        );
    }
}

fn near(actual: &[f64], expected: &[f64]) {
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() < 2e-6,
            "index={i}, actual={a}, expected={e}"
        );
    }
}

#[test]
fn complete_causal_layer_matches_dense_oracle_for_two_positions_and_all_bml_nodes() {
    let w = weights(4, 6);
    let layout = layout(13);
    let (first, mut buffer) = fixture(config(0, 6), layout, &w);
    let (bases, _) = addresses(&w, 20 * layout.slot_size + 1);
    let second = compile_layer_fragments_checked(3, layout, bases, config(1, 6)).unwrap();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (position, (plan, input)) in [
        (&first, [0.7, -0.3, 1.2, -0.4]),
        (&second, [-0.2, 0.9, -0.5, 0.8]),
    ]
    .into_iter()
    .enumerate()
    {
        for (i, &x) in input.iter().enumerate() {
            buffer.write_indexed(layout.input_base, i as u32, x);
        }
        let before = buffer.data().to_vec();
        let expected = reference(&input, &w, plan.config(), &mut keys, &mut values);
        checked(plan, &mut buffer);
        near(buffer.slot(layout.output_slot as usize), &expected);
        near(buffer.slot(layout.key_cache_slot as usize), &keys);
        near(buffer.slot(layout.value_cache_slot as usize), &values);
        for (i, (&old, &new)) in before.iter().zip(buffer.data()).enumerate() {
            let slot = i / layout.slot_size as usize;
            let offset = i % layout.slot_size as usize;
            let scratch = (1..=15).contains(&slot);
            let output = slot == layout.output_slot as usize && offset < 4;
            let cache = [
                layout.key_cache_slot as usize,
                layout.value_cache_slot as usize,
            ]
            .contains(&slot)
                && (position * 4..(position + 1) * 4).contains(&offset);
            if !(scratch || output || cache) {
                assert_eq!(old.to_bits(), new.to_bits(), "address={i}");
            }
        }
    }
}

#[test]
fn layer_dependency_scheduler_preserves_same_causal_result_and_owned_pools() {
    let w = weights(4, 6);
    let input = [-0.6, 0.4, 0.3, 1.0];
    let layout = layout(13);
    let mut keys = vec![0.2, -0.1, 0.3, -0.4];
    let mut values = vec![0.5, -0.2, 0.4, -0.1];
    let expected = reference(&input, &w, config(1, 6), &mut keys, &mut values);
    for cores in [0, 1, 2, 4, 8] {
        let (plan, mut buffer) = fixture(config(1, 6), layout, &w);
        buffer.slot_mut(layout.key_cache_slot as usize)[..4].copy_from_slice(&keys[..4]);
        buffer.slot_mut(layout.value_cache_slot as usize)[..4].copy_from_slice(&values[..4]);
        for (i, &x) in input.iter().enumerate() {
            buffer.write_indexed(layout.input_base, i as u32, x);
        }
        let mut runtime = Runtime::new(8192, 0);
        runtime.execute_with_cores(
            plan.operations(),
            &EvalContext::new(&[], &[f64::NAN]),
            &mut buffer,
            cores,
        );
        near(buffer.slot(layout.output_slot as usize), &expected);
        near(buffer.slot(layout.key_cache_slot as usize), &keys);
        near(buffer.slot(layout.value_cache_slot as usize), &values);
        assert_eq!(runtime.stack_depth(), 0);
    }
}

#[test]
fn every_learned_layer_tensor_changes_the_nontrivial_causal_result() {
    let original = weights(4, 6);
    let input = [0.7, -0.3, 1.2, -0.4];
    let past_k = vec![0.4, -0.3, 0.2, -0.1];
    let past_v = vec![-0.7, 0.5, -0.2, 0.3];
    let expected_original = reference(
        &input,
        &original,
        config(1, 6),
        &mut past_k.clone(),
        &mut past_v.clone(),
    );
    for tensor in 0..9 {
        let mut w = original.clone();
        let arrays = [
            &mut w.norm,
            &mut w.q,
            &mut w.k,
            &mut w.v,
            &mut w.o,
            &mut w.ffn_norm,
            &mut w.gate,
            &mut w.up,
            &mut w.down,
        ];
        arrays.into_iter().nth(tensor).unwrap()[0] += 0.17;
        let expected = reference(
            &input,
            &w,
            config(1, 6),
            &mut past_k.clone(),
            &mut past_v.clone(),
        );
        assert!(
            expected
                .iter()
                .zip(&expected_original)
                .any(|(a, b)| (a - b).abs() > 1e-8),
            "tensor={tensor}"
        );
        let layout = layout(13);
        let (plan, mut buffer) = fixture(config(1, 6), layout, &w);
        buffer.slot_mut(layout.key_cache_slot as usize)[..4].copy_from_slice(&past_k);
        buffer.slot_mut(layout.value_cache_slot as usize)[..4].copy_from_slice(&past_v);
        for (i, &x) in input.iter().enumerate() {
            buffer.write_indexed(layout.input_base, i as u32, x);
        }
        checked(&plan, &mut buffer);
        near(buffer.slot(layout.output_slot as usize), &expected);
    }
}

#[test]
fn layer_rejects_bad_shapes_overflows_and_aliased_buffers() {
    let w = weights(4, 6);
    let (weights, _) = addresses(&w, 261);
    let good = layout(13);
    let cfg = config(1, 6);
    for layout in [
        LayerLayout {
            slot_size: 0,
            ..good
        },
        LayerLayout {
            slot_size: 7,
            ..good
        },
        LayerLayout {
            input_base: 13,
            ..good
        },
        LayerLayout {
            input_base: u32::MAX,
            ..good
        },
        LayerLayout {
            output_slot: 1,
            ..good
        },
        LayerLayout {
            key_cache_slot: 18,
            ..good
        },
        LayerLayout {
            value_cache_slot: 16,
            ..good
        },
        LayerLayout {
            first_scratch_slot: u32::MAX,
            ..good
        },
        LayerLayout {
            output_slot: u32::MAX,
            ..good
        },
        LayerLayout {
            output_slot: 20,
            ..good
        },
    ] {
        assert!(compile_layer_fragments_checked(0, layout, weights, cfg).is_err());
    }
    for config in [
        LayerConfig {
            embedding: 0,
            ..cfg
        },
        LayerConfig {
            embedding: 3,
            ..cfg
        },
        LayerConfig {
            feed_forward: 0,
            ..cfg
        },
        LayerConfig {
            feed_forward: 14,
            ..cfg
        },
        LayerConfig {
            position: u32::MAX,
            ..cfg
        },
        LayerConfig {
            epsilon: 0.0,
            ..cfg
        },
        LayerConfig {
            epsilon: f64::NAN,
            ..cfg
        },
        LayerConfig {
            frequency_base: 0.0,
            ..cfg
        },
        LayerConfig {
            frequency_base: f64::INFINITY,
            ..cfg
        },
    ] {
        assert!(compile_layer_fragments_checked(0, good, weights, config).is_err());
    }
    assert!(compile_layer_fragments_checked(
        0,
        good,
        LayerWeights {
            down: u32::MAX,
            ..weights
        },
        cfg
    )
    .is_err());
}

#[test]
fn legacy_layer_converts_nonzero_slots_to_absolute_bases_and_keeps_both_residuals() {
    let w = weights(4, 4);
    let (weights, reads) = addresses(&w, 101);
    let fragments = compile_layer_fragments(
        2,
        2,
        4,
        weights.norm,
        weights.query,
        weights.key,
        weights.value,
        weights.output,
        weights.ffn_norm,
        weights.gate,
        weights.up,
        weights.down,
        u32::MAX,
    );
    assert_eq!(fragments[0].meta.input_slot, 8);
    assert_eq!(fragments[1].meta.input_slot, 12);
    assert_eq!(fragments.last().unwrap().meta.output_slot, 20);
    let mut buffer = ResultBuffer::new(100, 4);
    let input = [0.3, -0.7, 0.4, 0.8];
    buffer.slot_mut(2).copy_from_slice(&input);
    for (base, values) in reads {
        for (i, &v) in values.iter().enumerate() {
            buffer.write_indexed(base, i as u32, v);
        }
    }
    let mut runtime = Runtime::new(8192, 0);
    runtime.execute_fragments_sequential(&fragments, &EvalContext::new(&[], &[]), &mut buffer);
    let cfg = LayerConfig {
        epsilon: 1e-5,
        frequency_base: 10000.0,
        ..config(0, 4)
    };
    let expected = reference(&input, &w, cfg, &mut Vec::new(), &mut Vec::new());
    near(buffer.slot(20), &expected);
    assert_eq!(buffer.slot(2), input);
    assert_eq!(runtime.stack_depth(), 0);
}

#[test]
fn large_layer_keeps_each_atomic_fragment_within_declared_instruction_budget() {
    let mut counts = Vec::new();
    for (embedding, feed_forward) in [(4, 6), (2048, 5632)] {
        let size = feed_forward.max(2 * embedding);
        let mut cursor = 20 * size;
        let mut allocate = |count| {
            let base = cursor;
            cursor += count;
            base
        };
        let weights = LayerWeights {
            norm: allocate(embedding),
            query: allocate(embedding * embedding),
            key: allocate(embedding * embedding),
            value: allocate(embedding * embedding),
            output: allocate(embedding * embedding),
            ffn_norm: allocate(embedding),
            gate: allocate(embedding * feed_forward),
            up: allocate(embedding * feed_forward),
            down: allocate(embedding * feed_forward),
        };
        let mut bounds = BoundedProfile::default();
        bounds.rms.weight = 4.0;
        let cfg = LayerConfig {
            embedding,
            feed_forward,
            position: 1,
            bounds,
            ..config(1, feed_forward)
        };
        let plan = compile_layer_fragments_checked(
            0,
            LayerLayout {
                input_base: 0,
                ..layout(size)
            },
            weights,
            cfg,
        )
        .unwrap();
        let sizes: Vec<_> = plan
            .operations()
            .iter()
            .map(|op| (op.meta.name.clone(), op.fragment.byte_size()))
            .collect();
        println!("layer embedding={embedding} FFN={feed_forward} fragment bytes: {sizes:?}");
        for (name, bytes) in &sizes {
            assert!(
                *bytes <= 32768,
                "{name}: {bytes} bytes exceeds fragment budget"
            );
        }
        counts.push(sizes.iter().map(|(_, bytes)| bytes).sum::<usize>());
    }
    assert_eq!(
        counts[0], counts[1],
        "stored code must not unroll matrix dimensions"
    );
}
