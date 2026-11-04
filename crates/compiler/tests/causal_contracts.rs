//! Closed-form causal attention oracle: Q=K=0 gives uniform attention over
//! exactly the available prefix; two query heads share one two-component KV head.
use bml_compiler::{gguf_compiler::InferenceCompiler, HashConsedKV};
#[path = "support/causal_fixture.rs"]
mod causal_fixture;
use causal_fixture::fixture;

#[test]
fn mapped_tensor_dimensions_include_vector_and_matrix_weights() {
    let file = fixture(false);
    let model = InferenceCompiler::open_mapped(&file.0).unwrap();
    let dims = model.tensor_dims();
    assert_eq!(dims.get("output_norm.weight"), Some(&(4, 1)));
    assert_eq!(dims.get("blk.0.attn_norm.weight"), Some(&(4, 1)));
    assert_eq!(dims.get("blk.0.ffn_norm.weight"), Some(&(4, 1)));
    assert_eq!(dims.get("blk.0.attn_q.weight"), Some(&(4, 4)));
    assert_eq!(dims.get("blk.0.attn_k.weight"), Some(&(4, 2)));
}

fn expected(actual: &[f64], hidden: [f64; 4]) {
    let denominator = (hidden.iter().map(|x| x * x).sum::<f64>() / 4.0 + 3.0).sqrt();
    for i in 0..4 {
        assert!(
            (actual[i] - hidden[i] / denominator).abs() < 1e-6,
            "{actual:?}"
        );
    }
    assert!(actual[4..].iter().all(|&x| x == 0.0));
}

#[test]
fn causal_prefix_and_incremental_gqa_match_uniform_attention_and_raw_residuals() {
    let file = fixture(false);
    let model = InferenceCompiler::open(&file.0).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 8);
    expected(
        &model.forward_cached(&[69], &mut cache).unwrap(),
        [3., 0., 1., 0.],
    );
    let incremental = model.forward_cached(&[70], &mut cache).unwrap();
    expected(&incremental, [0.5, 2.5, 0.5, 0.5]);
    assert_eq!(model.forward(&[69, 70]).unwrap(), incremental);
    expected(&model.forward(&[70, 69]).unwrap(), [2.5, 0.5, 0.5, 0.5]);
    expected(&model.forward(&[69]).unwrap(), [3., 0., 1., 0.]);
    assert_eq!(cache.current_pos(), 2);
    assert_eq!(cache.load_v(0, 0, 0), [1., 0.]);
    assert_eq!(cache.load_v(0, 1, 0), [0., 1.]);
    let mut batched = HashConsedKV::new(1, 1, 2, 8);
    assert_eq!(
        model.forward_cached(&[69, 70], &mut batched).unwrap(),
        incremental
    );
    assert_eq!(batched.current_pos(), 2);
    let third = model.forward_cached(&[69], &mut batched).unwrap();
    expected(&third, [8.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0]);
    assert_eq!(third, model.forward(&[69, 70, 69]).unwrap());
    assert_eq!(batched.current_pos(), 3);
}

#[test]
fn invalid_prefix_cache_layout_and_context_fail_before_committing_history() {
    let file = fixture(false);
    let model = InferenceCompiler::open(&file.0).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 1);
    assert!(model.forward_cached(&[69, 70], &mut cache).is_err());
    assert!(model.forward_cached(&[u32::MAX], &mut cache).is_err());
    assert_eq!(cache.current_pos(), 0);
    assert_eq!(cache.k_stats().0, 0);
    model.forward_cached(&[69], &mut cache).unwrap();
    let old = cache.load_v(0, 0, 0);
    assert!(model.forward_cached(&[70], &mut cache).is_err());
    assert_eq!(cache.current_pos(), 1);
    assert_eq!(cache.load_v(0, 0, 0), old);
    let mut wrong = HashConsedKV::new(1, 2, 2, 8);
    assert!(model.forward_cached(&[69], &mut wrong).is_err());
    assert_eq!(wrong.current_pos(), 0);
    assert!(model.forward(&[]).is_err());
    assert!(model.forward(&[69; 9]).is_err());
}

#[test]
fn arithmetic_failure_keeps_committed_prefix_and_retry_overwrites_uncommitted_rows() {
    let file = fixture(true);
    let model = InferenceCompiler::open(&file.0).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 8);
    model.forward_cached(&[70], &mut cache).unwrap();
    let committed = cache.load_v(0, 0, 0);
    let error = model.forward_cached(&[69], &mut cache).unwrap_err();
    assert!(error.contains("non-finite"), "{error}");
    assert_eq!(cache.current_pos(), 1);
    assert_eq!(cache.load_v(0, 0, 0), committed);
    assert_eq!(
        model.forward_cached(&[70], &mut cache).unwrap(),
        model.forward(&[70, 70]).unwrap()
    );
}

#[test]
fn cache_allocation_budget_and_size_arithmetic_are_checked_before_allocating() {
    assert!(HashConsedKV::try_new(1, 1, 2, 8, 127).is_err());
    assert_eq!(
        HashConsedKV::try_new(1, 1, 2, 8, 128).unwrap().byte_size(),
        128
    );
    assert!(HashConsedKV::try_new(u32::MAX, u32::MAX, u32::MAX, u32::MAX, usize::MAX).is_err());
    assert!(HashConsedKV::try_new(1, 1, 2, 0, usize::MAX).is_err());
    assert!(HashConsedKV::try_new(1, 0, 2, 8, usize::MAX).is_err());
}

#[test]
fn default_cache_budget_accepts_a_mib_and_rejects_payloads_above_512_mib() {
    use bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES;
    let cache = HashConsedKV::try_new(1, 1, 128, 1024, DEFAULT_MAX_KV_BYTES).unwrap();
    assert_eq!(cache.byte_size(), 1_048_576);
    // Rejection happens before allocating the oversized payload.
    assert!(HashConsedKV::try_new(1, 1, 128, 524_289, DEFAULT_MAX_KV_BYTES).is_err());
}

#[test]
fn rotary_pairs_use_distinct_frequencies_and_absolute_cached_positions() {
    let source = causal_fixture::wide_rope_fixture();
    for mapped in [false, true] {
        let model = if mapped {
            InferenceCompiler::open_mapped(&source.0)
        } else {
            InferenceCompiler::open(&source.0)
        }
        .unwrap();
        assert_eq!(model.head_dim(), 4);
        let mut cache = HashConsedKV::new(1, 1, 4, 8);
        model.forward_cached(&[69, 70], &mut cache).unwrap();
        let logits = model.forward_cached(&[69], &mut cache).unwrap();
        assert_eq!(logits, model.forward(&[69, 70, 69]).unwrap());
        // Each embedding has two entries of 2: RMS denominator sqrt(8/4+3).
        // Identity K then rotates adjacent pairs at angles pos and pos/4
        // because base=16 and head width=4. Compare the persisted F32 keys.
        let scale = 2.0 / 5.0_f64.sqrt();
        for pos in 0..3 {
            let mut expected = Vec::new();
            for angle in [f64::from(pos), f64::from(pos) / 4.0] {
                let pair = if pos == 1 {
                    [-scale * angle.sin(), scale * angle.cos()]
                } else {
                    [scale * angle.cos(), scale * angle.sin()]
                };
                expected.extend(pair);
            }
            for (actual, expected) in cache.load_k(0, pos, 0).into_iter().zip(expected) {
                assert!(
                    (f64::from(actual) - expected).abs() < 1e-7,
                    "mapped={mapped} position={pos}: {actual} != {expected}"
                );
            }
        }
        assert_eq!(cache.current_pos(), 3);
    }
}

#[test]
fn distributed_legacy_compilers_reject_placeholder_output() {
    use bml_compiler::distributed::{compile_distributed, compile_distributed_bml};
    let file = fixture(false);
    let plain = compile_distributed(&file.0, 1, true).unwrap_err();
    let encoded = compile_distributed_bml(&file.0, 1).err().unwrap();
    for error in [plain, encoded] {
        assert!(error.contains("retired"), "{error}");
        assert!(error.contains("compile_directory_bounded"), "{error}");
    }
}

#[test]
fn projection_backend_errors_and_invalid_outputs_do_not_commit_cache_positions() {
    let model = fixture(false);
    let compiler = InferenceCompiler::open(&model.0).unwrap();
    for bad in [0, 1, 2] {
        let mut cache = HashConsedKV::new(1, 1, 2, 4);
        let mut calls = 0;
        let result = compiler.forward_cached_with_projection(&[69], &mut cache, &mut |_, _| {
            calls += 1;
            match bad {
                0 => Err("backend failure".into()),
                1 => Ok(vec![0.0]),
                _ => Ok(vec![f64::NAN; 4]),
            }
        });
        assert!(result.is_err());
        assert_eq!(calls, 1);
        assert_eq!(cache.current_pos(), 0);
        assert_eq!(
            compiler.forward_cached(&[69], &mut cache).unwrap(),
            compiler.forward(&[69]).unwrap()
        );
    }
}

#[test]
fn nonfinite_activations_are_rejected_before_calling_the_projection_backend() {
    let source = fixture(false);
    let model = InferenceCompiler::open(&source.0).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 4);
    let mut down_called = false;
    let result = model.forward_cached_with_projection(&[69], &mut cache, &mut |input, name| {
        assert!(
            input.iter().all(|x| x.is_finite()),
            "backend input for {name}"
        );
        down_called |= name.ends_with("ffn_down.weight");
        let &(_, rows) = model.tensor_dims().get(name).unwrap();
        // Individually finite gate/up projections overflow in SwiGLU. A backend
        // must never receive that nonfinite activation as its down-projection input.
        let value = if name.ends_with("ffn_gate.weight") || name.ends_with("ffn_up.weight") {
            f64::MAX
        } else {
            0.0
        };
        Ok(vec![value; rows])
    });
    assert_eq!(
        result.unwrap_err(),
        "invalid projection input for blk.0.ffn_down.weight"
    );
    assert!(!down_called);
    assert_eq!(cache.current_pos(), 0);
    assert_eq!(
        model.forward_cached(&[69], &mut cache).unwrap(),
        model.forward(&[69]).unwrap()
    );
}

#[test]
fn late_projection_failure_preserves_committed_history_for_retry() {
    let model = fixture(false);
    let compiler = InferenceCompiler::open(&model.0).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 4);
    compiler.forward_cached(&[69], &mut cache).unwrap();
    let mut head_called = false;
    let result = compiler.forward_cached_with_projection(&[70], &mut cache, &mut |input, name| {
        if name == compiler.output_weight_name() {
            head_called = true;
            return Err("output backend failed".into());
        }
        let &(n_in, n_out) = compiler.tensor_dims().get(name).unwrap();
        let offset = compiler.weight_offsets()[name] as usize;
        Ok((0..n_out)
            .map(|row| {
                input
                    .iter()
                    .enumerate()
                    .map(|(col, x)| {
                        x * f64::from(compiler.weight_pool()[offset + row * n_in + col])
                    })
                    .sum()
            })
            .collect())
    });
    assert!(result.is_err());
    assert!(head_called);
    assert_eq!(cache.current_pos(), 1);
    assert_eq!(
        compiler.forward_cached(&[70], &mut cache).unwrap(),
        compiler.forward(&[69, 70]).unwrap()
    );
}
