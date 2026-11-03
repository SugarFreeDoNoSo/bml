use bml_compiler::{gguf_compiler::InferenceCompiler, HashConsedKV};
use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
#[path = "support/symbolic_fixture.rs"]
mod symbolic_fixture;

// Independent closed-size oracle. No production projection, normalization,
// rotary, attention, activation or evaluator helper participates here.
fn oracle(ids: &[u32], with_mlp: bool) -> Vec<f64> {
    fn norm(x: [f64; 4]) -> [f64; 4] {
        let root = (x.iter().map(|v| v * v).sum::<f64>() / 4.0 + 0.5).sqrt();
        x.map(|v| 0.5 * v / root)
    }
    fn rotate(x: [f64; 2], position: usize) -> [f64; 2] {
        let (s, c) = (position as f64).sin_cos();
        [x[0] * c - x[1] * s, x[0] * s + x[1] * c]
    }
    let mut keys = Vec::<[f32; 2]>::new();
    let mut values = Vec::<[f32; 2]>::new();
    let mut hidden = [0.0; 4];
    for (position, &id) in ids.iter().enumerate() {
        hidden = [0.0; 4];
        if id == 69 {
            hidden[0] = 0.25;
        }
        if id == 70 {
            hidden[1] = 0.25;
        }
        let n = norm(hidden);
        let q = n.map(|v| v * 0.25);
        keys.push(rotate([q[0], q[1]], position).map(|v| v as f32));
        values.push([q[0] as f32, q[1] as f32]);
        for head in 0..2 {
            let query = rotate([q[head * 2], q[head * 2 + 1]], position);
            let scores: Vec<_> = keys
                .iter()
                .map(|k| (query[0] * f64::from(k[0]) + query[1] * f64::from(k[1])) / 2.0_f64.sqrt())
                .collect();
            let exps: Vec<_> = scores.iter().map(|s| s.exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..2 {
                let attention: f64 = values
                    .iter()
                    .zip(&exps)
                    .map(|(v, e)| e / sum * f64::from(v[d]))
                    .sum();
                hidden[head * 2 + d] += 0.25 * attention;
            }
        }
        if with_mlp {
            let n = norm(hidden);
            for (h, v) in hidden.iter_mut().zip(n) {
                let g = 0.25 * v;
                let up = -0.25 * v;
                *h += 0.25 * g / (1.0 + (-g).exp()) * up;
            }
        }
    }
    let mut logits = vec![0.0; 260];
    for (out, v) in logits.iter_mut().zip(norm(hidden)) {
        *out = 0.5 * v;
    }
    logits
}
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 2e-7, "logit {i}: {a} != {b}");
    }
}
#[test]
fn full_symbolic_causal_model_matches_independent_gqa_rope_and_nonzero_mlp_oracle() {
    let source = symbolic_fixture::fixture(0.25);
    let mut model = SymbolicCausalModel::open(&source.0, SymbolicLimits::default()).unwrap();
    let native = InferenceCompiler::open_mapped(&source.0).unwrap();
    for ids in [&[69][..], &[69, 70][..], &[70, 69][..], &[69, 70, 69][..]] {
        let actual = model.forward(ids).unwrap();
        close(&actual, &oracle(ids, true));
        close(&actual, &native.forward(ids).unwrap());
        assert!(
            oracle(ids, true)
                .iter()
                .zip(oracle(ids, false))
                .any(|(with, without)| (with - without).abs() > 1e-6),
            "fixture must expose missing MLP beyond the comparison tolerance"
        );
    }
    assert_eq!(
        model.executed_operations(),
        vec![
            "dot",
            "residual_add",
            "rms_norm",
            "rope",
            "scale",
            "softmax",
            "swiglu"
        ]
    );
    assert!(model.compiler().mapped_bytes() > 0);
    let mut cache = HashConsedKV::new(1, 1, 2, 4);
    model.forward_cached(&[69, 70], &mut cache).unwrap();
    let actual = model.forward_cached(&[69], &mut cache).unwrap();
    close(&actual, &oracle(&[69, 70, 69], true));
    assert_eq!(cache.current_pos(), 3);
}

#[test]
fn converted_autonomous_model_tokenizer_logits_and_generation_match_independent_oracle() {
    use bml_compiler::gguf_compiler::prepared_model::WeightEvaluation;
    use bml_model_weights::{convert, ConversionOptions, ReadLimits};
    let source = symbolic_fixture::fixture(0.25);
    let output = symbolic_fixture::Fixture(source.0.with_extension("bmlgguf"));
    let mut options = ConversionOptions::default();
    options.write.conversion.max_cache_bytes = 1 << 20;
    let report = convert::from_gguf(&source.0, &output.0, options).unwrap();
    assert!(report.weights_reused > 0);
    std::fs::remove_file(&source.0).unwrap();
    let error = SymbolicCausalModel::open_bmlgguf(
        &output.0,
        ReadLimits::default(),
        1,
        WeightEvaluation::CheckedExactF32,
        SymbolicLimits::default(),
    )
    .err()
    .unwrap();
    assert!(error.contains("materialization limit"));
    let mut model = SymbolicCausalModel::open_bmlgguf(
        &output.0,
        ReadLimits::default(),
        10000,
        WeightEvaluation::CheckedExactF32,
        SymbolicLimits::default(),
    )
    .unwrap();
    assert_eq!(
        model.profile(),
        "bmlgguf-checked-exact-f32-symbolic-causal-v1"
    );
    assert_eq!(model.compiler().mapped_bytes(), 0);
    // Literal byte-token oracle, not another tokenizer implementation.
    let ids = model.compiler().vocab().encode("AB").unwrap();
    assert_eq!(ids, vec![1, 3, 69, 70]);
    // The decoder preserves the explicit SentencePiece leading-space token.
    assert_eq!(model.compiler().vocab().decode(&ids).unwrap(), " AB");
    for prefix in [&[69][..], &[69, 70][..], ids.as_slice()] {
        close(&model.forward(prefix).unwrap(), &oracle(prefix, true));
    }
    let mut prefix = ids.clone();
    let mut cache = HashConsedKV::new(1, 1, 2, 8);
    let mut logits = model.forward_cached(&ids, &mut cache).unwrap();
    let mut generated = Vec::new();
    for _ in 0..2 {
        let expected = oracle(&prefix, true);
        close(&logits, &expected);
        let expected_token =
            expected
                .iter()
                .enumerate()
                .fold(0, |best, (i, &v)| if v > expected[best] { i } else { best })
                as u32;
        let token = bml_compiler::sampler::sample(&logits, 0.0, 0).unwrap();
        assert_eq!(token, expected_token);
        generated.push(token);
        prefix.push(token);
        logits = model.forward_cached(&[token], &mut cache).unwrap();
    }
    assert_eq!(generated.len(), 2);
    assert_eq!(cache.current_pos(), prefix.len() as u32);
    assert_eq!(
        model.executed_operations(),
        vec![
            "dot",
            "residual_add",
            "rms_norm",
            "rope",
            "scale",
            "softmax",
            "swiglu"
        ]
    );
}
#[test]
fn symbolic_failures_preserve_committed_positions_and_allow_retry() {
    let source = symbolic_fixture::fixture(0.25);
    let limits = SymbolicLimits {
        // Nine operation/width templates serve the first token in v2; the
        // second token requires a new softmax-width-two program after KV writes.
        max_programs: 9,
        ..SymbolicLimits::default()
    };
    let mut limited = SymbolicCausalModel::open(&source.0, limits).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 4);
    limited.forward_cached(&[69], &mut cache).unwrap();
    assert_eq!(cache.current_pos(), 1);
    assert!(limited
        .forward_cached(&[70], &mut cache)
        .unwrap_err()
        .contains("budget"));
    assert_eq!(cache.current_pos(), 1);
    let mut model = SymbolicCausalModel::open(&source.0, SymbolicLimits::default()).unwrap();
    close(
        &model.forward_cached(&[70], &mut cache).unwrap(),
        &oracle(&[69, 70], true),
    );
    assert_eq!(cache.current_pos(), 2);
    for ids in [&[][..], &[u32::MAX][..], &[69, 70, 69][..]] {
        assert!(model.forward_cached(ids, &mut cache).is_err());
        assert_eq!(cache.current_pos(), 2);
    }
    let mut wrong = HashConsedKV::new(1, 2, 2, 4);
    assert!(model.forward_cached(&[69], &mut wrong).is_err());
    assert_eq!(wrong.current_pos(), 0);
}
#[test]
fn symbolic_numerical_and_construction_limits_are_explicit() {
    let source = symbolic_fixture::fixture(1024.0);
    let mut model = SymbolicCausalModel::open(&source.0, SymbolicLimits::default()).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 2);
    assert!(InferenceCompiler::open_mapped(&source.0)
        .unwrap()
        .forward(&[69])
        .unwrap()
        .iter()
        .all(|x| x.is_finite()));
    assert!(model
        .forward_cached(&[69], &mut cache)
        .unwrap_err()
        .contains("domain/range"));
    assert_eq!(cache.current_pos(), 0);
    let source = symbolic_fixture::fixture(0.25);
    for limits in [
        SymbolicLimits {
            max_nodes: 1,
            ..SymbolicLimits::default()
        },
        SymbolicLimits {
            max_cached_ops: 1,
            ..SymbolicLimits::default()
        },
    ] {
        let mut model = SymbolicCausalModel::open(&source.0, limits).unwrap();
        assert!(model
            .forward_cached(&[69], &mut cache)
            .unwrap_err()
            .contains("budget"));
        assert_eq!(cache.current_pos(), 0);
    }
}
