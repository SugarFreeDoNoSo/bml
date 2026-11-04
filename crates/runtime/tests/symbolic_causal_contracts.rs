use bml_compiler::{gguf_compiler::InferenceCompiler, HashConsedKV};
use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
#[path = "support/symbolic_fixture.rs"]
mod symbolic_fixture;
#[path = "support/symbolic_oracle.rs"]
mod symbolic_oracle;
use symbolic_oracle::logits as oracle;

/// Opt-in measurement, not a numerical-parity gate. Preserve both complete
/// vectors so aggregate statistics cannot hide an individual bad logit.
#[test]
#[ignore = "requires BML_TEST_GGUF, BML_TEST_SYMBOLIC_ARTIFACT and a new BML_TEST_LOGITS_DIR; full real-model execution can take hours"]
fn real_model_bos_symbolic_logit_diagnostic() {
    use std::{io::Write, path::PathBuf, time::Instant};
    let source = PathBuf::from(std::env::var_os("BML_TEST_GGUF").expect("BML_TEST_GGUF"));
    let artifact = PathBuf::from(
        std::env::var_os("BML_TEST_SYMBOLIC_ARTIFACT").expect("BML_TEST_SYMBOLIC_ARTIFACT"),
    );
    let output =
        PathBuf::from(std::env::var_os("BML_TEST_LOGITS_DIR").expect("BML_TEST_LOGITS_DIR"));
    std::fs::create_dir(&output).expect("output directory must be new");
    let native = InferenceCompiler::open_mapped(&source).unwrap();
    let ids = native.vocab().encode("").unwrap();
    assert_eq!(ids, vec![native.vocab().bos_token_id]);
    let cache = || {
        HashConsedKV::try_new(
            native.config().n_layers,
            native.n_kv_heads(),
            native.head_dim(),
            2,
            bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES,
        )
        .unwrap()
    };
    let save = |name: &str, values: &[f64]| {
        let file = std::fs::File::create(output.join(name)).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        for value in values {
            writer.write_all(&value.to_le_bytes()).unwrap();
        }
        writer.flush().unwrap();
    };
    let argmax = |values: &[f64]| {
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0
    };
    let mut native_cache = cache();
    let reference = native.forward_cached(&ids, &mut native_cache).unwrap();
    assert!(reference.iter().all(|x| x.is_finite()));
    save("native-logits.f64le", &reference);
    let native_id = argmax(&reference);
    eprintln!(
        "native_bos_argmax={native_id} decoded={:?} logits={}",
        native.vocab().decode(&[native_id as u32]).unwrap(),
        reference.len()
    );
    let limits = SymbolicLimits {
        max_nodes: 2_000_000,
        max_cached_ops: 4_000_000,
        max_programs: 4096,
        graph: bml_runtime::graph::GraphLimits {
            max_ops: 2_000_000,
            max_steps: 100_000_000,
            max_values: 4_000_000,
        },
    };
    let mut symbolic =
        SymbolicCausalModel::open_artifact(&artifact, Some(&source), limits).unwrap();
    let mut symbolic_cache = cache();
    let started = Instant::now();
    let actual = symbolic.forward_cached(&ids, &mut symbolic_cache).unwrap();
    save("symbolic-logits.f64le", &actual);
    assert_eq!(native_cache.current_pos(), 1);
    assert_eq!(symbolic_cache.current_pos(), 1);
    assert_eq!(actual.len(), reference.len());
    assert!(!actual.is_empty());
    assert!(actual.iter().all(|x| x.is_finite()));
    let max_abs_error = actual
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let symbolic_id = argmax(&actual);
    let summary = format!(
        "input_ids={ids:?}\nlogits={}\nnative_argmax={native_id}\nsymbolic_argmax={symbolic_id}\nmax_abs_error={max_abs_error:.17e}\nsymbolic_elapsed_seconds={}\noperations={:?}\nqualification=diagnostic only; no numerical tolerance or parity claim\n",
        actual.len(), started.elapsed().as_secs_f64(), symbolic.executed_operations()
    );
    std::fs::write(output.join("summary.txt"), &summary).unwrap();
    eprintln!("{summary}");
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
