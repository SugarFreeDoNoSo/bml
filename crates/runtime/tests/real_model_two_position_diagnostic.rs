//! Opt-in complete vectors for two fixed TinyLlama positions, without a parity threshold.
use bml_compiler::{gguf_compiler::InferenceCompiler, HashConsedKV};
use bml_compiler::{
    symbolic_artifact,
    symbolic_bounded::{BoundedProfile, ProjectionBounds},
    symbolic_kernels::RmsBounds,
};
use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
use std::{io::Write, path::Path, time::Instant};

#[path = "support/prompt_diagnostic.rs"]
mod prompt_diagnostic;
#[path = "support/prompt_profile.rs"]
mod prompt_profile;
#[allow(dead_code)]
#[path = "support/symbolic_fixture.rs"]
mod symbolic_fixture;

#[test]
#[ignore = "requires pinned TinyLlama GGUF/AOT, BML_TEST_PROMPT=Hello|Hi, BML_TEST_SOURCE_REVISION and a new BML_TEST_LOGITS_DIR; executes native only"]
fn real_model_native_prompt_profile_preflight() {
    prompt_diagnostic::run_native_only().unwrap();
}

#[test]
#[ignore = "requires pinned TinyLlama GGUF/AOT, BML_TEST_PROMPT=Hello|Hi, BML_TEST_SOURCE_REVISION and a new BML_TEST_LOGITS_DIR; full BML can take hours"]
fn real_model_two_position_prompt_symbolic_logit_diagnostic() {
    prompt_diagnostic::run_full().unwrap();
}

#[test]
fn prompt_selection_requires_predeclared_text_exact_rust_ids_and_context_two() {
    for (text, ids) in prompt_diagnostic::CASES {
        assert_eq!(
            prompt_diagnostic::validate_prompt(text, &ids, 2).unwrap(),
            ids
        );
        assert!(prompt_diagnostic::validate_prompt(text, &ids[..1], 2).is_err());
        assert!(prompt_diagnostic::validate_prompt(text, &[ids[0], ids[1] + 1], 2).is_err());
        assert!(prompt_diagnostic::validate_prompt(text, &[ids[0], ids[1], 2], 2).is_err());
        assert!(prompt_diagnostic::validate_prompt(text, &ids, 1).is_err());
        assert!(prompt_diagnostic::validate_prompt(text, &ids, 3).is_err());
    }
    for text in ["", "hello", "Hi ", "Other"] {
        assert!(prompt_diagnostic::validate_prompt(text, &[1, 6324], 2).is_err());
    }
}

#[test]
fn native_role_probe_preserves_two_position_logits_and_checks_every_family() {
    let source = symbolic_fixture::fixture(0.25);
    let native = InferenceCompiler::open_mapped(&source.0).unwrap();
    let mut probe = prompt_profile::Probe::new(&native, prompt_diagnostic::profile());
    let mut observed_cache = HashConsedKV::new(1, 1, 2, 2);
    let mut reference_cache = HashConsedKV::new(1, 1, 2, 2);
    for (position, id) in [69, 70].into_iter().enumerate() {
        let actual = probe.position(&native, id, &mut observed_cache).unwrap();
        let expected = native.forward_cached(&[id], &mut reference_cache).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(actual.iter().all(|value| value.is_finite()));
        assert_eq!(observed_cache.current_pos(), position as u32 + 1);
        let report = probe.report();
        assert_eq!(report["projection_callbacks"], 8);
        assert_eq!(report["roles"].as_object().unwrap().len(), 10);
        assert_eq!(report["roles"]["rms_norm"]["calls"], 3);
        assert_eq!(report["roles"]["rope"]["calls"], 3);
        assert_eq!(report["roles"]["residual_add"]["calls"], 8);
        assert_eq!(report["roles"]["swiglu"]["calls"], 4);
        assert_eq!(
            report["roles"]["attention_dot"]["calls"],
            2 * (position + 1)
        );
        assert_eq!(report["roles"]["softmax"]["calls"], 2);
        assert_eq!(report["roles"]["weighted_sum"]["calls"], 4);
        assert_eq!(report["kv_head_rows_verified_in_bits"], 2 * (position + 1));
    }
}

#[test]
fn native_role_preflight_retains_a_real_bound_violation_before_committing() {
    let source = symbolic_fixture::fixture(2.0);
    let native = InferenceCompiler::open_mapped(&source.0).unwrap();
    let mut profile = prompt_diagnostic::profile();
    profile.rms.input = 1.0;
    profile
        .validate(native.config().n_embd as usize, native.rms_norm_epsilon())
        .unwrap();
    assert!(native
        .get_embedding(69)
        .unwrap()
        .iter()
        .any(|value| value.abs() > 1.0));
    let mut probe = prompt_profile::Probe::new(&native, profile);
    let mut cache = HashConsedKV::new(1, 1, 2, 2);
    let error = probe.position(&native, 69, &mut cache).unwrap_err();
    assert!(error.contains("attn_norm.weight rms_norm"), "{error}");
    assert_eq!(cache.current_pos(), 0);
    assert_eq!(probe.report()["projection_callbacks"], 0);
    assert_eq!(probe.report()["roles"]["rms_norm"]["calls"], 1);
}

fn save_vector(directory: &Path, name: &str, values: &[f64]) {
    let file = std::fs::File::create(directory.join(name)).unwrap();
    let mut writer = std::io::BufWriter::new(file);
    for value in values {
        writer.write_all(&value.to_le_bytes()).unwrap();
    }
    writer.flush().unwrap();
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0
}

#[test]
#[ignore = "requires real TinyLlama GGUF, a persisted context-2 symbolic artifact and a new output directory; can take hours"]
fn real_model_two_position_symbolic_logit_diagnostic() {
    let source = std::env::var_os("BML_TEST_GGUF").expect("BML_TEST_GGUF");
    let artifact =
        std::env::var_os("BML_TEST_SYMBOLIC_ARTIFACT").expect("BML_TEST_SYMBOLIC_ARTIFACT");
    let output = std::env::var_os("BML_TEST_LOGITS_DIR").expect("BML_TEST_LOGITS_DIR");
    let output = Path::new(&output);
    std::fs::create_dir(output).expect("output directory must be new");
    let native = InferenceCompiler::open_mapped(Path::new(&source)).unwrap();
    let ids = [1u32, 29958];
    assert_eq!(native.vocab().bos_token_id, ids[0]);
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
    let limits = SymbolicLimits {
        // All programs must be loaded from the authenticated AOT artifact.
        max_nodes: 0,
        max_cached_ops: 4_000_000,
        max_programs: 4096,
        graph: bml_runtime::graph::GraphLimits {
            max_ops: 2_000_000,
            max_steps: 100_000_000,
            max_values: 4_000_000,
        },
    };
    let mut symbolic =
        SymbolicCausalModel::open_artifact(Path::new(&artifact), Some(Path::new(&source)), limits)
            .unwrap();
    assert_eq!(symbolic.profile(), symbolic_artifact::BOUNDED_PROFILE);
    assert_eq!(symbolic.context_limit(), 2);
    let manifest: symbolic_artifact::Manifest = serde_json::from_slice(
        &std::fs::read(Path::new(&artifact).join(symbolic_artifact::MANIFEST)).unwrap(),
    )
    .unwrap();
    assert_eq!(
        manifest.source_sha256,
        "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"
    );
    assert_eq!(
        manifest.bounded_profile,
        Some(BoundedProfile {
            rms: RmsBounds {
                input: 160.0,
                weight: 4.0
            },
            projection_input: 64.0,
            projection_weight: 4.0,
            down_projection: Some(ProjectionBounds {
                input: 256.0,
                weight: 1.5
            }),
            attention: 20.0,
            value: 16.0,
            rope: 32.0,
            gate: 18.0,
            up: 16.0,
            residual: 160.0,
            scale_input: 256.0,
            score: 16.0,
        })
    );
    let mut native_cache = cache();
    let mut symbolic_cache = cache();
    let mut positions = Vec::new();
    for (position, &id) in ids.iter().enumerate() {
        let reference = native.forward_cached(&[id], &mut native_cache).unwrap();
        assert_eq!(reference.len(), 32_000);
        assert!(reference.iter().all(|value| value.is_finite()));
        let native_file = format!("native-{position}.f64le");
        save_vector(output, &native_file, &reference);
        eprintln!(
            "position={position} input={id} native_argmax={}",
            argmax(&reference)
        );
        let started = Instant::now();
        let actual = symbolic
            .forward_cached(&[id], &mut symbolic_cache)
            .unwrap_or_else(|error| panic!("symbolic position={position} input={id}: {error:?}"));
        let elapsed = started.elapsed().as_secs_f64();
        let symbolic_file = format!("symbolic-{position}.f64le");
        save_vector(output, &symbolic_file, &actual);
        assert_eq!(actual.len(), reference.len());
        assert!(actual.iter().all(|value| value.is_finite()));
        assert_eq!(native_cache.current_pos(), position as u32 + 1);
        assert_eq!(symbolic_cache.current_pos(), position as u32 + 1);
        let differences: Vec<_> = actual
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .collect();
        assert!(differences.iter().all(|value| value.is_finite()));
        let maximum = differences.iter().copied().fold(0.0, f64::max);
        let count = differences.len() as f64;
        let mean = differences.iter().map(|value| value / count).sum::<f64>();
        let rms = if maximum == 0.0 {
            0.0
        } else {
            maximum
                * (differences
                    .iter()
                    .map(|value| (value / maximum).powi(2))
                    .sum::<f64>()
                    / count)
                    .sqrt()
        };
        assert!(mean.is_finite() && rms.is_finite());
        let report = serde_json::json!({
            "position": position,
            "input_token": id,
            "logits": actual.len(),
            "native_argmax": argmax(&reference),
            "symbolic_argmax": argmax(&actual),
            "max_absolute_error": maximum,
            "mean_absolute_error": mean,
            "rms_error": rms,
            "native_vector": native_file,
            "symbolic_vector": symbolic_file,
            "native_cache_position": native_cache.current_pos(),
            "symbolic_cache_position": symbolic_cache.current_pos(),
            "symbolic_elapsed_seconds": elapsed,
        });
        std::fs::write(
            output.join(format!("position-{position}.json")),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        eprintln!("{report}");
        positions.push(report);
    }
    let summary = serde_json::json!({
        "input_ids": ids,
        "positions": positions,
        "operations": format!("{:?}", symbolic.executed_operations()),
        "qualification": "diagnostic only; no preset tolerance or numerical parity claim",
    });
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
}
