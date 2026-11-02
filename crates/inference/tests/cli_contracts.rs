//! Black-box CLI contracts with a tiny self-contained, quantized GGUF model.
//! The ordinary suite never invokes a Rust build for AOT or external models.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Files(PathBuf);

impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "bml_cli_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn source(&self) -> PathBuf {
        let path = self.path("source modelo ñ.gguf");
        std::fs::write(&path, fixture()).unwrap();
        path
    }

    fn command(&self, action: &str, model: &Path) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_bml-model"));
        command
            .arg(action)
            .arg("--model")
            .arg(model)
            .current_dir(&self.0);
        command
    }

    fn run(&self, model: &Path, args: &[&str]) -> Output {
        self.command("run", model).args(args).output().unwrap()
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn json(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("CLI must emit exactly one JSON document")
}

fn rejected(output: &Output) -> String {
    assert!(
        !output.status.success(),
        "unexpected success: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        output.stdout.is_empty(),
        "failed commands must not emit partial JSON"
    );
    let message = String::from_utf8(output.stderr.clone()).unwrap();
    assert!(!message.is_empty());
    assert!(!message.contains("panicked"), "{message}");
    message
}

fn without_times(mut value: Value) -> Value {
    value.as_object_mut().unwrap().remove("prefill_ms");
    value.as_object_mut().unwrap().remove("decode_ms");
    value
}

fn string(value: &str) -> Vec<u8> {
    let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
    bytes.extend(value.as_bytes());
    bytes
}

fn array(kind: u32, values: Vec<Vec<u8>>) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend((values.len() as u64).to_le_bytes());
    bytes.extend(values.into_iter().flatten());
    bytes
}

fn fixture() -> Vec<u8> {
    let mut tokens = vec!["<unk>".to_owned(), "<s>".into(), "</s>".into(), "▁".into()];
    tokens.extend((0..256).map(|byte| format!("<0x{byte:02X}>")));
    let vocab = tokens.len() as u64;
    let mut metadata = vec![
        ("general.architecture", 8u32, string("llama")),
        ("tokenizer.ggml.model", 8, string("llama")),
        (
            "tokenizer.ggml.tokens",
            9,
            array(8, tokens.iter().map(|token| string(token)).collect()),
        ),
        (
            "tokenizer.ggml.scores",
            9,
            array(
                6,
                tokens
                    .iter()
                    .map(|_| (-1.0f32).to_le_bytes().to_vec())
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            9,
            array(
                5,
                (0..tokens.len())
                    .map(|i| {
                        let kind: i32 = match i {
                            0 => 2,
                            1 | 2 => 3,
                            3 => 1,
                            _ => 6,
                        };
                        kind.to_le_bytes().to_vec()
                    })
                    .collect(),
            ),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            6,
            1e-5f32.to_le_bytes().to_vec(),
        ),
        ("llama.rope.freq_base", 6, 10000.0f32.to_le_bytes().to_vec()),
    ];
    for (name, value) in [
        ("llama.embedding_length", 32u32),
        ("llama.block_count", 1),
        ("llama.attention.head_count", 4),
        ("llama.attention.head_count_kv", 2),
        ("llama.feed_forward_length", 32),
        ("llama.context_length", 8),
        ("llama.rope.dimension_count", 8),
        ("tokenizer.ggml.bos_token_id", 1),
        ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.unknown_token_id", 0),
    ] {
        metadata.push((name, 4, value.to_le_bytes().to_vec()));
    }
    for (name, value) in [
        ("tokenizer.ggml.add_bos_token", true),
        ("tokenizer.ggml.add_eos_token", false),
        ("tokenizer.ggml.add_space_prefix", true),
        ("tokenizer.ggml.remove_extra_whitespaces", false),
        ("tokenizer.ggml.byte_fallback", true),
    ] {
        metadata.push((name, 7, vec![u8::from(value)]));
    }
    let quantized = |rows: u64, value: u8| {
        let mut bytes = Vec::new();
        for _ in 0..rows {
            bytes.extend(0x3800u16.to_le_bytes()); // Independent half encoding: 0.5.
            bytes.extend([value; 16]);
        }
        bytes
    };
    let norm = || {
        (0..32)
            .flat_map(|_| 1.0f32.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut tensors = vec![
        (
            "token_embd.weight",
            vec![32, vocab],
            2u32,
            quantized(vocab, 0xc2),
        ),
        ("output_norm.weight", vec![32], 0, norm()),
        ("blk.0.attn_norm.weight", vec![32], 0, norm()),
        ("blk.0.ffn_norm.weight", vec![32], 0, norm()),
    ];
    for (name, rows) in [
        ("blk.0.attn_q.weight", 32u64),
        ("blk.0.attn_k.weight", 16),
        ("blk.0.attn_v.weight", 16),
        ("blk.0.attn_output.weight", 32),
        ("blk.0.ffn_gate.weight", 32),
        ("blk.0.ffn_up.weight", 32),
        ("blk.0.ffn_down.weight", 32),
    ] {
        tensors.push((name, vec![32, rows], 2, quantized(rows, 0x88)));
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (name, kind, value) in metadata {
        bytes.extend(string(name));
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    let mut data = Vec::new();
    for (name, dims, kind, payload) in tensors {
        data.resize(data.len().div_ceil(32) * 32, 0);
        bytes.extend(string(name));
        bytes.extend((dims.len() as u32).to_le_bytes());
        for dimension in dims {
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(kind.to_le_bytes());
        bytes.extend((data.len() as u64).to_le_bytes());
        data.extend(payload);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(data);
    bytes
}

#[test]
fn gguf_export_inspection_and_execution_preserve_the_same_model() {
    let files = Files::new();
    let source = files.source();
    let destination = files.path("normalized modelo ñ.bmlres");
    let manifest = json(
        &files
            .command("export", &source)
            .arg("--output")
            .arg(&destination)
            .output()
            .unwrap(),
    );
    assert!(destination.is_file());
    assert_eq!(manifest["config"]["embedding_length"], 32);
    assert_eq!(manifest["config"]["vocab_size"], 260);
    let original = json(&files.command("inspect", &source).output().unwrap());
    let normalized = json(&files.command("inspect", &destination).output().unwrap());
    assert_eq!(original, normalized);
    assert_eq!(original["config"]["head_count"], 4);
    assert_eq!(original["config"]["head_count_kv"], 2);
    assert!(original["plan"].as_array().unwrap().len() > 10);
    let args = [
        "--tokens",
        "1,3,69",
        "--context",
        "8",
        "--predict",
        "2",
        "--logits",
    ];
    let direct = without_times(json(&files.run(&source, &args)));
    let exported = without_times(json(&files.run(&destination, &args)));
    assert_eq!(direct, exported);
    assert_eq!(direct["prompt_tokens"], serde_json::json!([1, 3, 69]));
    assert_eq!(direct["tokens"], serde_json::json!([0, 0]));
    assert_eq!(direct["context"], 8);
    assert_eq!(direct["kv_bytes"], 1024);
    let logits = direct["logits"].as_array().unwrap();
    assert_eq!(logits.len(), 3);
    // Each token has sixteen -3 and sixteen +2 embedding components.
    // Zero projections preserve the residual; final RMSNorm and tied logits
    // therefore equal 208 / sqrt(208/32 + epsilon) for every vocabulary row.
    let expected = 208.0_f64 / (6.5_f64 + 1e-5).sqrt();
    for row in logits {
        let values = row.as_array().unwrap();
        assert_eq!(values.len(), 260);
        for value in values {
            assert!((value.as_f64().unwrap() - expected).abs() < 1e-4);
        }
    }
}

#[test]
fn prompt_and_explicit_ids_are_equivalent_and_predict_zero_is_supported() {
    let files = Files::new();
    let source = files.source();
    let prompted = without_times(json(
        &files.run(&source, &["--prompt", "A", "--predict", "0", "--logits"]),
    ));
    let explicit = without_times(json(&files.run(
        &source,
        &["--tokens", "1,3,69", "--predict", "0", "--logits"],
    )));
    assert_eq!(prompted, explicit);
    assert_eq!(prompted["tokens"], serde_json::json!([]));
    assert_eq!(prompted["logits"].as_array().unwrap().len(), 1);
    assert_eq!(prompted["text"], "");
    let without = json(&files.run(&source, &["--tokens", "259", "--predict", "0"]));
    assert!(without.get("logits").is_none());
    assert_eq!(without["prompt_tokens"], serde_json::json!([259]));
    let unicode = json(&files.run(&source, &["--prompt", "é", "--predict", "0"]));
    assert_eq!(
        unicode["prompt_tokens"],
        serde_json::json!([1, 3, 199, 173])
    );
}

#[test]
fn malformed_and_repeated_arguments_are_errors_without_partial_output() {
    let files = Files::new();
    let source = files.source();
    let invalid = vec![
        vec!["--tokens", "1", "--predict", "0", "--logits", "--logits"],
        vec![
            "--tokens",
            "1",
            "--predict",
            "0",
            "--logits",
            "--logits",
            "ignored",
        ],
        vec!["--tokens", "1", "--predict", "0", "--predict", "0"],
        vec!["--tokens", "1", "--predict", "0", "--model", "ignored"],
        vec!["--tokens", "1", "--predict", "0", "--logits", "true"],
        vec!["--tokens", "1", "--prompt", "A", "--predict", "0"],
        vec!["--tokens", "1", "--predict", "0", "--unknown", "value"],
        vec!["--tokens", "1", "--predict"],
        vec!["--tokens"],
        vec!["--predict", "0"],
        vec!["--tokens", "1", "--predict", "-1"],
        vec!["--tokens", "1", "--predict", "0", "--context", "NaN"],
        vec!["--tokens", "1", "--predict", "0", "--context", "-1"],
    ];
    for args in invalid {
        rejected(&files.run(&source, &args));
    }
    rejected(
        &files
            .command("inspect", &source)
            .arg("--logits")
            .output()
            .unwrap(),
    );
    rejected(&files.command("unknown-command", &source).output().unwrap());
    rejected(
        &Command::new(env!("CARGO_BIN_EXE_bml-model"))
            .arg("inspect")
            .output()
            .unwrap(),
    );
}

#[test]
fn invalid_token_lists_and_context_boundaries_fail_before_generation() {
    let files = Files::new();
    let source = files.source();
    for tokens in [
        "",
        "-1",
        "4294967296",
        "260",
        "1,260",
        "1,",
        ",1",
        "1,,2",
        "NaN",
        "1, 3",
    ] {
        rejected(&files.run(&source, &["--tokens", tokens, "--predict", "0"]));
    }
    for (tokens, predict, context) in [
        ("1", "0", "0"),
        ("1", "0", "9"),
        ("1,3", "0", "1"),
        ("1", "8", "8"),
        ("1", "18446744073709551615", "8"),
        ("1", "0", "18446744073709551615"),
    ] {
        rejected(&files.run(
            &source,
            &[
                "--tokens",
                tokens,
                "--predict",
                predict,
                "--context",
                context,
            ],
        ));
    }
    let boundary = json(&files.run(
        &source,
        &["--tokens", "1", "--predict", "1", "--context", "2"],
    ));
    assert_eq!(boundary["tokens"].as_array().unwrap().len(), 1);
}

#[test]
fn export_and_compile_never_overwrite_an_existing_output() {
    let files = Files::new();
    let source = files.source();
    let destination = files.path("existing file");
    let sentinel = b"an existing unrelated file must remain intact";
    std::fs::write(&destination, sentinel).unwrap();
    for command in ["export", "compile"] {
        let output = files
            .command(command, &source)
            .arg("--output")
            .arg(&destination)
            .output()
            .unwrap();
        rejected(&output);
        assert_eq!(std::fs::read(&destination).unwrap(), sentinel);
    }
    let source_before = std::fs::read(&source).unwrap();
    rejected(
        &files
            .command("export", &source)
            .arg("--output")
            .arg(&source)
            .output()
            .unwrap(),
    );
    assert_eq!(std::fs::read(&source).unwrap(), source_before);
}

#[test]
fn invalid_models_leave_no_output_artifact_or_success_json() {
    let files = Files::new();
    for (index, contents) in [
        Vec::new(),
        b"GGUF".to_vec(),
        b"not a model".to_vec(),
        fixture()[..100].to_vec(),
        unsupported_architecture_fixture(),
        corrupt_head_count_fixture(),
    ]
    .into_iter()
    .enumerate()
    {
        let source = files.path(&format!("invalid-{index}.gguf"));
        std::fs::write(&source, contents).unwrap();
        for command in ["export", "compile"] {
            let destination = files.path(&format!("must-not-exist-{index}-{command}"));
            rejected(
                &files
                    .command(command, &source)
                    .arg("--output")
                    .arg(&destination)
                    .output()
                    .unwrap(),
            );
            assert!(!destination.exists());
        }
        rejected(&files.command("inspect", &source).output().unwrap());
        rejected(&files.run(&source, &["--tokens", "1", "--predict", "0"]));
    }
}

fn unsupported_architecture_fixture() -> Vec<u8> {
    let mut bytes = fixture();
    let encoded = string("llama");
    let start = bytes
        .windows(encoded.len())
        .position(|window| window == encoded)
        .unwrap();
    bytes[start + 8..start + encoded.len()].copy_from_slice(b"gemma");
    bytes
}

fn replace_u32_metadata(bytes: &mut [u8], name: &str, old: u32, new: u32) {
    let mut encoded = string(name);
    encoded.extend(4_u32.to_le_bytes());
    encoded.extend(old.to_le_bytes());
    let start = bytes
        .windows(encoded.len())
        .position(|window| window == encoded)
        .unwrap();
    bytes[start + encoded.len() - 4..start + encoded.len()].copy_from_slice(&new.to_le_bytes());
}

fn corrupt_head_count_fixture() -> Vec<u8> {
    let mut bytes = fixture();
    replace_u32_metadata(&mut bytes, "llama.attention.head_count", 4, 0);
    bytes
}

#[test]
fn diagnostic_logit_budget_is_checked_before_expensive_generation() {
    let files = Files::new();
    let source = files.path("large-context.gguf");
    let mut bytes = fixture();
    replace_u32_metadata(&mut bytes, "llama.context_length", 8, 70_000);
    std::fs::write(&source, bytes).unwrap();
    // 65,537 x 260 x 4 bytes exceeds the 64 MiB diagnostic limit while the
    // tiny model's KV allocation still fits the ordinary session budget.
    let mut child = files
        .command("run", &source)
        .args([
            "--tokens",
            "1",
            "--predict",
            "65536",
            "--context",
            "70000",
            "--logits",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            let output = child.wait_with_output().unwrap();
            let error = rejected(&output);
            assert!(
                error.contains("logits") || error.contains("diagnostic"),
                "{error}"
            );
            break;
        }
        if start.elapsed() > Duration::from_secs(4) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("CLI began generating instead of rejecting oversized diagnostic output");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_arguments_are_rejected_without_panicking() {
    use std::os::unix::ffi::OsStringExt;
    let invalid = std::ffi::OsString::from_vec(vec![0xff]);
    let output = Command::new(env!("CARGO_BIN_EXE_bml-model"))
        .args(["inspect", "--model"])
        .arg(invalid)
        .output()
        .unwrap();
    rejected(&output);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
mod native_orchestration {
    use super::*;
    use bml_inference::resources::Artifact;
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::PermissionsExt;

    // This subprocess substitutes the Cargo protocol only. It copies this
    // already-built test ELF; native lowering itself has a separate real gate.
    const STUB: &str = r#"#!/usr/bin/env python3
import json, os, pathlib, shutil, sys
phase = sys.argv[1]
manifest = pathlib.Path(sys.argv[sys.argv.index('--manifest-path') + 1])
name = json.loads(next(line.split('=', 1)[1].strip() for line in manifest.read_text().splitlines() if line.startswith('name = ')))
target = pathlib.Path(os.environ['CARGO_TARGET_DIR']) / 'release' / name
record = {'arguments': sys.argv[1:], 'manifest': str(manifest), 'launcher': str(target), 'rustflags': os.environ.get('RUSTFLAGS'), 'encoded_rustflags': os.environ.get('CARGO_ENCODED_RUSTFLAGS')}
with open(os.environ['BML_STUB_RECORD'], 'a') as output:
    output.write(json.dumps(record) + '\n')
assert phase in ('metadata', 'build')
if os.environ.get('BML_STUB_FAIL') == phase:
    sys.exit(17)
if phase == 'build' and os.environ.get('BML_STUB_FAIL') != 'missing-launcher':
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.environ['BML_STUB_LAUNCHER'], target)
"#;

    struct CompilerStub {
        executable: PathBuf,
        record: PathBuf,
    }

    impl CompilerStub {
        fn new(files: &Files) -> Self {
            let executable = files.path("cargo protocol stub.py");
            std::fs::write(&executable, STUB).unwrap();
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
            Self {
                executable,
                record: files.path("cargo-invocations.jsonl"),
            }
        }

        fn command(&self, files: &Files, source: &Path, output: &Path) -> Command {
            let mut command = files.command("compile", source);
            command
                .arg("--output")
                .arg(output)
                .env("CARGO", &self.executable)
                .env("BML_STUB_RECORD", &self.record)
                .env("BML_STUB_LAUNCHER", std::env::current_exe().unwrap());
            command
        }

        fn records(&self) -> Vec<Value> {
            std::fs::read_to_string(&self.record)
                .unwrap_or_default()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        }
    }

    impl Drop for CompilerStub {
        fn drop(&mut self) {
            // Generated build names are unique; remove only paths recorded by
            // this test's subprocess, including when an assertion fails.
            for record in self.records() {
                if let Some(launcher) = record["launcher"].as_str() {
                    let _ = std::fs::remove_file(launcher);
                }
                if let Some(manifest) = record["manifest"].as_str() {
                    if let Some(build) = Path::new(manifest).parent() {
                        let _ = std::fs::remove_dir_all(build);
                    }
                }
            }
        }
    }

    fn hash_file(path: &Path) -> String {
        format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
    }

    #[test]
    fn native_build_orchestration_binds_source_resources_lock_and_target() {
        let files = Files::new();
        let source = files.source();
        let output = files.path("compiled model");
        let stub = CompilerStub::new(&files);
        let report = json(&stub.command(&files, &source, &output).output().unwrap());
        let records = stub.records();
        assert_eq!(records.len(), 2);
        let manifest_path = records[0]["manifest"].as_str().unwrap();
        assert_eq!(
            records[0]["arguments"],
            serde_json::json!([
                "metadata",
                "--offline",
                "--format-version",
                "1",
                "--manifest-path",
                manifest_path
            ])
        );
        assert_eq!(
            records[1]["arguments"],
            serde_json::json!([
                "build",
                "--release",
                "--locked",
                "--offline",
                "--manifest-path",
                manifest_path
            ])
        );
        for invocation in &records {
            assert_eq!(invocation["rustflags"], "");
            assert_eq!(invocation["encoded_rustflags"], "-Ctarget-cpu=x86-64");
        }
        assert_eq!(report["output"], output.to_str().unwrap());
        assert_eq!(report["sha256"], hash_file(&output));
        assert_eq!(report["source_sha256"], hash_file(&source));
        assert_eq!(report["bytes"], std::fs::metadata(&output).unwrap().len());
        assert_eq!(report["target"], "x86_64-unknown-linux-gnu");
        assert_eq!(report["cpu"], "x86-64");
        assert_eq!(
            report["build_flags"],
            serde_json::json!(["--release", "--locked", "--offline", "-Ctarget-cpu=x86-64"])
        );
        assert!(report["rustc_version"]
            .as_str()
            .unwrap()
            .starts_with("rustc "));
        assert!(report["elapsed_ms"].as_f64().unwrap() > 0.0);
        let build = Path::new(manifest_path).parent().unwrap();
        assert_eq!(report["build_directory"], build.to_str().unwrap());
        assert_eq!(
            report["generated_source_sha256"],
            hash_file(&build.join("src/main.rs"))
        );
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        assert_eq!(
            report["generated_lock_sha256"],
            hash_file(&workspace.join("Cargo.lock"))
        );
        assert_eq!(
            std::fs::read(build.join("rust-toolchain.toml")).unwrap(),
            std::fs::read(workspace.join("rust-toolchain.toml")).unwrap()
        );
        let artifact = Artifact::open(&output).unwrap();
        let model = artifact.model().unwrap();
        assert_eq!(model.config().block_count, 1);
        assert_eq!(model.config().head_count_kv, 2);
        let generated = std::fs::read_to_string(build.join("src/main.rs")).unwrap();
        assert!(generated.contains(&format!("{:?}", artifact.resource_sha256())));
        assert!(generated.contains("session.begin_token(token)?"));
        assert!(generated.contains("let step = step.layer(0)?;"));
        assert!(!generated.contains("step.layer(1)"));
        assert!(generated.contains("step.finish()"));
        for operation in model.plan() {
            assert!(generated.contains(&format!("Operation::{operation:?}")));
        }
        let expected = 208.0_f64 / (6.5_f64 + 1e-5).sqrt();
        for logit in model.session(2).unwrap().decode(69).unwrap() {
            assert!((f64::from(logit) - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn failed_compiler_phases_and_missing_launcher_leave_no_output() {
        for failure in ["metadata", "build", "missing-launcher"] {
            let files = Files::new();
            let source = files.source();
            let output = files.path("never-created");
            let stub = CompilerStub::new(&files);
            let result = stub
                .command(&files, &source, &output)
                .env("BML_STUB_FAIL", failure)
                .output()
                .unwrap();
            let message = rejected(&result);
            assert!(!output.exists());
            if failure != "missing-launcher" {
                assert!(message.contains("native compilation failed"));
            }
            assert_eq!(
                stub.records().len(),
                if failure == "metadata" { 1 } else { 2 }
            );
        }
    }
}
