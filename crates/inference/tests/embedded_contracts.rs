//! Exercise the native entrypoint with resources appended to a real test ELF.

#![cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]

use bml_gguf::GgufReader;
use bml_inference::driver::{embedded_main, run_embedded_with_arguments, NativeToken};
use bml_inference::model::{Operation, Session};
use bml_inference::resources::{append_to_executable, write_model_file, Artifact};
use bml_inference::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const CHILD_REQUEST: &str = "BML_EMBEDDED_TEST_REQUEST";
const OUTPUT_BEGIN: &str = "BML_EMBEDDED_OUTPUT_BEGIN\n";
const RESULT_BEGIN: &str = "BML_EMBEDDED_RESULT:";
static NEXT: AtomicU64 = AtomicU64::new(0);
static NATIVE_CALLS: AtomicUsize = AtomicUsize::new(0);

struct Files(PathBuf);

impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "bml_embedded_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn artifact(&self) -> (PathBuf, ChildRequest) {
        let gguf = self.path("model.gguf");
        std::fs::write(&gguf, fixture()).unwrap();
        let reader = GgufReader::open(gguf).unwrap();
        let resources = self.path("model.bmlres");
        write_model_file(&reader, &resources).unwrap();
        let model = Artifact::open(&resources).unwrap();
        let request = ChildRequest {
            resource_sha256: model.resource_sha256(),
            plan: model.model().unwrap().plan().to_vec(),
            arguments: arguments(&[
                "--prompt",
                "ab",
                "--predict",
                "2",
                "--context",
                "8",
                "--logits",
            ]),
            fail_native: false,
        };
        let executable = self.path("embedded-tests");
        append_to_executable(std::env::current_exe().unwrap(), resources, &executable).unwrap();
        (executable, request)
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Serialize, Deserialize)]
struct ChildRequest {
    resource_sha256: String,
    plan: Vec<Operation>,
    arguments: Vec<String>,
    fail_native: bool,
}

struct ChildOutput {
    output: Option<Value>,
    error: Option<String>,
    native_calls: usize,
}

fn arguments(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn native_token(session: &mut Session<'_>, token: u32) -> Result<Vec<f32>> {
    NATIVE_CALLS.fetch_add(1, Ordering::Relaxed);
    session.begin_token(token)?.layer(0)?.finish()
}

fn failing_native(_: &mut Session<'_>, _: u32) -> Result<Vec<f32>> {
    NATIVE_CALLS.fetch_add(1, Ordering::Relaxed);
    Err(Error("native evaluator rejected the token".into()))
}

#[test]
#[ignore = "child helper; launched only by the exact ordinary embedded tests"]
fn embedded_child() {
    let request: ChildRequest =
        serde_json::from_str(&std::env::var(CHILD_REQUEST).expect("parent request")).unwrap();
    NATIVE_CALLS.store(0, Ordering::Relaxed);
    let evaluator: NativeToken = if request.fail_native {
        failing_native
    } else {
        native_token
    };
    println!("\n{OUTPUT_BEGIN}");
    let result = run_embedded_with_arguments(
        &request.resource_sha256,
        &request.plan,
        evaluator,
        &request.arguments,
    );
    println!(
        "{RESULT_BEGIN}{}",
        serde_json::json!({
            "error": result.err().map(|error| error.to_string()),
            "native_calls": NATIVE_CALLS.load(Ordering::Relaxed),
        })
    );
}

#[test]
#[ignore = "child helper; verifies the process entrypoint's invalid-argument exit"]
fn embedded_entrypoint_child() {
    let request: ChildRequest =
        serde_json::from_str(&std::env::var(CHILD_REQUEST).expect("parent request")).unwrap();
    // The test harness flags are deliberately invalid model-runner arguments.
    // The public entrypoint must report this error and terminate the process.
    embedded_main(&request.resource_sha256, &request.plan, native_token);
}

fn child(executable: &Path, request: &ChildRequest) -> ChildOutput {
    let result = Command::new(executable)
        .args(["--ignored", "--exact", "embedded_child", "--nocapture"])
        .env(CHILD_REQUEST, serde_json::to_string(request).unwrap())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "child failed: stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    let body = stdout.split_once(OUTPUT_BEGIN).unwrap().1;
    let (output, result) = body.split_once(RESULT_BEGIN).unwrap();
    let result: Value = serde_json::from_str(result.lines().next().unwrap()).unwrap();
    ChildOutput {
        output: if output.trim().is_empty() {
            None
        } else {
            Some(serde_json::from_str(output.trim()).unwrap())
        },
        error: result["error"].as_str().map(str::to_owned),
        native_calls: result["native_calls"].as_u64().unwrap() as usize,
    }
}

fn expect_failure(executable: &Path, request: &ChildRequest, message: &str) {
    let result = child(executable, request);
    assert!(
        result
            .error
            .as_deref()
            .is_some_and(|error| error.contains(message)),
        "expected {message:?}, got {:?}",
        result.error
    );
    assert!(result.output.is_none());
    assert_eq!(result.native_calls, 0, "validation must precede evaluation");
}

#[test]
fn embedded_native_output_matches_interpreted_prefill_and_decode() {
    let files = Files::new();
    let (executable, request) = files.artifact();
    let artifact = Artifact::open(files.path("model.bmlres")).unwrap();
    let model = artifact.model().unwrap();
    let prompt = model.tokenizer().encode("ab", true).unwrap();
    assert_eq!(prompt, [1, 260, 261]);

    // Check the explicit native layer sequence against the interpreter before
    // invoking the copied ELF. This fixture has nonzero attention and MLPs.
    let mut interpreted = model.session(8).unwrap();
    let mut native = model.session(8).unwrap();
    for &token in &prompt {
        assert_eq!(
            interpreted.decode(token).unwrap(),
            native_token(&mut native, token).unwrap()
        );
    }
    let mut session = model.session(8).unwrap();
    let mut expected_logits = vec![session.prefill(&prompt).unwrap()];
    for _ in 0..2 {
        let scores = expected_logits.last().unwrap();
        assert!(scores[261] > 0.0);
        assert!(scores
            .iter()
            .enumerate()
            .all(|(id, &score)| id == 261 || score == 0.0));
        expected_logits.push(session.decode(261).unwrap());
    }

    let result = child(&executable, &request);
    assert!(result.error.is_none());
    assert_eq!(result.native_calls, prompt.len() + 2);
    let output = result.output.unwrap();
    assert_eq!(output["prompt_tokens"], serde_json::json!(prompt));
    assert_eq!(output["tokens"], serde_json::json!([261, 261]));
    assert_eq!(output["bytes"], serde_json::json!([b'b', b'b']));
    assert_eq!(output["text"], "bb");
    assert_eq!(output["eos_id"], 2);
    assert_eq!(output["context"], 8);
    assert_eq!(output["kv_bytes"], model.memory_plan(8).unwrap().kv_bytes);
    let logits: Vec<Vec<f32>> = serde_json::from_value(output["logits"].clone()).unwrap();
    assert_eq!(logits, expected_logits);
    for field in ["prefill_ms", "decode_ms"] {
        assert!(output[field]
            .as_f64()
            .is_some_and(|value| value.is_finite() && value >= 0.0));
    }
}

#[test]
fn embedded_requires_the_compiled_resource_hash_and_operation_plan() {
    let files = Files::new();
    let (executable, mut request) = files.artifact();
    let valid_hash = request.resource_sha256.clone();
    request.resource_sha256 = "0".repeat(64);
    expect_failure(&executable, &request, "resource identity differs");
    request.resource_sha256 = valid_hash;
    request.plan.swap(0, 1);
    expect_failure(&executable, &request, "execution plan differs");
}

#[test]
fn embedded_rejects_corrupted_payload_before_native_execution() {
    let files = Files::new();
    let (executable, request) = files.artifact();
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&executable)
        .unwrap();
    file.seek(SeekFrom::End(-65)).unwrap();
    let mut byte = [0];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 1;
    file.seek(SeekFrom::End(-65)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_all().unwrap();
    drop(file);
    expect_failure(&executable, &request, "checksum mismatch");
}

#[test]
fn embedded_rejects_invalid_arguments_and_context_before_evaluation() {
    let files = Files::new();
    let (executable, mut request) = files.artifact();
    for (values, message) in [
        (
            vec!["--model", "unexpected.gguf"],
            "unknown or repeated argument",
        ),
        (vec!["--prompt"], "missing value"),
        (vec!["--logits", "--logits"], "repeated argument"),
        (
            vec!["--prompt", "a", "--tokens", "1"],
            "either --prompt or --tokens",
        ),
        (vec!["--tokens", "oops"], "invalid --tokens"),
        (
            vec!["--prompt", "ab", "--predict", "6", "--context", "8"],
            "exceeds context",
        ),
        (
            vec!["--prompt", "ab", "--predict", "0", "--context", "9"],
            "context",
        ),
        (
            vec!["--prompt", "ab", "--predict", "0", "--context", "0"],
            "exceeds context",
        ),
        (
            vec!["--prompt", "ab", "--predict", "70000", "--logits"],
            "64 MiB diagnostic limit",
        ),
    ] {
        request.arguments = arguments(&values);
        expect_failure(&executable, &request, message);
    }
}

#[test]
fn embedded_supports_explicit_tokens_and_optional_diagnostics() {
    let files = Files::new();
    let (executable, mut request) = files.artifact();
    request.arguments = arguments(&["--tokens", "1,260", "--predict", "1", "--context", "3"]);
    let result = child(&executable, &request);
    assert!(result.error.is_none());
    assert_eq!(result.native_calls, 3);
    let output = result.output.unwrap();
    assert_eq!(output["prompt_tokens"], serde_json::json!([1, 260]));
    assert_eq!(output["tokens"], serde_json::json!([261]));
    assert!(output.get("logits").is_none());

    request.arguments = arguments(&[
        "--prompt",
        "ab",
        "--predict",
        "0",
        "--context",
        "3",
        "--logits",
    ]);
    let result = child(&executable, &request);
    assert!(result.error.is_none());
    assert_eq!(result.native_calls, 3);
    let output = result.output.unwrap();
    assert_eq!(output["tokens"], serde_json::json!([]));
    assert_eq!(output["text"], "");
    assert_eq!(output["bytes"], serde_json::json!([]));
    assert_eq!(output["logits"].as_array().unwrap().len(), 1);
}

#[test]
fn embedded_propagates_native_evaluation_errors() {
    let files = Files::new();
    let (executable, mut request) = files.artifact();
    request.fail_native = true;
    let result = child(&executable, &request);
    assert_eq!(
        result.error.as_deref(),
        Some("native evaluator rejected the token")
    );
    assert_eq!(result.native_calls, 1);
    assert!(result.output.is_none());
}

#[test]
fn embedded_entrypoint_reports_invalid_process_arguments_and_exits_failure() {
    let files = Files::new();
    let (executable, request) = files.artifact();
    let result = Command::new(executable)
        .args([
            "--ignored",
            "--exact",
            "embedded_entrypoint_child",
            "--nocapture",
        ])
        .env(CHILD_REQUEST, serde_json::to_string(&request).unwrap())
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(
        stderr.contains("bml-model: unknown or repeated argument: --ignored"),
        "unexpected entrypoint diagnostic: {stderr}"
    );
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
    tokens.extend(["▁a".into(), "b".into()]);
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
                    .map(|_| 0.0f32.to_le_bytes().to_vec())
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
                            4..=259 => 6,
                            _ => 1,
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
    ];
    for (name, value) in [
        ("llama.embedding_length", 4u32),
        ("llama.block_count", 1),
        ("llama.attention.head_count", 2),
        ("llama.attention.head_count_kv", 1),
        ("llama.feed_forward_length", 6),
        ("llama.context_length", 8),
        ("tokenizer.ggml.bos_token_id", 1),
        ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.unknown_token_id", 0),
    ] {
        metadata.push((name, 4, value.to_le_bytes().to_vec()));
    }
    let mut tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
        (
            "token_embd.weight",
            vec![4, vocab],
            (0..vocab * 4)
                .map(|i| 0.2 + (i % 13) as f32 * 0.01)
                .collect(),
        ),
        ("output_norm.weight", vec![4], vec![0.8, 1.0, 1.2, 0.9]),
        ("blk.0.attn_norm.weight", vec![4], vec![1.0; 4]),
        ("blk.0.ffn_norm.weight", vec![4], vec![1.0; 4]),
    ];
    for (name, width, rows) in [
        ("blk.0.attn_q.weight", 4u64, 4u64),
        ("blk.0.attn_k.weight", 4, 2),
        ("blk.0.attn_v.weight", 4, 2),
        ("blk.0.attn_output.weight", 4, 4),
        ("blk.0.ffn_gate.weight", 4, 6),
        ("blk.0.ffn_up.weight", 4, 6),
        ("blk.0.ffn_down.weight", 6, 4),
    ] {
        tensors.push((
            name,
            vec![width, rows],
            (0..width * rows)
                .map(|i| (i % 7) as f32 * 0.01 - 0.02)
                .collect(),
        ));
    }
    let mut output = vec![0.0; (vocab * 4) as usize];
    output[261 * 4..262 * 4].fill(0.4);
    tensors.push(("output.weight", vec![4, vocab], output));
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
    for (name, dims, values) in tensors {
        data.resize(data.len().div_ceil(32) * 32, 0);
        bytes.extend(string(name));
        bytes.extend((dims.len() as u32).to_le_bytes());
        for dimension in dims {
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(0u32.to_le_bytes());
        bytes.extend((data.len() as u64).to_le_bytes());
        data.extend(values.into_iter().flat_map(f32::to_le_bytes));
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(data);
    bytes
}
