use bml_runtime::{
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
    symbolic_transport::{SymbolicLayerServer, SymbolicTransportLimits},
};
use std::{
    io::{BufRead, BufReader},
    net::{SocketAddr, TcpListener},
    path::Path,
    process::{Child, Command, Stdio},
    sync::Arc,
    thread,
};
#[allow(dead_code)]
#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;
struct Directory(std::path::PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
enum Server {
    Thread(Option<thread::JoinHandle<()>>),
    Process(Child),
}
impl Drop for Server {
    fn drop(&mut self) {
        match self {
            Server::Thread(task) => {
                if let Some(task) = task.take() {
                    let _ = task.join();
                }
            }
            Server::Process(child) => {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}
fn server(path: &Path, source: &Path, start: u32, end: u32) -> (Server, SocketAddr) {
    server_with_limits(path, source, start, end, 6, 4096)
}
fn server_with_limits(
    path: &Path,
    source: &Path,
    start: u32,
    end: u32,
    context: u32,
    kv_bytes: usize,
) -> (Server, SocketAddr) {
    if let Some(binary) = std::env::var_os("BML_WORKER_BIN") {
        let mut child = Command::new(binary)
            .args(["--port", "0", "--symbolic-artifact"])
            .arg(path)
            .arg("--symbolic-source")
            .arg(source)
            .args([
                "--layer-start",
                &start.to_string(),
                "--layer-end",
                &end.to_string(),
                "--symbolic-context",
                &context.to_string(),
                "--symbolic-kv-bytes",
                &kv_bytes.to_string(),
                "--symbolic-large-budget",
                "--symbolic-timeout-seconds",
                "600",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        assert!(line.starts_with("symbolic worker listening on "), "{line}");
        let address: SocketAddr = line.split_whitespace().nth(4).unwrap().parse().unwrap();
        return (
            Server::Process(child),
            SocketAddr::from(([127, 0, 0, 1], address.port())),
        );
    }
    let service = Arc::new(
        SymbolicLayerServer::new(
            path,
            Some(source),
            start..end,
            SymbolicTransportLimits::default(),
        )
        .unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let task = thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && std::time::Instant::now() < deadline =>
                {
                    thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(error) => panic!("symbolic test server: {error}"),
            }
        };
        service.serve_connection(stream).unwrap();
    });
    (Server::Thread(Some(task)), address)
}
#[test]
fn cli_generates_from_persisted_model_over_two_worker_ranges_with_relocated_weights() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("cli-distributed"));
    std::fs::create_dir(&directory.0).unwrap();
    let compiled = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args([
            "compile",
            "--distributed",
            "--context-size",
            "6",
            "--distributed-rms-input-bound",
            "4",
            "--distributed-rms-weight-bound",
            "4",
            "--distributed-projection-input-bound",
            "4",
            "--distributed-projection-weight-bound",
            "4",
            "--model",
        ])
        .arg(&source.0)
        .arg("--output")
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let manifest = bml_compiler::symbolic_artifact::load_directory(&directory.0, None)
        .unwrap()
        .manifest;
    let rms = manifest.bounded_profile.unwrap().rms;
    assert_eq!((rms.input, rms.weight), (4.0, 4.0));
    let bounds = manifest.bounded_profile.unwrap();
    assert_eq!(
        (bounds.projection_input, bounds.projection_weight),
        (4.0, 4.0)
    );
    let moved = symbolic_fixture::Fixture(source.0.with_extension("moved.gguf"));
    std::fs::rename(&source.0, &moved.0).unwrap();
    let mut reference = SymbolicCausalModel::open_artifact(
        &directory.0,
        Some(&moved.0),
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let native = bml_compiler::gguf_compiler::InferenceCompiler::open_mapped(&moved.0).unwrap();
    let vocab = reference.compiler().vocab().clone();
    let ids = vocab.encode("A").unwrap();
    let mut cache = bml_compiler::HashConsedKV::new(3, 1, 2, 6);
    let mut native_cache = bml_compiler::HashConsedKV::new(3, 1, 2, 6);
    let mut logits = reference.forward_cached(&ids, &mut cache).unwrap();
    let mut native_logits = native.forward_cached(&ids, &mut native_cache).unwrap();
    let mut expected = Vec::new();
    for step in 0..2 {
        assert!(logits
            .iter()
            .zip(&native_logits)
            .all(|(a, b)| (a - b).abs() < 2e-7));
        let token = bml_compiler::sampler::sample(&logits, 0.0, step).unwrap();
        assert_eq!(
            Some(token),
            bml_compiler::sampler::sample(&native_logits, 0.0, step)
        );
        assert_ne!(token, vocab.eos_token_id);
        expected.extend(vocab.token_piece_bytes(token).unwrap());
        if step == 0 {
            logits = reference.forward_cached(&[token], &mut cache).unwrap();
            native_logits = native.forward_cached(&[token], &mut native_cache).unwrap();
        }
    }
    assert!(!expected.is_empty());
    for ranges in [
        std::iter::once(0..3).collect(),
        vec![0..1, 1..3],
        vec![0..1, 1..2, 2..3],
    ] {
        let mut workers = Vec::new();
        let mut addresses = Vec::new();
        for range in &ranges {
            let (process, address) = server(&directory.0, &moved.0, range.start, range.end);
            workers.push(process);
            addresses.push(address.to_string());
        }
        let idle = if ranges.len() == 3 {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            addresses.push(listener.local_addr().unwrap().to_string());
            Some(listener)
        } else {
            None
        };
        let output = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["distribute", "--model"])
            .arg(&directory.0)
            .arg("--weights")
            .arg(&moved.0)
            .args([
                "--nodes",
                &addresses.join(","),
                "--prompt",
                "A",
                "--num-tokens",
                "2",
                "--temp",
                "0",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected);
        if let Some(idle) = idle {
            assert_eq!(
                idle.accept().unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
        }
        drop(workers);
    }
}
#[test]
fn cli_rejects_missing_symbolic_artifact_before_using_legacy_fragments() {
    let root = std::env::temp_dir().join(format!("bml-missing-symbolic-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let _cleanup = Directory(root.clone());
    let legacy = bml_compiler::distributed::DistributedFragment {
        fragment_id: 0,
        layer_start: 0,
        layer_end: 1,
        ops: vec![bml_compiler::RpnOp::Var(0)],
        weights: vec![],
        tensors: vec![],
        config: bml_compiler::gguf_compiler::ModelConfig {
            architecture: "llama".into(),
            n_layers: 1,
            n_heads: 1,
            n_embd: 2,
            context_length: 4,
            vocab_size: 260,
        },
        n_kv_heads: 1,
        head_dim: 2,
    };
    bml_compiler::distributed::serialize_distributed(&[legacy], &root).unwrap();
    assert!(root.join("header.bmlgraph").is_file());
    let output = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["distribute", "--model"])
        .arg(&root)
        .args([
            "--nodes",
            "127.0.0.1:1",
            "--prompt",
            "A",
            "--num-tokens",
            "1",
            "--temp",
            "0",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("fragmentos distribuidos históricos están incompletos"));
}

#[test]
fn cli_stops_at_eos_and_emits_partial_utf8_bytes_without_replacement() {
    for (token, expected) in [(2usize, Vec::new()), (199usize, vec![0xc3])] {
        let source = symbolic_fixture::fixture_layers_with_output(0.25, 3, token);
        let directory = Directory(source.0.with_extension("cli-eos-bytes"));
        std::fs::create_dir(&directory.0).unwrap();
        let compiled = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["compile", "--distributed", "--context-size", "6", "--model"])
            .arg(&source.0)
            .arg("--output")
            .arg(&directory.0)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let (_worker, address) = server(&directory.0, &source.0, 0, 3);
        let output = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["distribute", "--model"])
            .arg(&directory.0)
            .args([
                "--nodes",
                &address.to_string(),
                "--prompt",
                "A",
                "--num-tokens",
                "1",
                "--temp",
                "0",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected);
    }
}

#[test]
fn distributed_compile_rejects_partial_or_invalid_rms_bounds() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("bad-rms-profile"));
    for args in [
        vec!["--distributed-rms-input-bound", "4"],
        vec!["--distributed-projection-weight-bound", "4"],
        vec![
            "--distributed-projection-input-bound",
            "128",
            "--distributed-projection-weight-bound",
            "4",
        ],
        vec![
            "--distributed-rms-input-bound",
            "4",
            "--distributed-rms-weight-bound",
            "0",
        ],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["compile", "--distributed", "--context-size", "6", "--model"])
            .arg(&source.0)
            .arg("--output")
            .arg(&directory.0)
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(!directory
            .0
            .join(bml_compiler::symbolic_artifact::MANIFEST)
            .exists());
    }
}

/// Exact opt-in real-model integration; never selected by ordinary tests.
#[test]
#[ignore = "requires BML_TEST_GGUF, BML_TEST_SYMBOLIC_ARTIFACT and a release BML_WORKER_BIN"]
fn real_stories15m_distributed_token_matches_native_reference() {
    use bml_runtime::graph::GraphLimits;
    use bml_runtime::symbolic_transport::SymbolicLayerClient;
    let source =
        std::path::PathBuf::from(std::env::var_os("BML_TEST_GGUF").expect("BML_TEST_GGUF"));
    let directory = std::path::PathBuf::from(
        std::env::var_os("BML_TEST_SYMBOLIC_ARTIFACT").expect("BML_TEST_SYMBOLIC_ARTIFACT"),
    );
    assert!(
        std::env::var_os("BML_WORKER_BIN").is_some(),
        "release BML_WORKER_BIN required"
    );
    let reference = bml_compiler::gguf_compiler::InferenceCompiler::open_mapped(&source).unwrap();
    assert_eq!(reference.config().n_layers, 6);
    let ids = reference.vocab().encode("").unwrap();
    assert_eq!(ids, [1], "empty prompt must denote the single BOS token");
    let native = reference.forward(&ids).unwrap();
    let expected_token = bml_compiler::sampler::sample(&native, 0.0, 0).unwrap();
    let identity =
        bml_compiler::symbolic_artifact::checked_identity(&directory, Some(&source)).unwrap();
    let (_first, a) = server_with_limits(&directory, &source, 0, 3, 2, 1024 * 1024);
    let (_second, b) = server_with_limits(&directory, &source, 3, 6, 2, 1024 * 1024);
    let transport = SymbolicTransportLimits {
        timeout: std::time::Duration::from_secs(600),
        ..Default::default()
    };
    let first = SymbolicLayerClient::connect(a, &identity, 0..3, 2, transport).unwrap();
    let second = SymbolicLayerClient::connect(b, &identity, 3..6, 2, transport).unwrap();
    let runtime_limits = SymbolicLimits {
        max_nodes: 0,
        max_cached_ops: 4_000_000,
        max_programs: 4096,
        graph: GraphLimits {
            max_ops: 2_000_000,
            max_steps: 4_000_000,
            max_values: 4_000_000,
        },
    };
    let mut pipeline = bml_runtime::symbolic_transport::SymbolicPipeline::open(
        &directory,
        Some(&source),
        2,
        vec![first, second],
        runtime_limits,
    )
    .unwrap();
    let start = std::time::Instant::now();
    let actual = pipeline.forward(ids[0]).unwrap();
    assert_eq!(actual.len(), 32000);
    assert_eq!(actual.len(), native.len());
    assert!(actual.iter().all(|v| v.is_finite()));
    let token = bml_compiler::sampler::sample(&actual, 0.0, 0).unwrap();
    assert_eq!(token, expected_token);
    let max_error = actual
        .iter()
        .zip(&native)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let rms_error = (actual
        .iter()
        .zip(&native)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / actual.len() as f64)
        .sqrt();
    assert!(max_error < 1e-4, "maximum error {max_error}");
    assert_eq!(pipeline.position(), 1);
    pipeline.close().unwrap();
    println!("stories15m input_ids={ids:?} logits={} native_argmax={expected_token} symbolic_argmax={token} max_abs_error={max_error} rms_error={rms_error} elapsed_seconds={}", actual.len(), start.elapsed().as_secs_f64());
}

#[test]
fn cli_exports_trusted_shards_for_source_free_workers_and_generates() {
    use bml_compiler::{
        symbolic_artifact::{checked_identity, compile_directory_bounded},
        symbolic_bounded::BoundedProfile,
        symbolic_kernels::KernelLimits,
    };
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("cli-shard-export"));
    std::fs::create_dir(&directory.0).unwrap();
    let artifact = directory.0.join("artifact");
    std::fs::create_dir(&artifact).unwrap();
    compile_directory_bounded(
        &source.0,
        &artifact,
        6,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&artifact, None).unwrap();
    let mut addresses = Vec::new();
    let mut tasks = Vec::new();
    for range in [0..1, 1..3] {
        let shard = directory.0.join(format!("shard-{}", range.start));
        let output = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["export-shard", "--model"])
            .arg(&artifact)
            .arg("--output")
            .arg(&shard)
            .args([
                "--layer-start",
                &range.start.to_string(),
                "--layer-end",
                &range.end.to_string(),
                "--max-bytes",
                "1000000",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        let get = |key: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(key))
                .unwrap()
                .to_owned()
        };
        assert_eq!(get("artifact_sha256="), identity);
        let hash = get("shard_sha256=");
        let server = Arc::new(
            SymbolicLayerServer::new_authenticated(
                &artifact,
                &identity,
                range,
                SymbolicTransportLimits::default(),
                &shard,
                &hash,
                1_000_000,
            )
            .unwrap(),
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        addresses.push(listener.local_addr().unwrap().to_string());
        tasks.push(thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            server.serve_connection(stream).unwrap();
        }));
    }
    let mut reference = SymbolicCausalModel::open_artifact(
        &artifact,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let vocab = reference.compiler().vocab().clone();
    let mut cache = bml_compiler::HashConsedKV::new(3, 1, 2, 6);
    let mut logits = reference
        .forward_cached(&vocab.encode("A").unwrap(), &mut cache)
        .unwrap();
    let mut expected = Vec::new();
    for step in 0..2 {
        let token = bml_compiler::sampler::sample(&logits, 0.0, step).unwrap();
        assert_ne!(token, vocab.eos_token_id);
        expected.extend(vocab.token_piece_bytes(token).unwrap());
        if step == 0 {
            logits = reference.forward_cached(&[token], &mut cache).unwrap();
        }
    }
    let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["distribute", "--model"])
        .arg(&artifact)
        .args([
            "--nodes",
            &addresses.join(","),
            "--prompt",
            "A",
            "--num-tokens",
            "2",
            "--temp",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.stdout, expected);
    for task in tasks {
        task.join().unwrap();
    }
    let absent = directory.0.join("budget-error");
    let rejected = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["export-shard", "--model"])
        .arg(&artifact)
        .arg("--output")
        .arg(&absent)
        .args(["--layer-start", "0", "--layer-end", "1", "--max-bytes", "1"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(!absent.exists());
    let legacy = directory.0.join("legacy-symbolic");
    std::fs::create_dir(&legacy).unwrap();
    bml_compiler::symbolic_artifact::compile_directory(
        &source.0,
        &legacy,
        2,
        KernelLimits::default(),
    )
    .unwrap();
    let unsupported = directory.0.join("unsupported-profile");
    let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["export-shard", "--model"])
        .arg(&legacy)
        .arg("--output")
        .arg(&unsupported)
        .args(["--layer-start", "0", "--layer-end", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("bounded symbolic v4"));
    assert!(!unsupported.exists());
}

#[test]
fn symbolic_step_budget_is_explicit_and_enforced() {
    use bml_compiler::{
        symbolic_artifact::compile_directory_bounded, symbolic_bounded::BoundedProfile,
        symbolic_kernels::KernelLimits,
    };
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("cli-step-budget"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let run = |steps: &str, large: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_bml-cli"));
        command.args(["run", "--model"]).arg(&directory.0).args([
            "--prompt",
            "A",
            "--num-tokens",
            "1",
            "--context-size",
            "3",
            "--temp",
            "0",
            "--symbolic-max-steps",
            steps,
        ]);
        if large {
            command.arg("--symbolic-large-budget");
        }
        command.output().unwrap()
    };
    let rejected = run("1", true);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("resource limit"));
    let missing_flag = run("1000000", false);
    assert!(!missing_flag.status.success());
    let accepted = run("1000000", true);
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stderr)
    );
}

#[test]
fn cli_prepares_source_free_shards_with_coordinator_partition() {
    use bml_compiler::{
        symbolic_artifact::{
            checked_identity, compile_directory_bounded, load_directory_authenticated,
        },
        symbolic_bounded::BoundedProfile,
        symbolic_kernels::KernelLimits,
        symbolic_weight_shard::verify_layer_weights,
    };
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let root = Directory(source.0.with_extension("cli-prepare-shards"));
    std::fs::create_dir(&root.0).unwrap();
    let artifact = root.0.join("model");
    std::fs::create_dir(&artifact).unwrap();
    compile_directory_bounded(
        &source.0,
        &artifact,
        6,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&artifact, None).unwrap();
    let prepared = root.0.join("prepared");
    let run = |output: &std::path::Path, workers: &str, budget: &str| {
        Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["prepare-shards", "--model"])
            .arg(&artifact)
            .arg("--output")
            .arg(output)
            .args(["--workers", workers, "--max-bytes", budget])
            .output()
            .unwrap()
    };
    let result = run(&prepared, "2", "1000000");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let plan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(prepared.join("plan.json")).unwrap()).unwrap();
    assert_eq!(plan["artifact_sha256"], identity);
    let copy = prepared.join(plan["artifact"].as_str().unwrap());
    load_directory_authenticated(&copy, &identity).unwrap();
    let workers = plan["workers"].as_array().unwrap();
    assert_eq!(workers.len(), 2);
    for (entry, range) in workers.iter().zip([0..1, 1..3]) {
        assert_eq!(entry["layer_start"], range.start);
        assert_eq!(entry["layer_end"], range.end);
        let dir = prepared.join(entry["directory"].as_str().unwrap());
        verify_layer_weights(
            &dir,
            entry["shard_sha256"].as_str().unwrap(),
            &identity,
            range,
            1_000_000,
        )
        .unwrap();
    }
    let mut addresses = Vec::new();
    let mut tasks = Vec::new();
    for entry in workers {
        let range = entry["layer_start"].as_u64().unwrap() as u32
            ..entry["layer_end"].as_u64().unwrap() as u32;
        let shard = prepared.join(entry["directory"].as_str().unwrap());
        let server = Arc::new(
            SymbolicLayerServer::new_authenticated(
                &copy,
                &identity,
                range,
                SymbolicTransportLimits::default(),
                &shard,
                entry["shard_sha256"].as_str().unwrap(),
                1_000_000,
            )
            .unwrap(),
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        addresses.push(listener.local_addr().unwrap().to_string());
        tasks.push(thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            server.serve_connection(stream).unwrap();
        }));
    }
    let mut reference = SymbolicCausalModel::open_artifact(
        &artifact,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let vocab = reference.compiler().vocab().clone();
    let logits = reference.forward(&vocab.encode("A").unwrap()).unwrap();
    let token = bml_compiler::sampler::sample(&logits, 0.0, 0).unwrap();
    assert_ne!(token, vocab.eos_token_id);
    let expected = vocab.token_piece_bytes(token).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["distribute", "--model"])
        .arg(&artifact)
        .args([
            "--nodes",
            &addresses.join(","),
            "--prompt",
            "A",
            "--num-tokens",
            "1",
            "--temp",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.stdout, expected);
    for task in tasks {
        task.join().unwrap();
    }
    let existing = run(&prepared, "2", "1000000");
    assert!(!existing.status.success());
    assert!(prepared.join("plan.json").exists());
    let insufficient = root.0.join("insufficient");
    assert!(!run(&insufficient, "2", "1").status.success());
    assert!(!insufficient.exists());
    let extra = root.0.join("extra");
    assert!(run(&extra, "8", "1000000").status.success());
    let plan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(extra.join("plan.json")).unwrap()).unwrap();
    let workers = plan["workers"].as_array().unwrap();
    assert_eq!(workers.len(), 3);
    for (index, entry) in workers.iter().enumerate() {
        assert_eq!(entry["layer_start"], index);
        assert_eq!(entry["layer_end"], index + 1);
    }
}
