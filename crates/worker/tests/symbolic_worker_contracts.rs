use bml_compiler::{
    symbolic_artifact::{checked_identity, compile_directory_bounded},
    symbolic_bounded::BoundedProfile,
    symbolic_kernels::KernelLimits,
    HashConsedKV,
};
use bml_runtime::{
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
    symbolic_transport::{SymbolicLayerClient, SymbolicPipeline, SymbolicTransportLimits},
};
use std::{
    io::{BufRead, BufReader},
    net::SocketAddr,
    process::{Child, Command, Stdio},
    time::Duration,
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
struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn worker(path: &std::path::Path, start: u32, end: u32) -> (Worker, SocketAddr) {
    worker_with_shard(path, start, end, None)
}

fn worker_with_shard(
    path: &std::path::Path,
    start: u32,
    end: u32,
    shard: Option<(&std::path::Path, &str, Option<&str>)>,
) -> (Worker, SocketAddr) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_bml-worker"));
    if let Some((dir, hash, identity)) = shard {
        if let Some(identity) = identity {
            command.args(["--symbolic-artifact-sha256", identity]);
        }
        command.arg("--symbolic-shard").arg(dir).args([
            "--symbolic-shard-sha256",
            hash,
            "--symbolic-shard-bytes",
            "536870912",
        ]);
    }
    let mut process = command
        .args(["--port", "0", "--symbolic-artifact"])
        .arg(path)
        .args([
            "--layer-start",
            &start.to_string(),
            "--layer-end",
            &end.to_string(),
            "--symbolic-large-budget",
            "--symbolic-timeout-seconds",
            "600",
            "--symbolic-context",
            "3",
            "--symbolic-kv-bytes",
            "1048576",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(process.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    if !line.starts_with("symbolic worker listening on ") {
        use std::io::Read;
        let mut error = String::new();
        process
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut error)
            .unwrap();
        panic!("unexpected worker startup: {line}; stderr: {error}");
    }
    let bound: SocketAddr = line.split_whitespace().nth(4).unwrap().parse().unwrap();
    (
        Worker(process),
        SocketAddr::from(([127, 0, 0, 1], bound.port())),
    )
}
#[test]
fn real_worker_processes_execute_persisted_bml_ranges_with_independent_sessions() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("worker-artifact"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&directory.0, None).unwrap();
    let shard_a = directory.0.join("shard-a");
    let shard_b = directory.0.join("shard-b");
    let hash_a = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &directory.0,
        None,
        0..1,
        &shard_a,
        1_000_000,
    )
    .unwrap();
    let hash_b = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &directory.0,
        None,
        1..3,
        &shard_b,
        1_000_000,
    )
    .unwrap();
    let (_worker_a, address_a) =
        worker_with_shard(&directory.0, 0, 1, Some((&shard_a, &hash_a, None)));
    let (_worker_b, address_b) =
        worker_with_shard(&directory.0, 1, 3, Some((&shard_b, &hash_b, None)));
    let limits = SymbolicTransportLimits {
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let mut a = SymbolicLayerClient::connect(address_a, &identity, 0..1, 3, limits).unwrap();
    let mut b = SymbolicLayerClient::connect(address_b, &identity, 1..3, 3, limits).unwrap();
    let mut other = SymbolicLayerClient::connect(address_a, &identity, 0..1, 3, limits).unwrap();
    let mut model = SymbolicCausalModel::open_artifact(
        &directory.0,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut reference = SymbolicCausalModel::open_artifact(
        &directory.0,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut cache = HashConsedKV::new(3, 1, 2, 3);
    for (position, token) in [69, 70, 69].into_iter().enumerate() {
        let input = model.compiler().get_embedding(token).unwrap();
        let intermediate = a.prepare(position as u32, &input).unwrap();
        if position == 0 {
            other.prepare(0, &input).unwrap();
            other.commit(0).unwrap();
        } else if position == 1 {
            other.prepare(1, &input).unwrap();
            other.abort(1).unwrap();
        }
        let hidden = b.prepare(position as u32, &intermediate).unwrap();
        let logits = model.project_output(&hidden).unwrap();
        let expected = reference.forward_cached(&[token], &mut cache).unwrap();
        assert_eq!(
            logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        a.commit(position as u32).unwrap();
        b.commit(position as u32).unwrap();
    }
    assert!(a.prepare(3, &[0.0; 4]).is_err());
    other.reset().unwrap();
    other.close().unwrap();
    a.close().unwrap();
    b.close().unwrap();
}

#[test]
fn pipeline_covers_all_layers_on_one_two_or_extra_worker_processes() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("worker-pipeline"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&directory.0, None).unwrap();
    let limits = SymbolicTransportLimits {
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    for ranges in [
        std::iter::once(0..3).collect(),
        vec![0..1, 1..3],
        vec![0..1, 1..2, 2..3],
    ] {
        let mut workers = Vec::new();
        let mut clients = Vec::new();
        for range in &ranges {
            let (process, address) = worker(&directory.0, range.start, range.end);
            clients.push(
                SymbolicLayerClient::connect(address, &identity, range.clone(), 3, limits).unwrap(),
            );
            workers.push(process);
        }
        // An additional available node is deliberately left unassigned.
        if ranges.len() == 3 {
            workers.push(worker(&directory.0, 0, 1).0);
        }
        let mut pipeline = SymbolicPipeline::open(
            &directory.0,
            None,
            3,
            clients,
            SymbolicLimits {
                max_nodes: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut reference = SymbolicCausalModel::open_artifact(
            &directory.0,
            None,
            SymbolicLimits {
                max_nodes: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut cache = HashConsedKV::new(3, 1, 2, 3);
        let encoded = pipeline.model().compiler().vocab().encode("AB").unwrap();
        assert!(!encoded.is_empty());
        assert!(pipeline.forward(u32::MAX).is_err());
        assert!(pipeline.is_healthy());
        for (position, token) in [69, 70, 69].into_iter().enumerate() {
            let actual = pipeline.forward(token).unwrap();
            let expected = reference.forward_cached(&[token], &mut cache).unwrap();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(pipeline.position(), position as u32 + 1);
        }
        assert!(pipeline.forward(69).is_err());
        assert!(pipeline.is_healthy());
        pipeline.reset().unwrap();
        let actual = pipeline.forward(69).unwrap();
        let expected = reference.forward(&[69]).unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        pipeline.close().unwrap();
        assert!(!pipeline.is_healthy());
    }
}

#[test]
fn pipeline_rejects_gaps_and_duplicate_layer_assignments() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("worker-coverage"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&directory.0, None).unwrap();
    let (_process, address) = worker(&directory.0, 0, 1);
    let limits = SymbolicTransportLimits {
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let first = SymbolicLayerClient::connect(address, &identity, 0..1, 3, limits).unwrap();
    assert!(SymbolicPipeline::open(
        &directory.0,
        None,
        3,
        vec![first],
        SymbolicLimits::default()
    )
    .is_err());
    let first = SymbolicLayerClient::connect(address, &identity, 0..1, 3, limits).unwrap();
    let second = SymbolicLayerClient::connect(address, &identity, 0..1, 3, limits).unwrap();
    assert!(SymbolicPipeline::open(
        &directory.0,
        None,
        3,
        vec![first, second],
        SymbolicLimits::default()
    )
    .is_err());
}

#[test]
fn loss_of_a_real_worker_invalidates_the_coordinator_without_advancing_it() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("worker-failure"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&directory.0, None).unwrap();
    let (_worker_a, address_a) = worker(&directory.0, 0, 1);
    let (mut worker_b, address_b) = worker(&directory.0, 1, 3);
    let limits = SymbolicTransportLimits {
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let first = SymbolicLayerClient::connect(address_a, &identity, 0..1, 3, limits).unwrap();
    let second = SymbolicLayerClient::connect(address_b, &identity, 1..3, 3, limits).unwrap();
    let mut pipeline = SymbolicPipeline::open(
        &directory.0,
        None,
        3,
        vec![first, second],
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    worker_b.0.kill().unwrap();
    worker_b.0.wait().unwrap();
    assert!(pipeline.forward(69).is_err());
    assert_eq!(pipeline.position(), 0);
    assert!(!pipeline.is_healthy());
    assert!(pipeline.forward(69).is_err());
    assert!(pipeline.reset().is_err());
}

#[test]
fn pipeline_accepts_only_fresh_unprepared_worker_sessions() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("worker-nonfresh"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&directory.0, None).unwrap();
    let (_process, address) = worker(&directory.0, 0, 3);
    let limits = SymbolicTransportLimits {
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let model = SymbolicCausalModel::open_artifact(
        &directory.0,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let input = model.compiler().get_embedding(69).unwrap();
    let connect = || SymbolicLayerClient::connect(address, &identity, 0..3, 3, limits).unwrap();
    let mut prepared = connect();
    prepared.prepare(0, &input).unwrap();
    assert!(prepared.is_prepared());
    assert!(SymbolicPipeline::open(
        &directory.0,
        None,
        3,
        vec![prepared],
        SymbolicLimits::default()
    )
    .is_err());
    let mut advanced = connect();
    advanced.prepare(0, &input).unwrap();
    advanced.commit(0).unwrap();
    assert_eq!(advanced.position(), 1);
    assert!(SymbolicPipeline::open(
        &directory.0,
        None,
        3,
        vec![advanced],
        SymbolicLimits::default()
    )
    .is_err());
    let mut aborted = connect();
    aborted.prepare(0, &input).unwrap();
    aborted.abort(0).unwrap();
    assert_eq!(aborted.position(), 0);
    assert!(!aborted.is_prepared());
    let mut pipeline = SymbolicPipeline::open(
        &directory.0,
        None,
        3,
        vec![aborted],
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    pipeline.forward(69).unwrap();
    pipeline.close().unwrap();
}

#[test]
fn source_free_worker_executes_with_only_authenticated_artifact_and_shard() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("source-free-worker"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&directory.0, None).unwrap();
    let shard_a = directory.0.join("shard-a");
    let shard_b = directory.0.join("shard-b");
    let hash_a = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &directory.0,
        None,
        0..1,
        &shard_a,
        1_000_000,
    )
    .unwrap();
    let hash_b = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &directory.0,
        None,
        1..3,
        &shard_b,
        1_000_000,
    )
    .unwrap();
    let limits = SymbolicLimits {
        max_nodes: 0,
        ..Default::default()
    };
    let mut reference = SymbolicCausalModel::open_artifact(&directory.0, None, limits).unwrap();
    let mut head = SymbolicCausalModel::open_artifact(&directory.0, None, limits).unwrap();
    let inputs = [69, 70, 69].map(|id| reference.compiler().get_embedding(id).unwrap());
    let mut cache = HashConsedKV::new(3, 1, 2, 3);
    let expected = [69, 70, 69].map(|id| reference.forward_cached(&[id], &mut cache).unwrap());
    let hidden_source = source.0.with_extension("temporarily-unavailable");
    struct Restore(std::path::PathBuf, std::path::PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = std::fs::rename(&self.0, &self.1);
        }
    }
    std::fs::rename(&source.0, &hidden_source).unwrap();
    let _restore = Restore(hidden_source, source.0.clone());
    assert!(!source.0.exists());
    assert!(SymbolicCausalModel::open_authenticated_shard(
        &directory.0,
        &"0".repeat(64),
        limits,
        &shard_a,
        &hash_a,
        0..1,
        1_000_000
    )
    .is_err());
    let shard_model = SymbolicCausalModel::open_authenticated_shard(
        &directory.0,
        &identity,
        limits,
        &shard_a,
        &hash_a,
        0..1,
        1_000_000,
    )
    .unwrap();
    assert!(shard_model.compiler().get_embedding(69).is_err());
    let (_worker_a, address_a) = worker_with_shard(
        &directory.0,
        0,
        1,
        Some((&shard_a, &hash_a, Some(&identity))),
    );
    let (_worker_b, address_b) = worker_with_shard(
        &directory.0,
        1,
        3,
        Some((&shard_b, &hash_b, Some(&identity))),
    );
    let mut a = SymbolicLayerClient::connect(
        address_a,
        &identity,
        0..1,
        3,
        SymbolicTransportLimits::default(),
    )
    .unwrap();
    let mut b = SymbolicLayerClient::connect(
        address_b,
        &identity,
        1..3,
        3,
        SymbolicTransportLimits::default(),
    )
    .unwrap();
    for (position, (input, expected)) in inputs.iter().zip(expected).enumerate() {
        let partial = a.prepare(position as u32, input).unwrap();
        let hidden = b.prepare(position as u32, &partial).unwrap();
        let actual = head.project_output(&hidden).unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        a.commit(position as u32).unwrap();
        b.commit(position as u32).unwrap();
    }
    a.close().unwrap();
    b.close().unwrap();
    let kernel = directory.0.join("kernel_0_0.bmlgraph");
    let mut bytes = std::fs::read(&kernel).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&kernel, bytes).unwrap();
    assert!(SymbolicCausalModel::open_authenticated_shard(
        &directory.0,
        &identity,
        limits,
        &shard_a,
        &hash_a,
        0..1,
        1_000_000
    )
    .is_err());
}

#[test]
#[ignore = "requires BML_TEST_GGUF and BML_TEST_SYMBOLIC_ARTIFACT"]
fn source_free_stories15m_workers_match_native_logits() {
    let source = std::path::PathBuf::from(std::env::var_os("BML_TEST_GGUF").unwrap());
    let artifact =
        std::path::PathBuf::from(std::env::var_os("BML_TEST_SYMBOLIC_ARTIFACT").unwrap());
    let directory =
        Directory(std::env::temp_dir().join(format!("bml-stories-shards-{}", std::process::id())));
    std::fs::create_dir(&directory.0).unwrap();
    let identity = checked_identity(&artifact, Some(&source)).unwrap();
    let shard_a = directory.0.join("a");
    let shard_b = directory.0.join("b");
    let hash_a = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact,
        Some(&source),
        0..3,
        &shard_a,
        536_870_912,
    )
    .unwrap();
    let hash_b = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact,
        Some(&source),
        3..6,
        &shard_b,
        536_870_912,
    )
    .unwrap();
    let reference = bml_compiler::gguf_compiler::InferenceCompiler::open_mapped(&source).unwrap();
    let ids = reference.vocab().encode("").unwrap();
    assert_eq!(ids, [1]);
    let expected = reference.forward(&ids).unwrap();
    let runtime = SymbolicLimits {
        max_nodes: 0,
        max_cached_ops: 4_000_000,
        max_programs: 4096,
        graph: bml_runtime::graph::GraphLimits {
            max_ops: 2_000_000,
            max_steps: 4_000_000,
            max_values: 4_000_000,
        },
    };
    let mut head = SymbolicCausalModel::open_artifact(&artifact, Some(&source), runtime).unwrap();
    let input = reference.get_embedding(ids[0]).unwrap();
    let (_worker_a, addr_a) =
        worker_with_shard(&artifact, 0, 3, Some((&shard_a, &hash_a, Some(&identity))));
    let (_worker_b, addr_b) =
        worker_with_shard(&artifact, 3, 6, Some((&shard_b, &hash_b, Some(&identity))));
    let transport = SymbolicTransportLimits {
        timeout: Duration::from_secs(600),
        ..Default::default()
    };
    let mut a = SymbolicLayerClient::connect(addr_a, &identity, 0..3, 2, transport).unwrap();
    let mut b = SymbolicLayerClient::connect(addr_b, &identity, 3..6, 2, transport).unwrap();
    let partial = a.prepare(0, &input).unwrap();
    let hidden = b.prepare(0, &partial).unwrap();
    let logits = head.project_output(&hidden).unwrap();
    assert_eq!(logits.len(), expected.len());
    assert_eq!(logits.len(), 32000);
    assert!(logits.iter().all(|v| v.is_finite()));
    let max = logits
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let x = bml_compiler::sampler::sample(&logits, 0.0, 0).unwrap();
    let y = bml_compiler::sampler::sample(&expected, 0.0, 0).unwrap();
    assert!(max < 1e-4);
    assert_eq!(x, y);
    a.commit(0).unwrap();
    b.commit(0).unwrap();
    a.close().unwrap();
    b.close().unwrap();
    println!(
        "source_free_stories15m logits={} argmax={x} native_argmax={y} max_abs_error={max}",
        logits.len()
    );
}

#[test]
fn worker_explicit_step_budget_is_enforced_regardless_of_argument_order() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let directory = Directory(source.0.with_extension("worker-step-budget"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    for args in [
        ["--symbolic-large-budget", "--symbolic-max-steps", "1"],
        ["--symbolic-max-steps", "1", "--symbolic-large-budget"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
            .args(["--symbolic-artifact"])
            .arg(&directory.0)
            .args(["--layer-start", "0", "--layer-end", "3"])
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("resource limit"));
    }
}

enum PreparedStartup {
    Ready(Worker, SocketAddr),
    Rejected(std::process::ExitStatus, String),
}

fn bounded_prepared_start(directory: &std::path::Path) -> PreparedStartup {
    use std::io::Read;
    let child = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
        .args(["--port", "0", "--prepared"])
        .arg(directory)
        .args(["--worker-index", "0", "--symbolic-context", "3"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut worker = Worker(child);
    let stdout = worker.0.stdout.take().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(stdout).read_line(&mut line).map(|_| line);
        let _ = send.send(result);
    });
    let line = match receive.recv_timeout(Duration::from_secs(5)) {
        Ok(result) => result.unwrap(),
        Err(error) => {
            drop(worker);
            reader.join().unwrap();
            panic!("prepared worker did not report startup or rejection: {error}");
        }
    };
    reader.join().unwrap();
    if line.starts_with("symbolic worker listening on ") {
        let bound: SocketAddr = line.split_whitespace().nth(4).unwrap().parse().unwrap();
        return PreparedStartup::Ready(worker, SocketAddr::from(([127, 0, 0, 1], bound.port())));
    }
    let started = std::time::Instant::now();
    loop {
        if let Some(status) = worker.0.try_wait().unwrap() {
            let mut error = String::new();
            worker
                .0
                .stderr
                .take()
                .unwrap()
                .read_to_string(&mut error)
                .unwrap();
            return PreparedStartup::Rejected(status, error);
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "worker remained alive after {line:?}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn prepared_budget_fixture() -> (Directory, serde_json::Value) {
    let source = symbolic_fixture::fixture_layers(0.25, 1);
    let root = Directory(source.0.with_extension("prepared-budget"));
    std::fs::create_dir(&root.0).unwrap();
    let artifact = root.0.join("artifact");
    std::fs::create_dir(&artifact).unwrap();
    compile_directory_bounded(
        &source.0,
        &artifact,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&artifact, None).unwrap();
    let hash = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact,
        None,
        0..1,
        &root.0.join("shard-0"),
        1_000_000,
    )
    .unwrap();
    let plan = serde_json::json!({"version":1,"artifact":"artifact","artifact_sha256":identity,
        "workers":[{"index":0,"layer_start":0,"layer_end":1,"directory":"shard-0","shard_sha256":hash}]});
    (root, plan)
}

fn assert_prepared_session(directory: &std::path::Path, identity: &str) {
    let (_worker, address) = match bounded_prepared_start(directory) {
        PreparedStartup::Ready(worker, address) => (worker, address),
        PreparedStartup::Rejected(status, error) => {
            panic!("valid plan rejected: {status}: {error}")
        }
    };
    let mut client = SymbolicLayerClient::connect(
        address,
        identity,
        0..1,
        3,
        SymbolicTransportLimits::default(),
    )
    .unwrap();
    let output = client.prepare(0, &[0.25, -0.5, 0.75, 0.0]).unwrap();
    assert_eq!(output.len(), 4);
    assert!(output.iter().all(|value| value.is_finite()));
    client.commit(0).unwrap();
    assert_eq!(client.position(), 1);
    client.reset().unwrap();
    client.close().unwrap();
}

#[test]
fn prepared_plan_byte_budget_is_inclusive_and_does_not_accept_truncated_padding() {
    const LIMIT: usize = 1024 * 1024;
    let (directory, plan) = prepared_budget_fixture();
    let encoded = serde_json::to_vec(&plan).unwrap();
    let identity = plan["artifact_sha256"].as_str().unwrap();
    let path = directory.0.join("plan.json");
    std::fs::write(&path, &encoded).unwrap();
    assert_prepared_session(&directory.0, identity);
    let mut exact = encoded;
    exact.resize(LIMIT, b' ');
    std::fs::write(&path, &exact).unwrap();
    assert_prepared_session(&directory.0, identity);
    // Truncating valid trailing whitespace conceals an input-byte overflow.
    exact.push(b' ');
    std::fs::write(&path, &exact).unwrap();
    match bounded_prepared_start(&directory.0) {
        PreparedStartup::Rejected(status, error) => {
            assert_eq!(status.code(), Some(1));
            assert!(error.contains("plan exceeds byte budget"), "{error}");
        }
        PreparedStartup::Ready(_worker, _) => panic!("oversized plan was admitted"),
    }
    assert_eq!(std::fs::read(&path).unwrap(), exact);
}

#[test]
fn prepared_plan_assignment_index_is_checked_even_when_all_provider_bytes_are_valid() {
    let (directory, mut plan) = prepared_budget_fixture();
    let path = directory.0.join("plan.json");
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    assert_prepared_session(&directory.0, plan["artifact_sha256"].as_str().unwrap());
    plan["workers"][0]["index"] = 1.into();
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    match bounded_prepared_start(&directory.0) {
        PreparedStartup::Rejected(status, error) => {
            assert_eq!(status.code(), Some(1));
            assert!(error.contains("invalid plan assignment"), "{error}");
        }
        PreparedStartup::Ready(_worker, _) => panic!("incorrect assignment index was admitted"),
    }
    plan["workers"][0]["index"] = 0.into();
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    assert_prepared_session(&directory.0, plan["artifact_sha256"].as_str().unwrap());
}

#[test]
fn prepared_plan_checks_version_and_canonical_directories_with_valid_relocated_bytes() {
    let (directory, mut plan) = prepared_budget_fixture();
    let path = directory.0.join("plan.json");
    let identity = plan["artifact_sha256"].as_str().unwrap().to_owned();
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    assert_prepared_session(&directory.0, &identity);
    let rejected = |message: &str| match bounded_prepared_start(&directory.0) {
        PreparedStartup::Rejected(status, error) => {
            assert_eq!(status.code(), Some(1));
            assert!(error.contains(message), "{error}");
        }
        PreparedStartup::Ready(_worker, _) => panic!("invalid prepared schema was admitted"),
    };
    plan["version"] = 0.into();
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    rejected("invalid plan header");
    plan["version"] = 1.into();

    // Relocate real, authenticated provider bytes. If a schema guard is
    // removed, a later provider error cannot incidentally reject this plan.
    let artifact = directory.0.join("artifact");
    let alias = directory.0.join("artifact-alias");
    std::fs::rename(&artifact, &alias).unwrap();
    bml_runtime::symbolic_transport::SymbolicLayerServer::new_authenticated(
        &alias,
        &identity,
        0..1,
        SymbolicTransportLimits {
            context: 3,
            ..Default::default()
        },
        &directory.0.join("shard-0"),
        plan["workers"][0]["shard_sha256"].as_str().unwrap(),
        1_000_000,
    )
    .unwrap();
    plan["artifact"] = "artifact-alias".into();
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    rejected("invalid plan header");
    std::fs::rename(&alias, &artifact).unwrap();
    plan["artifact"] = "artifact".into();

    let shard = directory.0.join("shard-0");
    let shard_alias = directory.0.join("shard-alias");
    std::fs::rename(&shard, &shard_alias).unwrap();
    bml_runtime::symbolic_transport::SymbolicLayerServer::new_authenticated(
        &artifact,
        &identity,
        0..1,
        SymbolicTransportLimits {
            context: 3,
            ..Default::default()
        },
        &shard_alias,
        plan["workers"][0]["shard_sha256"].as_str().unwrap(),
        1_000_000,
    )
    .unwrap();
    plan["workers"][0]["directory"] = "shard-alias".into();
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    rejected("invalid plan assignment");
    std::fs::rename(&shard_alias, &shard).unwrap();
    plan["workers"][0]["directory"] = "shard-0".into();
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    assert_prepared_session(&directory.0, &identity);
}

#[test]
fn prepared_plan_starts_worker_and_rejects_invalid_or_conflicting_assignments() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let root = Directory(source.0.with_extension("worker-prepared"));
    std::fs::create_dir(&root.0).unwrap();
    let artifact = root.0.join("artifact");
    std::fs::create_dir(&artifact).unwrap();
    compile_directory_bounded(
        &source.0,
        &artifact,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&artifact, None).unwrap();
    let shard = root.0.join("shard-0");
    let hash = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact,
        None,
        0..3,
        &shard,
        1_000_000,
    )
    .unwrap();
    let mut plan = serde_json::json!({"version":1,"artifact":"artifact","artifact_sha256":identity,"workers":[{"index":0,"layer_start":0,"layer_end":3,"directory":"shard-0","shard_sha256":hash}]});
    let plan_path = root.0.join("plan.json");
    std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
    let mut reference = SymbolicCausalModel::open_artifact(
        &artifact,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let input = reference.compiler().get_embedding(69).unwrap();
    let mut cache = HashConsedKV::new(3, 1, 2, 3);
    let expected = reference
        .forward_layer_range(&input, 0, &mut cache, 0..3)
        .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
        .args(["--port", "0", "--prepared"])
        .arg(&root.0)
        .args(["--worker-index", "0", "--symbolic-context", "3"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    assert!(line.starts_with("symbolic worker listening on "), "{line}");
    let bound: SocketAddr = line.split_whitespace().nth(4).unwrap().parse().unwrap();
    let _worker = Worker(child);
    let address = SocketAddr::from(([127, 0, 0, 1], bound.port()));
    let mut client = SymbolicLayerClient::connect(
        address,
        &identity,
        0..3,
        3,
        SymbolicTransportLimits::default(),
    )
    .unwrap();
    let actual = client.prepare(0, &input).unwrap();
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    client.abort(0).unwrap();
    client.close().unwrap();
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_bml-worker"))
            .arg("--prepared")
            .arg(&root.0)
            .args(extra)
            .output()
            .unwrap()
    };
    assert!(!run(&["--worker-index", "1"]).status.success());
    assert!(!run(&["--worker-index", "0", "--layer-start", "0"])
        .status
        .success());
    assert!(!run(&[]).status.success());
    plan["workers"][0]["directory"] = "../escape".into();
    std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
    assert!(!run(&["--worker-index", "0"]).status.success());
}

#[test]
#[ignore = "requires BML_TEST_GGUF, BML_TEST_SYMBOLIC_ARTIFACT, BML_TEST_DEPLOYED_A and BML_TEST_DEPLOYED_B"]
fn transferred_stories_workers_match_native_logits() {
    let source = std::path::PathBuf::from(std::env::var_os("BML_TEST_GGUF").unwrap());
    let artifact =
        std::path::PathBuf::from(std::env::var_os("BML_TEST_SYMBOLIC_ARTIFACT").unwrap());
    let deployed_a = std::path::PathBuf::from(std::env::var_os("BML_TEST_DEPLOYED_A").unwrap());
    let deployed_b = std::path::PathBuf::from(std::env::var_os("BML_TEST_DEPLOYED_B").unwrap());
    let identity = checked_identity(&artifact, Some(&source)).unwrap();
    let reference = bml_compiler::gguf_compiler::InferenceCompiler::open_mapped(&source).unwrap();
    let ids = reference.vocab().encode("").unwrap();
    assert_eq!(ids, [1]);
    let expected = reference.forward(&ids).unwrap();
    let input = reference.get_embedding(ids[0]).unwrap();
    let runtime = SymbolicLimits {
        max_nodes: 0,
        max_cached_ops: 4_000_000,
        max_programs: 4096,
        graph: bml_runtime::graph::GraphLimits {
            max_ops: 2_000_000,
            max_steps: 4_000_000,
            max_values: 4_000_000,
        },
    };
    let mut head = SymbolicCausalModel::open_artifact(&artifact, Some(&source), runtime).unwrap();
    let open = |dir: &std::path::Path, index: &str| {
        let mut process = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
            .args(["--port", "0", "--prepared"])
            .arg(dir)
            .args([
                "--worker-index",
                index,
                "--symbolic-context",
                "2",
                "--symbolic-kv-bytes",
                "1048576",
                "--symbolic-large-budget",
                "--symbolic-timeout-seconds",
                "600",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(process.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        if !line.starts_with("symbolic worker listening on ") {
            use std::io::Read;
            let mut error = String::new();
            process
                .stderr
                .take()
                .unwrap()
                .read_to_string(&mut error)
                .unwrap();
            panic!("worker startup: {line} {error}");
        }
        let bound: SocketAddr = line.split_whitespace().nth(4).unwrap().parse().unwrap();
        (
            Worker(process),
            SocketAddr::from(([127, 0, 0, 1], bound.port())),
        )
    };
    let (_a, address_a) = open(&deployed_a, "0");
    let (_b, address_b) = open(&deployed_b, "1");
    let transport = SymbolicTransportLimits {
        timeout: Duration::from_secs(600),
        ..Default::default()
    };
    let mut a = SymbolicLayerClient::connect(address_a, &identity, 0..3, 2, transport).unwrap();
    let mut b = SymbolicLayerClient::connect(address_b, &identity, 3..6, 2, transport).unwrap();
    let intermediate = a.prepare(0, &input).unwrap();
    let hidden = b.prepare(0, &intermediate).unwrap();
    let actual = head.project_output(&hidden).unwrap();
    assert_eq!(actual.len(), 32000);
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|x| x.is_finite()));
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let selected = bml_compiler::sampler::sample(&actual, 0.0, 0).unwrap();
    let native = bml_compiler::sampler::sample(&expected, 0.0, 0).unwrap();
    assert!(max < 1e-4);
    assert_eq!(selected, native);
    a.commit(0).unwrap();
    b.commit(0).unwrap();
    a.close().unwrap();
    b.close().unwrap();
    println!("transferred_stories logits={} argmax={selected} native_argmax={native} max_abs_error={max}", actual.len());
}
