use bml_compiler::{
    symbolic_artifact::compile_directory_bounded, symbolic_bounded::BoundedProfile,
    symbolic_kernels::KernelLimits, HashConsedKV,
};
use bml_runtime::{
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
    symbolic_transport::{
        SymbolicLayerClient, SymbolicLayerServer, SymbolicPipeline, SymbolicTransportLimits,
    },
};
use std::{
    io::{self, Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
#[allow(dead_code)]
#[path = "support/symbolic_fixture.rs"]
mod symbolic_fixture;

struct Directory(std::path::PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn artifact() -> (symbolic_fixture::Fixture, Directory) {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let dir = Directory(source.0.with_extension("transport-artifact"));
    std::fs::create_dir(&dir.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &dir.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    (source, dir)
}
fn limits() -> SymbolicTransportLimits {
    SymbolicTransportLimits {
        timeout: Duration::from_secs(10),
        ..Default::default()
    }
}
fn listener(
    server: Arc<SymbolicLayerServer>,
    count: usize,
) -> (std::net::SocketAddr, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let task = thread::spawn(move || {
        let mut workers = Vec::new();
        for _ in 0..count {
            let (stream, _) = listener.accept().unwrap();
            let server = Arc::clone(&server);
            workers.push(thread::spawn(move || server.serve_connection(stream)));
        }
        for worker in workers {
            worker.join().unwrap().unwrap();
        }
    });
    (address, task)
}

fn test_socket(address: std::net::SocketAddr) -> TcpStream {
    let stream = TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream
}

fn test_write_frame(stream: &mut TcpStream, payload: &[u8]) {
    stream
        .write_all(&(payload.len() as u32).to_le_bytes())
        .unwrap();
    stream.write_all(payload).unwrap();
}

fn test_read_frame(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut prefix = [0; 4];
    stream.read_exact(&mut prefix)?;
    let count = u32::from_le_bytes(prefix) as usize;
    assert!((1..=1024).contains(&count), "unexpected test response size");
    let mut payload = vec![0; count];
    stream.read_exact(&mut payload)?;
    Ok(payload)
}

fn test_open(identity: &str, context: u32) -> Vec<u8> {
    let mut request = vec![0];
    for field in [1_u32, context, 0, 1] {
        request.extend_from_slice(&field.to_le_bytes());
    }
    request.extend_from_slice(identity.as_bytes());
    assert_eq!(request.len(), 81);
    request
}

fn short_limits(frame_bytes: usize) -> SymbolicTransportLimits {
    SymbolicTransportLimits {
        frame_bytes,
        context: 2,
        timeout: Duration::from_secs(3),
        ..Default::default()
    }
}

#[test]
fn raw_open_checks_each_owned_field_and_the_servers_own_context_budget() {
    let (_source, artifact) = artifact();
    let policy = short_limits(128);
    let server = Arc::new(SymbolicLayerServer::new(&artifact.0, None, 0..1, policy).unwrap());
    let original = test_open(server.identity(), 2);
    let mut requests = Vec::new();
    for (name, offset, value) in [
        ("version", 1, 2_u32),
        ("start", 9, 1),
        ("end", 13, 2),
        // The artifact admits three positions. Only the server policy limits
        // this request to two, so a later session check cannot mask this guard.
        ("context", 5, 3),
    ] {
        let mut request = original.clone();
        request[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        requests.push((name, request));
    }
    let mut wrong_type = original.clone();
    wrong_type[0] = 1;
    requests.push(("type", wrong_type));
    let mut wrong_identity = original.clone();
    wrong_identity[80] ^= 1;
    requests.push(("identity", wrong_identity));
    let mut trailing = original.clone();
    trailing.push(0);
    requests.push(("length", trailing));
    for (name, request) in requests {
        let (address, task) = listener(Arc::clone(&server), 1);
        let mut stream = test_socket(address);
        test_write_frame(&mut stream, &request);
        let response = test_read_frame(&mut stream).expect("an explicit rejection, not EOF");
        assert_eq!(response[0], 1, "accepted altered {name}");
        let message = std::str::from_utf8(&response[1..]).unwrap();
        if name == "context" {
            assert_eq!(message, "invalid context");
        } else {
            assert_eq!(message, "invalid identity, version or range");
        }
        assert!(task.join().is_ok());
    }
    // Independent subsequent admission and a complete transaction must work.
    let mut reference = SymbolicCausalModel::open_artifact(
        &artifact.0,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let input = reference.compiler().get_embedding(69).unwrap();
    let mut cache = HashConsedKV::new(3, 1, 2, 2);
    let expected = reference
        .forward_layer_range(&input, 0, &mut cache, 0..1)
        .unwrap();
    let (address, task) = listener(server.clone(), 1);
    let mut client =
        SymbolicLayerClient::connect(address, server.identity(), 0..1, 2, policy).unwrap();
    let actual = client.prepare(0, &input).unwrap();
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    client.commit(0).unwrap();
    assert_eq!(client.position(), 1);
    assert!(client.is_healthy());
    client.reset().unwrap();
    assert_eq!(client.prepare(0, &input).unwrap(), actual);
    client.abort(0).unwrap();
    client.close().unwrap();
    task.join().unwrap();
}

#[test]
fn frame_budgets_include_exact_open_size_and_maximum_u32_without_large_frames() {
    let (_source, artifact) = artifact();
    for budget in [81, u32::MAX as usize] {
        let policy = short_limits(budget);
        let server = Arc::new(SymbolicLayerServer::new(&artifact.0, None, 0..1, policy).unwrap());
        let (address, task) = listener(server.clone(), 1);
        // Even the maximum policy sends only the 81-byte Open and tiny controls;
        // no allocation scales with the configured frame budget.
        let mut client = SymbolicLayerClient::connect(address, server.identity(), 0..1, 2, policy)
            .expect("both inclusive frame endpoints must admit an ordinary session");
        client.reset().unwrap();
        assert!(client.is_healthy());
        client.close().unwrap();
        task.join().unwrap();
    }
}

#[test]
fn invalid_inbound_sizes_fail_in_framing_before_any_open_rejection() {
    let (_source, artifact) = artifact();
    let server =
        Arc::new(SymbolicLayerServer::new(&artifact.0, None, 0..1, short_limits(128)).unwrap());
    for payload in [Vec::new(), vec![0_u8; 130]] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = Arc::clone(&server);
        let (written, inspect) = std::sync::mpsc::channel();
        let task = thread::spawn(move || {
            let stream = listener.accept().unwrap().0;
            inspect.recv().unwrap();
            server.serve_connection(stream)
        });
        let mut stream = test_socket(address);
        // Supply the entire declared payload and finish the write side. A peer
        // that incorrectly proceeds cannot turn an absent body into a false hit.
        test_write_frame(&mut stream, &payload);
        stream.shutdown(Shutdown::Write).unwrap();
        written.send(()).unwrap();
        let error = task
            .join()
            .unwrap()
            .expect_err("invalid frame reached Open admission");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "symbolic frame size");
    }
}

fn assert_local_rejection<T: std::fmt::Debug>(result: io::Result<T>, message: &str) {
    let error = result.unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(error.to_string(), message);
}

#[test]
fn client_prepare_and_commit_preflight_never_sends_or_consumes_rejected_work() {
    let trace = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
    let recorded = Arc::clone(&trace);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let task = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let open = test_read_frame(&mut stream).unwrap();
        assert_eq!(open.len(), 81);
        recorded.lock().unwrap().push(open);
        test_write_frame(&mut stream, &[0, 2, 0, 0, 0]);
        loop {
            let request = match test_read_frame(&mut stream) {
                Ok(request) => request,
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return,
                Err(error) => panic!("test peer: {error}"),
            };
            let code = request[0];
            recorded.lock().unwrap().push(request);
            // Deliberately accept every prepare/control: rejecting invalid
            // client work here would mask missing *local* validation.
            if code == 1 {
                let mut response = vec![0, 2, 0, 0, 0];
                for value in [10.0_f64, 20.0] {
                    response.extend_from_slice(&value.to_le_bytes());
                }
                test_write_frame(&mut stream, &response);
            } else {
                test_write_frame(&mut stream, &[0]);
                if code == 5 {
                    return;
                }
            }
        }
    });
    let policy = SymbolicTransportLimits {
        context: 3,
        ..short_limits(81)
    };
    let mut client =
        SymbolicLayerClient::connect(address, &"a".repeat(64), 0..1, 3, policy).unwrap();
    let codes = || {
        trace
            .lock()
            .unwrap()
            .iter()
            .map(|frame| frame[0])
            .collect::<Vec<_>>()
    };
    assert_local_rejection(client.commit(0), "no matching symbolic row to commit");
    for (position, input) in [
        (1, vec![1.0, 2.0]),
        (0, vec![1.0]),
        (0, vec![f64::NAN, 2.0]),
        (0, vec![1.0, f64::INFINITY]),
    ] {
        assert_local_rejection(
            client.prepare(position, &input),
            "invalid symbolic input vector",
        );
        assert!(client.is_healthy());
        assert_eq!(client.position(), 0);
        assert!(!client.is_prepared());
        assert_eq!(codes(), [0]);
    }
    assert_eq!(client.prepare(0, &[1.0, 2.0]).unwrap(), [10.0, 20.0]);
    assert_eq!(codes(), [0, 1]);
    assert_local_rejection(
        client.prepare(0, &[1.0, 2.0]),
        "invalid symbolic input vector",
    );
    assert_local_rejection(client.commit(1), "no matching symbolic row to commit");
    assert_local_rejection(client.abort(1), "no matching symbolic row to abort");
    assert!(client.is_healthy());
    assert!(client.is_prepared());
    assert_eq!(client.position(), 0);
    assert_eq!(codes(), [0, 1]);
    client.abort(0).unwrap();
    assert!(!client.is_prepared());
    for position in 0..3 {
        assert_eq!(client.prepare(position, &[1.0, 2.0]).unwrap(), [10.0, 20.0]);
        assert!(client.is_prepared());
        assert_eq!(client.position(), position);
        client.commit(position).unwrap();
        assert!(!client.is_prepared());
        assert_eq!(client.position(), position + 1);
    }
    let before = codes();
    assert_local_rejection(
        client.prepare(3, &[1.0, 2.0]),
        "invalid symbolic input vector",
    );
    assert_eq!(codes(), before);
    assert_eq!(client.position(), 3);
    assert!(client.is_healthy());
    client.reset().unwrap();
    assert_eq!(client.position(), 0);
    assert_eq!(client.prepare(0, &[1.0, 2.0]).unwrap(), [10.0, 20.0]);
    client.abort(0).unwrap();
    client.close().unwrap();
    assert!(!client.is_healthy());
    task.join().unwrap();
}

#[test]
fn two_tcp_ranges_match_persisted_causal_logits_and_keep_sessions_independent() {
    let (_source, artifact) = artifact();
    let shard_a = artifact.0.join("shard_a");
    let shard_b = artifact.0.join("shard_b");
    let hash_a = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact.0,
        None,
        0..1,
        &shard_a,
        1_000_000,
    )
    .unwrap();
    let hash_b = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact.0,
        None,
        1..3,
        &shard_b,
        1_000_000,
    )
    .unwrap();
    let server_a = Arc::new(
        SymbolicLayerServer::new(&artifact.0, None, 0..1, limits())
            .unwrap()
            .with_weight_shard(&shard_a, &hash_a, 1_000_000)
            .unwrap(),
    );
    let server_b = Arc::new(
        SymbolicLayerServer::new(&artifact.0, None, 1..3, limits())
            .unwrap()
            .with_weight_shard(&shard_b, &hash_b, 1_000_000)
            .unwrap(),
    );
    assert_eq!(server_a.identity(), server_b.identity());
    let (address_a, task_a) = listener(Arc::clone(&server_a), 2);
    let (address_b, task_b) = listener(Arc::clone(&server_b), 1);
    let mut a =
        SymbolicLayerClient::connect(address_a, server_a.identity(), 0..1, 3, limits()).unwrap();
    let mut other =
        SymbolicLayerClient::connect(address_a, server_a.identity(), 0..1, 3, limits()).unwrap();
    let mut b =
        SymbolicLayerClient::connect(address_b, server_b.identity(), 1..3, 3, limits()).unwrap();
    let open = || {
        SymbolicCausalModel::open_artifact(
            &artifact.0,
            None,
            SymbolicLimits {
                max_nodes: 0,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let mut model = open();
    let mut full = open();
    let mut cache = HashConsedKV::new(3, 1, 2, 3);
    for (position, id) in [69, 70, 69].into_iter().enumerate() {
        let position = position as u32;
        let input = model.compiler().get_embedding(id).unwrap();
        let initial = a.prepare(position, &input).unwrap();
        assert_eq!(server_a.width(), input.len());
        assert_eq!(initial.len(), server_a.width());
        assert!(a.prepare(position, &input).is_err());
        assert!(a.is_healthy());
        a.abort(position).unwrap();
        let next = a.prepare(position, &input).unwrap();
        assert_eq!(
            initial.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            next.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        if position == 0 {
            other.prepare(0, &input).unwrap();
            other.commit(0).unwrap();
        } else if position == 1 {
            other.prepare(1, &input).unwrap();
            other.abort(1).unwrap();
        }
        let hidden = b.prepare(position, &next).unwrap();
        assert_eq!(server_b.width(), next.len());
        assert_eq!(hidden.len(), server_b.width());
        let actual = model.project_output(&hidden).unwrap();
        let expected = full.forward_cached(&[id], &mut cache).unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        a.commit(position).unwrap();
        b.commit(position).unwrap();
    }
    let input = model.compiler().get_embedding(69).unwrap();
    assert!(other.prepare(2, &input).is_err());
    other.reset().unwrap();
    other.prepare(0, &input).unwrap();
    other.abort(0).unwrap();
    a.close().unwrap();
    b.close().unwrap();
    other.close().unwrap();
    task_a.join().unwrap();
    task_b.join().unwrap();
}
#[test]
fn handshake_rejects_wrong_identity_range_context_and_frame_limits() {
    let (_source, artifact) = artifact();
    let server = Arc::new(SymbolicLayerServer::new(&artifact.0, None, 0..1, limits()).unwrap());
    let (address, task) = listener(Arc::clone(&server), 3);
    assert!(SymbolicLayerClient::connect(address, &"0".repeat(64), 0..1, 3, limits()).is_err());
    assert!(SymbolicLayerClient::connect(address, server.identity(), 1..2, 3, limits()).is_err());
    let mut excessive = limits();
    excessive.context = 99;
    assert!(SymbolicLayerClient::connect(address, server.identity(), 0..1, 9, excessive).is_err());
    task.join().unwrap();
    let mut small = limits();
    small.frame_bytes = 80;
    assert!(SymbolicLayerServer::new(&artifact.0, None, 0..1, small).is_err());
}
#[test]
fn lost_commit_response_invalidates_client_and_is_not_retried() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let task = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut prefix = [0; 4];
        stream.read_exact(&mut prefix).unwrap();
        let mut request = vec![0; u32::from_le_bytes(prefix) as usize];
        stream.read_exact(&mut request).unwrap();
        stream.write_all(&5u32.to_le_bytes()).unwrap();
        stream.write_all(&[0, 4, 0, 0, 0]).unwrap();
        stream.read_exact(&mut prefix).unwrap();
        let mut prepare = vec![0; u32::from_le_bytes(prefix) as usize];
        stream.read_exact(&mut prepare).unwrap();
        assert_eq!(prepare[0], 1);
        let mut response = vec![0];
        response.extend_from_slice(&prepare[5..]);
        stream
            .write_all(&(response.len() as u32).to_le_bytes())
            .unwrap();
        stream.write_all(&response).unwrap();
        stream.read_exact(&mut prefix).unwrap();
        let mut commit = vec![0; u32::from_le_bytes(prefix) as usize];
        stream.read_exact(&mut commit).unwrap();
        assert_eq!(commit[0], 2);
    });
    let mut client =
        SymbolicLayerClient::connect(address, &"a".repeat(64), 0..1, 3, limits()).unwrap();
    client.prepare(0, &[0.0; 4]).unwrap();
    assert!(client.commit(0).is_err());
    assert!(!client.is_healthy());
    assert!(client.commit(0).is_err());
    task.join().unwrap();
}

#[test]
fn pipeline_invalidates_all_sessions_after_a_lost_commit_response() {
    let (_source, artifact) = artifact();
    let server = Arc::new(SymbolicLayerServer::new(&artifact.0, None, 1..3, limits()).unwrap());
    let identity = server.identity().to_owned();
    let real_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let real_address = real_listener.local_addr().unwrap();
    let real_task = thread::spawn(move || {
        let (stream, _) = real_listener.accept().unwrap();
        let _ = server.serve_connection(stream); // The invalidated client closes its uncommitted session.
    });
    let fake_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let fake_address = fake_listener.local_addr().unwrap();
    let fake_task = thread::spawn(move || {
        let (mut stream, _) = fake_listener.accept().unwrap();
        let read = |stream: &mut std::net::TcpStream| {
            let mut prefix = [0; 4];
            stream.read_exact(&mut prefix).unwrap();
            let mut bytes = vec![0; u32::from_le_bytes(prefix) as usize];
            stream.read_exact(&mut bytes).unwrap();
            bytes
        };
        let open = read(&mut stream);
        assert_eq!(open[0], 0);
        stream.write_all(&5u32.to_le_bytes()).unwrap();
        stream.write_all(&[0, 4, 0, 0, 0]).unwrap();
        let prepare = read(&mut stream);
        assert_eq!(prepare[0], 1);
        // Return a valid finite vector; this peer deliberately exists only to inject a transport fault.
        let mut output = vec![0];
        output.extend_from_slice(&prepare[5..]);
        stream
            .write_all(&(output.len() as u32).to_le_bytes())
            .unwrap();
        stream.write_all(&output).unwrap();
        let commit = read(&mut stream);
        assert_eq!(commit[0], 2);
        // The peer has received the commit but closes without acknowledging it.
    });
    let first = SymbolicLayerClient::connect(fake_address, &identity, 0..1, 3, limits()).unwrap();
    let second = SymbolicLayerClient::connect(real_address, &identity, 1..3, 3, limits()).unwrap();
    let mut pipeline = SymbolicPipeline::open(
        &artifact.0,
        None,
        3,
        vec![first, second],
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(pipeline.forward(69).is_err());
    assert_eq!(pipeline.position(), 0);
    assert!(!pipeline.is_healthy());
    assert!(pipeline.forward(69).is_err());
    assert!(pipeline.reset().is_err());
    drop(pipeline);
    fake_task.join().unwrap();
    real_task.join().unwrap();
}

#[test]
fn changing_a_valid_manifest_changes_identity_and_old_server_refuses_it() {
    let (_source, artifact) = artifact();
    let server = Arc::new(SymbolicLayerServer::new(&artifact.0, None, 0..1, limits()).unwrap());
    let old_identity = server.identity().to_owned();
    let path = artifact.0.join(bml_compiler::symbolic_artifact::MANIFEST);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(b'\n');
    std::fs::write(&path, bytes).unwrap();
    let new_identity =
        bml_compiler::symbolic_artifact::checked_identity(&artifact.0, None).unwrap();
    assert_ne!(old_identity, new_identity);
    let socket = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = socket.local_addr().unwrap();
    let task = thread::spawn(move || {
        let (stream, _) = socket.accept().unwrap();
        assert!(server.serve_connection(stream).is_err());
    });
    assert!(SymbolicLayerClient::connect(address, &old_identity, 0..1, 3, limits()).is_err());
    task.join().unwrap();
}

#[test]
fn later_commit_loss_invalidates_pipeline_even_after_an_earlier_worker_acknowledges() {
    let (_source, artifact) = artifact();
    let server = Arc::new(SymbolicLayerServer::new(&artifact.0, None, 0..1, limits()).unwrap());
    let identity = server.identity().to_owned();
    let real = TcpListener::bind("127.0.0.1:0").unwrap();
    let real_address = real.local_addr().unwrap();
    let real_task = thread::spawn(move || {
        let (stream, _) = real.accept().unwrap();
        let _ = server.serve_connection(stream);
    });
    let fake = TcpListener::bind("127.0.0.1:0").unwrap();
    let fake_address = fake.local_addr().unwrap();
    let fake_task = thread::spawn(move || {
        let (mut stream, _) = fake.accept().unwrap();
        let read = |stream: &mut std::net::TcpStream| {
            let mut prefix = [0; 4];
            stream.read_exact(&mut prefix).unwrap();
            let mut bytes = vec![0; u32::from_le_bytes(prefix) as usize];
            stream.read_exact(&mut bytes).unwrap();
            bytes
        };
        assert_eq!(read(&mut stream)[0], 0);
        stream.write_all(&5u32.to_le_bytes()).unwrap();
        stream.write_all(&[0, 4, 0, 0, 0]).unwrap();
        let prepare = read(&mut stream);
        assert_eq!(prepare[0], 1);
        let mut response = vec![0];
        response.extend_from_slice(&prepare[5..]);
        stream
            .write_all(&(response.len() as u32).to_le_bytes())
            .unwrap();
        stream.write_all(&response).unwrap();
        assert_eq!(read(&mut stream)[0], 2);
    });
    let first = SymbolicLayerClient::connect(real_address, &identity, 0..1, 3, limits()).unwrap();
    let second = SymbolicLayerClient::connect(fake_address, &identity, 1..3, 3, limits()).unwrap();
    let mut pipeline = SymbolicPipeline::open(
        &artifact.0,
        None,
        3,
        vec![first, second],
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(pipeline.forward(69).is_err());
    assert_eq!(pipeline.position(), 0);
    assert!(!pipeline.is_healthy());
    assert!(pipeline.forward(69).is_err());
    drop(pipeline);
    fake_task.join().unwrap();
    real_task.join().unwrap();
}

#[test]
fn oversized_frame_is_rejected_before_its_body_is_received() {
    let (_source, artifact) = artifact();
    let mut restricted = limits();
    restricted.frame_bytes = 128;
    let server = SymbolicLayerServer::new(&artifact.0, None, 0..1, restricted).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let task = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        assert!(server.serve_connection(stream).is_err());
    });
    let mut stream = std::net::TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream.write_all(&129u32.to_le_bytes()).unwrap();
    let mut response = [0; 1];
    assert!(matches!(stream.read(&mut response), Ok(0) | Err(_)));
    task.join().unwrap();
}

#[test]
fn worker_admission_rejects_a_different_relocated_source_model() {
    let (_source, artifact) = artifact();
    let other = symbolic_fixture::fixture_layers(0.75, 3);
    assert!(SymbolicLayerServer::new(&artifact.0, Some(&other.0), 0..1, limits()).is_err());
}

#[test]
fn source_free_sessions_keep_verified_bytes_and_reject_replaced_shards() {
    let (_source, artifact) = artifact();
    let identity = bml_compiler::symbolic_artifact::checked_identity(&artifact.0, None).unwrap();
    let shard = artifact.0.join("replacement-shard");
    let hash = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact.0,
        None,
        0..1,
        &shard,
        1_000_000,
    )
    .unwrap();
    let mut reference = SymbolicCausalModel::open_artifact(
        &artifact.0,
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
        .forward_layer_range(&input, 0, &mut cache, 0..1)
        .unwrap();
    let server = Arc::new(
        SymbolicLayerServer::new_authenticated(
            &artifact.0,
            &identity,
            0..1,
            limits(),
            &shard,
            &hash,
            1_000_000,
        )
        .unwrap(),
    );
    let (address, task) = listener(server, 2);
    let mut established =
        SymbolicLayerClient::connect(address, &identity, 0..1, 3, limits()).unwrap();
    let payload_path = shard.join("weights.bin");
    let mut payload = std::fs::read(&payload_path).unwrap();
    payload[0] ^= 1;
    std::fs::write(&payload_path, payload).unwrap();
    assert!(SymbolicLayerClient::connect(address, &identity, 0..1, 3, limits()).is_err());
    let actual = established.prepare(0, &input).unwrap();
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    established.commit(0).unwrap();
    established.close().unwrap();
    task.join().unwrap();
}
