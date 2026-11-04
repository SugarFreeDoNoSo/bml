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
    io::{Read, Write},
    net::TcpListener,
    sync::Arc,
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
#[test]
fn two_tcp_ranges_match_persisted_causal_logits_and_keep_sessions_independent() {
    let (_source, artifact) = artifact();
    let server_a = Arc::new(SymbolicLayerServer::new(&artifact.0, None, 0..1, limits()).unwrap());
    let server_b = Arc::new(SymbolicLayerServer::new(&artifact.0, None, 1..3, limits()).unwrap());
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
