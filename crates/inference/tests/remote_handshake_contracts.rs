#[path = "support/gguf_fixture.rs"]
mod gguf_fixture;

use bml_gguf::GgufReader;
use bml_inference::{
    model::Model,
    remote::{LayerClient, RemoteLimits, RemotePipeline},
};
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

fn receive(stream: &mut TcpStream) -> Value {
    let mut prefix = [0; 4];
    stream.read_exact(&mut prefix).unwrap();
    let count = u32::from_le_bytes(prefix) as usize;
    assert!((1..=4096).contains(&count));
    let mut bytes = vec![0; count];
    stream.read_exact(&mut bytes).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn forward_rejects_shape_and_nonfinite_values_locally_without_poisoning_the_session() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let received = Arc::clone(&requests);
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let opening = receive(&mut stream);
        assert_eq!(opening["Open"]["context"], 2);
        received.lock().unwrap().push(opening);
        send(
            &mut stream,
            json!({"Ready": {"start": 0, "end": 1, "width": 4}}),
        );
        loop {
            let request = receive(&mut stream);
            received.lock().unwrap().push(request.clone());
            if let Some(forward) = request.get("Forward") {
                // Deliberately admit every hidden payload, including wrong
                // shapes and JSON nulls produced by nonfinite f32 values. A
                // server-side rejection must not mask a missing client guard.
                send(
                    &mut stream,
                    json!({"Output": {
                        "position": forward["position"],
                        "hidden": [4.0, 3.0, 2.0, 1.0],
                    }}),
                );
            } else if request == json!("Reset") {
                send(&mut stream, json!("Ack"));
            } else {
                assert_eq!(request, json!("Close"));
                send(&mut stream, json!("Ack"));
                break;
            }
        }
    });
    let mut client = LayerClient::connect(
        address,
        2,
        [7; 32],
        RemoteLimits {
            frame_bytes: 4096,
            timeout: Duration::from_secs(3),
            ..Default::default()
        },
    )
    .unwrap();
    let opening = requests.lock().unwrap().clone();
    assert_eq!(opening.len(), 1);
    // Each input violates exactly one admission condition. Infinity/NaN use
    // the agreed width; every incorrectly sized input contains finite values.
    for hidden in [
        vec![],
        vec![1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![f32::INFINITY, 2.0, 3.0, 4.0],
        vec![f32::NEG_INFINITY, 2.0, 3.0, 4.0],
        vec![1.0, f32::NAN, 3.0, 4.0],
    ] {
        let error = client.forward(0, &hidden).unwrap_err();
        assert_eq!(error.to_string(), "invalid remote hidden state");
        assert_eq!(
            *requests.lock().unwrap(),
            opening,
            "local rejection must not send a Forward request"
        );
    }
    let valid = [1.0, -0.0, -2.0, 0.5];
    assert_eq!(client.forward(0, &valid).unwrap(), [4.0, 3.0, 2.0, 1.0]);
    client.reset().unwrap();
    assert_eq!(client.forward(0, &[0.0; 4]).unwrap(), [4.0, 3.0, 2.0, 1.0]);
    client.close().unwrap();
    server.join().unwrap();
    assert_eq!(
        requests.lock().unwrap()[1..],
        [
            json!({"Forward": {"position": 0, "hidden": valid}}),
            json!("Reset"),
            json!({"Forward": {"position": 0, "hidden": [0.0, 0.0, 0.0, 0.0]}}),
            json!("Close"),
        ]
    );
}

fn send(stream: &mut TcpStream, value: Value) {
    let bytes = serde_json::to_vec(&value).unwrap();
    stream
        .write_all(&(bytes.len() as u32).to_le_bytes())
        .unwrap();
    stream.write_all(&bytes).unwrap();
}

#[test]
fn forward_rejects_remote_metadata_after_a_valid_local_request() {
    for (name, response, valid) in [
        (
            "valid",
            json!({"Output": {"position": 0, "hidden": [4.0, 3.0, 2.0, 1.0]}}),
            true,
        ),
        (
            "wrong-position",
            json!({"Output": {"position": 1, "hidden": [4.0, 3.0, 2.0, 1.0]}}),
            false,
        ),
        (
            "wrong-width",
            json!({"Output": {"position": 0, "hidden": [4.0, 3.0, 2.0]}}),
            false,
        ),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "peer accept deadline: {name}");
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("peer accept: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let opening = receive(&mut stream);
            assert_eq!(opening["Open"]["context"], 2);
            send(
                &mut stream,
                json!({"Ready": {"start": 0, "end": 1, "width": 4}}),
            );
            let request = receive(&mut stream);
            assert_eq!(request["Forward"]["position"], 0);
            let hidden: Vec<_> = request["Forward"]["hidden"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect();
            assert_eq!(hidden, [1.0, 2.0, 3.0, 4.0]);
            // The peer does not reject malformed response metadata on the
            // client's behalf. It sends a complete valid JSON frame in all cases.
            send(&mut stream, response);
            let mut byte = [0_u8; 1];
            assert_eq!(
                stream.read(&mut byte).unwrap(),
                0,
                "client must close: {name}"
            );
        });
        let mut client = LayerClient::connect(
            address,
            2,
            [7; 32],
            RemoteLimits {
                frame_bytes: 4096,
                timeout: Duration::from_secs(3),
                ..RemoteLimits::default()
            },
        )
        .unwrap();
        let result = client.forward(0, &[1.0, 2.0, 3.0, 4.0]);
        // Close and collect the peer before asserting the response contract,
        // including when a mutated guard admits the wrong metadata.
        drop(client);
        server.join().unwrap();
        if valid {
            assert_eq!(result.unwrap(), [4.0, 3.0, 2.0, 1.0]);
        } else {
            let error = result.unwrap_err().to_string();
            assert!(
                error.contains("invalid remote response; discard pipeline"),
                "{name}: {error}"
            );
        }
    }
}

#[test]
fn opening_requires_a_nonempty_ordered_range_and_positive_hidden_width() {
    // Each invalid reply violates exactly one part of the Ready contract.
    // The valid control also exercises a real forward and orderly close.
    for (start, end, width, valid) in [
        (2, 2, 4, false),
        (3, 2, 4, false),
        (2, 3, 0, false),
        (2, 3, 4, true),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let request = receive(&mut stream);
            assert_eq!(request["Open"]["version"], 2);
            assert_eq!(request["Open"]["context"], 2);
            assert_eq!(request["Open"]["fingerprint"], json!(vec![7_u8; 32]));
            send(
                &mut stream,
                json!({"Ready": {"start": start, "end": end, "width": width}}),
            );
            if valid {
                let request = receive(&mut stream);
                assert_eq!(request["Forward"]["position"], 0);
                assert_eq!(request["Forward"]["hidden"], json!([1.0, 2.0, 3.0, 4.0]));
                send(
                    &mut stream,
                    json!({"Output": {"position": 0, "hidden": [4.0, 3.0, 2.0, 1.0]}}),
                );
                assert_eq!(receive(&mut stream), json!("Close"));
                send(&mut stream, json!("Ack"));
            }
        });
        let result = LayerClient::connect(
            address,
            2,
            [7; 32],
            RemoteLimits {
                frame_bytes: 4096,
                timeout: Duration::from_secs(3),
                ..Default::default()
            },
        );
        if valid {
            let mut client = result.unwrap();
            assert_eq!(client.layers(), 2..3);
            assert_eq!(
                client.forward(0, &[1.0, 2.0, 3.0, 4.0]).unwrap(),
                [4.0, 3.0, 2.0, 1.0]
            );
            client.close().unwrap();
        } else {
            let error = result.err().expect("invalid Ready metadata was accepted");
            assert!(error
                .to_string()
                .contains("invalid session opening response"));
        }
        server.join().unwrap();
    }
}

#[test]
fn client_setup_rejects_frame_budget_above_u32_max() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let _ = listener.accept();
    });
    if usize::BITS > 32 {
        let err = LayerClient::connect(
            address,
            2,
            [7; 32],
            RemoteLimits {
                frame_bytes: (u32::MAX as usize) + 1,
                timeout: Duration::from_secs(3),
                ..Default::default()
            },
        )
        .err()
        .unwrap();
        assert!(
            err.to_string().contains("invalid transport limits"),
            "{err}"
        );
    }
    server.join().unwrap();
}

#[test]
fn client_setup_enforces_zero_frame_zero_timeout_and_exact_u32_max_frame_budget() {
    for (name, frame_bytes, timeout) in [
        ("zero-frame", 0usize, Duration::from_secs(2)),
        ("zero-timeout", 4096usize, Duration::ZERO),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut byte = [0_u8; 1];
            assert_eq!(
                stream.read(&mut byte).unwrap(),
                0,
                "invalid transport limits must fail before sending: {name}"
            );
        });
        let err = LayerClient::connect(
            address,
            2,
            [7; 32],
            RemoteLimits {
                frame_bytes,
                timeout,
                ..Default::default()
            },
        )
        .err()
        .expect("invalid transport limits must be rejected");
        assert!(
            err.to_string().contains("invalid transport limits"),
            "{name}: {err}"
        );
        server.join().unwrap();
    }

    // Exact u32::MAX frame_bytes is inclusive and must pass setup without
    // allocating beyond the small actual handshake frame.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let opening = receive(&mut stream);
        assert_eq!(opening["Open"]["context"], 2);
        send(
            &mut stream,
            json!({"Ready": {"start": 0, "end": 1, "width": 4}}),
        );
        assert_eq!(receive(&mut stream), json!("Close"));
        send(&mut stream, json!("Ack"));
    });
    let client = LayerClient::connect(
        address,
        2,
        [7; 32],
        RemoteLimits {
            frame_bytes: u32::MAX as usize,
            timeout: Duration::from_secs(2),
            ..Default::default()
        },
    )
    .expect("u32::MAX frame budget is inclusive");
    client.close().unwrap();
    server.join().unwrap();
}

#[test]
fn receive_and_send_enforce_exact_frame_boundaries_and_reject_zero_length() {
    let open_len = serde_json::to_vec(&json!({
        "Open": {
            "version": 2u32,
            "context": 2usize,
            "fingerprint": vec![7u8; 32],
        }
    }))
    .unwrap()
    .len();

    // Send boundary: exact open_len succeeds; open_len - 1 and open_len - 2 fail before I/O.
    for (name, frame_bytes, should_succeed) in [
        ("send-exact", open_len, true),
        ("send-one-short", open_len - 1, false),
        ("send-two-short", open_len - 2, false),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            if should_succeed {
                let opening = receive(&mut stream);
                assert_eq!(opening["Open"]["context"], 2);
                send(
                    &mut stream,
                    json!({"Ready": {"start": 0, "end": 1, "width": 4}}),
                );
                assert_eq!(receive(&mut stream), json!("Close"));
                send(&mut stream, json!("Ack"));
            } else {
                let mut byte = [0_u8; 1];
                assert_eq!(
                    stream.read(&mut byte).unwrap(),
                    0,
                    "oversized outbound frame must not write to socket: {name}"
                );
            }
        });
        let result = LayerClient::connect(
            address,
            2,
            [7; 32],
            RemoteLimits {
                frame_bytes,
                timeout: Duration::from_secs(2),
                ..Default::default()
            },
        );
        if should_succeed {
            result.expect(name).close().unwrap();
        } else {
            let err = result
                .err()
                .expect("outbound frame exceeding limit must fail");
            assert!(
                err.to_string().contains("frame size exceeds limit"),
                "{name}: {err}"
            );
        }
        server.join().unwrap();
    }

    // Receive boundary: zero-length prefix rejected; exact limit succeeds;
    // limit + 1 and limit + 2 rejected.
    let recv_limit = open_len + 64;
    let base_ready =
        serde_json::to_vec(&json!({"Ready": {"start": 0, "end": 1, "width": 4}})).unwrap();
    assert!(base_ready.len() < recv_limit);
    let padded_ready = |target_len: usize| -> Vec<u8> {
        let mut bytes = base_ready.clone();
        bytes.resize(target_len, b' ');
        bytes
    };

    for (name, reply_bytes, should_succeed) in [
        ("recv-zero", Vec::new(), false),
        ("recv-exact", padded_ready(recv_limit), true),
        ("recv-plus-one", padded_ready(recv_limit + 1), false),
        ("recv-plus-two", padded_ready(recv_limit + 2), false),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let opening = receive(&mut stream);
            assert_eq!(opening["Open"]["context"], 2);
            stream
                .write_all(&(reply_bytes.len() as u32).to_le_bytes())
                .unwrap();
            if !reply_bytes.is_empty() {
                let _ = stream.write_all(&reply_bytes);
            }
            if should_succeed {
                assert_eq!(receive(&mut stream), json!("Close"));
                send(&mut stream, json!("Ack"));
            }
        });
        let result = LayerClient::connect(
            address,
            2,
            [7; 32],
            RemoteLimits {
                frame_bytes: recv_limit,
                timeout: Duration::from_secs(2),
                ..Default::default()
            },
        );
        if should_succeed {
            result.expect(name).close().unwrap();
        } else {
            let err = result.err().expect("invalid inbound frame size must fail");
            assert!(
                err.to_string().contains("frame size exceeds limit"),
                "{name}: {err}"
            );
        }
        server.join().unwrap();
    }
}

#[test]
fn remote_pipeline_rejects_empty_addresses_zero_context_and_excess_layer_end() {
    struct TempFile(PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "bml-remote-pipeline-contract-{}-{}.gguf",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let _guard = TempFile(path.clone());
    std::fs::write(&path, gguf_fixture::fixture()).unwrap();
    let reader = GgufReader::open(&path).unwrap();
    let model = Model::from_gguf(&reader).unwrap();
    assert_eq!(model.config().block_count, 1);
    assert_eq!(model.config().embedding_length, 32);

    let limits = RemoteLimits {
        frame_bytes: 4096,
        timeout: Duration::from_secs(2),
        ..Default::default()
    };

    // 1. Empty endpoint list alone (valid context) must fail locally.
    let empty_err = RemotePipeline::connect(&model, &[], 4, limits)
        .err()
        .expect("empty remote endpoints must fail");
    assert_eq!(
        empty_err.to_string(),
        "invalid remote pipeline endpoints/context"
    );

    // 2. Zero context alone (non-empty endpoint list) must fail locally before I/O.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let done_peer = Arc::clone(&done);
    let peer = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !done_peer.load(Ordering::Acquire) {
            match listener.accept() {
                Ok(_) => panic!("zero context must not connect to peer"),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept failed: {error}"),
            }
        }
    });
    let zero_ctx_err = RemotePipeline::connect(&model, &[address], 0, limits)
        .err()
        .expect("zero context must fail");
    done.store(true, Ordering::Release);
    peer.join().unwrap();
    assert_eq!(
        zero_ctx_err.to_string(),
        "invalid remote pipeline endpoints/context"
    );

    // 3. Permissive peer advertising end > block_count (0..2 when block_count == 1)
    // must be rejected even though start == 0 and width == embedding_length.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let peer = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let opening = receive(&mut stream);
        assert_eq!(opening["Open"]["context"], 4);
        send(
            &mut stream,
            json!({"Ready": {"start": 0, "end": 2, "width": 32}}),
        );
        let mut byte = [0_u8; 1];
        let _ = stream.read(&mut byte);
    });
    let excess_end_err = RemotePipeline::connect(&model, &[address], 4, limits)
        .err()
        .expect("layer range past block_count must fail");
    assert_eq!(
        excess_end_err.to_string(),
        "remote layers must cover the model once in order"
    );
    peer.join().unwrap();
}
