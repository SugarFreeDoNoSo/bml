use bml_inference::remote::{LayerClient, RemoteLimits};
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
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
