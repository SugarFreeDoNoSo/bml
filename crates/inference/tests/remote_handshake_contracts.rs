use bml_inference::remote::{LayerClient, RemoteLimits};
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
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
