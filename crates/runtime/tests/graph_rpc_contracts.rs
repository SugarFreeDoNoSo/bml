use bml_compiler::{BmlGraph, Fragment, RpnOp};
use bml_runtime::{
    buffer::ResultBuffer,
    graph_rpc::{encode_request, handle_request},
};
fn fixture() -> (BmlGraph, ResultBuffer) {
    (
        BmlGraph {
            fragments: vec![Fragment {
                ops: vec![RpnOp::Var(0), RpnOp::Const(0), RpnOp::FMul],
            }],
            const_pool: vec![3.0],
            threshold: 32,
        },
        ResultBuffer::new(1, 1),
    )
}
fn rejected(reply: &[u8]) {
    assert_eq!(reply, &[1, 0, 0, 0, 1, 0, 0, 0]);
}
#[test]
fn graph_rpc_rejects_every_truncation_trailing_data_versions_and_large_counts() {
    let (g, b) = fixture();
    let request = encode_request(&g, &[2.0], &b).unwrap();
    for end in 0..request.len() {
        rejected(&handle_request(&request[..end]));
    }
    let mut trailing = request.clone();
    trailing.push(0);
    rejected(&handle_request(&trailing));
    for offset in [0, 4, 8, 12, 16] {
        let mut corrupt = request.clone();
        corrupt[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        rejected(&handle_request(&corrupt));
    }
    let response = handle_request(&request);
    assert_eq!(&response[..8], &[1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(f64::from_le_bytes(response[8..16].try_into().unwrap()), 6.0);
    assert_eq!(response.len(), 28);
}
#[test]
fn bounded_graph_decoder_rejects_declared_allocations_before_payload_reads() {
    let (g, _) = fixture();
    let bytes = g.serialize();
    assert!(BmlGraph::deserialize_with_limits(&bytes, 32, 0, 100, 100).is_err());
    assert!(BmlGraph::deserialize_with_limits(&bytes, 32, 1, 2, 100).is_err());
    assert!(BmlGraph::deserialize_with_limits(&bytes, 32, 1, 3, 0).is_err());
    assert!(BmlGraph::deserialize_with_limits(&bytes, 32, 1, 3, 1).is_ok());
    for offset in [8, 12] {
        let mut corrupt = bytes.clone();
        corrupt[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(BmlGraph::deserialize_with_limits(&corrupt, 32, 100, 100, 100).is_err());
    }
}

#[test]
fn graph_encoder_accepts_exact_value_quota_and_rejects_one_more_input() {
    let quota = bml_runtime::graph::GraphLimits::default().max_values;
    assert!(
        (4..=1_000_000).contains(&quota),
        "keep the fixture allocation bounded"
    );
    let graph = BmlGraph {
        fragments: vec![Fragment {
            ops: vec![RpnOp::One],
        }],
        const_pool: vec![],
        threshold: 32,
    };
    let mut buffer = ResultBuffer::new(1, quota - 2);
    let first = 0x7ff8_0000_0000_0042_u64;
    buffer.write(0, 0, f64::from_bits(first));
    buffer.write(0, (quota - 3) as u32, -0.0);
    // This quota belongs to the encoder. Execution separately accounts for
    // constants and stack, so do not interpret this as server admission.
    let encoded = encode_request(&graph, &[2.0, 5.0], &buffer)
        .expect("the declared value quota is inclusive");
    assert_eq!(encoded.len(), 20 + 8 * quota + 25);
    let header: Vec<u32> = encoded[..20]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(header, [1, 2, 1, (quota - 2) as u32, 25]);
    assert_eq!(
        u64::from_le_bytes(encoded[36..44].try_into().unwrap()),
        first
    );
    let last = 20 + 8 * (quota - 1);
    assert_eq!(
        u64::from_le_bytes(encoded[last..last + 8].try_into().unwrap()),
        (-0.0_f64).to_bits()
    );
    assert_eq!(buffer.read(0, 0).to_bits(), first);
    assert_eq!(
        buffer.read(0, (quota - 3) as u32).to_bits(),
        (-0.0_f64).to_bits()
    );
    drop(encoded);
    let error = encode_request(&graph, &[2.0, 5.0, 7.0], &buffer).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(error.to_string(), "graph value budget exceeded");
}

#[test]
fn graph_rpc_zero_and_two_inputs_execute_closed_sum_and_preserve_buffer_bits() {
    let stored = [
        (-0.0_f64).to_bits(),
        0x7ff8_0000_0000_0042,
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        1,
        42.5_f64.to_bits(),
    ];
    for (inputs, ops, constants) in [
        (
            vec![],
            vec![RpnOp::Const(0), RpnOp::Const(1), RpnOp::FAdd],
            vec![2.0, 5.0],
        ),
        (
            vec![2.0, 5.0],
            vec![RpnOp::Var(0), RpnOp::Var(1), RpnOp::FAdd],
            vec![],
        ),
    ] {
        let graph = BmlGraph {
            fragments: vec![Fragment { ops }],
            const_pool: constants,
            threshold: 32,
        };
        let buffer =
            ResultBuffer::from_vec(2, 3, stored.into_iter().map(f64::from_bits).collect()).unwrap();
        let request = encode_request(&graph, &inputs, &buffer).unwrap();
        assert_eq!(
            u32::from_le_bytes(request[4..8].try_into().unwrap()) as usize,
            inputs.len()
        );
        assert_eq!(&request[8..16], &[2, 0, 0, 0, 3, 0, 0, 0]);
        let response = handle_request(&request);
        assert_eq!(&response[..8], &[1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(response.len(), 20 + 8 * stored.len());
        assert_eq!(
            u64::from_le_bytes(response[8..16].try_into().unwrap()),
            7.0_f64.to_bits()
        );
        assert_eq!(
            u32::from_le_bytes(response[16..20].try_into().unwrap()) as usize,
            stored.len()
        );
        let returned: Vec<u64> = response[20..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| u64::from_le_bytes(*bytes))
            .collect();
        assert_eq!(returned, stored);
        assert_eq!(
            buffer
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            stored
        );
        // Change only the declared input cardinality, leaving the valid body
        // and graph intact. The fixed rejection cannot be mistaken for 7.
        let mut corrupt = request;
        corrupt[4..8].copy_from_slice(&((inputs.len() + 1) as u32).to_le_bytes());
        rejected(&handle_request(&corrupt));
    }
}

#[test]
fn graph_encoder_accepts_exact_wire_limit_and_rejects_one_more_opcode_byte() {
    use bml_runtime::graph_rpc::MAX_WIRE_BYTES;
    // Independent wire arithmetic: 20 request bytes, 24 graph-header bytes,
    // 99_996 nine-byte Loop opcodes and two declared f64 arrays.
    // The fixture stays around 50 MiB at peak, including temporary serialization.
    // This is an encoding boundary, not a claim of server execution admission:
    // ValidatedGraph has a separate combined-value budget.
    const OPERATIONS: usize = 99_996;
    const CONSTANTS: usize = 1_000_000;
    const INPUTS: usize = 984_651;
    let mut graph = BmlGraph {
        fragments: vec![Fragment {
            ops: vec![
                RpnOp::Loop {
                    count: 0,
                    body_len: 0
                };
                OPERATIONS
            ],
        }],
        const_pool: vec![1.0; CONSTANTS],
        threshold: 32,
    };
    let inputs = vec![2.0; INPUTS];
    let buffer = ResultBuffer::new(0, 0);
    assert_eq!(
        44 + 9 * graph.fragments[0].ops.len() + 8 * (graph.const_pool.len() + inputs.len()),
        MAX_WIRE_BYTES
    );
    let encoded = encode_request(&graph, &inputs, &buffer)
        .expect("a frame exactly at the wire limit must be encodable");
    assert_eq!(encoded.len(), MAX_WIRE_BYTES);
    let fields: Vec<u32> = encoded[..20]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(&fields[..4], &[1, INPUTS as u32, 0, 0]);
    assert_eq!(fields[4] as usize, MAX_WIRE_BYTES - 20 - 8 * INPUTS);
    drop(encoded);
    graph.fragments[0].ops.push(RpnOp::One); // One byte, still below the opcode count limit.
    let error = encode_request(&graph, &inputs, &buffer).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("graph frame budget exceeded"));
}

#[test]
fn graph_rpc_rejects_invalid_execution_and_preserves_ieee_bits() {
    let (mut g, b) = fixture();
    g.fragments[0].ops = vec![RpnOp::FAdd];
    rejected(&handle_request(&encode_request(&g, &[], &b).unwrap()));
    g.fragments[0].ops = vec![
        RpnOp::One,
        RpnOp::Loop {
            count: u32::MAX,
            body_len: 1,
        },
        RpnOp::Drop,
    ];
    rejected(&handle_request(&encode_request(&g, &[], &b).unwrap()));
    g.fragments[0].ops = vec![RpnOp::Var(0)];
    for value in [-0.0, f64::INFINITY, f64::from_bits(1)] {
        let response = handle_request(&encode_request(&g, &[value], &b).unwrap());
        assert_eq!(
            u64::from_le_bytes(response[8..16].try_into().unwrap()),
            value.to_bits()
        );
    }
}

#[test]
fn graph_frame_limit_is_enforced_before_body_even_with_larger_legacy_budget() {
    use bml_runtime::net::{recv_msg, MsgType};
    use std::{
        io::{ErrorKind, Write},
        net::{TcpListener, TcpStream},
        thread,
        time::Duration,
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        recv_msg(&mut stream).unwrap_err().kind()
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .write_all(&(MsgType::ExecuteGraph as u32).to_le_bytes())
        .unwrap();
    client
        .write_all(&((bml_runtime::graph_rpc::MAX_WIRE_BYTES + 1) as u32).to_le_bytes())
        .unwrap();
    assert_eq!(server.join().unwrap(), ErrorKind::InvalidData);
    drop(client);
}

#[test]
fn client_rejects_malformed_or_wrong_type_graph_replies() {
    use bml_runtime::net::{recv_msg, send_msg, Message, MsgType, NodeHandle};
    use std::{net::TcpListener, thread, time::Duration};
    let (g, b) = fixture();
    let valid = handle_request(&encode_request(&g, &[2.0], &b).unwrap());
    let mut cases: Vec<_> = (0..valid.len())
        .map(|end| (MsgType::GraphResult, valid[..end].to_vec()))
        .collect();
    cases.push((MsgType::ReportResult, valid.clone()));
    let mut wrong_version = valid.clone();
    wrong_version[0] = 2;
    cases.push((MsgType::GraphResult, wrong_version));
    let mut trailing = valid;
    trailing.push(0);
    cases.push((MsgType::GraphResult, trailing));
    for (kind, payload) in cases {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            assert_eq!(
                recv_msg(&mut stream).unwrap().msg_type,
                MsgType::ExecuteGraph
            );
            send_msg(&mut stream, &Message::new(kind, payload)).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            assert_eq!(std::io::Read::read(&mut stream, &mut [0u8; 1]).unwrap(), 0);
        });
        let mut node = NodeHandle::connect(&address.to_string()).unwrap();
        node.stream_mut()
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        assert!(node.execute_graph(&g, &[2.0], &b).is_err());
        server.join().unwrap();
    }
}

#[test]
fn completed_rpc_results_publish_all_fields_or_return_unchanged() {
    use bml_runtime::{append_only::AppendOnlyResults, graph_rpc::GraphResult};
    let limited = AppendOnlyResults::new(3).unwrap();
    limited.append(99.0).unwrap();
    let nan = f64::from_bits(0x7ff8_0000_0000_0123);
    let result = GraphResult {
        value: nan,
        buffer: vec![-0.0, 42.0],
    };
    let pointer = result.buffer.as_ptr();
    let retained = result.publish(&limited).unwrap_err();
    assert_eq!(retained.buffer.as_ptr(), pointer);
    assert_eq!(retained.value.to_bits(), nan.to_bits());
    assert_eq!(limited.reserved_count(), 1);
    assert_eq!(limited.get(0), Some(99.0));
    assert!(limited.get(1).is_none());
    let enough = AppendOnlyResults::new(3).unwrap();
    let published = retained.publish(&enough).unwrap();
    assert_eq!(published.value_index, 0);
    assert_eq!(published.buffer_indices, 1..3);
    assert_eq!(enough.get(0).unwrap().to_bits(), nan.to_bits());
    assert_eq!(enough.get(1).unwrap().to_bits(), (-0.0f64).to_bits());
    assert_eq!(enough.get(2), Some(42.0));
    assert_eq!(enough.append(1.0), Err(1.0));
}
