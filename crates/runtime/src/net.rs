//! Protocolo de comunicación interna entre nodos BML.
//!
//! Usa TCP raw con framing simple: `[u32 msg_type][u32 payload_len][payload]`.
//! Sin protobuf, sin gRPC, sin deps externas. Solo `std::net`.
//!
//! # Mensajes
//!
//! | msg_type | Nombre | Payload |
//! |---|---|---|
//! | 0 | `ExecuteFragment` | admisión de un fragmento `.bmlgraph` serializado |
//! | 1 | `ReportResult` | respuesta f64 legacy; el worker confirma admisión, no inferencia |
//! | 2 | `StealWork` | vacío (request) / fragmento (response) |
//! | 3 | `HealthCheck` | vacío (request) / status (response) |
//! | 4 | `BatchRequest` | solicitud legacy retirada; migrar al transporte simbólico causal |
//! | 5 | `BatchResult` | vacío: rechazo de la inferencia legacy retirada |
//! | 8 | `EnqueueFragment` | fragmento pendiente serializado |
//! | 9 | `EnqueueResult` | byte: 1 aceptado, 0 cola llena, 2 inválido |

use std::io::{self, Read, Write};
use std::net::TcpStream;

/// Tipos de mensaje del protocolo BML.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MsgType {
    /// Legacy model-shard upload; its response acknowledges loading only.
    ExecuteFragment = 0,
    /// Legacy f64 response; the worker uses it to acknowledge admission only.
    ReportResult = 1,
    /// Roba trabajo de otro nodo.
    StealWork = 2,
    /// Verifica que un nodo está vivo.
    HealthCheck = 3,
    /// Retired legacy hidden-state request; the worker explicitly rejects it.
    BatchRequest = 4,
    /// Legacy response tag; the worker returns an empty rejection payload.
    BatchResult = 5,
    /// Ejecuta un VectorFragment (matmul de columnas).
    VectorMatmul = 6,
    /// Reporta el resultado de un VectorFragment.
    VectorResult = 7,
    /// Enqueue a serialized fragment as pending, stealable work.
    EnqueueFragment = 8,
    /// Admission status: 1 accepted, 0 full, 2 invalid.
    EnqueueResult = 9,
    /// Execute a bounded compiled graph with inputs and an initial result buffer.
    ExecuteGraph = 10,
    /// Versioned graph result or explicit validation failure.
    GraphResult = 11,
}

impl MsgType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::ExecuteFragment),
            1 => Some(Self::ReportResult),
            2 => Some(Self::StealWork),
            3 => Some(Self::HealthCheck),
            4 => Some(Self::BatchRequest),
            5 => Some(Self::BatchResult),
            6 => Some(Self::VectorMatmul),
            7 => Some(Self::VectorResult),
            8 => Some(Self::EnqueueFragment),
            9 => Some(Self::EnqueueResult),
            10 => Some(Self::ExecuteGraph),
            11 => Some(Self::GraphResult),
            _ => None,
        }
    }
}

/// Un mensaje del protocolo BML.
#[derive(Debug, Clone)]
pub struct Message {
    pub msg_type: MsgType,
    pub payload: Vec<u8>,
}

impl Message {
    pub fn new(msg_type: MsgType, payload: Vec<u8>) -> Self {
        Self { msg_type, payload }
    }

    /// Serializa el mensaje a bytes con framing.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + self.payload.len());
        bytes.extend_from_slice(&(self.msg_type as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }
}

/// Envía un mensaje por un TcpStream.
#[inline(never)]
pub fn send_msg(stream: &mut TcpStream, msg: &Message) -> io::Result<()> {
    u32::try_from(msg.payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "message exceeds u32 framing"))?;
    let bytes = msg.to_bytes();
    stream.write_all(&bytes)?;
    Ok(())
}

/// Recibe un mensaje de un TcpStream.
///
/// Lee el header (8 bytes: msg_type + payload_len) y luego el payload.
pub fn recv_msg(stream: &mut TcpStream) -> io::Result<Message> {
    recv_msg_with_limit(stream, DEFAULT_MAX_PAYLOAD_BYTES)
}

/// Default maximum payload accepted by worker connections (64 MiB).
pub const DEFAULT_MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

/// Receive one frame, rejecting its declared size before reserving memory.
/// Callers transferring larger model shards must supply an explicit budget.
/// On failure the connection must be closed: an unread body may remain.
pub fn recv_msg_with_limit(
    stream: &mut TcpStream,
    max_payload_bytes: usize,
) -> io::Result<Message> {
    let mut header = [0u8; 8];
    stream.read_exact(&mut header)?;
    let msg_type_raw = u32::from_le_bytes(header[0..4].try_into().unwrap());
    let payload_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    let msg_type = MsgType::from_u32(msg_type_raw).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("msg_type desconocido: {msg_type_raw}"),
        )
    })?;

    let max_payload_bytes = if matches!(msg_type, MsgType::ExecuteGraph | MsgType::GraphResult) {
        max_payload_bytes.min(crate::graph_rpc::MAX_WIRE_BYTES)
    } else {
        max_payload_bytes
    };
    if payload_len > max_payload_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "message payload budget exceeded",
        ));
    }
    let mut payload = Vec::new();
    payload.try_reserve_exact(payload_len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::OutOfMemory,
            "message payload allocation failed",
        )
    })?;
    payload.resize(payload_len, 0);
    if payload_len > 0 {
        stream.read_exact(&mut payload)?;
    }

    Ok(Message { msg_type, payload })
}

/// Remote worker health snapshot. Queue length may change immediately after sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeHealth {
    /// Whether the worker reports itself alive.
    pub alive: bool,
    /// Pending work count; absent for legacy one-byte responses.
    pub pending: Option<u64>,
}
impl NodeHealth {
    fn decode(payload: &[u8]) -> io::Result<Self> {
        if !matches!(payload.len(), 1 | 9) || payload[0] > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid health payload",
            ));
        }
        Ok(Self {
            alive: payload[0] == 1,
            pending: if payload.len() == 9 {
                Some(u64::from_le_bytes(payload[1..].try_into().unwrap()))
            } else {
                None
            },
        })
    }
}

/// Handle a un nodo remoto via TCP.
pub struct NodeHandle {
    stream: TcpStream,
}

impl NodeHandle {
    /// Conecta a un nodo en la dirección dada.
    pub fn connect(addr: &str) -> io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    /// Submit a legacy fragment for admission, without executing inference.
    pub fn send_fragment(&mut self, fragment_bytes: &[u8]) -> io::Result<()> {
        let msg = Message::new(MsgType::ExecuteFragment, fragment_bytes.to_vec());
        send_msg(&mut self.stream, &msg)
    }

    /// Read the legacy f64 response; a worker upload acknowledgement is not inference.
    pub fn recv_result(&mut self) -> io::Result<f64> {
        let msg = recv_msg(&mut self.stream)?;
        if msg.msg_type != MsgType::ReportResult {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "esperaba ReportResult",
            ));
        }
        if msg.payload.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "payload demasiado pequeño",
            ));
        }
        Ok(f64::from_le_bytes(msg.payload[0..8].try_into().unwrap()))
    }

    /// Submit pending work; false indicates queue saturation. An I/O error
    /// leaves admission uncertain, so automatic retries can duplicate work.
    pub fn enqueue_fragment(&mut self, bytes: &[u8]) -> io::Result<bool> {
        send_msg(
            &mut self.stream,
            &Message::new(MsgType::EnqueueFragment, bytes.to_vec()),
        )?;
        let response = recv_msg(&mut self.stream)?;
        if response.msg_type != MsgType::EnqueueResult {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected EnqueueResult",
            ));
        }
        match response.payload.as_slice() {
            [1] => Ok(true),
            [0] => Ok(false),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "fragment rejected or invalid admission response",
            )),
        }
    }
    /// Roba trabajo del nodo remoto.
    pub fn steal_work(&mut self) -> io::Result<Option<Vec<u8>>> {
        let msg = Message::new(MsgType::StealWork, vec![]);
        send_msg(&mut self.stream, &msg)?;
        let response = recv_msg(&mut self.stream)?;
        if response.msg_type != MsgType::StealWork {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected StealWork response",
            ));
        }
        if response.payload.is_empty() {
            Ok(None)
        } else {
            Ok(Some(response.payload))
        }
    }

    /// Health check, compatible with legacy workers without queue counts.
    pub fn health_check(&mut self) -> io::Result<bool> {
        Ok(self.health_status()?.alive)
    }
    /// Fetch liveness and a snapshot of queued work.
    pub fn health_status(&mut self) -> io::Result<NodeHealth> {
        let msg = Message::new(MsgType::HealthCheck, vec![]);
        send_msg(&mut self.stream, &msg)?;
        let response = recv_msg(&mut self.stream)?;
        if response.msg_type != MsgType::HealthCheck {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected HealthCheck response",
            ));
        }
        NodeHealth::decode(&response.payload)
    }

    /// Acceso mutable al stream subyacente.
    pub fn stream_mut(&mut self) -> &mut TcpStream {
        &mut self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    #[test]
    fn message_serialization() {
        let msg = Message::new(MsgType::ExecuteFragment, vec![1, 2, 3, 4]);
        let bytes = msg.to_bytes();
        assert_eq!(bytes.len(), 12); // 4 (type) + 4 (len) + 4 (payload)
        let msg_type = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let payload_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(msg_type, 0);
        assert_eq!(payload_len, 4);
        assert_eq!(&bytes[8..], &[1, 2, 3, 4]);
    }

    #[test]
    fn send_recv_message() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server_thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let msg = recv_msg(&mut stream).unwrap();
            assert_eq!(msg.msg_type, MsgType::ExecuteFragment);
            assert_eq!(msg.payload, vec![1, 2, 3]);

            // Responder
            let response = Message::new(MsgType::ReportResult, 42.0_f64.to_le_bytes().to_vec());
            send_msg(&mut stream, &response).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();
        let msg = Message::new(MsgType::ExecuteFragment, vec![1, 2, 3]);
        send_msg(&mut client, &msg).unwrap();

        let response = recv_msg(&mut client).unwrap();
        assert_eq!(response.msg_type, MsgType::ReportResult);
        let result = f64::from_le_bytes(response.payload[0..8].try_into().unwrap());
        assert!((result - 42.0).abs() < 1e-12);

        server_thread.join().unwrap();
    }

    #[test]
    fn node_handle_connect_and_health() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let addr_str = format!("127.0.0.1:{}", addr.port());

        let server_thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = recv_msg(&mut stream);
            if request.is_ok() {
                let response = Message::new(MsgType::HealthCheck, vec![1]); // alive
                send_msg(&mut stream, &response).unwrap();
            }
            request
        });

        let mut node = NodeHandle::connect(&addr_str).unwrap();
        let health = node.health_check();
        // A missing request must become observable EOF at the peer, rather
        // than joining a reader while the client still owns the open socket.
        drop(node);
        let request = server_thread
            .join()
            .unwrap()
            .expect("health_check must transmit a request before EOF");
        assert_eq!(request.msg_type, MsgType::HealthCheck);
        assert!(request.payload.is_empty());
        assert!(health.unwrap());
    }

    #[test]
    fn node_handle_send_fragment_recv_result() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let addr_str = format!("127.0.0.1:{}", addr.port());

        let server_thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let msg = recv_msg(&mut stream).unwrap();
            assert_eq!(msg.msg_type, MsgType::ExecuteFragment);
            assert!(!msg.payload.is_empty());

            // Responder con un resultado
            let result = 1.23456_f64;
            let response = Message::new(MsgType::ReportResult, result.to_le_bytes().to_vec());
            send_msg(&mut stream, &response).unwrap();
        });

        let mut node = NodeHandle::connect(&addr_str).unwrap();
        node.send_fragment(&[0, 1, 2, 3]).unwrap();
        let result = node.recv_result().unwrap();
        assert!((result - 1.23456).abs() < 1e-5);

        server_thread.join().unwrap();
    }

    #[test]
    fn node_handle_steal_work() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let addr_str = format!("127.0.0.1:{}", addr.port());

        let server_thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let msg = recv_msg(&mut stream).unwrap();
            assert_eq!(msg.msg_type, MsgType::StealWork);
            // Responder con un fragmento robado
            let response = Message::new(MsgType::StealWork, vec![10, 20, 30]);
            send_msg(&mut stream, &response).unwrap();
        });

        let mut node = NodeHandle::connect(&addr_str).unwrap();
        let stolen = node.steal_work().unwrap();
        assert!(stolen.is_some());
        assert_eq!(stolen.unwrap(), vec![10, 20, 30]);

        server_thread.join().unwrap();
    }

    #[test]
    fn msg_type_roundtrip() {
        for i in 0..12 {
            let mt = MsgType::from_u32(i).unwrap();
            assert_eq!(mt as u32, i);
        }
        assert!(MsgType::from_u32(99).is_none());
    }
    #[test]
    fn node_rejects_wrong_response_types() {
        for request in [MsgType::HealthCheck, MsgType::StealWork] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                assert_eq!(recv_msg(&mut stream).unwrap().msg_type, request);
                send_msg(&mut stream, &Message::new(MsgType::ReportResult, vec![1])).unwrap();
            });
            let mut node = NodeHandle::connect(&address.to_string()).unwrap();
            let error = match request {
                MsgType::HealthCheck => node.health_check().unwrap_err(),
                _ => node.steal_work().unwrap_err(),
            };
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            server.join().unwrap();
        }
    }
    #[test]
    fn health_payload_accepts_legacy_and_counted_frames_only() {
        assert_eq!(
            NodeHealth::decode(&[1]).unwrap(),
            NodeHealth {
                alive: true,
                pending: None
            }
        );
        assert_eq!(
            NodeHealth::decode(&[0]).unwrap(),
            NodeHealth {
                alive: false,
                pending: None
            }
        );
        for count in [0, 64, u64::MAX] {
            let mut payload = vec![1];
            payload.extend_from_slice(&count.to_le_bytes());
            assert_eq!(NodeHealth::decode(&payload).unwrap().pending, Some(count));
        }
        for payload in [vec![], vec![2], vec![1, 0], vec![1; 8], vec![1; 10]] {
            assert_eq!(
                NodeHealth::decode(&payload).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
    }

    #[test]
    fn frame_budget_rejects_header_before_body_and_accepts_exact_limit() {
        for (budget, declared, body) in [
            (0, 0u32, Some(vec![])),
            (3, 3, Some(vec![1, 2, 3])),
            (3, 4, None),
            (DEFAULT_MAX_PAYLOAD_BYTES, u32::MAX, None),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                    .unwrap();
                recv_msg_with_limit(&mut stream, budget)
            });
            let mut client = TcpStream::connect(address).unwrap();
            client
                .write_all(&(MsgType::HealthCheck as u32).to_le_bytes())
                .unwrap();
            client.write_all(&declared.to_le_bytes()).unwrap();
            if let Some(bytes) = &body {
                client.write_all(bytes).unwrap();
            }
            // Keep the socket open: oversized requests must be rejected from
            // the header alone, without waiting for an absent body or EOF.
            let result = server.join().unwrap();
            if let Some(bytes) = body {
                assert_eq!(result.unwrap().payload, bytes);
            } else {
                assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
            }
            drop(client);
        }
    }
}
