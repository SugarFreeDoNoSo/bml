//! Protocolo de comunicación interna entre nodos BML.
//!
//! Usa TCP raw con framing simple: `[u32 msg_type][u32 payload_len][payload]`.
//! Sin protobuf, sin gRPC, sin deps externas. Solo `std::net`.
//!
//! # Mensajes
//!
//! | msg_type | Nombre | Payload |
//! |---|---|---|
//! | 0 | `ExecuteFragment` | fragmento `.bmlgraph` serializado |
//! | 1 | `ReportResult` | resultado f64 + metadata |
//! | 2 | `StealWork` | vacío (request) / fragmento (response) |
//! | 3 | `HealthCheck` | vacío (request) / status (response) |
//! | 4 | `BatchRequest` | lista de prompts |
//! | 5 | `BatchResult` | lista de tokens generados |

use std::io::{self, Read, Write};
use std::net::TcpStream;

/// Tipos de mensaje del protocolo BML.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MsgType {
    /// Ejecuta un fragmento `.bmlgraph`.
    ExecuteFragment = 0,
    /// Reporta el resultado de una ejecución.
    ReportResult = 1,
    /// Roba trabajo de otro nodo.
    StealWork = 2,
    /// Verifica que un nodo está vivo.
    HealthCheck = 3,
    /// Envía un batch de prompts a procesar.
    BatchRequest = 4,
    /// Reporta los tokens generados de un batch.
    BatchResult = 5,
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
pub fn send_msg(stream: &mut TcpStream, msg: &Message) -> io::Result<()> {
    let bytes = msg.to_bytes();
    stream.write_all(&bytes)?;
    Ok(())
}

/// Recibe un mensaje de un TcpStream.
///
/// Lee el header (8 bytes: msg_type + payload_len) y luego el payload.
pub fn recv_msg(stream: &mut TcpStream) -> io::Result<Message> {
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

    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        stream.read_exact(&mut payload)?;
    }

    Ok(Message { msg_type, payload })
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

    /// Envía un fragmento a ejecutar.
    pub fn send_fragment(&mut self, fragment_bytes: &[u8]) -> io::Result<()> {
        let msg = Message::new(MsgType::ExecuteFragment, fragment_bytes.to_vec());
        send_msg(&mut self.stream, &msg)
    }

    /// Recibe el resultado de una ejecución.
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

    /// Roba trabajo del nodo remoto.
    pub fn steal_work(&mut self) -> io::Result<Option<Vec<u8>>> {
        let msg = Message::new(MsgType::StealWork, vec![]);
        send_msg(&mut self.stream, &msg)?;
        let response = recv_msg(&mut self.stream)?;
        if response.payload.is_empty() {
            Ok(None)
        } else {
            Ok(Some(response.payload))
        }
    }

    /// Health check.
    pub fn health_check(&mut self) -> io::Result<bool> {
        let msg = Message::new(MsgType::HealthCheck, vec![]);
        send_msg(&mut self.stream, &msg)?;
        let response = recv_msg(&mut self.stream)?;
        Ok(!response.payload.is_empty())
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
            let msg = recv_msg(&mut stream).unwrap();
            assert_eq!(msg.msg_type, MsgType::HealthCheck);
            let response = Message::new(MsgType::HealthCheck, vec![1]); // alive
            send_msg(&mut stream, &response).unwrap();
        });

        let mut node = NodeHandle::connect(&addr_str).unwrap();
        let alive = node.health_check().unwrap();
        assert!(alive);

        server_thread.join().unwrap();
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
        for i in 0..6 {
            let mt = MsgType::from_u32(i).unwrap();
            assert_eq!(mt as u32, i);
        }
        assert!(MsgType::from_u32(99).is_none());
    }
}
