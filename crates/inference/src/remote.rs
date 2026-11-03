//! Bounded synchronous TCP transport for causal layer sessions.
//!
//! Each connection owns one session. This protocol is distinct from the legacy
//! fragment worker protocol. A failed transport makes commit status uncertain:
//! callers must discard the pipeline instead of blindly retrying a token.
use crate::model::Model;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::ops::Range;
use std::time::Duration;

/// Resource and timeout policy for one connected layer session.
#[derive(Clone, Copy, Debug)]
pub struct RemoteLimits {
    /// Maximum encoded JSON frame, excluding the four-byte length prefix.
    pub frame_bytes: usize,
    /// Maximum token context admitted by the worker.
    pub context: usize,
    /// Maximum numerical session payload (KV and transient buffers).
    pub memory_bytes: usize,
    /// Socket read/write timeout; network scheduling is not real-time.
    pub timeout: Duration,
}
impl Default for RemoteLimits {
    fn default() -> Self {
        Self {
            frame_bytes: 1024 * 1024,
            context: 4096,
            memory_bytes: crate::model::DEFAULT_SESSION_MEMORY_LIMIT,
            timeout: Duration::from_secs(30),
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Request {
    Open {
        version: u32,
        context: usize,
        fingerprint: [u8; 32],
    },
    Forward {
        position: usize,
        hidden: Vec<f32>,
    },
    Reset,
    Close,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Response {
    Ready {
        start: usize,
        end: usize,
        width: usize,
    },
    Output {
        position: usize,
        hidden: Vec<f32>,
    },
    Ack,
    Rejected(String),
}
fn setup(stream: &TcpStream, limits: RemoteLimits) -> Result<()> {
    if limits.frame_bytes == 0 || limits.frame_bytes > u32::MAX as usize || limits.timeout.is_zero()
    {
        return Err(Error("invalid transport limits".into()));
    }
    stream.set_read_timeout(Some(limits.timeout))?;
    stream.set_write_timeout(Some(limits.timeout))?;
    stream.set_nodelay(true)?;
    Ok(())
}
fn receive<T: serde::de::DeserializeOwned>(stream: &mut TcpStream, limit: usize) -> Result<T> {
    let mut prefix = [0; 4];
    stream.read_exact(&mut prefix)?;
    let count = u32::from_le_bytes(prefix) as usize;
    if count == 0 || count > limit {
        return Err(Error("remote frame size exceeds limit".into()));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(count)
        .map_err(|_| Error("remote frame allocation failed".into()))?;
    bytes.resize(count, 0);
    stream.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}
fn send<T: Serialize>(stream: &mut TcpStream, value: &T, limit: usize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > limit {
        return Err(Error("remote frame size exceeds limit".into()));
    }
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

/// Serve one dedicated connection using the validated local layer kernels.
/// The owner selects the model and range; there is no remote model upload or
/// filesystem path access. Different connections receive independent KV state.
/// Opening checks the expected numerical model fingerprint before allocating KV.
pub fn serve_layer_connection(
    model: &Model<'_>,
    layers: Range<usize>,
    mut stream: TcpStream,
    limits: RemoteLimits,
) -> Result<()> {
    setup(&stream, limits)?;
    let context = match receive(&mut stream, limits.frame_bytes)? {
        Request::Open {
            version: 2,
            context,
            fingerprint,
        } if context <= limits.context && fingerprint == model.fingerprint() => context,
        _ => {
            send(
                &mut stream,
                &Response::Rejected("invalid open/version/context/model identity".into()),
                limits.frame_bytes,
            )?;
            return Ok(());
        }
    };
    let mut session = match model.layer_session(layers.clone(), context, limits.memory_bytes) {
        Ok(session) => session,
        Err(error) => {
            send(
                &mut stream,
                &Response::Rejected(error.to_string()),
                limits.frame_bytes,
            )?;
            return Ok(());
        }
    };
    send(
        &mut stream,
        &Response::Ready {
            start: layers.start,
            end: layers.end,
            width: model.config().embedding_length,
        },
        limits.frame_bytes,
    )?;
    loop {
        let response = match receive(&mut stream, limits.frame_bytes)? {
            Request::Forward { position, hidden } => match session.forward(position, &hidden) {
                Ok(hidden) => Response::Output { position, hidden },
                Err(error) => Response::Rejected(error.to_string()),
            },
            Request::Reset => {
                session.reset();
                Response::Ack
            }
            Request::Close => {
                send(&mut stream, &Response::Ack, limits.frame_bytes)?;
                return Ok(());
            }
            Request::Open { .. } => Response::Rejected("session already open".into()),
        };
        send(&mut stream, &response, limits.frame_bytes)?;
    }
}

/// Client for one causal layer session. There is no automatic retry or reconnect.
pub struct LayerClient {
    stream: TcpStream,
    limits: RemoteLimits,
    layers: Range<usize>,
    width: usize,
    healthy: bool,
}
impl LayerClient {
    /// Connect and admit one context. Caller must verify returned range against
    /// the planned pipeline; the handshake verifies the expected numerical model identity.
    pub fn connect(
        address: impl ToSocketAddrs,
        context: usize,
        expected_fingerprint: [u8; 32],
        limits: RemoteLimits,
    ) -> Result<Self> {
        let stream = TcpStream::connect(address)?;
        setup(&stream, limits)?;
        let mut client = Self {
            stream,
            limits,
            layers: 0..0,
            width: 0,
            healthy: true,
        };
        match client.exchange(Request::Open {
            version: 2,
            context,
            fingerprint: expected_fingerprint,
        })? {
            Response::Ready { start, end, width } if start < end && width > 0 => {
                client.layers = start..end;
                client.width = width;
                Ok(client)
            }
            Response::Rejected(message) => Err(Error(message)),
            _ => Err(Error("invalid session opening response".into())),
        }
    }
    fn exchange(&mut self, request: Request) -> Result<Response> {
        if !self.healthy {
            return Err(Error(
                "remote session has uncertain state; discard pipeline".into(),
            ));
        }
        let result = send(&mut self.stream, &request, self.limits.frame_bytes)
            .and_then(|()| receive(&mut self.stream, self.limits.frame_bytes));
        if result.is_err() {
            self.healthy = false;
            let _ = self.stream.shutdown(Shutdown::Both);
        }
        result
    }
    fn unexpected(&mut self) -> Error {
        self.healthy = false;
        let _ = self.stream.shutdown(Shutdown::Both);
        Error("invalid remote response; discard pipeline".into())
    }
    /// Fixed layer range reported by the worker.
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }
    /// Execute one hidden state at its explicit causal position.
    pub fn forward(&mut self, position: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.width || hidden.iter().any(|x| !x.is_finite()) {
            return Err(Error("invalid remote hidden state".into()));
        }
        match self.exchange(Request::Forward {
            position,
            hidden: hidden.to_vec(),
        })? {
            Response::Output {
                position: returned,
                hidden,
            } if returned == position
                && hidden.len() == self.width
                && hidden.iter().all(|x| x.is_finite()) =>
            {
                Ok(hidden)
            }
            Response::Rejected(message) => Err(Error(message)),
            _ => Err(self.unexpected()),
        }
    }
    /// Discard this stage's context; callers must reset every stage together.
    pub fn reset(&mut self) -> Result<()> {
        match self.exchange(Request::Reset)? {
            Response::Ack => Ok(()),
            _ => Err(self.unexpected()),
        }
    }
    /// Explicit orderly shutdown. Socket errors still make delivery uncertain.
    pub fn close(mut self) -> Result<()> {
        match self.exchange(Request::Close)? {
            Response::Ack => Ok(()),
            _ => Err(self.unexpected()),
        }
    }
}

/// Coordinator for one request across ordered, non-overlapping remote stages.
/// The coordinator retains the validated model for embeddings/output projection;
/// this is distributed computation, not model-memory sharding. A stage failure
/// poisons the complete pipeline because earlier stages may have committed.
pub struct RemotePipeline<'a> {
    model: &'a Model<'a>,
    stages: Vec<LayerClient>,
    position: usize,
    context: usize,
    healthy: bool,
}
impl<'a> RemotePipeline<'a> {
    /// Open dedicated sessions and require exact coverage of every model layer.
    /// Endpoints are ordered by layer range, not load-balanced replicas.
    pub fn connect(
        model: &'a Model<'a>,
        addresses: &[std::net::SocketAddr],
        context: usize,
        limits: RemoteLimits,
    ) -> Result<Self> {
        if addresses.is_empty() || context == 0 || context > model.config().context_length {
            return Err(Error("invalid remote pipeline endpoints/context".into()));
        }
        let mut stages = Vec::new();
        let mut next = 0;
        for address in addresses {
            let client = LayerClient::connect(address, context, model.fingerprint(), limits)?;
            if client.layers.start != next
                || client.layers.end > model.config().block_count
                || client.width != model.config().embedding_length
            {
                return Err(Error(
                    "remote layers must cover the model once in order".into(),
                ));
            }
            next = client.layers.end;
            stages.push(client);
        }
        if next != model.config().block_count {
            return Err(Error(
                "remote pipeline does not cover every model layer".into(),
            ));
        }
        Ok(Self {
            model,
            stages,
            position: 0,
            context,
            healthy: true,
        })
    }
    fn poison(&mut self) {
        self.healthy = false;
        for stage in &mut self.stages {
            stage.healthy = false;
            let _ = stage.stream.shutdown(Shutdown::Both);
        }
    }
    fn check_health(&self) -> Result<()> {
        if self.healthy {
            Ok(())
        } else {
            Err(Error(
                "remote pipeline is invalid; create a new session".into(),
            ))
        }
    }
    /// Append a token and return logits. Invalid token/context checks occur
    /// before network mutation. A numerical/transport failure invalidates all
    /// stages, rather than pretending to roll back already committed workers.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        self.check_health()?;
        if self.position >= self.context {
            return Err(Error("remote context exhausted".into()));
        }
        let mut hidden = self.model.embed_token(token)?;
        for stage in &mut self.stages {
            match stage.forward(self.position, &hidden) {
                Ok(next) => hidden = next,
                Err(error) => {
                    self.poison();
                    return Err(error);
                }
            }
        }
        match self.model.final_logits(&hidden) {
            Ok(logits) => {
                self.position += 1;
                Ok(logits)
            }
            Err(error) => {
                self.poison();
                Err(error)
            }
        }
    }
    /// Validate all prompt IDs/capacity before appending. A mid-prompt execution
    /// failure invalidates the pipeline; distributed rollback is not provided.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.check_health()?;
        if tokens.is_empty()
            || tokens.len() > self.context.saturating_sub(self.position)
            || tokens
                .iter()
                .any(|&id| id as usize >= self.model.config().vocab_size)
        {
            return Err(Error("invalid remote prompt or context overflow".into()));
        }
        let mut logits = Vec::new();
        for &token in tokens {
            logits = self.decode(token)?;
        }
        Ok(logits)
    }
    /// Number of tokens confirmed by every stage and output projection.
    pub fn position(&self) -> usize {
        self.position
    }
    /// Reset all healthy stages. Any partial reset failure invalidates the pipeline.
    pub fn reset(&mut self) -> Result<()> {
        self.check_health()?;
        for stage in &mut self.stages {
            if let Err(error) = stage.reset() {
                self.poison();
                return Err(error);
            }
        }
        self.position = 0;
        Ok(())
    }
    /// Close every stage, preserving the first error without skipping cleanup.
    pub fn close(self) -> Result<()> {
        let mut first_error = None;
        for stage in self.stages {
            if let Err(error) = stage.close() {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn lost_response_poisoning_prevents_an_ambiguous_retry() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let limits = RemoteLimits::default();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _: Request = receive(&mut stream, limits.frame_bytes).unwrap();
            send(
                &mut stream,
                &Response::Ready {
                    start: 0,
                    end: 1,
                    width: 2,
                },
                limits.frame_bytes,
            )
            .unwrap();
            let request: Request = receive(&mut stream, limits.frame_bytes).unwrap();
            assert!(matches!(request, Request::Forward { position: 0, .. }));
            // The server may have committed this token, but its response is lost.
            stream.shutdown(Shutdown::Both).unwrap();
        });
        let mut client = LayerClient::connect(address, 2, [0; 32], limits).unwrap();
        assert!(client.forward(0, &[1., 2.]).is_err());
        assert!(client
            .forward(0, &[1., 2.])
            .unwrap_err()
            .to_string()
            .contains("uncertain state"));
        server.join().unwrap();
    }
}
