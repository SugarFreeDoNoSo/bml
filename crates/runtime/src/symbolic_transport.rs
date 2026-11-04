//! Versioned TCP transport for independent symbolic layer sessions.
//!
//! The server owns its artifact path and range; clients cannot choose files.
//! Frames and KV storage have explicit bounds, values are exact little-endian
//! f64, and any transport or framing failure invalidates the client. In
//! particular, an unanswered commit is ambiguous and is never retried here.
//! Artifact identity detects mismatches; this protocol does not authenticate
//! peers or encrypt traffic and requires an appropriately protected network.
use crate::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
use crate::symbolic_session::SymbolicLayerSession;
use bml_compiler::symbolic_artifact::checked_identity;
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Duration;

const VERSION: u32 = 1;
const OPEN: u8 = 0;
const PREPARE: u8 = 1;
const COMMIT: u8 = 2;
const ABORT: u8 = 3;
const RESET: u8 = 4;
const CLOSE: u8 = 5;

/// Per-connection resource and timeout limits.
#[derive(Clone, Copy, Debug)]
pub struct SymbolicTransportLimits {
    /// Maximum bytes in one frame, excluding its length prefix.
    pub frame_bytes: usize,
    /// Maximum admitted token context.
    pub context: u32,
    /// Maximum F32 KV payload per connection, for its assigned layer range.
    pub kv_bytes: usize,
    /// Socket read and write timeout.
    pub timeout: Duration,
    /// Limits for loading and executing persisted kernels.
    pub model: SymbolicLimits,
}
impl Default for SymbolicTransportLimits {
    fn default() -> Self {
        Self {
            frame_bytes: 16 * 1024 * 1024,
            context: 4096,
            kv_bytes: 512 * 1024 * 1024,
            timeout: Duration::from_secs(30),
            model: SymbolicLimits {
                max_nodes: 0,
                ..Default::default()
            },
        }
    }
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn setup(stream: &TcpStream, limits: SymbolicTransportLimits) -> io::Result<()> {
    if limits.frame_bytes < 81
        || limits.frame_bytes > u32::MAX as usize
        || limits.context == 0
        || limits.timeout.is_zero()
    {
        return Err(invalid("invalid symbolic transport limits"));
    }
    stream.set_read_timeout(Some(limits.timeout))?;
    stream.set_write_timeout(Some(limits.timeout))?;
    stream.set_nodelay(true)
}
fn read_frame(stream: &mut TcpStream, limit: usize) -> io::Result<Vec<u8>> {
    let mut prefix = [0; 4];
    stream.read_exact(&mut prefix)?;
    let len = u32::from_le_bytes(prefix) as usize;
    if len == 0 || len > limit {
        return Err(invalid("symbolic frame size"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "symbolic frame allocation"))?;
    bytes.resize(len, 0);
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn write_frame(stream: &mut TcpStream, bytes: &[u8], limit: usize) -> io::Result<()> {
    if bytes.is_empty() || bytes.len() > limit {
        return Err(invalid("symbolic response size"));
    }
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(bytes)
}
fn u32_at(bytes: &[u8], start: usize) -> io::Result<u32> {
    let value = bytes
        .get(start..start + 4)
        .ok_or_else(|| invalid("truncated symbolic request"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}
fn reject(message: &str, limit: usize) -> Vec<u8> {
    let mut out = vec![1];
    let mut end = message.len().min(limit.saturating_sub(1));
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    out.extend_from_slice(&message.as_bytes()[..end]);
    out
}
fn position_request(bytes: &[u8]) -> io::Result<u32> {
    if bytes.len() != 5 {
        return Err(invalid("invalid position request"));
    }
    u32_at(bytes, 1)
}
fn vectors(bytes: &[u8], width: usize) -> io::Result<(u32, Vec<f64>)> {
    if bytes.len() < 9 {
        return Err(invalid("truncated symbolic vector"));
    }
    let position = u32_at(bytes, 1)?;
    let count = u32_at(bytes, 5)? as usize;
    if count != width || count.checked_mul(8).and_then(|n| n.checked_add(9)) != Some(bytes.len()) {
        return Err(invalid("symbolic vector width"));
    }
    let values: Vec<_> = bytes[9..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_le_bytes(*chunk))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite symbolic input"));
    }
    Ok((position, values))
}

/// Server-owned artifact and assigned range. No network request can change them.
/// Artifacts and mapped source files must not be modified while this is active.
pub struct SymbolicLayerServer {
    directory: PathBuf,
    source_override: Option<PathBuf>,
    shard: Option<(PathBuf, String, u64)>,
    identity: String,
    layers: Range<u32>,
    width: usize,
    limits: SymbolicTransportLimits,
}
impl SymbolicLayerServer {
    /// Validate the entire persisted artifact and assign a nonempty layer range.
    pub fn new(
        directory: &Path,
        source_override: Option<&Path>,
        layers: Range<u32>,
        limits: SymbolicTransportLimits,
    ) -> Result<Self, String> {
        if limits.frame_bytes < 81
            || limits.frame_bytes > u32::MAX as usize
            || limits.context == 0
            || limits.timeout.is_zero()
        {
            return Err("invalid symbolic transport limits".into());
        }
        let identity = checked_identity(directory, source_override)?;
        let model = SymbolicCausalModel::open_artifact(directory, source_override, limits.model)?;
        if checked_identity(directory, source_override)? != identity {
            return Err("symbolic artifact changed during admission".into());
        }
        if model.profile() != bml_compiler::symbolic_artifact::BOUNDED_PROFILE {
            return Err("symbolic server requires the complete bounded profile".into());
        }
        if layers.start >= layers.end || layers.end > model.compiler().config().n_layers {
            return Err("invalid symbolic server layer range".into());
        }
        let width = model.compiler().config().n_embd as usize;
        if width
            .checked_mul(8)
            .and_then(|n| n.checked_add(9))
            .is_none_or(|n| n > limits.frame_bytes)
        {
            return Err("symbolic vector exceeds frame budget".into());
        }
        Ok(Self {
            directory: directory.into(),
            source_override: source_override.map(Path::to_owned),
            shard: None,
            identity,
            layers,
            width,
            limits,
        })
    }
    /// Configure authenticated shard execution; the GGUF is still required
    /// for artifact admission until source-independent loading is available.
    pub fn with_weight_shard(
        mut self,
        directory: &Path,
        expected_sha256: &str,
        max_bytes: u64,
    ) -> Result<Self, String> {
        SymbolicCausalModel::open_artifact_with_shard(
            &self.directory,
            self.source_override.as_deref(),
            self.limits.model,
            directory,
            expected_sha256,
            self.layers.clone(),
            max_bytes,
        )?;
        self.shard = Some((directory.to_owned(), expected_sha256.to_owned(), max_bytes));
        Ok(self)
    }

    /// SHA-256 of the validated manifest binding all persisted graph and source hashes.
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Assigned contiguous range.
    pub fn layers(&self) -> Range<u32> {
        self.layers.clone()
    }
    /// Hidden vector width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Serve one dedicated connection with a fresh model and KV session.
    /// Closing a connection discards any uncommitted row. No model is uploaded.
    pub fn serve_connection(&self, mut stream: TcpStream) -> io::Result<()> {
        setup(&stream, self.limits)?;
        let open = read_frame(&mut stream, self.limits.frame_bytes)?;
        if open.len() != 81
            || open[0] != OPEN
            || u32_at(&open, 1)? != VERSION
            || u32_at(&open, 9)? != self.layers.start
            || u32_at(&open, 13)? != self.layers.end
            || open.get(17..) != Some(self.identity.as_bytes())
        {
            write_frame(
                &mut stream,
                &reject(
                    "invalid identity, version or range",
                    self.limits.frame_bytes,
                ),
                self.limits.frame_bytes,
            )?;
            return Ok(());
        }
        let context = u32_at(&open, 5)?;
        if context == 0 || context > self.limits.context {
            write_frame(
                &mut stream,
                &reject("invalid context", self.limits.frame_bytes),
                self.limits.frame_bytes,
            )?;
            return Ok(());
        }
        let opened = match &self.shard {
            Some((directory, hash, budget)) => SymbolicCausalModel::open_artifact_with_shard(
                &self.directory,
                self.source_override.as_deref(),
                self.limits.model,
                directory,
                hash,
                self.layers.clone(),
                *budget,
            ),
            None => SymbolicCausalModel::open_artifact(
                &self.directory,
                self.source_override.as_deref(),
                self.limits.model,
            ),
        };
        let model = match opened {
            Ok(model) => model,
            Err(error) => {
                write_frame(
                    &mut stream,
                    &reject(&error, self.limits.frame_bytes),
                    self.limits.frame_bytes,
                )?;
                return Ok(());
            }
        };
        if checked_identity(&self.directory, self.source_override.as_deref())
            .map_err(|e| invalid(&e))?
            != self.identity
        {
            return Err(invalid("symbolic artifact identity changed"));
        }
        let mut session = match SymbolicLayerSession::new(
            model,
            self.layers.clone(),
            context,
            self.limits.kv_bytes,
        ) {
            Ok(session) => session,
            Err(error) => {
                write_frame(
                    &mut stream,
                    &reject(&error, self.limits.frame_bytes),
                    self.limits.frame_bytes,
                )?;
                return Ok(());
            }
        };
        let mut ready = vec![0];
        ready.extend_from_slice(&(self.width as u32).to_le_bytes());
        write_frame(&mut stream, &ready, self.limits.frame_bytes)?;
        loop {
            let request = read_frame(&mut stream, self.limits.frame_bytes)?;
            let (response, close) = match request[0] {
                PREPARE => {
                    let (position, values) = vectors(&request, self.width)?;
                    match session.prepare(position, &values) {
                        Ok(output) => {
                            let mut response = vec![0];
                            response.extend_from_slice(&(output.len() as u32).to_le_bytes());
                            for value in output {
                                response.extend_from_slice(&value.to_le_bytes());
                            }
                            (response, false)
                        }
                        Err(error) => (reject(&error, self.limits.frame_bytes), false),
                    }
                }
                COMMIT | ABORT => {
                    let position = position_request(&request)?;
                    let result = if request[0] == COMMIT {
                        session.commit(position)
                    } else {
                        session.abort(position)
                    };
                    (
                        match result {
                            Ok(()) => vec![0],
                            Err(error) => reject(&error, self.limits.frame_bytes),
                        },
                        false,
                    )
                }
                RESET if request.len() == 1 => (
                    match session.reset() {
                        Ok(()) => vec![0],
                        Err(error) => reject(&error, self.limits.frame_bytes),
                    },
                    false,
                ),
                CLOSE if request.len() == 1 => (vec![0], true),
                _ => return Err(invalid("invalid symbolic request")),
            };
            write_frame(&mut stream, &response, self.limits.frame_bytes)?;
            if close {
                return Ok(());
            }
        }
    }
}

/// Client for one dedicated range session. It never reconnects or retries.
pub struct SymbolicLayerClient {
    stream: TcpStream,
    limits: SymbolicTransportLimits,
    width: usize,
    healthy: bool,
    identity: String,
    layers: Range<u32>,
    context: u32,
    position: u32,
    prepared: bool,
}
impl SymbolicLayerClient {
    /// Connect using a validated manifest identity, exact range and context.
    pub fn connect(
        address: impl ToSocketAddrs,
        identity: &str,
        layers: Range<u32>,
        context: u32,
        limits: SymbolicTransportLimits,
    ) -> io::Result<Self> {
        if identity.len() != 64
            || !identity.bytes().all(|v| v.is_ascii_hexdigit())
            || layers.start >= layers.end
            || context == 0
            || context > limits.context
        {
            return Err(invalid("invalid symbolic client admission"));
        }
        let stream = TcpStream::connect(address)?;
        setup(&stream, limits)?;
        let mut client = Self {
            stream,
            limits,
            width: 0,
            healthy: true,
            identity: identity.to_owned(),
            layers: layers.clone(),
            context,
            position: 0,
            prepared: false,
        };
        let mut request = vec![OPEN];
        for value in [VERSION, context, layers.start, layers.end] {
            request.extend_from_slice(&value.to_le_bytes());
        }
        request.extend_from_slice(identity.as_bytes());
        let response = client.exchange(&request)?;
        if response.len() != 4 {
            return Err(invalid("invalid symbolic handshake"));
        }
        client.width = u32_at(&response, 0)? as usize;
        if client.width == 0
            || client
                .width
                .checked_mul(8)
                .and_then(|n| n.checked_add(9))
                .is_none_or(|n| n > limits.frame_bytes)
        {
            return Err(invalid("invalid symbolic handshake width"));
        }
        Ok(client)
    }
    fn exchange(&mut self, request: &[u8]) -> io::Result<Vec<u8>> {
        if !self.healthy {
            return Err(io::Error::other("symbolic connection is invalidated"));
        }
        let response = write_frame(&mut self.stream, request, self.limits.frame_bytes)
            .and_then(|_| read_frame(&mut self.stream, self.limits.frame_bytes));
        let bytes = match response {
            Ok(bytes) => bytes,
            Err(error) => {
                self.healthy = false;
                return Err(error);
            }
        };
        match bytes[0] {
            0 => Ok(bytes[1..].to_vec()),
            1 => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                String::from_utf8_lossy(&bytes[1..]).into_owned(),
            )),
            _ => {
                self.healthy = false;
                Err(invalid("invalid symbolic response status"))
            }
        }
    }
    /// Whether the connection remains usable after transport/framing failures.
    pub fn is_healthy(&self) -> bool {
        self.healthy
    }
    /// Last acknowledged position; an unanswered commit invalidates the client.
    pub fn position(&self) -> u32 {
        self.position
    }
    /// Whether a successful preparation awaits commit or abort.
    pub fn is_prepared(&self) -> bool {
        self.prepared
    }
    /// Prepare one token without advancing the remote position.
    pub fn prepare(&mut self, position: u32, hidden: &[f64]) -> io::Result<Vec<f64>> {
        if self.prepared
            || position != self.position
            || position >= self.context
            || hidden.len() != self.width
            || hidden.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid symbolic input vector"));
        }
        let mut request = vec![PREPARE];
        request.extend_from_slice(&position.to_le_bytes());
        request.extend_from_slice(&(hidden.len() as u32).to_le_bytes());
        for value in hidden {
            request.extend_from_slice(&value.to_le_bytes());
        }
        let response = self.exchange(&request)?;
        let mut packed = vec![PREPARE];
        packed.extend_from_slice(&position.to_le_bytes());
        packed.extend_from_slice(&response);
        match vectors(&packed, self.width) {
            Ok((_, output)) => {
                self.prepared = true;
                Ok(output)
            }
            Err(error) => {
                self.healthy = false;
                Err(error)
            }
        }
    }
    fn control(&mut self, code: u8, position: Option<u32>) -> io::Result<()> {
        let mut request = vec![code];
        if let Some(position) = position {
            request.extend_from_slice(&position.to_le_bytes());
        }
        let response = self.exchange(&request)?;
        if !response.is_empty() {
            self.healthy = false;
            return Err(invalid("invalid symbolic acknowledgement"));
        }
        Ok(())
    }
    /// Confirm a prepared position. An unanswered response invalidates this client.
    pub fn commit(&mut self, position: u32) -> io::Result<()> {
        if !self.prepared || position != self.position {
            return Err(invalid("no matching symbolic row to commit"));
        }
        self.control(COMMIT, Some(position))?;
        self.position += 1;
        self.prepared = false;
        Ok(())
    }
    /// Discard an uncommitted row and allow retry at the same position.
    pub fn abort(&mut self, position: u32) -> io::Result<()> {
        if !self.prepared || position != self.position {
            return Err(invalid("no matching symbolic row to abort"));
        }
        self.control(ABORT, Some(position))?;
        self.prepared = false;
        Ok(())
    }
    /// Clear the remote KV session.
    pub fn reset(&mut self) -> io::Result<()> {
        self.control(RESET, None)?;
        self.position = 0;
        self.prepared = false;
        Ok(())
    }
    /// Close the session; the client is unusable afterwards.
    pub fn close(&mut self) -> io::Result<()> {
        let result = self.control(CLOSE, None);
        self.healthy = false;
        result
    }
}

/// A complete ordered symbolic causal pipeline over independently owned sessions.
///
/// All clients must cover `[0, n_layers)` exactly once, in order, and use the
/// same validated artifact identity, width and context. The local model supplies
/// real embeddings and the persisted output head; workers execute BML layers.
/// If any remote or head operation fails, every client is invalidated at the
/// coordinator: commit status may be ambiguous and the pipeline cannot resume.
pub struct SymbolicPipeline {
    head: SymbolicCausalModel,
    clients: Vec<SymbolicLayerClient>,
    position: u32,
    context: u32,
    healthy: bool,
}
impl SymbolicPipeline {
    /// Load a checked persisted artifact and verify complete, ordered coverage.
    pub fn open(
        directory: &Path,
        source_override: Option<&Path>,
        context: u32,
        clients: Vec<SymbolicLayerClient>,
        limits: SymbolicLimits,
    ) -> Result<Self, String> {
        let identity = checked_identity(directory, source_override)?;
        let head = SymbolicCausalModel::open_artifact(directory, source_override, limits)?;
        if checked_identity(directory, source_override)? != identity {
            return Err("symbolic artifact changed during pipeline admission".into());
        }
        if head.profile() != bml_compiler::symbolic_artifact::BOUNDED_PROFILE {
            return Err("symbolic pipeline requires the complete bounded profile".into());
        }
        if context == 0 || context > head.context_limit() {
            return Err("invalid symbolic pipeline context".into());
        }
        let mut next = 0;
        for client in &clients {
            if !client.healthy
                || client.position != 0
                || client.prepared
                || client.identity != identity
                || client.context != context
                || client.width != head.compiler().config().n_embd as usize
                || client.layers.start != next
                || client.layers.end <= next
            {
                return Err("incomplete or inconsistent symbolic pipeline coverage".into());
            }
            next = client.layers.end;
        }
        if next != head.compiler().config().n_layers {
            return Err("incomplete or overlapping symbolic layer coverage".into());
        }
        Ok(Self {
            head,
            clients,
            position: 0,
            context,
            healthy: true,
        })
    }
    /// Confirmed prefix length, excluding any partial remote operation.
    pub fn position(&self) -> u32 {
        self.position
    }
    /// Whether all connections remain safe to use.
    pub fn is_healthy(&self) -> bool {
        self.healthy
    }
    /// Read-only model metadata and vocabulary for tokenization.
    pub fn model(&self) -> &SymbolicCausalModel {
        &self.head
    }

    /// Evaluate one token and return all logits, committing only after the head succeeds.
    /// Any execution, transport or commit failure invalidates the whole pipeline.
    /// An invalid token or exhausted context is rejected before sending anything.
    pub fn forward(&mut self, token: u32) -> Result<Vec<f64>, String> {
        if !self.healthy {
            return Err("symbolic pipeline is invalidated".into());
        }
        if self.position >= self.context || token as usize >= self.head.compiler().vocab().len() {
            return Err("invalid symbolic token or exhausted context".into());
        }
        let result = (|| {
            let mut hidden = self.head.compiler().get_embedding(token)?;
            for client in &mut self.clients {
                hidden = client
                    .prepare(self.position, &hidden)
                    .map_err(|e| e.to_string())?;
            }
            let logits = self.head.project_output(&hidden)?;
            if logits.len() != self.head.compiler().vocab().len()
                || logits.iter().any(|v| !v.is_finite())
            {
                return Err("invalid symbolic pipeline logits".into());
            }
            for client in &mut self.clients {
                client.commit(self.position).map_err(|e| e.to_string())?;
            }
            Ok(logits)
        })();
        match result {
            Ok(logits) => {
                self.position += 1;
                Ok(logits)
            }
            Err(error) => {
                self.healthy = false;
                Err(error)
            }
        }
    }

    /// Reset all healthy sessions and the coordinator position. A partial reset
    /// invalidates the pipeline because remote positions may differ.
    pub fn reset(&mut self) -> Result<(), String> {
        if !self.healthy {
            return Err("symbolic pipeline is invalidated".into());
        }
        for client in &mut self.clients {
            if let Err(error) = client.reset() {
                self.healthy = false;
                return Err(error.to_string());
            }
        }
        self.position = 0;
        Ok(())
    }
    /// Close all connections. This object cannot subsequently execute tokens.
    pub fn close(&mut self) -> Result<(), String> {
        self.healthy = false;
        let mut first_error = None;
        for client in &mut self.clients {
            if let Err(error) = client.close() {
                if first_error.is_none() {
                    first_error = Some(error.to_string());
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_framing_preserves_finite_f64_bits_and_rejects_bad_lengths() {
        let values = [0.0, -0.0, f64::from_bits(1), -3.25];
        let mut request = vec![PREPARE];
        request.extend_from_slice(&7u32.to_le_bytes());
        request.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for value in values {
            request.extend_from_slice(&value.to_le_bytes());
        }
        let (position, decoded) = vectors(&request, 4).unwrap();
        assert_eq!(position, 7);
        assert_eq!(
            decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert!(vectors(&request[..request.len() - 1], 4).is_err());
        assert!(vectors(&request, 3).is_err());
        let start = request.len() - 8;
        request[start..].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(vectors(&request, 4).is_err());
    }
}
