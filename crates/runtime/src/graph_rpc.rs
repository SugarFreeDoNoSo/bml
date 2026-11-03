//! Stateless bounded compiled-graph RPC. Floating values use little-endian IEEE bits.
//! Request v1: five u32 fields (version, input count, slots, slot size, graph bytes),
//! then inputs, flat buffer values and a complete serialized .bmlgraph.
//! Response v1: u32 version, u32 status; success adds f64 result, u32 count, buffer.
use crate::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
    net::{recv_msg_with_limit, send_msg, Message, MsgType, NodeHandle},
};
use bml_compiler::BmlGraph;
use std::io;

/// Maximum graph-RPC frame body, independent of the legacy shard transfer limit.
pub const MAX_WIRE_BYTES: usize = 16 * 1024 * 1024;
/// Complete result and updated flat result buffer; no remote mutable state persists.
#[derive(Debug, PartialEq)]
pub struct GraphResult {
    /// Final scalar value under native runtime arithmetic.
    pub value: f64,
    /// Updated buffer in the request's original slot layout.
    pub buffer: Vec<f64>,
}
/// Stable location of a complete RPC result in append-only storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedGraphResult {
    /// Index holding the final scalar result.
    pub value_index: usize,
    /// Contiguous indices holding the returned flat buffer.
    pub buffer_indices: std::ops::Range<usize>,
}
impl GraphResult {
    /// Publish the complete received result without copying its buffer. On full
    /// storage returns the original result, permitting local retry without RPC.
    pub fn publish(
        self,
        output: &crate::append_only::AppendOnlyResults,
    ) -> Result<PublishedGraphResult, Self> {
        match output.append_record(self.value, &self.buffer) {
            Ok(range) => Ok(PublishedGraphResult {
                value_index: range.start,
                buffer_indices: range.start + 1..range.end,
            }),
            Err(_) => Err(self),
        }
    }
}
/// Remote execution or local publication failed; completed work is preserved.
#[derive(Debug)]
pub enum GraphPublishError {
    /// Transport/protocol failure; remote completion may be unknown.
    Rpc(io::Error),
    /// RPC completed, but storage is full. Retry publication of this value only.
    StorageFull(GraphResult),
}
impl std::fmt::Display for GraphPublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rpc(error) => write!(f, "graph RPC failed: {error}"),
            Self::StorageFull(_) => {
                f.write_str("graph completed; append-only publication capacity exhausted")
            }
        }
    }
}
impl std::error::Error for GraphPublishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rpc(error) => Some(error),
            Self::StorageFull(_) => None,
        }
    }
}
fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn u32_at(bytes: &[u8], offset: usize) -> io::Result<usize> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| invalid("truncated graph RPC header"))?
            .try_into()
            .unwrap(),
    ) as usize)
}
fn values(bytes: &[u8]) -> io::Result<Vec<f64>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes.len() / 8)
        .map_err(|_| invalid("graph value allocation failed"))?;
    for chunk in bytes.as_chunks::<8>().0 {
        output.push(f64::from_le_bytes(*chunk));
    }
    Ok(output)
}
/// Encode an owned-context request with explicit wire/resource bounds.
pub fn encode_request(
    graph: &BmlGraph,
    inputs: &[f64],
    buffer: &ResultBuffer,
) -> io::Result<Vec<u8>> {
    let limits = GraphLimits::default();
    let stored = graph
        .fragments
        .iter()
        .try_fold(0usize, |n, f| n.checked_add(f.ops.len()))
        .ok_or_else(|| invalid("graph opcode overflow"))?;
    if graph.fragments.len() > limits.max_ops
        || stored > limits.max_ops
        || graph.const_pool.len() > limits.max_values
    {
        return Err(invalid("graph encoding budget exceeded"));
    }
    let count = inputs
        .len()
        .checked_add(buffer.len())
        .ok_or_else(|| invalid("graph value overflow"))?;
    if count > limits.max_values {
        return Err(invalid("graph value budget exceeded"));
    }
    let graph_bytes = graph.serialize();
    let size = count
        .checked_mul(8)
        .and_then(|n| n.checked_add(20))
        .and_then(|n| n.checked_add(graph_bytes.len()))
        .ok_or_else(|| invalid("graph frame overflow"))?;
    if size > MAX_WIRE_BYTES {
        return Err(invalid("graph frame budget exceeded"));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(size)
        .map_err(|_| invalid("graph frame allocation failed"))?;
    for value in [
        1,
        inputs.len(),
        buffer.n_slots(),
        buffer.slot_size(),
        graph_bytes.len(),
    ] {
        let value = u32::try_from(value).map_err(|_| invalid("graph header does not fit u32"))?;
        output.extend_from_slice(&value.to_le_bytes());
    }
    for value in inputs.iter().chain(buffer.data()) {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output.extend_from_slice(&graph_bytes);
    Ok(output)
}
fn execute_request(payload: &[u8]) -> io::Result<GraphResult> {
    if payload.len() > MAX_WIRE_BYTES || u32_at(payload, 0)? != 1 {
        return Err(invalid("unsupported graph frame"));
    }
    let inputs = u32_at(payload, 4)?;
    let slots = u32_at(payload, 8)?;
    let slot_size = u32_at(payload, 12)?;
    let graph_len = u32_at(payload, 16)?;
    let buffer_len = slots
        .checked_mul(slot_size)
        .ok_or_else(|| invalid("buffer length overflow"))?;
    let count = inputs
        .checked_add(buffer_len)
        .ok_or_else(|| invalid("value length overflow"))?;
    let limits = GraphLimits::default();
    if count > limits.max_values {
        return Err(invalid("graph values exceed budget"));
    }
    let graph_start = count
        .checked_mul(8)
        .and_then(|n| n.checked_add(20))
        .ok_or_else(|| invalid("frame length overflow"))?;
    if graph_start.checked_add(graph_len) != Some(payload.len()) {
        return Err(invalid("graph frame length mismatch"));
    }
    let graph = BmlGraph::deserialize_with_limits(
        &payload[graph_start..],
        32768,
        limits.max_ops,
        limits.max_ops,
        limits.max_values,
    )
    .map_err(|_| invalid("invalid or oversized serialized graph"))?;
    let validated = ValidatedGraph::new(graph, inputs, slots, slot_size, limits)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let input_values = values(&payload[20..20 + inputs * 8])?;
    let initial = values(&payload[20 + inputs * 8..graph_start])?;
    let mut buffer = ResultBuffer::from_vec(slots, slot_size, initial)
        .map_err(|_| invalid("buffer layout mismatch"))?;
    let value = validated
        .execute(&input_values, &mut buffer)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(GraphResult {
        value,
        buffer: buffer.into_vec(),
    })
}
/// Worker entrypoint. Failures return a fixed nonzero status, never partial data.
/// Status 1 means the request failed wire, graph, resource or allocation validation.
pub fn handle_request(payload: &[u8]) -> Vec<u8> {
    let mut output = 1u32.to_le_bytes().to_vec();
    match execute_request(payload) {
        Err(_) => output.extend_from_slice(&1u32.to_le_bytes()),
        Ok(result) => {
            output.extend_from_slice(&0u32.to_le_bytes());
            output.extend_from_slice(&result.value.to_le_bytes());
            output.extend_from_slice(&(result.buffer.len() as u32).to_le_bytes());
            for value in result.buffer {
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    output
}
fn decode_response(payload: &[u8], expected_values: usize) -> io::Result<GraphResult> {
    if payload.len() > MAX_WIRE_BYTES || u32_at(payload, 0)? != 1 {
        return Err(invalid("unsupported graph response"));
    }
    if u32_at(payload, 4)? != 0 {
        return Err(invalid("remote graph request rejected"));
    }
    let count = u32_at(payload, 16)?;
    if count != expected_values
        || count.checked_mul(8).and_then(|n| n.checked_add(20)) != Some(payload.len())
    {
        return Err(invalid("graph response length mismatch"));
    }
    let value = f64::from_le_bytes(payload[8..16].try_into().unwrap());
    Ok(GraphResult {
        value,
        buffer: values(&payload[20..])?,
    })
}
impl NodeHandle {
    /// Execute remotely, then publish its scalar and full buffer into new slots.
    /// On StorageFull retain/re-publish the returned result; do not repeat RPC.
    pub fn execute_graph_append_only(
        &mut self,
        graph: &BmlGraph,
        inputs: &[f64],
        buffer: &ResultBuffer,
        output: &crate::append_only::AppendOnlyResults,
    ) -> Result<PublishedGraphResult, GraphPublishError> {
        self.execute_graph(graph, inputs, buffer)
            .map_err(GraphPublishError::Rpc)?
            .publish(output)
            .map_err(GraphPublishError::StorageFull)
    }
    /// Execute a complete compiled graph on the peer with explicit inputs/buffer.
    /// No automatic retry: transport failure leaves remote completion unknown.
    /// Transport or malformed-response failures close the connection. A complete
    /// remote validation rejection leaves the stateless session reusable.
    pub fn execute_graph(
        &mut self,
        graph: &BmlGraph,
        inputs: &[f64],
        buffer: &ResultBuffer,
    ) -> io::Result<GraphResult> {
        let payload = encode_request(graph, inputs, buffer)?;
        let response = send_msg(
            self.stream_mut(),
            &Message::new(MsgType::ExecuteGraph, payload),
        )
        .and_then(|()| recv_msg_with_limit(self.stream_mut(), MAX_WIRE_BYTES));
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                let _ = self.stream_mut().shutdown(std::net::Shutdown::Both);
                return Err(error);
            }
        };
        if response.msg_type != MsgType::GraphResult {
            let _ = self.stream_mut().shutdown(std::net::Shutdown::Both);
            return Err(invalid("unexpected graph response type"));
        }
        let result = decode_response(&response.payload, buffer.len());
        if result.is_err() && response.payload != [1, 0, 0, 0, 1, 0, 0, 0] {
            let _ = self.stream_mut().shutdown(std::net::Shutdown::Both);
        }
        result
    }
}
