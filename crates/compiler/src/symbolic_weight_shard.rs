//! Authenticated export of the encoded tensors belonging to a layer range.
//!
//! A trusted caller must distribute the returned descriptor hash separately.
//! This format is an intermediate building block: current workers still load
//! the complete GGUF and do not consume these files.
use crate::symbolic_artifact::{checked_identity, load_directory};
use bml_parser::GgufParser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, ops::Range, path::Path};

const TENSORS: [&str; 9] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];
const MAX_DESCRIPTOR: u64 = 16 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
    length: u64,
    sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    version: u32,
    artifact_sha256: String,
    source_sha256: String,
    start: u32,
    end: u32,
    payload_sha256: String,
    payload_length: u64,
    tensors: Vec<Entry>,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Export encoded GGUF tensor bytes for one nonempty layer range. The output
/// directory must not exist. `max_bytes` applies to the encoded tensor payload.
/// The returned SHA-256 authenticates the descriptor and must be delivered to
/// the consumer over a trusted channel; it is not an embedded signature.
pub fn export_layer_weights(
    directory: &Path,
    source: Option<&Path>,
    layers: Range<u32>,
    output: &Path,
    max_bytes: u64,
) -> Result<String, String> {
    let identity = checked_identity(directory, source)?;
    let artifact = load_directory(directory, source)?;
    if identity != checked_identity(directory, source)? {
        return Err("artifact changed during export".into());
    }
    if layers.start >= layers.end || layers.end > artifact.manifest.config.n_layers {
        return Err("invalid layer range".into());
    }
    let parser =
        GgufParser::open(source.unwrap_or(&artifact.manifest.source)).map_err(|e| e.to_string())?;
    if hash(parser.bytes()) != artifact.manifest.source_sha256 {
        return Err("source changed during export".into());
    }
    let mut entries = Vec::new();
    let mut total = 0u64;
    let mut payload_hash = Sha256::new();
    for layer in layers.clone() {
        for suffix in TENSORS {
            let name = format!("blk.{layer}.{suffix}");
            let info = parser
                .find_tensor(&name)
                .ok_or_else(|| format!("missing tensor {name}"))?;
            let data = parser
                .tensor_data(info)
                .ok_or_else(|| format!("unsupported tensor {name}"))?;
            let length = u64::try_from(data.len()).map_err(|_| "tensor size overflow")?;
            let offset = total;
            total = total.checked_add(length).ok_or("shard size overflow")?;
            if total > max_bytes {
                return Err("shard exceeds byte budget".into());
            }
            payload_hash.update(data);
            entries.push(Entry {
                name,
                dims: info.dims.clone(),
                dtype: info.data_type as u32,
                offset,
                length,
                sha256: hash(data),
            });
        }
    }
    let desc = Descriptor {
        version: 1,
        artifact_sha256: identity,
        source_sha256: artifact.manifest.source_sha256,
        start: layers.start,
        end: layers.end,
        payload_sha256: format!("{:x}", payload_hash.finalize()),
        payload_length: total,
        tensors: entries,
    };
    let encoded = serde_json::to_vec(&desc).map_err(|e| e.to_string())?;
    if encoded.len() as u64 > MAX_DESCRIPTOR {
        return Err("shard descriptor exceeds budget".into());
    }
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let result = (|| {
        let mut f = fs::File::create(output.join("weights.bin")).map_err(|e| e.to_string())?;
        for entry in &desc.tensors {
            let info = parser
                .find_tensor(&entry.name)
                .ok_or("source tensor disappeared")?;
            let data = parser
                .tensor_data(info)
                .ok_or("source tensor unavailable")?;
            if hash(data) != entry.sha256 {
                return Err("source changed during export".into());
            }
            f.write_all(data).map_err(|e| e.to_string())?;
        }
        f.sync_all().map_err(|e| e.to_string())?;
        fs::write(output.join("weights.json"), &encoded).map_err(|e| e.to_string())?;
        Ok(hash(&encoded))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(output);
    }
    result
}

/// Verify a shard against an externally trusted descriptor hash and artifact
/// identity, including all bytes, layout, ordering and the exact layer range.
/// Verification currently reads the payload into memory under `max_bytes`.
pub fn verify_layer_weights(
    directory: &Path,
    expected_descriptor: &str,
    expected_artifact: &str,
    layers: Range<u32>,
    max_bytes: u64,
) -> Result<(), String> {
    VerifiedLayerWeights::open(
        directory,
        expected_descriptor,
        expected_artifact,
        layers,
        max_bytes,
    )
    .map(|_| ())
}

/// Owned, authenticated encoded weights. Retaining the verified bytes prevents
/// a file replacement between integrity validation and decoding.
pub struct VerifiedLayerWeights {
    descriptor: Descriptor,
    payload: Vec<u8>,
}

impl VerifiedLayerWeights {
    /// Load and authenticate one range under an explicit encoded-byte budget.
    pub fn open(
        directory: &Path,
        expected_descriptor: &str,
        expected_artifact: &str,
        layers: Range<u32>,
        max_bytes: u64,
    ) -> Result<Self, String> {
        let descriptor_path = directory.join("weights.json");
        let bytes =
            crate::causal_artifact::read_bounded(&descriptor_path, MAX_DESCRIPTOR as usize)?;
        if hash(&bytes) != expected_descriptor {
            return Err("shard descriptor identity mismatch".into());
        }
        let d: Descriptor = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if d.version != 1
            || d.artifact_sha256 != expected_artifact
            || d.start != layers.start
            || d.end != layers.end
            || layers.start >= layers.end
            || d.payload_length > max_bytes
        {
            return Err("shard identity, range or budget mismatch".into());
        }
        let expected_count = usize::try_from(layers.end - layers.start)
            .ok()
            .and_then(|count| count.checked_mul(TENSORS.len()))
            .ok_or("shard tensor count overflow")?;
        if d.tensors.len() != expected_count {
            return Err("shard tensor count mismatch".into());
        }
        let payload = crate::causal_artifact::read_bounded(
            &directory.join("weights.bin"),
            usize::try_from(max_bytes).unwrap_or(usize::MAX),
        )?;
        if payload.len() as u64 != d.payload_length || hash(&payload) != d.payload_sha256 {
            return Err("shard payload identity mismatch".into());
        }
        let mut offset = 0usize;
        for (i, entry) in d.tensors.iter().enumerate() {
            let layer = layers.start + (i / TENSORS.len()) as u32;
            if entry.name != format!("blk.{layer}.{}", TENSORS[i % TENSORS.len()])
                || entry.offset != offset as u64
                || entry.dims.is_empty()
                || entry.dims.contains(&0)
            {
                return Err("invalid shard tensor layout".into());
            }
            let end = offset
                .checked_add(usize::try_from(entry.length).map_err(|_| "tensor length overflow")?)
                .ok_or("tensor length overflow")?;
            let data = payload.get(offset..end).ok_or("truncated shard tensor")?;
            if hash(data) != entry.sha256 {
                return Err("shard tensor identity mismatch".into());
            }
            offset = end;
        }
        if offset != payload.len() {
            return Err("trailing shard bytes".into());
        }
        Ok(Self {
            descriptor: d,
            payload,
        })
    }

    /// The global layer range included in this verified payload.
    pub fn layers(&self) -> Range<u32> {
        self.descriptor.start..self.descriptor.end
    }

    /// Shape in GGML axis order; axis zero is contiguous.
    pub fn dimensions(&self, name: &str) -> Option<&[u64]> {
        self.descriptor
            .tensors
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.dims.as_slice())
    }

    /// Decode one contiguous GGML row with the independent GGUF codecs.
    pub fn decode_row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>, String> {
        let entry = self
            .descriptor
            .tensors
            .iter()
            .find(|entry| entry.name == name)
            .ok_or("missing shard tensor")?;
        let (&columns, outer) = entry
            .dims
            .split_first()
            .ok_or("missing tensor dimensions")?;
        let columns = usize::try_from(columns).map_err(|_| "tensor dimension overflow")?;
        let rows = outer.iter().try_fold(1usize, |n, &dimension| {
            n.checked_mul(usize::try_from(dimension).map_err(|_| "tensor dimension overflow")?)
                .ok_or("tensor dimension overflow")
        })?;
        let length = usize::try_from(entry.length).map_err(|_| "tensor length overflow")?;
        if columns == 0 || rows == 0 || row >= rows || !length.is_multiple_of(rows) {
            return Err("invalid shard row layout".into());
        }
        let stride = length / rows;
        let start = usize::try_from(entry.offset)
            .map_err(|_| "tensor offset overflow")?
            .checked_add(row.checked_mul(stride).ok_or("tensor row overflow")?)
            .ok_or("tensor row overflow")?;
        let end = start.checked_add(stride).ok_or("tensor row overflow")?;
        let data = self
            .payload
            .get(start..end)
            .ok_or("tensor row outside payload")?;
        let dtype = bml_gguf::GgufDataType::from_u32(entry.dtype).ok_or("unknown tensor dtype")?;
        bml_gguf::decode_f32(dtype, data, columns).map_err(|error| error.to_string())
    }
}
