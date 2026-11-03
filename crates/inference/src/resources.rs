//! Normalized, hashed model resources and an ELF appendix for autonomous CPU use.
//!
//! Weights retain their original F32/F16/Q4_0/Q8_0/Q6_K encoding. The container stores
//! model configuration and tokenizer data once, and borrows tensors from a
//! read-only mapping. It does not contain or parse a GGUF at runtime.

use crate::model::{Config, Model, Operation};
use crate::tokenizer::TokenizerSpec;
use crate::{Error, Result, Tensor};
use bml_gguf::{GgufDataType, GgufReader, MmapFile};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const RESOURCE_MAGIC: &[u8; 8] = b"BMLRS001";
const FOOTER_MAGIC: &[u8; 8] = b"BMLAOT01";
const HEADER_BYTES: usize = 48;
const FOOTER_BYTES: usize = 64;
const FORMAT_VERSION: u64 = 1;
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_TENSORS: usize = 100_000;
const TENSOR_ALIGNMENT: usize = 32;
const RESOURCE_ALIGNMENT: usize = 4096;
const TARGET: &str = "x86_64-unknown-linux-gnu";
const ABI: &str = "bml-inference-cpu-v1";
const PROFILE: &str = "llama-dense-rmsnorm-rope-swiglu-v1";

/// A tensor's canonical range in the normalized compressed tensor section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDescriptor {
    /// Unique canonical model tensor name.
    pub name: String,
    /// Dimensions in GGML order, with axis zero contiguous.
    pub dims: Vec<u64>,
    /// GGML dtype number; only F32/F16/Q4_0/Q8_0/Q6_K are accepted.
    pub dtype: u32,
    /// Byte offset relative to the aligned tensor section.
    pub offset: u64,
    /// Exact encoded length, excluding inter-tensor padding.
    pub length: u64,
    /// Lowercase SHA-256 of the exact encoded tensor bytes.
    pub sha256: String,
}

/// Resource bounds derived from the admitted model configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResourceLimits {
    /// Maximum sequence length admitted by this model.
    pub max_context_tokens: usize,
    /// Bytes for both F32 KV caches at the declared context limit.
    pub kv_cache_bytes: u64,
    /// Conservative numerical scratch bound for one prefill/decode call.
    ///
    /// Includes attention scores, decoded rows, intermediate activations and
    /// two logits buffers. Excludes model/tokenizer metadata, Vec headers,
    /// allocator overhead and outputs retained by the caller. This is an
    /// analytical payload bound, not a measurement of resident memory.
    pub scratch_payload_bytes_bound: u64,
}

/// Identification of the exporter build, without local source/toolchain paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildIdentity {
    /// Exporting Cargo package, independently of the model's source identity.
    pub package: String,
    /// Exporter package version at compilation time.
    pub version: String,
    /// Exact source commit if supplied as `BML_BUILD_REVISION` at build time.
    /// `None` means the revision was not recorded, rather than a guessed commit.
    pub source_revision: Option<String>,
    /// Rust compiler identification if supplied as `BML_BUILD_RUSTC`.
    pub rustc_version: Option<String>,
}

/// Fixed operation sequence and precision semantics admitted by this ABI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlan {
    /// Version of the operation vocabulary and ordering semantics.
    pub version: u64,
    /// Activations and KV storage; this ABI admits only `f32`.
    pub activation_dtype: String,
    /// Scalar dot/softmax reductions; this ABI admits only `f64`.
    pub reduction_dtype: String,
    /// Explicit operations, checked against the model configuration at load.
    pub operations: Vec<Operation>,
}

/// Versioned metadata for a normalized model, with no build-machine paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Version of the binary header, tensor layout and JSON schema.
    pub format_version: u64,
    /// Supported executable target; the first profile uses Linux x86_64 GNU.
    pub target: String,
    /// Runtime resource ABI identifier.
    pub abi: String,
    /// Explicit admitted Llama architecture profile.
    pub profile: String,
    /// SHA-256 of the original GGUF, used for provenance rather than loading.
    pub source_sha256: String,
    /// Exporter identity; launcher build/link flags belong in its build report.
    pub producer: BuildIdentity,
    /// Model dimensions and numerical parameters validated by the runtime.
    pub config: Config,
    /// Exact tokenizer vocabulary, scores and special-token policy.
    pub tokenizer: TokenizerSpec,
    /// Context and KV memory bounds derived from `config`.
    pub limits: ResourceLimits,
    /// Versioned tensor execution sequence and numerical policy.
    pub execution_plan: ExecutionPlan,
    /// Canonically ordered, nonoverlapping encoded tensor resources.
    pub tensors: Vec<TensorDescriptor>,
}

/// An owned read-only mapping and validated normalized resource descriptors.
///
/// The mapped file must remain unchanged and untruncated while this value or
/// any borrowed [`Model`] exists. Checksums detect accidental corruption; they
/// are not signatures and do not establish the origin of untrusted artifacts.
pub struct Artifact {
    mapping: MmapFile,
    manifest: Manifest,
    ranges: Vec<Range<usize>>,
    resource_range: Range<usize>,
}

impl Artifact {
    /// Open a normalized resource file or an ELF carrying a model appendix.
    ///
    /// Validates bounded metadata, version/target/ABI, shape-derived lengths,
    /// alignment, canonical ranges and SHA-256 before exposing tensor slices.
    /// Extra trailing bytes, missing ranges and duplicate names are rejected.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mapping = MmapFile::open(path)?;
        let bytes = mapping.as_bytes();
        let resource_range = resource_range(bytes)?;
        let (manifest, relative_ranges) = parse_resources(&bytes[resource_range.clone()])?;
        let ranges = relative_ranges
            .into_iter()
            .map(|range| range.start + resource_range.start..range.end + resource_range.start)
            .collect();
        let artifact = Self {
            mapping,
            manifest,
            ranges,
            resource_range,
        };
        // The runtime owns semantic checks for required tensors and tokenizer.
        artifact.model()?;
        Ok(artifact)
    }

    /// Validated configuration, provenance and tensor table.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// SHA-256 of the entire normalized resource, including manifest and padding.
    ///
    /// An AOT launcher can bind this value at compilation time to detect a
    /// different resource even if its untrusted source-identity label matches.
    pub fn resource_sha256(&self) -> String {
        sha256(&self.mapping.as_bytes()[self.resource_range.clone()])
    }

    /// Create a model borrowing the mapping; no tensor payload is copied.
    pub fn model(&self) -> Result<Model<'_>> {
        let tensors = self
            .manifest
            .tensors
            .iter()
            .zip(&self.ranges)
            .map(|(tensor, range)| {
                Ok(Tensor {
                    name: tensor.name.clone(),
                    dims: tensor.dims.clone(),
                    dtype: dtype(tensor.dtype)?,
                    data: &self.mapping.as_bytes()[range.clone()],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let model = Model::from_tensors(
            self.manifest.config.clone(),
            self.manifest.tokenizer.clone(),
            tensors,
        )?;
        if self.manifest.execution_plan.operations != model.plan() {
            return Err(Error(
                "resource execution plan does not match model configuration".into(),
            ));
        }
        Ok(model)
    }
}

/// Validate and stream a GGUF's model into normalized resources.
///
/// Only the bounded manifest is serialized in memory. Each tensor is written
/// directly from the source mapping with its quantization unchanged. The
/// returned manifest records the source hash and exact emitted resource ranges.
pub fn write_model(reader: &GgufReader, output: &mut impl Write) -> Result<Manifest> {
    let model = Model::from_gguf(reader)?;
    let mut tensors = Vec::new();
    let mut names = HashSet::new();
    let mut offset = 0usize;
    if model.tensors().len() > MAX_TENSORS {
        return Err(Error("too many tensor resources".into()));
    }
    for tensor in model.tensors() {
        if !names.insert(&tensor.name) {
            return Err(Error("duplicate tensor resource name".into()));
        }
        let length = encoded_length(tensor.dtype, &tensor.dims)?;
        if length != tensor.data.len() {
            return Err(Error("tensor resource byte length mismatch".into()));
        }
        offset = align(offset, TENSOR_ALIGNMENT)?;
        tensors.push(TensorDescriptor {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
            dtype: tensor.dtype as u32,
            offset: to_u64(offset)?,
            length: to_u64(length)?,
            sha256: sha256(tensor.data),
        });
        offset = offset
            .checked_add(length)
            .ok_or_else(|| Error("tensor resource size overflow".into()))?;
    }
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        target: TARGET.into(),
        abi: ABI.into(),
        profile: PROFILE.into(),
        source_sha256: sha256(reader.bytes()),
        producer: BuildIdentity {
            package: env!("CARGO_PKG_NAME").into(),
            version: env!("CARGO_PKG_VERSION").into(),
            source_revision: option_env!("BML_BUILD_REVISION").map(str::to_owned),
            rustc_version: option_env!("BML_BUILD_RUSTC").map(str::to_owned),
        },
        config: model.config().clone(),
        tokenizer: model.tokenizer().spec().clone(),
        limits: resource_limits(model.config())?,
        execution_plan: ExecutionPlan {
            version: FORMAT_VERSION,
            activation_dtype: "f32".into(),
            reduction_dtype: "f64".into(),
            operations: model.plan().to_vec(),
        },
        tensors,
    };
    validate_identity(&manifest.producer)?;
    let json = serde_json::to_vec(&manifest)?;
    if json.is_empty() || json.len() > MAX_MANIFEST_BYTES {
        return Err(Error("resource manifest exceeds 16 MiB limit".into()));
    }
    output.write_all(RESOURCE_MAGIC)?;
    output.write_all(&to_u64(json.len())?.to_le_bytes())?;
    output.write_all(&Sha256::digest(&json))?;
    output.write_all(&json)?;
    let header_end = HEADER_BYTES + json.len();
    write_padding(output, align(header_end, TENSOR_ALIGNMENT)? - header_end)?;
    let mut written = 0usize;
    for (descriptor, tensor) in manifest.tensors.iter().zip(model.tensors()) {
        let offset = to_usize(descriptor.offset)?;
        write_padding(output, offset - written)?;
        output.write_all(tensor.data)?;
        written = offset + tensor.data.len();
    }
    Ok(manifest)
}

/// Validate and atomically publish normalized model resources to a new path.
///
/// A sibling temporary file is synced before publishing with a no-overwrite
/// hard link. On failure only the private temporary file is removed, so an
/// invalid input never leaves an empty or partial destination behind.
pub fn write_model_file(reader: &GgufReader, output: impl AsRef<Path>) -> Result<Manifest> {
    let mut pending = PendingOutput::new(output.as_ref())?;
    let manifest = write_model(reader, &mut pending.file)?;
    pending.commit()?;
    Ok(manifest)
}

/// Append validated model resources to a Linux x86_64 ELF launcher.
///
/// Copies both inputs in bounded chunks and appends a versioned SHA-256 footer.
/// The output must not already exist, so input files and existing artifacts are
/// never truncated. Publication is atomic; failed copies remove their temporary
/// file without creating an empty or partial output.
/// The executable remains dynamically dependent on its launcher's system ABI.
pub fn append_to_executable(
    launcher: impl AsRef<Path>,
    resources: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<()> {
    let resources = Artifact::open(resources.as_ref())?;
    let mut input = File::open(launcher.as_ref())?;
    let mut elf_header = [0u8; 64];
    input.read_exact(&mut elf_header)?;
    check_elf(&elf_header)?;
    input.seek(SeekFrom::Start(0))?;
    let launcher_len = to_usize(input.metadata()?.len())?;
    let resource_offset = align(launcher_len, RESOURCE_ALIGNMENT)?;
    let bytes = &resources.mapping.as_bytes()[resources.resource_range.clone()];
    let mut pending = PendingOutput::new(output.as_ref())?;
    let mut destination = &mut pending.file;
    let result = (|| -> Result<()> {
        let copied = std::io::copy(&mut input, &mut destination)?;
        if copied != to_u64(launcher_len)? {
            return Err(Error("launcher changed while being copied".into()));
        }
        write_padding(&mut destination, resource_offset - launcher_len)?;
        let mut digest = Sha256::new();
        for chunk in bytes.chunks(64 * 1024) {
            digest.update(chunk);
            destination.write_all(chunk)?;
        }
        destination.write_all(FOOTER_MAGIC)?;
        destination.write_all(&FORMAT_VERSION.to_le_bytes())?;
        destination.write_all(&to_u64(resource_offset)?.to_le_bytes())?;
        destination.write_all(&to_u64(bytes.len())?.to_le_bytes())?;
        destination.write_all(&digest.finalize())?;
        destination.flush()?;
        destination.set_permissions(input.metadata()?.permissions())?;
        Ok(())
    })();
    result?;
    pending.commit()
}

struct PendingOutput {
    file: File,
    temporary: PathBuf,
    destination: PathBuf,
}

impl PendingOutput {
    fn new(destination: &Path) -> Result<Self> {
        static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);
        let name = destination
            .file_name()
            .ok_or_else(|| Error("output path must name a file".into()))?;
        match fs::symlink_metadata(destination) {
            Ok(_) => return Err(Error("output already exists".into())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| Error(error.to_string()))?
            .as_nanos();
        for _ in 0..64 {
            let mut temporary_name = std::ffi::OsString::from(".");
            temporary_name.push(name);
            temporary_name.push(format!(
                ".bml-tmp-{}-{time}-{}",
                std::process::id(),
                NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed)
            ));
            let temporary = destination.with_file_name(temporary_name);
            match File::options()
                .write(true)
                .create_new(true)
                .open(&temporary)
            {
                Ok(file) => {
                    return Ok(Self {
                        file,
                        temporary,
                        destination: destination.to_owned(),
                    })
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        Err(Error("unable to create a unique temporary output".into()))
    }

    fn commit(self) -> Result<()> {
        self.file.sync_all()?;
        // Unlike rename, hard_link cannot replace a file that another process
        // creates after our initial existence check. Both paths share a parent.
        fs::hard_link(&self.temporary, &self.destination)?;
        Ok(())
    }
}

impl Drop for PendingOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temporary);
    }
}

fn resource_range(bytes: &[u8]) -> Result<Range<usize>> {
    if bytes.starts_with(RESOURCE_MAGIC) {
        return Ok(0..bytes.len());
    }
    check_elf(bytes)?;
    let footer_offset = bytes
        .len()
        .checked_sub(FOOTER_BYTES)
        .ok_or_else(|| Error("missing AOT footer".into()))?;
    let footer = &bytes[footer_offset..];
    if &footer[..8] != FOOTER_MAGIC || read_u64(footer, 8)? != FORMAT_VERSION {
        return Err(Error("unsupported AOT footer magic or version".into()));
    }
    let offset = to_usize(read_u64(footer, 16)?)?;
    let length = to_usize(read_u64(footer, 24)?)?;
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error("AOT resource range overflow".into()))?;
    if offset < 64
        || !offset.is_multiple_of(RESOURCE_ALIGNMENT)
        || length < HEADER_BYTES
        || end != footer_offset
    {
        return Err(Error("invalid AOT resource range".into()));
    }
    let payload = bytes
        .get(offset..end)
        .ok_or_else(|| Error("truncated AOT resources".into()))?;
    if Sha256::digest(payload)[..] != footer[32..64] {
        return Err(Error("AOT resource checksum mismatch".into()));
    }
    Ok(offset..end)
}

fn parse_resources(bytes: &[u8]) -> Result<(Manifest, Vec<Range<usize>>)> {
    if bytes.len() < HEADER_BYTES || &bytes[..8] != RESOURCE_MAGIC {
        return Err(Error("invalid normalized resource header".into()));
    }
    let manifest_len = to_usize(read_u64(bytes, 8)?)?;
    if manifest_len == 0 || manifest_len > MAX_MANIFEST_BYTES {
        return Err(Error("resource manifest exceeds 16 MiB limit".into()));
    }
    let manifest_end = HEADER_BYTES + manifest_len;
    let json = bytes
        .get(HEADER_BYTES..manifest_end)
        .ok_or_else(|| Error("truncated resource manifest".into()))?;
    if Sha256::digest(json)[..] != bytes[16..48] {
        return Err(Error("resource manifest checksum mismatch".into()));
    }
    let manifest: Manifest = serde_json::from_slice(json)?;
    if manifest.format_version != FORMAT_VERSION
        || manifest.target != TARGET
        || manifest.abi != ABI
        || manifest.profile != PROFILE
        || !is_sha256(&manifest.source_sha256)
    {
        return Err(Error(
            "unsupported resource version, target, ABI or provenance".into(),
        ));
    }
    if manifest.limits != resource_limits(&manifest.config)? {
        return Err(Error(
            "resource limits do not match model configuration".into(),
        ));
    }
    validate_identity(&manifest.producer)?;
    if manifest.execution_plan.version != FORMAT_VERSION
        || manifest.execution_plan.activation_dtype != "f32"
        || manifest.execution_plan.reduction_dtype != "f64"
    {
        return Err(Error(
            "unsupported resource execution plan or precision".into(),
        ));
    }
    if manifest.tensors.is_empty() || manifest.tensors.len() > MAX_TENSORS {
        return Err(Error("invalid tensor resource count".into()));
    }
    let data_base = align(manifest_end, TENSOR_ALIGNMENT)?;
    check_padding(bytes, manifest_end..data_base)?;
    let mut previous_end = 0usize;
    let mut names = HashSet::new();
    let mut ranges = Vec::new();
    for tensor in &manifest.tensors {
        if tensor.name.is_empty() || !names.insert(&tensor.name) {
            return Err(Error("empty or duplicate tensor resource name".into()));
        }
        let dtype = dtype(tensor.dtype)?;
        let length = encoded_length(dtype, &tensor.dims)?;
        let offset = to_usize(tensor.offset)?;
        if tensor.length != to_u64(length)?
            || offset != align(previous_end, TENSOR_ALIGNMENT)?
            || !is_sha256(&tensor.sha256)
        {
            return Err(Error(
                "invalid tensor resource length, offset or checksum".into(),
            ));
        }
        let start = data_base
            .checked_add(offset)
            .ok_or_else(|| Error("tensor resource offset overflow".into()))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| Error("tensor resource end overflow".into()))?;
        check_padding(bytes, data_base + previous_end..start)?;
        let payload = bytes
            .get(start..end)
            .ok_or_else(|| Error("truncated tensor resource".into()))?;
        if sha256(payload) != tensor.sha256 {
            return Err(Error("tensor resource checksum mismatch".into()));
        }
        ranges.push(start..end);
        previous_end = offset + length;
    }
    if data_base.checked_add(previous_end) != Some(bytes.len()) {
        return Err(Error("trailing bytes after tensor resources".into()));
    }
    Ok((manifest, ranges))
}

fn resource_limits(config: &Config) -> Result<ResourceLimits> {
    config.validate()?;
    let head_dim = config.embedding_length / config.head_count;
    let kv_cache_bytes = [
        config.context_length,
        config.block_count,
        config.head_count_kv,
        head_dim,
        2,
        std::mem::size_of::<f32>(),
    ]
    .into_iter()
    .try_fold(1u64, |product, value| {
        product
            .checked_mul(to_u64(value)?)
            .ok_or_else(|| Error("KV resource budget overflow".into()))
    })?;
    // A deliberately conservative upper bound for the reference implementation:
    // per-layer activations, one decoded projection row, scores, and both the
    // preceding/current logits while prefill replaces its output buffer.
    let scratch_payload_bytes_bound = [
        (config.embedding_length, 16 * 4u64),
        (config.feed_forward_length, 8 * 4),
        (config.head_count_kv * head_dim, 4 * 4),
        (config.vocab_size, 2 * 4),
        (config.context_length, 8),
    ]
    .into_iter()
    .try_fold(0u64, |total, (count, bytes)| {
        to_u64(count)?
            .checked_mul(bytes)
            .and_then(|size| total.checked_add(size))
            .ok_or_else(|| Error("scratch resource budget overflow".into()))
    })?;
    Ok(ResourceLimits {
        max_context_tokens: config.context_length,
        kv_cache_bytes,
        scratch_payload_bytes_bound,
    })
}

fn validate_identity(identity: &BuildIdentity) -> Result<()> {
    let version_valid = !identity.version.is_empty()
        && identity.version.len() <= 64
        && identity
            .version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".+-".contains(&byte));
    let revision_valid = identity.source_revision.as_ref().is_none_or(|revision| {
        matches!(revision.len(), 40 | 64) && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    });
    let compiler_valid = identity.rustc_version.as_ref().is_none_or(|compiler| {
        compiler.starts_with("rustc ")
            && compiler.len() <= 256
            && compiler
                .bytes()
                .all(|byte| byte.is_ascii_graphic() || byte == b' ')
            && !compiler.contains(['/', '\\'])
    });
    if identity.package != "bml-inference" || !version_valid || !revision_valid || !compiler_valid {
        return Err(Error("invalid exporter build identity".into()));
    }
    Ok(())
}

fn dtype(number: u32) -> Result<GgufDataType> {
    match GgufDataType::from_u32(number) {
        Some(
            value @ (GgufDataType::F32
            | GgufDataType::F16
            | GgufDataType::Q4_0
            | GgufDataType::Q8_0
            | GgufDataType::Q6_K),
        ) => Ok(value),
        _ => Err(Error("unsupported normalized tensor dtype".into())),
    }
}

fn encoded_length(dtype: GgufDataType, dims: &[u64]) -> Result<usize> {
    if dims.is_empty() || dims.len() > 4 || dims.contains(&0) {
        return Err(Error("invalid normalized tensor shape".into()));
    }
    let elements = dims.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(to_usize(dimension)?)
            .ok_or_else(|| Error("tensor shape overflow".into()))
    })?;
    let length = match dtype {
        GgufDataType::F32 => elements.checked_mul(4),
        GgufDataType::F16 => elements.checked_mul(2),
        GgufDataType::Q6_K if dims[0].is_multiple_of(256) => (elements / 256).checked_mul(210),
        GgufDataType::Q4_0 | GgufDataType::Q8_0 if dims[0].is_multiple_of(32) => {
            (elements / 32).checked_mul(if dtype == GgufDataType::Q4_0 { 18 } else { 34 })
        }
        _ => {
            return Err(Error(
                "unsupported dtype or incomplete quantized row".into(),
            ))
        }
    };
    length.ok_or_else(|| Error("encoded tensor length overflow".into()))
}

fn check_elf(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 64
        || &bytes[..4] != b"\x7fELF"
        || bytes[4] != 2
        || bytes[5] != 1
        || bytes[6] != 1
        || !matches!(u16::from_le_bytes([bytes[16], bytes[17]]), 2 | 3)
        || u16::from_le_bytes([bytes[18], bytes[19]]) != 62
    {
        return Err(Error(
            "launcher must be a 64-bit little-endian x86_64 ELF executable".into(),
        ));
    }
    Ok(())
}

fn check_padding(bytes: &[u8], range: Range<usize>) -> Result<()> {
    if bytes
        .get(range)
        .is_none_or(|padding| padding.iter().any(|&byte| byte != 0))
    {
        return Err(Error(
            "invalid or truncated resource alignment padding".into(),
        ));
    }
    Ok(())
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let data = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error("truncated resource integer".into()))?;
    Ok(u64::from_le_bytes(
        data.try_into()
            .map_err(|_| Error("invalid resource integer".into()))?,
    ))
}

fn align(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or_else(|| Error("resource alignment overflow".into()))
}

fn to_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error("resource size exceeds address space".into()))
}

fn to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error("resource size exceeds u64".into()))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn write_padding(output: &mut impl Write, mut count: usize) -> Result<()> {
    let zeros = [0u8; RESOURCE_ALIGNMENT];
    while count != 0 {
        let length = count.min(zeros.len());
        output.write_all(&zeros[..length])?;
        count -= length;
    }
    Ok(())
}
