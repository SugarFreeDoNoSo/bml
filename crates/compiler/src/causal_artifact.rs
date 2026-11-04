//! Compiled row-projection graphs plus a versioned native causal execution profile.
//! Matrix bytes remain in an immutable, SHA-256-identified GGUF mapping. This
//! format does not claim that native normalization/attention are symbolic BML.
use crate::{
    bml_ops,
    fragment::{BmlGraph, Fragment},
    gguf_compiler::{InferenceCompiler, ModelConfig},
    tokenizer::Vocabulary,
};
use bml_inference::tokenizer::TokenizerSpec;
use bml_parser::GgufParser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

/// Manifest filename distinguishing this format from legacy scalar directories.
pub const MANIFEST: &str = "model.bmlgraph.json";
const PROFILE: &str = "llama-causal-f64-f32kv-rpn-v1";
const MAX_MANIFEST: usize = 16 * 1024 * 1024;
const MAX_PROGRAM_BYTES: usize = 64 * 1024;
const MAX_PROGRAMS: usize = 256;
const MAX_PROJECTIONS: usize = 65_536;

/// One stored RPN program shared by projections having the same input width.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    /// Number of input and weight values in each row.
    pub columns: usize,
    /// SHA-256 of the complete serialized program file.
    pub sha256: String,
}

/// Binding of an encoded tensor to its stored row program.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    /// Exact GGUF tensor name.
    pub tensor: String,
    /// Input width, GGML axis zero.
    pub columns: usize,
    /// Output width, GGML axis one.
    pub rows: usize,
    /// Index into the program table.
    pub program: usize,
}

/// Complete binding and tokenizer metadata for the versioned causal profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Container schema version.
    pub version: u32,
    /// Numerical execution profile, including non-RPN host operations.
    pub profile: String,
    /// Default external GGUF locator; callers may override it, with the same hash.
    pub source: PathBuf,
    /// SHA-256 of the complete GGUF, including metadata and encoded tensor bytes.
    pub source_sha256: String,
    /// Model dimensions bound to the referenced source.
    pub config: ModelConfig,
    /// Number of grouped KV heads.
    pub kv_heads: u32,
    /// Positive normalization epsilon.
    pub rms_norm_epsilon: f64,
    /// Rotary frequency base.
    pub rope_frequency_base: f64,
    /// Persisted, validated BPE vocabulary and special-token policy.
    pub tokenizer: TokenizerSpec,
    /// Stored program descriptors in file-index order.
    pub programs: Vec<Program>,
    /// Exactly the matrix projections required by this causal profile.
    pub projections: Vec<Projection>,
}

/// A loaded artifact before runtime stack/loop validation.
pub struct LoadedArtifact {
    /// Validated source mapping and causal numerical implementation.
    pub compiler: InferenceCompiler,
    /// Verified metadata and source bindings.
    pub manifest: Manifest,
    /// Deserialized stored programs; never regenerated on load.
    pub graphs: Vec<BmlGraph>,
}

pub(crate) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > limit as u64 {
        return Err("artifact file exceeds its byte budget".into());
    }
    let mut bytes = Vec::new();
    file.take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > limit {
        return Err("artifact file grew past its byte budget".into());
    }
    Ok(bytes)
}

pub(crate) fn required_projections(
    model: &InferenceCompiler,
) -> Result<Vec<(String, usize, usize)>, String> {
    let count = (model.config().n_layers as usize)
        .checked_mul(7)
        .and_then(|n| n.checked_add(1))
        .ok_or("projection count overflow")?;
    if count > MAX_PROJECTIONS {
        return Err("too many causal projections".into());
    }
    let mut names = Vec::with_capacity(count);
    for layer in 0..model.config().n_layers {
        for suffix in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            names.push(format!("blk.{layer}.{suffix}.weight"));
        }
    }
    names.push(model.output_weight_name().into());
    names
        .into_iter()
        .map(|name| {
            let &(columns, rows) = model
                .tensor_dims()
                .get(&name)
                .ok_or("missing projection dimensions")?;
            if columns == 0 || rows == 0 {
                return Err("empty projection".into());
            }
            Ok((name, columns, rows))
        })
        .collect()
}

/// Compile executable row graphs and persist BPE/configuration with an external
/// GGUF reference. Reject nonempty output directories rather than mix artifacts.
pub fn compile_directory(
    source: &Path,
    output: &Path,
    fragment_bytes: usize,
) -> Result<Manifest, String> {
    let source = source.canonicalize().map_err(|e| e.to_string())?;
    let parser = GgufParser::open(&source).map_err(|e| e.to_string())?;
    let source_sha256 = hash(parser.bytes());
    let tokenizer = TokenizerSpec::from_gguf(&parser).map_err(|e| e.to_string())?;
    let model = InferenceCompiler::from_mapped_parser(parser)?;
    let required = required_projections(&model)?;
    let mut programs = Vec::new();
    let mut encoded = Vec::new();
    let mut widths = BTreeMap::new();
    let mut projections = Vec::new();
    for (tensor, columns, rows) in required {
        let program = if let Some(&id) = widths.get(&columns) {
            id
        } else {
            let n = u32::try_from(columns).map_err(|_| "RPN row width overflow")?;
            n.checked_mul(2).ok_or("RPN buffer index overflow")?;
            let row = bml_ops::bml_dot_program(0, n, n);
            let fragment = Fragment { ops: row.ops };
            if fragment.byte_size() > fragment_bytes {
                return Err("fragment threshold cannot contain the atomic row loop".into());
            }
            if programs.len() >= MAX_PROGRAMS {
                return Err("too many distinct row programs".into());
            }
            let graph = BmlGraph {
                fragments: vec![fragment],
                const_pool: row.const_pool,
                threshold: fragment_bytes,
            };
            let bytes = graph.serialize();
            let id = programs.len();
            programs.push(Program {
                columns,
                sha256: hash(&bytes),
            });
            encoded.push(bytes);
            widths.insert(columns, id);
            id
        };
        projections.push(Projection {
            tensor,
            columns,
            rows,
            program,
        });
    }
    let manifest = Manifest {
        version: 1,
        profile: PROFILE.into(),
        source,
        source_sha256,
        config: model.config().clone(),
        kv_heads: model.n_kv_heads(),
        rms_norm_epsilon: model.rms_norm_epsilon(),
        rope_frequency_base: model.rope_frequency_base(),
        tokenizer,
        programs,
        projections,
    };
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_MANIFEST {
        return Err("compiled manifest exceeds byte budget".into());
    }
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    if fs::read_dir(output)
        .map_err(|e| e.to_string())?
        .next()
        .is_some()
    {
        return Err("output directory is not empty".into());
    }
    for (i, bytes) in encoded.iter().enumerate() {
        fs::write(output.join(format!("program_{i}.bmlgraph")), bytes)
            .map_err(|e| e.to_string())?;
    }
    // Publish the manifest last: partial writes cannot look like a loadable bundle.
    fs::write(output.join(MANIFEST), bytes).map_err(|e| e.to_string())?;
    Ok(manifest)
}

/// Load saved graphs, verify their hashes and bind them to the exact GGUF bytes.
/// A source override changes only the locator, never the expected model identity.
pub fn load_directory(
    directory: &Path,
    source_override: Option<&Path>,
) -> Result<LoadedArtifact, String> {
    let bytes = read_bounded(&directory.join(MANIFEST), MAX_MANIFEST)?;
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|e| format!("artifact manifest: {e}"))?;
    if manifest.version != 1
        || manifest.profile != PROFILE
        || manifest.programs.is_empty()
        || manifest.programs.len() > MAX_PROGRAMS
        || manifest.projections.len() > MAX_PROJECTIONS
    {
        return Err("unsupported causal artifact profile or table size".into());
    }
    let parser =
        GgufParser::open(source_override.unwrap_or(&manifest.source)).map_err(|e| e.to_string())?;
    if hash(parser.bytes()) != manifest.source_sha256 {
        return Err("referenced GGUF identity mismatch".into());
    }
    let actual_tokenizer = TokenizerSpec::from_gguf(&parser).map_err(|e| e.to_string())?;
    if serde_json::to_vec(&actual_tokenizer).map_err(|e| e.to_string())?
        != serde_json::to_vec(&manifest.tokenizer).map_err(|e| e.to_string())?
    {
        return Err("artifact tokenizer differs from referenced model".into());
    }
    let mut compiler = InferenceCompiler::from_mapped_parser(parser)?;
    if compiler.config() != &manifest.config
        || compiler.n_kv_heads() != manifest.kv_heads
        || compiler.rms_norm_epsilon().to_bits() != manifest.rms_norm_epsilon.to_bits()
        || compiler.rope_frequency_base().to_bits() != manifest.rope_frequency_base.to_bits()
    {
        return Err("artifact numerical configuration differs from referenced model".into());
    }
    compiler.use_artifact_vocabulary(Vocabulary::from_spec(manifest.tokenizer.clone())?);
    let required = required_projections(&compiler)?;
    if required.len() != manifest.projections.len() {
        return Err("artifact projection count mismatch".into());
    }
    let mut used = BTreeSet::new();
    for ((name, columns, rows), binding) in required.iter().zip(&manifest.projections) {
        let program = manifest
            .programs
            .get(binding.program)
            .ok_or("invalid program binding")?;
        if name != &binding.tensor
            || *columns != binding.columns
            || *rows != binding.rows
            || program.columns != *columns
        {
            return Err("artifact projection binding mismatch".into());
        }
        used.insert(binding.program);
    }
    if used.len() != manifest.programs.len() {
        return Err("unreferenced artifact program".into());
    }
    let mut graphs = Vec::with_capacity(manifest.programs.len());
    for (i, program) in manifest.programs.iter().enumerate() {
        let bytes = read_bounded(
            &directory.join(format!("program_{i}.bmlgraph")),
            MAX_PROGRAM_BYTES,
        )?;
        if hash(&bytes) != program.sha256 {
            return Err("stored RPN program hash mismatch".into());
        }
        graphs.push(BmlGraph::deserialize_with_limits(
            &bytes,
            MAX_PROGRAM_BYTES,
            64,
            4096,
            4096,
        )?);
    }
    Ok(LoadedArtifact {
        compiler,
        manifest,
        graphs,
    })
}
