//! Persisted AOT programs for the experimental symbolic causal profile.
//! Model weights remain external; source identity and all operation bindings
//! are checked without regenerating any stored arithmetic program on load.
use crate::{
    causal_artifact::{hash, read_bounded, required_projections},
    gguf_compiler::{InferenceCompiler, ModelConfig},
    symbolic_kernels::{
        compile_operation, is_symbolic_opcode, CompiledKernel, KernelLimits, Operation,
        OutputLayout,
    },
    tokenizer::Vocabulary,
    BmlGraph, RpnOp,
};
use bml_inference::tokenizer::TokenizerSpec;
use bml_parser::GgufParser;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
/// Filename distinguishing symbolic AOT artifacts from native causal artifacts.
pub const MANIFEST: &str = "symbolic.bmlgraph.json";
/// Versioned numerical profile for persisted BML-only causal arithmetic.
pub const PROFILE: &str = "llama-symbolic-bml-f64-f32kv-v2";
const LEGACY_PROFILE: &str = "llama-symbolic-bml-f64-f32kv-v1";
const MAX_MANIFEST: usize = 16 * 1024 * 1024;
const MAX_FILE: usize = 8 * 1024 * 1024;
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_PROGRAMS: usize = 4096;
const MAX_OPS: usize = 4_000_000;
const MAX_FRAGMENTS: usize = 65_536;
/// One operation template with an explicit output layout and fixed literals.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelDescriptor {
    /// BML operation family.
    pub operation: Operation,
    /// Vector width or canonical scalar-operation width.
    pub width: usize,
    /// Scalar programs in v1, or the canonical shared-output layout in v2.
    #[serde(default)]
    pub layout: OutputLayout,
    /// Fixed literal prefix, validated against the numerical profile.
    pub fixed: Vec<f64>,
    /// Hashes of kernel_INDEX_PROGRAM.bmlgraph files in execution order.
    pub programs: Vec<String>,
}
/// Complete source binding and finite-context AOT operation plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Container schema version.
    pub version: u32,
    /// Explicit symbolic numerical execution profile.
    pub profile: String,
    /// Default external source locator.
    pub source: PathBuf,
    /// SHA-256 of the entire immutable GGUF source.
    pub source_sha256: String,
    /// Validated source model dimensions.
    pub config: ModelConfig,
    /// Grouped KV-head count.
    pub kv_heads: u32,
    /// Learned-normalization epsilon.
    pub rms_norm_epsilon: f64,
    /// Rotary frequency base.
    pub rope_frequency_base: f64,
    /// Persisted tokenizer and token policy.
    pub tokenizer: TokenizerSpec,
    /// Maximum number of tokens supported by the stored causal operation plan.
    pub context_length: u32,
    /// Target fragment size used for these programs.
    pub fragment_bytes: usize,
    /// Exact required operation templates, in canonical order.
    pub kernels: Vec<KernelDescriptor>,
}
/// Loaded bytecode and source metadata before runtime stack/step validation.
pub struct LoadedArtifact {
    /// Verified mapped source and restored tokenizer.
    pub compiler: InferenceCompiler,
    /// Verified manifest.
    pub manifest: Manifest,
    /// Actual stored programs, never recompiled during loading.
    pub kernels: Vec<CompiledKernel>,
}
fn required_kernels(
    model: &InferenceCompiler,
    context: u32,
    program_limit: usize,
    shared_outputs: bool,
) -> Result<Vec<(Operation, usize)>, String> {
    if context == 0 || context > model.config().context_length {
        return Err("invalid symbolic artifact context".into());
    }
    let mut keys = BTreeSet::new();
    for (_, columns, _) in required_projections(model)? {
        keys.insert((Operation::Dot, columns));
    }
    if model.config().n_layers > 0 || model.weight_offsets().contains_key("output_norm.weight") {
        keys.insert((Operation::Norm, model.config().n_embd as usize));
    }
    if model.config().n_layers > 0 {
        let count = context as usize;
        let minimum = if shared_outputs {
            // One softmax program per prefix, before allocating the plan.
            count
        } else {
            count
                .checked_add(1)
                .and_then(|n| count.checked_mul(n))
                .map(|n| n / 2)
                .ok_or("symbolic context plan overflow")?
        };
        if minimum > program_limit {
            return Err("symbolic AOT program budget exceeded".into());
        }
        keys.extend([
            (Operation::Rope, 2),
            (Operation::Swiglu, 1),
            (Operation::Add, 1),
            (Operation::Scale, 1),
            (Operation::Dot, model.head_dim() as usize),
        ]);
        for width in 1..=count {
            keys.insert((Operation::Softmax, width));
            keys.insert((Operation::Dot, width));
        }
    }
    let mut outputs = 0usize;
    for &(op, width) in &keys {
        let programs = if shared_outputs {
            1
        } else {
            op.signature(width, model.rms_norm_epsilon())?.outputs
        };
        outputs = outputs
            .checked_add(programs)
            .ok_or("symbolic output count overflow")?;
        if outputs > program_limit {
            return Err("symbolic AOT program budget exceeded".into());
        }
    }
    Ok(keys.into_iter().collect())
}
fn check_output_directory(output: &Path) -> Result<(), String> {
    if output.exists()
        && fs::read_dir(output)
            .map_err(|e| e.to_string())?
            .next()
            .is_some()
    {
        return Err("output directory is not empty".into());
    }
    Ok(())
}
/// Compile every operation needed up to context_length and publish the manifest
/// last. Construction/byte budgets and a nonempty destination are hard errors.
/// This does not establish numerical evaluability for a particular prompt.
pub fn compile_directory(
    source: &Path,
    output: &Path,
    context_length: u32,
    mut limits: KernelLimits,
) -> Result<Manifest, String> {
    check_output_directory(output)?;
    limits.max_programs = limits.max_programs.min(MAX_PROGRAMS);
    limits.max_ops = limits.max_ops.min(MAX_OPS);
    if limits.fragment_bytes < std::mem::size_of::<RpnOp>() || limits.fragment_bytes > MAX_FILE {
        return Err("invalid symbolic fragment threshold".into());
    }
    let source = source.canonicalize().map_err(|e| e.to_string())?;
    let parser = GgufParser::open(&source).map_err(|e| e.to_string())?;
    let source_sha256 = hash(parser.bytes());
    let tokenizer = TokenizerSpec::from_gguf(&parser).map_err(|e| e.to_string())?;
    let model = InferenceCompiler::from_mapped_parser(parser)?;
    let required = required_kernels(&model, context_length, limits.max_programs, true)?;
    let mut descriptors = Vec::with_capacity(required.len());
    let mut encoded = Vec::with_capacity(required.len());
    let mut bytes_used = 0usize;
    let mut remaining = limits;
    for (operation, width) in required {
        let compiled = compile_operation(operation, width, model.rms_norm_epsilon(), remaining)?;
        if compiled.layout != OutputLayout::for_operation(operation)
            || compiled.signature
                != operation.signature_for_layout(
                    width,
                    model.rms_norm_epsilon(),
                    compiled.layout,
                )?
        {
            return Err("compiler symbolic signature mismatch".into());
        }
        remaining.max_programs -= compiled.graphs.len();
        let mut hashes = Vec::with_capacity(compiled.graphs.len());
        let mut files = Vec::with_capacity(compiled.graphs.len());
        for graph in compiled.graphs {
            if graph.fragments.len() > MAX_FRAGMENTS {
                return Err("symbolic fragment count exceeds budget".into());
            }
            remaining.max_ops -= graph.fragments.iter().map(|f| f.ops.len()).sum::<usize>();
            let bytes = graph.serialize();
            bytes_used = bytes_used
                .checked_add(bytes.len())
                .ok_or("symbolic byte count overflow")?;
            if bytes.len() > MAX_FILE || bytes_used > MAX_BYTES {
                return Err("symbolic artifact exceeds byte budget".into());
            }
            hashes.push(hash(&bytes));
            files.push(bytes);
        }
        descriptors.push(KernelDescriptor {
            operation,
            width,
            layout: compiled.layout,
            fixed: compiled.signature.fixed,
            programs: hashes,
        });
        encoded.push(files);
    }
    let manifest = Manifest {
        version: 2,
        profile: PROFILE.into(),
        source,
        source_sha256,
        config: model.config().clone(),
        kv_heads: model.n_kv_heads(),
        rms_norm_epsilon: model.rms_norm_epsilon(),
        rope_frequency_base: model.rope_frequency_base(),
        tokenizer,
        context_length,
        fragment_bytes: limits.fragment_bytes,
        kernels: descriptors,
    };
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_MANIFEST {
        return Err("symbolic manifest exceeds byte budget".into());
    }
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    check_output_directory(output)?;
    for (kernel, files) in encoded.iter().enumerate() {
        for (index, bytes) in files.iter().enumerate() {
            fs::write(
                output.join(format!("kernel_{kernel}_{index}.bmlgraph")),
                bytes,
            )
            .map_err(|e| e.to_string())?;
        }
    }
    fs::write(output.join(MANIFEST), bytes).map_err(|e| e.to_string())?;
    Ok(manifest)
}
/// Load a bounded artifact and verify source, metadata, literal bindings, hashes
/// and the complete operation plan. Runtime validation remains mandatory.
pub fn load_directory(
    directory: &Path,
    source_override: Option<&Path>,
) -> Result<LoadedArtifact, String> {
    let bytes = read_bounded(&directory.join(MANIFEST), MAX_MANIFEST)?;
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|e| format!("symbolic manifest: {e}"))?;
    let shared_outputs = match (manifest.version, manifest.profile.as_str()) {
        (1, LEGACY_PROFILE) => false,
        (2, PROFILE) => true,
        _ => return Err("unsupported symbolic artifact profile".into()),
    };
    if manifest.kernels.is_empty()
        || manifest.kernels.len() > MAX_PROGRAMS
        || manifest.fragment_bytes < std::mem::size_of::<RpnOp>()
        || manifest.fragment_bytes > MAX_FILE
    {
        return Err("unsupported symbolic artifact profile or size".into());
    }
    let count = manifest
        .kernels
        .iter()
        .try_fold(0usize, |n, k| n.checked_add(k.programs.len()))
        .ok_or("symbolic program count overflow")?;
    if count > MAX_PROGRAMS {
        return Err("symbolic program count exceeds budget".into());
    }
    let parser =
        GgufParser::open(source_override.unwrap_or(&manifest.source)).map_err(|e| e.to_string())?;
    if hash(parser.bytes()) != manifest.source_sha256 {
        return Err("referenced GGUF identity mismatch".into());
    }
    let tokenizer = TokenizerSpec::from_gguf(&parser).map_err(|e| e.to_string())?;
    if serde_json::to_vec(&tokenizer).map_err(|e| e.to_string())?
        != serde_json::to_vec(&manifest.tokenizer).map_err(|e| e.to_string())?
    {
        return Err("symbolic tokenizer differs from source".into());
    }
    let mut compiler = InferenceCompiler::from_mapped_parser(parser)?;
    if compiler.config() != &manifest.config
        || compiler.n_kv_heads() != manifest.kv_heads
        || compiler.rms_norm_epsilon().to_bits() != manifest.rms_norm_epsilon.to_bits()
        || compiler.rope_frequency_base().to_bits() != manifest.rope_frequency_base.to_bits()
    {
        return Err("symbolic numerical configuration differs from source".into());
    }
    compiler.use_artifact_vocabulary(Vocabulary::from_spec(manifest.tokenizer.clone())?);
    let required = required_kernels(
        &compiler,
        manifest.context_length,
        MAX_PROGRAMS,
        shared_outputs,
    )?;
    if required.len() != manifest.kernels.len() {
        return Err("symbolic operation plan mismatch".into());
    }
    let mut kernels = Vec::with_capacity(required.len());
    let mut bytes_used = 0usize;
    let mut ops_used = 0usize;
    for (index, (&(operation, width), descriptor)) in
        required.iter().zip(&manifest.kernels).enumerate()
    {
        let layout = if shared_outputs {
            OutputLayout::for_operation(operation)
        } else {
            OutputLayout::Scalars
        };
        // Check the bounded manifest's literal count before materializing the
        // output-index literals for a potentially large source-model width.
        let base_signature = operation.signature(width, compiler.rms_norm_epsilon())?;
        let fixed_count = base_signature
            .fixed
            .len()
            .checked_add(if layout == OutputLayout::Buffer {
                base_signature.outputs
            } else {
                0
            })
            .ok_or("symbolic literal count overflow")?;
        if descriptor.fixed.len() != fixed_count {
            return Err("symbolic kernel literal count mismatch".into());
        }
        let signature =
            operation.signature_for_layout(width, compiler.rms_norm_epsilon(), layout)?;
        let programs = if layout == OutputLayout::Buffer {
            1
        } else {
            signature.outputs
        };
        if descriptor.operation != operation
            || descriptor.width != width
            || descriptor.layout != layout
            || descriptor.fixed != signature.fixed
            || descriptor.programs.len() != programs
        {
            return Err("symbolic kernel binding mismatch".into());
        }
        let mut graphs = Vec::with_capacity(programs);
        for (output, digest) in descriptor.programs.iter().enumerate() {
            let bytes = read_bounded(
                &directory.join(format!("kernel_{index}_{output}.bmlgraph")),
                MAX_FILE.min(MAX_BYTES - bytes_used),
            )?;
            bytes_used += bytes.len();
            if hash(&bytes) != *digest {
                return Err("symbolic stored program hash mismatch".into());
            }
            let graph = BmlGraph::deserialize_with_limits(
                &bytes,
                manifest.fragment_bytes,
                MAX_FRAGMENTS,
                MAX_OPS - ops_used,
                0,
            )?;
            for fragment in &graph.fragments {
                if fragment.byte_size() > manifest.fragment_bytes
                    || !fragment.ops.iter().all(is_symbolic_opcode)
                {
                    return Err("invalid symbolic fragment or native opcode".into());
                }
                ops_used += fragment.ops.len();
            }
            graphs.push(graph);
        }
        kernels.push(CompiledKernel {
            operation,
            width,
            layout,
            signature,
            graphs,
        });
    }
    Ok(LoadedArtifact {
        compiler,
        manifest,
        kernels,
    })
}
