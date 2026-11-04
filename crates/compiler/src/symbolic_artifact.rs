//! Persisted AOT programs for the experimental symbolic causal profile.
//! Model weights remain external; source identity and all operation bindings
//! are checked without regenerating any stored arithmetic program on load.
use crate::{
    causal_artifact::{hash, read_bounded, required_projections},
    gguf_compiler::{InferenceCompiler, ModelConfig},
    symbolic_bounded::{is_bounded_opcode, BoundedProfile},
    symbolic_kernels::{
        compile_operation_with_rms_bounds, is_symbolic_opcode, CompiledKernel, KernelLimits,
        Operation, OutputLayout, RmsBounds,
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
/// Opt-in profile replacing RMSNorm only with explicitly bounded identities.
pub const BOUNDED_RMS_PROFILE: &str = "llama-symbolic-bml-bounded-rms-f64-f32kv-v3";
/// All activation arithmetic checked, with compact indexed BML projections.
pub const BOUNDED_PROFILE: &str = "llama-symbolic-bml-bounded-f64-f32kv-v4";
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
    /// Runtime-enforced RMS operand bounds, required only by the v3 profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_bounds: Option<RmsBounds>,
    /// Complete operand declarations for v4, mutually exclusive with v3 bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounded_profile: Option<BoundedProfile>,
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

/// Complete, bounded AOT bytecode ready to publish without recompiling.
///
/// The private encoded programs and manifest are checked together at construction.
/// Learned weights remain in the hash-identified GGUF; this object owns only
/// bytecode, operation literals and metadata. Construction does not certify the
/// numerical range of an eventual prompt.
#[derive(Debug)]
pub struct CompiledArtifact {
    manifest: Manifest,
    manifest_bytes: Vec<u8>,
    encoded: Vec<Vec<Vec<u8>>>,
}

impl CompiledArtifact {
    /// Source binding, operation plan and effective context limit.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Publish into a new or empty directory, writing the manifest last.
    /// A write failure can leave uncommitted program files in the destination.
    pub fn publish(&self, output: &Path) -> Result<(), String> {
        check_output_directory(output)?;
        fs::create_dir_all(output).map_err(|e| e.to_string())?;
        check_output_directory(output)?;
        for (kernel, files) in self.encoded.iter().enumerate() {
            for (index, bytes) in files.iter().enumerate() {
                fs::write(
                    output.join(format!("kernel_{kernel}_{index}.bmlgraph")),
                    bytes,
                )
                .map_err(|e| e.to_string())?;
            }
        }
        fs::write(output.join(MANIFEST), &self.manifest_bytes).map_err(|e| e.to_string())
    }
}
fn required_kernels(
    model: &InferenceCompiler,
    context: u32,
    program_limit: usize,
    shared_outputs: bool,
    bounded: bool,
    separate_down: bool,
) -> Result<Vec<(Operation, usize)>, String> {
    if context == 0 || context > model.config().context_length {
        return Err("invalid symbolic artifact context".into());
    }
    let mut keys = BTreeSet::new();
    for (name, columns, _) in required_projections(model)? {
        let operation = if separate_down && name.ends_with(".ffn_down.weight") {
            Operation::DownDot
        } else {
            Operation::Dot
        };
        keys.insert((operation, columns));
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
            (
                if bounded {
                    Operation::AttentionDot
                } else {
                    Operation::Dot
                },
                model.head_dim() as usize,
            ),
        ]);
        for width in 1..=count {
            keys.insert((Operation::Softmax, width));
            keys.insert((
                if bounded {
                    Operation::WeightedSum
                } else {
                    Operation::Dot
                },
                width,
            ));
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
    limits: KernelLimits,
) -> Result<Manifest, String> {
    check_output_directory(output)?;
    let compiled = compile(source, context_length, limits)?;
    compiled.publish(output)?;
    Ok(compiled.manifest)
}

/// Publish an opt-in bounded-RMS artifact. Other arithmetic remains unchanged;
/// successful compilation does not certify full-model numerical evaluability.
pub fn compile_directory_with_rms_bounds(
    source: &Path,
    output: &Path,
    context_length: u32,
    bounds: RmsBounds,
    limits: KernelLimits,
) -> Result<Manifest, String> {
    check_output_directory(output)?;
    let compiled = compile_with_rms_bounds(source, context_length, bounds, limits)?;
    compiled.publish(output)?;
    Ok(compiled.manifest)
}

/// Construct the complete AOT plan with an explicit bounded RMS profile.
pub fn compile_with_rms_bounds(
    source: &Path,
    context_length: u32,
    bounds: RmsBounds,
    limits: KernelLimits,
) -> Result<CompiledArtifact, String> {
    compile_context_with_rms_bounds(source, Some(context_length), Some(bounds), None, limits)
}

/// Compile every operation with the explicit complete bounded arithmetic profile.
pub fn compile_bounded(
    source: &Path,
    context_length: u32,
    profile: BoundedProfile,
    limits: KernelLimits,
) -> Result<CompiledArtifact, String> {
    compile_context_with_rms_bounds(source, Some(context_length), None, Some(profile), limits)
}

/// Publish a complete bounded-profile artifact without embedding model weights.
pub fn compile_directory_bounded(
    source: &Path,
    output: &Path,
    context_length: u32,
    profile: BoundedProfile,
    limits: KernelLimits,
) -> Result<Manifest, String> {
    check_output_directory(output)?;
    let compiled = compile_bounded(source, context_length, profile, limits)?;
    compiled.publish(output)?;
    Ok(compiled.manifest)
}

/// Construct every operation needed for an explicit causal context, with no
/// filesystem output. The same bytecode is published by [`compile_directory`].
pub fn compile(
    source: &Path,
    context_length: u32,
    limits: KernelLimits,
) -> Result<CompiledArtifact, String> {
    compile_with_context(source, Some(context_length), limits)
}

pub(crate) fn compile_with_context(
    source: &Path,
    context_length: Option<u32>,
    limits: KernelLimits,
) -> Result<CompiledArtifact, String> {
    compile_context_with_rms_bounds(source, context_length, None, None, limits)
}

fn compile_context_with_rms_bounds(
    source: &Path,
    context_length: Option<u32>,
    rms_bounds: Option<RmsBounds>,
    bounded_profile: Option<BoundedProfile>,
    mut limits: KernelLimits,
) -> Result<CompiledArtifact, String> {
    limits.max_programs = limits.max_programs.min(MAX_PROGRAMS);
    limits.max_ops = limits.max_ops.min(MAX_OPS);
    if limits.fragment_bytes < std::mem::size_of::<RpnOp>() || limits.fragment_bytes > MAX_FILE {
        return Err("invalid symbolic fragment threshold".into());
    }
    let source = source.canonicalize().map_err(|e| e.to_string())?;
    let parser = GgufParser::open(&source).map_err(|e| e.to_string())?;
    // Architecture rejection precedes tokenizer/weight loading at every entry.
    crate::gguf_compiler::read_model_config(&parser)?;
    let source_sha256 = hash(parser.bytes());
    let tokenizer = TokenizerSpec::from_gguf(&parser).map_err(|e| e.to_string())?;
    let model = InferenceCompiler::from_mapped_parser(parser)?;
    if let Some(bounds) = rms_bounds {
        bounds.config(model.config().n_embd as usize, model.rms_norm_epsilon())?;
    }
    if let Some(profile) = bounded_profile {
        profile.validate(model.config().n_embd as usize, model.rms_norm_epsilon())?;
    }
    let context_length = context_length.unwrap_or(model.config().context_length);
    let required = required_kernels(
        &model,
        context_length,
        limits.max_programs,
        true,
        bounded_profile.is_some(),
        bounded_profile.is_some_and(|profile| profile.down_projection.is_some()),
    )?;
    let mut descriptors = Vec::with_capacity(required.len());
    let mut encoded = Vec::with_capacity(required.len());
    let mut bytes_used = 0usize;
    let mut remaining = limits;
    for (operation, width) in required {
        let compiled = if let Some(profile) = bounded_profile {
            crate::symbolic_bounded::compile_operation(
                operation,
                width,
                model.rms_norm_epsilon(),
                profile,
                remaining,
            )?
        } else {
            compile_operation_with_rms_bounds(
                operation,
                width,
                model.rms_norm_epsilon(),
                OutputLayout::for_operation(operation),
                rms_bounds,
                remaining,
            )?
        };
        let signature = if let Some(profile) = bounded_profile {
            profile.signature(operation, width, model.rms_norm_epsilon(), compiled.layout)?
        } else {
            operation.signature_for_layout_with_rms_bounds(
                width,
                model.rms_norm_epsilon(),
                compiled.layout,
                rms_bounds,
            )?
        };
        let expected_layout = if bounded_profile.is_some() {
            BoundedProfile::output_layout(operation)
        } else {
            OutputLayout::for_operation(operation)
        };
        if compiled.layout != expected_layout || compiled.signature != signature {
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
        version: if bounded_profile.is_some() {
            4
        } else if rms_bounds.is_some() {
            3
        } else {
            2
        },
        profile: if bounded_profile.is_some() {
            BOUNDED_PROFILE
        } else if rms_bounds.is_some() {
            BOUNDED_RMS_PROFILE
        } else {
            PROFILE
        }
        .into(),
        rms_bounds,
        bounded_profile,
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
    Ok(CompiledArtifact {
        manifest,
        manifest_bytes: bytes,
        encoded,
    })
}
/// Verify a complete artifact and return the SHA-256 of its exact manifest bytes.
/// The manifest binds the source weights and every persisted kernel by hash.
/// Files must remain unchanged while the returned identity is in use; this
/// function rejects a manifest changed during validation.
pub fn checked_identity(
    directory: &Path,
    source_override: Option<&Path>,
) -> Result<String, String> {
    let before = read_bounded(&directory.join(MANIFEST), MAX_MANIFEST)?;
    let loaded = load_directory(directory, source_override)?;
    drop(loaded);
    let after = read_bounded(&directory.join(MANIFEST), MAX_MANIFEST)?;
    if before != after {
        return Err("symbolic manifest changed during identity validation".into());
    }
    Ok(hash(&before))
}

/// Load a bounded artifact and verify source, metadata, literal bindings, hashes
/// and the complete operation plan. Runtime validation remains mandatory.
pub fn load_directory(
    directory: &Path,
    source_override: Option<&Path>,
) -> Result<LoadedArtifact, String> {
    load_directory_inner(directory, source_override, None)
}

/// Load a bounded worker artifact using a manifest hash obtained from a trusted
/// coordinator. The hash must identify an artifact previously validated against
/// its complete source; this method does not revalidate source model contents.
pub fn load_directory_authenticated(
    directory: &Path,
    expected_identity: &str,
) -> Result<LoadedArtifact, String> {
    load_directory_inner(directory, None, Some(expected_identity))
}

fn load_directory_inner(
    directory: &Path,
    source_override: Option<&Path>,
    trusted_identity: Option<&str>,
) -> Result<LoadedArtifact, String> {
    let bytes = read_bounded(&directory.join(MANIFEST), MAX_MANIFEST)?;
    if trusted_identity.is_some_and(|expected| hash(&bytes) != expected) {
        return Err("symbolic manifest identity mismatch".into());
    }
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|e| format!("symbolic manifest: {e}"))?;
    let shared_outputs = match (
        manifest.version,
        manifest.profile.as_str(),
        manifest.rms_bounds,
        manifest.bounded_profile,
    ) {
        (1, LEGACY_PROFILE, None, None) => false,
        (2, PROFILE, None, None) => true,
        (3, BOUNDED_RMS_PROFILE, Some(_), None) => true,
        (4, BOUNDED_PROFILE, None, Some(_)) => true,
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
    let mut compiler = if trusted_identity.is_some() {
        if manifest.version != 4 || manifest.profile != BOUNDED_PROFILE {
            return Err("source-free workers require bounded profile v4".into());
        }
        InferenceCompiler::from_authenticated_manifest(&manifest)?
    } else {
        let parser = GgufParser::open(source_override.unwrap_or(&manifest.source))
            .map_err(|e| e.to_string())?;
        if hash(parser.bytes()) != manifest.source_sha256 {
            return Err("referenced GGUF identity mismatch".into());
        }
        let tokenizer = TokenizerSpec::from_gguf(&parser).map_err(|e| e.to_string())?;
        if serde_json::to_vec(&tokenizer).map_err(|e| e.to_string())?
            != serde_json::to_vec(&manifest.tokenizer).map_err(|e| e.to_string())?
        {
            return Err("symbolic tokenizer differs from source".into());
        }
        InferenceCompiler::from_mapped_parser(parser)?
    };
    if let Some(profile) = manifest.bounded_profile {
        profile.validate(
            compiler.config().n_embd as usize,
            compiler.rms_norm_epsilon(),
        )?;
    }
    if let Some(bounds) = manifest.rms_bounds {
        bounds.config(
            compiler.config().n_embd as usize,
            compiler.rms_norm_epsilon(),
        )?;
    }
    if compiler.config() != &manifest.config
        || compiler.n_kv_heads() != manifest.kv_heads
        || compiler.rms_norm_epsilon().to_bits() != manifest.rms_norm_epsilon.to_bits()
        || compiler.rope_frequency_base().to_bits() != manifest.rope_frequency_base.to_bits()
    {
        return Err("symbolic numerical configuration differs from source".into());
    }
    compiler.use_artifact_vocabulary(Vocabulary::from_spec(manifest.tokenizer.clone())?);
    let required = if trusted_identity.is_some() {
        if manifest.context_length == 0
            || manifest.context_length > compiler.config().context_length
        {
            return Err("invalid symbolic artifact context".into());
        }
        let mut keys = BTreeSet::new();
        for descriptor in &manifest.kernels {
            if descriptor.width == 0 || !keys.insert((descriptor.operation, descriptor.width)) {
                return Err("invalid symbolic operation plan".into());
            }
        }
        keys.into_iter().collect::<Vec<_>>()
    } else {
        required_kernels(
            &compiler,
            manifest.context_length,
            MAX_PROGRAMS,
            shared_outputs,
            manifest.bounded_profile.is_some(),
            manifest
                .bounded_profile
                .is_some_and(|profile| profile.down_projection.is_some()),
        )?
    };
    if required.len() != manifest.kernels.len() {
        return Err("symbolic operation plan mismatch".into());
    }
    let mut kernels = Vec::with_capacity(required.len());
    let mut bytes_used = 0usize;
    let mut ops_used = 0usize;
    for (index, (&(operation, width), descriptor)) in
        required.iter().zip(&manifest.kernels).enumerate()
    {
        let layout = if manifest.bounded_profile.is_some()
            && matches!(operation, Operation::Norm | Operation::Softmax)
            && descriptor.layout == OutputLayout::IndexedBuffer
        {
            OutputLayout::IndexedBuffer
        } else if shared_outputs {
            OutputLayout::for_operation(operation)
        } else {
            OutputLayout::Scalars
        };
        // Check the bounded manifest's literal count before materializing the
        // output-index literals for a potentially large source-model width.
        let base_signature = if let Some(profile) = manifest.bounded_profile {
            profile.base_signature(operation, width, compiler.rms_norm_epsilon())?
        } else {
            operation.signature_with_rms_bounds(
                width,
                compiler.rms_norm_epsilon(),
                manifest.rms_bounds,
            )?
        };
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
        let signature = if let Some(profile) = manifest.bounded_profile {
            profile.signature(operation, width, compiler.rms_norm_epsilon(), layout)?
        } else {
            operation.signature_for_layout_with_rms_bounds(
                width,
                compiler.rms_norm_epsilon(),
                layout,
                manifest.rms_bounds,
            )?
        };
        let programs = if layout.uses_buffer() {
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
                    || !fragment
                        .ops
                        .iter()
                        .all(if manifest.bounded_profile.is_some() {
                            is_bounded_opcode
                        } else {
                            is_symbolic_opcode
                        })
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
