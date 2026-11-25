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

#[derive(Clone, Copy)]
struct ArtifactBudgets {
    max_manifest: usize,
    max_file: usize,
    max_bytes: usize,
    max_programs: usize,
    max_ops: usize,
    max_fragments: usize,
}

const DEFAULT_BUDGETS: ArtifactBudgets = ArtifactBudgets {
    max_manifest: MAX_MANIFEST,
    max_file: MAX_FILE,
    max_bytes: MAX_BYTES,
    max_programs: MAX_PROGRAMS,
    max_ops: MAX_OPS,
    max_fragments: MAX_FRAGMENTS,
};
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
    if model.weight_offsets().contains_key("output_norm.weight") {
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
        if minimum >= program_limit {
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
    limits: KernelLimits,
) -> Result<CompiledArtifact, String> {
    compile_context_with_budgets(
        source,
        context_length,
        rms_bounds,
        bounded_profile,
        limits,
        DEFAULT_BUDGETS,
    )
}

fn compile_context_with_budgets(
    source: &Path,
    context_length: Option<u32>,
    rms_bounds: Option<RmsBounds>,
    bounded_profile: Option<BoundedProfile>,
    mut limits: KernelLimits,
    budgets: ArtifactBudgets,
) -> Result<CompiledArtifact, String> {
    limits.max_programs = limits.max_programs.min(budgets.max_programs);
    limits.max_ops = limits.max_ops.min(budgets.max_ops);
    if limits.fragment_bytes < std::mem::size_of::<RpnOp>()
        || limits.fragment_bytes > budgets.max_file
    {
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
        if compiled.layout != expected_layout {
            return Err("compiler symbolic signature mismatch".into());
        }
        if compiled.signature != signature {
            return Err("compiler symbolic signature mismatch".into());
        }
        remaining.max_programs = remaining
            .max_programs
            .checked_sub(compiled.graphs.len())
            .ok_or("symbolic AOT program budget exceeded")?;
        let mut hashes = Vec::with_capacity(compiled.graphs.len());
        let mut files = Vec::with_capacity(compiled.graphs.len());
        for graph in compiled.graphs {
            if graph.fragments.len() > budgets.max_fragments {
                return Err("symbolic fragment count exceeds budget".into());
            }
            remaining.max_ops -= graph.fragments.iter().map(|f| f.ops.len()).sum::<usize>();
            let bytes = graph.serialize();
            bytes_used = bytes_used
                .checked_add(bytes.len())
                .ok_or("symbolic byte count overflow")?;
            if bytes.len() > budgets.max_file || bytes_used > budgets.max_bytes {
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
    if bytes.len() > budgets.max_manifest {
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
    load_directory_with_budgets(
        directory,
        source_override,
        trusted_identity,
        DEFAULT_BUDGETS,
    )
}

fn load_directory_with_budgets(
    directory: &Path,
    source_override: Option<&Path>,
    trusted_identity: Option<&str>,
    budgets: ArtifactBudgets,
) -> Result<LoadedArtifact, String> {
    let bytes = read_bounded(&directory.join(MANIFEST), budgets.max_manifest)?;
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
        || manifest.kernels.len() > budgets.max_programs
        || manifest.fragment_bytes < std::mem::size_of::<RpnOp>()
        || manifest.fragment_bytes > budgets.max_file
    {
        return Err("unsupported symbolic artifact profile or size".into());
    }
    let count = manifest
        .kernels
        .iter()
        .try_fold(0usize, |n, k| n.checked_add(k.programs.len()))
        .ok_or("symbolic program count overflow")?;
    if count > budgets.max_programs {
        return Err("symbolic program count exceeds budget".into());
    }
    let mut compiler = if trusted_identity.is_some() {
        if manifest.version != 4 {
            return Err("source-free workers require bounded profile v4".into());
        }
        if manifest.profile != BOUNDED_PROFILE {
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
            budgets.max_programs,
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
                budgets.max_file.min(budgets.max_bytes - bytes_used),
            )?;
            bytes_used += bytes.len();
            if hash(&bytes) != *digest {
                return Err("symbolic stored program hash mismatch".into());
            }
            let graph = BmlGraph::deserialize_with_limits(
                &bytes,
                manifest.fragment_bytes,
                budgets.max_fragments,
                budgets.max_ops - ops_used,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempFile(PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn gguf_string(value: &str) -> Vec<u8> {
        let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
        bytes.extend(value.as_bytes());
        bytes
    }

    fn gguf_array(kind: u32, values: Vec<Vec<u8>>) -> Vec<u8> {
        let mut bytes = kind.to_le_bytes().to_vec();
        bytes.extend((values.len() as u64).to_le_bytes());
        bytes.extend(values.into_iter().flatten());
        bytes
    }

    fn mini_gguf(include_output_norm: bool) -> TempFile {
        let mut tokens = vec!["<unk>".into(), "<s>".into(), "</s>".into(), "▁".into()];
        tokens.extend((0..256).map(|b| format!("<0x{b:02X}>")));
        let vocab = tokens.len();
        let mut meta = vec![
            ("general.architecture", 8u32, gguf_string("llama")),
            ("tokenizer.ggml.model", 8, gguf_string("llama")),
            (
                "tokenizer.ggml.tokens",
                9,
                gguf_array(8, tokens.iter().map(|t| gguf_string(t)).collect()),
            ),
            (
                "tokenizer.ggml.scores",
                9,
                gguf_array(
                    6,
                    (0..vocab).map(|_| (-1f32).to_le_bytes().to_vec()).collect(),
                ),
            ),
            (
                "tokenizer.ggml.token_type",
                9,
                gguf_array(
                    5,
                    (0..vocab)
                        .map(|i| {
                            let kind: i32 = match i {
                                0 => 2,
                                1 | 2 => 3,
                                3 => 1,
                                _ => 6,
                            };
                            kind.to_le_bytes().to_vec()
                        })
                        .collect(),
                ),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon",
                6,
                0.5f32.to_le_bytes().to_vec(),
            ),
            ("llama.rope.freq_base", 6, 16f32.to_le_bytes().to_vec()),
        ];
        for (name, value) in [
            ("llama.block_count", 1u32),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 1),
            ("llama.embedding_length", 4),
            ("llama.feed_forward_length", 4),
            ("llama.context_length", 8),
            ("llama.rope.dimension_count", 2),
            ("llama.vocab_size", vocab as u32),
            ("tokenizer.ggml.bos_token_id", 1),
            ("tokenizer.ggml.eos_token_id", 2),
            ("tokenizer.ggml.unknown_token_id", 0),
        ] {
            meta.push((name, 4, value.to_le_bytes().to_vec()));
        }
        let mut weights: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            (
                "token_embd.weight",
                vec![4, vocab as u64],
                vec![0.1; 4 * vocab],
            ),
            ("output.weight", vec![4, vocab as u64], vec![0.1; 4 * vocab]),
        ];
        if include_output_norm {
            weights.push(("output_norm.weight", vec![4], vec![0.5; 4]));
        }
        weights.extend([
            ("blk.0.attn_norm.weight", vec![4], vec![0.5; 4]),
            ("blk.0.ffn_norm.weight", vec![4], vec![0.5; 4]),
            ("blk.0.attn_q.weight", vec![4, 4], vec![0.1; 16]),
            ("blk.0.attn_k.weight", vec![4, 2], vec![0.1; 8]),
            ("blk.0.attn_v.weight", vec![4, 2], vec![0.1; 8]),
            ("blk.0.attn_output.weight", vec![4, 4], vec![0.1; 16]),
            ("blk.0.ffn_gate.weight", vec![4, 4], vec![0.1; 16]),
            ("blk.0.ffn_up.weight", vec![4, 4], vec![0.1; 16]),
            ("blk.0.ffn_down.weight", vec![4, 4], vec![0.1; 16]),
        ]);
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend((weights.len() as u64).to_le_bytes());
        bytes.extend((meta.len() as u64).to_le_bytes());
        for (name, kind, value) in meta {
            bytes.extend(gguf_string(name));
            bytes.extend(kind.to_le_bytes());
            bytes.extend(value);
        }
        let mut data = Vec::new();
        for (name, dims, values) in weights {
            data.resize(data.len().div_ceil(32) * 32, 0);
            bytes.extend(gguf_string(name));
            bytes.extend((dims.len() as u32).to_le_bytes());
            for dim in dims {
                bytes.extend(dim.to_le_bytes());
            }
            bytes.extend(0u32.to_le_bytes());
            bytes.extend((data.len() as u64).to_le_bytes());
            data.extend(values.into_iter().flat_map(f32::to_le_bytes));
        }
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        bytes.extend(data);
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_symbolic_artifact_unit_{}_{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).unwrap();
        TempFile(path)
    }

    #[test]
    fn required_kernels_covers_layer_norm_without_output_norm_and_exact_scalar_program_limit() {
        let gguf = mini_gguf(true);
        let model = InferenceCompiler::open_mapped(&gguf.0).unwrap();
        let all_keys = required_kernels(&model, 8, usize::MAX, false, false, false).unwrap();
        let total_outputs: usize = all_keys
            .iter()
            .map(|&(op, width)| {
                op.signature(width, model.rms_norm_epsilon())
                    .unwrap()
                    .outputs
            })
            .sum();
        // Context 8 => 8*9/2 = 36 <= total_outputs < 8*9*2 = 144, catching / -> * as well as > -> ==/>.
        assert!(total_outputs < 8 * 9 * 2);
        assert_eq!(
            required_kernels(&model, 8, total_outputs, false, false, false).unwrap(),
            all_keys
        );
        assert_eq!(
            required_kernels(&model, 8, total_outputs - 1, false, false, false).unwrap_err(),
            "symbolic AOT program budget exceeded"
        );
    }

    #[test]
    fn compile_and_authenticated_load_enforce_exact_file_total_manifest_context_and_op_budgets() {
        let gguf = mini_gguf(true);

        // Public fragment_bytes exact upper boundary MAX_FILE (ok) and MAX_FILE + 1 (err).
        assert!(compile(
            &gguf.0,
            1,
            KernelLimits {
                fragment_bytes: MAX_FILE,
                ..Default::default()
            },
        )
        .is_ok());
        assert_eq!(
            compile(
                &gguf.0,
                1,
                KernelLimits {
                    fragment_bytes: MAX_FILE + 1,
                    ..Default::default()
                },
            )
            .err()
            .unwrap(),
            "invalid symbolic fragment threshold"
        );

        // Exact compile byte/file/manifest boundaries using parameterized budgets.
        let small_limits = KernelLimits {
            fragment_bytes: 512,
            ..Default::default()
        };
        let baseline = compile_context_with_budgets(
            &gguf.0,
            Some(2),
            None,
            None,
            small_limits,
            DEFAULT_BUDGETS,
        )
        .unwrap();
        let max_kernel_bytes = baseline
            .encoded
            .iter()
            .flatten()
            .map(Vec::len)
            .max()
            .unwrap();
        let total_kernel_bytes: usize = baseline.encoded.iter().flatten().map(Vec::len).sum();
        let manifest_len = baseline.manifest_bytes.len();
        assert!(max_kernel_bytes >= small_limits.fragment_bytes);
        assert!(total_kernel_bytes > max_kernel_bytes);

        let exact_compile_budgets = ArtifactBudgets {
            max_file: max_kernel_bytes,
            max_bytes: total_kernel_bytes,
            max_manifest: manifest_len,
            ..DEFAULT_BUDGETS
        };
        assert!(compile_context_with_budgets(
            &gguf.0,
            Some(2),
            None,
            None,
            small_limits,
            exact_compile_budgets,
        )
        .is_ok());
        assert_eq!(
            compile_context_with_budgets(
                &gguf.0,
                Some(2),
                None,
                None,
                small_limits,
                ArtifactBudgets {
                    max_file: max_kernel_bytes - 1,
                    ..exact_compile_budgets
                },
            )
            .err()
            .unwrap(),
            "symbolic artifact exceeds byte budget"
        );
        assert_eq!(
            compile_context_with_budgets(
                &gguf.0,
                Some(2),
                None,
                None,
                small_limits,
                ArtifactBudgets {
                    max_bytes: total_kernel_bytes - 1,
                    ..exact_compile_budgets
                },
            )
            .err()
            .unwrap(),
            "symbolic artifact exceeds byte budget"
        );
        assert_eq!(
            compile_context_with_budgets(
                &gguf.0,
                Some(2),
                None,
                None,
                small_limits,
                ArtifactBudgets {
                    max_manifest: manifest_len - 1,
                    ..exact_compile_budgets
                },
            )
            .err()
            .unwrap(),
            "symbolic manifest exceeds byte budget"
        );

        // Authenticated load at exact model context_length (8) and exact total op budget.
        let dir = TempDir(gguf.0.with_extension("auth-budget-dir"));
        fs::create_dir(&dir.0).unwrap();
        compile_directory_bounded(
            &gguf.0,
            &dir.0,
            8,
            BoundedProfile::default(),
            KernelLimits::default(),
        )
        .unwrap();
        let identity = checked_identity(&dir.0, None).unwrap();
        let loaded = load_directory_authenticated(&dir.0, &identity).unwrap();
        assert_eq!(loaded.manifest.context_length, 8);
        let total_ops: usize = loaded
            .kernels
            .iter()
            .flat_map(|k| &k.graphs)
            .flat_map(|g| &g.fragments)
            .map(|f| f.ops.len())
            .sum();
        assert!(total_ops > 10);
        assert!(load_directory_with_budgets(
            &dir.0,
            None,
            Some(&identity),
            ArtifactBudgets {
                max_ops: total_ops,
                ..DEFAULT_BUDGETS
            },
        )
        .is_ok());
        assert!(load_directory_with_budgets(
            &dir.0,
            None,
            Some(&identity),
            ArtifactBudgets {
                max_ops: total_ops - 1,
                ..DEFAULT_BUDGETS
            },
        )
        .is_err());

        // Exact max_fragments, max_programs, max_file, and max_bytes on load & compile.
        let max_graph_fragments = loaded
            .kernels
            .iter()
            .flat_map(|k| &k.graphs)
            .map(|g| g.fragments.len())
            .max()
            .unwrap();
        let total_programs: usize = loaded
            .manifest
            .kernels
            .iter()
            .map(|k| k.programs.len())
            .sum();
        let mut disk_file_sizes = Vec::new();
        for (index, k) in loaded.manifest.kernels.iter().enumerate() {
            for output in 0..k.programs.len() {
                disk_file_sizes.push(
                    fs::metadata(dir.0.join(format!("kernel_{index}_{output}.bmlgraph")))
                        .unwrap()
                        .len() as usize,
                );
            }
        }
        let max_disk_file = *disk_file_sizes.iter().max().unwrap();
        let total_disk_bytes: usize = disk_file_sizes.iter().sum();

        let baseline_max_fragments = baseline
            .encoded
            .iter()
            .flatten()
            .map(|bytes| {
                BmlGraph::deserialize_with_limits(
                    bytes,
                    small_limits.fragment_bytes,
                    MAX_FRAGMENTS,
                    MAX_OPS,
                    0,
                )
                .unwrap()
                .fragments
                .len()
            })
            .max()
            .unwrap();
        assert!(compile_context_with_budgets(
            &gguf.0,
            Some(2),
            None,
            None,
            small_limits,
            ArtifactBudgets {
                max_fragments: baseline_max_fragments,
                ..exact_compile_budgets
            },
        )
        .is_ok());
        assert_eq!(
            compile_context_with_budgets(
                &gguf.0,
                Some(2),
                None,
                None,
                small_limits,
                ArtifactBudgets {
                    max_fragments: baseline_max_fragments - 1,
                    ..exact_compile_budgets
                },
            )
            .err()
            .unwrap(),
            "symbolic fragment count exceeds budget"
        );

        let exact_load_budgets = ArtifactBudgets {
            max_file: max_disk_file.max(loaded.manifest.fragment_bytes),
            max_bytes: total_disk_bytes,
            max_programs: total_programs,
            max_ops: total_ops,
            max_fragments: max_graph_fragments,
            ..DEFAULT_BUDGETS
        };
        assert!(
            load_directory_with_budgets(&dir.0, None, Some(&identity), exact_load_budgets).is_ok()
        );
        assert!(load_directory_with_budgets(
            &dir.0,
            None,
            Some(&identity),
            ArtifactBudgets {
                max_bytes: total_disk_bytes - 1,
                ..exact_load_budgets
            },
        )
        .is_err());
        assert!(load_directory_with_budgets(
            &dir.0,
            None,
            Some(&identity),
            ArtifactBudgets {
                max_programs: total_programs - 1,
                ..exact_load_budgets
            },
        )
        .is_err());
        assert!(load_directory_with_budgets(
            &dir.0,
            None,
            Some(&identity),
            ArtifactBudgets {
                max_fragments: max_graph_fragments - 1,
                ..exact_load_budgets
            },
        )
        .is_err());
    }
}
