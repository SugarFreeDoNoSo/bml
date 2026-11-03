//! Experimental causal Llama execution with BML-only arithmetic programs.
//!
//! Embedding lookup, source-row decoding, F32 KV storage, and precomputation of
//! position/dimension constants remain host responsibilities. Every activation
//! operation (including attention products, scaling and residuals) executes a
//! BMLTransformer-derived program. The source-model constructor compiles lazy
//! templates under budgets; open_artifact loads a complete persisted AOT plan
//! and disables recompilation. IEEE hot-loop semantics and finite output checks
//! do not certify every real intermediate.
use crate::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
};
use bml_compiler::{
    gguf_compiler::InferenceCompiler,
    symbolic_bounded::{is_bounded_opcode, BoundedProfile},
    symbolic_kernels::{
        compile_operation_with_rms_bounds, is_symbolic_opcode, CompiledKernel, KernelLimits,
        Operation, OutputLayout, RmsBounds,
    },
    HashConsedKV,
};
use std::{collections::BTreeMap, path::Path};

/// Resource budgets for the experimental symbolic execution profile.
#[derive(Clone, Copy, Debug)]
pub struct SymbolicLimits {
    /// Maximum nodes in any constructed operation DAG.
    pub max_nodes: usize,
    /// Maximum stored opcodes across all cached output programs.
    pub max_cached_ops: usize,
    /// Maximum cached programs, including shared vector-output programs.
    pub max_programs: usize,
    /// Per-program runtime validation limits.
    pub graph: GraphLimits,
}
impl Default for SymbolicLimits {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_cached_ops: 1_000_000,
            max_programs: 1024,
            graph: GraphLimits::default(),
        }
    }
}
struct Kernel {
    programs: Vec<ValidatedGraph>,
    fixed: Vec<f64>,
    external: usize,
    outputs: usize,
    layout: OutputLayout,
    ops: usize,
    calls: usize,
    checked_real: bool,
    indexed: bool,
}
impl Kernel {
    fn compile(
        op: Operation,
        width: usize,
        epsilon: f64,
        rms_bounds: Option<RmsBounds>,
        bounded_profile: Option<BoundedProfile>,
        limits: SymbolicLimits,
        remaining: (usize, usize),
    ) -> Result<Self, String> {
        let kernel_limits = KernelLimits {
            max_nodes: limits.max_nodes,
            max_ops: remaining.0,
            max_output_ops: limits.graph.max_ops,
            max_programs: remaining.1,
            fragment_bytes: 32768,
        };
        let compiled = if let Some(profile) = bounded_profile {
            bml_compiler::symbolic_bounded::compile_operation(
                op,
                width,
                epsilon,
                profile,
                kernel_limits,
            )?
        } else {
            compile_operation_with_rms_bounds(
                op,
                width,
                epsilon,
                OutputLayout::for_operation(op),
                rms_bounds,
                kernel_limits,
            )?
        };
        Self::from_compiled(
            compiled,
            limits.graph,
            rms_bounds.is_some() && op == Operation::Norm,
            bounded_profile.is_some(),
        )
    }
    fn from_compiled(
        compiled: CompiledKernel,
        limits: GraphLimits,
        checked_real: bool,
        indexed: bool,
    ) -> Result<Self, String> {
        let signature = compiled.signature;
        let expected_programs = match compiled.layout {
            OutputLayout::Scalars => signature.outputs,
            OutputLayout::Buffer => 1,
        };
        if signature.outputs == 0 || compiled.graphs.len() != expected_programs {
            return Err("symbolic output count mismatch".into());
        }
        let constants = signature
            .fixed
            .len()
            .checked_add(signature.external)
            .ok_or("symbolic constant count overflow")?;
        let mut programs = Vec::with_capacity(compiled.graphs.len());
        let mut ops = 0usize;
        for graph in compiled.graphs {
            for fragment in &graph.fragments {
                if !fragment.ops.iter().all(if indexed {
                    is_bounded_opcode
                } else {
                    is_symbolic_opcode
                }) {
                    return Err("native arithmetic in symbolic program".into());
                }
                ops = ops
                    .checked_add(fragment.ops.len())
                    .ok_or("symbolic opcode overflow")?;
            }
            programs.push(
                ValidatedGraph::new_with_external_constants(
                    graph,
                    signature.inputs,
                    constants,
                    usize::from(compiled.layout == OutputLayout::Buffer),
                    if compiled.layout == OutputLayout::Buffer {
                        signature.outputs
                    } else {
                        0
                    },
                    limits,
                )
                .map_err(|e| format!("symbolic preflight: {e}"))?,
            );
        }
        Ok(Self {
            programs,
            fixed: signature.fixed,
            external: signature.external,
            outputs: signature.outputs,
            layout: compiled.layout,
            ops,
            calls: 0,
            checked_real,
            indexed,
        })
    }
    fn execute(&mut self, inputs: &[f64], external: &[f64]) -> Result<Vec<f64>, String> {
        if external.len() != self.external || inputs.iter().chain(external).any(|x| !x.is_finite())
        {
            return Err("invalid symbolic context".into());
        }
        let joined;
        let constants = if self.fixed.is_empty() {
            external
        } else {
            joined = self
                .fixed
                .iter()
                .chain(external)
                .copied()
                .collect::<Vec<_>>();
            &joined
        };
        let mut result = if self.layout == OutputLayout::Scalars {
            Vec::with_capacity(self.outputs)
        } else {
            Vec::new()
        };
        for program in &self.programs {
            let mut buffer = if self.layout == OutputLayout::Buffer {
                // Unwritten outputs must fail; zero-initialization would hide
                // missing/out-of-range stores in a persisted program.
                ResultBuffer::from_vec(1, self.outputs, vec![f64::NAN; self.outputs])
                    .map_err(|_| "symbolic output buffer dimensions")?
            } else {
                ResultBuffer::new(0, 0)
            };
            let value = if self.indexed {
                program.execute_indexed_symbolic_with_constants(inputs, constants, &mut buffer)
            } else if self.checked_real {
                program.execute_symbolic_with_constants(inputs, constants, &mut buffer)
            } else {
                program.execute_with_constants(inputs, constants, &mut buffer)
            }
            .map_err(|e| e.to_string())?;
            if !value.is_finite() {
                return Err("symbolic BML evaluation exceeded its numerical domain/range".into());
            }
            if self.layout == OutputLayout::Buffer {
                result = buffer.into_vec();
                if result.iter().any(|x| !x.is_finite()) {
                    return Err("symbolic outputs missing or outside numerical domain/range".into());
                }
            } else {
                result.push(value);
            }
        }
        self.calls += 1;
        Ok(result)
    }
}

/// A mapped model executed through budgeted BML-only arithmetic templates.
/// This experimental profile may reject inputs that native inference accepts.
pub struct SymbolicCausalModel {
    compiler: InferenceCompiler,
    kernels: BTreeMap<(Operation, usize), Kernel>,
    limits: SymbolicLimits,
    cached_ops: usize,
    cached_programs: usize,
    compiled_context: Option<u32>,
    profile: String,
    rms_bounds: Option<RmsBounds>,
    bounded_profile: Option<BoundedProfile>,
}
impl SymbolicCausalModel {
    /// Open an autonomous BMLGGUF model with explicit checked F32 weight evaluation.
    /// Every numeric weight must match the original term's exact rational target.
    /// Activations execute the existing BML-only causal RPN profile; no source
    /// GGUF, weight resynthesis or fallback to native activation kernels is used.
    pub fn open_bmlgguf(
        path: &Path,
        read_limits: bml_model_weights::ReadLimits,
        max_evaluated_weights: usize,
        evaluation: bml_compiler::gguf_compiler::prepared_model::WeightEvaluation,
        limits: SymbolicLimits,
    ) -> Result<Self, String> {
        let mut reader =
            bml_model_weights::WeightReader::open(path, read_limits).map_err(|e| e.to_string())?;
        let compiler =
            InferenceCompiler::from_bmlgguf(&mut reader, max_evaluated_weights, evaluation)?;
        Ok(Self {
            compiler,
            kernels: BTreeMap::new(),
            limits,
            cached_ops: 0,
            cached_programs: 0,
            compiled_context: None,
            profile: "bmlgguf-checked-exact-f32-symbolic-causal-v1".into(),
            rms_bounds: None,
            bounded_profile: None,
        })
    }
    /// Open a validated mapped Llama model. No matrix weights are expanded here.
    pub fn open(path: &Path, limits: SymbolicLimits) -> Result<Self, String> {
        Self::open_with_optional_rms_bounds(path, limits, None, None)
    }

    /// Opt into bounded RMSNorm while preserving the other symbolic identities.
    /// Every RMS input and learned weight is checked against the supplied bounds.
    pub fn open_with_rms_bounds(
        path: &Path,
        limits: SymbolicLimits,
        bounds: RmsBounds,
    ) -> Result<Self, String> {
        Self::open_with_optional_rms_bounds(path, limits, Some(bounds), None)
    }

    /// Open with checked arithmetic for all families and compact indexed dots.
    /// Softmax receives a copied maximum score as an auxiliary shift; all
    /// subtraction, exponential, reduction and division arithmetic stays BML.
    pub fn open_bounded(
        path: &Path,
        limits: SymbolicLimits,
        profile: BoundedProfile,
    ) -> Result<Self, String> {
        Self::open_with_optional_rms_bounds(path, limits, None, Some(profile))
    }

    fn open_with_optional_rms_bounds(
        path: &Path,
        limits: SymbolicLimits,
        rms_bounds: Option<RmsBounds>,
        bounded_profile: Option<BoundedProfile>,
    ) -> Result<Self, String> {
        let compiler = InferenceCompiler::open_mapped(path)?;
        if let Some(profile) = bounded_profile {
            profile.validate(
                compiler.config().n_embd as usize,
                compiler.rms_norm_epsilon(),
            )?;
        }
        if let Some(bounds) = rms_bounds {
            bounds.config(
                compiler.config().n_embd as usize,
                compiler.rms_norm_epsilon(),
            )?;
        }
        Ok(Self {
            compiler,
            kernels: BTreeMap::new(),
            limits,
            cached_ops: 0,
            cached_programs: 0,
            compiled_context: None,
            profile: if bounded_profile.is_some() {
                bml_compiler::symbolic_artifact::BOUNDED_PROFILE
            } else if rms_bounds.is_some() {
                bml_compiler::symbolic_artifact::BOUNDED_RMS_PROFILE
            } else {
                bml_compiler::symbolic_artifact::PROFILE
            }
            .into(),
            rms_bounds,
            bounded_profile,
        })
    }
    /// Load every persisted symbolic program and validate it before execution.
    /// Missing kernels never trigger recompilation in this mode. Source relocation
    /// is permitted only when the complete GGUF hash still matches the manifest.
    pub fn open_artifact(
        directory: &Path,
        source_override: Option<&Path>,
        limits: SymbolicLimits,
    ) -> Result<Self, String> {
        let artifact = bml_compiler::symbolic_artifact::load_directory(directory, source_override)?;
        let mut kernels = BTreeMap::new();
        let mut cached_ops = 0usize;
        let mut cached_programs = 0usize;
        for compiled in artifact.kernels {
            let key = (compiled.operation, compiled.width);
            let programs = compiled.graphs.len();
            let ops = compiled
                .graphs
                .iter()
                .try_fold(0usize, |sum, g| {
                    g.fragments
                        .iter()
                        .try_fold(sum, |n, f| n.checked_add(f.ops.len()))
                })
                .ok_or("symbolic opcode overflow")?;
            cached_ops = cached_ops
                .checked_add(ops)
                .ok_or("symbolic opcode overflow")?;
            cached_programs = cached_programs
                .checked_add(programs)
                .ok_or("symbolic program overflow")?;
            if cached_ops > limits.max_cached_ops || cached_programs > limits.max_programs {
                return Err("symbolic artifact runtime budget exceeded".into());
            }
            let checked_real = artifact.manifest.rms_bounds.is_some() && key.0 == Operation::Norm;
            kernels.insert(
                key,
                Kernel::from_compiled(
                    compiled,
                    limits.graph,
                    checked_real,
                    artifact.manifest.bounded_profile.is_some(),
                )?,
            );
        }
        Ok(Self {
            compiler: artifact.compiler,
            kernels,
            limits,
            cached_ops,
            cached_programs,
            compiled_context: Some(artifact.manifest.context_length),
            profile: artifact.manifest.profile,
            rms_bounds: artifact.manifest.rms_bounds,
            bounded_profile: artifact.manifest.bounded_profile,
        })
    }

    /// Read-only source configuration and tokenizer.
    pub fn compiler(&self) -> &InferenceCompiler {
        &self.compiler
    }
    /// Persisted profile identifier, or the current profile for a lazy model.
    pub fn profile(&self) -> &str {
        &self.profile
    }
    /// Effective context limit, including the coverage of a loaded AOT artifact.
    pub fn context_limit(&self) -> u32 {
        self.compiled_context
            .unwrap_or(self.compiler.config().context_length)
    }

    /// Operation families that have successfully executed, for diagnostics.
    pub fn executed_operations(&self) -> Vec<&'static str> {
        let mut names = std::collections::BTreeSet::new();
        for ((op, _), kernel) in &self.kernels {
            if kernel.calls > 0 {
                names.insert(op.name());
            }
        }
        names.into_iter().collect()
    }
    fn ensure_kernel(&mut self, op: Operation, width: usize) -> Result<(), String> {
        if let std::collections::btree_map::Entry::Vacant(entry) = self.kernels.entry((op, width)) {
            if self.compiled_context.is_some() {
                return Err("stored symbolic kernel missing; recompilation is disabled".into());
            }
            let kernel = Kernel::compile(
                op,
                width,
                self.compiler.rms_norm_epsilon(),
                self.rms_bounds,
                self.bounded_profile,
                self.limits,
                (
                    self.limits.max_cached_ops - self.cached_ops,
                    self.limits.max_programs - self.cached_programs,
                ),
            )?;
            self.cached_ops += kernel.ops;
            self.cached_programs += kernel.programs.len();
            entry.insert(kernel);
        }
        Ok(())
    }
    fn run(
        &mut self,
        op: Operation,
        width: usize,
        inputs: &[f64],
        constants: &[f64],
    ) -> Result<Vec<f64>, String> {
        let shifted;
        let inputs = if self.bounded_profile.is_some() && op == Operation::Softmax {
            let maximum = inputs
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .ok_or("empty bounded softmax")?;
            shifted = inputs
                .iter()
                .copied()
                .chain(std::iter::once(maximum))
                .collect::<Vec<_>>();
            &shifted
        } else {
            inputs
        };
        if let Some(profile) = self.bounded_profile {
            profile
                .validate_values(
                    op,
                    width,
                    self.compiler.rms_norm_epsilon(),
                    inputs,
                    constants,
                )
                .map_err(|e| format!("{}: {e}", op.name()))?;
        } else if op == Operation::Norm {
            if let Some(bounds) = self.rms_bounds {
                bounds
                    .config(width, self.compiler.rms_norm_epsilon())?
                    .validate_values(inputs, constants)
                    .map_err(|e| format!("rms_norm: {e}"))?;
            }
        }
        self.ensure_kernel(op, width)?;
        self.kernels
            .get_mut(&(op, width))
            .expect("kernel inserted")
            .execute(inputs, constants)
            .map_err(|e| format!("{}: {e}", op.name()))
    }
    fn project(&mut self, input: &[f64], name: &str) -> Result<Vec<f64>, String> {
        let width = input.len();
        let profile = self.bounded_profile;
        let epsilon = self.compiler.rms_norm_epsilon();
        self.ensure_kernel(Operation::Dot, width)?;
        let kernel = self
            .kernels
            .get_mut(&(Operation::Dot, width))
            .expect("kernel inserted");
        self.compiler.project_rows(input, name, |x, row| {
            let weights: Vec<_> = row.iter().map(|&v| f64::from(v)).collect();
            if let Some(profile) = profile {
                profile.validate_values(Operation::Dot, width, epsilon, x, &weights)?;
            }
            Ok(kernel.execute(x, &weights)?[0])
        })
    }
    fn norm(&mut self, input: &[f64], name: &str) -> Result<Vec<f64>, String> {
        let start = *self
            .compiler
            .weight_offsets()
            .get(name)
            .ok_or("missing normalization weights")? as usize;
        let end = start
            .checked_add(input.len())
            .ok_or("normalization range overflow")?;
        let weights: Vec<_> = self
            .compiler
            .weight_pool()
            .get(start..end)
            .ok_or("invalid normalization range")?
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        self.run(Operation::Norm, input.len(), input, &weights)
    }
    fn rotate(&mut self, input: &[f64], position: u32) -> Result<Vec<f64>, String> {
        let dim = self.compiler.head_dim() as usize;
        let base = self.compiler.rope_frequency_base();
        let mut output = Vec::with_capacity(input.len());
        for (index, pair) in input.as_chunks::<2>().0.iter().enumerate() {
            let within = (index * 2) % dim;
            let angle = position as f64 / base.powf(within as f64 / dim as f64);
            output.extend(self.run(Operation::Rope, 2, pair, &[angle.cos(), angle.sin()])?);
        }
        Ok(output)
    }
    fn residual(&mut self, left: &[f64], right: &[f64]) -> Result<Vec<f64>, String> {
        if left.len() != right.len() {
            return Err("symbolic residual dimensions differ".into());
        }
        left.iter()
            .zip(right)
            .map(|(&x, &y)| Ok(self.run(Operation::Add, 1, &[x, y], &[])?[0]))
            .collect()
    }
    /// Execute a complete prefix with a fresh bounded F32 KV cache.
    pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f64>, String> {
        let count = u32::try_from(ids.len()).map_err(|_| "context length overflow")?;
        if count == 0
            || count
                > self
                    .compiled_context
                    .unwrap_or(self.compiler.config().context_length)
        {
            return Err("invalid symbolic context length".into());
        }
        let mut cache = HashConsedKV::try_new(
            self.compiler.config().n_layers,
            self.compiler.n_kv_heads(),
            self.compiler.head_dim(),
            count,
            bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES,
        )?;
        self.forward_cached(ids, &mut cache)
    }
    /// Execute prefill/decode. Failures do not advance the committed KV position;
    /// uncommitted rows may be overwritten by a later retry, as in native mode.
    pub fn forward_cached(
        &mut self,
        ids: &[u32],
        cache: &mut HashConsedKV,
    ) -> Result<Vec<f64>, String> {
        let config = self.compiler.config().clone();
        let dim = self.compiler.head_dim() as usize;
        let kv_heads = self.compiler.n_kv_heads() as usize;
        if !cache.matches_layout(config.n_layers, kv_heads as u32, dim as u32) {
            return Err("KV cache layout does not match symbolic model".into());
        }
        let count = u32::try_from(ids.len()).map_err(|_| "context length overflow")?;
        let end = cache
            .current_pos()
            .checked_add(count)
            .ok_or("context position overflow")?;
        if ids.is_empty()
            || end
                > cache
                    .capacity()
                    .min(config.context_length)
                    .min(self.compiled_context.unwrap_or(config.context_length))
            || ids
                .iter()
                .any(|&id| id as usize >= self.compiler.vocab().len())
        {
            return Err("invalid symbolic causal input".into());
        }
        let start = cache.current_pos();
        let mut hidden = Vec::new();
        for (offset, &id) in ids.iter().enumerate() {
            let pos = start + offset as u32;
            hidden = self.compiler.get_embedding(id)?;
            for layer in 0..config.n_layers {
                let prefix = format!("blk.{layer}");
                let normalized = self.norm(&hidden, &format!("{prefix}.attn_norm.weight"))?;
                let q = self.project(&normalized, &format!("{prefix}.attn_q.weight"))?;
                let k = self.project(&normalized, &format!("{prefix}.attn_k.weight"))?;
                let v = self.project(&normalized, &format!("{prefix}.attn_v.weight"))?;
                let q = self.rotate(&q, pos)?;
                let k = self.rotate(&k, pos)?;
                for head in 0..kv_heads {
                    let range = head * dim..(head + 1) * dim;
                    let key: Vec<_> = k[range.clone()].iter().map(|&v| v as f32).collect();
                    let value: Vec<_> = v[range].iter().map(|&v| v as f32).collect();
                    if key.iter().chain(&value).any(|v| !v.is_finite()) {
                        return Err("non-finite symbolic KV value".into());
                    }
                    cache.store_k(layer, pos, head as u32, &key);
                    cache.store_v(layer, pos, head as u32, &value);
                }
                let mut attention = Vec::with_capacity(config.n_embd as usize);
                for head in 0..config.n_heads as usize {
                    let kv_head = head / (config.n_heads as usize / kv_heads);
                    let query = &q[head * dim..(head + 1) * dim];
                    let mut scores = Vec::with_capacity(pos as usize + 1);
                    for previous in 0..=pos {
                        let key: Vec<_> = cache
                            .load_k(layer, previous, kv_head as u32)
                            .into_iter()
                            .map(f64::from)
                            .collect();
                        let dot_op = if self.bounded_profile.is_some() {
                            Operation::AttentionDot
                        } else {
                            Operation::Dot
                        };
                        let dot = self.run(dot_op, dim, query, &key)?[0];
                        scores.push(
                            self.run(Operation::Scale, 1, &[dot, 1.0 / (dim as f64).sqrt()], &[])?
                                [0],
                        );
                    }
                    let probabilities = self.run(Operation::Softmax, scores.len(), &scores, &[])?;
                    for d in 0..dim {
                        let values: Vec<_> = (0..=pos)
                            .map(|previous| {
                                f64::from(cache.load_v(layer, previous, kv_head as u32)[d])
                            })
                            .collect();
                        attention.push(
                            self.run(
                                if self.bounded_profile.is_some() {
                                    Operation::WeightedSum
                                } else {
                                    Operation::Dot
                                },
                                probabilities.len(),
                                &probabilities,
                                &values,
                            )?[0],
                        );
                    }
                }
                let projected =
                    self.project(&attention, &format!("{prefix}.attn_output.weight"))?;
                hidden = self.residual(&hidden, &projected)?;
                let normalized = self.norm(&hidden, &format!("{prefix}.ffn_norm.weight"))?;
                let gate = self.project(&normalized, &format!("{prefix}.ffn_gate.weight"))?;
                let up = self.project(&normalized, &format!("{prefix}.ffn_up.weight"))?;
                let mut activated = Vec::with_capacity(gate.len());
                for (&g, &u) in gate.iter().zip(&up) {
                    activated.push(self.run(Operation::Swiglu, 1, &[g, u], &[])?[0]);
                }
                let down = self.project(&activated, &format!("{prefix}.ffn_down.weight"))?;
                hidden = self.residual(&hidden, &down)?;
            }
        }
        if self
            .compiler
            .weight_offsets()
            .contains_key("output_norm.weight")
        {
            hidden = self.norm(&hidden, "output_norm.weight")?;
        }
        let logits = self.project(&hidden, self.compiler.output_weight_name())?;
        cache.advance_by(count);
        Ok(logits)
    }
}
