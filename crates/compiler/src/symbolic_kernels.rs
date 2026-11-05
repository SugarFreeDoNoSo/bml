//! Compile BML-only operation templates independently of the runtime.
use crate::{
    fragment_program, linearize, rpn::linearize_outputs, symbolic_projection::compile_row,
    BmlGraph, RpnOp, RpnProgram,
};
use bml_domain::{BMLTransformer, NodeId, NodeKind, NodeSoA};
use serde::{Deserialize, Serialize};

/// Explicit operand bounds for the experimental bounded RMS arithmetic.
/// Other operation families retain their existing numerical identities.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RmsBounds {
    /// Maximum absolute activation supplied to each normalization.
    pub input: f64,
    /// Maximum absolute learned normalization weight.
    pub weight: f64,
}
impl RmsBounds {
    /// Validate the bounds and dimension-dependent arithmetic constants.
    pub fn config(
        self,
        width: usize,
        epsilon: f64,
    ) -> Result<bml_domain::transformer::BoundedRmsConfig, String> {
        bml_domain::transformer::BoundedRmsConfig::new(width, epsilon, self.input, self.weight)
            .map_err(|e| e.to_string())
    }
}

/// Arithmetic operation families used by the symbolic causal execution profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// Signed dot product with external weight constants.
    Dot,
    /// Bounded-profile MLP down-projection dot product.
    DownDot,
    /// Bounded-profile query/key dot product, with its own operand bounds.
    AttentionDot,
    /// Bounded-profile probability/value dot product.
    WeightedSum,
    /// Learned RMS normalization.
    Norm,
    /// Adjacent-pair rotary embedding.
    Rope,
    /// Natural-exponential softmax.
    Softmax,
    /// Standard Llama SwiGLU.
    Swiglu,
    /// Signed residual addition.
    Add,
    /// Signed activation multiplied by a positive scale.
    Scale,
}
impl Operation {
    /// Stable diagnostic label.
    pub fn name(self) -> &'static str {
        match self {
            Self::Dot => "dot",
            Self::DownDot => "down_dot",
            Self::AttentionDot => "attention_dot",
            Self::WeightedSum => "weighted_sum",
            Self::Norm => "rms_norm",
            Self::Rope => "rope",
            Self::Softmax => "softmax",
            Self::Swiglu => "swiglu",
            Self::Add => "residual_add",
            Self::Scale => "scale",
        }
    }
    /// Canonical context and literal bindings for this version of the profile.
    /// This validates metadata without regenerating any operation bytecode.
    pub fn signature(self, width: usize, epsilon: f64) -> Result<KernelSignature, String> {
        if width == 0 {
            return Err("empty symbolic operation width".into());
        }
        let (inputs, external, outputs, fixed) = match self {
            Self::Dot | Self::DownDot => (width, width, 1, vec![]),
            Self::Norm => {
                if !epsilon.is_finite() || epsilon <= 0.0 {
                    return Err("invalid symbolic RMS epsilon".into());
                }
                (width, width, width, vec![1.0 / width as f64, epsilon, 0.5])
            }
            Self::Softmax => (width, 0, width, vec![std::f64::consts::LOG2_E]),
            Self::Rope if width == 2 => (2, 2, 2, vec![]),
            Self::Swiglu if width == 1 => (2, 0, 1, vec![std::f64::consts::LOG2_E]),
            Self::Add | Self::Scale if width == 1 => (2, 0, 1, vec![]),
            _ => return Err("invalid symbolic operation width".into()),
        };
        Ok(KernelSignature {
            inputs,
            external,
            outputs,
            fixed,
        })
    }

    /// Context declaration for the explicitly selected RMS arithmetic.
    pub fn signature_with_rms_bounds(
        self,
        width: usize,
        epsilon: f64,
        bounds: Option<RmsBounds>,
    ) -> Result<KernelSignature, String> {
        let mut signature = self.signature(width, epsilon)?;
        if self == Self::Norm {
            if let Some(bounds) = bounds {
                signature.fixed = bounds.config(width, epsilon)?.fixed_literals();
            }
        }
        Ok(signature)
    }

    /// Canonical bindings for an explicit persisted output representation.
    /// Buffer output adds exact integer offsets after the operation literals;
    /// external weights follow that complete fixed prefix. This materializes
    /// O(width) literals: callers must apply their construction/storage budgets
    /// before requesting a buffered signature for an untrusted width.
    pub fn signature_for_layout(
        self,
        width: usize,
        epsilon: f64,
        layout: OutputLayout,
    ) -> Result<KernelSignature, String> {
        self.signature_for_layout_with_rms_bounds(width, epsilon, layout, None)
    }

    /// Output binding including the canonical bounded RMS literal prefix.
    pub fn signature_for_layout_with_rms_bounds(
        self,
        width: usize,
        epsilon: f64,
        layout: OutputLayout,
        bounds: Option<RmsBounds>,
    ) -> Result<KernelSignature, String> {
        if layout == OutputLayout::IndexedBuffer {
            return Err("indexed buffer layout requires the complete bounded profile".into());
        }
        let mut signature = self.signature_with_rms_bounds(width, epsilon, bounds)?;
        if layout == OutputLayout::Buffer {
            if OutputLayout::for_operation(self) != OutputLayout::Buffer {
                return Err("buffer output is unsupported for this symbolic operation".into());
            }
            signature.buffer_literal_count()?;
            signature
                .fixed
                .try_reserve_exact(signature.outputs)
                .map_err(|_| "cannot reserve symbolic output constants")?;
            signature
                .fixed
                .extend((0..signature.outputs).map(|i| i as f64));
        }
        Ok(signature)
    }
}

/// How the persisted program exposes logical scalar outputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputLayout {
    /// One graph and one returned scalar per logical output (legacy format).
    #[default]
    Scalars,
    /// One graph writes all logical outputs to result-buffer slot zero.
    Buffer,
    /// One indexed graph writes all outputs using loop counters. Only the
    /// complete bounded profile supports this representation; fixed literals
    /// contain configuration, without a width-sized array of output offsets.
    IndexedBuffer,
}
impl OutputLayout {
    /// Whether execution gathers outputs from result-buffer slot zero.
    pub fn uses_buffer(self) -> bool {
        matches!(self, Self::Buffer | Self::IndexedBuffer)
    }
    /// Default representation used for newly compiled operation programs.
    pub fn for_operation(operation: Operation) -> Self {
        match operation {
            Operation::Norm | Operation::Softmax | Operation::Rope => Self::Buffer,
            _ => Self::Scalars,
        }
    }
}
/// Constant and input dimensions independent of serialized program contents.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSignature {
    /// Activation values supplied on each execution.
    pub inputs: usize,
    /// External weight/coefficient values following the fixed literal prefix.
    pub external: usize,
    /// Number of logical scalar outputs, independent of stored program count.
    pub outputs: usize,
    /// Canonical fixed literals, never model matrix weights.
    pub fixed: Vec<f64>,
}
impl KernelSignature {
    /// Preflight the fixed literal count for materialized [`OutputLayout::Buffer`].
    ///
    /// Call this on a base signature from [`Operation::signature`] or
    /// [`Operation::signature_with_rms_bounds`], before appending output offsets.
    /// The count includes the existing fixed prefix and one offset per output;
    /// external weights are excluded. Checking arithmetic and the u32 count
    /// range allocates no storage. Success does not guarantee allocation,
    /// construction budgets, or that an operation supports this output layout.
    pub fn buffer_literal_count(&self) -> Result<usize, &'static str> {
        let count = self
            .fixed
            .len()
            .checked_add(self.outputs)
            .ok_or("symbolic output constant size overflow")?;
        if count > u32::MAX as usize {
            return Err("symbolic output constant index overflow");
        }
        Ok(count)
    }
}
/// Construction and storage budgets, independent of runtime execution budgets.
#[derive(Clone, Copy, Debug)]
pub struct KernelLimits {
    /// Maximum nodes in one operation DAG.
    pub max_nodes: usize,
    /// Maximum stored opcodes across the output programs.
    pub max_ops: usize,
    /// Maximum opcodes in any one persisted program.
    pub max_output_ops: usize,
    /// Maximum persisted programs.
    pub max_programs: usize,
    /// Target bytes per fragment.
    pub fragment_bytes: usize,
}
impl Default for KernelLimits {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_ops: 1_000_000,
            max_output_ops: 100_000,
            max_programs: 1024,
            fragment_bytes: 32768,
        }
    }
}
/// Compiled programs and their external context declaration.
pub struct CompiledKernel {
    /// Operation family.
    pub operation: Operation,
    /// Vector width or canonical scalar-operation width.
    pub width: usize,
    /// Declared activation, external constant and output dimensions.
    pub signature: KernelSignature,
    /// Whether outputs are returned individually or written to one buffer slot.
    pub layout: OutputLayout,
    /// Programs containing no owned constant pool; fixed literals are in signature.
    pub graphs: Vec<BmlGraph>,
}
/// True only for the extended BML basis, stack operations and output stores.
pub fn is_symbolic_opcode(op: &RpnOp) -> bool {
    matches!(
        op,
        RpnOp::One
            | RpnOp::Var(_)
            | RpnOp::Const(_)
            | RpnOp::Bml
            | RpnOp::Pick { .. }
            | RpnOp::Dup
            | RpnOp::Drop
            | RpnOp::Swap
            | RpnOp::StoreResult { .. }
    )
}
/// Compile an operation under explicit node, opcode, program and fragment budgets.
pub fn compile_operation(
    op: Operation,
    width: usize,
    epsilon: f64,
    limits: KernelLimits,
) -> Result<CompiledKernel, String> {
    compile_operation_with_layout(op, width, epsilon, OutputLayout::for_operation(op), limits)
}

/// Compile the same arithmetic using either the legacy scalar representation or
/// one shared vector program. Explicit layout is useful when validating older
/// artifacts against new programs without changing their mathematical DAGs.
pub fn compile_operation_with_layout(
    op: Operation,
    width: usize,
    epsilon: f64,
    layout: OutputLayout,
    limits: KernelLimits,
) -> Result<CompiledKernel, String> {
    compile_operation_with_rms_bounds(op, width, epsilon, layout, None, limits)
}

/// Compile an explicit arithmetic selection without changing legacy programs.
/// Bounds affect RMSNorm only; all other families retain their existing DAGs.
pub fn compile_operation_with_rms_bounds(
    op: Operation,
    width: usize,
    epsilon: f64,
    layout: OutputLayout,
    bounds: Option<RmsBounds>,
    limits: KernelLimits,
) -> Result<CompiledKernel, String> {
    let base_signature = op.signature_with_rms_bounds(width, epsilon, bounds)?;
    if layout == OutputLayout::Buffer && OutputLayout::for_operation(op) != layout {
        return Err("buffer output is unsupported for this symbolic operation".into());
    }
    if limits.fragment_bytes < std::mem::size_of::<RpnOp>() {
        return Err("symbolic fragment threshold too small".into());
    }
    let program_count = match layout {
        OutputLayout::Scalars => base_signature.outputs,
        OutputLayout::Buffer => 1,
        OutputLayout::IndexedBuffer => {
            return Err("indexed buffer layout requires the complete bounded profile".into());
        }
    };
    if program_count > limits.max_programs {
        return Err("symbolic program budget exceeded".into());
    }
    if op == Operation::Dot {
        let row = compile_row(width, limits.max_nodes)?;
        return finish(op, width, layout, vec![row.program], base_signature, limits);
    }

    // RMSNorm consists of a per-input square and output expansion, n-1 sum
    // additions, and a fixed normalization block. Softmax likewise has per-input
    // exponential/output expansions, n-1 additions, and a fixed reciprocal.
    // Thus their constructed node counts are affine for all nonempty widths.
    // Small probes measure the current identities, and check the increment,
    // before any user-width allocation. Fixed-width families use one exact probe.
    let nodes = required_nodes(op, width, epsilon, bounds)?;
    if nodes > limits.max_nodes || nodes > u32::MAX as usize {
        return Err("symbolic node budget exceeded".into());
    }
    let OperationDag {
        mut soa,
        roots,
        weights,
    } = build_dag(op, width, epsilon, nodes, bounds)?;
    if soa.len() != nodes || soa.len() > limits.max_nodes {
        return Err("symbolic construction node count changed".into());
    }
    if soa.const_pool != base_signature.fixed || weights.len() != base_signature.external {
        return Err("symbolic construction signature changed".into());
    }
    let output_constants_start =
        u32::try_from(soa.const_pool.len()).map_err(|_| "constant index overflow")?;
    let signature = op.signature_for_layout_with_rms_bounds(width, epsilon, layout, bounds)?;
    // Owned operation literals and output indices occupy a fixed prefix. Every
    // external weight is remapped after that prefix before either linearizer runs.
    soa.const_pool = signature.fixed.clone();
    for (index, &node) in weights.iter().enumerate() {
        let id = signature
            .fixed
            .len()
            .checked_add(index)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("constant index overflow")?;
        soa.kinds[node as usize] = NodeKind::Const(id);
    }
    let mut programs = Vec::with_capacity(program_count);
    match layout {
        OutputLayout::Buffer => {
            let mut program = linearize_outputs(&soa, &roots, 0, output_constants_start)?;
            program.const_pool.clear();
            programs.push(program);
        }
        OutputLayout::Scalars => {
            let mut stored = 0usize;
            for root in roots {
                let mut program = linearize(&soa, root);
                stored = stored
                    .checked_add(program.ops.len())
                    .ok_or("symbolic opcode overflow")?;
                if stored > limits.max_ops {
                    return Err("symbolic opcode cache budget exceeded".into());
                }
                program.const_pool.clear();
                programs.push(program);
            }
        }
        OutputLayout::IndexedBuffer => {
            return Err("indexed buffer layout requires the complete bounded profile".into());
        }
    }
    finish(op, width, layout, programs, signature, limits)
}

struct OperationDag {
    soa: NodeSoA,
    roots: Vec<NodeId>,
    weights: Vec<NodeId>,
}

fn required_nodes(
    op: Operation,
    width: usize,
    epsilon: f64,
    bounds: Option<RmsBounds>,
) -> Result<usize, String> {
    if matches!(op, Operation::Norm | Operation::Softmax) {
        let one = build_dag(op, 1, epsilon, 0, bounds)?.soa.len();
        let two = build_dag(op, 2, epsilon, 0, bounds)?.soa.len();
        let three = build_dag(op, 3, epsilon, 0, bounds)?.soa.len();
        let increment = two.checked_sub(one).ok_or("invalid symbolic node count")?;
        if three.checked_sub(two) != Some(increment) {
            return Err("symbolic vector node construction is not affine".into());
        }
        (width - 1)
            .checked_mul(increment)
            .and_then(|n| n.checked_add(one))
            .ok_or_else(|| "symbolic node size overflow".into())
    } else {
        Ok(build_dag(op, width, epsilon, 0, bounds)?.soa.len())
    }
}

fn build_dag(
    op: Operation,
    width: usize,
    epsilon: f64,
    capacity: usize,
    bounds: Option<RmsBounds>,
) -> Result<OperationDag, String> {
    let mut t = BMLTransformer::with_capacity(capacity);
    let mut weights = Vec::new();
    let roots = match op {
        Operation::Norm => {
            let x: Vec<_> = (0..width).map(|i| t.var(i as u32)).collect();
            weights = (0..width).map(|i| t.external_constant(i as u32)).collect();
            if let Some(bounds) = bounds {
                let config = bounds.config(width, epsilon)?;
                t.rms_norm_bounded(&x, &weights, &config)
                    .map_err(|e| e.to_string())?
            } else {
                t.rms_norm(&x, &weights, epsilon)
                    .map_err(|e| e.to_string())?
            }
        }
        Operation::Softmax => {
            let x: Vec<_> = (0..width).map(|i| t.var(i as u32)).collect();
            t.softmax(&x).map_err(|e| e.to_string())?
        }
        Operation::Rope => {
            let x = [t.var(0), t.var(1)];
            weights = vec![t.external_constant(0), t.external_constant(1)];
            t.rope(&x, &weights[..1], &weights[1..])
                .map_err(|e| e.to_string())?
        }
        Operation::Swiglu => {
            let g = t.var(0);
            let u = t.var(1);
            t.swiglu(&[g], &[u]).map_err(|e| e.to_string())?
        }
        Operation::Add | Operation::Scale => {
            let x = t.var(0);
            let y = t.var(1);
            vec![if op == Operation::Add {
                t.add(x, y)
            } else {
                t.scale_signed(x, y)
            }]
        }
        Operation::Dot | Operation::DownDot | Operation::AttentionDot | Operation::WeightedSum => {
            return Err("dot products require their row compiler".into())
        }
    };
    Ok(OperationDag {
        soa: t.into_soa(),
        roots,
        weights,
    })
}

fn finish(
    operation: Operation,
    width: usize,
    layout: OutputLayout,
    programs: Vec<RpnProgram>,
    signature: KernelSignature,
    limits: KernelLimits,
) -> Result<CompiledKernel, String> {
    finish_with_basis(operation, width, layout, programs, signature, limits, false)
}

pub(super) fn finish_with_basis(
    operation: Operation,
    width: usize,
    layout: OutputLayout,
    programs: Vec<RpnProgram>,
    signature: KernelSignature,
    limits: KernelLimits,
    indexed: bool,
) -> Result<CompiledKernel, String> {
    if layout == OutputLayout::IndexedBuffer
        && (!indexed || !matches!(operation, Operation::Norm | Operation::Softmax))
    {
        return Err("indexed buffer layout requires bounded RMS or softmax".into());
    }
    let expected = match layout {
        OutputLayout::Scalars => signature.outputs,
        OutputLayout::Buffer | OutputLayout::IndexedBuffer => 1,
    };
    if programs.len() != expected || programs.len() > limits.max_programs {
        return Err("symbolic program budget exceeded".into());
    }
    let mut graphs = Vec::with_capacity(programs.len());
    let mut ops = 0usize;
    for program in programs {
        if program.ops.len() > limits.max_output_ops {
            return Err("symbolic output opcode budget exceeded".into());
        }
        ops = ops
            .checked_add(program.ops.len())
            .ok_or("symbolic opcode overflow")?;
        if ops > limits.max_ops {
            return Err("symbolic opcode cache budget exceeded".into());
        }
        let valid_basis = if indexed {
            program
                .ops
                .iter()
                .all(crate::symbolic_bounded::is_bounded_opcode)
        } else {
            program.ops.iter().all(is_symbolic_opcode)
        };
        if !program.const_pool.is_empty() || !valid_basis {
            return Err("invalid symbolic basis or owned weights".into());
        }
        if program
            .ops
            .iter()
            .any(|op| matches!(op, RpnOp::Loop { .. }))
            && program
                .ops
                .len()
                .checked_mul(std::mem::size_of::<RpnOp>())
                .is_none_or(|bytes| bytes > limits.fragment_bytes)
        {
            return Err("symbolic loop body exceeds fragment budget".into());
        }
        graphs.push(fragment_program(&program, limits.fragment_bytes));
    }
    Ok(CompiledKernel {
        operation,
        width,
        signature,
        layout,
        graphs,
    })
}
