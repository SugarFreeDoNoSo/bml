//! Explicit finite-real operation profiles with compact indexed dot programs.
use crate::{
    linearize,
    rpn::linearize_outputs,
    symbolic_kernels::{
        finish_with_basis, is_symbolic_opcode, CompiledKernel, KernelLimits, KernelSignature,
        Operation, OutputLayout, RmsBounds,
    },
    RpnOp, RpnProgram,
};
use bml_domain::{transformer::BoundedOpConfig, BMLTransformer, NodeKind};
use serde::{Deserialize, Serialize};

#[path = "symbolic_bounded_rms.rs"]
mod rms;
pub(crate) use rms::{append_stage, compact_stage, compile_rms_program, linearize_stage};
#[path = "symbolic_bounded_softmax.rs"]
mod softmax;

/// Operand declarations for the opt-in complete bounded arithmetic profile.
/// These limits reject unsupported contexts; they do not promise evaluability
/// or uniform accuracy. Every arithmetic instruction is checked at execution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedProfile {
    /// RMS activation and learned weight bounds.
    pub rms: RmsBounds,
    /// Absolute input bound for learned matrix projections.
    pub projection_input: f64,
    /// Absolute learned matrix element bound.
    pub projection_weight: f64,
    /// Absolute query and key element bound.
    pub attention: f64,
    /// Absolute cached attention value bound.
    pub value: f64,
    /// Absolute unrotated query/key bound.
    pub rope: f64,
    /// Absolute SwiGLU gate bound.
    pub gate: f64,
    /// Absolute SwiGLU up-projection bound.
    pub up: f64,
    /// Absolute bound for each residual operand.
    pub residual: f64,
    /// Absolute unscaled attention-score bound.
    pub scale_input: f64,
    /// Absolute scaled softmax-score bound.
    pub score: f64,
}
impl Default for BoundedProfile {
    fn default() -> Self {
        Self {
            rms: RmsBounds {
                input: 16.0,
                weight: 16.0,
            },
            projection_input: 128.0,
            projection_weight: 2.0,
            attention: 16.0,
            value: 16.0,
            rope: 16.0,
            gate: 16.0,
            up: 16.0,
            residual: 32.0,
            scale_input: 256.0,
            score: 16.0,
        }
    }
}
impl BoundedProfile {
    /// New complete-profile vector kernels index their outputs directly.
    /// Historical Buffer signatures remain accepted when loading old v4 data.
    pub fn output_layout(op: Operation) -> OutputLayout {
        match op {
            Operation::Norm | Operation::Softmax => OutputLayout::IndexedBuffer,
            _ => OutputLayout::for_operation(op),
        }
    }
    /// Validate all declared families and model-dependent RMS parameters.
    pub fn validate(self, model_width: usize, epsilon: f64) -> Result<(), String> {
        self.rms.config(model_width, epsilon)?;
        for op in [
            Operation::Dot,
            Operation::AttentionDot,
            Operation::WeightedSum,
            Operation::Rope,
            Operation::Swiglu,
            Operation::Softmax,
            Operation::Add,
            Operation::Scale,
        ] {
            self.config(op, if op == Operation::Rope { 2 } else { 1 })?;
        }
        Ok(())
    }
    fn config(self, op: Operation, width: usize) -> Result<BoundedOpConfig, String> {
        let config = match op {
            Operation::Dot => {
                BoundedOpConfig::dot_step(self.projection_input, self.projection_weight)
            }
            Operation::AttentionDot => BoundedOpConfig::dot_step(self.attention, self.attention),
            Operation::WeightedSum => BoundedOpConfig::dot_step(1.0, self.value),
            Operation::Rope if width == 2 => BoundedOpConfig::rope(self.rope),
            Operation::Swiglu if width == 1 => BoundedOpConfig::swiglu(self.gate, self.up),
            Operation::Softmax => BoundedOpConfig::softmax(width, self.score),
            Operation::Add if width == 1 => BoundedOpConfig::add(self.residual, self.residual),
            Operation::Scale if width == 1 => BoundedOpConfig::product(self.scale_input, 1.0),
            _ => return Err("invalid bounded operation width".into()),
        };
        config.map_err(|e| e.to_string())
    }
    /// Validate every runtime operand, including the selected softmax shift.
    pub fn validate_values(
        self,
        op: Operation,
        width: usize,
        epsilon: f64,
        inputs: &[f64],
        external: &[f64],
    ) -> Result<(), String> {
        if op == Operation::Norm {
            return self
                .rms
                .config(width, epsilon)?
                .validate_values(inputs, external)
                .map_err(|e| e.to_string());
        }
        if is_dot(op) {
            if inputs.len() != width || external.len() != width || width == 0 {
                return Err("bounded dot context dimensions differ".into());
            }
            let (left, right) = match op {
                Operation::Dot => (self.projection_input, self.projection_weight),
                Operation::AttentionDot => (self.attention, self.attention),
                _ => (1.0, self.value),
            };
            if let Some(value) = inputs.iter().find(|x| !x.is_finite() || x.abs() > left) {
                return Err(format!(
                    "bounded dot operand input {value} exceeds declaration {left}"
                ));
            }
            if let Some(value) = external.iter().find(|x| !x.is_finite() || x.abs() > right) {
                return Err(format!(
                    "bounded dot operand weight {value} exceeds declaration {right}"
                ));
            }
            return Ok(());
        }
        self.config(op, width)?
            .validate_values(inputs, external)
            .map_err(|e| e.to_string())
    }
    /// Canonical dimensions and literals, independent of stored bytecode.
    pub fn base_signature(
        self,
        op: Operation,
        width: usize,
        epsilon: f64,
    ) -> Result<KernelSignature, String> {
        if width == 0 || width > u32::MAX as usize {
            return Err("invalid bounded symbolic width".into());
        }
        if op == Operation::Norm {
            return op.signature_with_rms_bounds(width, epsilon, Some(self.rms));
        }
        let config = self.config(op, width)?;
        let mut fixed = config.fixed_literals();
        let (inputs, external, outputs) = if is_dot(op) {
            fixed.push(0.0);
            (width, width, 1)
        } else {
            (config.inputs(), config.external(), config.outputs())
        };
        fixed
            .len()
            .checked_add(external)
            .filter(|&count| count as u64 <= u64::from(u32::MAX) + 1)
            .ok_or("bounded constant context index overflow")?;
        Ok(KernelSignature {
            inputs,
            external,
            outputs,
            fixed,
        })
    }
    /// Add output bindings after validating a bounded manifest's literal count.
    pub fn signature(
        self,
        op: Operation,
        width: usize,
        epsilon: f64,
        layout: OutputLayout,
    ) -> Result<KernelSignature, String> {
        let historical_buffer =
            layout == OutputLayout::Buffer && matches!(op, Operation::Norm | Operation::Softmax);
        if width == 0
            || width > u32::MAX as usize
            || (layout != Self::output_layout(op) && !historical_buffer)
        {
            return Err("invalid bounded symbolic shape/layout".into());
        }
        let mut signature = self.base_signature(op, width, epsilon)?;
        if layout == OutputLayout::Buffer {
            signature
                .fixed
                .len()
                .checked_add(signature.outputs)
                .filter(|&n| n <= u32::MAX as usize)
                .ok_or("output literal index overflow")?;
            signature
                .fixed
                .try_reserve_exact(signature.outputs)
                .map_err(|_| "cannot reserve output literals")?;
            signature
                .fixed
                .extend((0..signature.outputs).map(|i| i as f64));
        }
        Ok(signature)
    }
}
fn is_dot(op: Operation) -> bool {
    matches!(
        op,
        Operation::Dot | Operation::AttentionDot | Operation::WeightedSum
    )
}
/// BML arithmetic, stack/output instructions and bounded indexed iteration.
pub fn is_bounded_opcode(op: &RpnOp) -> bool {
    is_symbolic_opcode(op)
        || matches!(
            op,
            RpnOp::Loop { .. }
                | RpnOp::ContextVarIndexed { .. }
                | RpnOp::ContextConstIndexed { .. }
        )
}

fn dag(
    config: &BoundedOpConfig,
    capacity: usize,
) -> Result<(bml_domain::NodeSoA, Vec<u32>, Vec<u32>), String> {
    let mut builder = BMLTransformer::with_capacity(capacity);
    let inputs: Vec<_> = (0..config.inputs())
        .map(|i| builder.var(i as u32))
        .collect();
    let external: Vec<_> = (0..config.external())
        .map(|i| builder.external_constant(i as u32))
        .collect();
    let roots = builder
        .operation_bounded(&inputs, &external, config)
        .map_err(|e| e.to_string())?;
    Ok((builder.into_soa(), roots, external))
}

/// Compile the selected family. Dot storage is independent of vector width;
/// runtime work and context sizes remain bounded and linear in that width.
pub fn compile_operation(
    op: Operation,
    width: usize,
    epsilon: f64,
    profile: BoundedProfile,
    limits: KernelLimits,
) -> Result<CompiledKernel, String> {
    if op == Operation::Norm {
        let config = profile.rms.config(width, epsilon)?;
        let external_base = u32::try_from(config.fixed_literals().len())
            .map_err(|_| "bounded RMS constant index overflow")?;
        let program = compile_rms_program(&config, external_base, 0, limits)?;
        let signature = profile.signature(op, width, epsilon, OutputLayout::IndexedBuffer)?;
        return finish_with_basis(
            op,
            width,
            OutputLayout::IndexedBuffer,
            vec![program],
            signature,
            limits,
            true,
        );
    }
    if op == Operation::Softmax {
        let config = profile.config(op, width)?;
        let program = softmax::compile_softmax_program(&config, 0, limits)?;
        let signature = profile.signature(op, width, epsilon, OutputLayout::IndexedBuffer)?;
        return finish_with_basis(
            op,
            width,
            OutputLayout::IndexedBuffer,
            vec![program],
            signature,
            limits,
            true,
        );
    }
    if limits.max_programs == 0 || limits.fragment_bytes < std::mem::size_of::<RpnOp>() {
        return Err("bounded symbolic program/fragment budget exceeded".into());
    }
    let config = profile.config(op, width)?;
    let nodes = dag(&config, 0)?.0.len();
    if nodes > limits.max_nodes || nodes > u32::MAX as usize {
        return Err("symbolic node budget exceeded".into());
    }
    let layout = BoundedProfile::output_layout(op);
    let signature = profile.signature(op, width, epsilon, layout)?;
    let (mut soa, roots, external) = dag(&config, nodes)?;
    if soa.len() != nodes || soa.const_pool != config.fixed_literals() {
        return Err("bounded symbolic construction changed".into());
    }
    let prefix = soa.const_pool.len();
    for (index, &node) in external.iter().enumerate() {
        soa.kinds[node as usize] = NodeKind::Const(
            u32::try_from(signature.fixed.len() + index).map_err(|_| "constant index overflow")?,
        );
    }
    soa.const_pool = signature.fixed.clone();
    let mut program = if is_dot(op) {
        let step = linearize(&soa, roots[0]);
        let mut body = Vec::new();
        let mut height = 0usize;
        for instruction in step.ops {
            match instruction {
                RpnOp::Var(id) => {
                    let depth = u32::try_from(height + usize::from(id == 0))
                        .map_err(|_| "stack index overflow")?;
                    body.push(RpnOp::Pick { depth });
                    match id {
                        0 => {}
                        1 => body.push(RpnOp::ContextVarIndexed { base: 0 }),
                        2 => body.push(RpnOp::ContextConstIndexed {
                            base: signature.fixed.len() as u32,
                        }),
                        _ => return Err("invalid bounded dot variable".into()),
                    }
                    height += 1;
                }
                other => {
                    match other {
                        RpnOp::One | RpnOp::Const(_) | RpnOp::Pick { .. } | RpnOp::Dup => {
                            height += 1
                        }
                        RpnOp::Bml | RpnOp::Drop => {
                            height = height.checked_sub(1).ok_or("invalid dot stack")?
                        }
                        RpnOp::Swap => {}
                        _ => return Err("invalid bounded dot step opcode".into()),
                    }
                    body.push(other);
                }
            }
        }
        if height != 1 {
            return Err("bounded dot step must return one scalar".into());
        }
        body.extend([RpnOp::Swap, RpnOp::Drop, RpnOp::Swap, RpnOp::Drop]);
        let mut program = RpnProgram::new();
        program.push(RpnOp::Const(prefix as u32));
        program.push(RpnOp::Loop {
            count: u32::try_from(width).map_err(|_| "dot width overflow")?,
            body_len: u32::try_from(body.len()).map_err(|_| "dot body overflow")?,
        });
        program.ops.extend(body);
        program
    } else if layout == OutputLayout::Buffer {
        linearize_outputs(&soa, &roots, 0, prefix as u32)?
    } else {
        linearize(&soa, roots[0])
    };
    program.const_pool.clear();
    finish_with_basis(op, width, layout, vec![program], signature, limits, true)
}
