//! Complete RMS operation fragments with explicit literal ownership.

use super::{FragmentMeta, OperationFragment};
use crate::{
    symbolic_kernels::{KernelLimits, RmsBounds},
    Fragment, RpnOp,
};
use bml_domain::{transformer::BoundedRmsConfig, BMLTransformer, NodeKind, NodeSoA};

#[derive(Clone, Debug)]
struct Stage {
    dag: NodeSoA,
    roots: Vec<u32>,
}
impl Stage {
    fn evaluate(&self, inputs: &[f64], values: &mut [f64]) -> Result<(), String> {
        for (node, kind) in self.dag.kinds.iter().enumerate() {
            values[node] = match *kind {
                NodeKind::One => 1.0,
                NodeKind::Zero => 0.0,
                NodeKind::Var(id) => inputs[id as usize],
                NodeKind::Const(id) => self.dag.const_pool[id as usize],
                NodeKind::Bml => bml_domain::symbolic::Basis::Bml
                    .apply_real(
                        values[self.dag.lefts[node] as usize],
                        values[self.dag.rights[node] as usize],
                    )
                    .map_err(|error| format!("RMS BML node {node}: {error}"))?,
            };
        }
        Ok(())
    }
}

/// A complete learned RMS normalization with its numerical declarations,
/// configuration constants and finite-real preflight.
///
/// All input and weight addresses are absolute `ResultBuffer` element bases;
/// output identifies a slot. The caller must provide the same values used by
/// preflight and reserve sufficient disjoint input, weight and output storage.
/// This is a partial BML numerical contract, not universal IEEE arithmetic.
#[derive(Clone, Debug)]
pub struct CheckedRmsFragment {
    operation: OperationFragment,
    config: BoundedRmsConfig,
    stages: [Stage; 3],
}
impl CheckedRmsFragment {
    /// Immutable bytecode, layout and owned constants for execution.
    pub fn operation(&self) -> &OperationFragment {
        &self.operation
    }
    /// Consume this wrapper while retaining the operation's constant pool.
    pub fn into_operation(self) -> OperationFragment {
        self.operation
    }
    /// Complete fixed configuration pool, without model weights.
    pub fn constants(&self) -> &[f64] {
        self.operation
            .const_pool
            .as_deref()
            .expect("RMS owns its configuration")
    }
    /// Explicit width, epsilon and numerical bounds.
    pub fn config(&self) -> &BoundedRmsConfig {
        &self.config
    }

    /// Check operand declarations and every BML intermediate before writes.
    /// Scratch storage is independent of vector width. This evaluates the same
    /// staged expression as the bytecode; it does not substitute native RMS.
    pub fn validate_values(&self, inputs: &[f64], weights: &[f64]) -> Result<(), String> {
        self.config
            .validate_values(inputs, weights)
            .map_err(|e| e.to_string())?;
        let [reduce, prepare, output] = &self.stages;
        let capacity = self
            .stages
            .iter()
            .map(|stage| stage.dag.len())
            .max()
            .unwrap();
        let mut values = vec![0.0; capacity];
        let mut total = self.constants()[4];
        for (index, &input) in inputs.iter().enumerate() {
            reduce
                .evaluate(&[total, input], &mut values)
                .map_err(|error| format!("RMS reduce element {index}: {error}"))?;
            total = values[reduce.roots[0] as usize];
        }
        prepare.evaluate(&[total], &mut values)?;
        let inverse_root = values[prepare.roots[0] as usize];
        let shift = values[prepare.roots[1] as usize];
        for (index, (&input, &weight)) in inputs.iter().zip(weights).enumerate() {
            output
                .evaluate(&[input, weight, inverse_root, shift], &mut values)
                .map_err(|error| format!("RMS output element {index}: {error}"))?;
        }
        Ok(())
    }
}

fn stages(config: &BoundedRmsConfig) -> Result<[Stage; 3], String> {
    let mut builder = BMLTransformer::new();
    let inputs = [builder.var(0), builder.var(1)];
    let root = builder
        .rms_norm_bounded_reduce(&inputs, config)
        .map_err(|e| e.to_string())?;
    let reduce = Stage {
        dag: builder.into_soa(),
        roots: vec![root],
    };
    let mut builder = BMLTransformer::new();
    let total = builder.var(0);
    let roots = builder
        .rms_norm_bounded_prepare(&[total], config)
        .map_err(|e| e.to_string())?;
    let prepare = Stage {
        dag: builder.into_soa(),
        roots: roots.to_vec(),
    };
    let mut builder = BMLTransformer::new();
    let inputs = [
        builder.var(0),
        builder.var(1),
        builder.var(2),
        builder.var(3),
    ];
    let root = builder
        .rms_norm_bounded_output(&inputs, config)
        .map_err(|e| e.to_string())?;
    let output = Stage {
        dag: builder.into_soa(),
        roots: vec![root],
    };
    Ok([reduce, prepare, output])
}

/// Compile `x[i] * weight[i] / sqrt(mean(x*x) + epsilon)` entirely as BML.
///
/// Configuration belongs to the returned operation. The runtime operation APIs
/// bind it automatically; raw `HotLoop` callers must use `operation.context()`.
/// Width must be nonzero, read ranges must fit u32, and the declarations must
/// satisfy `BoundedRmsConfig`. Output capacity and overlap require the caller's
/// actual buffer layout and remain execution preconditions.
#[allow(clippy::too_many_arguments)]
pub fn compile_rmsnorm_fragment_checked(
    name: &str,
    input_base: u32,
    output_slot: u32,
    weight_base: u32,
    n_elems: u32,
    epsilon: f64,
    bounds: RmsBounds,
) -> Result<CheckedRmsFragment, String> {
    let config = bounds.config(n_elems as usize, epsilon)?;
    input_base
        .checked_add(n_elems - 1)
        .ok_or("RMS input address overflow")?;
    weight_base
        .checked_add(n_elems - 1)
        .ok_or("RMS weight address overflow")?;
    let constants = config.fixed_literals();
    let prefix = u32::try_from(constants.len()).map_err(|_| "RMS literal overflow")?;
    let mut program = crate::symbolic_bounded::compile_rms_program(
        &config,
        prefix,
        output_slot,
        KernelLimits::default(),
    )?;
    for op in &mut program.ops {
        *op = match *op {
            RpnOp::ContextVarIndexed { base: 0 } => RpnOp::VarIndexed { base: input_base },
            RpnOp::ContextConstIndexed { base } if base == prefix => {
                RpnOp::VarIndexed { base: weight_base }
            }
            other => other,
        };
    }
    // The vector-kernel convention returns an administrative scalar. An
    // operation fragment instead preserves the caller's entire stack prefix.
    if program.ops.pop() != Some(RpnOp::One) {
        return Err("RMS vector-kernel return convention changed".into());
    }
    let operation = OperationFragment {
        fragment: Fragment { ops: program.ops },
        const_pool: Some(constants),
        meta: FragmentMeta {
            name: name.into(),
            input_slot: input_base,
            output_slot,
            weight_base,
            n_rows: n_elems,
            n_cols: 1,
        },
    };
    Ok(CheckedRmsFragment {
        operation,
        config: config.clone(),
        stages: stages(&config)?,
    })
}

/// Complete RMS with default epsilon `1e-5` and owned configuration constants.
///
/// Input/weight bases are absolute buffer addresses; output is a slot. The
/// input bound is 16; the weight bound is `min(16,479/K)` with
/// `K=ceil(sqrt(n_elems))+1`, leaving headroom in the bounded rescaling identity.
/// This declares accepted data; it never clamps inputs or infers their range.
/// Use the checked constructor for explicit bounds/epsilon and real preflight.
/// Zero width emits no operations or writes.
///
/// # Panics
/// Nonzero widths above the bounded RMS construction limit or overflowing
/// absolute input/weight ranges require the checked constructor's error path.
pub fn compile_rmsnorm_fragment(
    name: &str,
    input_base: u32,
    output_slot: u32,
    weight_base: u32,
    n_elems: u32,
) -> OperationFragment {
    if n_elems == 0 {
        return OperationFragment {
            fragment: Fragment { ops: Vec::new() },
            const_pool: Some(Vec::new()),
            meta: FragmentMeta {
                name: name.into(),
                input_slot: input_base,
                output_slot,
                weight_base,
                n_rows: 0,
                n_cols: 1,
            },
        };
    }
    let k = f64::from(n_elems).sqrt().ceil() + 1.0;
    compile_rmsnorm_fragment_checked(
        name,
        input_base,
        output_slot,
        weight_base,
        n_elems,
        1e-5,
        RmsBounds {
            input: 16.0,
            weight: (479.0 / k).min(16.0),
        },
    )
    .expect("unsupported default RMS layout; use compile_rmsnorm_fragment_checked")
    .into_operation()
}
