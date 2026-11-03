//! Stable shifted softmax with constant stored code and execution stack.

use super::rms::{append_stage, compact_stage, linearize_stage};
use crate::{symbolic_kernels::KernelLimits, RpnOp, RpnProgram};
use bml_domain::{transformer::BoundedOpConfig, BMLTransformer, NodeSoA};

fn stage(
    config: &BoundedOpConfig,
    inputs: u32,
    build: impl FnOnce(&mut BMLTransformer, &[u32]) -> Result<u32, String>,
) -> Result<(NodeSoA, Vec<u32>), String> {
    let mut builder = BMLTransformer::new();
    let variables: Vec<_> = (0..inputs).map(|index| builder.var(index)).collect();
    let root = build(&mut builder, &variables)?;
    let dag = builder.into_soa();
    if dag.const_pool != config.fixed_literals() {
        return Err("bounded softmax stage literal layout changed".into());
    }
    Ok((dag, vec![root]))
}

/// Fixed literals occupy indices 0..10. Inputs are `width` scores followed by
/// their caller-selected maximum; profile validation checks that maximum before
/// execution. Recomputed exponentials use precisely the same BML operands as
/// the historical vector DAG, and the reduction keeps its left-to-right order.
pub(super) fn compile_softmax_program(
    config: &BoundedOpConfig,
    output_slot: u32,
    limits: KernelLimits,
) -> Result<RpnProgram, String> {
    if limits.max_programs == 0 || limits.max_nodes == 0 {
        return Err("bounded softmax program/node budget exceeded".into());
    }
    let width = u32::try_from(config.outputs()).map_err(|_| "softmax width overflow")?;
    if width == 0 || config.inputs() != width as usize + 1 {
        return Err("invalid bounded softmax stage configuration".into());
    }
    let (exp, exp_roots) = stage(config, 2, |builder, inputs| {
        builder
            .softmax_bounded_exp(inputs, config)
            .map_err(|e| e.to_string())
    })?;
    let (reduce, reduce_roots) = stage(config, 3, |builder, inputs| {
        builder
            .softmax_bounded_reduce(inputs, config)
            .map_err(|e| e.to_string())
    })?;
    let (inverse, inverse_roots) = stage(config, 1, |builder, inputs| {
        builder
            .softmax_bounded_inverse(inputs, config)
            .map_err(|e| e.to_string())
    })?;
    let (output, output_roots) = stage(config, 3, |builder, inputs| {
        builder
            .softmax_bounded_output(inputs, config)
            .map_err(|e| e.to_string())
    })?;
    let nodes = exp
        .len()
        .checked_add(reduce.len())
        .and_then(|n| n.checked_add(inverse.len()))
        .and_then(|n| n.checked_add(output.len()))
        .ok_or("bounded softmax node count overflow")?;
    if nodes > limits.max_nodes {
        return Err("symbolic node budget exceeded".into());
    }
    let (exp, exp_roots) = compact_stage(exp, exp_roots);
    let (reduce, reduce_roots) = compact_stage(reduce, reduce_roots);
    let (inverse, inverse_roots) = compact_stage(inverse, inverse_roots);
    let (output, output_roots) = compact_stage(output, output_roots);
    let mut ops = Vec::new();
    append_stage(&mut ops, linearize_stage(&exp, exp_roots[0]), |id, _| {
        Ok(vec![RpnOp::Var(match id {
            0 => 0,
            1 => width,
            _ => return Err("invalid softmax exponential variable".into()),
        })])
    })?;
    let mut reduction = Vec::new();
    append_stage(
        &mut reduction,
        linearize_stage(&reduce, reduce_roots[0]),
        |id, height| {
            Ok(match id {
                0 => vec![RpnOp::Pick { depth: height + 1 }],
                1 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::ContextVarIndexed { base: 1 },
                ],
                2 => vec![RpnOp::Var(width)],
                _ => return Err("invalid softmax reduction variable".into()),
            })
        },
    )?;
    reduction.extend([RpnOp::Swap, RpnOp::Drop, RpnOp::Swap, RpnOp::Drop]);
    ops.push(RpnOp::Loop {
        count: width - 1,
        body_len: reduction.len() as u32,
    });
    ops.extend(reduction);
    append_stage(
        &mut ops,
        linearize_stage(&inverse, inverse_roots[0]),
        |id, height| {
            if id != 0 {
                return Err("invalid softmax inverse variable".into());
            }
            Ok(vec![RpnOp::Pick { depth: height }])
        },
    )?;
    ops.extend([RpnOp::Swap, RpnOp::Drop]);
    let mut emission = Vec::new();
    append_stage(
        &mut emission,
        linearize_stage(&output, output_roots[0]),
        |id, height| {
            Ok(match id {
                0 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::ContextVarIndexed { base: 0 },
                ],
                1 => vec![RpnOp::Var(width)],
                2 => vec![RpnOp::Pick { depth: height + 1 }],
                _ => return Err("invalid softmax output variable".into()),
            })
        },
    )?;
    emission.extend([RpnOp::Swap, RpnOp::StoreResult { slot: output_slot }]);
    ops.push(RpnOp::Loop {
        count: width,
        body_len: emission.len() as u32,
    });
    ops.extend(emission);
    ops.extend([RpnOp::Drop, RpnOp::One]);
    if ops.len() > limits.max_ops || ops.len() > limits.max_output_ops {
        return Err("bounded softmax opcode budget exceeded".into());
    }
    let bytes = ops
        .len()
        .checked_mul(std::mem::size_of::<RpnOp>())
        .ok_or("softmax program byte size overflow")?;
    if bytes > limits.fragment_bytes {
        return Err(format!(
            "symbolic loop body exceeds fragment budget: {bytes} > {}",
            limits.fragment_bytes
        ));
    }
    Ok(RpnProgram {
        ops,
        const_pool: Vec::new(),
    })
}
