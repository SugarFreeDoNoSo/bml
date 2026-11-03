//! Width-independent RMS programs for the complete bounded profile.

use crate::{symbolic_kernels::KernelLimits, RpnOp, RpnProgram};
use bml_domain::{transformer::BoundedRmsConfig, BMLTransformer, NodeKind, NodeSoA};
use std::collections::BTreeMap;

// Local common subexpressions retain exactly the same operand bits. This does
// not reassociate any BML operation or retain an input vector between loops.
pub(super) fn compact_stage(dag: NodeSoA, roots: Vec<u32>) -> (NodeSoA, Vec<u32>) {
    let mut compact = NodeSoA::new();
    compact.const_pool = dag.const_pool;
    let mut unique = BTreeMap::new();
    let mut mapping = Vec::with_capacity(dag.kinds.len());
    for (index, kind) in dag.kinds.into_iter().enumerate() {
        let key = match kind {
            NodeKind::One => (0, 0, 0),
            NodeKind::Zero => (1, 0, 0),
            NodeKind::Var(id) => (2, id, 0),
            NodeKind::Const(id) => (3, id, 0),
            NodeKind::Bml => (
                4,
                mapping[dag.lefts[index] as usize],
                mapping[dag.rights[index] as usize],
            ),
        };
        let node = *unique.entry(key).or_insert_with(|| match key {
            (0, _, _) => compact.push_one(),
            (1, _, _) => compact.push_zero(),
            (2, id, _) => compact.push_var(id),
            (3, id, _) => compact.push_const(id),
            (_, left, right) => compact.push_bml(left, right),
        });
        mapping.push(node);
    }
    let roots = roots
        .into_iter()
        .map(|root| mapping[root as usize])
        .collect();
    (compact, roots)
}

// Consume a last-use value directly when it is already at the top of the
// stack. Other shared values retain stable relative positions until their
// last consumer. This changes only stack administration, never BML operands.
pub(super) fn linearize_stage(dag: &NodeSoA, root: u32) -> RpnProgram {
    let mut pending = vec![(root, false)];
    let mut visited = vec![false; dag.len()];
    let mut order = Vec::new();
    let mut uses = vec![0usize; dag.len()];
    uses[root as usize] = 1;
    while let Some((node, expanded)) = pending.pop() {
        let index = node as usize;
        if visited[index] {
            continue;
        }
        if dag.kinds[index] == NodeKind::Bml && !expanded {
            pending.push((node, true));
            pending.push((dag.rights[index], false));
            pending.push((dag.lefts[index], false));
            continue;
        }
        visited[index] = true;
        if dag.kinds[index] == NodeKind::Bml {
            uses[dag.lefts[index] as usize] += 1;
            uses[dag.rights[index] as usize] += 1;
        }
        order.push(node);
    }
    let mut program = RpnProgram::new();
    let mut stack = Vec::<u32>::new();
    for node in order {
        let index = node as usize;
        match dag.kinds[index] {
            NodeKind::One => program.push(RpnOp::One),
            NodeKind::Zero => program.push(RpnOp::Zero),
            NodeKind::Var(id) => program.push(RpnOp::Var(id)),
            NodeKind::Const(id) => program.push(RpnOp::Const(id)),
            NodeKind::Bml => {
                let left = dag.lefts[index];
                let right = dag.rights[index];
                let left_at = stack.iter().position(|&id| id == left).unwrap();
                let right_at = stack.iter().position(|&id| id == right).unwrap();
                let len = stack.len();
                if left != right
                    && left_at + 2 == len
                    && right_at + 1 == len
                    && uses[left as usize] == 1
                    && uses[right as usize] == 1
                {
                    stack.truncate(len - 2);
                } else if left == right && left_at + 1 == len && uses[left as usize] == 2 {
                    program.push(RpnOp::Dup);
                    stack.pop();
                } else if left_at + 1 == len && uses[left as usize] == 1 {
                    program.push(RpnOp::Pick {
                        depth: (len - 1 - right_at) as u32,
                    });
                    stack.pop();
                } else if right_at + 1 == len && uses[right as usize] == 1 {
                    program.push(RpnOp::Pick {
                        depth: (len - 1 - left_at) as u32,
                    });
                    program.push(RpnOp::Swap);
                    stack.pop();
                } else {
                    program.push(RpnOp::Pick {
                        depth: (len - 1 - left_at) as u32,
                    });
                    program.push(RpnOp::Pick {
                        depth: (len - right_at) as u32,
                    });
                }
                program.push(RpnOp::Bml);
                uses[left as usize] -= 1;
                uses[right as usize] -= 1;
            }
        }
        stack.push(node);
        while stack.len() > 1 && uses[stack[stack.len() - 2] as usize] == 0 {
            program.push(RpnOp::Swap);
            program.push(RpnOp::Drop);
            stack.remove(stack.len() - 2);
        }
    }
    debug_assert_eq!(stack, vec![root]);
    program
}

fn stage(
    config: &BoundedRmsConfig,
    inputs: u32,
    build: impl FnOnce(&mut BMLTransformer, &[u32]) -> Result<Vec<u32>, String>,
) -> Result<(NodeSoA, Vec<u32>), String> {
    let mut builder = BMLTransformer::new();
    let variables: Vec<_> = (0..inputs).map(|index| builder.var(index)).collect();
    let roots = build(&mut builder, &variables)?;
    let dag = builder.into_soa();
    if dag.const_pool != config.fixed_literals() {
        return Err("bounded RMS stage literal layout changed".into());
    }
    Ok((dag, roots))
}

pub(super) fn append_stage(
    output: &mut Vec<RpnOp>,
    stage: RpnProgram,
    mut variable: impl FnMut(u32, u32) -> Result<Vec<RpnOp>, String>,
) -> Result<(), String> {
    let mut height = 0u32;
    for instruction in stage.ops {
        match instruction {
            RpnOp::Var(id) => {
                output.extend(variable(id, height)?);
                height += 1;
            }
            other => {
                match other {
                    RpnOp::One | RpnOp::Const(_) | RpnOp::Pick { .. } | RpnOp::Dup => height += 1,
                    RpnOp::Bml | RpnOp::Drop => {
                        height = height.checked_sub(1).ok_or("invalid RMS stage stack")?
                    }
                    RpnOp::Swap => {}
                    _ => return Err("invalid RMS stage instruction".into()),
                }
                output.push(other);
            }
        }
    }
    if height != 1 {
        return Err("bounded RMS stage must return one scalar".into());
    }
    Ok(())
}

/// Emit RMS with the canonical 17 literals at constant indices 0..17.
///
/// Inputs are loaded from context base zero; learned weights begin at
/// `weight_const_base`. Every output writes `output_slot[index]`, then one
/// administrative `One` remains. The caller owns and binds the constants. No
/// input-sized scratch, output-index literals or native arithmetic is emitted.
/// An older Buffer binding can still keep its unused output literals and place
/// weights after them; new IndexedBuffer kernels bind weights directly at 17.
///
/// The three construction DAGs, stored opcodes and peak execution stack have
/// constant size with respect to width. Runtime steps remain linear in width
/// and are checked by the existing graph execution budget.
pub(crate) fn compile_rms_program(
    config: &BoundedRmsConfig,
    weight_const_base: u32,
    output_slot: u32,
    limits: KernelLimits,
) -> Result<RpnProgram, String> {
    if limits.max_programs == 0 || limits.max_nodes == 0 {
        return Err("bounded RMS program/node budget exceeded".into());
    }
    let width = u32::try_from(config.width()).map_err(|_| "RMS width overflow")?;
    weight_const_base
        .checked_add(width - 1)
        .ok_or("RMS weight constant index overflow")?;
    let (reduce, reduce_roots) = stage(config, 2, |builder, inputs| {
        builder
            .rms_norm_bounded_reduce(inputs, config)
            .map(|root| vec![root])
            .map_err(|error| error.to_string())
    })?;
    let (prepare, prepare_roots) = stage(config, 1, |builder, inputs| {
        builder
            .rms_norm_bounded_prepare(inputs, config)
            .map(|roots| roots.to_vec())
            .map_err(|error| error.to_string())
    })?;
    let (output, output_roots) = stage(config, 4, |builder, inputs| {
        builder
            .rms_norm_bounded_output(inputs, config)
            .map(|root| vec![root])
            .map_err(|error| error.to_string())
    })?;
    let nodes = reduce
        .len()
        .checked_add(prepare.len())
        .and_then(|n| n.checked_add(output.len()))
        .ok_or("bounded RMS node count overflow")?;
    if nodes > limits.max_nodes {
        return Err("symbolic node budget exceeded".into());
    }
    let (reduce, reduce_roots) = compact_stage(reduce, reduce_roots);
    let (prepare, prepare_roots) = compact_stage(prepare, prepare_roots);
    let (output, output_roots) = compact_stage(output, output_roots);

    // Each iteration replaces [total, index] by its newly accumulated total.
    let mut reduction = Vec::new();
    append_stage(
        &mut reduction,
        linearize_stage(&reduce, reduce_roots[0]),
        |id, height| {
            Ok(match id {
                0 => vec![RpnOp::Pick { depth: height + 1 }],
                1 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::ContextVarIndexed { base: 0 },
                ],
                _ => return Err("invalid RMS reduction variable".into()),
            })
        },
    )?;
    reduction.extend([RpnOp::Swap, RpnOp::Drop, RpnOp::Swap, RpnOp::Drop]);
    let mut ops = vec![
        RpnOp::Const(4), // epsilon * width / input_bound^2, identical to v3.
        RpnOp::Loop {
            count: width,
            body_len: reduction.len() as u32,
        },
    ];
    ops.extend(reduction);

    // Retain the fixed-size preparation DAG once. Both roots share its values;
    // no recomputation of inverse_root or width-sized input cache is needed.
    // Stack after this block: [total, prepare_node_0, ..., prepare_node_n].
    for (index, kind) in prepare.kinds.iter().enumerate() {
        let index = index as u32;
        match *kind {
            NodeKind::One => ops.push(RpnOp::One),
            NodeKind::Const(id) => ops.push(RpnOp::Const(id)),
            NodeKind::Var(0) => ops.push(RpnOp::Pick { depth: index }),
            NodeKind::Bml => {
                let left = prepare.lefts[index as usize];
                let right = prepare.rights[index as usize];
                if left >= index || right >= index {
                    return Err("invalid RMS preparation topology".into());
                }
                ops.extend([
                    RpnOp::Pick {
                        depth: index - left - 1,
                    },
                    RpnOp::Pick {
                        depth: index - right,
                    },
                    RpnOp::Bml,
                ]);
            }
            _ => return Err("invalid RMS preparation node".into()),
        }
    }
    let prepare_nodes = prepare.len() as u32;
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
                1 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::ContextConstIndexed {
                        base: weight_const_base,
                    },
                ],
                2 | 3 => vec![RpnOp::Pick {
                    depth: height + prepare_nodes - prepare_roots[(id - 2) as usize],
                }],
                _ => return Err("invalid RMS output variable".into()),
            })
        },
    )?;
    emission.extend([RpnOp::Swap, RpnOp::StoreResult { slot: output_slot }]);
    ops.push(RpnOp::Loop {
        count: width,
        body_len: emission.len() as u32,
    });
    ops.extend(emission);
    ops.extend(std::iter::repeat_n(RpnOp::Drop, prepare.len() + 1));
    ops.push(RpnOp::One);
    if ops.len() > limits.max_ops || ops.len() > limits.max_output_ops {
        return Err("bounded RMS opcode budget exceeded".into());
    }
    let bytes = ops
        .len()
        .checked_mul(std::mem::size_of::<RpnOp>())
        .ok_or("RMS program byte size overflow")?;
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
