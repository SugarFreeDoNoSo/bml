//! Complete bounded BML MLP without implicit result-buffer scratch slots.

use super::{FragmentMeta, OperationFragment};
use crate::{symbolic_bounded, Fragment, RpnOp, RpnProgram};
use bml_domain::{transformer::BoundedOpConfig, BMLTransformer, NodeKind, NodeSoA};
use std::collections::BTreeMap;

/// Inclusive absolute operand declarations for each MLP stage.
///
/// Bounds declare accepted data; they never clamp values. Dot stages retain
/// the bounded recurrence's accumulator precondition (`accumulator < 256`
/// before each step). Successful construction does not guarantee evaluability;
/// use [`CheckedMlpFragment::validate_values`] on the actual operands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlpBounds {
    /// Input vector element bound.
    pub input: f64,
    /// Gate projection matrix element bound.
    pub gate_weight: f64,
    /// Up projection matrix element bound.
    pub up_weight: f64,
    /// Computed gate projection bound; at most 16 for bounded sigmoid.
    pub gate: f64,
    /// Computed up projection bound.
    pub up: f64,
    /// Computed SwiGLU element bound used by the down projection.
    pub hidden: f64,
    /// Down projection matrix element bound.
    pub down_weight: f64,
}

impl Default for MlpBounds {
    fn default() -> Self {
        Self {
            input: 16.0,
            gate_weight: 2.0,
            up_weight: 2.0,
            gate: 16.0,
            up: 16.0,
            hidden: 256.0,
            down_weight: 1.0,
        }
    }
}

/// Explicit temporary slots for the linear-work MLP implementation.
///
/// Each base is the absolute address of offset zero in its paired slot. All
/// pairs must imply the same nonzero `ResultBuffer` slot size, at least
/// `max(n_embd, n_ffn)`. The actual buffer must use that size. These slots and
/// the output slot must be distinct and disjoint from every operand range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlpScratch {
    /// Slot receiving the gate projection.
    pub gate_slot: u32,
    /// Absolute base of the gate slot.
    pub gate_base: u32,
    /// Slot receiving the up projection.
    pub up_slot: u32,
    /// Absolute base of the up slot.
    pub up_base: u32,
    /// Slot receiving the SwiGLU activation.
    pub hidden_slot: u32,
    /// Absolute base of the activation slot.
    pub hidden_base: u32,
}

#[derive(Clone, Debug)]
struct Stage {
    config: BoundedOpConfig,
    dag: NodeSoA,
    root: u32,
    program: RpnProgram,
    constant_base: u32,
}

impl Stage {
    fn new(config: BoundedOpConfig, constants: &mut Vec<f64>) -> Result<Self, String> {
        let mut builder = BMLTransformer::new();
        let inputs: Vec<_> = (0..config.inputs())
            .map(|index| builder.var(index as u32))
            .collect();
        let root = builder
            .operation_bounded(&inputs, &[], &config)
            .map_err(|error| error.to_string())?[0];
        let dag = builder.into_soa();
        let constant_base = u32::try_from(constants.len()).map_err(|_| "MLP constant overflow")?;
        let (folded, root_folded) = fold_configuration(&dag, root)?;
        constants.extend_from_slice(&folded.const_pool);
        let (compact, roots) = symbolic_bounded::compact_stage(folded, vec![root_folded]);
        let program = symbolic_bounded::linearize_stage(&compact, roots[0]);
        Ok(Self {
            config,
            dag,
            root,
            program,
            constant_base,
        })
    }

    fn evaluate(&self, inputs: &[f64], values: &mut [f64]) -> Result<f64, String> {
        self.config
            .validate_values(inputs, &[])
            .map_err(|error| error.to_string())?;
        for (node, kind) in self.dag.kinds.iter().enumerate() {
            values[node] = match *kind {
                NodeKind::One => 1.0,
                NodeKind::Zero => 0.0,
                NodeKind::Var(index) => inputs[index as usize],
                NodeKind::Const(index) => self.dag.const_pool[index as usize],
                NodeKind::Bml => bml_domain::symbolic::Basis::Bml
                    .apply_real(
                        values[self.dag.lefts[node] as usize],
                        values[self.dag.rights[node] as usize],
                    )
                    .map_err(|error| format!("BML node {node}: {error}"))?,
            };
        }
        Ok(values[self.root as usize])
    }

    fn append(
        &self,
        output: &mut Vec<RpnOp>,
        variable: impl FnMut(u32, u32) -> Result<Vec<RpnOp>, String>,
    ) -> Result<(), String> {
        let mut program = self.program.clone();
        for instruction in &mut program.ops {
            if let RpnOp::Const(index) = instruction {
                *index = index
                    .checked_add(self.constant_base)
                    .ok_or("MLP constant index overflow")?;
            }
        }
        symbolic_bounded::append_stage(output, program, variable)
    }
}

// Only closed configuration subgraphs are evaluated at compilation. Var leaves
// represent all activations/weights, so anything depending on model data remains
// BML bytecode. Preserve exact finite-real operation order and signed-zero bits.
fn fold_configuration(dag: &NodeSoA, root: u32) -> Result<(NodeSoA, u32), String> {
    let mut folded = NodeSoA::new();
    folded.const_pool = dag.const_pool.clone();
    let mut literals = BTreeMap::new();
    for (index, value) in folded.const_pool.iter().enumerate() {
        literals.entry(value.to_bits()).or_insert(index as u32);
    }
    let mut mapping = Vec::with_capacity(dag.len());
    let mut known = Vec::<Option<f64>>::with_capacity(dag.len());
    for (index, kind) in dag.kinds.iter().enumerate() {
        let (node, value) = match *kind {
            NodeKind::One => (folded.push_one(), Some(1.0)),
            NodeKind::Zero => (folded.push_zero(), Some(0.0)),
            NodeKind::Var(id) => (folded.push_var(id), None),
            NodeKind::Const(id) => (folded.push_const(id), Some(dag.const_pool[id as usize])),
            NodeKind::Bml => {
                let left = dag.lefts[index] as usize;
                let right = dag.rights[index] as usize;
                match (known[left], known[right]) {
                    (Some(left), Some(right)) => {
                        let value = bml_domain::symbolic::Basis::Bml
                            .apply_real(left, right)
                            .map_err(|error| {
                                format!("MLP configuration BML node {index}: {error}")
                            })?;
                        let id = *literals.entry(value.to_bits()).or_insert_with(|| {
                            let id = folded.const_pool.len() as u32;
                            folded.const_pool.push(value);
                            id
                        });
                        (folded.push_const(id), Some(value))
                    }
                    _ => (folded.push_bml(mapping[left], mapping[right]), None),
                }
            }
        };
        mapping.push(node);
        known.push(value);
    }
    Ok((folded, mapping[root as usize]))
}

/// Complete `down * (gate * sigmoid(gate) * up)` with full matrix projections.
///
/// Gate and up matrices are row-major `[n_ffn, n_embd]`; down is row-major
/// `[n_embd, n_ffn]`. All input/matrix bases are absolute buffer element
/// addresses, while output identifies one slot. No extra slots are required.
/// Outputs must be disjoint from all operands, and the caller must reserve the
/// actual buffer ranges and output capacity. Native FAdd/FMul only form indices.
///
/// Without explicit scratch, gate/up/SwiGLU are recomputed for each output row:
/// execution and preflight cost O(n_embd² * n_ffn). This path has three nested
/// loop levels and is executed by HotLoop/Runtime operation APIs; the persisted
/// `ValidatedGraph` contract permits only two and rejects this representation.
/// With [`MlpScratch`], projection work is O(n_embd * n_ffn), and loop nesting
/// is at most two. Bytecode and execution stack size are dimension-independent.
#[derive(Clone, Debug)]
pub struct CheckedMlpFragment {
    operation: OperationFragment,
    operations: Vec<OperationFragment>,
    bounds: MlpBounds,
    n_embd: u32,
    n_ffn: u32,
    stages: [Stage; 4],
    scratch: Option<MlpScratch>,
}

impl CheckedMlpFragment {
    /// Bytecode, buffer metadata and owned configuration constants.
    pub fn operation(&self) -> &OperationFragment {
        &self.operation
    }

    /// Consume the preflight wrapper, preserving the operation's owned constants.
    pub fn into_operation(self) -> OperationFragment {
        self.operation
    }

    /// Balanced executable blocks with the complete owned pool on each block.
    ///
    /// Explicit-scratch MLP has four indivisible loop blocks, each at most
    /// 32 KiB, in gate/up/SwiGLU/down order (empty for zero embedding width).
    /// The recomputing variant retains its single three-level loop (32,640
    /// bytes with default bounds), which cannot be split without changing its
    /// representation. Check its size when using custom numerical bounds.
    pub fn operations(&self) -> &[OperationFragment] {
        &self.operations
    }

    /// Consume this wrapper and return the safe complete-loop execution plan.
    pub fn into_operations(self) -> Vec<OperationFragment> {
        self.operations
    }

    /// Complete configuration pool, containing no activation or learned weight.
    pub fn constants(&self) -> &[f64] {
        self.operation
            .const_pool
            .as_deref()
            .expect("MLP owns its configuration")
    }

    /// Numerical declarations for inputs, projections, SwiGLU and down weights.
    pub fn bounds(&self) -> MlpBounds {
        self.bounds
    }

    /// Embedding and hidden FFN widths, in that order.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.n_embd, self.n_ffn)
    }

    /// Declared temporary slots, or `None` for the recomputing representation.
    pub fn scratch(&self) -> Option<MlpScratch> {
        self.scratch
    }

    /// Validate row-major operands and every BML intermediate before writes.
    ///
    /// This evaluates the same bounded formulas as the emitted bytecode, using
    /// fixed-size DAG scratch, plus O(n_ffn) host values for the explicit-scratch
    /// variant. It does not replace any result with native MLP
    /// arithmetic. Execute on the same operands, bytecode and constants after
    /// this check; mutation invalidates the preflight. Buffer overlap/capacity
    /// require the caller's layout and remain separate execution preconditions.
    pub fn validate_values(
        &self,
        inputs: &[f64],
        gate_weights: &[f64],
        up_weights: &[f64],
        down_weights: &[f64],
    ) -> Result<(), String> {
        let embd = self.n_embd as usize;
        let ffn = self.n_ffn as usize;
        let count = embd.checked_mul(ffn).ok_or("MLP operand shape overflow")?;
        if inputs.len() != embd
            || gate_weights.len() != count
            || up_weights.len() != count
            || down_weights.len() != count
        {
            return Err("MLP operand shape mismatch".into());
        }
        for (index, &value) in inputs.iter().enumerate() {
            if !value.is_finite() || value.abs() > self.bounds.input {
                return Err(format!("MLP input bound at element {index}"));
            }
        }
        let capacity = self
            .stages
            .iter()
            .map(|stage| stage.dag.len())
            .max()
            .unwrap();
        let mut values = vec![0.0; capacity];
        let [gate, up, swiglu, down] = &self.stages;
        let activate = |hidden: usize, values: &mut [f64]| -> Result<f64, String> {
            let mut g = 0.0;
            let mut u = 0.0;
            for (column, &input) in inputs.iter().enumerate() {
                g = gate
                    .evaluate(&[g, input, gate_weights[hidden * embd + column]], values)
                    .map_err(|error| format!("MLP gate row {hidden}, column {column}: {error}"))?;
                u = up
                    .evaluate(&[u, input, up_weights[hidden * embd + column]], values)
                    .map_err(|error| format!("MLP up row {hidden}, column {column}: {error}"))?;
            }
            swiglu
                .evaluate(&[g, u], values)
                .map_err(|error| format!("MLP SwiGLU element {hidden}: {error}"))
        };
        let cached = if self.scratch.is_some() && embd != 0 {
            let mut hidden = Vec::new();
            hidden
                .try_reserve_exact(ffn)
                .map_err(|_| "MLP preflight allocation")?;
            for index in 0..ffn {
                hidden.push(activate(index, &mut values)?);
            }
            Some(hidden)
        } else {
            None
        };
        for row in 0..embd {
            let mut total = 0.0;
            for hidden in 0..ffn {
                let activated = match &cached {
                    Some(values) => values[hidden],
                    None => activate(hidden, &mut values)?,
                };
                total = down
                    .evaluate(
                        &[total, activated, down_weights[row * ffn + hidden]],
                        &mut values,
                    )
                    .map_err(|error| format!("MLP down row {row}, column {hidden}: {error}"))?;
            }
        }
        Ok(())
    }
}

fn check_read(base: u32, length: u32, name: &str) -> Result<(), String> {
    if length != 0 {
        base.checked_add(length - 1)
            .ok_or_else(|| format!("MLP {name} address overflow"))?;
    }
    Ok(())
}

fn projection(
    stage: &Stage,
    input_base: u32,
    weight_base: u32,
    n_embd: u32,
    hidden_depth: u32,
) -> Result<Vec<RpnOp>, String> {
    let mut body = Vec::new();
    // Frame: [..., hidden_index, (retained gate), accumulator, column].
    stage.append(&mut body, |id, height| {
        Ok(match id {
            0 => vec![RpnOp::Pick { depth: height + 1 }],
            1 => vec![
                RpnOp::Pick { depth: height },
                RpnOp::VarIndexed { base: input_base },
            ],
            2 => vec![
                RpnOp::Pick {
                    depth: height + hidden_depth,
                },
                RpnOp::Const(0),
                RpnOp::FMul,
                RpnOp::Pick { depth: height + 1 },
                RpnOp::FAdd,
                RpnOp::VarIndexed { base: weight_base },
            ],
            _ => return Err("invalid MLP projection variable".into()),
        })
    })?;
    // Replace the old accumulator and consume the loop column.
    body.extend([RpnOp::Swap, RpnOp::Drop, RpnOp::Swap, RpnOp::Drop]);
    let mut result = vec![
        RpnOp::Zero,
        RpnOp::Loop {
            count: n_embd,
            body_len: body.len() as u32,
        },
    ];
    result.extend(body);
    Ok(result)
}

/// Compile a rectangular MLP with explicit data ranges and owned constants.
///
/// The input has `n_embd` elements; gate/up contain `n_ffn * n_embd` weights
/// each; down contains `n_embd * n_ffn` weights. Zero embedding width performs
/// no writes; zero FFN width writes a zero vector. Nonempty declared read ranges
/// and matrix counts must fit u32. The output slot must hold `n_embd` elements
/// and not overlap any operand. All model weights remain in the caller's buffer.
///
/// Runtime operation entrypoints bind constants automatically; direct HotLoop
/// callers must pass `operation.context(parent)`. The returned profile requires
/// finite-real preflight; unchecked legacy execution can fail outside it.
#[allow(clippy::too_many_arguments)]
pub fn compile_mlp_fragment_checked(
    name: &str,
    input_base: u32,
    output_slot: u32,
    gate_base: u32,
    up_base: u32,
    down_base: u32,
    n_embd: u32,
    n_ffn: u32,
    bounds: MlpBounds,
) -> Result<CheckedMlpFragment, String> {
    let count = n_embd
        .checked_mul(n_ffn)
        .ok_or("MLP matrix dimensions overflow")?;
    check_read(input_base, n_embd, "input")?;
    check_read(gate_base, count, "gate")?;
    check_read(up_base, count, "up")?;
    check_read(down_base, count, "down")?;
    let configs = [
        BoundedOpConfig::dot_step(bounds.input, bounds.gate_weight),
        BoundedOpConfig::dot_step(bounds.input, bounds.up_weight),
        BoundedOpConfig::swiglu(bounds.gate, bounds.up),
        BoundedOpConfig::dot_step(bounds.hidden, bounds.down_weight),
    ];
    let mut constants = vec![f64::from(n_embd), f64::from(n_ffn)];
    let mut stages = Vec::with_capacity(4);
    for config in configs {
        stages.push(Stage::new(
            config.map_err(|error| error.to_string())?,
            &mut constants,
        )?);
    }
    let stages: [Stage; 4] = stages.try_into().map_err(|_| "MLP stage count")?;
    let [gate, up, swiglu, down] = &stages;
    let mut hidden = projection(gate, input_base, gate_base, n_embd, 2)?;
    hidden.extend(projection(up, input_base, up_base, n_embd, 3)?);
    // Frame: [output_row, down_accumulator, hidden_index, gate, up].
    swiglu.append(&mut hidden, |id, height| match id {
        0 | 1 => Ok(vec![RpnOp::Pick {
            depth: height + 1 - id,
        }]),
        _ => Err("invalid MLP SwiGLU variable".into()),
    })?;
    hidden.extend([RpnOp::Swap, RpnOp::Drop, RpnOp::Swap, RpnOp::Drop]);
    // Frame: [output_row, down_accumulator, hidden_index, activated].
    down.append(&mut hidden, |id, height| {
        Ok(match id {
            0 => vec![RpnOp::Pick { depth: height + 2 }],
            1 => vec![RpnOp::Pick { depth: height }],
            2 => vec![
                RpnOp::Pick { depth: height + 3 },
                RpnOp::Const(1),
                RpnOp::FMul,
                RpnOp::Pick { depth: height + 2 },
                RpnOp::FAdd,
                RpnOp::VarIndexed { base: down_base },
            ],
            _ => return Err("invalid MLP down variable".into()),
        })
    })?;
    hidden.extend([
        RpnOp::Swap,
        RpnOp::Drop,
        RpnOp::Swap,
        RpnOp::Drop,
        RpnOp::Swap,
        RpnOp::Drop,
    ]);
    let mut outer = vec![
        RpnOp::Zero,
        RpnOp::Loop {
            count: n_ffn,
            body_len: hidden.len() as u32,
        },
    ];
    outer.extend(hidden);
    outer.extend([RpnOp::Swap, RpnOp::StoreResult { slot: output_slot }]);
    let mut ops = vec![RpnOp::Loop {
        count: n_embd,
        body_len: outer.len() as u32,
    }];
    ops.extend(outer);
    let operation = OperationFragment {
        fragment: Fragment { ops },
        const_pool: Some(constants),
        meta: FragmentMeta {
            name: name.into(),
            input_slot: input_base,
            output_slot,
            weight_base: gate_base,
            n_rows: n_embd,
            n_cols: n_ffn,
        },
    };
    Ok(CheckedMlpFragment {
        operations: vec![operation.clone()],
        operation,
        bounds,
        n_embd,
        n_ffn,
        stages,
        scratch: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn scratch_layout(
    scratch: MlpScratch,
    input_base: u32,
    output_slot: u32,
    gate_base: u32,
    up_base: u32,
    down_base: u32,
    n_embd: u32,
    n_ffn: u32,
) -> Result<(), String> {
    let pairs = [
        (scratch.gate_slot, scratch.gate_base),
        (scratch.up_slot, scratch.up_base),
        (scratch.hidden_slot, scratch.hidden_base),
    ];
    let slots = [
        scratch.gate_slot,
        scratch.up_slot,
        scratch.hidden_slot,
        output_slot,
    ];
    for (index, slot) in slots.iter().enumerate() {
        if slots[..index].contains(slot) {
            return Err("MLP scratch and output slots must be distinct".into());
        }
    }
    let mut stride = None;
    for (slot, base) in pairs {
        if slot == 0 {
            if base != 0 {
                return Err("MLP scratch slot/base mismatch".into());
            }
        } else {
            if base == 0 || base % slot != 0 {
                return Err("MLP scratch slot/base mismatch".into());
            }
            let size = base / slot;
            if stride.is_some_and(|previous| previous != size) {
                return Err("MLP scratch slots imply different buffer strides".into());
            }
            stride = Some(size);
        }
    }
    let stride = stride.ok_or("MLP scratch buffer stride is missing")?;
    if stride < n_embd.max(n_ffn) {
        return Err("MLP scratch/output slot capacity is too small".into());
    }
    let output_base = output_slot
        .checked_mul(stride)
        .ok_or("MLP output address overflow")?;
    check_read(output_base, n_embd, "output")?;
    let count = n_embd
        .checked_mul(n_ffn)
        .ok_or("MLP matrix dimensions overflow")?;
    let reads = [
        (input_base, n_embd),
        (gate_base, count),
        (up_base, count),
        (down_base, count),
    ];
    let writes = [
        (scratch.gate_base, n_ffn),
        (scratch.up_base, n_ffn),
        (scratch.hidden_base, n_ffn),
        (output_base, n_embd),
    ];
    for (base, length) in writes {
        check_read(base, length, "scratch")?;
        for (read_base, read_length) in reads {
            if length != 0
                && read_length != 0
                && u64::from(base) < u64::from(read_base) + u64::from(read_length)
                && u64::from(read_base) < u64::from(base) + u64::from(length)
            {
                return Err("MLP scratch/output overlaps an operand range".into());
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matrix_program(
    stage: &Stage,
    input_base: u32,
    weight_base: u32,
    output_slot: u32,
    rows: u32,
    columns: u32,
    stride_constant: u32,
) -> Result<Vec<RpnOp>, String> {
    let mut inner = Vec::new();
    // Frame: [row, row_start, accumulator, column].
    stage.append(&mut inner, |id, height| {
        Ok(match id {
            0 => vec![RpnOp::Pick { depth: height + 1 }],
            1 => vec![
                RpnOp::Pick { depth: height },
                RpnOp::VarIndexed { base: input_base },
            ],
            2 => vec![
                RpnOp::Pick { depth: height + 2 },
                RpnOp::Pick { depth: height + 1 },
                RpnOp::FAdd,
                RpnOp::VarIndexed { base: weight_base },
            ],
            _ => return Err("invalid MLP matrix variable".into()),
        })
    })?;
    inner.extend([RpnOp::Swap, RpnOp::Drop, RpnOp::Swap, RpnOp::Drop]);
    let mut outer = vec![
        RpnOp::Dup,
        RpnOp::Const(stride_constant),
        RpnOp::FMul,
        RpnOp::Zero,
        RpnOp::Loop {
            count: columns,
            body_len: inner.len() as u32,
        },
    ];
    outer.extend(inner);
    outer.extend([
        RpnOp::Pick { depth: 2 },
        RpnOp::StoreResult { slot: output_slot },
        RpnOp::Drop,
        RpnOp::Drop,
    ]);
    let mut ops = vec![RpnOp::Loop {
        count: rows,
        body_len: outer.len() as u32,
    }];
    ops.extend(outer);
    Ok(ops)
}

/// Compile a complete MLP using three explicitly declared temporary slots.
///
/// Gate and up projections execute once, followed by one SwiGLU pass and the
/// complete down projection: O(n_embd * n_ffn) arithmetic, with at most two
/// nested loops. [`CheckedMlpFragment::operations`] exposes four balanced blocks
/// below 32 KiB, with identical configuration pools. The concatenated
/// [`CheckedMlpFragment::operation`] may exceed 32 KiB and must not be split
/// arbitrarily inside loops. Scratch slot
/// pairings, capacity, addressing and overlap are checked against the declared
/// layout; the caller must supply a buffer with that slot size and enough slots.
/// Zero embedding width makes no writes; zero FFN width writes a zero output.
#[allow(clippy::too_many_arguments)]
pub fn compile_mlp_fragment_with_scratch_checked(
    name: &str,
    input_base: u32,
    output_slot: u32,
    gate_base: u32,
    up_base: u32,
    down_base: u32,
    n_embd: u32,
    n_ffn: u32,
    bounds: MlpBounds,
    scratch: MlpScratch,
) -> Result<CheckedMlpFragment, String> {
    scratch_layout(
        scratch,
        input_base,
        output_slot,
        gate_base,
        up_base,
        down_base,
        n_embd,
        n_ffn,
    )?;
    let mut compiled = compile_mlp_fragment_checked(
        name,
        input_base,
        output_slot,
        gate_base,
        up_base,
        down_base,
        n_embd,
        n_ffn,
        bounds,
    )?;
    let [gate, up, swiglu, down] = &compiled.stages;
    let mut ops = matrix_program(
        gate,
        input_base,
        gate_base,
        scratch.gate_slot,
        n_ffn,
        n_embd,
        0,
    )?;
    ops.extend(matrix_program(
        up,
        input_base,
        up_base,
        scratch.up_slot,
        n_ffn,
        n_embd,
        0,
    )?);
    let mut activation = Vec::new();
    swiglu.append(&mut activation, |id, height| {
        let base = match id {
            0 => scratch.gate_base,
            1 => scratch.up_base,
            _ => return Err("invalid MLP scratch activation variable".into()),
        };
        Ok(vec![
            RpnOp::Pick { depth: height },
            RpnOp::VarIndexed { base },
        ])
    })?;
    activation.extend([
        RpnOp::Swap,
        RpnOp::StoreResult {
            slot: scratch.hidden_slot,
        },
    ]);
    ops.push(RpnOp::Loop {
        count: n_ffn,
        body_len: activation.len() as u32,
    });
    ops.extend(activation);
    ops.extend(matrix_program(
        down,
        scratch.hidden_base,
        down_base,
        output_slot,
        n_embd,
        n_ffn,
        1,
    )?);
    if n_embd == 0 {
        ops.clear();
    }
    let stages = [
        (
            "gate",
            input_base,
            scratch.gate_slot,
            gate_base,
            n_ffn,
            n_embd,
        ),
        ("up", input_base, scratch.up_slot, up_base, n_ffn, n_embd),
        (
            "swiglu",
            scratch.gate_base,
            scratch.hidden_slot,
            scratch.up_base,
            n_ffn,
            1,
        ),
        (
            "down",
            scratch.hidden_base,
            output_slot,
            down_base,
            n_embd,
            n_ffn,
        ),
    ];
    let mut operations = Vec::new();
    let mut cursor = 0;
    if n_embd != 0 {
        for (stage, input_slot, output_slot, weight_base, n_rows, n_cols) in stages {
            let Some(RpnOp::Loop { body_len, .. }) = ops.get(cursor) else {
                return Err("MLP stage boundary is not a complete loop".into());
            };
            let end = cursor + 1 + *body_len as usize;
            let block = ops
                .get(cursor..end)
                .ok_or("MLP stage loop escapes operation")?;
            if std::mem::size_of_val(block) > 32768 {
                return Err(format!("MLP {stage} loop exceeds the 32 KiB plan budget"));
            }
            operations.push(OperationFragment {
                fragment: Fragment {
                    ops: block.to_vec(),
                },
                const_pool: compiled.operation.const_pool.clone(),
                meta: FragmentMeta {
                    name: format!("{name}.{stage}"),
                    input_slot,
                    output_slot,
                    weight_base,
                    n_rows,
                    n_cols,
                },
            });
            cursor = end;
        }
    }
    if cursor != ops.len() {
        return Err("MLP plan does not cover the full operation".into());
    }
    compiled.operation.fragment.ops = ops;
    compiled.operations = operations;
    compiled.scratch = Some(scratch);
    Ok(compiled)
}

/// Complete square MLP preserving the historical constructor signature.
///
/// This means `n_embd = n_ffn = n_elems`, with [`MlpBounds::default`] and an
/// owned configuration pool. It requires no implicit scratch slots. Use the
/// configurable constructor for rectangular FFNs and finite-real preflight.
/// Output storage must be disjoint from inputs and all three weight matrices.
/// The default program occupies 32,640 bytes, independent of width. Only closed
/// configuration BML is folded at compilation; data arithmetic remains BML.
///
/// # Panics
/// Panics on overflowing dimensions or declared read addresses. The fallible
/// [`compile_mlp_fragment_checked`] reports these errors without panicking.
pub fn compile_mlp_fragment(
    name: &str,
    input_base: u32,
    output_slot: u32,
    gate_base: u32,
    up_base: u32,
    down_base: u32,
    n_elems: u32,
) -> OperationFragment {
    compile_mlp_fragment_checked(
        name,
        input_base,
        output_slot,
        gate_base,
        up_base,
        down_base,
        n_elems,
        n_elems,
        MlpBounds::default(),
    )
    .expect("invalid square bounded MLP dimensions or addresses")
    .into_operation()
}
