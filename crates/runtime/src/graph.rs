//! Preflight for bounded execution of compiled graphs with complete context.
use crate::{buffer::ResultBuffer, HotLoop};
use bml_compiler::{BmlGraph, RpnOp};
use bml_domain::EvalContext;
use std::fmt;

/// Resource limits checked before entering the hot loop.
#[derive(Clone, Copy, Debug)]
pub struct GraphLimits {
    /// Maximum number of stored opcodes across all fragments.
    pub max_ops: usize,
    /// Maximum executed opcodes plus loop-iteration pushes.
    pub max_steps: usize,
    /// Maximum input values, constants, buffer values and stack values combined.
    pub max_values: usize,
}
impl Default for GraphLimits {
    fn default() -> Self {
        Self {
            max_ops: 100_000,
            max_steps: 1_000_000,
            max_values: 1_000_000,
        }
    }
}
/// Validation or execution failure. Discard partial buffer outputs on error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// Stored instructions, expanded work or values exceed their budget.
    Limit,
    /// A loop extends outside its fragment or nests beyond two levels.
    Loop,
    /// An input, constant or output slot is outside the declared context.
    Reference,
    /// An instruction consumes absent values or the graph lacks one final result.
    Stack,
    /// Execution input/buffer dimensions differ from the validated dimensions.
    Context,
    /// The bounded stack allocation failed.
    Allocation,
    /// A checked BML intermediate left the finite-real numerical domain/range.
    Numerical,
    /// Checked symbolic execution does not support an instruction in the graph.
    SymbolicBasis,
}
impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limit => "graph resource limit exceeded",
            Self::Loop => "invalid graph loop",
            Self::Reference => "invalid graph context reference",
            Self::Stack => "invalid graph stack effect",
            Self::Context => "graph execution context differs from preflight",
            Self::Allocation => "graph stack allocation failed",
            Self::Numerical => "symbolic BML intermediate outside numerical domain/range",
            Self::SymbolicBasis => "instruction outside checked symbolic basis",
        })
    }
}
impl std::error::Error for GraphError {}
/// An owned graph whose instructions and execution bounds cannot be modified.
pub struct ValidatedGraph {
    graph: BmlGraph,
    inputs: usize,
    slots: usize,
    slot_size: usize,
    peak: usize,
    steps: usize,
    external_constants: Option<usize>,
}
impl ValidatedGraph {
    /// Validate every fragment, including loop bodies with zero iterations.
    /// Buffers preserve the runtime's native IEEE and invalid-dynamic-index policy.
    pub fn new(
        graph: BmlGraph,
        inputs: usize,
        slots: usize,
        slot_size: usize,
        limits: GraphLimits,
    ) -> Result<Self, GraphError> {
        Self::validate(graph, inputs, slots, slot_size, None, limits)
    }

    /// Validate a program whose Const indices address an external weight slice.
    ///
    /// The stored pool must be empty: owned and external constant namespaces
    /// cannot be mixed implicitly. The declared count participates in reference
    /// and value-budget validation. Execution requires exactly that many values
    /// through [`Self::execute_with_constants`]; weights are never retained.
    pub fn new_with_external_constants(
        graph: BmlGraph,
        inputs: usize,
        constants: usize,
        slots: usize,
        slot_size: usize,
        limits: GraphLimits,
    ) -> Result<Self, GraphError> {
        if !graph.const_pool.is_empty() {
            return Err(GraphError::Context);
        }
        Self::validate(graph, inputs, slots, slot_size, Some(constants), limits)
    }

    fn validate(
        graph: BmlGraph,
        inputs: usize,
        slots: usize,
        slot_size: usize,
        external_constants: Option<usize>,
        limits: GraphLimits,
    ) -> Result<Self, GraphError> {
        let constant_count = external_constants.unwrap_or(graph.const_pool.len());
        let values = slots
            .checked_mul(slot_size)
            .and_then(|n| n.checked_add(inputs))
            .and_then(|n| n.checked_add(constant_count))
            .ok_or(GraphError::Limit)?;
        if values > limits.max_values || graph.fragments.len() > limits.max_ops {
            return Err(GraphError::Limit);
        }
        let mut stored = 0usize;
        for fragment in &graph.fragments {
            stored = stored
                .checked_add(fragment.ops.len())
                .ok_or(GraphError::Limit)?;
            if stored > limits.max_ops {
                return Err(GraphError::Limit);
            }
            structure(&fragment.ops, 0, inputs, constant_count, slots)?;
        }
        let mut state = StackBudget {
            height: 0,
            peak: 0,
            steps: 0,
            max_steps: limits.max_steps,
            max_stack: limits.max_values - values,
        };
        for fragment in &graph.fragments {
            state.walk(&fragment.ops)?;
        }
        if state.height != 1 {
            return Err(GraphError::Stack);
        }
        Ok(Self {
            graph,
            inputs,
            slots,
            slot_size,
            peak: state.peak,
            steps: state.steps,
            external_constants,
        })
    }
    /// Exact preflight count of opcodes plus implicit loop-index pushes.
    pub fn execution_steps(&self) -> usize {
        self.steps
    }
    /// Maximum live stack values, used for allocation before evaluation.
    pub fn stack_capacity(&self) -> usize {
        self.peak
    }
    /// Execute the same immutable graph with inputs and a caller-owned buffer.
    /// Context and allocation failures occur before modifying that buffer.
    pub fn execute(&self, inputs: &[f64], buffer: &mut ResultBuffer) -> Result<f64, GraphError> {
        if self.external_constants.is_some() {
            return Err(GraphError::Context);
        }
        self.execute_context(inputs, &self.graph.const_pool, buffer)
    }

    /// Execute using the explicitly declared external constants without copying
    /// or retaining them. Length/mode/context checks precede any buffer writes.
    /// Numeric semantics remain IEEE; this is not a finite-real domain check.
    pub fn execute_with_constants(
        &self,
        inputs: &[f64],
        constants: &[f64],
        buffer: &mut ResultBuffer,
    ) -> Result<f64, GraphError> {
        if self.external_constants != Some(constants.len()) {
            return Err(GraphError::Context);
        }
        self.execute_context(inputs, constants, buffer)
    }

    /// Execute BML-only bytecode, checking every arithmetic intermediate.
    ///
    /// This opt-in path rejects nonfinite operands, invalid logarithms,
    /// exponential overflow/underflow and invalid output offsets. Legacy
    /// execution retains IEEE semantics. Context/basis errors precede writes;
    /// a numerical error may leave earlier output cells modified.
    pub fn execute_symbolic_with_constants(
        &self,
        inputs: &[f64],
        constants: &[f64],
        buffer: &mut ResultBuffer,
    ) -> Result<f64, GraphError> {
        self.execute_symbolic_context(inputs, constants, buffer, false, false, false)
    }

    /// Execute the finite-real BML basis with bounded loops and explicit indexed
    /// context loads. Loops preserve the same preflight depth/step/stack limits.
    /// Invalid dynamic indices fail before a load; preceding writes may remain.
    /// The historical checked entry point still rejects these extra opcodes.
    pub fn execute_indexed_symbolic_with_constants(
        &self,
        inputs: &[f64],
        constants: &[f64],
        buffer: &mut ResultBuffer,
    ) -> Result<f64, GraphError> {
        self.execute_symbolic_context(inputs, constants, buffer, true, false, false)
    }

    /// Opt in to finite maximum selection alongside indexed checked BML.
    /// Existing symbolic profiles continue to reject FMax. Nonfinite operands
    /// fail with Numerical; all existing context, loop and stack limits apply.
    pub fn execute_indexed_symbolic_with_selection(
        &self,
        inputs: &[f64],
        constants: &[f64],
        buffer: &mut ResultBuffer,
    ) -> Result<f64, GraphError> {
        self.execute_symbolic_context(inputs, constants, buffer, true, true, false)
    }

    /// Execute an explicitly general operation program with finite-real checks.
    ///
    /// Unlike persisted symbolic profiles this permits native FAdd/FMul, Zero,
    /// result-buffer loads and maximum selection. Each arithmetic result must
    /// be finite; dynamic addresses must be exact nonnegative integers in range.
    /// It does not prove that native arithmetic is used only for addressing.
    /// Errors may leave earlier output writes, as in other checked entrypoints.
    pub fn execute_operation_with_constants(
        &self,
        inputs: &[f64],
        constants: &[f64],
        buffer: &mut ResultBuffer,
    ) -> Result<f64, GraphError> {
        self.execute_symbolic_context(inputs, constants, buffer, true, true, true)
    }

    fn execute_symbolic_context(
        &self,
        inputs: &[f64],
        constants: &[f64],
        buffer: &mut ResultBuffer,
        indexed: bool,
        selection: bool,
        operation: bool,
    ) -> Result<f64, GraphError> {
        if self.external_constants != Some(constants.len())
            || inputs.len() != self.inputs
            || buffer.n_slots() != self.slots
            || buffer.slot_size() != self.slot_size
        {
            return Err(GraphError::Context);
        }
        if !self.graph.fragments.iter().flat_map(|f| &f.ops).all(|op| {
            operation
                || bml_compiler::symbolic_kernels::is_symbolic_opcode(op)
                || (selection && matches!(op, RpnOp::FMax))
                || (indexed
                    && matches!(
                        op,
                        RpnOp::Loop { .. }
                            | RpnOp::ContextVarIndexed { .. }
                            | RpnOp::ContextConstIndexed { .. }
                    ))
        }) {
            return Err(GraphError::SymbolicBasis);
        }
        if inputs.iter().chain(constants).any(|v| !v.is_finite()) {
            return Err(GraphError::Numerical);
        }
        let mut stack = Vec::new();
        stack
            .try_reserve_exact(self.peak)
            .map_err(|_| GraphError::Allocation)?;
        for fragment in &self.graph.fragments {
            execute_checked_ops(&fragment.ops, inputs, constants, &mut stack, buffer)?;
        }
        stack.pop().ok_or(GraphError::Stack)
    }

    fn execute_context(
        &self,
        inputs: &[f64],
        constants: &[f64],
        buffer: &mut ResultBuffer,
    ) -> Result<f64, GraphError> {
        if inputs.len() != self.inputs
            || buffer.n_slots() != self.slots
            || buffer.slot_size() != self.slot_size
        {
            return Err(GraphError::Context);
        }
        let mut hot = HotLoop::try_with_capacity(self.peak).map_err(|_| GraphError::Allocation)?;
        let context = EvalContext::new(inputs, constants);
        Ok(hot.execute_fragments_full(&self.graph.fragments, &context, buffer))
    }
}
fn execute_checked_ops(
    ops: &[RpnOp],
    inputs: &[f64],
    constants: &[f64],
    stack: &mut Vec<f64>,
    buffer: &mut ResultBuffer,
) -> Result<(), GraphError> {
    let mut position = 0;
    while position < ops.len() {
        match ops[position] {
            RpnOp::One => stack.push(1.0),
            RpnOp::Zero => stack.push(0.0),
            RpnOp::VarIndexed { base } => {
                let offset = stack.pop().ok_or(GraphError::Stack)?;
                let index = bml_compiler::rpn::context_index(base, offset)
                    .filter(|&index| (index as usize) < buffer.len())
                    .ok_or(GraphError::Reference)?;
                let value = buffer.read_indexed(index, 0);
                if !value.is_finite() {
                    return Err(GraphError::Numerical);
                }
                stack.push(value);
            }
            RpnOp::FAdd | RpnOp::FMul => {
                let right = stack.pop().ok_or(GraphError::Stack)?;
                let left = stack.pop().ok_or(GraphError::Stack)?;
                let value = if matches!(ops[position], RpnOp::FAdd) {
                    left + right
                } else {
                    left * right
                };
                if !value.is_finite() {
                    return Err(GraphError::Numerical);
                }
                stack.push(value);
            }
            RpnOp::Var(id) => stack.push(inputs[id as usize]),
            RpnOp::Const(id) => stack.push(constants[id as usize]),
            RpnOp::ContextVarIndexed { base } | RpnOp::ContextConstIndexed { base } => {
                let offset = stack.pop().ok_or(GraphError::Stack)?;
                let index = bml_compiler::rpn::context_index(base, offset)
                    .ok_or(GraphError::Reference)? as usize;
                let context = if matches!(ops[position], RpnOp::ContextVarIndexed { .. }) {
                    inputs
                } else {
                    constants
                };
                stack.push(*context.get(index).ok_or(GraphError::Reference)?);
            }
            RpnOp::FMax => {
                let right = stack.pop().ok_or(GraphError::Stack)?;
                let left = stack.pop().ok_or(GraphError::Stack)?;
                let result = bml_compiler::rpn::finite_max(left, right);
                if !result.is_finite() {
                    return Err(GraphError::Numerical);
                }
                stack.push(result);
            }
            RpnOp::Bml => {
                let right = stack.pop().ok_or(GraphError::Stack)?;
                let left = stack.pop().ok_or(GraphError::Stack)?;
                stack.push(
                    bml_domain::symbolic::Basis::Bml
                        .apply_real(left, right)
                        .map_err(|_| GraphError::Numerical)?,
                );
            }
            RpnOp::Pick { depth } => {
                let index = stack
                    .len()
                    .checked_sub(depth as usize + 1)
                    .ok_or(GraphError::Stack)?;
                stack.push(stack[index]);
            }
            RpnOp::Dup => stack.push(*stack.last().ok_or(GraphError::Stack)?),
            RpnOp::Drop => {
                stack.pop().ok_or(GraphError::Stack)?;
            }
            RpnOp::Swap => {
                let n = stack.len();
                if n < 2 {
                    return Err(GraphError::Stack);
                }
                stack.swap(n - 1, n - 2);
            }
            RpnOp::StoreResult { slot } => {
                let offset = stack.pop().ok_or(GraphError::Stack)?;
                let value = stack.pop().ok_or(GraphError::Stack)?;
                let index = bml_compiler::rpn::context_index(0, offset)
                    .filter(|i| (*i as usize) < buffer.slot_size())
                    .ok_or(GraphError::Reference)?;
                buffer.write(slot, index, value);
            }
            RpnOp::Loop { count, body_len } => {
                let end = position + 1 + body_len as usize; // bounded during preflight
                let body = &ops[position + 1..end];
                for index in 0..count {
                    stack.push(f64::from(index));
                    execute_checked_ops(body, inputs, constants, stack, buffer)?;
                }
                position = end;
                continue;
            }
        }
        position += 1;
    }
    Ok(())
}
fn structure(
    ops: &[RpnOp],
    depth: usize,
    inputs: usize,
    constants: usize,
    slots: usize,
) -> Result<(), GraphError> {
    let mut i = 0;
    while i < ops.len() {
        match ops[i] {
            RpnOp::Var(id) | RpnOp::ContextVarIndexed { base: id } if id as usize >= inputs => {
                return Err(GraphError::Reference)
            }
            RpnOp::Const(id) | RpnOp::ContextConstIndexed { base: id }
                if id as usize >= constants =>
            {
                return Err(GraphError::Reference)
            }
            RpnOp::StoreResult { slot } if slot as usize >= slots => {
                return Err(GraphError::Reference)
            }
            RpnOp::Loop { body_len, .. } => {
                if depth >= 2 {
                    return Err(GraphError::Loop);
                }
                let end = (i + 1)
                    .checked_add(body_len as usize)
                    .filter(|end| *end <= ops.len())
                    .ok_or(GraphError::Loop)?;
                structure(&ops[i + 1..end], depth + 1, inputs, constants, slots)?;
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    Ok(())
}
struct StackBudget {
    height: usize,
    peak: usize,
    steps: usize,
    max_steps: usize,
    max_stack: usize,
}
impl StackBudget {
    fn step(&mut self) -> Result<(), GraphError> {
        self.steps = self.steps.checked_add(1).ok_or(GraphError::Limit)?;
        if self.steps > self.max_steps {
            return Err(GraphError::Limit);
        }
        Ok(())
    }
    fn effect(&mut self, take: usize, give: usize) -> Result<(), GraphError> {
        self.height = self.height.checked_sub(take).ok_or(GraphError::Stack)?;
        self.height = self.height.checked_add(give).ok_or(GraphError::Limit)?;
        if self.height > self.max_stack {
            return Err(GraphError::Limit);
        }
        self.peak = self.peak.max(self.height);
        Ok(())
    }
    fn walk(&mut self, ops: &[RpnOp]) -> Result<(), GraphError> {
        let mut i = 0;
        while i < ops.len() {
            self.step()?;
            match ops[i] {
                RpnOp::One | RpnOp::Zero | RpnOp::Var(_) | RpnOp::Const(_) => self.effect(0, 1)?,
                RpnOp::Bml | RpnOp::FAdd | RpnOp::FMul | RpnOp::FMax => self.effect(2, 1)?,
                RpnOp::Dup => self.effect(1, 2)?,
                RpnOp::Pick { depth } => {
                    if depth as usize >= self.height {
                        return Err(GraphError::Stack);
                    }
                    self.effect(0, 1)?;
                }
                RpnOp::Drop => self.effect(1, 0)?,
                RpnOp::Swap => self.effect(2, 2)?,
                RpnOp::VarIndexed { .. }
                | RpnOp::ContextVarIndexed { .. }
                | RpnOp::ContextConstIndexed { .. } => self.effect(1, 1)?,
                RpnOp::StoreResult { .. } => self.effect(2, 0)?,
                RpnOp::Loop { count, body_len } => {
                    let end = i + 1 + body_len as usize; // checked by structure
                                                         // Even an empty loop incurs one push per iteration.
                    if count as usize > self.max_steps - self.steps {
                        return Err(GraphError::Limit);
                    }
                    for _ in 0..count {
                        self.step()?;
                        self.effect(0, 1)?;
                        self.walk(&ops[i + 1..end])?;
                    }
                    i = end;
                    continue;
                }
            }
            i += 1;
        }
        Ok(())
    }
}
