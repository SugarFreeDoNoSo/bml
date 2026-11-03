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
/// Validation or execution setup failure; no partial result is returned.
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
            RpnOp::Var(id) if id as usize >= inputs => return Err(GraphError::Reference),
            RpnOp::Const(id) if id as usize >= constants => return Err(GraphError::Reference),
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
                RpnOp::Bml | RpnOp::FAdd | RpnOp::FMul => self.effect(2, 1)?,
                RpnOp::Dup => self.effect(1, 2)?,
                RpnOp::Pick { depth } => {
                    if depth as usize >= self.height {
                        return Err(GraphError::Stack);
                    }
                    self.effect(0, 1)?;
                }
                RpnOp::Drop => self.effect(1, 0)?,
                RpnOp::Swap => self.effect(2, 2)?,
                RpnOp::VarIndexed { .. } => self.effect(1, 1)?,
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
