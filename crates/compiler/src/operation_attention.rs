//! Single-head attention with explicit storage and finite-real BML arithmetic.

use super::{FragmentMeta, OperationFragment};
use crate::symbolic_bounded::{append_stage, compact_stage, linearize_stage};
use crate::{Fragment, RpnOp, RpnProgram};
use bml_domain::{transformer::BoundedOpConfig, BMLTransformer, NodeKind, NodeSoA};

/// Inclusive absolute operand declarations, never data-dependent clamps.
///
/// Products use the bounded BML templates: operand bounds are in
/// `[1/256,256]`, their product is at most 480, and each dot accumulator must
/// be below 256 before a step. Scores must have bound at most 16. These
/// declarations alone do not certify every intermediate; use preflight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionBounds {
    /// Query and key element bound.
    pub attention: f64,
    /// Cached value element bound.
    pub value: f64,
    /// Unscaled query-key dot product bound.
    pub scale_input: f64,
    /// Scaled softmax score bound.
    pub score: f64,
}

impl Default for AttentionBounds {
    fn default() -> Self {
        Self {
            attention: 16.0,
            value: 16.0,
            scale_input: 256.0,
            score: 16.0,
        }
    }
}

/// Optional scratch for linear-work attention. Its first `context` elements
/// hold scores and then probabilities. `base` must equal `slot * slot_size`.
/// Scratch and output must be disjoint from each other and from all operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionScratch {
    /// Result buffer slot to write.
    pub slot: u32,
    /// Absolute result buffer address of that slot's first element.
    pub base: u32,
}

#[derive(Clone, Debug)]
struct Stage {
    dag: NodeSoA,
    root: u32,
    code: RpnProgram,
}

impl Stage {
    fn build(
        pool: &mut Vec<f64>,
        inputs: u32,
        build: impl FnOnce(&mut BMLTransformer, &[u32]) -> Result<u32, String>,
    ) -> Result<Self, String> {
        let mut builder = BMLTransformer::new();
        let variables: Vec<_> = (0..inputs).map(|id| builder.var(id)).collect();
        let root = build(&mut builder, &variables)?;
        let (dag, roots) = compact_stage(builder.into_soa(), vec![root]);
        let root = roots[0];
        let mut code = linearize_stage(&dag, root);
        let base = u32::try_from(pool.len()).map_err(|_| "attention literal overflow")?;
        for op in &mut code.ops {
            if let RpnOp::Const(id) = op {
                *id = id.checked_add(base).ok_or("attention literal overflow")?;
            }
        }
        pool.extend_from_slice(&dag.const_pool);
        Ok(Self { dag, root, code })
    }

    fn emit(
        &self,
        output: &mut Vec<RpnOp>,
        variable: impl FnMut(u32, u32) -> Result<Vec<RpnOp>, String>,
    ) -> Result<(), String> {
        append_stage(output, self.code.clone(), variable)
    }

    fn evaluate(&self, inputs: &[f64], values: &mut [f64]) -> Result<f64, String> {
        for (index, kind) in self.dag.kinds.iter().enumerate() {
            values[index] = match *kind {
                NodeKind::One => 1.0,
                NodeKind::Zero => 0.0,
                NodeKind::Var(id) => inputs[id as usize],
                NodeKind::Const(id) => self.dag.const_pool[id as usize],
                NodeKind::Bml => bml_domain::symbolic::Basis::Bml
                    .apply_real(
                        values[self.dag.lefts[index] as usize],
                        values[self.dag.rights[index] as usize],
                    )
                    .map_err(|error| format!("attention BML node {index}: {error}"))?,
            };
        }
        Ok(values[self.root as usize])
    }
}

#[derive(Clone, Debug)]
struct Stages {
    dot: Stage,
    scale: Stage,
    exp: Stage,
    reduce: Stage,
    inverse: Stage,
    probability: Stage,
    cached_sum: Stage,
    cached_normalize: Stage,
    weighted: Stage,
    dot_config: BoundedOpConfig,
    scale_config: BoundedOpConfig,
    softmax_config: BoundedOpConfig,
    weighted_config: BoundedOpConfig,
}

impl Stages {
    fn new(pool: &mut Vec<f64>, bounds: AttentionBounds, context: u32) -> Result<Self, String> {
        let error = |e: bml_domain::transformer::BoundedBuildError| e.to_string();
        let dot_config =
            BoundedOpConfig::dot_step(bounds.attention, bounds.attention).map_err(error)?;
        let scale_config = BoundedOpConfig::product(bounds.scale_input, 1.0).map_err(error)?;
        let softmax_config =
            BoundedOpConfig::softmax(context as usize, bounds.score).map_err(error)?;
        let weighted_config = BoundedOpConfig::dot_step(1.0, bounds.value).map_err(error)?;
        let ordinary = |pool: &mut Vec<f64>, config: &BoundedOpConfig| {
            Stage::build(pool, config.inputs() as u32, |builder, variables| {
                builder
                    .operation_bounded(variables, &[], config)
                    .map(|roots| roots[0])
                    .map_err(error)
            })
        };
        let dot = ordinary(pool, &dot_config)?;
        let scale = ordinary(pool, &scale_config)?;
        let exp = Stage::build(pool, 2, |builder, inputs| {
            builder
                .softmax_bounded_exp(inputs, &softmax_config)
                .map_err(error)
        })?;
        let reduce = Stage::build(pool, 3, |builder, inputs| {
            builder
                .softmax_bounded_reduce(inputs, &softmax_config)
                .map_err(error)
        })?;
        let inverse = Stage::build(pool, 1, |builder, inputs| {
            builder
                .softmax_bounded_inverse(inputs, &softmax_config)
                .map_err(error)
        })?;
        let probability = Stage::build(pool, 3, |builder, inputs| {
            builder
                .softmax_bounded_output(inputs, &softmax_config)
                .map_err(error)
        })?;
        let cached_sum = Stage::build(pool, 2, |builder, inputs| {
            builder
                .softmax_bounded_sum(inputs, &softmax_config)
                .map_err(error)
        })?;
        let cached_normalize = Stage::build(pool, 2, |builder, inputs| {
            builder
                .softmax_bounded_normalize(inputs, &softmax_config)
                .map_err(error)
        })?;
        let weighted = ordinary(pool, &weighted_config)?;
        Ok(Self {
            dot,
            scale,
            exp,
            reduce,
            inverse,
            probability,
            cached_sum,
            cached_normalize,
            weighted,
            dot_config,
            scale_config,
            softmax_config,
            weighted_config,
        })
    }

    fn capacity(&self) -> usize {
        [
            &self.dot,
            &self.scale,
            &self.exp,
            &self.reduce,
            &self.inverse,
            &self.probability,
            &self.cached_sum,
            &self.cached_normalize,
            &self.weighted,
        ]
        .into_iter()
        .map(|stage| stage.dag.len())
        .max()
        .unwrap()
    }
}

/// Complete single-head `softmax(Q Kᵀ / sqrt(head_dim)) V`.
///
/// Query has `head_dim` elements; keys and values are row-major
/// `[context, head_dim]`. All three bases are absolute result buffer addresses;
/// output is a slot. Constants are owned and bound by the operation runtime.
/// The fragment preserves the caller's stack prefix exactly. Native FAdd/FMul
/// operate only on indices; finite FMax selects a maximum. All data arithmetic
/// (dot, scale, softmax and weighted sum) consists of BML operations.
///
/// Without scratch, scores are recomputed for each output dimension, requiring
/// O(context * head_dim²) work. Explicit scratch reduces work to
/// O(context * head_dim). Both have constant stored code and execution stack.
/// The scratch and singleton programs fit 32 KiB. The general scratch-free
/// program is larger and nests three loops: use `HotLoop` with its own strict
/// preflight, or the scratch constructor with the graph validator's two-loop
/// limit. No historical graph-validation limit is implicitly relaxed.
#[derive(Clone, Debug)]
pub struct CheckedAttentionFragment {
    operation: OperationFragment,
    q_base: u32,
    k_base: u32,
    v_base: u32,
    head_dim: u32,
    context: u32,
    bounds: AttentionBounds,
    scratch: Option<AttentionScratch>,
    stages: Stages,
}

impl CheckedAttentionFragment {
    /// Immutable operation, including bytecode, metadata and owned constants.
    pub fn operation(&self) -> &OperationFragment {
        &self.operation
    }
    /// Consume the wrapper, retaining the operation's constant pool.
    pub fn into_operation(self) -> OperationFragment {
        self.operation
    }
    /// Complete configuration pool, without model operands.
    pub fn constants(&self) -> &[f64] {
        self.operation.const_pool.as_deref().unwrap()
    }
    /// Inclusive absolute declarations used by all arithmetic stages.
    pub fn bounds(&self) -> AttentionBounds {
        self.bounds
    }
    /// Number of elements in the query and each cached key/value row.
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }
    /// Number of key/value rows included in the normalization.
    pub fn context(&self) -> u32 {
        self.context
    }
    /// Explicit temporary storage, if the linear-work constructor was used.
    pub fn scratch(&self) -> Option<AttentionScratch> {
        self.scratch
    }

    /// Validate capacity, checked address arithmetic and storage separation
    /// against the actual `ResultBuffer` layout before executing any writes.
    /// Read-only Q/K/V ranges may overlap each other.
    pub fn validate_layout(&self, slot_size: usize, slots: usize) -> Result<(), String> {
        let total = slot_size
            .checked_mul(slots)
            .ok_or("attention buffer size overflow")?;
        let width = self.head_dim as usize;
        let cache = width
            .checked_mul(self.context as usize)
            .ok_or("attention cache size overflow")?;
        let range = |base: usize, len: usize| -> Result<std::ops::Range<usize>, String> {
            let end = base.checked_add(len).ok_or("attention address overflow")?;
            if end > total || end as u64 > u64::from(u32::MAX) + 1 {
                return Err("attention operand exceeds buffer".into());
            }
            Ok(base..end)
        };
        let reads = [
            range(self.q_base as usize, width)?,
            range(self.k_base as usize, cache)?,
            range(self.v_base as usize, cache)?,
        ];
        if slot_size < width || self.operation.meta.output_slot as usize >= slots {
            return Err("attention output slot capacity".into());
        }
        let output_base = (self.operation.meta.output_slot as usize)
            .checked_mul(slot_size)
            .ok_or("attention output address overflow")?;
        let output = range(output_base, width)?;
        let overlaps = |a: &std::ops::Range<usize>, b: &std::ops::Range<usize>| {
            a.start < b.end && b.start < a.end
        };
        if reads.iter().any(|read| overlaps(read, &output)) {
            return Err("attention output overlaps operands".into());
        }
        if let Some(scratch) = self.scratch {
            if scratch.slot as usize >= slots
                || slot_size < self.context as usize
                || (scratch.slot as usize).checked_mul(slot_size) != Some(scratch.base as usize)
            {
                return Err("attention scratch slot/base/capacity mismatch".into());
            }
            let temporary = range(scratch.base as usize, self.context as usize)?;
            if overlaps(&temporary, &output) || reads.iter().any(|read| overlaps(read, &temporary))
            {
                return Err("attention scratch overlaps operands or output".into());
            }
        }
        Ok(())
    }

    /// Check actual operand shapes, declared bounds and every BML intermediate.
    /// This performs no buffer writes. Execute the unchanged operation against
    /// the same operands after preflight. Scratch for this check is O(context),
    /// separate from the constant execution stack of the emitted program.
    pub fn validate_values(
        &self,
        query: &[f64],
        keys: &[f64],
        values: &[f64],
    ) -> Result<(), String> {
        let width = self.head_dim as usize;
        let cache = width * self.context as usize;
        if query.len() != width || keys.len() != cache || values.len() != cache {
            return Err("attention operand shape mismatch".into());
        }
        for (name, data, bound) in [
            ("query", query, self.bounds.attention),
            ("key", keys, self.bounds.attention),
            ("value", values, self.bounds.value),
        ] {
            for (index, &value) in data.iter().enumerate() {
                if !value.is_finite() || value.abs() > bound {
                    return Err(format!("attention {name} bound at element {index}"));
                }
            }
        }
        let mut workspace = vec![0.0; self.stages.capacity()];
        let mut scores = Vec::with_capacity(self.context as usize + 1);
        let scale = 1.0 / f64::from(self.head_dim).sqrt();
        for (position, row) in keys.chunks_exact(width).enumerate() {
            let mut sum = 0.0;
            for (&q, &k) in query.iter().zip(row) {
                self.stages
                    .dot_config
                    .validate_values(&[sum, q, k], &[])
                    .map_err(|e| format!("attention dot {position}: {e}"))?;
                sum = self.stages.dot.evaluate(&[sum, q, k], &mut workspace)?;
            }
            self.stages
                .scale_config
                .validate_values(&[sum, scale], &[])
                .map_err(|e| format!("attention scale {position}: {e}"))?;
            scores.push(self.stages.scale.evaluate(&[sum, scale], &mut workspace)?);
        }
        let maximum = scores
            .iter()
            .copied()
            .fold(-self.bounds.score, crate::rpn::finite_max);
        scores.push(maximum);
        self.stages
            .softmax_config
            .validate_values(&scores, &[])
            .map_err(|e| format!("attention softmax: {e}"))?;
        let mut total = self
            .stages
            .exp
            .evaluate(&[scores[0], maximum], &mut workspace)?;
        for &score in &scores[1..self.context as usize] {
            total = self
                .stages
                .reduce
                .evaluate(&[total, score, maximum], &mut workspace)?;
        }
        let inverse = self.stages.inverse.evaluate(&[total], &mut workspace)?;
        let mut probabilities = Vec::with_capacity(self.context as usize);
        for &score in &scores[..self.context as usize] {
            probabilities.push(
                self.stages
                    .probability
                    .evaluate(&[score, maximum, inverse], &mut workspace)?,
            );
        }
        for dimension in 0..width {
            let mut sum = 0.0;
            for (position, &probability) in probabilities.iter().enumerate() {
                let value = values[position * width + dimension];
                self.stages
                    .weighted_config
                    .validate_values(&[sum, probability, value], &[])
                    .map_err(|e| {
                        format!("attention weighted output {dimension}, position {position}: {e}")
                    })?;
                sum = self
                    .stages
                    .weighted
                    .evaluate(&[sum, probability, value], &mut workspace)?;
            }
        }
        Ok(())
    }
}

fn loop_body(output: &mut Vec<RpnOp>, count: u32, body: Vec<RpnOp>) -> Result<(), String> {
    output.push(RpnOp::Loop {
        count,
        body_len: u32::try_from(body.len()).map_err(|_| "attention body length overflow")?,
    });
    output.extend(body);
    Ok(())
}

fn remove_below_top(output: &mut Vec<RpnOp>, count: usize) {
    for _ in 0..count {
        output.extend([RpnOp::Swap, RpnOp::Drop]);
    }
}

struct Emitter<'a> {
    stages: &'a Stages,
    query: u32,
    key: u32,
    value: u32,
    width: u32,
    stride: u32,
    scale: u32,
}

impl Emitter<'_> {
    /// Replace [..., position] by [..., position, scaled_score].
    fn score(&self, output: &mut Vec<RpnOp>) -> Result<(), String> {
        let mut body = Vec::new();
        self.stages.dot.emit(&mut body, |id, height| {
            Ok(match id {
                0 => vec![RpnOp::Pick { depth: height + 1 }],
                1 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::VarIndexed { base: self.query },
                ],
                2 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::Pick { depth: height + 3 },
                    RpnOp::Const(self.stride),
                    RpnOp::FMul,
                    RpnOp::FAdd,
                    RpnOp::VarIndexed { base: self.key },
                ],
                _ => return Err("attention dot variable".into()),
            })
        })?;
        remove_below_top(&mut body, 2);
        output.push(RpnOp::Zero);
        loop_body(output, self.width, body)?;
        self.stages.scale.emit(output, |id, height| {
            Ok(match id {
                0 => vec![RpnOp::Pick { depth: height }],
                1 => vec![RpnOp::Const(self.scale)],
                _ => return Err("attention scale variable".into()),
            })
        })?;
        remove_below_top(output, 1);
        Ok(())
    }

    /// Cache each exponential once, then retain the original ordered sum and
    /// normalization. Every BML operand is identical to the uncached stages.
    /// Input: [maximum]. Output: empty frame; scratch holds probabilities.
    fn cached_probabilities(
        &self,
        output: &mut Vec<RpnOp>,
        context: u32,
        scratch: AttentionScratch,
    ) -> Result<(), String> {
        let mut exponentials = Vec::new();
        self.stages.exp.emit(&mut exponentials, |id, height| {
            Ok(match id {
                0 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::VarIndexed { base: scratch.base },
                ],
                1 => vec![RpnOp::Pick { depth: height + 1 }],
                _ => return Err("attention cached exponential variable".into()),
            })
        })?;
        exponentials.extend([RpnOp::Swap, RpnOp::StoreResult { slot: scratch.slot }]);
        loop_body(output, context, exponentials)?;
        output.extend([
            RpnOp::Drop,
            RpnOp::Zero,
            RpnOp::VarIndexed { base: scratch.base },
        ]);
        let mut reduction = Vec::new();
        self.stages.cached_sum.emit(&mut reduction, |id, height| {
            Ok(match id {
                0 => vec![RpnOp::Pick { depth: height + 1 }],
                1 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::One,
                    RpnOp::FAdd,
                    RpnOp::VarIndexed { base: scratch.base },
                ],
                _ => return Err("attention cached sum variable".into()),
            })
        })?;
        remove_below_top(&mut reduction, 2);
        loop_body(output, context - 1, reduction)?;
        self.stages.inverse.emit(output, |id, height| {
            if id != 0 {
                return Err("attention inverse variable".into());
            }
            Ok(vec![RpnOp::Pick { depth: height }])
        })?;
        remove_below_top(output, 1);
        let mut probabilities = Vec::new();
        self.stages
            .cached_normalize
            .emit(&mut probabilities, |id, height| {
                Ok(match id {
                    0 => vec![
                        RpnOp::Pick { depth: height },
                        RpnOp::VarIndexed { base: scratch.base },
                    ],
                    1 => vec![RpnOp::Pick { depth: height + 1 }],
                    _ => return Err("attention cached normalization variable".into()),
                })
            })?;
        probabilities.extend([RpnOp::Swap, RpnOp::StoreResult { slot: scratch.slot }]);
        loop_body(output, context, probabilities)?;
        output.push(RpnOp::Drop);
        Ok(())
    }

    /// The one-token maximum is the score itself. Keep its exponential and
    /// reciprocal on the stack to execute the complete singleton softmax.
    fn singleton(&self, output: &mut Vec<RpnOp>, output_slot: u32) -> Result<(), String> {
        output.push(RpnOp::Zero);
        self.score(output)?;
        remove_below_top(output, 1);
        self.stages.exp.emit(output, |id, height| {
            if id > 1 {
                return Err("attention singleton exponential variable".into());
            }
            Ok(vec![RpnOp::Pick { depth: height }])
        })?;
        remove_below_top(output, 1);
        self.stages.inverse.emit(output, |id, height| {
            if id != 0 {
                return Err("attention singleton inverse variable".into());
            }
            Ok(vec![RpnOp::Pick { depth: height }])
        })?;
        self.stages.cached_normalize.emit(output, |id, height| {
            Ok(vec![RpnOp::Pick {
                depth: match id {
                    0 => height + 1,
                    1 => height,
                    _ => return Err("attention singleton normalization variable".into()),
                },
            }])
        })?;
        remove_below_top(output, 2);
        let mut row = Vec::new();
        self.stages.weighted.emit(&mut row, |id, height| {
            Ok(match id {
                0 => vec![RpnOp::Zero],
                1 => vec![RpnOp::Pick { depth: height + 1 }],
                2 => vec![
                    RpnOp::Pick { depth: height },
                    RpnOp::VarIndexed { base: self.value },
                ],
                _ => return Err("attention singleton weighted variable".into()),
            })
        })?;
        row.extend([RpnOp::Swap, RpnOp::StoreResult { slot: output_slot }]);
        loop_body(output, self.width, row)?;
        output.push(RpnOp::Drop);
        Ok(())
    }

    /// Initialize total from the first score, then reduce in token order.
    /// Input: [maximum]. Output: [maximum, inverse_total].
    fn denominator(&self, output: &mut Vec<RpnOp>, context: u32) -> Result<(), String> {
        output.push(RpnOp::Zero);
        self.score(output)?;
        remove_below_top(output, 1);
        self.stages.exp.emit(output, |id, height| {
            Ok(vec![RpnOp::Pick {
                depth: match id {
                    0 => height,
                    1 => height + 1,
                    _ => return Err("attention exp variable".into()),
                },
            }])
        })?;
        remove_below_top(output, 1);
        let mut reduction = vec![RpnOp::One, RpnOp::FAdd];
        self.score(&mut reduction)?;
        self.stages.reduce.emit(&mut reduction, |id, height| {
            Ok(vec![RpnOp::Pick {
                depth: match id {
                    0 => height + 2,
                    1 => height,
                    2 => height + 3,
                    _ => return Err("attention softmax reduction variable".into()),
                },
            }])
        })?;
        remove_below_top(&mut reduction, 3);
        loop_body(output, context - 1, reduction)?;
        self.stages.inverse.emit(output, |id, height| {
            if id != 0 {
                return Err("attention inverse variable".into());
            }
            Ok(vec![RpnOp::Pick { depth: height }])
        })?;
        remove_below_top(output, 1);
        Ok(())
    }

    fn weighted_outputs(
        &self,
        output: &mut Vec<RpnOp>,
        context: u32,
        output_slot: u32,
        scratch: Option<AttentionScratch>,
    ) -> Result<(), String> {
        let mut row = vec![RpnOp::Zero];
        let mut terms = Vec::new();
        if scratch.is_none() {
            self.score(&mut terms)?;
            self.stages.probability.emit(&mut terms, |id, height| {
                Ok(vec![RpnOp::Pick {
                    depth: match id {
                        0 => height,
                        1 => height + 5,
                        2 => height + 4,
                        _ => return Err("attention probability variable".into()),
                    },
                }])
            })?;
            remove_below_top(&mut terms, 1);
        }
        self.stages.weighted.emit(&mut terms, |id, height| {
            let extra = u32::from(scratch.is_none());
            Ok(match id {
                0 => vec![RpnOp::Pick {
                    depth: height + 1 + extra,
                }],
                1 => match scratch {
                    Some(scratch) => vec![
                        RpnOp::Pick { depth: height },
                        RpnOp::VarIndexed { base: scratch.base },
                    ],
                    None => vec![RpnOp::Pick { depth: height }],
                },
                2 => vec![
                    RpnOp::Pick {
                        depth: height + extra,
                    },
                    RpnOp::Const(self.stride),
                    RpnOp::FMul,
                    RpnOp::Pick {
                        depth: height + 3 + extra,
                    },
                    RpnOp::FAdd,
                    RpnOp::VarIndexed { base: self.value },
                ],
                _ => return Err("attention weighted variable".into()),
            })
        })?;
        remove_below_top(&mut terms, if scratch.is_some() { 2 } else { 3 });
        loop_body(&mut row, context, terms)?;
        row.extend([RpnOp::Swap, RpnOp::StoreResult { slot: output_slot }]);
        loop_body(output, self.width, row)?;
        if scratch.is_none() {
            output.extend([RpnOp::Drop, RpnOp::Drop]);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn compile(
    name: &str,
    q_base: u32,
    k_base: u32,
    v_base: u32,
    output_slot: u32,
    head_dim: u32,
    context: u32,
    bounds: AttentionBounds,
    scratch: Option<AttentionScratch>,
) -> Result<CheckedAttentionFragment, String> {
    if head_dim == 0 || head_dim > 65536 || context == 0 || context > 65536 {
        return Err("attention dimensions must be in 1..=65536".into());
    }
    let cache = head_dim
        .checked_mul(context)
        .ok_or("attention cache size overflow")?;
    for (base, len) in [(q_base, head_dim), (k_base, cache), (v_base, cache)] {
        base.checked_add(len - 1)
            .ok_or("attention operand address overflow")?;
    }
    if let Some(scratch) = scratch {
        scratch
            .base
            .checked_add(context - 1)
            .ok_or("attention scratch address overflow")?;
        if scratch.slot == output_slot {
            return Err("attention scratch is the output slot".into());
        }
    }
    let mut pool = Vec::new();
    let stages = Stages::new(&mut pool, bounds, context)?;
    let stride = pool.len() as u32;
    pool.push(f64::from(head_dim));
    let scale = pool.len() as u32;
    pool.push(1.0 / f64::from(head_dim).sqrt());
    let initial_max = pool.len() as u32;
    pool.push(-bounds.score);
    let emitter = Emitter {
        stages: &stages,
        query: q_base,
        key: k_base,
        value: v_base,
        width: head_dim,
        stride,
        scale,
    };
    let mut ops = Vec::new();
    if context == 1 && scratch.is_none() {
        emitter.singleton(&mut ops, output_slot)?;
    } else {
        ops.push(RpnOp::Const(initial_max));
        let mut maximum = Vec::new();
        emitter.score(&mut maximum)?;
        if let Some(scratch) = scratch {
            maximum.extend([
                RpnOp::Dup,
                RpnOp::Pick { depth: 2 },
                RpnOp::StoreResult { slot: scratch.slot },
            ]);
        }
        remove_below_top(&mut maximum, 1);
        maximum.push(RpnOp::FMax);
        loop_body(&mut ops, context, maximum)?;
        if let Some(scratch) = scratch {
            emitter.cached_probabilities(&mut ops, context, scratch)?;
        } else {
            emitter.denominator(&mut ops, context)?;
        }
        emitter.weighted_outputs(&mut ops, context, output_slot, scratch)?;
    }
    let operation = OperationFragment {
        fragment: Fragment { ops },
        const_pool: Some(pool),
        meta: FragmentMeta {
            name: name.into(),
            input_slot: q_base,
            output_slot,
            weight_base: k_base,
            n_rows: head_dim,
            n_cols: context,
        },
    };
    Ok(CheckedAttentionFragment {
        operation,
        q_base,
        k_base,
        v_base,
        head_dim,
        context,
        bounds,
        scratch,
        stages,
    })
}

/// Compile attention without external scratch. This preserves compatibility
/// with callers supplying only operand bases and an output slot. See
/// [`CheckedAttentionFragment`] for the recomputation cost and preconditions.
#[allow(clippy::too_many_arguments)]
pub fn compile_attention_fragment_checked(
    name: &str,
    q_base: u32,
    k_base: u32,
    v_base: u32,
    output_slot: u32,
    head_dim: u32,
    context: u32,
    bounds: AttentionBounds,
) -> Result<CheckedAttentionFragment, String> {
    compile(
        name,
        q_base,
        k_base,
        v_base,
        output_slot,
        head_dim,
        context,
        bounds,
        None,
    )
}

/// Compile linear-work attention with explicit scratch for `context` values.
/// The scratch base/slot correspondence and all buffer capacities are checked
/// by [`CheckedAttentionFragment::validate_layout`], using the actual layout.
#[allow(clippy::too_many_arguments)]
pub fn compile_attention_fragment_with_scratch_checked(
    name: &str,
    q_base: u32,
    k_base: u32,
    v_base: u32,
    output_slot: u32,
    head_dim: u32,
    context: u32,
    bounds: AttentionBounds,
    scratch: AttentionScratch,
) -> Result<CheckedAttentionFragment, String> {
    compile(
        name,
        q_base,
        k_base,
        v_base,
        output_slot,
        head_dim,
        context,
        bounds,
        Some(scratch),
    )
}

/// Historical signature: one query head and exactly one cached key/value token.
/// All bases are absolute buffer addresses; `n_elems` is the head dimension.
/// This executes dot, scale, singleton softmax and weighted values through BML
/// under [`AttentionBounds::default`]. It owns its constants and needs no extra
/// buffer slots. Zero width is a no-op. For explicit bounds, multiple tokens,
/// errors and finite-real preflight, use the checked constructors.
///
/// # Panics
/// Nonzero unsupported dimensions or overflowing bases require the checked API.
pub fn compile_attention_fragment(
    name: &str,
    q_base: u32,
    k_base: u32,
    v_base: u32,
    output_slot: u32,
    n_elems: u32,
) -> OperationFragment {
    if n_elems == 0 {
        return OperationFragment {
            fragment: Fragment { ops: Vec::new() },
            const_pool: Some(Vec::new()),
            meta: FragmentMeta {
                name: name.into(),
                input_slot: q_base,
                output_slot,
                weight_base: k_base,
                n_rows: 0,
                n_cols: 1,
            },
        };
    }
    compile_attention_fragment_checked(
        name,
        q_base,
        k_base,
        v_base,
        output_slot,
        n_elems,
        1,
        AttentionBounds::default(),
    )
    .expect("unsupported default attention; use the checked constructor")
    .into_operation()
}
