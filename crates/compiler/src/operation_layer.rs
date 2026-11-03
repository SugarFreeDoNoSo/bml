//! Complete single-head causal layer composition with explicit buffer addresses.

use super::{
    compile_attention_fragment_with_scratch_checked, compile_matmul_fragment_checked,
    compile_mlp_fragment_with_scratch_checked, compile_rmsnorm_fragment_checked, AttentionBounds,
    AttentionScratch, FragmentMeta, MatmulBounds, MlpBounds, MlpScratch, OperationFragment,
};
use crate::{
    bml_ops::{bml_rope_vector_program, RopeVectorConfig},
    symbolic_bounded::{append_stage, compact_stage, linearize_stage, BoundedProfile},
    Fragment, RpnOp,
};
use bml_domain::{transformer::BoundedOpConfig, BMLTransformer};

/// Absolute element bases of the nine learned tensors of one layer.
/// Matrices are row-major: Q/K/V/O are `[embedding, embedding]`, gate/up
/// are `[feed_forward, embedding]`, and down is `[embedding, feed_forward]`.
#[derive(Clone, Copy, Debug)]
pub struct LayerWeights {
    /// Attention RMS learned vector.
    pub norm: u32,
    /// Query matrix.
    pub query: u32,
    /// Key matrix.
    pub key: u32,
    /// Value matrix.
    pub value: u32,
    /// Attention output matrix.
    pub output: u32,
    /// Feed-forward RMS learned vector.
    pub ffn_norm: u32,
    /// Gate matrix.
    pub gate: u32,
    /// Up matrix.
    pub up: u32,
    /// Down matrix.
    pub down: u32,
}

/// One query head, one KV head, and the explicit current causal position.
/// This operation-level API does not infer GQA, tokenization or model metadata.
#[derive(Clone, Copy, Debug)]
pub struct LayerConfig {
    /// Positive even head and embedding width.
    pub embedding: u32,
    /// Positive feed-forward width, independent of embedding width.
    pub feed_forward: u32,
    /// Current token position; cache rows before this position must exist.
    pub position: u32,
    /// Positive finite epsilon used by both RMS normalizations.
    pub epsilon: f64,
    /// Positive finite RoPE frequency base.
    pub frequency_base: f64,
    /// Partial finite-real declarations for all arithmetic families.
    pub bounds: BoundedProfile,
}

/// Disjoint writable slots and an absolute input address.
/// The fifteen scratch slots, output, and two KV slots must be distinct and
/// must not overlap the input vector or any learned tensor. The KV slots store
/// row-major `[context, embedding]` values, including earlier rotated keys.
#[derive(Clone, Copy, Debug)]
pub struct LayerLayout {
    /// Shared slot size of the execution buffer, in f64 elements.
    pub slot_size: u32,
    /// Absolute input vector base; it need not begin at a slot boundary.
    pub input_base: u32,
    /// Output slot for the final residual sum.
    pub output_slot: u32,
    /// First of fifteen consecutive temporary slots.
    pub first_scratch_slot: u32,
    /// Persistent rotated-key cache slot.
    pub key_cache_slot: u32,
    /// Persistent value cache slot.
    pub value_cache_slot: u32,
}

/// Executable operations and the validated single-head buffer contract.
/// Construction checks shapes, addresses, overlap and declared numerical
/// profiles. It cannot inspect runtime operands: bounds and evaluability of
/// every BML remain execution preconditions, as for the component APIs.
/// The legacy IEEE executor is not transactional; callers must only advance
/// their session position after successful numerical validation of the output.
#[derive(Clone, Debug)]
pub struct CompiledLayerFragments {
    operations: Vec<OperationFragment>,
    config: LayerConfig,
    layout: LayerLayout,
    required_slots: usize,
}

impl CompiledLayerFragments {
    /// Ordered operations; dependency-aware runtime APIs preserve this order.
    pub fn operations(&self) -> &[OperationFragment] {
        &self.operations
    }

    /// Consume the plan, retaining each operation's owned constants.
    pub fn into_operations(self) -> Vec<OperationFragment> {
        self.operations
    }

    /// Validated dimensions, current position and numerical declarations.
    pub fn config(&self) -> LayerConfig {
        self.config
    }

    /// Validated slot/address contract.
    pub fn layout(&self) -> LayerLayout {
        self.layout
    }

    /// Minimum number of slots, covering all read and written ranges.
    pub fn required_slots(&self) -> usize {
        self.required_slots
    }
}

fn range(base: u32, count: u32) -> Result<(u64, u64), String> {
    let end = u64::from(base) + u64::from(count);
    if end > u64::from(u32::MAX) + 1 {
        return Err("layer absolute address overflow".into());
    }
    Ok((u64::from(base), end))
}

fn slot_base(slot: u32, size: u32) -> Result<u32, String> {
    slot.checked_mul(size)
        .ok_or_else(|| "layer slot address overflow".into())
}

fn validate_layout(
    layout: LayerLayout,
    weights: LayerWeights,
    config: LayerConfig,
) -> Result<usize, String> {
    let n = config.embedding;
    let ffn = config.feed_forward;
    if n == 0 || !n.is_multiple_of(2) || ffn == 0 {
        return Err("layer requires a positive even embedding and a positive FFN width".into());
    }
    let context = config
        .position
        .checked_add(1)
        .ok_or("layer context overflow")?;
    let cache_len = context.checked_mul(n).ok_or("layer cache size overflow")?;
    if layout.slot_size < cache_len || layout.slot_size < ffn {
        return Err("layer slots cannot hold the complete KV prefix and FFN vectors".into());
    }
    let matrix = n.checked_mul(n).ok_or("layer matrix size overflow")?;
    let ffn_matrix = n.checked_mul(ffn).ok_or("layer FFN matrix size overflow")?;
    let mut reads = vec![range(layout.input_base, n)?];
    for (base, count) in [
        (weights.norm, n),
        (weights.query, matrix),
        (weights.key, matrix),
        (weights.value, matrix),
        (weights.output, matrix),
        (weights.ffn_norm, n),
        (weights.gate, ffn_matrix),
        (weights.up, ffn_matrix),
        (weights.down, ffn_matrix),
    ] {
        reads.push(range(base, count)?);
    }
    let mut slots = vec![
        layout.output_slot,
        layout.key_cache_slot,
        layout.value_cache_slot,
    ];
    for index in 0..15 {
        slots.push(
            layout
                .first_scratch_slot
                .checked_add(index)
                .ok_or("layer scratch slot overflow")?,
        );
    }
    slots.sort_unstable();
    if slots.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("layer writable slots overlap".into());
    }
    let mut end = reads.iter().map(|r| r.1).max().unwrap();
    for slot in slots {
        let write = range(slot_base(slot, layout.slot_size)?, layout.slot_size)?;
        if reads
            .iter()
            .any(|read| read.0 < write.1 && write.0 < read.1)
        {
            return Err("layer output, cache or scratch overlaps an input/weight tensor".into());
        }
        end = end.max(write.1);
    }
    usize::try_from(end.div_ceil(u64::from(layout.slot_size)))
        .map_err(|_| "layer buffer size overflow".into())
}

fn copy_cache_row(
    name: String,
    input: u32,
    output: u32,
    width: u32,
    row: u32,
) -> OperationFragment {
    let body = vec![
        RpnOp::Dup,
        RpnOp::VarIndexed { base: input },
        RpnOp::Pick { depth: 1 },
        RpnOp::Const(0),
        RpnOp::FAdd,
        RpnOp::StoreResult { slot: output },
        RpnOp::Drop,
    ];
    let mut ops = vec![RpnOp::Loop {
        count: width,
        body_len: body.len() as u32,
    }];
    ops.extend(body);
    OperationFragment {
        fragment: Fragment { ops },
        const_pool: Some(vec![f64::from(row) * f64::from(width)]),
        meta: FragmentMeta {
            name,
            input_slot: input,
            output_slot: output,
            weight_base: 0,
            n_rows: width,
            n_cols: 1,
        },
    }
}

fn residual(
    name: String,
    left: u32,
    right: u32,
    output: u32,
    width: u32,
    bound: f64,
) -> Result<OperationFragment, String> {
    let config = BoundedOpConfig::add(bound, bound).map_err(|e| e.to_string())?;
    let mut builder = BMLTransformer::new();
    let inputs = [builder.var(0), builder.var(1)];
    let root = builder
        .operation_bounded(&inputs, &[], &config)
        .map_err(|e| e.to_string())?[0];
    let (dag, roots) = compact_stage(builder.into_soa(), vec![root]);
    let mut body = Vec::new();
    append_stage(&mut body, linearize_stage(&dag, roots[0]), |id, height| {
        Ok(vec![
            RpnOp::Pick { depth: height },
            RpnOp::VarIndexed {
                base: if id == 0 { left } else { right },
            },
        ])
    })?;
    body.extend([RpnOp::Swap, RpnOp::StoreResult { slot: output }]);
    let mut ops = vec![RpnOp::Loop {
        count: width,
        body_len: body.len() as u32,
    }];
    ops.extend(body);
    Ok(OperationFragment {
        fragment: Fragment { ops },
        const_pool: Some(dag.const_pool),
        meta: FragmentMeta {
            name,
            input_slot: left,
            output_slot: output,
            weight_base: right,
            n_rows: width,
            n_cols: 1,
        },
    })
}

/// Compose a complete causal layer, including both original residuals.
///
/// Earlier cache rows are read-only. The current projected K/V row is written
/// after RoPE on K; attention reads precisely `position + 1` rows. Configuration
/// and addressing are checked before returning any plan. Each returned fragment
/// owns its constants, so caller context constants cannot alter its layout.
/// KV storage is f64 here, not the separate model engine's F32 cache contract.
pub fn compile_layer_fragments_checked(
    layer: u32,
    layout: LayerLayout,
    weights: LayerWeights,
    config: LayerConfig,
) -> Result<CompiledLayerFragments, String> {
    let required_slots = validate_layout(layout, weights, config)?;
    config
        .bounds
        .validate(config.embedding as usize, config.epsilon)?;
    let slots: Vec<u32> = (0..15).map(|i| layout.first_scratch_slot + i).collect();
    let bases: Vec<u32> = slots
        .iter()
        .map(|&s| slot_base(s, layout.slot_size))
        .collect::<Result<_, _>>()?;
    let n = config.embedding;
    let bounds = config.bounds;
    let name = |suffix: &str| format!("blk.{layer}.{suffix}");
    let mut operations = Vec::new();
    operations.push(
        compile_rmsnorm_fragment_checked(
            &name("attn_norm"),
            layout.input_base,
            slots[0],
            weights.norm,
            n,
            config.epsilon,
            bounds.rms,
        )?
        .into_operation(),
    );
    let projection = MatmulBounds {
        input: bounds.projection_input,
        weight: bounds.projection_weight,
    };
    for (suffix, slot, weight) in [
        ("attn_q", slots[1], weights.query),
        ("attn_k", slots[2], weights.key),
        ("attn_v", slots[3], weights.value),
    ] {
        operations.push(
            compile_matmul_fragment_checked(
                &name(suffix),
                bases[0],
                slot,
                weight,
                n,
                n,
                projection,
            )?
            .operation()
            .clone(),
        );
    }
    let rope = RopeVectorConfig {
        heads: 1,
        head_dim: n,
        position: config.position,
        frequency_base: config.frequency_base,
        input_bound: bounds.rope,
    };
    for (suffix, input, slot) in [
        ("rope_q", bases[1], slots[4]),
        ("rope_k", bases[2], slots[5]),
    ] {
        let compiled = bml_rope_vector_program(input, slot, rope)?;
        let program = compiled.program();
        operations.push(OperationFragment {
            fragment: Fragment {
                ops: program.ops.clone(),
            },
            const_pool: Some(program.const_pool.clone()),
            meta: FragmentMeta {
                name: name(suffix),
                input_slot: input,
                output_slot: slot,
                weight_base: 0,
                n_rows: n,
                n_cols: 1,
            },
        });
    }
    operations.push(copy_cache_row(
        name("cache_k"),
        bases[5],
        layout.key_cache_slot,
        n,
        config.position,
    ));
    operations.push(copy_cache_row(
        name("cache_v"),
        bases[3],
        layout.value_cache_slot,
        n,
        config.position,
    ));
    operations.push(
        compile_attention_fragment_with_scratch_checked(
            &name("attention"),
            bases[4],
            slot_base(layout.key_cache_slot, layout.slot_size)?,
            slot_base(layout.value_cache_slot, layout.slot_size)?,
            slots[6],
            n,
            config.position + 1,
            AttentionBounds {
                attention: bounds.attention,
                value: bounds.value,
                score: bounds.score,
                scale_input: bounds.scale_input,
            },
            AttentionScratch {
                slot: slots[14],
                base: bases[14],
            },
        )?
        .into_operation(),
    );
    operations.push(
        compile_matmul_fragment_checked(
            &name("attn_output"),
            bases[6],
            slots[7],
            weights.output,
            n,
            n,
            projection,
        )?
        .operation()
        .clone(),
    );
    operations.push(residual(
        name("attn_residual"),
        layout.input_base,
        bases[7],
        slots[8],
        n,
        bounds.residual,
    )?);
    operations.push(
        compile_rmsnorm_fragment_checked(
            &name("ffn_norm"),
            bases[8],
            slots[9],
            weights.ffn_norm,
            n,
            config.epsilon,
            bounds.rms,
        )?
        .into_operation(),
    );
    operations.extend(
        compile_mlp_fragment_with_scratch_checked(
            &name("mlp"),
            bases[9],
            slots[10],
            weights.gate,
            weights.up,
            weights.down,
            n,
            config.feed_forward,
            MlpBounds {
                input: bounds.projection_input,
                gate_weight: bounds.projection_weight,
                up_weight: bounds.projection_weight,
                gate: bounds.gate,
                up: bounds.up,
                hidden: bounds.projection_input,
                down_weight: bounds.projection_weight,
            },
            MlpScratch {
                gate_slot: slots[11],
                gate_base: bases[11],
                up_slot: slots[12],
                up_base: bases[12],
                hidden_slot: slots[13],
                hidden_base: bases[13],
            },
        )?
        .into_operations(),
    );
    operations.push(residual(
        name("ffn_residual"),
        bases[8],
        bases[10],
        layout.output_slot,
        n,
        bounds.residual,
    )?);
    Ok(CompiledLayerFragments {
        operations,
        config,
        layout,
        required_slots,
    })
}

/// Compatibility entry point for a single head at position zero with square
/// FFN, epsilon 1e-5 and RoPE base 10,000. `input_slot` is a slot index, while
/// learned tensor bases are absolute element addresses. The actual buffer's
/// slot size must equal `n_embd`. Seventeen consecutive slots after the input
/// hold scratch and KV; the final result is in `input_slot + 18`.
/// `n_cols_const_id` is retained for source compatibility but is unused: every
/// operation owns the correct stride and configuration constants.
///
/// # Panics
/// Invalid dimensions, overflow or overlap of reserved writable slots with
/// learned tensors require the explicit checked constructor's error path.
#[allow(clippy::too_many_arguments)]
pub fn compile_layer_fragments(
    layer: u32,
    input_slot: u32,
    n_embd: u32,
    norm_base: u32,
    q_base: u32,
    k_base: u32,
    v_base: u32,
    o_base: u32,
    ffn_norm_base: u32,
    gate_base: u32,
    up_base: u32,
    down_base: u32,
    _n_cols_const_id: u32,
) -> Vec<OperationFragment> {
    let offset = |n| {
        input_slot
            .checked_add(n)
            .expect("layer slot overflow; use checked constructor")
    };
    let k = f64::from(n_embd).sqrt().ceil() + 1.0;
    let mut bounds = BoundedProfile::default();
    bounds.rms.weight = (479.0 / k).min(16.0);
    compile_layer_fragments_checked(
        layer,
        LayerLayout {
            slot_size: n_embd,
            input_base: slot_base(input_slot, n_embd).expect("layer input address overflow"),
            output_slot: offset(18),
            first_scratch_slot: offset(1),
            key_cache_slot: offset(16),
            value_cache_slot: offset(17),
        },
        LayerWeights {
            norm: norm_base,
            query: q_base,
            key: k_base,
            value: v_base,
            output: o_base,
            ffn_norm: ffn_norm_base,
            gate: gate_base,
            up: up_base,
            down: down_base,
        },
        LayerConfig {
            embedding: n_embd,
            feed_forward: n_embd,
            position: 0,
            epsilon: 1e-5,
            frequency_base: 10000.0,
            bounds,
        },
    )
    .expect("unsupported layer layout; use compile_layer_fragments_checked")
    .into_operations()
}
