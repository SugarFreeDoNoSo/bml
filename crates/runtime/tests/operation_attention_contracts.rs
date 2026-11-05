use bml_compiler::{
    op_fragments::{
        compile_attention_fragment, compile_attention_fragment_checked,
        compile_attention_fragment_with_scratch_checked, AttentionBounds, AttentionScratch,
        CheckedAttentionFragment,
    },
    BmlGraph, Fragment, RpnOp,
};
use bml_domain::EvalContext;
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
    HotLoop, Runtime,
};

fn oracle(query: &[f64], keys: &[f64], values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let width = query.len();
    let scores: Vec<_> = keys
        .chunks_exact(width)
        .map(|row| row.iter().zip(query).map(|(k, q)| k * q).sum::<f64>() / (width as f64).sqrt())
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let total: f64 = exps.iter().sum();
    let probabilities: Vec<_> = exps.iter().map(|value| value / total).collect();
    let result = (0..width)
        .map(|dimension| {
            probabilities
                .iter()
                .enumerate()
                .map(|(position, probability)| probability * values[position * width + dimension])
                .sum()
        })
        .collect();
    (result, probabilities)
}

fn near(actual: &[f64], expected: &[f64]) {
    assert!(actual.len() >= expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() < 2e-6,
            "index={index}, actual={actual}, expected={expected}"
        );
    }
}

fn fixture(
    query: &[f64],
    keys: &[f64],
    values: &[f64],
    scratch: bool,
) -> (CheckedAttentionFragment, ResultBuffer) {
    let width = query.len();
    let context = keys.len() / width;
    let size = keys.len().max(width).max(context) + 5;
    // Every operand straddles a slot boundary; bases are absolute addresses.
    let q_base = size as u32 - 1;
    let k_base = 3 * size as u32 - 2;
    let v_base = 5 * size as u32 - 1;
    let bounds = AttentionBounds {
        attention: 4.0,
        value: 8.0,
        scale_input: 128.0,
        score: 16.0,
    };
    let compiled = if scratch {
        compile_attention_fragment_with_scratch_checked(
            "attention",
            q_base,
            k_base,
            v_base,
            7,
            width as u32,
            context as u32,
            bounds,
            AttentionScratch {
                slot: 8,
                base: 8 * size as u32,
            },
        )
        .unwrap()
    } else {
        compile_attention_fragment_checked(
            "attention",
            q_base,
            k_base,
            v_base,
            7,
            width as u32,
            context as u32,
            bounds,
        )
        .unwrap()
    };
    if scratch {
        // Consumers use the public scratch descriptor to provision enough
        // slots; it must describe the layout accepted by the same operation.
        let storage = compiled
            .scratch()
            .expect("linear attention declares its scratch");
        assert_eq!(storage.base as usize, storage.slot as usize * size);
        compiled
            .validate_layout(size, storage.slot as usize + 1)
            .unwrap();
    } else {
        assert!(compiled.scratch().is_none());
    }
    compiled.validate_layout(size, 10).unwrap();
    compiled.validate_values(query, keys, values).unwrap();
    let mut buffer = ResultBuffer::new(10, size);
    for slot in 0..10 {
        buffer.slot_mut(slot).fill(77.0);
    }
    for (base, data) in [(q_base, query), (k_base, keys), (v_base, values)] {
        for (index, &value) in data.iter().enumerate() {
            buffer.write_indexed(base, index as u32, value);
        }
    }
    (compiled, buffer)
}

fn checked_graph(compiled: &CheckedAttentionFragment, buffer: &ResultBuffer) -> ValidatedGraph {
    let mut fragment = compiled.operation().fragment.clone();
    fragment.ops.push(RpnOp::One);
    let graph = BmlGraph {
        fragments: vec![fragment],
        const_pool: vec![],
        threshold: 32768,
    };
    let graph = BmlGraph::deserialize(&graph.serialize(), 32768).unwrap();
    ValidatedGraph::new_with_external_constants(
        graph,
        0,
        compiled.constants().len(),
        buffer.n_slots(),
        buffer.slot_size(),
        GraphLimits {
            max_ops: 100_000,
            max_steps: 100_000_000,
            max_values: 1_000_000,
        },
    )
    .unwrap()
}

#[test]
fn complete_attention_matches_nonuniform_signed_oracle_with_and_without_scratch() {
    let cases = [
        (
            vec![1.0, -0.5],
            vec![1.0, 0.0, -1.0, 1.0, 0.0, -2.0],
            vec![2.0, -3.0, -4.0, 0.0, 0.0, 5.0],
        ),
        (
            vec![0.0, -0.0, 0.0],
            vec![1.0, -1.0, 0.0, -2.0, 3.0, 1.0],
            vec![0.0, 2.0, -4.0, -2.0, -2.0, 0.0],
        ),
        // All scores negative; a fixed upper bound is not the actual maximum.
        (
            vec![1.0, 1.0],
            vec![-3.0, -3.0, -2.0, -2.0, -1.0, -1.0],
            vec![-1.0, 2.0, 3.0, -4.0, 5.0, 0.0],
        ),
        (
            vec![-1.0],
            vec![2.0, -1.0, 0.0, 1.0],
            vec![1.0, -2.0, 0.0, 3.0],
        ),
        (
            vec![-1.0, 0.0, 0.5],
            vec![1.0, -1.0, 0.5],
            vec![-2.0, 0.0, 3.0],
        ),
    ];
    for (query, keys, values) in cases {
        let (expected, probabilities) = oracle(&query, &keys, &values);
        for scratch in [false, true] {
            let (compiled, mut buffer) = fixture(&query, &keys, &values, scratch);
            let before = buffer.data().to_vec();
            let operation = compiled.operation();
            if scratch || compiled.context() > 1 {
                assert!(operation.fragment.ops.contains(&RpnOp::FMax));
            }
            assert!(operation.fragment.ops.contains(&RpnOp::Bml));
            let mut hot = HotLoop::with_capacity(operation.fragment.ops.len());
            hot.execute_fragment(&Fragment {
                ops: vec![RpnOp::One, RpnOp::Zero],
            });
            let initial_capacity = hot.stack_capacity();
            let parent = EvalContext::new(&[f64::NAN], &[f64::NAN]);
            for _ in 0..3 {
                hot.execute_fragment_full(
                    &operation.fragment,
                    &operation.context(&parent),
                    &mut buffer,
                );
                assert_eq!(hot.stack_depth(), 2);
                assert_eq!(hot.stack_capacity(), initial_capacity);
                near(buffer.slot(7), &expected);
                if scratch {
                    near(buffer.slot(8), &probabilities);
                }
            }
            assert_eq!(hot.stack_pop(), Some(0.0));
            assert_eq!(hot.stack_pop(), Some(1.0));
            let size = buffer.slot_size();
            for (index, (&actual, &original)) in buffer.data().iter().zip(&before).enumerate() {
                if !(7 * size..7 * size + query.len()).contains(&index)
                    && !(scratch && (8 * size..8 * size + probabilities.len()).contains(&index))
                {
                    assert_eq!(
                        actual.to_bits(),
                        original.to_bits(),
                        "unexpected write at {index}"
                    );
                }
            }
            if scratch || compiled.context() == 1 {
                let checked = checked_graph(&compiled, &buffer);
                assert_eq!(
                    checked
                        .execute_operation_with_constants(&[], compiled.constants(), &mut buffer)
                        .unwrap(),
                    1.0
                );
                near(buffer.slot(7), &expected);
                assert_eq!(
                    checked.execute_indexed_symbolic_with_constants(
                        &[],
                        compiled.constants(),
                        &mut buffer
                    ),
                    Err(GraphError::SymbolicBasis)
                );
            }
        }
    }
}

#[test]
fn historical_attention_is_one_kv_token_with_owned_constants_and_real_outputs() {
    let query = [-1.0, 0.5, 0.0, 1.0];
    let keys = [1.0, -0.5, 0.0, -1.0];
    let values = [-2.0, 0.0, 3.0, -0.5];
    let operation = compile_attention_fragment("legacy", 1, 7, 13, 3, 4);
    assert!(operation.fragment.ops.len() * std::mem::size_of::<RpnOp>() < 32768);
    let mut buffer = ResultBuffer::new(5, 6);
    for (base, data) in [(1, &query), (7, &keys), (13, &values)] {
        for (i, &value) in data.iter().enumerate() {
            buffer.write_indexed(base, i as u32, value);
        }
    }
    let mut runtime = Runtime::new(8192, 0);
    runtime.execute_fragments_sequential(
        &[operation],
        &EvalContext::new(&[], &[f64::NAN]),
        &mut buffer,
    );
    near(buffer.slot(3), &values);
    let before = buffer.data().to_vec();
    let no_op = compile_attention_fragment("empty", u32::MAX, u32::MAX, u32::MAX, u32::MAX, 0);
    assert!(no_op.fragment.ops.is_empty());
    runtime.execute_fragments_sequential(&[no_op], &EvalContext::new(&[], &[]), &mut buffer);
    assert_eq!(buffer.data(), before);
}

#[test]
fn attention_operand_bounds_are_inclusive_in_both_executable_layouts() {
    for scratch in [false, true] {
        for (query, key, value) in [
            (0.5, -0.25, 1.25),
            (4.0, 0.0, 8.0),
            (-4.0, 0.0, -8.0),
            (0.0, 4.0, 8.0),
            (0.0, -4.0, -8.0),
        ] {
            let (checked, mut buffer) = fixture(&[query], &[key], &[value], scratch);
            // A single causal cache row has probability one. Its output is
            // the literal value, including when query/key/value lie exactly
            // at either sign of their inclusive absolute declarations.
            Runtime::new(4096, 0).execute_fragments_sequential(
                std::slice::from_ref(checked.operation()),
                &EvalContext::new(&[], &[]),
                &mut buffer,
            );
            near(buffer.slot(7), &[value]);

            let outside_attention = f64::from_bits(4.0_f64.to_bits() + 1);
            let outside_value = f64::from_bits(8.0_f64.to_bits() + 1);
            for (bad_query, bad_key, bad_value) in [
                (outside_attention, 0.0, 0.0),
                (-outside_attention, 0.0, 0.0),
                (0.0, outside_attention, 0.0),
                (0.0, -outside_attention, 0.0),
                (0.0, 0.0, outside_value),
                (0.0, 0.0, -outside_value),
            ] {
                assert!(checked
                    .validate_values(&[bad_query], &[bad_key], &[bad_value])
                    .is_err());
            }
        }
    }
}

#[test]
fn attention_rejects_bad_dimensions_addresses_profiles_and_actual_intermediates() {
    let default = AttentionBounds::default();
    for (q, k, v, width, context) in [
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 65537, 1),
        (0, 0, 0, 1, 65537),
        (u32::MAX, 0, 0, 2, 1),
        (0, u32::MAX, 0, 1, 2),
        (0, 0, u32::MAX, 1, 2),
        (0, 0, 0, 65536, 65536),
    ] {
        assert!(
            compile_attention_fragment_checked("bad", q, k, v, 3, width, context, default).is_err()
        );
    }
    for bounds in [
        AttentionBounds {
            attention: 32.0,
            ..default
        },
        AttentionBounds {
            value: 0.0,
            ..default
        },
        AttentionBounds {
            scale_input: f64::NAN,
            ..default
        },
        AttentionBounds {
            score: 17.0,
            ..default
        },
    ] {
        assert!(compile_attention_fragment_checked("bad", 0, 4, 8, 3, 2, 2, bounds).is_err());
    }
    let compiled = compile_attention_fragment_checked("test", 0, 4, 8, 3, 2, 2, default).unwrap();
    for (query, keys, values) in [
        (vec![1.0], vec![1.0; 4], vec![1.0; 4]),
        (vec![1.0; 2], vec![1.0; 3], vec![1.0; 4]),
        (vec![1.0; 2], vec![1.0; 4], vec![1.0; 3]),
        (vec![f64::NAN, 0.0], vec![1.0; 4], vec![1.0; 4]),
        (vec![1.0; 2], vec![f64::INFINITY; 4], vec![1.0; 4]),
        (vec![1.0; 2], vec![1.0; 4], vec![16.01; 4]),
        // Declared operands pass, but the dot recurrence reaches 256.
        (vec![16.0; 2], vec![16.0; 4], vec![1.0; 4]),
        // Dot is evaluable but the scaled score exceeds its declared bound.
        (vec![4.0; 2], vec![4.0; 4], vec![1.0; 4]),
    ] {
        assert!(compiled.validate_values(&query, &keys, &values).is_err());
    }
    let narrow = compile_attention_fragment_checked(
        "scale",
        0,
        4,
        8,
        3,
        2,
        2,
        AttentionBounds {
            scale_input: 1.0,
            ..default
        },
    )
    .unwrap();
    assert!(narrow
        .validate_values(&[1.0; 2], &[1.0; 4], &[1.0; 4])
        .is_err());
}

#[test]
#[cfg(target_pointer_width = "64")]
fn attention_layout_admits_the_last_u32_element_without_allocating_a_buffer() {
    let checked = compile_attention_fragment_checked(
        "last-address",
        u32::MAX,
        u32::MAX,
        u32::MAX,
        0,
        1,
        1,
        AttentionBounds::default(),
    )
    .unwrap();
    checked.validate_values(&[0.25], &[0.5], &[-0.75]).unwrap();
    let domain = u32::MAX as usize + 1;
    // Read-only operands may alias. Their one element occupies the final u32
    // address, with an exclusive end of 2^32. Output is the disjoint address 0.
    // This API checks a logical layout only; no ResultBuffer is allocated.
    checked.validate_layout(domain, 1).unwrap();
    assert_eq!(
        checked.validate_layout(domain - 1, 1).unwrap_err(),
        "attention operand exceeds buffer"
    );
}

#[test]
fn attention_output_stride_is_checked_independently_of_a_valid_output_slot() {
    let checked = compile_attention_fragment_checked(
        "output-stride",
        0,
        4,
        8,
        12,
        2,
        1,
        AttentionBounds::default(),
    )
    .unwrap();
    let query = [1.0, -1.0];
    let keys = [0.5, -0.25];
    let values = [2.0, 3.0];
    checked.validate_values(&query, &keys, &values).unwrap();
    checked.validate_layout(2, 16).unwrap();
    let mut buffer = ResultBuffer::new(16, 2);
    for (base, data) in [(0, &query), (4, &keys), (8, &values)] {
        for (offset, value) in data.iter().enumerate() {
            buffer.write_indexed(base, offset as u32, *value);
        }
    }
    Runtime::new(4096, 0).execute_fragments_sequential(
        std::slice::from_ref(checked.operation()),
        &EvalContext::new(&[], &[]),
        &mut buffer,
    );
    // One cache row has probability one, so output is exactly its value row
    // up to the existing checked-attention numerical contract.
    near(buffer.slot(12), &values);
    // With stride 1, reads and output remain disjoint and inside 16 slots;
    // only the per-slot output width is invalid, not its slot number.
    assert_eq!(
        checked.validate_layout(1, 16).unwrap_err(),
        "attention output slot capacity"
    );
}

#[test]
fn attention_rejects_read_capacity_even_when_addresses_fit_u32() {
    let checked = compile_attention_fragment_checked(
        "capacity",
        2,
        4,
        6,
        0,
        2,
        1,
        AttentionBounds::default(),
    )
    .unwrap();
    // All three reads exactly fit four slots, disjoint from output slot zero.
    checked.validate_layout(2, 4).unwrap();
    // Only the physical buffer capacity is invalid; every address still fits
    // the interpreter's u32 address domain and the output slot is available.
    let error = checked.validate_layout(2, 1).unwrap_err();
    assert_eq!(error, "attention operand exceeds buffer");
}

#[test]
fn attention_layout_preflight_rejects_overlap_bad_scratch_and_capacity() {
    let bounds = AttentionBounds::default();
    let make = |q, k, v, out, scratch| {
        compile_attention_fragment_with_scratch_checked(
            "layout", q, k, v, out, 2, 3, bounds, scratch,
        )
    };
    let good = AttentionScratch { slot: 4, base: 32 };
    make(0, 8, 16, 3, good)
        .unwrap()
        .validate_layout(8, 5)
        .unwrap();
    assert!(make(0, 8, 16, 3, good)
        .unwrap()
        .validate_layout(8, 4)
        .is_err());
    assert!(make(0, 8, 16, 3, good)
        .unwrap()
        .validate_layout(1, 40)
        .is_err());
    assert!(make(24, 8, 16, 3, good)
        .unwrap()
        .validate_layout(8, 5)
        .is_err());
    assert!(make(0, 8, 32, 3, good)
        .unwrap()
        .validate_layout(8, 5)
        .is_err());
    assert!(make(0, 8, 16, 3, AttentionScratch { slot: 4, base: 31 })
        .unwrap()
        .validate_layout(8, 5)
        .is_err());
    assert!(make(0, 8, 16, 3, AttentionScratch { slot: 3, base: 24 }).is_err());
    assert!(make(
        0,
        8,
        16,
        3,
        AttentionScratch {
            slot: 4,
            base: u32::MAX
        }
    )
    .is_err());
    let valid = make(0, 8, 16, 3, good).unwrap();
    assert!(valid.validate_layout(usize::MAX, 2).is_err());
    assert!(valid.validate_layout(8, 0).is_err());
}

#[test]
fn scratch_attention_code_literals_and_stack_are_constant_across_dimensions() {
    let mut dimensions = Vec::new();
    for (width, context) in [(1, 1), (2, 4), (3, 8), (8, 16), (32, 3)] {
        let query: Vec<_> = (0..width).map(|i| (i as f64 % 3.0 - 1.0) / 8.0).collect();
        let keys: Vec<_> = (0..width * context)
            .map(|i| ((i * 7 % 17) as f64 - 8.0) / 8.0)
            .collect();
        let values: Vec<_> = (0..width * context)
            .map(|i| ((i * 3 % 13) as f64 - 6.0) / 2.0)
            .collect();
        let (compiled, mut buffer) = fixture(&query, &keys, &values, true);
        let graph = checked_graph(&compiled, &buffer);
        dimensions.push((
            compiled.operation().fragment.ops.len(),
            compiled.constants().len(),
            graph.stack_capacity(),
        ));
        assert_eq!(
            graph
                .execute_operation_with_constants(&[], compiled.constants(), &mut buffer)
                .unwrap(),
            1.0
        );
        near(buffer.slot(7), &oracle(&query, &keys, &values).0);
    }
    assert!(
        dimensions.iter().all(|size| *size == dimensions[0]),
        "{dimensions:?}"
    );
    assert!(
        dimensions[0].0 * std::mem::size_of::<RpnOp>() < 32768,
        "{dimensions:?}"
    );
    println!(
        "attention scratch ops={}, bytes={}, literals={}, peak={}",
        dimensions[0].0,
        dimensions[0].0 * std::mem::size_of::<RpnOp>(),
        dimensions[0].1,
        dimensions[0].2
    );
}

// Track whether stack entries depend on model operands. This independently
// rejects native arithmetic involving a loaded activation, even after copies,
// reductions or nested-loop carry. Configuration literals and counters are
// allowed for addressing; BML and maximum selection propagate data dependence.
fn check_index_arithmetic(ops: &[RpnOp], stack: &mut Vec<bool>) {
    let mut pc = 0;
    while pc < ops.len() {
        match ops[pc] {
            RpnOp::One | RpnOp::Zero | RpnOp::Const(_) => stack.push(false),
            RpnOp::VarIndexed { .. } => {
                assert!(!stack.pop().unwrap(), "activation used as address");
                stack.push(true);
            }
            RpnOp::Bml | RpnOp::FMax => {
                stack.pop().unwrap();
                stack.pop().unwrap();
                stack.push(true);
            }
            RpnOp::FAdd | RpnOp::FMul => {
                assert!(!stack.pop().unwrap(), "native arithmetic on data");
                assert!(!stack.pop().unwrap(), "native arithmetic on data");
                stack.push(false);
            }
            RpnOp::Pick { depth } => stack.push(stack[stack.len() - depth as usize - 1]),
            RpnOp::Dup => stack.push(*stack.last().unwrap()),
            RpnOp::Drop => {
                stack.pop().unwrap();
            }
            RpnOp::Swap => {
                let n = stack.len();
                stack.swap(n - 1, n - 2);
            }
            RpnOp::StoreResult { .. } => {
                assert!(!stack.pop().unwrap(), "activation used as output address");
                stack.pop().unwrap();
            }
            RpnOp::Loop { count, body_len } => {
                let end = pc + 1 + body_len as usize;
                let mut iteration = stack.clone();
                // The second pass includes data carried from the first pass.
                for _ in 0..2 {
                    iteration.push(false);
                    check_index_arithmetic(&ops[pc + 1..end], &mut iteration);
                }
                if count != 0 {
                    *stack = iteration;
                }
                pc = end;
                continue;
            }
            other => panic!("unexpected attention context operation: {other:?}"),
        }
        pc += 1;
    }
}

#[test]
fn attention_native_arithmetic_is_only_used_for_indices() {
    for context in [1, 3] {
        for scratch in [false, true] {
            let (compiled, _) = fixture(
                &[-1.0, 0.5],
                &vec![0.25; context * 2],
                &vec![-0.5; context * 2],
                scratch,
            );
            let mut stack = vec![true]; // unrelated caller prefix
            check_index_arithmetic(&compiled.operation().fragment.ops, &mut stack);
            assert_eq!(stack, [true]);
        }
    }
}
