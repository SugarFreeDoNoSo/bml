use bml_compiler::{
    op_fragments::{FragmentMeta, OperationFragment},
    Fragment, RpnOp,
};
use bml_domain::EvalContext;
use bml_runtime::{buffer::ResultBuffer, Runtime};
use std::sync::{Arc, Mutex};

fn fragment(ops: Vec<RpnOp>) -> OperationFragment {
    OperationFragment {
        const_pool: None,
        fragment: Fragment { ops },
        // Deliberately unrelated metadata: bytecode decides dependencies.
        meta: FragmentMeta {
            name: "fixture".into(),
            input_slot: 0,
            output_slot: 0,
            weight_base: 0,
            n_rows: 0,
            n_cols: 0,
        },
    }
}
fn store(constant: u32, slot: u32) -> OperationFragment {
    fragment(vec![
        RpnOp::Const(constant),
        RpnOp::Zero,
        RpnOp::StoreResult { slot },
    ])
}
fn copy(base: u32, slot: u32) -> OperationFragment {
    fragment(vec![
        RpnOp::Zero,
        RpnOp::VarIndexed { base },
        RpnOp::Zero,
        RpnOp::StoreResult { slot },
    ])
}
fn bits(buffer: &ResultBuffer) -> Vec<u64> {
    buffer.data().iter().map(|value| value.to_bits()).collect()
}

#[test]
fn loop_carried_swaps_preserve_the_actual_read_after_a_prior_write() {
    let constants = [0.0, 1.0, 13.0];
    let context = EvalContext::new(&[], &constants);
    for (count, expected) in [(1, [3.0_f64, 13.0, 3.0]), (2, [3.0, 13.0, 13.0])] {
        let fragments = [
            store(2, 1),
            fragment(vec![
                RpnOp::Const(0),
                RpnOp::Const(1),
                RpnOp::Loop { count, body_len: 2 },
                RpnOp::Drop,
                RpnOp::Swap,
                RpnOp::VarIndexed { base: 0 },
                RpnOp::Zero,
                RpnOp::StoreResult { slot: 2 },
                RpnOp::Drop,
            ]),
        ];
        // One swap selects original slot 0; two swaps select slot 1, which
        // must observe the preceding writer. These literal answers depend on
        // program order, not on which worker thread happens to run first.
        for cores in [1, 2] {
            let mut buffer = ResultBuffer::from_vec(3, 1, vec![3.0, 5.0, -1.0]).unwrap();
            let mut runtime = Runtime::new(16, 0);
            runtime.execute_with_cores(&fragments, &context, &mut buffer, cores);
            assert_eq!(
                bits(&buffer),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "count={count}, cores={cores}",
            );
            assert_eq!(runtime.stack_depth(), 0);
        }
    }
}

#[test]
fn zero_trip_loop_keeps_its_body_inactive_with_multiple_cores() {
    let context = EvalContext::new(&[], &[13.0, 5.0]);
    for (count, expected) in [(0, [3.0_f64, 5.0]), (1, [13.0, 5.0])] {
        let fragments = [
            fragment(vec![
                RpnOp::Loop { count, body_len: 4 },
                RpnOp::Drop,
                RpnOp::Const(0),
                RpnOp::Zero,
                RpnOp::StoreResult { slot: 0 },
            ]),
            store(1, 1),
        ];
        for cores in [1, 2] {
            let mut buffer = ResultBuffer::from_vec(2, 1, vec![3.0, -1.0]).unwrap();
            let mut runtime = Runtime::new(16, 0);
            runtime.execute_with_cores(&fragments, &context, &mut buffer, cores);
            assert_eq!(
                bits(&buffer),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "count={count}, cores={cores}",
            );
            assert_eq!(runtime.stack_depth(), 0);
        }
    }
}

#[test]
fn dependent_fragments_preserve_read_write_and_overwrite_order() {
    let fragments = vec![
        store(0, 1),
        copy(2, 2),
        store(1, 1),
        copy(4, 3),
        store(2, 4),
        store(3, 5),
        fragment(vec![
            RpnOp::Zero,
            RpnOp::VarIndexed { base: 8 },
            RpnOp::Zero,
            RpnOp::VarIndexed { base: 10 },
            RpnOp::FAdd,
            RpnOp::Zero,
            RpnOp::StoreResult { slot: 6 },
        ]),
    ];
    let constants = [11.0, 17.0, -3.0, 8.0];
    let context = EvalContext::new(&[], &constants);
    for cores in [0, 1, 2, 3, 7, 12, usize::MAX] {
        let mut buffer = ResultBuffer::new(7, 2);
        let address = buffer.data().as_ptr();
        let mut runtime = Runtime::new(32, 0);
        runtime.execute_with_cores(&fragments, &context, &mut buffer, cores);
        assert_eq!(
            buffer.data(),
            &[0.0, 0.0, 17.0, 0.0, 11.0, 0.0, 11.0, 0.0, -3.0, 0.0, 8.0, 0.0, 5.0, 0.0],
            "cores={cores}"
        );
        assert_eq!(
            buffer.data().as_ptr(),
            address,
            "must retain the caller's storage"
        );
        assert_eq!(runtime.stack_depth(), 0);
    }
}

#[test]
fn absolute_index_reads_cross_slots_and_keep_legacy_float_casts() {
    let fragments = vec![
        store(0, 1),
        fragment(vec![
            RpnOp::Var(0),
            RpnOp::VarIndexed { base: 1 },
            RpnOp::Zero,
            RpnOp::StoreResult { slot: 3 },
        ]),
    ];
    for offset in [1.0, 1.5, -9.0, f64::NAN, f64::INFINITY, f64::from(u32::MAX)] {
        let inputs = [offset];
        let constants = [47.0];
        let context = EvalContext::new(&inputs, &constants);
        let mut sequential = ResultBuffer::new(4, 2);
        sequential.slot_mut(0).copy_from_slice(&[3.0, 7.0]);
        Runtime::new(32, 0).execute_fragments_sequential(&fragments, &context, &mut sequential);
        let mut parallel = ResultBuffer::new(4, 2);
        parallel.slot_mut(0).copy_from_slice(&[3.0, 7.0]);
        Runtime::new(32, 0).execute_with_cores(&fragments, &context, &mut parallel, 2);
        assert_eq!(bits(&parallel), bits(&sequential), "offset={offset}");
        if offset == 1.0 || offset == 1.5 {
            assert_eq!(parallel.read(3, 0), 47.0);
        }
    }
}

#[test]
fn loop_carried_indices_observe_writes_from_prior_fragments() {
    let fragments = vec![
        store(0, 2),
        fragment(vec![
            RpnOp::Zero,
            RpnOp::Loop {
                count: 3,
                body_len: 7,
            },
            RpnOp::Drop,
            RpnOp::Dup,
            RpnOp::VarIndexed { base: 0 },
            RpnOp::Zero,
            RpnOp::StoreResult { slot: 4 },
            RpnOp::One,
            RpnOp::FAdd,
            RpnOp::Drop,
        ]),
    ];
    let constants = [99.0];
    let context = EvalContext::new(&[], &constants);
    let mut buffer = ResultBuffer::new(5, 1);
    Runtime::new(32, 0).execute_with_cores(&fragments, &context, &mut buffer, 2);
    assert_eq!(buffer.read(4, 0), 99.0);
}

#[test]
fn own_output_can_be_read_and_updated_inside_an_independent_wave() {
    let increment = |slot: u32| {
        fragment(vec![
            RpnOp::Loop {
                count: 3,
                body_len: 7,
            },
            RpnOp::Drop,
            RpnOp::Zero,
            RpnOp::VarIndexed { base: slot },
            RpnOp::One,
            RpnOp::FAdd,
            RpnOp::Zero,
            RpnOp::StoreResult { slot },
        ])
    };
    let fragments = [increment(0), increment(1)];
    let context = EvalContext::new(&[], &[]);
    let mut buffer = ResultBuffer::new(2, 1);
    buffer.write(0, 0, 4.0);
    buffer.write(1, 0, -9.0);
    Runtime::new(32, 0).execute_with_cores(&fragments, &context, &mut buffer, 2);
    assert_eq!(buffer.data(), &[7.0, -6.0]);
}

#[test]
fn shared_or_preexisting_stack_keeps_sequential_semantics() {
    let context = EvalContext::new(&[], &[23.0]);
    let fragments = [
        fragment(vec![RpnOp::Const(0)]),
        fragment(vec![RpnOp::Zero, RpnOp::StoreResult { slot: 0 }]),
    ];
    let mut buffer = ResultBuffer::new(2, 1);
    let mut runtime = Runtime::new(32, 0);
    runtime.execute_with_cores(&fragments, &context, &mut buffer, 4);
    assert_eq!(buffer.read(0, 0), 23.0);
    assert_eq!(runtime.stack_depth(), 0);

    runtime.execute_fragment(&Fragment {
        ops: vec![RpnOp::One],
    });
    runtime.execute_with_cores(&[store(0, 0), store(0, 1)], &context, &mut buffer, 4);
    assert_eq!(runtime.stack_depth(), 1);
    assert_eq!(buffer.data(), &[23.0, 23.0]);
}

#[test]
fn mutex_entrypoint_finishes_the_same_dependency_chain() {
    let fragments = [store(0, 1), copy(1, 2), copy(2, 3), copy(3, 4)];
    for cores in [0, 1, 2, 4, 16] {
        let buffer = Arc::new(Mutex::new(ResultBuffer::new(5, 1)));
        Runtime::new(32, 0).execute_fragments_parallel(&fragments, &[], &[61.0], &buffer, cores);
        assert_eq!(
            buffer.lock().unwrap().data(),
            &[0.0, 61.0, 61.0, 61.0, 61.0]
        );
        assert_eq!(Arc::strong_count(&buffer), 1);
    }
}

#[test]
fn empty_or_zero_sized_buffers_keep_sequential_behavior() {
    let context = EvalContext::new(&[], &[13.0]);
    for (slots, size) in [(0, 0), (3, 0), (0, 2)] {
        let mut buffer = ResultBuffer::new(slots, size);
        let mut runtime = Runtime::new(32, 0);
        runtime.execute_with_cores(&[], &context, &mut buffer, 2);
        runtime.execute_with_cores(&[store(0, 0), store(0, 1)], &context, &mut buffer, 2);
        assert!(buffer.data().is_empty());
    }
}

#[test]
fn per_operation_constant_pools_bind_in_sequential_and_parallel_execution() {
    let mut first = store(0, 0);
    first.const_pool = Some(vec![11.0]);
    let mut second = store(0, 1);
    second.const_pool = Some(vec![17.0]);
    let third = store(0, 2); // Keeps the shared context.
    let fragments = [first, second, third];
    let context = EvalContext::new(&[], &[61.0]);
    for cores in [0, 1, 2, 3, 8] {
        let mut buffer = ResultBuffer::new(3, 1);
        Runtime::new(32, 0).execute_with_cores(&fragments, &context, &mut buffer, cores);
        assert_eq!(buffer.data(), &[11.0, 17.0, 61.0]);
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(48))]
    #[test]
    fn scheduled_buffer_programs_match_an_independent_sequential_oracle(
        commands in proptest::collection::vec((0u32..25, 0u32..25, 0u32..8, 0u32..5), 1..14),
        cores in 2usize..8,
    ) {
        let mut expected: Vec<f64> = (0..18).map(|i| f64::from(i) + 0.5).collect();
        let mut buffer = ResultBuffer::from_vec(6, 3, expected.clone()).unwrap();
        let mut constants = Vec::new();
        let mut fragments = Vec::new();
        for (a, b, slot, offset) in commands {
            let value = expected.get(a as usize).copied().unwrap_or(f64::NAN)
                + expected.get(b as usize).copied().unwrap_or(f64::NAN);
            if slot < 6 && offset < 3 { expected[(slot * 3 + offset) as usize] = value; }
            let base = constants.len() as u32;
            constants.extend([f64::from(a), f64::from(b), f64::from(offset)]);
            fragments.push(fragment(vec![RpnOp::Const(base), RpnOp::VarIndexed { base: 0 },
                RpnOp::Const(base + 1), RpnOp::VarIndexed { base: 0 }, RpnOp::FAdd,
                RpnOp::Const(base + 2), RpnOp::StoreResult { slot }]));
        }
        let context = EvalContext::new(&[], &constants);
        let mut runtime = Runtime::new(32, 0);
        runtime.execute_with_cores(&fragments, &context, &mut buffer, cores);
        for (&actual, &expected) in buffer.data().iter().zip(&expected) {
            proptest::prop_assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
        }
        proptest::prop_assert_eq!(runtime.stack_depth(), 0);
    }
}
