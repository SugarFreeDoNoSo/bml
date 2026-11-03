use bml_compiler::{fragment_program, Fragment, RpnOp, RpnProgram};
use bml_domain::EvalContext;
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
    hot_loop::HotLoop,
    shm::ShmChannel,
};

fn program(ops: Vec<RpnOp>, constants: Vec<f64>) -> RpnProgram {
    RpnProgram {
        ops,
        const_pool: constants,
    }
}
fn checked(program: &RpnProgram, inputs: usize, slots: usize, size: usize) -> ValidatedGraph {
    let mut graph = fragment_program(program, 32768);
    graph.const_pool.clear();
    ValidatedGraph::new_with_external_constants(
        graph,
        inputs,
        program.const_pool.len(),
        slots,
        size,
        GraphLimits::default(),
    )
    .unwrap()
}
fn same(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn finite_max_is_identical_in_reference_fragment_native_and_checked_evaluators() {
    let code = program(vec![RpnOp::Var(0), RpnOp::Var(1), RpnOp::FMax], vec![]);
    for (left, right, expected) in [
        (-7.0, -2.0, -2.0),
        (4.0, -2.0, 4.0),
        (-2.0, 4.0, 4.0),
        (3.25, 3.25, 3.25),
        (f64::MIN_POSITIVE, f64::from_bits(1), f64::MIN_POSITIVE),
        (-0.0, 0.0, 0.0),
        (0.0, -0.0, 0.0),
        (-0.0, -0.0, -0.0),
        (f64::NAN, 1.0, f64::NAN),
        (1.0, f64::NAN, f64::NAN),
        (f64::INFINITY, 1.0, f64::NAN),
        (1.0, f64::NEG_INFINITY, f64::NAN),
    ] {
        let inputs = [left, right];
        let context = EvalContext::new(&inputs, &[]);
        same(code.evaluate_with_ctx(&context), expected);
        let fragment = Fragment {
            ops: code.ops.clone(),
        };
        let mut stack = vec![];
        fragment.evaluate_on_stack_with_ctx(&mut stack, &context);
        same(stack[0], expected);
        same(
            HotLoop::with_capacity(8).execute_with_ctx(&code, &context),
            expected,
        );
        let graph = checked(&code, 2, 0, 0);
        let mut buffer = ResultBuffer::new(0, 0);
        same(
            graph
                .execute_with_constants(&inputs, &[], &mut buffer)
                .unwrap(),
            expected,
        );
        let selected = graph.execute_indexed_symbolic_with_selection(&inputs, &[], &mut buffer);
        if expected.is_nan() {
            assert!(matches!(selected, Err(GraphError::Numerical)));
        } else {
            same(selected.unwrap(), expected);
        }
        assert!(matches!(
            graph.execute_symbolic_with_constants(&inputs, &[], &mut buffer),
            Err(GraphError::SymbolicBasis)
        ));
        assert!(matches!(
            graph.execute_indexed_symbolic_with_constants(&inputs, &[], &mut buffer),
            Err(GraphError::SymbolicBasis)
        ));
    }
}

#[test]
fn maximum_selection_uses_runtime_loop_counters_and_preserves_cross_fragment_stack() {
    let code = program(
        vec![
            RpnOp::Const(0),
            RpnOp::Loop {
                count: 4,
                body_len: 1,
            },
            RpnOp::FMax,
        ],
        vec![-100.0],
    );
    assert_eq!(code.evaluate(), 3.0);
    assert_eq!(HotLoop::with_capacity(8).execute(&code), 3.0);
    let graph = checked(&code, 0, 0, 0);
    assert_eq!(graph.execution_steps(), 10);
    assert_eq!(
        graph
            .execute_indexed_symbolic_with_selection(&[], &[-100.0], &mut ResultBuffer::new(0, 0))
            .unwrap(),
        3.0
    );
    let nested = program(
        vec![
            RpnOp::Const(0),
            RpnOp::Loop {
                count: 2,
                body_len: 3,
            },
            RpnOp::Loop {
                count: 3,
                body_len: 1,
            },
            RpnOp::FMax,
            RpnOp::FMax,
        ],
        vec![-9.0],
    );
    assert_eq!(nested.evaluate(), 2.0);
    assert_eq!(HotLoop::with_capacity(8).execute(&nested), 2.0);
    let fragmented = program(
        vec![RpnOp::Const(0), RpnOp::Const(1), RpnOp::FMax],
        vec![-9.0, -4.0],
    );
    let graph = fragment_program(&fragmented, std::mem::size_of::<RpnOp>());
    assert_eq!(graph.evaluate(0.0), -4.0);
}

#[test]
fn checked_general_operations_validate_buffer_indices_native_arithmetic_and_every_bml() {
    let code = program(
        vec![
            RpnOp::Const(0),
            RpnOp::Loop {
                count: 3,
                body_len: 2,
            },
            RpnOp::VarIndexed { base: 2 },
            RpnOp::FMax,
            RpnOp::Zero,
            RpnOp::StoreResult { slot: 1 },
            RpnOp::One,
        ],
        vec![-100.0],
    );
    let graph = checked(&code, 0, 2, 5);
    let mut buffer = ResultBuffer::new(2, 5);
    buffer.slot_mut(0)[2..].copy_from_slice(&[-7.0, -3.0, -9.0]);
    assert_eq!(
        graph
            .execute_operation_with_constants(&[], &[-100.0], &mut buffer)
            .unwrap(),
        1.0
    );
    assert_eq!(buffer.read(1, 0), -3.0);
    assert!(matches!(
        graph.execute_indexed_symbolic_with_selection(&[], &[-100.0], &mut buffer),
        Err(GraphError::SymbolicBasis)
    ));
    buffer.write(0, 3, f64::INFINITY);
    assert!(matches!(
        graph.execute_operation_with_constants(&[], &[-100.0], &mut buffer),
        Err(GraphError::Numerical)
    ));
    for offset in [-1.0, 0.5, 10.0] {
        let code = program(
            vec![RpnOp::Const(0), RpnOp::VarIndexed { base: 0 }],
            vec![offset],
        );
        assert!(matches!(
            checked(&code, 0, 2, 5).execute_operation_with_constants(&[], &[offset], &mut buffer),
            Err(GraphError::Reference)
        ));
    }
    let overflow = program(
        vec![RpnOp::Const(0), RpnOp::Const(0), RpnOp::FMul],
        vec![f64::MAX],
    );
    assert!(matches!(
        checked(&overflow, 0, 0, 0).execute_operation_with_constants(
            &[],
            &[f64::MAX],
            &mut ResultBuffer::new(0, 0)
        ),
        Err(GraphError::Numerical)
    ));
    let bad_bml = program(vec![RpnOp::One, RpnOp::Zero, RpnOp::Bml], vec![]);
    assert!(matches!(
        checked(&bad_bml, 0, 0, 0).execute_operation_with_constants(
            &[],
            &[],
            &mut ResultBuffer::new(0, 0)
        ),
        Err(GraphError::Numerical)
    ));
    let addressed = program(
        vec![
            RpnOp::One,
            RpnOp::One,
            RpnOp::FAdd,
            RpnOp::One,
            RpnOp::FMul,
            RpnOp::VarIndexed { base: 0 },
        ],
        vec![],
    );
    assert_eq!(
        checked(&addressed, 0, 2, 5)
            .execute_operation_with_constants(&[], &[], &mut buffer)
            .unwrap(),
        -7.0
    );
}

#[test]
fn shm_maximum_tag_roundtrips_without_changing_legacy_records() {
    let channel = ShmChannel::new();
    for (ops, expected) in [
        (
            vec![RpnOp::One, RpnOp::One, RpnOp::FAdd],
            vec![3, 0, 0, 0, 0, 0, 9],
        ),
        (
            vec![RpnOp::One, RpnOp::One, RpnOp::FMax],
            vec![3, 0, 0, 0, 0, 0, 16],
        ),
    ] {
        let fragment = Fragment { ops };
        let path = channel.write_fragment(&fragment).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        assert_eq!(ShmChannel::read_fragment(&path).unwrap(), fragment);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn scheduling_respects_reads_selected_by_maximum_even_with_incomplete_metadata() {
    use bml_compiler::op_fragments::{FragmentMeta, OperationFragment};
    use bml_runtime::runtime::Runtime;
    let operation = |ops, output_slot| OperationFragment {
        fragment: Fragment { ops },
        const_pool: None,
        meta: FragmentMeta {
            name: "selection".into(),
            input_slot: 0,
            output_slot,
            weight_base: 0,
            n_rows: 1,
            n_cols: 1,
        },
    };
    let fragments = [
        operation(
            vec![
                RpnOp::Const(0),
                RpnOp::Const(1),
                RpnOp::StoreResult { slot: 1 },
            ],
            1,
        ),
        operation(
            vec![
                RpnOp::Const(1),
                RpnOp::Const(2),
                RpnOp::FMax,
                RpnOp::VarIndexed { base: 0 },
                RpnOp::Const(1),
                RpnOp::StoreResult { slot: 2 },
            ],
            2,
        ),
    ];
    let mut buffer = ResultBuffer::new(3, 2);
    let mut runtime = Runtime::new(32, 8);
    for cores in [0, 1, 2, 4] {
        buffer.slot_mut(1).fill(-99.0);
        runtime.execute_with_cores(
            &fragments,
            &EvalContext::new(&[], &[5.0, 0.0, 2.0]),
            &mut buffer,
            cores,
        );
        assert_eq!(buffer.read(2, 0), 5.0);
        assert_eq!(runtime.stack_depth(), 0);
    }
}
