use bml_compiler::{BmlGraph, Fragment, RpnOp};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
};

fn checked_graph(
    parts: Vec<Vec<RpnOp>>,
    inputs: usize,
    constants: usize,
    slots: usize,
    slot_size: usize,
) -> ValidatedGraph {
    let graph = BmlGraph {
        fragments: parts.into_iter().map(|ops| Fragment { ops }).collect(),
        const_pool: Vec::new(),
        threshold: 32768,
    };
    ValidatedGraph::new_with_external_constants(
        graph,
        inputs,
        constants,
        slots,
        slot_size,
        GraphLimits::default(),
    )
    .unwrap()
}

#[test]
fn checked_symbolic_rejects_invalid_intermediates_hidden_by_finite_legacy_results() {
    let cases = [
        // Overflow is discarded by Drop; checking only the final value misses it.
        (
            vec![
                RpnOp::Const(0),
                RpnOp::One,
                RpnOp::Bml,
                RpnOp::Drop,
                RpnOp::One,
            ],
            vec![1024.0],
            1.0_f64,
        ),
        // The IEEE infinity becomes -infinity, then zero, without a Drop.
        (
            vec![
                RpnOp::Const(0),
                RpnOp::One,
                RpnOp::Bml,
                RpnOp::One,
                RpnOp::Swap,
                RpnOp::Bml,
                RpnOp::One,
                RpnOp::Bml,
            ],
            vec![1024.0],
            0.0,
        ),
        // exp2 underflow produces a finite subtraction result directly.
        (
            vec![RpnOp::Const(0), RpnOp::Const(1), RpnOp::Bml],
            vec![-1075.0, 2.0],
            -1.0,
        ),
        (
            vec![
                RpnOp::One,
                RpnOp::Const(0),
                RpnOp::Bml,
                RpnOp::Drop,
                RpnOp::One,
            ],
            vec![0.0],
            1.0,
        ),
        (
            vec![
                RpnOp::One,
                RpnOp::Const(0),
                RpnOp::Bml,
                RpnOp::Drop,
                RpnOp::One,
            ],
            vec![-1.0],
            1.0,
        ),
    ];
    for (ops, constants, expected_legacy) in cases {
        let graph = checked_graph(vec![ops], 0, constants.len(), 0, 0);
        let legacy = graph
            .execute_with_constants(&[], &constants, &mut ResultBuffer::new(0, 0))
            .unwrap();
        assert_eq!(legacy.to_bits(), expected_legacy.to_bits());
        assert_eq!(
            graph.execute_symbolic_with_constants(&[], &constants, &mut ResultBuffer::new(0, 0)),
            Err(GraphError::Numerical)
        );
    }
}

#[test]
fn checked_symbolic_matches_finite_legacy_bits_across_fragments_and_output_stores() {
    let graph = checked_graph(
        vec![
            vec![RpnOp::Var(0), RpnOp::Const(0), RpnOp::Bml, RpnOp::Dup],
            vec![
                RpnOp::One,
                RpnOp::Swap,
                RpnOp::Bml,
                RpnOp::Pick { depth: 1 },
            ],
            vec![
                RpnOp::Const(1),
                RpnOp::StoreResult { slot: 1 },
                RpnOp::Swap,
                RpnOp::Drop,
            ],
        ],
        1,
        2,
        2,
        2,
    );
    for value in [0.5_f64, 1.0, 2.0] {
        let mut checked = ResultBuffer::from_vec(2, 2, vec![77.0; 4]).unwrap();
        let mut legacy = ResultBuffer::from_vec(2, 2, vec![77.0; 4]).unwrap();
        let actual = graph
            .execute_symbolic_with_constants(&[value], &[2.0, 1.0], &mut checked)
            .unwrap();
        let prior = graph
            .execute_with_constants(&[value], &[2.0, 1.0], &mut legacy)
            .unwrap();
        let inner = value.exp2() - 1.0;
        assert_eq!(actual.to_bits(), (2.0 - inner.log2()).to_bits());
        assert_eq!(actual.to_bits(), prior.to_bits());
        assert_eq!(checked.into_vec(), vec![77.0, 77.0, 77.0, inner]);
        assert_eq!(legacy.into_vec(), vec![77.0, 77.0, 77.0, inner]);
    }
}

#[test]
fn checked_symbolic_rejects_native_opcodes_before_earlier_output_writes() {
    for native in [
        vec![RpnOp::One, RpnOp::One, RpnOp::FAdd],
        vec![RpnOp::One, RpnOp::One, RpnOp::FMul],
        vec![RpnOp::Zero],
        vec![RpnOp::Const(0), RpnOp::VarIndexed { base: 0 }],
        vec![
            RpnOp::Loop {
                count: 1,
                body_len: 1,
            },
            RpnOp::Drop,
            RpnOp::One,
        ],
    ] {
        let graph = checked_graph(
            vec![
                vec![RpnOp::One, RpnOp::Const(0), RpnOp::StoreResult { slot: 0 }],
                native,
            ],
            1,
            1,
            1,
            1,
        );
        let mut buffer = ResultBuffer::from_vec(1, 1, vec![77.0]).unwrap();
        assert_eq!(
            graph.execute_symbolic_with_constants(&[0.0], &[0.0], &mut buffer),
            Err(GraphError::SymbolicBasis)
        );
        assert_eq!(buffer.into_vec(), vec![77.0]);
    }
}

#[test]
fn checked_symbolic_store_offsets_require_exact_in_range_nonnegative_integers() {
    let graph = checked_graph(
        vec![vec![
            RpnOp::One,
            RpnOp::Const(0),
            RpnOp::StoreResult { slot: 0 },
            RpnOp::One,
        ]],
        0,
        1,
        2,
        2,
    );
    for offset in [0.5, -1.0, -0.5, 2.0, u32::MAX as f64, u32::MAX as f64 + 1.0] {
        let mut buffer = ResultBuffer::from_vec(2, 2, vec![77.0, 88.0, 99.0, 111.0]).unwrap();
        assert_eq!(
            graph.execute_symbolic_with_constants(&[], &[offset], &mut buffer),
            Err(GraphError::Reference)
        );
        assert_eq!(buffer.into_vec(), vec![77.0, 88.0, 99.0, 111.0]);
    }
    for offset in [0.0, -0.0, 1.0] {
        let mut buffer = ResultBuffer::from_vec(2, 2, vec![77.0, 88.0, 99.0, 111.0]).unwrap();
        assert_eq!(
            graph.execute_symbolic_with_constants(&[], &[offset], &mut buffer),
            Ok(1.0)
        );
        let expected = if offset == 0.0 {
            vec![1.0, 88.0, 99.0, 111.0]
        } else {
            vec![77.0, 1.0, 99.0, 111.0]
        };
        assert_eq!(buffer.into_vec(), expected);
    }
}

#[test]
fn checked_symbolic_rejects_context_and_nonfinite_external_values_before_writes() {
    let graph = checked_graph(
        vec![vec![
            RpnOp::One,
            RpnOp::Const(0),
            RpnOp::StoreResult { slot: 0 },
            RpnOp::Var(0),
        ]],
        1,
        1,
        1,
        1,
    );
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (input, constant) in [(nonfinite, 0.0), (0.0, nonfinite)] {
            let mut buffer = ResultBuffer::from_vec(1, 1, vec![77.0]).unwrap();
            assert_eq!(
                graph.execute_symbolic_with_constants(&[input], &[constant], &mut buffer),
                Err(GraphError::Numerical)
            );
            assert_eq!(buffer.into_vec(), vec![77.0]);
        }
    }
    for (input, constants) in [(vec![], vec![0.0]), (vec![0.0], vec![])] {
        let mut buffer = ResultBuffer::from_vec(1, 1, vec![77.0]).unwrap();
        assert_eq!(
            graph.execute_symbolic_with_constants(&input, &constants, &mut buffer),
            Err(GraphError::Context)
        );
        assert_eq!(buffer.into_vec(), vec![77.0]);
    }
    let mut wrong_size = ResultBuffer::new(1, 2);
    assert_eq!(
        graph.execute_symbolic_with_constants(&[0.0], &[0.0], &mut wrong_size),
        Err(GraphError::Context)
    );
    let mut wrong_slots = ResultBuffer::new(2, 1);
    assert_eq!(
        graph.execute_symbolic_with_constants(&[0.0], &[0.0], &mut wrong_slots),
        Err(GraphError::Context)
    );
}
