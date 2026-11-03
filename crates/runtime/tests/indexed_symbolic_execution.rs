use bml_compiler::{BmlGraph, Fragment, RpnOp};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
};

fn validate(
    parts: Vec<Vec<RpnOp>>,
    inputs: usize,
    constants: usize,
    limits: GraphLimits,
) -> Result<ValidatedGraph, GraphError> {
    let graph = BmlGraph {
        fragments: parts.into_iter().map(|ops| Fragment { ops }).collect(),
        const_pool: vec![],
        threshold: 32768,
    };
    let graph = BmlGraph::deserialize(&graph.serialize(), 32768).unwrap();
    ValidatedGraph::new_with_external_constants(graph, inputs, constants, 0, 0, limits)
}
fn execute(graph: &ValidatedGraph, input: &[f64], constants: &[f64]) -> Result<f64, GraphError> {
    graph.execute_indexed_symbolic_with_constants(input, constants, &mut ResultBuffer::new(0, 0))
}
fn loop_parts() -> Vec<Vec<RpnOp>> {
    vec![
        vec![RpnOp::Const(0)],
        vec![
            RpnOp::Loop {
                count: 3,
                body_len: 6,
            },
            RpnOp::Dup,
            RpnOp::ContextVarIndexed { base: 1 },
            RpnOp::Swap,
            RpnOp::ContextConstIndexed { base: 2 },
            RpnOp::Bml,
            RpnOp::Bml,
        ],
        vec![RpnOp::One, RpnOp::Bml],
    ]
}

#[test]
fn indexed_symbolic_loops_preserve_context_and_cross_fragment_stack_with_exact_budgets() {
    let limits = GraphLimits {
        max_ops: 10,
        max_steps: 25,
        max_values: 12,
    };
    let graph = validate(loop_parts(), 4, 5, limits).unwrap();
    assert_eq!(graph.execution_steps(), 25);
    assert_eq!(graph.stack_capacity(), 3);
    let input = [777.0, 1.0, 1.0, 1.0];
    let constants = [1.0, 998.0, 2.0, 2.0, 2.0];
    assert_eq!(execute(&graph, &input, &constants), Ok(65536.0));
    assert_eq!(
        graph.execute_with_constants(&input, &constants, &mut ResultBuffer::new(0, 0)),
        Ok(65536.0)
    );
    assert_eq!(
        graph.execute_symbolic_with_constants(&input, &constants, &mut ResultBuffer::new(0, 0)),
        Err(GraphError::SymbolicBasis)
    );
    for bad in [
        GraphLimits {
            max_ops: 9,
            ..limits
        },
        GraphLimits {
            max_steps: 24,
            ..limits
        },
        GraphLimits {
            max_values: 11,
            ..limits
        },
    ] {
        assert!(matches!(
            validate(loop_parts(), 4, 5, bad),
            Err(GraphError::Limit)
        ));
    }
}

#[test]
fn indexed_symbolic_rejects_fractional_negative_overflowing_and_out_of_context_offsets() {
    for op in [
        RpnOp::ContextVarIndexed { base: 1 },
        RpnOp::ContextConstIndexed { base: 1 },
    ] {
        let graph = validate(vec![vec![RpnOp::Var(0), op]], 3, 3, GraphLimits::default()).unwrap();
        for offset in [-1.0, -0.5, 0.5, 2.0, u32::MAX as f64, u32::MAX as f64 + 1.0] {
            let input = [offset, 11.0, 22.0];
            assert_eq!(
                execute(&graph, &input, &[999.0, 11.0, 22.0]),
                Err(GraphError::Reference)
            );
            assert!(graph
                .execute_with_constants(&input, &[999.0, 11.0, 22.0], &mut ResultBuffer::new(0, 0))
                .unwrap()
                .is_nan());
        }
        for (offset, answer) in [(0.0, 11.0), (-0.0, 11.0), (1.0, 22.0)] {
            assert_eq!(
                execute(&graph, &[offset, 11.0, 22.0], &[999.0, 11.0, 22.0]),
                Ok(answer)
            );
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                execute(&graph, &[value, 11.0, 22.0], &[999.0, 11.0, 22.0]),
                Err(GraphError::Numerical)
            );
        }
    }
}

#[test]
fn indexed_symbolic_checks_every_bml_inside_loops_even_when_the_result_is_discarded() {
    let graph = validate(
        vec![vec![
            RpnOp::Loop {
                count: 1,
                body_len: 4,
            },
            RpnOp::Const(0),
            RpnOp::One,
            RpnOp::Bml,
            RpnOp::Drop,
            RpnOp::Drop,
            RpnOp::One,
        ]],
        0,
        1,
        GraphLimits::default(),
    )
    .unwrap();
    for value in [1024.0, -1075.0] {
        assert_eq!(
            graph.execute_with_constants(&[], &[value], &mut ResultBuffer::new(0, 0)),
            Ok(1.0)
        );
        assert_eq!(execute(&graph, &[], &[value]), Err(GraphError::Numerical));
    }
}

#[test]
fn indexed_symbolic_nested_loops_and_zero_iteration_structure_are_bounded() {
    let nested = vec![
        RpnOp::Loop {
            count: 2,
            body_len: 4,
        },
        RpnOp::Drop,
        RpnOp::Loop {
            count: 2,
            body_len: 1,
        },
        RpnOp::ContextVarIndexed { base: 0 },
        RpnOp::Bml,
        RpnOp::Bml,
    ];
    let graph = validate(vec![nested], 2, 0, GraphLimits::default()).unwrap();
    assert_eq!(execute(&graph, &[1.0, 1.0], &[]), Ok(3.0));
    for op in [
        RpnOp::ContextVarIndexed { base: 1 },
        RpnOp::ContextConstIndexed { base: 1 },
    ] {
        let bad = vec![
            RpnOp::Loop {
                count: 0,
                body_len: 1,
            },
            op,
            RpnOp::One,
        ];
        assert!(matches!(
            validate(vec![bad], 1, 1, GraphLimits::default()),
            Err(GraphError::Reference)
        ));
    }
    let malformed = vec![
        RpnOp::Loop {
            count: 0,
            body_len: 50,
        },
        RpnOp::One,
    ];
    assert!(matches!(
        validate(vec![malformed], 0, 0, GraphLimits::default()),
        Err(GraphError::Loop)
    ));
    let too_deep = vec![
        RpnOp::Loop {
            count: 0,
            body_len: 3,
        },
        RpnOp::Loop {
            count: 0,
            body_len: 2,
        },
        RpnOp::Loop {
            count: 0,
            body_len: 1,
        },
        RpnOp::Drop,
        RpnOp::One,
    ];
    assert!(matches!(
        validate(vec![too_deep], 0, 0, GraphLimits::default()),
        Err(GraphError::Loop)
    ));
    let native = vec![
        RpnOp::Loop {
            count: 0,
            body_len: 3,
        },
        RpnOp::One,
        RpnOp::One,
        RpnOp::FAdd,
        RpnOp::One,
    ];
    let graph = validate(vec![native], 0, 0, GraphLimits::default()).unwrap();
    assert_eq!(execute(&graph, &[], &[]), Err(GraphError::SymbolicBasis));
}

#[test]
fn legacy_shm_rejects_new_context_loads_explicitly() {
    let channel = bml_runtime::shm::ShmChannel::new();
    for op in [
        RpnOp::ContextVarIndexed { base: 0 },
        RpnOp::ContextConstIndexed { base: 0 },
    ] {
        let error = channel
            .write_fragment(&Fragment { ops: vec![op] })
            .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }
}
