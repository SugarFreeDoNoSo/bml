use bml_compiler::{BmlGraph, Fragment, RpnOp};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
};
fn graph(parts: Vec<Vec<RpnOp>>, constants: Vec<f64>) -> BmlGraph {
    BmlGraph {
        fragments: parts.into_iter().map(|ops| Fragment { ops }).collect(),
        const_pool: constants,
        threshold: 32768,
    }
}
#[test]
fn graph_preserves_stack_constants_inputs_and_results_across_fragments() {
    let g = graph(
        vec![
            vec![RpnOp::Var(0), RpnOp::Const(0), RpnOp::FMul],
            vec![RpnOp::One, RpnOp::FAdd],
        ],
        vec![3.0],
    );
    let validated = ValidatedGraph::new(g, 1, 0, 0, GraphLimits::default()).unwrap();
    assert_eq!(validated.stack_capacity(), 2);
    assert_eq!(validated.execution_steps(), 5);
    let mut buffer = ResultBuffer::new(0, 0);
    assert_eq!(validated.execute(&[2.0], &mut buffer), Ok(7.0));
    assert_eq!(validated.execute(&[-2.0], &mut buffer), Ok(-5.0));
}
#[test]
fn nested_loops_use_inclusive_work_and_stack_budgets() {
    let g = graph(
        vec![vec![
            RpnOp::Zero,
            RpnOp::Loop {
                count: 2,
                body_len: 3,
            },
            RpnOp::Drop,
            RpnOp::Loop {
                count: 3,
                body_len: 1,
            },
            RpnOp::FAdd,
        ]],
        vec![],
    );
    let limits = GraphLimits {
        max_ops: 5,
        max_steps: 20,
        max_values: 2,
    };
    let valid = ValidatedGraph::new(g.clone(), 0, 0, 0, limits).unwrap();
    assert_eq!(valid.execution_steps(), 20);
    assert_eq!(valid.execute(&[], &mut ResultBuffer::new(0, 0)), Ok(6.0));
    for limited in [
        GraphLimits {
            max_ops: 4,
            ..limits
        },
        GraphLimits {
            max_steps: 19,
            ..limits
        },
        GraphLimits {
            max_values: 1,
            ..limits
        },
    ] {
        assert!(matches!(
            ValidatedGraph::new(g.clone(), 0, 0, 0, limited),
            Err(GraphError::Limit)
        ));
    }
    let huge = graph(
        vec![vec![RpnOp::Loop {
            count: u32::MAX,
            body_len: 0,
        }]],
        vec![],
    );
    assert!(matches!(
        ValidatedGraph::new(huge, 0, 0, 0, limits),
        Err(GraphError::Limit)
    ));
}
#[test]
fn malformed_references_loops_and_stack_effects_fail_before_execution() {
    for ops in [
        vec![RpnOp::FAdd],
        vec![RpnOp::Dup],
        vec![RpnOp::One, RpnOp::Pick { depth: 1 }],
        vec![RpnOp::One, RpnOp::One],
    ] {
        assert!(matches!(
            ValidatedGraph::new(graph(vec![ops], vec![]), 0, 0, 0, GraphLimits::default()),
            Err(GraphError::Stack)
        ));
    }
    for ops in [
        vec![
            RpnOp::Loop {
                count: 1,
                body_len: 2,
            },
            RpnOp::One,
        ],
        vec![
            RpnOp::Loop {
                count: 0,
                body_len: 2,
            },
            RpnOp::Loop {
                count: 0,
                body_len: 1,
            },
            RpnOp::Loop {
                count: 0,
                body_len: 0,
            },
        ],
    ] {
        assert!(matches!(
            ValidatedGraph::new(graph(vec![ops], vec![]), 0, 0, 0, GraphLimits::default()),
            Err(GraphError::Loop)
        ));
    }
    for op in [
        RpnOp::Var(0),
        RpnOp::Const(0),
        RpnOp::StoreResult { slot: 0 },
    ] {
        let g = graph(
            vec![vec![
                RpnOp::One,
                RpnOp::Loop {
                    count: 0,
                    body_len: 1,
                },
                op,
            ]],
            vec![],
        );
        assert!(matches!(
            ValidatedGraph::new(g, 0, 0, 0, GraphLimits::default()),
            Err(GraphError::Reference)
        ));
    }
}
#[test]
fn buffer_context_mismatch_is_transactional_and_valid_stores_are_visible() {
    let g = graph(
        vec![
            vec![RpnOp::Const(0), RpnOp::Zero, RpnOp::StoreResult { slot: 0 }],
            vec![RpnOp::Zero, RpnOp::VarIndexed { base: 0 }],
        ],
        vec![17.0],
    );
    let v = ValidatedGraph::new(g, 0, 1, 1, GraphLimits::default()).unwrap();
    let mut wrong = ResultBuffer::new(2, 1);
    wrong.write(0, 0, 3.0);
    assert_eq!(v.execute(&[], &mut wrong), Err(GraphError::Context));
    assert_eq!(wrong.data(), &[3.0, 0.0]);
    let mut right = ResultBuffer::new(1, 1);
    assert_eq!(v.execute(&[], &mut right), Ok(17.0));
    assert_eq!(right.data(), &[17.0]);
    assert_eq!(v.execute(&[1.0], &mut right), Err(GraphError::Context));
    assert_eq!(right.data(), &[17.0]);
}

#[test]
fn compiled_serialized_shared_dag_executes_after_preflight() {
    use bml_compiler::{fragment_program, linearize, HashConsRegistry};
    let mut registry = HashConsRegistry::new();
    let x = registry.var(0);
    let four = registry.const_value(4.0);
    let inner = registry.bml(x, four);
    let root = registry.bml(inner, x);
    let program = linearize(&registry.into_soa(), root);
    let encoded = fragment_program(&program, 32).serialize();
    let decoded = BmlGraph::deserialize(&encoded, 32).unwrap();
    assert!(decoded.fragments.len() > 1);
    let validated = ValidatedGraph::new(decoded, 1, 0, 0, GraphLimits::default()).unwrap();
    for (input, expected) in [(1.0, 1.0), (2.0, 3.0)] {
        assert_eq!(
            validated.execute(&[input], &mut ResultBuffer::new(0, 0)),
            Ok(expected)
        );
    }
}

#[test]
fn external_constants_are_borrowed_per_execution_and_do_not_replace_owned_pools() {
    let raw = graph(
        vec![vec![RpnOp::Var(0), RpnOp::Const(0), RpnOp::Bml]],
        vec![],
    );
    let bytes = raw.serialize();
    let decoded = BmlGraph::deserialize(&bytes, 32768).unwrap();
    assert!(decoded.const_pool.is_empty());
    let validated =
        ValidatedGraph::new_with_external_constants(decoded, 1, 1, 0, 0, GraphLimits::default())
            .unwrap();
    let mut buffer = ResultBuffer::new(0, 0);
    assert_eq!(
        validated.execute_with_constants(&[1.0], &[2.0], &mut buffer),
        Ok(1.0)
    );
    assert_eq!(
        validated.execute_with_constants(&[1.0], &[4.0], &mut buffer),
        Ok(0.0)
    );
    assert_eq!(
        validated.execute(&[1.0], &mut buffer),
        Err(GraphError::Context)
    );

    let empty = ValidatedGraph::new_with_external_constants(
        graph(vec![vec![RpnOp::One]], vec![]),
        0,
        0,
        0,
        0,
        GraphLimits::default(),
    )
    .unwrap();
    assert_eq!(empty.execute_with_constants(&[], &[], &mut buffer), Ok(1.0));
    assert_eq!(empty.execute(&[], &mut buffer), Err(GraphError::Context));

    let owned = graph(vec![vec![RpnOp::Const(0)]], vec![7.0]);
    assert!(matches!(
        ValidatedGraph::new_with_external_constants(
            owned.clone(),
            0,
            1,
            0,
            0,
            GraphLimits::default()
        ),
        Err(GraphError::Context)
    ));
    let owned = ValidatedGraph::new(owned, 0, 0, 0, GraphLimits::default()).unwrap();
    assert_eq!(
        owned.execute_with_constants(&[], &[9.0], &mut buffer),
        Err(GraphError::Context)
    );
    assert_eq!(owned.execute(&[], &mut buffer), Ok(7.0));
}

#[test]
fn external_constant_context_is_checked_before_buffer_writes() {
    let raw = graph(
        vec![vec![
            RpnOp::Const(0),
            RpnOp::Dup,
            RpnOp::Zero,
            RpnOp::StoreResult { slot: 0 },
        ]],
        vec![],
    );
    let validated =
        ValidatedGraph::new_with_external_constants(raw, 0, 1, 1, 1, GraphLimits::default())
            .unwrap();
    let mut buffer = ResultBuffer::new(1, 1);
    buffer.write(0, 0, 19.0);
    for constants in [&[][..], &[1.0, 2.0][..]] {
        assert_eq!(
            validated.execute_with_constants(&[], constants, &mut buffer),
            Err(GraphError::Context)
        );
        assert_eq!(buffer.read(0, 0), 19.0);
    }
    assert_eq!(
        validated.execute_with_constants(&[1.0], &[2.0], &mut buffer),
        Err(GraphError::Context)
    );
    assert_eq!(buffer.read(0, 0), 19.0);
    let mut wrong = ResultBuffer::new(1, 2);
    wrong.write(0, 0, 23.0);
    assert_eq!(
        validated.execute_with_constants(&[], &[2.0], &mut wrong),
        Err(GraphError::Context)
    );
    assert_eq!(wrong.read(0, 0), 23.0);
    assert_eq!(
        validated.execute_with_constants(&[], &[7.0], &mut buffer),
        Ok(7.0)
    );
    assert_eq!(buffer.read(0, 0), 7.0);
}

#[test]
fn external_constant_references_and_value_budgets_are_inclusive() {
    let raw = graph(vec![vec![RpnOp::Const(1)]], vec![]);
    let limits = GraphLimits {
        max_ops: 1,
        max_steps: 1,
        max_values: 3,
    };
    let valid =
        ValidatedGraph::new_with_external_constants(raw.clone(), 0, 2, 0, 0, limits).unwrap();
    assert_eq!(
        valid.execute_with_constants(&[], &[0.0, 11.0], &mut ResultBuffer::new(0, 0)),
        Ok(11.0)
    );
    assert!(matches!(
        ValidatedGraph::new_with_external_constants(raw.clone(), 0, 1, 0, 0, limits),
        Err(GraphError::Reference)
    ));
    assert!(matches!(
        ValidatedGraph::new_with_external_constants(
            raw.clone(),
            0,
            2,
            0,
            0,
            GraphLimits {
                max_values: 2,
                ..limits
            }
        ),
        Err(GraphError::Limit)
    ));
    assert!(matches!(
        ValidatedGraph::new_with_external_constants(
            raw,
            1,
            usize::MAX,
            0,
            0,
            GraphLimits::default()
        ),
        Err(GraphError::Limit)
    ));
    let zero_loop = graph(
        vec![vec![
            RpnOp::Loop {
                count: 0,
                body_len: 1,
            },
            RpnOp::Const(2),
            RpnOp::One,
        ]],
        vec![],
    );
    assert!(matches!(
        ValidatedGraph::new_with_external_constants(zero_loop, 0, 2, 0, 0, GraphLimits::default()),
        Err(GraphError::Reference)
    ));
}
