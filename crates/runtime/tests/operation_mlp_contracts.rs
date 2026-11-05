use bml_compiler::{
    op_fragments::{
        compile_mlp_fragment, compile_mlp_fragment_checked,
        compile_mlp_fragment_with_scratch_checked, CheckedMlpFragment, MlpBounds, MlpScratch,
        OperationFragment,
    },
    BmlGraph, Fragment, RpnOp,
};
use bml_domain::{
    symbolic::Basis, transformer::BoundedOpConfig, BMLTransformer, EvalContext, NodeKind,
};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
    HotLoop, Runtime,
};

#[derive(Clone, Copy)]
struct Layout {
    size: usize,
    input: u32,
    gate: u32,
    up: u32,
    down: u32,
    scratch: MlpScratch,
}
impl Layout {
    fn new(embd: u32, ffn: u32) -> Self {
        let size = (embd * ffn + 12).max(embd + 12).max(ffn + 12) as usize;
        Self {
            size,
            input: 3,
            gate: size as u32 + 5,
            up: size as u32 * 2 + 7,
            down: size as u32 * 3 + 9,
            scratch: MlpScratch {
                gate_slot: 4,
                gate_base: size as u32 * 4,
                up_slot: 5,
                up_base: size as u32 * 5,
                hidden_slot: 6,
                hidden_base: size as u32 * 6,
            },
        }
    }
    fn compile(self, embd: u32, ffn: u32, scratch: bool, bounds: MlpBounds) -> CheckedMlpFragment {
        if scratch {
            compile_mlp_fragment_with_scratch_checked(
                "mlp",
                self.input,
                7,
                self.gate,
                self.up,
                self.down,
                embd,
                ffn,
                bounds,
                self.scratch,
            )
            .unwrap()
        } else {
            compile_mlp_fragment_checked(
                "mlp", self.input, 7, self.gate, self.up, self.down, embd, ffn, bounds,
            )
            .unwrap()
        }
    }
    fn buffer(self, inputs: &[f64], gate: &[f64], up: &[f64], down: &[f64]) -> ResultBuffer {
        let mut buffer = ResultBuffer::new(8, self.size);
        for slot in 0..8 {
            buffer.slot_mut(slot).fill(-777.0);
        }
        for (base, values) in [
            (self.input, inputs),
            (self.gate, gate),
            (self.up, up),
            (self.down, down),
        ] {
            for (index, &value) in values.iter().enumerate() {
                buffer.write_indexed(base, index as u32, value);
            }
        }
        buffer
    }
}

fn oracle(inputs: &[f64], gate: &[f64], up: &[f64], down: &[f64], ffn: usize) -> Vec<f64> {
    let embd = inputs.len();
    let hidden: Vec<_> = (0..ffn)
        .map(|row| {
            let g: f64 = inputs
                .iter()
                .enumerate()
                .map(|(col, x)| x * gate[row * embd + col])
                .sum();
            let u: f64 = inputs
                .iter()
                .enumerate()
                .map(|(col, x)| x * up[row * embd + col])
                .sum();
            g / (1.0 + (-g).exp()) * u
        })
        .collect();
    (0..embd)
        .map(|row| {
            hidden
                .iter()
                .enumerate()
                .map(|(col, h)| h * down[row * ffn + col])
                .sum()
        })
        .collect()
}

// Build the full, unoptimized data DAG independently of compiler Stage and its
// constant folder. Every original BML executes through the real-domain checker.
fn unfolded_bml(
    inputs: &[f64],
    gate: &[f64],
    up: &[f64],
    down: &[f64],
    ffn: usize,
    bounds: MlpBounds,
) -> Vec<f64> {
    let embd = inputs.len();
    let mut builder = BMLTransformer::new();
    let values: Vec<_> = inputs
        .iter()
        .chain(gate)
        .chain(up)
        .chain(down)
        .copied()
        .collect();
    let variables: Vec<_> = (0..values.len())
        .map(|index| builder.var(index as u32))
        .collect();
    let zero = builder.constant(0.0);
    let gate_config = BoundedOpConfig::dot_step(bounds.input, bounds.gate_weight).unwrap();
    let up_config = BoundedOpConfig::dot_step(bounds.input, bounds.up_weight).unwrap();
    let swiglu_config = BoundedOpConfig::swiglu(bounds.gate, bounds.up).unwrap();
    let down_config = BoundedOpConfig::dot_step(bounds.hidden, bounds.down_weight).unwrap();
    let mut hidden = Vec::new();
    for row in 0..ffn {
        let mut projected = [zero; 2];
        for (matrix, config) in [&gate_config, &up_config].into_iter().enumerate() {
            for column in 0..embd {
                let weight = embd + matrix * embd * ffn + row * embd + column;
                projected[matrix] = builder
                    .operation_bounded(
                        &[projected[matrix], variables[column], variables[weight]],
                        &[],
                        config,
                    )
                    .unwrap()[0];
            }
        }
        hidden.push(
            builder
                .operation_bounded(&projected, &[], &swiglu_config)
                .unwrap()[0],
        );
    }
    let roots: Vec<_> = (0..embd)
        .map(|row| {
            let mut total = zero;
            for (column, &activation) in hidden.iter().enumerate() {
                let weight = embd + 2 * embd * ffn + row * ffn + column;
                total = builder
                    .operation_bounded(&[total, activation, variables[weight]], &[], &down_config)
                    .unwrap()[0];
            }
            total
        })
        .collect();
    let dag = builder.into_soa();
    let mut evaluated = Vec::with_capacity(dag.len());
    for (index, kind) in dag.kinds.iter().enumerate() {
        evaluated.push(match *kind {
            NodeKind::One => 1.0,
            NodeKind::Zero => 0.0,
            NodeKind::Var(id) => values[id as usize],
            NodeKind::Const(id) => dag.const_pool[id as usize],
            NodeKind::Bml => Basis::Bml
                .apply_real(
                    evaluated[dag.lefts[index] as usize],
                    evaluated[dag.rights[index] as usize],
                )
                .unwrap(),
        });
    }
    roots.iter().map(|&root| evaluated[root as usize]).collect()
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() < 2e-7,
            "element {index}: {actual} != {expected}"
        );
    }
}

fn run(
    operation: &OperationFragment,
    layout: Layout,
    inputs: &[f64],
    gate: &[f64],
    up: &[f64],
    down: &[f64],
    scratch: bool,
) -> Vec<f64> {
    let mut buffer = layout.buffer(inputs, gate, up, down);
    let before = buffer.data().to_vec();
    let mut hot = HotLoop::with_capacity(operation.fragment.ops.len() + 4);
    hot.execute_fragment_with_ctx(
        &Fragment {
            ops: vec![RpnOp::Const(0), RpnOp::Const(1), RpnOp::Const(2)],
        },
        &EvalContext::new(&[], &[7.0, -2.0, 99.0]),
    );
    let parent = EvalContext::new(&[], &[f64::NAN]);
    for _ in 0..2 {
        hot.execute_fragment_full(
            &operation.fragment,
            &operation.context(&parent),
            &mut buffer,
        );
        assert_eq!(
            hot.stack_depth(),
            3,
            "MLP changed its caller's stack prefix"
        );
    }
    assert_eq!(
        [hot.stack_pop(), hot.stack_pop(), hot.stack_pop()],
        [Some(99.0), Some(-2.0), Some(7.0)]
    );
    let embd = operation.meta.n_rows as usize;
    let ffn = operation.meta.n_cols as usize;
    for (index, (&old, &new)) in before.iter().zip(buffer.data()).enumerate() {
        let output = (7 * layout.size..7 * layout.size + embd).contains(&index);
        let temporary = scratch
            && embd != 0
            && (4..7).any(|slot| (slot * layout.size..slot * layout.size + ffn).contains(&index));
        if !output && !temporary {
            assert_eq!(old.to_bits(), new.to_bits(), "unexpected write at {index}");
        }
    }
    buffer.slot(7)[..embd].to_vec()
}

#[test]
fn mlp_preflight_rejects_each_oversized_matrix_independently() {
    let layout = Layout::new(1, 1);
    let inputs = [0.5];
    let matrices = [vec![0.25], vec![0.25], vec![0.5]];
    for scratch in [false, true] {
        let checked = layout.compile(1, 1, scratch, MlpBounds::default());
        checked
            .validate_values(&inputs, &matrices[0], &matrices[1], &matrices[2])
            .unwrap();
        let actual = run(
            checked.operation(),
            layout,
            &inputs,
            &matrices[0],
            &matrices[1],
            &matrices[2],
            scratch,
        );
        close(
            &actual,
            &oracle(&inputs, &matrices[0], &matrices[1], &matrices[2], 1),
        );
        for matrix in 0..3 {
            let mut changed = matrices.clone();
            // An unused extra element must be rejected by shape preflight,
            // rather than silently ignored or reached by an indexing panic.
            changed[matrix].push(f64::NAN);
            let error = checked
                .validate_values(&inputs, &changed[0], &changed[1], &changed[2])
                .unwrap_err();
            assert_eq!(
                error, "MLP operand shape mismatch",
                "matrix={matrix}, scratch={scratch}"
            );
        }
    }
}

#[test]
fn empty_ffn_still_checks_input_finiteness_and_declared_bound() {
    let bounds = MlpBounds::default();
    let layout = Layout::new(1, 0);
    for scratch in [false, true] {
        let checked = layout.compile(1, 0, scratch, bounds);
        checked
            .validate_values(&[bounds.input], &[], &[], &[])
            .unwrap();
        close(
            &run(
                checked.operation(),
                layout,
                &[bounds.input],
                &[],
                &[],
                &[],
                scratch,
            ),
            &[0.0],
        );
        // With no FFN rows, no later dot or SwiGLU stage can incidentally
        // enforce the input contract. It remains an entrypoint obligation.
        for input in [
            bounds.input + 1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let error = checked
                .validate_values(&[input], &[], &[], &[])
                .unwrap_err();
            assert_eq!(
                error, "MLP input bound at element 0",
                "input={input}, scratch={scratch}"
            );
        }
    }
}

#[test]
fn scratch_mlp_accepts_adjacent_operand_and_temporary_ranges() {
    // Eight one-element slots alternate operand and writable storage. Every
    // range is disjoint, including operands starting exactly at a write's end.
    let layout = Layout {
        size: 1,
        input: 2,
        gate: 4,
        up: 6,
        down: 0,
        scratch: MlpScratch {
            gate_slot: 1,
            gate_base: 1,
            up_slot: 3,
            up_base: 3,
            hidden_slot: 5,
            hidden_base: 5,
        },
    };
    let inputs = [1.0];
    let gate = [0.5];
    let up = [-0.25];
    let down = [0.5];
    let checked = layout.compile(1, 1, true, MlpBounds::default());
    checked.validate_values(&inputs, &gate, &up, &down).unwrap();
    let mut buffer = layout.buffer(&inputs, &gate, &up, &down);
    let before = buffer.data().to_vec();
    Runtime::new(4096, 0).execute_fragments_sequential(
        checked.operations(),
        &EvalContext::new(&[], &[]),
        &mut buffer,
    );
    close(buffer.slot(7), &oracle(&inputs, &gate, &up, &down, 1));
    for slot in [0, 2, 4, 6] {
        assert_eq!(
            buffer.slot(slot)[0].to_bits(),
            before[slot].to_bits(),
            "operand slot={slot}"
        );
    }
}

#[test]
fn public_mlp_uses_rectangular_gate_up_and_down_and_preserves_unrelated_state() {
    let cases = [
        (vec![2.0], vec![0.75], vec![1.5], vec![-0.5], 1),
        (
            vec![1.5, -0.5],
            vec![1.0, 0.25, -0.5, 1.5, 0.75, -1.0],
            vec![-0.25, 1.0, 0.5, -0.25, -1.0, 0.5],
            vec![0.25, -0.5, 0.75, -0.75, 0.125, 0.5],
            3,
        ),
        (
            vec![-1.0, 0.5, 2.0],
            vec![0.5, -0.25, 0.75, -0.5, 1.0, -0.25],
            vec![1.0, 0.5, -0.25, -1.0, 0.0, 0.5],
            vec![0.25, -0.75, 0.0, 0.5, -1.0, 0.125],
            2,
        ),
        (
            vec![0.0, -0.0],
            vec![1.0, -2.0, 0.5, 0.25],
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![1.0, -0.5, 0.75, 0.5],
            2,
        ),
        (vec![1.0, -2.0], vec![], vec![], vec![], 0),
        (vec![], vec![], vec![], vec![], 3),
        (vec![], vec![], vec![], vec![], 0),
    ];
    for (inputs, gate, up, down, ffn) in cases {
        let embd = inputs.len() as u32;
        let layout = Layout::new(embd, ffn);
        let mut prior = None;
        for scratch in [false, true] {
            let checked = layout.compile(embd, ffn, scratch, MlpBounds::default());
            assert_eq!(checked.dimensions(), (embd, ffn));
            assert_eq!(checked.scratch().is_some(), scratch);
            checked.validate_values(&inputs, &gate, &up, &down).unwrap();
            let actual = run(
                checked.operation(),
                layout,
                &inputs,
                &gate,
                &up,
                &down,
                scratch,
            );
            close(&actual, &oracle(&inputs, &gate, &up, &down, ffn as usize));
            let bits: Vec<_> = actual.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                bits,
                unfolded_bml(
                    &inputs,
                    &gate,
                    &up,
                    &down,
                    ffn as usize,
                    MlpBounds::default()
                )
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
                "closed configuration folding changed a result bit"
            );
            if let Some(old) = &prior {
                assert_eq!(&bits, old, "scratch changes numerical order");
            }
            prior = Some(bits);
        }
    }
}

#[test]
fn mlp_fold_preserves_unfolded_bits_for_custom_configuration() {
    let inputs = [1.5, -0.5];
    let gate = [0.75, -0.25, -0.5, 0.5, 0.25, 1.0];
    let up = [-0.5, 0.25, 0.75, -1.0, 0.5, 0.25];
    let down = [0.25, -0.5, 0.75, -0.5, 0.25, 0.125];
    let bounds = MlpBounds {
        input: 8.0,
        gate_weight: 1.0,
        up_weight: 1.5,
        gate: 8.0,
        up: 8.0,
        hidden: 64.0,
        down_weight: 2.0,
    };
    let expected = unfolded_bml(&inputs, &gate, &up, &down, 3, bounds);
    close(&expected, &oracle(&inputs, &gate, &up, &down, 3));
    for scratch in [false, true] {
        let layout = Layout::new(2, 3);
        let checked = layout.compile(2, 3, scratch, bounds);
        // Public numerical declarations must describe values this very
        // operation admits, so a caller can construct inputs from the getter.
        let declared = checked.bounds();
        checked
            .validate_values(&[declared.input; 2], &[0.0; 6], &[0.0; 6], &[0.0; 6])
            .unwrap();
        assert!(checked
            .validate_values(&[declared.input + 1.0; 2], &[0.0; 6], &[0.0; 6], &[0.0; 6])
            .is_err());
        checked.validate_values(&inputs, &gate, &up, &down).unwrap();
        let actual = run(
            checked.operation(),
            layout,
            &inputs,
            &gate,
            &up,
            &down,
            scratch,
        );
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn every_matrix_element_influences_the_complete_mlp_and_legacy_is_square() {
    let inputs = [1.0, 0.5];
    let gate = vec![0.25; 6];
    let up = vec![0.5; 6];
    let down = vec![0.25; 6];
    let layout = Layout::new(2, 3);
    let checked = layout.compile(2, 3, true, MlpBounds::default());
    let baseline = run(
        checked.operation(),
        layout,
        &inputs,
        &gate,
        &up,
        &down,
        true,
    );
    for matrix in 0..3 {
        for index in 0..6 {
            let mut matrices = [gate.clone(), up.clone(), down.clone()];
            matrices[matrix][index] += 0.125;
            checked
                .validate_values(&inputs, &matrices[0], &matrices[1], &matrices[2])
                .unwrap();
            let actual = run(
                checked.operation(),
                layout,
                &inputs,
                &matrices[0],
                &matrices[1],
                &matrices[2],
                true,
            );
            close(
                &actual,
                &oracle(&inputs, &matrices[0], &matrices[1], &matrices[2], 3),
            );
            assert!(
                actual
                    .iter()
                    .zip(&baseline)
                    .any(|(a, b)| (a - b).abs() > 1e-4),
                "matrix {matrix}, index {index} ignored"
            );
        }
    }
    let layout = Layout::new(2, 2);
    let operation = compile_mlp_fragment(
        "legacy",
        layout.input,
        7,
        layout.gate,
        layout.up,
        layout.down,
        2,
    );
    assert_eq!(operation.meta.n_cols, 2);
    close(
        &run(
            &operation,
            layout,
            &inputs,
            &gate[..4],
            &up[..4],
            &down[..4],
            false,
        ),
        &oracle(&inputs, &gate[..4], &up[..4], &down[..4], 2),
    );
}

#[test]
fn mlp_preflight_rejects_shapes_bounds_and_each_invalid_stage() {
    let layout = Layout::new(1, 1);
    let checked = layout.compile(1, 1, false, MlpBounds::default());
    assert!(checked
        .validate_values(&[], &[1.0], &[1.0], &[1.0])
        .unwrap_err()
        .contains("shape"));
    assert!(checked
        .validate_values(&[17.0], &[1.0], &[1.0], &[1.0])
        .unwrap_err()
        .contains("input"));
    for matrix in 0..3 {
        let mut matrices = [[1.0]; 3];
        matrices[matrix][0] = f64::NAN;
        assert!(checked
            .validate_values(&[1.0], &matrices[0], &matrices[1], &matrices[2])
            .is_err());
    }
    for (bounds, expected) in [
        (
            MlpBounds {
                gate: 0.5,
                ..MlpBounds::default()
            },
            "SwiGLU",
        ),
        (
            MlpBounds {
                up: 0.5,
                ..MlpBounds::default()
            },
            "SwiGLU",
        ),
        (
            MlpBounds {
                hidden: 0.25,
                ..MlpBounds::default()
            },
            "down",
        ),
        (
            MlpBounds {
                down_weight: 0.5,
                ..MlpBounds::default()
            },
            "down",
        ),
    ] {
        let compiled = layout.compile(1, 1, false, bounds);
        assert!(compiled
            .validate_values(&[1.0], &[1.0], &[1.0], &[1.0])
            .unwrap_err()
            .contains(expected));
    }
    let layout = Layout::new(1, 6);
    let checked = layout.compile(1, 6, false, MlpBounds::default());
    assert!(checked
        .validate_values(&[4.0], &[2.0; 6], &[2.0; 6], &[1.0; 6])
        .unwrap_err()
        .contains("down"));
}

fn graph(operation: &OperationFragment) -> BmlGraph {
    BmlGraph {
        fragments: vec![
            operation.fragment.clone(),
            Fragment {
                ops: vec![RpnOp::One],
            },
        ],
        const_pool: vec![],
        threshold: 32768,
    }
}

#[test]
fn scratch_mlp_executes_checked_graph_and_runtime_binds_its_owned_constants() {
    let layout = Layout::new(2, 3);
    let checked = layout.compile(2, 3, true, MlpBounds::default());
    let inputs = [1.0, -0.5];
    let gate = [0.25; 6];
    let up = [0.5; 6];
    let down = [-0.25; 6];
    checked.validate_values(&inputs, &gate, &up, &down).unwrap();
    let validated = ValidatedGraph::new_with_external_constants(
        graph(checked.operation()),
        0,
        checked.constants().len(),
        8,
        layout.size,
        GraphLimits::default(),
    )
    .unwrap();
    let mut buffer = layout.buffer(&inputs, &gate, &up, &down);
    assert_eq!(
        validated
            .execute_operation_with_constants(&[], checked.constants(), &mut buffer)
            .unwrap(),
        1.0
    );
    close(&buffer.slot(7)[..2], &oracle(&inputs, &gate, &up, &down, 3));
    let steps = validated.execution_steps();
    let error = ValidatedGraph::new_with_external_constants(
        graph(checked.operation()),
        0,
        checked.constants().len(),
        8,
        layout.size,
        GraphLimits {
            max_steps: steps - 1,
            ..GraphLimits::default()
        },
    );
    assert!(matches!(error, Err(GraphError::Limit)));
    let mut runtime_buffer = layout.buffer(&inputs, &gate, &up, &down);
    Runtime::new(4096, 0).execute_fragments_sequential(
        &[checked.operation().clone()],
        &EvalContext::new(&[], &[f64::NAN]),
        &mut runtime_buffer,
    );
    assert_eq!(runtime_buffer.data(), buffer.data());

    let blocks = checked.operations();
    assert_eq!(blocks.len(), 4);
    let mut plan_buffer = layout.buffer(&inputs, &gate, &up, &down);
    let mut prefixed_buffer = layout.buffer(&inputs, &gate, &up, &down);
    let mut hot = HotLoop::with_capacity(4096);
    hot.execute_fragment_with_ctx(
        &Fragment {
            ops: vec![RpnOp::Const(0), RpnOp::Const(1)],
        },
        &EvalContext::new(&[], &[42.0, -7.0]),
    );
    let expected_metadata = [
        ("mlp.gate", layout.input, 4, layout.gate, 3, 2),
        ("mlp.up", layout.input, 5, layout.up, 3, 2),
        (
            "mlp.swiglu",
            layout.scratch.gate_base,
            6,
            layout.scratch.up_base,
            3,
            1,
        ),
        ("mlp.down", layout.scratch.hidden_base, 7, layout.down, 2, 3),
    ];
    for (block, (name, input, output, weights, rows, columns)) in
        blocks.iter().zip(expected_metadata)
    {
        assert_eq!(block.const_pool.as_deref(), Some(checked.constants()));
        assert_eq!(block.meta.name, name);
        assert_eq!(
            (
                block.meta.input_slot,
                block.meta.output_slot,
                block.meta.weight_base,
                block.meta.n_rows,
                block.meta.n_cols
            ),
            (input, output, weights, rows, columns)
        );
        let RpnOp::Loop { body_len, .. } = block.fragment.ops[0] else {
            panic!("MLP plan block is not a complete loop");
        };
        assert_eq!(body_len as usize + 1, block.fragment.ops.len());
        assert!(block.fragment.ops.len() * std::mem::size_of::<RpnOp>() <= 32768);
        let validated = ValidatedGraph::new_with_external_constants(
            graph(block),
            0,
            checked.constants().len(),
            8,
            layout.size,
            GraphLimits::default(),
        )
        .unwrap();
        assert_eq!(
            validated
                .execute_operation_with_constants(&[], checked.constants(), &mut plan_buffer)
                .unwrap(),
            1.0
        );
        hot.execute_fragment_full(
            &block.fragment,
            &block.context(&EvalContext::new(&[], &[f64::NAN])),
            &mut prefixed_buffer,
        );
        assert_eq!(hot.stack_depth(), 2, "block changed caller stack prefix");
    }
    assert_eq!([hot.stack_pop(), hot.stack_pop()], [Some(-7.0), Some(42.0)]);
    assert_eq!(plan_buffer.data(), buffer.data());
    assert_eq!(prefixed_buffer.data(), buffer.data());
    for cores in [1, 2, 4] {
        let mut scheduled = layout.buffer(&inputs, &gate, &up, &down);
        Runtime::new(4096, 0).execute_with_cores(
            blocks,
            &EvalContext::new(&[], &[f64::NAN]),
            &mut scheduled,
            cores,
        );
        assert_eq!(
            scheduled.data(),
            buffer.data(),
            "plan changed with {cores} cores"
        );
    }
    assert_eq!(checked.clone().into_operations().len(), 4);
    let recomputed = layout.compile(2, 3, false, MlpBounds::default());
    assert!(matches!(
        ValidatedGraph::new_with_external_constants(
            graph(recomputed.operation()),
            0,
            recomputed.constants().len(),
            8,
            layout.size,
            GraphLimits::default()
        ),
        Err(GraphError::Loop)
    ));
}

#[test]
fn mlp_dimensions_addresses_and_explicit_scratch_layout_are_checked() {
    let build = |input, gate, up, down, embd, ffn, bounds| {
        compile_mlp_fragment_checked("bad", input, 7, gate, up, down, embd, ffn, bounds)
    };
    let bounds = MlpBounds::default();
    assert!(build(0, 0, 0, 0, u32::MAX, 2, bounds)
        .unwrap_err()
        .contains("dimensions"));
    for bases in [
        (u32::MAX, 0, 0, 0),
        (0, u32::MAX, 0, 0),
        (0, 0, u32::MAX, 0),
        (0, 0, 0, u32::MAX),
    ] {
        assert!(build(bases.0, bases.1, bases.2, bases.3, 2, 2, bounds)
            .unwrap_err()
            .contains("address"));
    }
    for invalid in [
        MlpBounds {
            input: f64::NAN,
            ..bounds
        },
        MlpBounds {
            gate: 18.0001,
            ..bounds
        },
        MlpBounds {
            input: 256.0,
            ..bounds
        },
        MlpBounds {
            down_weight: 2.0,
            ..bounds
        },
    ] {
        assert!(build(0, 10, 20, 30, 2, 2, invalid).is_err());
    }
    let layout = Layout::new(2, 3);
    let scratch = layout.scratch;
    let bad = [
        MlpScratch {
            up_slot: 4,
            ..scratch
        },
        MlpScratch {
            gate_base: scratch.gate_base + 1,
            ..scratch
        },
        MlpScratch {
            hidden_base: scratch.hidden_base + 6,
            ..scratch
        },
        MlpScratch {
            hidden_slot: 7,
            hidden_base: layout.size as u32 * 7,
            ..scratch
        },
    ];
    for bad in bad {
        assert!(compile_mlp_fragment_with_scratch_checked(
            "bad",
            layout.input,
            7,
            layout.gate,
            layout.up,
            layout.down,
            2,
            3,
            bounds,
            bad
        )
        .is_err());
    }
    assert!(compile_mlp_fragment_with_scratch_checked(
        "overlap",
        scratch.gate_base,
        7,
        layout.gate,
        layout.up,
        layout.down,
        2,
        3,
        bounds,
        scratch,
    )
    .unwrap_err()
    .contains("overlaps"));
    let too_small = MlpScratch {
        gate_slot: 4,
        gate_base: 4,
        up_slot: 5,
        up_base: 5,
        hidden_slot: 6,
        hidden_base: 6,
    };
    assert!(compile_mlp_fragment_with_scratch_checked(
        "bad", 0, 7, 10, 20, 30, 2, 3, bounds, too_small
    )
    .unwrap_err()
    .contains("capacity"));
}

// Independent taint analysis: only loop counters/strides may enter native
// arithmetic or addresses; all loaded data and activation arithmetic use BML.
fn address_only(ops: &[RpnOp], stack: &mut Vec<bool>) {
    let mut pc = 0;
    while pc < ops.len() {
        match ops[pc] {
            RpnOp::Zero | RpnOp::One => stack.push(false),
            RpnOp::Const(id) => stack.push(id < 2),
            RpnOp::Pick { depth } => stack.push(stack[stack.len() - 1 - depth as usize]),
            RpnOp::Dup => stack.push(*stack.last().unwrap()),
            RpnOp::Drop => {
                stack.pop().unwrap();
            }
            RpnOp::Swap => {
                let n = stack.len();
                stack.swap(n - 1, n - 2);
            }
            RpnOp::FAdd | RpnOp::FMul => {
                assert!(stack.pop().unwrap());
                assert!(stack.pop().unwrap());
                stack.push(true);
            }
            RpnOp::Bml => {
                assert!(!stack.pop().unwrap());
                assert!(!stack.pop().unwrap());
                stack.push(false);
            }
            RpnOp::VarIndexed { .. } => {
                assert!(stack.pop().unwrap());
                stack.push(false);
            }
            RpnOp::StoreResult { .. } => {
                assert!(stack.pop().unwrap());
                assert!(!stack.pop().unwrap());
            }
            RpnOp::Loop { body_len, .. } => {
                let before = stack.clone();
                stack.push(true);
                let end = pc + 1 + body_len as usize;
                address_only(&ops[pc + 1..end], stack);
                assert_eq!(*stack, before);
                pc = end;
                continue;
            }
            other => panic!("unexpected MLP opcode {other:?}"),
        }
        pc += 1;
    }
}

#[test]
fn mlp_storage_is_constant_and_native_arithmetic_is_only_addressing() {
    for scratch in [false, true] {
        let mut counts = Vec::new();
        let mut constants = Vec::new();
        for (embd, ffn) in [(1, 1), (2, 3), (32, 64), (256, 512)] {
            let layout = Layout::new(embd, ffn);
            let checked = layout.compile(embd, ffn, scratch, MlpBounds::default());
            constants.push(checked.constants().len());
            if scratch {
                assert_eq!(checked.operations().len(), 4);
                for block in checked.operations() {
                    assert!(block.fragment.ops.len() * std::mem::size_of::<RpnOp>() <= 32768);
                }
            } else {
                assert_eq!(checked.operations().len(), 1);
                assert!(
                    checked.operation().fragment.ops.len() * std::mem::size_of::<RpnOp>() <= 32768
                );
            }
            let mut stack = vec![false, false, false];
            address_only(&checked.operation().fragment.ops, &mut stack);
            assert_eq!(stack, vec![false, false, false]);
            counts.push(checked.operation().fragment.ops.len());
        }
        assert!(counts.iter().all(|count| *count == counts[0]));
        assert!(constants.iter().all(|count| *count == constants[0]));
        println!(
            "scratch={scratch}: {} ops, {} bytes",
            counts[0],
            counts[0] * std::mem::size_of::<RpnOp>()
        );
    }
}
