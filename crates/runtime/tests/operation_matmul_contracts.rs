use bml_compiler::op_fragments::{
    compile_matmul_fragment, compile_matmul_fragment_checked, MatmulBounds, OperationFragment,
};
use bml_compiler::{Fragment, RpnOp};
use bml_domain::EvalContext;
use bml_runtime::{buffer::ResultBuffer, hot_loop::HotLoop};

fn oracle(weights: &[f64], inputs: &[f64], rows: usize) -> Vec<f64> {
    (0..rows)
        .map(|row| {
            inputs
                .iter()
                .enumerate()
                .map(|(column, input)| weights[row * inputs.len() + column] * input)
                .sum()
        })
        .collect()
}

fn run(
    operation: &OperationFragment,
    constants: &[f64],
    inputs: &[f64],
    weights: &[f64],
) -> Vec<f64> {
    let rows = operation.meta.n_rows as usize;
    let slot_size = (weights.len() + 8).max(rows + 8).max(inputs.len() + 8);
    let mut buffer = ResultBuffer::new(4, slot_size);
    for slot in 0..4 {
        buffer.slot_mut(slot).fill(-999.0);
    }
    for (index, &value) in inputs.iter().enumerate() {
        buffer.write_indexed(operation.meta.input_slot, index as u32, value);
    }
    for (index, &value) in weights.iter().enumerate() {
        buffer.write_indexed(operation.meta.weight_base, index as u32, value);
    }
    let before = buffer.data().to_vec();
    let mut hot_loop = HotLoop::with_capacity(operation.fragment.ops.len() + 16);
    // Fragments share a stack; the matmul must preserve an unrelated prefix.
    hot_loop.execute_fragment(&Fragment {
        ops: vec![RpnOp::One],
    });
    let context = EvalContext::new(&[], constants);
    for _ in 0..3 {
        hot_loop.execute_fragment_full(&operation.fragment, &context, &mut buffer);
        assert_eq!(hot_loop.stack_depth(), 1, "loop frame leaked");
    }
    assert_eq!(hot_loop.stack_pop(), Some(1.0));
    let output_start = operation.meta.output_slot as usize * slot_size;
    for (index, (&actual, &original)) in buffer.data().iter().zip(&before).enumerate() {
        if !(output_start..output_start + rows).contains(&index) {
            assert_eq!(actual, original, "unexpected write at {index}");
        }
    }
    let output = buffer.slot(operation.meta.output_slot as usize)[..rows].to_vec();
    for (row, (&actual, expected)) in output.iter().zip(oracle(weights, inputs, rows)).enumerate() {
        assert!(actual.is_finite(), "nonfinite row {row}: {actual}");
        assert!(
            (actual - expected).abs() < 2e-7,
            "row {row}: {actual} != {expected}"
        );
    }
    output
}

fn layout(rows: usize, inputs: usize, weights: usize) -> (u32, u32, u32) {
    let slot_size = (weights + 8).max(rows + 8).max(inputs + 8);
    (3, (slot_size + 5) as u32, 3)
}

#[test]
fn legacy_public_matmul_computes_signed_rectangular_rows_and_preserves_stack() {
    let cases: &[(&[f64], &[f64], usize)] = &[
        (&[1.0], &[2.0], 1),
        (&[3.0, 4.0], &[1.0, 2.0, 3.0, 2.0], 2),
        (
            &[-1.0, 2.0, -3.0, 0.5],
            &[
                0.0, -1.0, 2.0, 0.5, 3.0, 0.0, -0.25, -2.0, -2.0, 4.0, 1.5, 0.0,
            ],
            3,
        ),
        (&[0.0, -0.0, 0.0], &[-2.0, 0.0, 3.0, 1.0, -1.0, 0.25], 2),
        (&[], &[], 3),
        (&[1.0, -2.0], &[], 0),
    ];
    for &(inputs, weights, rows) in cases {
        let (input_base, weight_base, output_slot) = layout(rows, inputs.len(), weights.len());
        let operation = compile_matmul_fragment(
            "legacy",
            input_base,
            output_slot,
            weight_base,
            rows as u32,
            inputs.len() as u32,
            2,
        );
        run(
            &operation,
            &[999.0, -123.0, inputs.len() as f64],
            inputs,
            weights,
        );
        assert_eq!(
            operation
                .fragment
                .ops
                .iter()
                .filter(|op| matches!(op, RpnOp::FAdd))
                .count(),
            1
        );
        assert_eq!(
            operation
                .fragment
                .ops
                .iter()
                .filter(|op| matches!(op, RpnOp::FMul))
                .count(),
            1
        );
        assert!(operation
            .fragment
            .ops
            .iter()
            .all(|op| !matches!(op, RpnOp::Const(id) if *id != 2)));
    }
}

#[test]
fn configurable_matmul_owns_constants_and_checks_actual_finite_real_recurrence() {
    let cases: &[(MatmulBounds, &[f64], &[f64], usize)] = &[
        (
            MatmulBounds {
                input: 128.0,
                weight: 2.0,
            },
            &[64.0, -32.0, 0.5],
            &[0.25, -0.5, 1.0, -0.5, -1.0, 2.0],
            2,
        ),
        (
            MatmulBounds {
                input: 2.5,
                weight: 1.25,
            },
            &[-2.5, 1.25, 0.0],
            &[1.25, -0.75, 0.0, -0.5, -1.0, 1.0],
            2,
        ),
        (MatmulBounds::default(), &[], &[], 3),
    ];
    for &(bounds, inputs, weights, rows) in cases {
        let (input_base, weight_base, output_slot) = layout(rows, inputs.len(), weights.len());
        let compiled = compile_matmul_fragment_checked(
            "checked",
            input_base,
            output_slot,
            weight_base,
            rows as u32,
            inputs.len() as u32,
            bounds,
        )
        .unwrap();
        assert_eq!(compiled.bounds(), bounds);
        compiled.validate_values(inputs, weights).unwrap();
        run(compiled.operation(), compiled.constants(), inputs, weights);
    }
}

#[test]
fn matmul_input_address_budget_includes_the_last_u32_element() {
    for columns in 1..=4 {
        let last_valid_base = u32::MAX - (columns - 1);
        let compiled = compile_matmul_fragment_checked(
            "last address",
            last_valid_base,
            3,
            0,
            1,
            columns,
            MatmulBounds::default(),
        )
        .expect("the final u32 address is an inclusive endpoint");
        compiled
            .validate_values(&vec![1.0; columns as usize], &vec![1.0; columns as usize])
            .unwrap();
        // This tests declaration arithmetic without allocating a buffer with
        // billions of elements or attempting a read at those addresses.
        if columns > 1 {
            let error = compile_matmul_fragment_checked(
                "one past the address space",
                last_valid_base + 1,
                3,
                0,
                1,
                columns,
                MatmulBounds::default(),
            )
            .unwrap_err();
            assert!(error.contains("matmul input address overflow"));
        }
    }
}

#[test]
fn configurable_matmul_rejects_invalid_shapes_addresses_bounds_and_numerical_range() {
    let compile = |rows, cols, input_base, weight_base, bounds| {
        compile_matmul_fragment_checked("checked", input_base, 3, weight_base, rows, cols, bounds)
    };
    for bounds in [
        MatmulBounds {
            input: f64::NAN,
            weight: 1.0,
        },
        MatmulBounds {
            input: 0.0,
            weight: 1.0,
        },
        MatmulBounds {
            input: 257.0,
            weight: 1.0,
        },
        MatmulBounds {
            input: 32.0,
            weight: 16.0,
        },
    ] {
        assert!(compile(1, 1, 0, 0, bounds).is_err());
    }
    let bounds = MatmulBounds::default();
    assert!(compile(u32::MAX, 2, 0, 0, bounds)
        .unwrap_err()
        .contains("dimensions"));
    assert!(compile(1, 2, u32::MAX, 0, bounds)
        .unwrap_err()
        .contains("input address"));
    assert!(compile(2, 2, 0, u32::MAX - 2, bounds)
        .unwrap_err()
        .contains("weight address"));
    let compiled = compile(1, 2, 0, 8, bounds).unwrap();
    assert!(compiled
        .validate_values(&[1.0], &[1.0, 1.0])
        .unwrap_err()
        .contains("shape"));
    assert!(compiled
        .validate_values(&[f64::NAN, 0.0], &[1.0, 1.0])
        .unwrap_err()
        .contains("column 0"));
    assert!(compiled
        .validate_values(&[17.0, 0.0], &[1.0, 1.0])
        .unwrap_err()
        .contains("input bound"));
    assert!(compiled
        .validate_values(&[1.0, 0.0], &[f64::INFINITY, 1.0])
        .unwrap_err()
        .contains("row 0, column 0"));
    // All declared operands are allowed, but the first product reaches 256.
    // The next recurrence is outside the explicit accumulator domain.
    assert!(compiled
        .validate_values(&[16.0, 1.0], &[16.0, 1.0])
        .unwrap_err()
        .contains("row 0, column 1"));
    // A finite declaration is not a promise that every BML intermediate fits.
    let negative = compile(
        1,
        5,
        0,
        8,
        MatmulBounds {
            input: 128.0,
            weight: 2.0,
        },
    )
    .unwrap();
    assert!(negative
        .validate_values(&[-128.0; 5], &[2.0; 5])
        .unwrap_err()
        .contains("BML node"));
}

#[test]
fn public_matmul_matches_independent_oracle_across_rectangular_dimensions() {
    for (rows, columns) in [(1, 7), (7, 1), (2, 3), (3, 2), (5, 17), (4, 64)] {
        let inputs: Vec<_> = (0..columns)
            .map(|column| ((column * 7 % 13) as f64 - 6.0) / 4.0)
            .collect();
        let weights: Vec<_> = (0..rows * columns)
            .map(|index| ((index * 11 % 17) as f64 - 8.0) / 8.0)
            .collect();
        let (input_base, weight_base, output_slot) = layout(rows, columns, weights.len());
        let compiled = compile_matmul_fragment_checked(
            "rectangular",
            input_base,
            output_slot,
            weight_base,
            rows as u32,
            columns as u32,
            MatmulBounds {
                input: 2.0,
                weight: 1.0,
            },
        )
        .unwrap();
        compiled.validate_values(&inputs, &weights).unwrap();
        run(
            compiled.operation(),
            compiled.constants(),
            &inputs,
            &weights,
        );
        let legacy = compile_matmul_fragment(
            "rectangular legacy",
            input_base,
            output_slot,
            weight_base,
            rows as u32,
            columns as u32,
            0,
        );
        run(&legacy, &[columns as f64], &inputs, &weights);
    }
}

#[test]
fn matmul_bytecode_size_is_independent_of_dimensions() {
    let small = compile_matmul_fragment("small", 0, 3, 10, 1, 1, 0);
    let large = compile_matmul_fragment("large", 0, 3, 10, 2048, 2048, 0);
    assert_eq!(small.fragment.ops.len(), large.fragment.ops.len());
    assert!(large.fragment.ops.len() * std::mem::size_of::<RpnOp>() < 32 * 1024);
    let checked = compile_matmul_fragment_checked(
        "wide",
        0,
        3,
        2048,
        2048,
        2048,
        MatmulBounds {
            input: 128.0,
            weight: 2.0,
        },
    )
    .unwrap();
    assert!(checked.operation().fragment.ops.len() <= large.fragment.ops.len());
}
