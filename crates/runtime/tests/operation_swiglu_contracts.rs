use bml_compiler::{bml_ops::bml_swiglu_program, Fragment, RpnOp};
use bml_domain::EvalContext;
use bml_runtime::{buffer::ResultBuffer, HotLoop};

const SLOT_SIZE: usize = 12;
const GATE_BASE: u32 = 13;
const UP_BASE: u32 = 27;
const OUTPUT_SLOT: u32 = 4;
const UNUSED_SCALE: u32 = 5;
const LOG2_E: u32 = 7;
const ONE: u32 = 9;

fn evaluate(gate: &[f64], up: &[f64], historical_scale: f64, log2_e: f64) -> Vec<f64> {
    assert_eq!(gate.len(), up.len());
    let mut buffer = ResultBuffer::new(5, SLOT_SIZE);
    for slot in 0..buffer.n_slots() {
        buffer.slot_mut(slot).fill(-987.0);
    }
    for (index, (&g, &u)) in gate.iter().zip(up).enumerate() {
        buffer.write_indexed(GATE_BASE, index as u32, g);
        buffer.write_indexed(UP_BASE, index as u32, u);
    }
    let before = buffer.data().to_vec();
    let mut constants = vec![f64::NAN; 10];
    constants[UNUSED_SCALE as usize] = historical_scale;
    constants[LOG2_E as usize] = log2_e;
    constants[ONE as usize] = 1.0;
    let program = bml_swiglu_program(
        GATE_BASE,
        UP_BASE,
        gate.len() as u32,
        UNUSED_SCALE,
        LOG2_E,
        ONE,
        OUTPUT_SLOT,
    );
    let mut hot = HotLoop::with_capacity(program.ops.len());
    hot.execute_fragment(&Fragment {
        ops: vec![RpnOp::One],
    });
    let fragment = Fragment { ops: program.ops };
    for _ in 0..3 {
        hot.execute_fragment_full(&fragment, &EvalContext::new(&[], &constants), &mut buffer);
        assert_eq!(hot.stack_depth(), 1, "vector program leaked stack values");
    }
    assert_eq!(
        hot.stack_pop(),
        Some(1.0),
        "changed the existing stack prefix"
    );
    assert_eq!(hot.stack_depth(), 0);
    for (index, (&old, &new)) in before.iter().zip(buffer.data()).enumerate() {
        if !(OUTPUT_SLOT as usize * SLOT_SIZE..OUTPUT_SLOT as usize * SLOT_SIZE + gate.len())
            .contains(&index)
        {
            assert_eq!(
                old.to_bits(),
                new.to_bits(),
                "changed unrelated cell {index}"
            );
        }
    }
    buffer.slot(OUTPUT_SLOT as usize)[..gate.len()].to_vec()
}

#[test]
fn swiglu_fragment_writes_standard_signed_values_and_preserves_other_cells() {
    let l = std::f64::consts::LN_2;
    let gate = [l, -l, 0.0, -0.0, 1.0, l, -l, l, -l];
    let up = [3.0, 3.0, 1.0, -2.0, 1.0, -3.0, -3.0, 0.0, -0.0];
    let expected = [
        2.0 * l,
        -l,
        0.0,
        0.0,
        1.0 / (1.0 + (-1.0_f64).exp()),
        -2.0 * l,
        l,
        0.0,
        0.0,
    ];
    let actual = evaluate(&gate, &up, 1.7, std::f64::consts::LOG2_E);
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(actual.is_finite(), "nonfinite output {index}: {actual}");
        assert!(
            (actual - expected).abs() < 1e-8,
            "output {index}: actual {actual}, expected {expected}"
        );
    }
}

#[test]
fn swiglu_fragment_uses_declared_exponent_and_ignores_obsolete_scale() {
    let gate = [0.5, -0.5];
    let up = [1.0, -1.0];
    let baseline = evaluate(&gate, &up, 1.7, std::f64::consts::LOG2_E);
    assert_eq!(
        baseline,
        evaluate(&gate, &up, f64::NAN, std::f64::consts::LOG2_E)
    );
    let base_two = evaluate(&gate, &up, -193.0, 1.0);
    for (index, &actual) in base_two.iter().enumerate() {
        let expected = gate[index] * up[index] / (1.0 + (-gate[index]).exp2());
        assert!((actual - expected).abs() < 1e-8);
        assert!((actual - baseline[index]).abs() > 1e-3);
    }
}

#[test]
fn swiglu_fragment_covers_signed_boundaries_and_small_inputs() {
    let gate = [16.0, -16.0, 16.0, -16.0, 1e-8, -1e-8];
    let up = [16.0, 16.0, -16.0, -16.0, 16.0, -16.0];
    let actual = evaluate(&gate, &up, 1.7, std::f64::consts::LOG2_E);
    for (index, &actual) in actual.iter().enumerate() {
        let expected = gate[index] * up[index] / (1.0 + (-gate[index]).exp());
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() < 1e-8,
            "output {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn swiglu_fragment_empty_and_compact_programs_use_only_bml_arithmetic() {
    assert!(evaluate(&[], &[], f64::NAN, std::f64::consts::LOG2_E).is_empty());
    let build = |width| {
        bml_swiglu_program(
            GATE_BASE,
            UP_BASE,
            width,
            UNUSED_SCALE,
            LOG2_E,
            ONE,
            OUTPUT_SLOT,
        )
    };
    let one = build(1);
    let wide = build(4096);
    assert_eq!(one.ops.len(), wide.ops.len());
    assert!(one.ops.iter().any(|op| matches!(op, RpnOp::Bml)));
    assert!(!one
        .ops
        .iter()
        .any(|op| matches!(op, RpnOp::FAdd | RpnOp::FMul)));
    assert!(one
        .ops
        .iter()
        .any(|op| matches!(op, RpnOp::Const(id) if *id == LOG2_E)));
    assert!(one
        .ops
        .iter()
        .any(|op| matches!(op, RpnOp::Const(id) if *id == ONE)));
    assert!(!one
        .ops
        .iter()
        .any(|op| matches!(op, RpnOp::Const(id) if *id == UNUSED_SCALE)));
    assert!(one.const_pool.is_empty());
}
