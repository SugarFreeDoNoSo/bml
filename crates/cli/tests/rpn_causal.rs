//! Independent closed-form GQA fixture, executed through actual RPN projections.
use bml_cli::bml_inference::{bml_matmul, forward_bml, forward_bml_cached};
use bml_compiler::{gguf_compiler::InferenceCompiler, HashConsedKV};
use bml_runtime::HotLoop;
#[path = "../../compiler/tests/support/causal_fixture.rs"]
mod causal_fixture;

#[test]
fn rpn_signed_rectangular_projection_uses_contiguous_rows_and_checked_ranges() {
    let mut hot = HotLoop::with_capacity(16);
    let weights = [99.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 123.0];
    assert_eq!(
        bml_matmul(&mut hot, &[2.0, -1.0, 0.5], &weights, 1, 3, 2).unwrap(),
        [5.5, -16.0]
    );
    assert_eq!(
        bml_matmul(&mut hot, &[2.0, -1.0], &weights, 1, 2, 3).unwrap(),
        [4.0, 10.0, 16.0]
    );
    for (offset, input, output) in [
        (1, 3, 3),
        (u64::MAX, 3, 2),
        (0, 0, 2),
        (0, 3, 0),
        (0, u32::MAX, 2),
    ] {
        assert!(bml_matmul(&mut hot, &[2.0, -1.0, 0.5], &weights, offset, input, output).is_err());
    }
    assert!(bml_matmul(&mut hot, &[2.0], &weights, 1, 3, 2).is_err());
}

#[test]
fn rpn_causal_prefix_and_incremental_gqa_match_closed_form_and_native() {
    let model = causal_fixture::fixture(false);
    for compiler in [
        InferenceCompiler::open(&model.0).unwrap(),
        InferenceCompiler::open_mapped(&model.0).unwrap(),
    ] {
        let mut hot = HotLoop::with_capacity(16);
        for (ids, hidden) in [
            (vec![69], [3.0_f64, 0.0, 1.0, 0.0]),
            (vec![69, 70], [0.5, 2.5, 0.5, 0.5]),
            (vec![70, 69], [2.5, 0.5, 0.5, 0.5]),
            (
                vec![69, 70, 69],
                [8.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0],
            ),
        ] {
            let actual = forward_bml(&mut hot, &compiler, &ids).unwrap();
            let denom = (hidden.iter().map(|v| v * v).sum::<f64>() / 4.0 + 3.0).sqrt();
            for (&value, expected) in actual.iter().zip(hidden.map(|v| v / denom)) {
                assert!((value - expected).abs() < 1e-6);
            }
            assert!(actual[4..].iter().all(|&v| v == 0.0));
            assert_eq!(actual, compiler.forward(&ids).unwrap());
            let mut cache = HashConsedKV::new(1, 1, 2, 4);
            let mut incremental = Vec::new();
            for id in &ids {
                incremental = forward_bml_cached(&mut hot, &compiler, &[*id], &mut cache).unwrap();
            }
            assert_eq!(actual, incremental);
            assert_eq!(cache.current_pos(), ids.len() as u32);
        }
    }
}
