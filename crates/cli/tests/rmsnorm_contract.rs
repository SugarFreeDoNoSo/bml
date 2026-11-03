//! Closed-form numerical checks for the public legacy RMSNorm helper.
use bml_cli::bml_inference::bml_rmsnorm;
use bml_runtime::HotLoop;

#[test]
fn rmsnorm_preserves_shape_sign_weights_offset_and_default_epsilon() {
    let mut hot = HotLoop::with_capacity(16);
    // Mean square is exactly 25/3; only weights at offset 1 belong to this norm.
    let output = bml_rmsnorm(&mut hot, &[3.0, -4.0, 0.0], &[99.0, 2.0, 0.5, -3.0], 1, 3);
    let denominator = (25.0_f64 / 3.0 + 1e-5).sqrt();
    let expected = [6.0 / denominator, -2.0 / denominator, 0.0];
    assert_eq!(
        output.len(),
        expected.len(),
        "RMSNorm preserves vector shape"
    );
    for (actual, expected) in output.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-12, "{actual} != {expected}");
    }

    // Here epsilon dominates: the denominator is sqrt(1.25e-5).
    let output = bml_rmsnorm(&mut hot, &[0.001, -0.002], &[2.0, -0.5], 0, 2);
    let expected = [0.002 / 1.25e-5_f64.sqrt(), 0.001 / 1.25e-5_f64.sqrt()];
    assert_eq!(output.len(), expected.len());
    for (actual, expected) in output.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-12, "{actual} != {expected}");
    }
}
