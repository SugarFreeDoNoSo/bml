use bml_compiler::distributed::{compile_distributed, compile_distributed_bml};
use std::path::Path;

#[test]
fn zero_layers_per_fragment_is_rejected_before_opening_model() {
    let absent = Path::new("this-model-must-not-be-opened.gguf");
    let plain = compile_distributed(absent, 0, false).unwrap_err();
    assert!(plain.contains("layers_per_fragment"));
    let encoded = compile_distributed_bml(absent, 0)
        .err()
        .expect("zero must fail");
    assert!(encoded.contains("layers_per_fragment"));
}
