//! A real optional-feature consumer, included in ordinary workspace CI.
use bml_domain::tensor::{matmul, Shape, Tensor};
use bml_gguf::GgufReader;

struct Fixture(std::path::PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn gguf_tensor_bridge_is_available_to_workspace_consumers() {
    // Independent GGUF v3 fixture, not the production synthetic-file helper.
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(6u64.to_le_bytes());
    bytes.extend(b"matrix");
    bytes.extend(2u32.to_le_bytes());
    bytes.extend(3u64.to_le_bytes()); // GGML ne[0]: contiguous columns.
    bytes.extend(2u64.to_le_bytes()); // GGML ne[1]: rows.
    bytes.extend(0u32.to_le_bytes()); // F32.
    bytes.extend(0u64.to_le_bytes()); // Relative tensor offset.
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    for value in [1f32, 2., 3., 4., 5., 6.] {
        bytes.extend(value.to_le_bytes());
    }
    let fixture = Fixture(
        std::env::temp_dir().join(format!("bml_bridge_consumer_{}.gguf", std::process::id())),
    );
    std::fs::write(&fixture.0, bytes).unwrap();
    let a = {
        let reader = GgufReader::open(&fixture.0).unwrap();
        let view = reader.tensor_view("matrix").unwrap();
        assert_eq!(view.ggml_dims(), [3, 2]);
        view.to_tensor_f32().unwrap()
    }; // The owned tensor survives the reader and mmap.
    assert_eq!(a.shape().dims(), [2, 3]);
    assert_eq!(a.as_slice(), [1., 2., 3., 4., 5., 6.]);
    let b = Tensor::from_vec(Shape::new([3, 1]).unwrap(), vec![2f32, -1., 0.5]).unwrap();
    let product = matmul(&a.view(), &b.view()).unwrap();
    assert_eq!(product.shape().dims(), [2, 1]);
    assert_eq!(product.as_slice(), [1.5, 6.]);
}
