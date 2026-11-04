use bml_compiler::{
    symbolic_artifact::{checked_identity, compile_directory_bounded},
    symbolic_bounded::BoundedProfile,
    symbolic_kernels::KernelLimits,
    symbolic_weight_shard::{export_layer_weights, verify_layer_weights, VerifiedLayerWeights},
};
#[allow(dead_code)]
#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;

#[test]
fn shard_export_is_range_bounded_and_requires_trusted_identity() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let root = source.0.with_extension("weight-shards");
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    std::fs::create_dir(&root).unwrap();
    let artifact = root.join("artifact");
    std::fs::create_dir(&artifact).unwrap();
    compile_directory_bounded(
        &source.0,
        &artifact,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&artifact, None).unwrap();
    let shard = root.join("shard");
    assert!(export_layer_weights(&artifact, None, 1..3, &shard, 1).is_err());
    assert!(!shard.exists());
    let digest = export_layer_weights(&artifact, None, 1..3, &shard, 1_000_000).unwrap();
    verify_layer_weights(&shard, &digest, &identity, 1..3, 1_000_000).unwrap();
    verify_layer_weights(&shard, &digest, &identity, 1..3, u64::MAX).unwrap();
    let descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(shard.join("weights.json")).unwrap()).unwrap();
    let tensors = descriptor["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 18);
    assert!(tensors.iter().all(|tensor| {
        let name = tensor["name"].as_str().unwrap();
        name.starts_with("blk.1.") || name.starts_with("blk.2.")
    }));
    assert!(
        std::fs::metadata(shard.join("weights.bin")).unwrap().len()
            < std::fs::metadata(&source.0).unwrap().len()
    );
    assert!(verify_layer_weights(&shard, &digest, &identity, 0..2, 1_000_000).is_err());
    assert!(verify_layer_weights(&shard, &digest, "wrong", 1..3, 1_000_000).is_err());
    assert!(verify_layer_weights(&shard, &digest, &identity, 1..3, 1).is_err());
    let opened = VerifiedLayerWeights::open(&shard, &digest, &identity, 1..3, 1_000_000).unwrap();
    let original = bml_parser::GgufParser::open(&source.0).unwrap();
    for tensor in tensors {
        let name = tensor["name"].as_str().unwrap();
        let view = original.tensor_view(name).unwrap();
        assert_eq!(opened.dimensions(name), Some(view.ggml_dims()));
        let rows = view.ggml_dims().iter().skip(1).product::<u64>() as usize;
        for row in 0..rows {
            assert_eq!(
                opened.decode_row_f32(name, row).unwrap(),
                view.decode_row_f32(row).unwrap()
            );
        }
        assert!(opened.decode_row_f32(name, rows).is_err());
    }
    assert!(opened.decode_row_f32("blk.0.attn_q.weight", 0).is_err());
    let file = shard.join("weights.bin");
    let mut bytes = std::fs::read(&file).unwrap();
    let before = opened.decode_row_f32("blk.1.attn_norm.weight", 0).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&file, bytes).unwrap();
    assert!(verify_layer_weights(&shard, &digest, &identity, 1..3, 1_000_000).is_err());
    assert_eq!(
        opened.decode_row_f32("blk.1.attn_norm.weight", 0).unwrap(),
        before
    );
    let mut desc = std::fs::read(shard.join("weights.json")).unwrap();
    desc.push(b' ');
    std::fs::write(shard.join("weights.json"), desc).unwrap();
    assert!(verify_layer_weights(&shard, &digest, &identity, 1..3, 1_000_000).is_err());
}

#[test]
#[ignore = "requires BML_TEST_GGUF and BML_TEST_SYMBOLIC_ARTIFACT"]
fn real_quantized_layer_shard_preserves_gguf_rows() {
    let source = std::path::PathBuf::from(std::env::var_os("BML_TEST_GGUF").unwrap());
    let artifact =
        std::path::PathBuf::from(std::env::var_os("BML_TEST_SYMBOLIC_ARTIFACT").unwrap());
    let output = std::env::temp_dir().join(format!("bml-real-weight-shard-{}", std::process::id()));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(output.clone());
    let identity = checked_identity(&artifact, Some(&source)).unwrap();
    let digest =
        export_layer_weights(&artifact, Some(&source), 0..1, &output, 536_870_912).unwrap();
    let shard = VerifiedLayerWeights::open(&output, &digest, &identity, 0..1, 536_870_912).unwrap();
    let parser = bml_parser::GgufParser::open(&source).unwrap();
    let descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("weights.json")).unwrap()).unwrap();
    let entries = descriptor["tensors"].as_array().unwrap();
    let mut quantized = 0;
    for entry in entries {
        let name = entry["name"].as_str().unwrap();
        let view = parser.tensor_view(name).unwrap();
        if view.dtype() == bml_gguf::GgufDataType::Q4_0 {
            quantized += 1;
        }
        assert_eq!(shard.dimensions(name), Some(view.ggml_dims()));
        let rows = view.ggml_dims().iter().skip(1).product::<u64>() as usize;
        for row in [0, rows / 2, rows - 1] {
            let expected = view.decode_row_f32(row).unwrap();
            let actual = shard.decode_row_f32(name, row).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name} row {row}"
            );
        }
    }
    assert!(
        quantized > 0,
        "model must actually contain Q4_0 layer tensors"
    );
    let bytes = std::fs::metadata(output.join("weights.bin")).unwrap().len();
    assert!(bytes < std::fs::metadata(&source).unwrap().len());
    println!(
        "layer=0 tensors={} q4_tensors={quantized} shard_bytes={bytes}",
        entries.len()
    );
}
