use bml_compiler::{
    symbolic_artifact::{checked_identity, compile_directory_bounded},
    symbolic_bounded::BoundedProfile,
    symbolic_kernels::KernelLimits,
    symbolic_weight_shard::{export_layer_weights, verify_layer_weights},
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
    let file = shard.join("weights.bin");
    let mut bytes = std::fs::read(&file).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&file, bytes).unwrap();
    assert!(verify_layer_weights(&shard, &digest, &identity, 1..3, 1_000_000).is_err());
    let mut desc = std::fs::read(shard.join("weights.json")).unwrap();
    desc.push(b' ');
    std::fs::write(shard.join("weights.json"), desc).unwrap();
    assert!(verify_layer_weights(&shard, &digest, &identity, 1..3, 1_000_000).is_err());
}
