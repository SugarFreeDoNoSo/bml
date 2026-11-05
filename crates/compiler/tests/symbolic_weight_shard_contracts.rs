use bml_compiler::{
    symbolic_artifact::{checked_identity, compile_directory_bounded},
    symbolic_bounded::BoundedProfile,
    symbolic_kernels::KernelLimits,
    symbolic_weight_shard::{export_layer_weights, verify_layer_weights, VerifiedLayerWeights},
};
use sha2::{Digest, Sha256};
#[allow(dead_code)]
#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;

#[test]
fn authenticated_tensor_layout_rejects_independent_names_and_offsets() {
    let source = symbolic_fixture::fixture(0.25);
    let root = source.0.with_extension("semantic-tensor-layout");
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
        1,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&artifact, None).unwrap();
    let original = root.join("original");
    let descriptor_hash =
        export_layer_weights(&artifact, None, 0..1, &original, 1_000_000).unwrap();
    let valid = VerifiedLayerWeights::open(&original, &descriptor_hash, &identity, 0..1, 1_000_000)
        .unwrap();
    assert_eq!(
        valid.decode_row_f32("blk.0.attn_norm.weight", 0).unwrap(),
        [0.5; 4]
    );
    let descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original.join("weights.json")).unwrap()).unwrap();
    assert_eq!(descriptor["tensors"][0]["name"], "blk.0.attn_norm.weight");
    assert_eq!(descriptor["tensors"][0]["offset"], 0);
    let payload = std::fs::read(original.join("weights.bin")).unwrap();
    for (name, field, value) in [
        (
            "wrong-name",
            "name",
            serde_json::json!("blk.0.unexpected.weight"),
        ),
        ("wrong-offset", "offset", serde_json::json!(1)),
    ] {
        let directory = root.join(name);
        std::fs::create_dir(&directory).unwrap();
        let mut changed = descriptor.clone();
        changed["tensors"][0][field] = value;
        let encoded = serde_json::to_vec(&changed).unwrap();
        let authenticated = format!("{:x}", Sha256::digest(&encoded));
        std::fs::write(directory.join("weights.json"), encoded).unwrap();
        std::fs::write(directory.join("weights.bin"), &payload).unwrap();
        // Trust the new descriptor hash while preserving every payload and
        // tensor hash: neither authentication nor a second invalid field may
        // mask the independent semantic name and contiguous-offset checks.
        let error =
            VerifiedLayerWeights::open(&directory, &authenticated, &identity, 0..1, 1_000_000)
                .err()
                .unwrap();
        assert_eq!(error, "invalid shard tensor layout", "{name}");
    }
}

#[test]
fn authenticated_layer_shards_reject_independent_range_mismatches_and_empty_ranges() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let root = source.0.with_extension("semantic-weight-ranges");
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
    let original = root.join("original");
    let descriptor_hash =
        export_layer_weights(&artifact, None, 1..3, &original, 1_000_000).unwrap();
    VerifiedLayerWeights::open(&original, &descriptor_hash, &identity, 1..3, 1_000_000).unwrap();
    let descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original.join("weights.json")).unwrap()).unwrap();
    let payload = std::fs::read(original.join("weights.bin")).unwrap();
    for (name, start, end, empty) in [
        ("wrong-start", 0_u32, 3_u32, false),
        ("wrong-end", 1, 4, false),
        ("empty-range", 1, 1, true),
    ] {
        let directory = root.join(name);
        std::fs::create_dir(&directory).unwrap();
        let mut changed = descriptor.clone();
        changed["start"] = start.into();
        changed["end"] = end.into();
        let bytes = if empty { &[][..] } else { payload.as_slice() };
        if empty {
            changed["tensors"] = serde_json::json!([]);
            changed["payload_length"] = 0.into();
            changed["payload_sha256"] = format!("{:x}", Sha256::digest(bytes)).into();
        }
        let encoded = serde_json::to_vec(&changed).unwrap();
        let authenticated = format!("{:x}", Sha256::digest(&encoded));
        std::fs::write(directory.join("weights.json"), encoded).unwrap();
        std::fs::write(directory.join("weights.bin"), bytes).unwrap();
        // Re-authenticate each descriptor: a stale digest must not mask the
        // independent semantic checks on its start, end, or nonempty range.
        let requested = if empty { 1..1 } else { 1..3 };
        let error =
            VerifiedLayerWeights::open(&directory, &authenticated, &identity, requested, 1_000_000)
                .err()
                .unwrap();
        assert_eq!(error, "shard identity, range or budget mismatch", "{name}");
    }
}

#[test]
fn authenticated_workers_reject_individually_invalid_model_metadata() {
    use bml_compiler::symbolic_artifact::{load_directory_authenticated, MANIFEST};
    use serde_json::json;
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let root = source.0.with_extension("authenticated-metadata");
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    std::fs::create_dir(&root).unwrap();
    compile_directory_bounded(
        &source.0,
        &root,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let identity = checked_identity(&root, None).unwrap();
    let original = std::fs::read(root.join(MANIFEST)).unwrap();
    let valid: serde_json::Value = serde_json::from_slice(&original).unwrap();
    // Exercise the worker-only metadata path without access to a source GGUF.
    std::fs::remove_file(&source.0).unwrap();
    let loaded = load_directory_authenticated(&root, &identity).unwrap();
    assert_eq!(loaded.compiler.config().n_layers, 3);
    assert_eq!(loaded.compiler.mapped_bytes(), 0);
    for (section, field, value) in [
        ("config", "architecture", json!("other")),
        ("config", "n_layers", json!(0)),
        ("config", "n_heads", json!(0)),
        ("config", "n_embd", json!(0)),
        ("config", "context_length", json!(0)),
        ("config", "n_embd", json!(3)),
        ("", "kv_heads", json!(0)),
        ("", "kv_heads", json!(3)),
        ("", "rms_norm_epsilon", json!(0.0)),
        ("", "rms_norm_epsilon", json!(-1.0)),
        ("", "rope_frequency_base", json!(0.0)),
        ("", "rope_frequency_base", json!(-1.0)),
        (
            "config",
            "vocab_size",
            json!(valid["config"]["vocab_size"].as_u64().unwrap() + 1),
        ),
    ] {
        let mut bad = valid.clone();
        let object = if section.is_empty() {
            &mut bad
        } else {
            &mut bad[section]
        };
        object[field] = value;
        let bytes = serde_json::to_vec(&bad).unwrap();
        std::fs::write(root.join(MANIFEST), &bytes).unwrap();
        // A matching trusted digest does not replace format validation.
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let error = load_directory_authenticated(&root, &digest)
            .err()
            .expect("individually invalid metadata must be rejected");
        let expected = if field == "vocab_size" {
            "symbolic vocabulary size mismatch"
        } else {
            "invalid authenticated symbolic model metadata"
        };
        assert_eq!(error, expected, "{section}.{field}");
    }
    std::fs::write(root.join(MANIFEST), original).unwrap();
    load_directory_authenticated(&root, &identity).unwrap();
}

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
