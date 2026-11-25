use bml_compiler::HashConsedKV;
use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
#[path = "support/symbolic_fixture.rs"]
mod symbolic_fixture;

#[test]
fn range_execution_matches_complete_causal_forward_and_never_commits() {
    let fixture = symbolic_fixture::fixture(0.25);
    let mut sharded = SymbolicCausalModel::open(&fixture.0, SymbolicLimits::default()).unwrap();
    let mut complete = SymbolicCausalModel::open(&fixture.0, SymbolicLimits::default()).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 4);
    let mut full_cache = HashConsedKV::new(1, 1, 2, 4);
    for (position, id) in [69, 70, 69].into_iter().enumerate() {
        let input = sharded.compiler().get_embedding(id).unwrap();
        let hidden = sharded
            .forward_layer_range(&input, position as u32, &mut cache, 0..1)
            .unwrap();
        assert_eq!(cache.current_pos(), position as u32);
        let actual = sharded.project_output(&hidden).unwrap();
        let expected = complete.forward_cached(&[id], &mut full_cache).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        for prior in 0..=position as u32 {
            assert_eq!(cache.load_k(0, prior, 0), full_cache.load_k(0, prior, 0));
            assert_eq!(cache.load_v(0, prior, 0), full_cache.load_v(0, prior, 0));
        }
        cache.advance();
    }
}

#[test]
fn invalid_range_position_shape_and_nonfinite_input_fail_before_writes() {
    let fixture = symbolic_fixture::fixture(0.25);
    let mut model = SymbolicCausalModel::open(&fixture.0, SymbolicLimits::default()).unwrap();
    let mut cache = HashConsedKV::new(1, 1, 2, 4);
    let input = model.compiler().get_embedding(69).unwrap();
    let before = cache.load_k(0, 0, 0);
    assert!(model
        .forward_layer_range(&input, 0, &mut cache, 0..0)
        .is_err());
    assert!(model
        .forward_layer_range(&input, 0, &mut cache, 0..2)
        .is_err());
    assert!(model
        .forward_layer_range(&input, 1, &mut cache, 0..1)
        .is_err());
    assert!(model
        .forward_layer_range(&input[..1], 0, &mut cache, 0..1)
        .is_err());
    let mut invalid = input.clone();
    invalid[0] = f64::NAN;
    assert!(model
        .forward_layer_range(&invalid, 0, &mut cache, 0..1)
        .is_err());
    assert!(model.project_output(&invalid).is_err());
    assert_eq!(cache.load_k(0, 0, 0), before);
    assert_eq!(cache.current_pos(), 0);
}

#[test]
fn three_layer_gqa_model_composes_separate_ranges_across_tokens() {
    use bml_compiler::symbolic_bounded::BoundedProfile;
    let fixture = symbolic_fixture::fixture_layers(0.25, 3);
    struct Artifact(std::path::PathBuf);
    impl Drop for Artifact {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let artifact = Artifact(fixture.0.with_extension("range-artifact"));
    std::fs::create_dir(&artifact.0).unwrap();
    bml_compiler::symbolic_artifact::compile_directory_bounded(
        &fixture.0,
        &artifact.0,
        3,
        BoundedProfile::default(),
        bml_compiler::symbolic_kernels::KernelLimits::default(),
    )
    .unwrap();
    for mode in 0..3 {
        let open = || {
            if mode == 2 {
                SymbolicCausalModel::open_artifact(
                    &artifact.0,
                    None,
                    SymbolicLimits {
                        max_nodes: 0,
                        ..SymbolicLimits::default()
                    },
                )
                .unwrap()
            } else if mode == 1 {
                SymbolicCausalModel::open_bounded(
                    &fixture.0,
                    SymbolicLimits::default(),
                    BoundedProfile::default(),
                )
                .unwrap()
            } else {
                SymbolicCausalModel::open(&fixture.0, SymbolicLimits::default()).unwrap()
            }
        };
        let mut first = open();
        let mut second = open();
        let mut full = open();
        let mut first_cache = HashConsedKV::new(3, 1, 2, 4);
        let mut second_cache = HashConsedKV::new(3, 1, 2, 4);
        let mut full_cache = HashConsedKV::new(3, 1, 2, 4);
        for (position, id) in [69, 70, 69].into_iter().enumerate() {
            let input = first.compiler().get_embedding(id).unwrap();
            let intermediate = first
                .forward_layer_range(&input, position as u32, &mut first_cache, 0..1)
                .unwrap();
            let hidden = second
                .forward_layer_range(&intermediate, position as u32, &mut second_cache, 1..3)
                .unwrap();
            assert_eq!(first_cache.current_pos(), position as u32);
            assert_eq!(second_cache.current_pos(), position as u32);
            let logits = second.project_output(&hidden).unwrap();
            let expected = full.forward_cached(&[id], &mut full_cache).unwrap();
            if position == 0 {
                let mut incomplete = open();
                let mut incomplete_cache = HashConsedKV::new(3, 1, 2, 4);
                let skipped = incomplete
                    .forward_layer_range(&input, 0, &mut incomplete_cache, 0..2)
                    .unwrap();
                let skipped_logits = incomplete.project_output(&skipped).unwrap();
                assert!(skipped_logits
                    .iter()
                    .zip(&expected)
                    .any(|(a, b)| (a - b).abs() > 1e-9));
            }
            assert_eq!(
                logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            for prior in 0..=position as u32 {
                for layer in 0..3 {
                    let shard = if layer == 0 {
                        &first_cache
                    } else {
                        &second_cache
                    };
                    assert_eq!(
                        shard.load_k(layer, prior, 0),
                        full_cache.load_k(layer, prior, 0)
                    );
                    assert_eq!(
                        shard.load_v(layer, prior, 0),
                        full_cache.load_v(layer, prior, 0)
                    );
                }
            }
            first_cache.advance();
            second_cache.advance();
        }
    }
}

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, &word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, delta) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(delta);
        }
    }
    let mut out = String::with_capacity(64);
    for word in h {
        use std::fmt::Write;
        write!(&mut out, "{word:08x}").unwrap();
    }
    out
}

#[test]
fn forward_layer_range_project_output_and_forward_enforce_each_guard_and_shard_subrange() {
    use bml_compiler::symbolic_bounded::BoundedProfile;
    struct Dir(std::path::PathBuf);
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    let single = symbolic_fixture::fixture(0.25);
    let mut model = SymbolicCausalModel::open(&single.0, SymbolicLimits::default()).unwrap();
    let input = model.compiler().get_embedding(69).unwrap();

    // Cache covers 0..2 so only layers.end > config.n_layers (1) triggers.
    let mut oversize_cache = HashConsedKV::new(2, 1, 2, 4);
    assert_eq!(
        model
            .forward_layer_range(&input, 0, &mut oversize_cache, 0..2)
            .unwrap_err(),
        "invalid symbolic layer range context"
    );

    // Valid length non-finite hidden rejected before kernel execution.
    let mut valid_cache = HashConsedKV::new(1, 1, 2, 4);
    let invalid = [f64::NAN, 0.0, 0.0, 0.0];
    assert_eq!(
        model
            .forward_layer_range(&invalid, 0, &mut valid_cache, 0..1)
            .unwrap_err(),
        "invalid symbolic layer range context"
    );
    assert_eq!(
        model.project_output(&invalid).unwrap_err(),
        "invalid symbolic output hidden state"
    );

    // Cache with mismatched kv_heads (2 != 1) rejected by covers_layers guard.
    let mut wrong_kv_cache = HashConsedKV::new(1, 2, 2, 4);
    assert_eq!(
        model
            .forward_layer_range(&input, 0, &mut wrong_kv_cache, 0..1)
            .unwrap_err(),
        "invalid symbolic layer range context"
    );

    // Non-empty prefix strictly above model context_length (8) rejected in forward().
    assert_eq!(
        model.forward(&[69; 9]).unwrap_err(),
        "invalid symbolic context length"
    );

    // Sharded model covering 1..3 accepts strict subranges 1..2 and 2..3 and rejects 0..1.
    let three = symbolic_fixture::fixture_layers(0.25, 3);
    let artifact = Dir(three.0.with_extension("shard-subrange-artifact"));
    std::fs::create_dir(&artifact.0).unwrap();
    bml_compiler::symbolic_artifact::compile_directory_bounded(
        &three.0,
        &artifact.0,
        3,
        BoundedProfile::default(),
        bml_compiler::symbolic_kernels::KernelLimits::default(),
    )
    .unwrap();
    let identity = bml_compiler::symbolic_artifact::checked_identity(&artifact.0, None).unwrap();
    let shard = artifact.0.join("shard-1-3");
    let shard_hash = bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact.0,
        None,
        1..3,
        &shard,
        1_000_000,
    )
    .unwrap();
    let mut sharded = SymbolicCausalModel::open_authenticated_shard(
        &artifact.0,
        &identity,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
        &shard,
        &shard_hash,
        1..3,
        1_000_000,
    )
    .unwrap();
    let mut cache_all = HashConsedKV::new(3, 1, 2, 4);
    // Out-of-shard range 0..1 must fail at the layer range guard, not in shard lookup.
    assert_eq!(
        sharded
            .forward_layer_range(&input, 0, &mut cache_all, 0..1)
            .unwrap_err(),
        "invalid symbolic layer range context"
    );
    // Strict subranges inside 1..3 succeed (kills < -> > and > -> < on shard bounds).
    let mid = sharded
        .forward_layer_range(&input, 0, &mut cache_all, 1..2)
        .unwrap();
    let _out = sharded
        .forward_layer_range(&mid, 0, &mut cache_all, 2..3)
        .unwrap();
}

#[test]
fn open_authenticated_shard_and_project_reject_out_of_range_and_bad_shard_dimensions() {
    use bml_compiler::symbolic_bounded::BoundedProfile;
    struct Dir(std::path::PathBuf);
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    let three = symbolic_fixture::fixture_layers(0.25, 3);
    let artifact = Dir(three.0.with_extension("forged-shard-artifact"));
    std::fs::create_dir(&artifact.0).unwrap();
    bml_compiler::symbolic_artifact::compile_directory_bounded(
        &three.0,
        &artifact.0,
        3,
        BoundedProfile::default(),
        bml_compiler::symbolic_kernels::KernelLimits::default(),
    )
    .unwrap();
    let identity = bml_compiler::symbolic_artifact::checked_identity(&artifact.0, None).unwrap();

    // Case 1: 2D shard tensor with dims[0] != width (dims = [8, 2] instead of [4, 4]).
    let bad_dims_dir = artifact.0.join("bad-dims-shard");
    bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact.0,
        None,
        0..1,
        &bad_dims_dir,
        1_000_000,
    )
    .unwrap();
    let mut desc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(bad_dims_dir.join("weights.json")).unwrap()).unwrap();
    for entry in desc["tensors"].as_array_mut().unwrap() {
        if entry["name"].as_str() == Some("blk.0.attn_q.weight") {
            entry["dims"] = serde_json::json!([8, 2]);
        }
    }
    let edited_bytes = serde_json::to_vec(&desc).unwrap();
    let edited_hash = sha256_hex(&edited_bytes);
    std::fs::write(bad_dims_dir.join("weights.json"), &edited_bytes).unwrap();
    let mut model = SymbolicCausalModel::open_authenticated_shard(
        &artifact.0,
        &identity,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
        &bad_dims_dir,
        &edited_hash,
        0..1,
        1_000_000,
    )
    .unwrap();
    let mut cache = HashConsedKV::new(3, 1, 2, 4);
    let hidden = [0.25, 0.0, 0.0, 0.0];
    assert_eq!(
        model
            .forward_layer_range(&hidden, 0, &mut cache, 0..1)
            .unwrap_err(),
        "shard projection dimension mismatch"
    );

    // Case 2: Forged self-consistent shard claiming layers 0..4 on a 3-layer artifact.
    let over_dir = artifact.0.join("over-layers-shard");
    bml_compiler::symbolic_weight_shard::export_layer_weights(
        &artifact.0,
        None,
        0..3,
        &over_dir,
        1_000_000,
    )
    .unwrap();
    let mut payload = std::fs::read(over_dir.join("weights.bin")).unwrap();
    let mut desc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(over_dir.join("weights.json")).unwrap()).unwrap();
    let tensors = desc["tensors"].as_array().unwrap().clone();
    let mut new_tensors = tensors.clone();
    for entry in &tensors {
        let name = entry["name"].as_str().unwrap();
        if let Some(suffix) = name.strip_prefix("blk.0.") {
            let offset = entry["offset"].as_u64().unwrap() as usize;
            let length = entry["length"].as_u64().unwrap() as usize;
            let new_offset = payload.len() as u64;
            payload.extend_from_within(offset..offset + length);
            let mut cloned = entry.clone();
            cloned["name"] = serde_json::json!(format!("blk.3.{suffix}"));
            cloned["offset"] = serde_json::json!(new_offset);
            new_tensors.push(cloned);
        }
    }
    desc["end"] = serde_json::json!(4);
    desc["payload_length"] = serde_json::json!(payload.len() as u64);
    desc["payload_sha256"] = serde_json::json!(sha256_hex(&payload));
    desc["tensors"] = serde_json::Value::Array(new_tensors);
    std::fs::write(over_dir.join("weights.bin"), &payload).unwrap();
    let over_bytes = serde_json::to_vec(&desc).unwrap();
    let over_hash = sha256_hex(&over_bytes);
    std::fs::write(over_dir.join("weights.json"), &over_bytes).unwrap();
    assert_eq!(
        SymbolicCausalModel::open_authenticated_shard(
            &artifact.0,
            &identity,
            SymbolicLimits {
                max_nodes: 0,
                ..Default::default()
            },
            &over_dir,
            &over_hash,
            0..4,
            1_000_000,
        )
        .err()
        .unwrap(),
        "invalid symbolic shard layer range"
    );
}
