use bml_compiler::{
    symbolic_artifact::compile_directory_bounded, symbolic_bounded::BoundedProfile,
    symbolic_kernels::KernelLimits, HashConsedKV,
};
use bml_runtime::{
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
    symbolic_session::SymbolicLayerSession,
};
#[allow(dead_code)]
#[path = "support/symbolic_fixture.rs"]
mod symbolic_fixture;

#[test]
fn persisted_sessions_prepare_abort_commit_and_reset_without_losing_causal_equivalence() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let directory = Directory(source.0.with_extension("session-artifact"));
    std::fs::create_dir(&directory.0).unwrap();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let open = || {
        SymbolicCausalModel::open_artifact(
            &directory.0,
            None,
            SymbolicLimits {
                max_nodes: 0,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let mut first = SymbolicLayerSession::new(open(), 0..1, 3, 48).unwrap();
    let mut second = SymbolicLayerSession::new(open(), 1..3, 3, 96).unwrap();
    let mut head = open();
    let mut complete = open();
    let mut cache = HashConsedKV::new(3, 1, 2, 3);
    for (position, id) in [69, 70, 69].into_iter().enumerate() {
        let input = first.model().compiler().get_embedding(id).unwrap();
        let position = position as u32;
        let initial = first.prepare(position, &input).unwrap();
        assert!(first.prepare(position, &input).is_err());
        assert!(first.commit(position + 1).is_err());
        first.abort(position).unwrap();
        let intermediate = first.prepare(position, &input).unwrap();
        assert_eq!(
            initial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            intermediate.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let hidden = second.prepare(position, &intermediate).unwrap();
        assert_eq!(first.position(), position);
        assert_eq!(second.position(), position);
        let actual = head.project_output(&hidden).unwrap();
        let expected = complete.forward_cached(&[id], &mut cache).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        first.commit(position).unwrap();
        second.commit(position).unwrap();
        assert!(first.commit(position).is_err());
    }
    assert!(first.prepare(3, &[0.0; 4]).is_err());
    first.reset().unwrap();
    second.reset().unwrap();
    assert_eq!((first.position(), second.position()), (0, 0));
    let input = first.model().compiler().get_embedding(69).unwrap();
    let intermediate = first.prepare(0, &input).unwrap();
    let hidden = second.prepare(0, &intermediate).unwrap();
    let actual = head.project_output(&hidden).unwrap();
    let expected = complete.forward(&[69]).unwrap();
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn session_rejects_invalid_layout_budget_and_uncommitted_errors() {
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    let open = || {
        SymbolicCausalModel::open_bounded(
            &source.0,
            SymbolicLimits::default(),
            BoundedProfile::default(),
        )
        .unwrap()
    };
    assert!(SymbolicLayerSession::new(open(), 0..0, 3, 4096).is_err());
    assert!(SymbolicLayerSession::new(open(), 0..4, 3, 4096).is_err());
    assert!(SymbolicLayerSession::new(open(), 0..1, 0, 4096).is_err());
    assert!(SymbolicLayerSession::new(open(), 0..1, 3, 47).is_err());
    assert!(SymbolicLayerSession::new(open(), 1..3, 3, 95).is_err());
    let mut session = SymbolicLayerSession::new(open(), 0..1, 1, 4096).unwrap();
    assert!(session.abort(0).is_err());
    assert!(session.commit(0).is_err());
    assert!(session.prepare(1, &[0.0; 4]).is_err());
    assert!(session.prepare(0, &[f64::NAN; 4]).is_err());
    assert_eq!(session.position(), 0);
    assert!(!session.is_prepared());
    let input = session.model().compiler().get_embedding(69).unwrap();
    session.prepare(0, &input).unwrap();
    session.reset().unwrap();
    assert_eq!(session.position(), 0);
    assert!(!session.is_prepared());
    session.prepare(0, &input).unwrap();
    session.commit(0).unwrap();
    assert!(session.prepare(1, &input).is_err());
}

#[test]
fn range_cache_uses_global_layer_indices_and_enforces_its_budget() {
    let mut cache = HashConsedKV::try_new_layer_range(2..4, 1, 2, 3, 96).unwrap();
    assert_eq!(cache.byte_size(), 96);
    assert!(cache.covers_layers(2..4, 1, 2));
    assert!(cache.covers_layers(3..4, 1, 2));
    assert!(!cache.covers_layers(1..3, 1, 2));
    assert!(!cache.covers_layers(2..5, 1, 2));
    assert!(!cache.matches_layout(2, 1, 2));
    cache.store_k(2, 1, 0, &[1.0, 2.0]);
    cache.store_v(3, 1, 0, &[3.0, 4.0]);
    assert_eq!(cache.load_k(2, 1, 0), [1.0, 2.0]);
    assert_eq!(cache.load_v(3, 1, 0), [3.0, 4.0]);
    assert!(HashConsedKV::try_new_layer_range(2..4, 1, 2, 3, 95).is_err());
    assert!(
        HashConsedKV::try_new_layer_range(std::ops::Range { start: 4, end: 2 }, 1, 2, 3, 96)
            .is_err()
    );
}

#[test]
fn verified_weight_shards_execute_layer_ranges_with_identical_logits() {
    use bml_compiler::symbolic_weight_shard::export_layer_weights;
    let source = symbolic_fixture::fixture_layers(0.25, 3);
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let dir = Directory(source.0.with_extension("sharded-execution"));
    std::fs::create_dir(&dir.0).unwrap();
    let artifact = dir.0.join("artifact");
    std::fs::create_dir(&artifact).unwrap();
    compile_directory_bounded(
        &source.0,
        &artifact,
        3,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    let shard0 = dir.0.join("shard0");
    let shard1 = dir.0.join("shard1");
    let hash0 = export_layer_weights(&artifact, None, 0..1, &shard0, 1_000_000).unwrap();
    let hash1 = export_layer_weights(&artifact, None, 1..3, &shard1, 1_000_000).unwrap();
    let limits = SymbolicLimits {
        max_nodes: 0,
        ..Default::default()
    };
    assert!(SymbolicCausalModel::open_artifact_with_shard(
        &artifact,
        None,
        limits,
        &shard0,
        "wrong",
        0..1,
        1_000_000
    )
    .is_err());
    let open = |path: &std::path::Path, hash: &str, range| {
        SymbolicCausalModel::open_artifact_with_shard(
            &artifact, None, limits, path, hash, range, 1_000_000,
        )
        .unwrap()
    };
    let mut a = SymbolicLayerSession::new(open(&shard0, &hash0, 0..1), 0..1, 3, 48).unwrap();
    let mut b = SymbolicLayerSession::new(open(&shard1, &hash1, 1..3), 1..3, 3, 96).unwrap();
    let mut local = SymbolicCausalModel::open_artifact(&artifact, None, limits).unwrap();
    let mut head = SymbolicCausalModel::open_artifact(&artifact, None, limits).unwrap();
    let mut kv = HashConsedKV::new(3, 1, 2, 3);
    for (pos, id) in [69, 70, 69].into_iter().enumerate() {
        let input = local.compiler().get_embedding(id).unwrap();
        let partial = a.prepare(pos as u32, &input).unwrap();
        let result = b.prepare(pos as u32, &partial).unwrap();
        let logits = head.project_output(&result).unwrap();
        let expected = local.forward_cached(&[id], &mut kv).unwrap();
        assert_eq!(
            logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        a.commit(pos as u32).unwrap();
        b.commit(pos as u32).unwrap();
    }
}
