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
