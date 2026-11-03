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
