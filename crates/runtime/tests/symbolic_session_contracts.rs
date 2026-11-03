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
    let mut first = SymbolicLayerSession::new(open(), 0..1, 3, 4096).unwrap();
    let mut second = SymbolicLayerSession::new(open(), 1..3, 3, 4096).unwrap();
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
    assert!(SymbolicLayerSession::new(open(), 0..1, 3, 1).is_err());
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
