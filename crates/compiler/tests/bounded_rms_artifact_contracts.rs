//! Explicit bounded RMS arithmetic, persisted separately from historical v2.
//! The causal oracle uses ordinary arithmetic and no production kernel helpers.
use bml_compiler::{
    symbolic_artifact::{
        compile_directory, compile_directory_with_rms_bounds, load_directory, Manifest,
        BOUNDED_RMS_PROFILE, MANIFEST, PROFILE,
    },
    symbolic_kernels::{
        compile_operation, compile_operation_with_rms_bounds, CompiledKernel, KernelLimits,
        Operation, OutputLayout, RmsBounds,
    },
    BmlGraph, HashConsedKV, RpnOp,
};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;
#[path = "../../runtime/tests/support/symbolic_oracle.rs"]
mod symbolic_oracle;

const BOUNDS: RmsBounds = RmsBounds {
    input: 16.0,
    weight: 16.0,
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_bounded_rms_artifact_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn manifest(&self) -> Manifest {
        serde_json::from_slice(&std::fs::read(self.0.join(MANIFEST)).unwrap()).unwrap()
    }
    fn save(&self, manifest: &Manifest) {
        std::fs::write(self.0.join(MANIFEST), serde_json::to_vec(manifest).unwrap()).unwrap();
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn artifact(source: &symbolic_fixture::Fixture, bounds: RmsBounds) -> Directory {
    let dir = Directory::new();
    compile_directory_with_rms_bounds(
        &source.0,
        &dir.0,
        3,
        bounds,
        KernelLimits {
            fragment_bytes: 256,
            ..Default::default()
        },
    )
    .unwrap();
    dir
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() < tolerance,
            "element {index}: {actual} != {expected}, tolerance {tolerance}"
        );
    }
}

// Serialize before execution, check the stored basis independently, and use the
// strict runtime entry point so finite final outputs cannot mask bad internals.
fn checked_kernel(
    kernel: &CompiledKernel,
    inputs: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, String> {
    assert_eq!(kernel.layout, OutputLayout::Buffer);
    assert_eq!(kernel.graphs.len(), 1);
    assert_eq!(inputs.len(), kernel.signature.inputs);
    assert_eq!(weights.len(), kernel.signature.external);
    let original = &kernel.graphs[0];
    let stored = BmlGraph::deserialize(&original.serialize(), original.threshold).unwrap();
    assert!(stored.const_pool.is_empty());
    assert!(stored.fragments.len() > 1);
    for op in stored.fragments.iter().flat_map(|fragment| &fragment.ops) {
        assert!(matches!(
            op,
            RpnOp::One
                | RpnOp::Var(_)
                | RpnOp::Const(_)
                | RpnOp::Bml
                | RpnOp::Pick { .. }
                | RpnOp::Dup
                | RpnOp::Drop
                | RpnOp::Swap
                | RpnOp::StoreResult { slot: 0 }
        ));
    }
    let mut constants = kernel.signature.fixed.clone();
    constants.extend_from_slice(weights);
    let graph = ValidatedGraph::new_with_external_constants(
        stored,
        inputs.len(),
        constants.len(),
        1,
        inputs.len(),
        GraphLimits::default(),
    )
    .unwrap();
    let mut output = ResultBuffer::new(1, inputs.len());
    output.slot_mut(0).fill(f64::NAN);
    graph
        .execute_symbolic_with_constants(inputs, &constants, &mut output)
        .map_err(|error| error.to_string())?;
    Ok(output.into_vec())
}

#[test]
fn bounded_rms_persisted_kernels_use_checked_bml_and_match_rms_oracle() {
    let cases = [
        (vec![1.0], vec![4.0]),
        (vec![0.0], vec![4.0]),
        (vec![-1.0], vec![-4.0]),
        (vec![16.0, -16.0, 0.0, 0.125], vec![16.0, -16.0, 4.0, -0.25]),
        (vec![0.0; 4], vec![-4.0, 4.0, 0.0, 16.0]),
    ];
    let epsilon = 1e-5;
    let limits = KernelLimits {
        fragment_bytes: 256,
        ..Default::default()
    };
    for (input, weights) in cases {
        let kernel = compile_operation_with_rms_bounds(
            Operation::Norm,
            input.len(),
            epsilon,
            OutputLayout::Buffer,
            Some(BOUNDS),
            limits,
        )
        .unwrap();
        let root = (input.iter().map(|value| value * value).sum::<f64>() / input.len() as f64
            + epsilon)
            .sqrt();
        let expected: Vec<_> = input
            .iter()
            .zip(&weights)
            .map(|(input, weight)| input * weight / root)
            .collect();
        close(
            &checked_kernel(&kernel, &input, &weights).unwrap(),
            &expected,
            2e-7,
        );
    }
    // This is the localized old-RMS failure, without claiming the rest of a
    // transformer now accepts arbitrary large intermediate activations.
    let legacy = compile_operation(Operation::Norm, 1, epsilon, limits).unwrap();
    assert!(checked_kernel(&legacy, &[1.0], &[4.0]).is_err());
}

#[test]
fn bounded_rms_v3_artifacts_match_causal_oracle_without_recompilation() {
    let source = symbolic_fixture::fixture(0.25);
    let directory = artifact(&source, BOUNDS);
    let manifest = directory.manifest();
    assert_eq!(manifest.version, 3);
    assert_eq!(manifest.profile, BOUNDED_RMS_PROFILE);
    assert_eq!(manifest.rms_bounds, Some(BOUNDS));
    let loaded = load_directory(&directory.0, None).unwrap();
    assert!(loaded.compiler.mapped_bytes() > 0);
    for kernel in &loaded.kernels {
        for graph in &kernel.graphs {
            assert!(
                graph.const_pool.is_empty(),
                "learned weights must stay external"
            );
            assert!(graph
                .fragments
                .iter()
                .flat_map(|f| &f.ops)
                .all(|op| matches!(
                    op,
                    RpnOp::One
                        | RpnOp::Var(_)
                        | RpnOp::Const(_)
                        | RpnOp::Bml
                        | RpnOp::Pick { .. }
                        | RpnOp::Dup
                        | RpnOp::Drop
                        | RpnOp::Swap
                        | RpnOp::StoreResult { .. }
                )));
        }
    }
    drop(loaded);
    let no_construction = SymbolicLimits {
        max_nodes: 0,
        ..Default::default()
    };
    let mut persisted =
        SymbolicCausalModel::open_artifact(&directory.0, None, no_construction).unwrap();
    let mut lazy =
        SymbolicCausalModel::open_with_rms_bounds(&source.0, SymbolicLimits::default(), BOUNDS)
            .unwrap();
    assert_eq!(persisted.profile(), BOUNDED_RMS_PROFILE);
    assert_eq!(lazy.profile(), BOUNDED_RMS_PROFILE);
    for ids in [&[1][..], &[69][..], &[69, 70][..], &[70, 69, 70][..]] {
        let expected = symbolic_oracle::logits(ids, true);
        close(&persisted.forward(ids).unwrap(), &expected, 2e-7);
        close(&lazy.forward(ids).unwrap(), &expected, 2e-7);
    }
    let mut cache = HashConsedKV::new(1, 1, 2, 8);
    persisted.forward_cached(&[69, 70], &mut cache).unwrap();
    close(
        &persisted.forward_cached(&[69], &mut cache).unwrap(),
        &symbolic_oracle::logits(&[69, 70, 69], true),
        2e-7,
    );
    assert_eq!(cache.current_pos(), 3);
    assert!(persisted.forward_cached(&[70], &mut cache).is_err());
    assert_eq!(cache.current_pos(), 3);
    drop(persisted);
    drop(lazy);
    let moved = directory.0.join("relocated.gguf");
    std::fs::rename(&source.0, &moved).unwrap();
    assert!(SymbolicCausalModel::open_artifact(&directory.0, None, no_construction).is_err());
    let mut moved_model =
        SymbolicCausalModel::open_artifact(&directory.0, Some(&moved), no_construction).unwrap();
    close(
        &moved_model.forward(&[69, 70]).unwrap(),
        &symbolic_oracle::logits(&[69, 70], true),
        2e-7,
    );
}

#[test]
fn bounded_rms_runtime_enforces_actual_operand_bounds_without_advancing_cache() {
    for (embedding, bounds, expected) in [
        (
            2.0,
            RmsBounds {
                input: 1.0,
                weight: 1.0,
            },
            "input 0",
        ),
        (
            0.25,
            RmsBounds {
                input: 1.0,
                weight: 0.25,
            },
            "weight 0",
        ),
    ] {
        let source = symbolic_fixture::fixture(embedding);
        let directory = artifact(&source, bounds);
        let mut models = [
            SymbolicCausalModel::open_with_rms_bounds(&source.0, SymbolicLimits::default(), bounds)
                .unwrap(),
            SymbolicCausalModel::open_artifact(
                &directory.0,
                None,
                SymbolicLimits {
                    max_nodes: 0,
                    ..Default::default()
                },
            )
            .unwrap(),
        ];
        for model in &mut models {
            let mut cache = HashConsedKV::new(1, 1, 2, 8);
            let error = model.forward_cached(&[69], &mut cache).unwrap_err();
            assert!(error.contains("rms_norm:"), "{error}");
            assert!(error.contains(expected), "{error}");
            assert_eq!(cache.current_pos(), 0);
            assert!(!model.executed_operations().contains(&"rms_norm"));
        }
    }
}

#[test]
fn bounded_rms_artifacts_reject_malformed_profiles_bounds_and_literals() {
    let source = symbolic_fixture::fixture(0.25);
    let directory = artifact(&source, BOUNDS);
    let original = directory.manifest();
    for case in 0..9 {
        let mut changed = original.clone();
        match case {
            0 => changed.rms_bounds = None,
            1 => changed.version = 2,
            2 => changed.profile = PROFILE.into(),
            3 => {
                changed.version = 2;
                changed.profile = PROFILE.into();
            }
            4 => changed.rms_bounds.as_mut().unwrap().input = 0.0,
            5 => changed.rms_bounds.as_mut().unwrap().weight = -1.0,
            6 => changed.rms_bounds.as_mut().unwrap().weight = 256.0,
            7 => changed.rms_bounds.as_mut().unwrap().input = 8.0,
            _ => changed.rms_bounds.as_mut().unwrap().weight = 8.0,
        }
        directory.save(&changed);
        assert!(load_directory(&directory.0, None).is_err(), "case {case}");
    }
    let norm = original
        .kernels
        .iter()
        .position(|k| k.operation == Operation::Norm)
        .unwrap();
    for index in 0..original.kernels[norm].fixed.len() {
        let mut changed = original.clone();
        changed.kernels[norm].fixed[index] += 0.125;
        directory.save(&changed);
        assert!(
            load_directory(&directory.0, None).is_err(),
            "fixed literal {index}"
        );
    }
    directory.save(&original);
    let mut malformed = serde_json::to_value(&original).unwrap();
    malformed["rms_bounds"]["unexpected"] = true.into();
    std::fs::write(
        directory.0.join(MANIFEST),
        serde_json::to_vec(&malformed).unwrap(),
    )
    .unwrap();
    assert!(load_directory(&directory.0, None).is_err());
    directory.save(&original);
    assert!(load_directory(&directory.0, None).is_ok());
}

#[test]
fn bounded_rms_is_opt_in_and_legacy_v2_bytes_stay_unchanged() {
    let source = symbolic_fixture::fixture(0.25);
    let directory = Directory::new();
    let manifest = compile_directory(&source.0, &directory.0, 3, KernelLimits::default()).unwrap();
    assert_eq!(manifest.version, 2);
    assert_eq!(manifest.profile, PROFILE);
    assert_eq!(manifest.rms_bounds, None);
    assert!(serde_json::to_value(&manifest)
        .unwrap()
        .get("rms_bounds")
        .is_none());
    let mut historical = SymbolicCausalModel::open_artifact(
        &directory.0,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    close(
        &historical.forward(&[69, 70]).unwrap(),
        &symbolic_oracle::logits(&[69, 70], true),
        2e-7,
    );
    for (operation, width) in [
        (Operation::Norm, 4),
        (Operation::Dot, 4),
        (Operation::Softmax, 3),
        (Operation::Rope, 2),
        (Operation::Swiglu, 1),
        (Operation::Add, 1),
        (Operation::Scale, 1),
    ] {
        let legacy = compile_operation(operation, width, 0.5, KernelLimits::default()).unwrap();
        let explicit = compile_operation_with_rms_bounds(
            operation,
            width,
            0.5,
            OutputLayout::for_operation(operation),
            None,
            KernelLimits::default(),
        )
        .unwrap();
        assert_eq!(legacy.signature, explicit.signature);
        assert_eq!(
            legacy
                .graphs
                .iter()
                .map(BmlGraph::serialize)
                .collect::<Vec<_>>(),
            explicit
                .graphs
                .iter()
                .map(BmlGraph::serialize)
                .collect::<Vec<_>>()
        );
        if operation != Operation::Norm {
            let bounded = compile_operation_with_rms_bounds(
                operation,
                width,
                0.5,
                OutputLayout::for_operation(operation),
                Some(BOUNDS),
                KernelLimits::default(),
            )
            .unwrap();
            assert_eq!(legacy.signature, bounded.signature);
            assert_eq!(
                legacy
                    .graphs
                    .iter()
                    .map(BmlGraph::serialize)
                    .collect::<Vec<_>>(),
                bounded
                    .graphs
                    .iter()
                    .map(BmlGraph::serialize)
                    .collect::<Vec<_>>()
            );
        }
    }
}
