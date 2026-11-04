//! Indexed bounded kernels and persisted complete-profile causal execution.
use bml_compiler::{
    symbolic_artifact::{
        compile_directory_bounded, compile_directory_with_rms_bounds, load_directory, Manifest,
        BOUNDED_PROFILE, BOUNDED_RMS_PROFILE, MANIFEST,
    },
    symbolic_bounded::{compile_operation, BoundedProfile},
    symbolic_kernels::{CompiledKernel, KernelLimits, Operation, OutputLayout, RmsBounds},
    BmlGraph, HashConsedKV, RpnOp,
};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
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

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_full_bounded_{}_{}",
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

fn artifact(source: &symbolic_fixture::Fixture, profile: BoundedProfile) -> Directory {
    let output = Directory::new();
    compile_directory_bounded(&source.0, &output.0, 3, profile, KernelLimits::default()).unwrap();
    output
}
fn graph_limits() -> GraphLimits {
    GraphLimits {
        max_ops: 100_000,
        max_steps: 10_000_000,
        max_values: 1_000_000,
    }
}
fn stored(kernel: &CompiledKernel) -> BmlGraph {
    assert_eq!(kernel.graphs.len(), 1);
    let graph = &kernel.graphs[0];
    assert!(
        graph.const_pool.is_empty(),
        "all model values must remain external"
    );
    BmlGraph::deserialize(&graph.serialize(), graph.threshold).unwrap()
}
fn validate(kernel: &CompiledKernel, limits: GraphLimits) -> Result<ValidatedGraph, GraphError> {
    let buffered = kernel.layout.uses_buffer();
    ValidatedGraph::new_with_external_constants(
        stored(kernel),
        kernel.signature.inputs,
        kernel.signature.fixed.len() + kernel.signature.external,
        usize::from(buffered),
        if buffered {
            kernel.signature.outputs
        } else {
            0
        },
        limits,
    )
}
fn execute(kernel: &CompiledKernel, input: &[f64], external: &[f64]) -> Vec<f64> {
    execute_checked(kernel, input, external).unwrap()
}
fn execute_checked(
    kernel: &CompiledKernel,
    input: &[f64],
    external: &[f64],
) -> Result<Vec<f64>, GraphError> {
    assert_eq!(input.len(), kernel.signature.inputs);
    assert_eq!(external.len(), kernel.signature.external);
    let graph = validate(kernel, graph_limits())?;
    let mut constants = kernel.signature.fixed.clone();
    constants.extend_from_slice(external);
    let mut buffer = if kernel.layout.uses_buffer() {
        ResultBuffer::from_vec(
            1,
            kernel.signature.outputs,
            vec![f64::NAN; kernel.signature.outputs],
        )
        .unwrap()
    } else {
        ResultBuffer::new(0, 0)
    };
    let value = graph.execute_indexed_symbolic_with_constants(input, &constants, &mut buffer)?;
    Ok(if kernel.layout.uses_buffer() {
        buffer.into_vec()
    } else {
        vec![value]
    })
}
fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() < tolerance,
            "output {index}: {actual} != {expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn indexed_dot_roles_preserve_external_context_and_independent_signed_results() {
    let profile = BoundedProfile::default();
    for (operation, left, right) in [
        (Operation::Dot, 64.0, 1.0),
        (Operation::AttentionDot, 8.0, 8.0),
        (Operation::WeightedSum, 0.5, 8.0),
    ] {
        for width in [1, 32, 288, 768] {
            let kernel =
                compile_operation(operation, width, 0.5, profile, KernelLimits::default()).unwrap();
            assert_eq!(kernel.signature.inputs, width);
            assert_eq!(kernel.signature.external, width);
            assert_eq!(kernel.signature.outputs, 1);
            assert_eq!(kernel.layout, OutputLayout::Scalars);
            let input: Vec<_> = (0..width)
                .map(|i| if i % 2 == 0 { left } else { -left })
                .collect();
            let weights: Vec<_> = (0..width)
                .map(|i| if i % 4 < 2 { right } else { right / 4.0 })
                .collect();
            profile
                .validate_values(operation, width, 0.5, &input, &weights)
                .unwrap();
            let reference: f64 = input.iter().zip(&weights).map(|(x, w)| x * w).sum();
            close(&execute(&kernel, &input, &weights), &[reference], 2e-6);
            let different: Vec<_> = weights.iter().map(|value| -value / 2.0).collect();
            let reference: f64 = input.iter().zip(&different).map(|(x, w)| x * w).sum();
            close(&execute(&kernel, &input, &different), &[reference], 2e-6);
            let ops: Vec<_> = kernel.graphs[0]
                .fragments
                .iter()
                .flat_map(|f| &f.ops)
                .collect();
            assert_eq!(
                ops.iter()
                    .filter(|op| matches!(op, RpnOp::Loop { .. }))
                    .count(),
                1
            );
            assert!(ops
                .iter()
                .any(|op| matches!(op, RpnOp::ContextVarIndexed { base: 0 })));
            assert!(ops
                .iter()
                .any(|op| matches!(op,RpnOp::ContextConstIndexed {base}
                if *base as usize == kernel.signature.fixed.len())));
            assert!(ops.iter().all(|op| matches!(
                op,
                RpnOp::One
                    | RpnOp::Const(_)
                    | RpnOp::Bml
                    | RpnOp::Pick { .. }
                    | RpnOp::Dup
                    | RpnOp::Drop
                    | RpnOp::Swap
                    | RpnOp::Loop { .. }
                    | RpnOp::ContextVarIndexed { .. }
                    | RpnOp::ContextConstIndexed { .. }
            )));
        }
    }
}

#[test]
fn indexed_dot_storage_is_constant_and_executed_work_has_exact_linear_budgets() {
    let profile = BoundedProfile::default();
    assert!(
        compile_operation(
            Operation::Dot,
            u32::MAX as usize,
            0.5,
            profile,
            KernelLimits::default(),
        )
        .is_err(),
        "indexed external constants exceed the u32 namespace after their fixed prefix"
    );
    let mut measurements = Vec::new();
    for width in [1, 32, 288, 768] {
        let kernel =
            compile_operation(Operation::Dot, width, 0.5, profile, KernelLimits::default())
                .unwrap();
        let graph = stored(&kernel);
        let ops: usize = graph.fragments.iter().map(|f| f.ops.len()).sum();
        let body = graph
            .fragments
            .iter()
            .flat_map(|f| &f.ops)
            .find_map(|op| match op {
                RpnOp::Loop { count, body_len } => {
                    assert_eq!(*count as usize, width);
                    Some(*body_len as usize)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(ops, body + 2);
        let validated = validate(&kernel, graph_limits()).unwrap();
        let steps = 2 + width * (body + 1);
        assert_eq!(validated.execution_steps(), steps);
        assert!(validate(
            &kernel,
            GraphLimits {
                max_steps: steps,
                ..graph_limits()
            }
        )
        .is_ok());
        assert!(matches!(
            validate(
                &kernel,
                GraphLimits {
                    max_steps: steps - 1,
                    ..graph_limits()
                }
            ),
            Err(GraphError::Limit)
        ));
        assert!(compile_operation(
            Operation::Dot,
            width,
            0.5,
            profile,
            KernelLimits {
                max_ops: ops - 1,
                ..Default::default()
            }
        )
        .is_err());
        assert!(compile_operation(
            Operation::Dot,
            width,
            0.5,
            profile,
            KernelLimits {
                max_output_ops: ops - 1,
                ..Default::default()
            }
        )
        .is_err());
        assert!(compile_operation(
            Operation::Dot,
            width,
            0.5,
            profile,
            KernelLimits {
                max_ops: ops,
                max_output_ops: ops,
                max_programs: 1,
                ..Default::default()
            }
        )
        .is_ok());
        measurements.push((graph.serialize().len(), ops, validated.stack_capacity()));
    }
    assert!(measurements.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(compile_operation(
        Operation::Dot,
        32,
        0.5,
        profile,
        KernelLimits {
            max_nodes: 0,
            ..Default::default()
        }
    )
    .is_err());
    assert!(compile_operation(
        Operation::Dot,
        32,
        0.5,
        profile,
        KernelLimits {
            fragment_bytes: 256,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn complete_bounded_role_bounds_and_explicit_softmax_shift_are_enforced() {
    let profile = BoundedProfile::default();
    for (role, input, weight) in [
        (Operation::Dot, 128.1, 1.0),
        (Operation::Dot, 1.0, 2.1),
        (Operation::AttentionDot, 16.1, 1.0),
        (Operation::AttentionDot, 1.0, 16.1),
        (Operation::WeightedSum, 1.1, 1.0),
        (Operation::WeightedSum, 0.5, 16.1),
    ] {
        assert!(profile
            .validate_values(role, 1, 0.5, &[input], &[weight])
            .is_err());
    }
    assert!(profile
        .validate_values(Operation::Dot, 1, 0.5, &[8.0], &[8.0])
        .is_err());
    assert!(profile
        .validate_values(Operation::AttentionDot, 1, 0.5, &[8.0], &[8.0])
        .is_ok());
    assert!(profile
        .validate_values(Operation::WeightedSum, 1, 0.5, &[8.0], &[8.0])
        .is_err());
    let scores = [-16.0, -16.0, -8.0, -12.0];
    let kernel =
        compile_operation(Operation::Softmax, 4, 0.5, profile, KernelLimits::default()).unwrap();
    assert_eq!(kernel.signature.inputs, 5);
    assert_eq!(kernel.signature.external, 0);
    let mut inputs = scores.to_vec();
    inputs.push(-8.0);
    profile
        .validate_values(Operation::Softmax, 4, 0.5, &inputs, &[])
        .unwrap();
    let exponentials: Vec<_> = scores.iter().map(|value| (value + 8.0).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    let expected: Vec<_> = exponentials.iter().map(|value| value / total).collect();
    close(&execute(&kernel, &inputs, &[]), &expected, 2e-9);
    inputs[4] = 16.0;
    assert!(profile
        .validate_values(Operation::Softmax, 4, 0.5, &inputs, &[])
        .is_err());
    assert!(profile
        .validate_values(Operation::Softmax, 4, 0.5, &scores, &[])
        .is_err());
}

#[test]
fn full_bounded_v4_artifacts_match_independent_causal_oracle_without_recompilation() {
    let source = symbolic_fixture::fixture(0.25);
    let profile = BoundedProfile::default();
    let directory = artifact(&source, profile);
    let manifest = directory.manifest();
    assert_eq!(manifest.version, 4);
    assert_eq!(manifest.profile, BOUNDED_PROFILE);
    assert_eq!(manifest.rms_bounds, None);
    assert_eq!(manifest.bounded_profile, Some(profile));
    for descriptor in &manifest.kernels {
        let expected = profile
            .signature(
                descriptor.operation,
                descriptor.width,
                manifest.rms_norm_epsilon,
                descriptor.layout,
            )
            .unwrap();
        assert_eq!(
            descriptor.fixed, expected.fixed,
            "roundtrip literal binding {:?} width {}",
            descriptor.operation, descriptor.width
        );
    }
    for role in [
        Operation::Dot,
        Operation::AttentionDot,
        Operation::WeightedSum,
    ] {
        assert!(manifest
            .kernels
            .iter()
            .any(|kernel| kernel.operation == role));
    }
    let loaded = load_directory(&directory.0, None).unwrap();
    assert!(loaded.compiler.mapped_bytes() > 0);
    assert!(loaded
        .kernels
        .iter()
        .flat_map(|k| &k.graphs)
        .all(|g| g.const_pool.is_empty()));
    drop(loaded);
    let no_construction = SymbolicLimits {
        max_nodes: 0,
        ..Default::default()
    };
    let mut persisted =
        SymbolicCausalModel::open_artifact(&directory.0, None, no_construction).unwrap();
    let mut lazy =
        SymbolicCausalModel::open_bounded(&source.0, SymbolicLimits::default(), profile).unwrap();
    assert_eq!(persisted.profile(), BOUNDED_PROFILE);
    assert_eq!(lazy.profile(), BOUNDED_PROFILE);
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
    for family in [
        "dot",
        "attention_dot",
        "weighted_sum",
        "rms_norm",
        "rope",
        "softmax",
        "swiglu",
        "residual_add",
        "scale",
    ] {
        assert!(
            persisted.executed_operations().contains(&family),
            "{family}"
        );
    }
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
fn full_bounded_runtime_rejects_each_dot_role_without_committing_cache() {
    let source = symbolic_fixture::fixture(0.25);
    for (role, profile) in [
        BoundedProfile {
            projection_input: 1.0 / 256.0,
            ..Default::default()
        },
        BoundedProfile {
            projection_weight: 0.125,
            ..Default::default()
        },
        BoundedProfile {
            attention: 1.0 / 256.0,
            ..Default::default()
        },
        BoundedProfile {
            value: 1.0 / 256.0,
            ..Default::default()
        },
        BoundedProfile {
            down_projection: Some(bml_compiler::symbolic_bounded::ProjectionBounds {
                input: 128.0,
                weight: 1.0 / 256.0,
            }),
            ..Default::default()
        },
    ]
    .into_iter()
    .enumerate()
    {
        let directory = artifact(&source, profile);
        let mut models = [
            SymbolicCausalModel::open_bounded(&source.0, SymbolicLimits::default(), profile)
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
            assert!(error.contains("bounded dot operand"), "{error}");
            if role < 2 {
                let expected = if role == 0 { "input" } else { "weight" };
                assert!(
                    error.contains(expected) && error.contains("blk.0.") && error.contains("row "),
                    "{error}"
                );
            }
            if role == 4 {
                assert!(
                    error.contains("ffn_down.weight") && error.contains("weight"),
                    "{error}"
                );
            }
            assert_eq!(cache.current_pos(), 0);
        }
    }
}

#[test]
fn full_bounded_artifacts_reject_malformed_profile_literals_layout_and_roles() {
    let source = symbolic_fixture::fixture(0.25);
    let directory = artifact(&source, BoundedProfile::default());
    let original = directory.manifest();
    for case in 0..9 {
        let mut changed = original.clone();
        match case {
            0 => changed.bounded_profile = None,
            1 => {
                changed.rms_bounds = Some(RmsBounds {
                    input: 16.0,
                    weight: 16.0,
                })
            }
            2 => changed.version = 3,
            3 => changed.profile = BOUNDED_RMS_PROFILE.into(),
            4 => changed.bounded_profile.as_mut().unwrap().projection_weight = 0.0,
            5 => changed.bounded_profile.as_mut().unwrap().attention = 32.0,
            6 => changed.bounded_profile.as_mut().unwrap().score = 17.0,
            7 => changed.bounded_profile.as_mut().unwrap().projection_input = 64.0,
            _ => changed.bounded_profile.as_mut().unwrap().value = 8.0,
        }
        directory.save(&changed);
        assert!(
            load_directory(&directory.0, None).is_err(),
            "profile case {case}"
        );
    }
    for (kernel, descriptor) in original.kernels.iter().enumerate() {
        for literal in 0..descriptor.fixed.len() {
            let mut changed = original.clone();
            changed.kernels[kernel].fixed[literal] += 0.125;
            directory.save(&changed);
            assert!(
                load_directory(&directory.0, None).is_err(),
                "kernel{kernel} literal{literal}"
            );
        }
        let mut changed = original.clone();
        changed.kernels[kernel].layout = if descriptor.layout == OutputLayout::Scalars {
            OutputLayout::Buffer
        } else {
            OutputLayout::Scalars
        };
        directory.save(&changed);
        assert!(
            load_directory(&directory.0, None).is_err(),
            "layout {kernel}"
        );
    }
    let mut changed = original.clone();
    let index = changed
        .kernels
        .iter()
        .position(|k| k.operation == Operation::AttentionDot)
        .unwrap();
    changed.kernels[index].operation = Operation::Dot;
    directory.save(&changed);
    assert!(load_directory(&directory.0, None).is_err());
    let mut malformed = serde_json::to_value(&original).unwrap();
    malformed["bounded_profile"]["unknown"] = true.into();
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
fn bounded_v3_artifacts_remain_separate_from_the_new_indexed_profile() {
    let source = symbolic_fixture::fixture(0.25);
    let directory = Directory::new();
    let manifest = compile_directory_with_rms_bounds(
        &source.0,
        &directory.0,
        3,
        RmsBounds {
            input: 16.0,
            weight: 16.0,
        },
        KernelLimits::default(),
    )
    .unwrap();
    assert_eq!(manifest.version, 3);
    assert_eq!(manifest.profile, BOUNDED_RMS_PROFILE);
    assert_eq!(manifest.bounded_profile, None);
    assert!(serde_json::to_value(&manifest)
        .unwrap()
        .get("bounded_profile")
        .is_none());
    let loaded = load_directory(&directory.0, None).unwrap();
    assert!(loaded.kernels.iter().all(|kernel| !matches!(
        kernel.operation,
        Operation::AttentionDot | Operation::WeightedSum
    )));
    assert!(loaded
        .kernels
        .iter()
        .flat_map(|k| &k.graphs)
        .flat_map(|g| &g.fragments)
        .flat_map(|f| &f.ops)
        .all(|op| !matches!(
            op,
            RpnOp::ContextVarIndexed { .. }
                | RpnOp::ContextConstIndexed { .. }
                | RpnOp::Loop { .. }
        )));
    drop(loaded);
    let mut model = SymbolicCausalModel::open_artifact(
        &directory.0,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    close(
        &model.forward(&[69, 70, 69]).unwrap(),
        &symbolic_oracle::logits(&[69, 70, 69], true),
        2e-7,
    );
}

#[test]
fn swiglu_domain_error_identifies_layer_operand_and_preserves_position() {
    let source = symbolic_fixture::fixture(0.25);
    for (profile, role) in [
        (
            BoundedProfile {
                gate: 1.0 / 256.0,
                ..Default::default()
            },
            "gate value",
        ),
        (
            BoundedProfile {
                up: 1.0 / 256.0,
                ..Default::default()
            },
            "up value",
        ),
    ] {
        let mut model =
            SymbolicCausalModel::open_bounded(&source.0, SymbolicLimits::default(), profile)
                .unwrap();
        let mut cache = HashConsedKV::new(1, 1, 2, 8);
        let error = model.forward_cached(&[69], &mut cache).unwrap_err();
        assert!(error.contains("blk.0.ffn_swiglu index "), "{error}");
        assert!(
            error.contains(role) && error.contains("exceeds declaration"),
            "{error}"
        );
        assert_eq!(cache.current_pos(), 0);
    }
}

#[test]
fn persisted_extended_swiglu_executes_checked_bml_across_boundary() {
    let source = symbolic_fixture::fixture(0.25);
    let profile = BoundedProfile {
        gate: 18.0,
        up: 16.0,
        ..Default::default()
    };
    let directory = artifact(&source, profile);
    let loaded = load_directory(&directory.0, None).unwrap();
    let kernel = loaded
        .kernels
        .iter()
        .find(|kernel| kernel.operation == Operation::Swiglu)
        .unwrap();
    assert_eq!(kernel.signature.fixed.len(), 25);
    for gate in [
        -18.0_f64,
        -17.2,
        -16.0,
        -1.0,
        0.0,
        1.0,
        16.0,
        17.196_898_568_148_562,
        18.0,
    ] {
        for up in [-16.0, -1.0, 0.0, 1.0, 16.0] {
            profile
                .validate_values(Operation::Swiglu, 1, 0.5, &[gate, up], &[])
                .unwrap();
            let actual = execute(kernel, &[gate, up], &[])[0];
            let expected = gate / (1.0 + (-gate).exp()) * up;
            assert!(
                (actual - expected).abs() < 3e-8,
                "{gate}, {up}: {actual} != {expected}"
            );
        }
    }
    assert!(profile
        .validate_values(Operation::Swiglu, 1, 0.5, &[18.01, 0.0], &[])
        .is_err());
    let legacy = compile_operation(
        Operation::Swiglu,
        1,
        0.5,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    assert_eq!(legacy.signature.fixed.len(), 24);
}

#[test]
fn persisted_down_projection_has_independent_bounds_and_checked_execution() {
    use bml_compiler::symbolic_bounded::ProjectionBounds;
    let source = symbolic_fixture::fixture(0.25);
    let profile = BoundedProfile {
        projection_input: 64.0,
        projection_weight: 4.0,
        down_projection: Some(ProjectionBounds {
            input: 256.0,
            weight: 1.5,
        }),
        ..Default::default()
    };
    let directory = artifact(&source, profile);
    let loaded = load_directory(&directory.0, None).unwrap();
    let kernel = loaded
        .kernels
        .iter()
        .find(|kernel| kernel.operation == Operation::DownDot)
        .unwrap();
    let mut input = vec![0.0; kernel.width];
    let mut weights = vec![0.0; kernel.width];
    input[0] = 167.621_176_174_126_73;
    weights[0] = 1.0;
    profile
        .validate_values(Operation::DownDot, kernel.width, 0.5, &input, &weights)
        .unwrap();
    assert!(profile
        .validate_values(Operation::Dot, kernel.width, 0.5, &input, &weights)
        .is_err());
    let actual = execute(kernel, &input, &weights)[0];
    assert!((actual - input[0]).abs() < 1e-7, "{actual}");
    for value in [
        -256.0,
        -167.621_176_174_126_73,
        -1.0,
        0.0,
        1.0,
        167.621_176_174_126_73,
        256.0,
    ] {
        for weight in [-1.5, -0.5, 0.0, 0.5, 1.5] {
            input[0] = value;
            weights[0] = weight;
            profile
                .validate_values(Operation::DownDot, kernel.width, 0.5, &input, &weights)
                .unwrap();
            let result = execute_checked(kernel, &input, &weights);
            if value.abs() == 256.0 && value * weight == 384.0 {
                assert!(
                    matches!(result, Err(GraphError::Numerical)),
                    "input={value} weight={weight}: {result:?}"
                );
                continue;
            }
            let actual = result.unwrap()[0];
            let expected = value * weight;
            assert!(
                (actual - expected).abs() < 1e-7,
                "input={value} weight={weight}: {actual} != {expected}"
            );
        }
    }
    weights[0] = 1.5001;
    assert!(profile
        .validate_values(Operation::DownDot, kernel.width, 0.5, &input, &weights)
        .is_err());
    let old = serde_json::to_value(BoundedProfile::default()).unwrap();
    assert!(old.get("down_projection").is_none());
    assert_eq!(
        serde_json::from_value::<BoundedProfile>(old)
            .unwrap()
            .down_projection,
        None
    );
}
