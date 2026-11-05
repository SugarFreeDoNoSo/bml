//! Indexed bounded kernels and persisted complete-profile causal execution.
use bml_compiler::{
    symbolic_artifact::{
        compile_directory_bounded, compile_directory_with_rms_bounds, load_directory, Manifest,
        BOUNDED_PROFILE, BOUNDED_RMS_PROFILE, MANIFEST,
    },
    symbolic_bounded::{compile_operation, BoundedProfile, ProjectionBounds},
    symbolic_kernels::{CompiledKernel, KernelLimits, Operation, OutputLayout, RmsBounds},
    BmlGraph, HashConsedKV, RpnOp,
};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphError, GraphLimits, ValidatedGraph},
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
};
use sha2::{Digest, Sha256};
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
fn bounded_profile_validation_checks_every_declared_family_before_compilation() {
    let profile = BoundedProfile {
        down_projection: Some(ProjectionBounds {
            input: 16.0,
            weight: 1.0,
        }),
        ..BoundedProfile::default()
    };
    profile.validate(32, 0.5).unwrap();
    for (name, invalid) in [
        (
            "RMS input",
            BoundedProfile {
                rms: RmsBounds {
                    input: f64::NAN,
                    ..profile.rms
                },
                ..profile
            },
        ),
        (
            "RMS weight",
            BoundedProfile {
                rms: RmsBounds {
                    weight: f64::NAN,
                    ..profile.rms
                },
                ..profile
            },
        ),
        (
            "projection input",
            BoundedProfile {
                projection_input: f64::NAN,
                ..profile
            },
        ),
        (
            "projection weight",
            BoundedProfile {
                projection_weight: f64::NAN,
                ..profile
            },
        ),
        (
            "attention",
            BoundedProfile {
                attention: f64::NAN,
                ..profile
            },
        ),
        (
            "value",
            BoundedProfile {
                value: f64::NAN,
                ..profile
            },
        ),
        (
            "RoPE",
            BoundedProfile {
                rope: f64::NAN,
                ..profile
            },
        ),
        (
            "gate",
            BoundedProfile {
                gate: f64::NAN,
                ..profile
            },
        ),
        (
            "up",
            BoundedProfile {
                up: f64::NAN,
                ..profile
            },
        ),
        (
            "residual",
            BoundedProfile {
                residual: f64::NAN,
                ..profile
            },
        ),
        (
            "scale",
            BoundedProfile {
                scale_input: f64::NAN,
                ..profile
            },
        ),
        (
            "score",
            BoundedProfile {
                score: f64::NAN,
                ..profile
            },
        ),
        (
            "down projection",
            BoundedProfile {
                down_projection: Some(ProjectionBounds {
                    input: f64::NAN,
                    weight: 1.0,
                }),
                ..profile
            },
        ),
    ] {
        assert!(invalid.validate(32, 0.5).is_err(), "invalid {name}");
    }
    assert!(profile.validate(0, 0.5).is_err());
    assert!(profile.validate(32, 0.0).is_err());
}

fn fixed_bounded_width(
    operation: Operation,
    width: usize,
    inputs: &[f64],
    external: &[f64],
    expected: &[f64],
) {
    let profile = BoundedProfile::default();
    let layout = BoundedProfile::output_layout(operation);
    let signature = profile.signature(operation, width, 0.5, layout).unwrap();
    assert_eq!(signature.inputs, inputs.len());
    assert_eq!(signature.external, external.len());
    let compiled =
        compile_operation(operation, width, 0.5, profile, KernelLimits::default()).unwrap();
    assert_eq!(compiled.signature, signature);
    profile
        .validate_values(operation, width, 0.5, inputs, external)
        .unwrap();
    close(&execute(&compiled, inputs, external), expected, 2e-6);
    // Reject a mismatched public metadata width even when its data would fit
    // the primitive's actual input count. Config guards must not ignore width.
    for invalid in [1, 2, 3] {
        if invalid != width {
            assert!(
                profile.base_signature(operation, invalid, 0.5).is_err(),
                "{operation:?} width={invalid}"
            );
            assert!(profile.signature(operation, invalid, 0.5, layout).is_err());
        }
    }
}

#[test]
fn bounded_rope_configuration_requires_exact_pair_width() {
    fixed_bounded_width(Operation::Rope, 2, &[0.25, -0.5], &[0.0, 1.0], &[0.5, 0.25]);
}

#[test]
fn bounded_swiglu_configuration_requires_exact_scalar_width() {
    fixed_bounded_width(Operation::Swiglu, 1, &[0.0, 4.0], &[], &[0.0]);
}

#[test]
fn bounded_residual_configuration_requires_exact_scalar_width() {
    fixed_bounded_width(Operation::Add, 1, &[1.0, 2.0], &[], &[3.0]);
}

#[test]
fn bounded_scale_configuration_requires_exact_scalar_width() {
    fixed_bounded_width(Operation::Scale, 1, &[1.5, 0.5], &[], &[0.75]);
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
fn bounded_v4_accepts_the_independently_canonical_residual_prefix() {
    const CANONICAL: [f64; 3] = [64.0, 2.0, 6.0];
    let source = symbolic_fixture::fixture(0.25);
    let profile = BoundedProfile {
        residual: 3.0,
        ..BoundedProfile::default()
    };
    let directory = artifact(&source, profile);
    let mut manifest = directory.manifest();
    assert_eq!(manifest.version, 4);
    assert_eq!(manifest.bounded_profile, Some(profile));
    let index = manifest
        .kernels
        .iter()
        .position(|kernel| kernel.operation == Operation::Add && kernel.width == 1)
        .unwrap();
    assert_eq!(manifest.kernels[index].layout, OutputLayout::Scalars);
    assert_eq!(manifest.kernels[index].programs.len(), 1);

    // First admit the writer's complete artifact. Then supply an independently
    // canonical prefix, so sharing a drifted formula between writer and loader
    // cannot turn this compatibility check into a self-comparison.
    let mut residual = {
        let loaded = load_directory(&directory.0, None).unwrap();
        loaded
            .kernels
            .into_iter()
            .find(|kernel| kernel.operation == Operation::Add && kernel.width == 1)
            .unwrap()
    };
    assert_eq!(residual.signature.external, 0);
    residual.signature.fixed = CANONICAL.to_vec();
    for (inputs, expected) in [
        ([-3.0, 3.0], 0.0),
        ([3.0, -3.0], 0.0),
        ([3.0, 3.0], 6.0),
        ([-3.0, -3.0], -6.0),
    ] {
        close(&execute(&residual, &inputs, &[]), &[expected], 2e-6);
    }
    // Persist the same validated program with an empty owned pool. Its
    // descriptor binds the canonical values and records the exact program hash.
    let program = stored(&residual).serialize();
    let path = directory.0.join(format!("kernel_{index}_0.bmlgraph"));
    manifest.kernels[index].fixed = CANONICAL.to_vec();
    manifest.kernels[index].programs[0] = format!("{:x}", Sha256::digest(&program));
    std::fs::write(&path, program).unwrap();
    directory.save(&manifest);
    assert_eq!(
        format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        manifest.kernels[index].programs[0],
    );
    let mut model = SymbolicCausalModel::open_artifact(
        &directory.0,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..SymbolicLimits::default()
        },
    )
    .unwrap();
    assert_eq!(model.profile(), BOUNDED_PROFILE);
    let mut cache = HashConsedKV::new(1, 1, 2, 3);
    let logits = model.forward_cached(&[69, 70], &mut cache).unwrap();
    close(&logits, &symbolic_oracle::logits(&[69, 70], true), 2e-7);
    assert_eq!(cache.current_pos(), 2);
    assert!(model.executed_operations().contains(&"residual_add"));
    drop(model);
    // A genuinely noncanonical prefix remains invalid with the same valid
    // program digest; rejection is about profile identity, not file damage.
    manifest.kernels[index].fixed[2] = 7.0;
    directory.save(&manifest);
    assert!(load_directory(&directory.0, None).is_err());
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

#[test]
fn residual_bound_errors_identify_layer_element_and_operands_without_committing() {
    let source = symbolic_fixture::fixture(0.25);
    let profile = BoundedProfile {
        residual: 1.0 / 256.0,
        ..Default::default()
    };
    let directory = artifact(&source, profile);
    let mut models = [
        SymbolicCausalModel::open_bounded(&source.0, SymbolicLimits::default(), profile).unwrap(),
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
        assert!(
            error.contains("blk.0.attention: residual element")
                && error.contains("left ")
                && error.contains("right ")
                && error.contains("residual_add"),
            "{error}"
        );
        assert_eq!(cache.current_pos(), 0);
    }
}

#[test]
fn rms_bound_errors_identify_tensor_and_operand_maxima_without_committing() {
    let source = symbolic_fixture::fixture(0.25);
    let profile = BoundedProfile {
        rms: bml_compiler::symbolic_kernels::RmsBounds {
            input: 16.0,
            weight: 1.0 / 256.0,
        },
        ..Default::default()
    };
    let directory = artifact(&source, profile);
    let mut models = [
        SymbolicCausalModel::open_bounded(&source.0, SymbolicLimits::default(), profile).unwrap(),
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
        assert!(
            error.contains("blk.0.attn_norm.weight")
                && error.contains("input max_abs")
                && error.contains("weight max_abs")
                && error.contains("bounded RMS weight"),
            "{error}"
        );
        assert_eq!(cache.current_pos(), 0);
    }
}

/// Requires a local TinyLlama GGUF; this checks SwiGLU on native projection
/// values for one token, not full symbolic inference or model parity.
#[test]
#[ignore = "requires BML_TEST_GGUF pointing to a local TinyLlama Q4_0 model"]
fn tinyllama_real_swiglu_inputs_match_checked_bml_oracle() {
    use bml_compiler::gguf_compiler::InferenceCompiler;
    let source = std::env::var_os("BML_TEST_GGUF").expect("BML_TEST_GGUF is required");
    let model = InferenceCompiler::open_mapped(std::path::Path::new(&source)).unwrap();
    let mut cache = HashConsedKV::try_new(
        model.config().n_layers,
        model.n_kv_heads(),
        model.head_dim(),
        1,
        512 * 1024 * 1024,
    )
    .unwrap();
    let mut pending: Option<(String, Vec<f64>)> = None;
    let mut samples = Vec::new();
    let logits = model
        .forward_cached_with_projection(&[1], &mut cache, &mut |input, name| {
            let output = model.project_rows(input, name, |x, weights| {
                Ok(x.iter()
                    .zip(weights)
                    .map(|(&value, &weight)| value * f64::from(weight))
                    .sum())
            })?;
            if name.ends_with(".ffn_gate.weight") {
                if pending
                    .replace((name.to_string(), output.clone()))
                    .is_some()
                {
                    return Err("overlapping gate projections".into());
                }
            } else if name.ends_with(".ffn_up.weight") {
                let (gate_name, gate) = pending.take().ok_or("up projection without gate")?;
                if gate_name.replace(".ffn_gate.weight", ".ffn_up.weight") != name
                    || gate.len() != output.len()
                {
                    return Err("gate/up projection mismatch".into());
                }
                samples.push((gate_name, gate, output.clone()));
            }
            Ok(output)
        })
        .unwrap();
    assert!(pending.is_none());
    assert_eq!(samples.len(), model.config().n_layers as usize);
    assert_eq!(logits.len(), 32000);
    let profile = BoundedProfile {
        gate: 18.0,
        up: 16.0,
        ..Default::default()
    };
    let kernel = compile_operation(
        Operation::Swiglu,
        1,
        model.rms_norm_epsilon(),
        profile,
        KernelLimits::default(),
    )
    .unwrap();
    let graph = validate(&kernel, graph_limits()).unwrap();
    let mut buffer = ResultBuffer::new(0, 0);
    let mut total = 0usize;
    let mut global_max = 0.0_f64;
    for (name, gates, ups) in samples {
        let mut max_error = 0.0_f64;
        for (index, (&gate, &up)) in gates.iter().zip(&ups).enumerate() {
            profile
                .validate_values(
                    Operation::Swiglu,
                    1,
                    model.rms_norm_epsilon(),
                    &[gate, up],
                    &[],
                )
                .unwrap_or_else(|e| panic!("{name}[{index}]: {e}"));
            let observed = graph
                .execute_indexed_symbolic_with_constants(
                    &[gate, up],
                    &kernel.signature.fixed,
                    &mut buffer,
                )
                .unwrap_or_else(|e| panic!("{name}[{index}]: {e}"));
            let expected = gate / (1.0 + (-gate).exp()) * up;
            let error = (observed - expected).abs();
            max_error = max_error.max(error);
            assert!(
                error < 3e-8,
                "{name}[{index}] error={error}: {observed} != {expected}"
            );
            total += 1;
        }
        global_max = global_max.max(max_error);
        eprintln!("{name} pairs={} max_abs_error={max_error}", gates.len());
    }
    eprintln!(
        "swiglu_layers={} pairs={total} max_abs_error={global_max}",
        model.config().n_layers
    );
}

/// Checks selected rows of every native TinyLlama projection for one BOS token.
/// This is not complete symbolic inference or full-logit parity.
#[test]
#[ignore = "requires BML_TEST_GGUF pointing to a local TinyLlama Q4_0 model"]
fn tinyllama_real_projection_rows_match_checked_bml_oracle() {
    use bml_compiler::{gguf_compiler::InferenceCompiler, symbolic_bounded::ProjectionBounds};
    use std::collections::BTreeMap;
    let source = std::env::var_os("BML_TEST_GGUF").expect("BML_TEST_GGUF is required");
    let model = InferenceCompiler::open_mapped(std::path::Path::new(&source)).unwrap();
    let profile = BoundedProfile {
        projection_input: 64.0,
        projection_weight: 4.0,
        down_projection: Some(ProjectionBounds {
            input: 256.0,
            weight: 1.5,
        }),
        ..Default::default()
    };
    let epsilon = model.rms_norm_epsilon();
    let mut kernels = BTreeMap::new();
    for (name, &(width, _)) in model.tensor_dims() {
        if name.ends_with(".attn_q.weight")
            || name.ends_with(".attn_k.weight")
            || name.ends_with(".attn_v.weight")
            || name.ends_with(".attn_output.weight")
            || name.ends_with(".ffn_gate.weight")
            || name.ends_with(".ffn_up.weight")
            || name.ends_with(".ffn_down.weight")
            || name == model.output_weight_name()
        {
            let op = if name.ends_with(".ffn_down.weight") {
                Operation::DownDot
            } else {
                Operation::Dot
            };
            if let std::collections::btree_map::Entry::Vacant(entry) = kernels.entry((op, width)) {
                let kernel =
                    compile_operation(op, width, epsilon, profile, KernelLimits::default())
                        .unwrap();
                let graph = validate(
                    &kernel,
                    GraphLimits {
                        max_steps: 100_000_000,
                        ..graph_limits()
                    },
                )
                .unwrap();
                entry.insert((kernel, graph));
            }
        }
    }
    let mut cache = HashConsedKV::try_new(
        model.config().n_layers,
        model.n_kv_heads(),
        model.head_dim(),
        1,
        512 * 1024 * 1024,
    )
    .unwrap();
    let mut sampled = 0usize;
    let mut tensors = 0usize;
    let mut global_max = 0.0_f64;
    let logits = model
        .forward_cached_with_projection(&[1], &mut cache, &mut |input, name| {
            let &(width, rows) = model
                .tensor_dims()
                .get(name)
                .ok_or("missing projection dimensions")?;
            if width != input.len() || rows < 3 {
                return Err("unexpected projection shape".into());
            }
            let op = if name.ends_with(".ffn_down.weight") {
                Operation::DownDot
            } else {
                Operation::Dot
            };
            let (kernel, graph) = kernels.get(&(op, width)).ok_or("missing checked kernel")?;
            let mut row = 0usize;
            let mut max_error = 0.0_f64;
            let output = model.project_rows(input, name, |x, weights| {
                let index = row;
                row += 1;
                let expected: f64 = x.iter().zip(weights).map(|(&v, &w)| v * f64::from(w)).sum();
                let selected = index == 0
                    || index == rows / 2
                    || index == rows - 1
                    || (name == "blk.2.ffn_gate.weight" && index == 5475)
                    || (name == "blk.2.ffn_down.weight" && index == 624);
                if selected {
                    let external: Vec<f64> = weights.iter().map(|&v| f64::from(v)).collect();
                    profile
                        .validate_values(op, width, epsilon, x, &external)
                        .map_err(|e| format!("{name} row {index}: {e}"))?;
                    let mut constants = kernel.signature.fixed.clone();
                    constants.extend_from_slice(&external);
                    let mut buffer = ResultBuffer::new(0, 0);
                    let actual = graph
                        .execute_indexed_symbolic_with_constants(x, &constants, &mut buffer)
                        .map_err(|e| format!("{name} row {index}: {e}"))?;
                    let error = (actual - expected).abs();
                    if error >= 3e-7 {
                        return Err(format!(
                            "{name} row {index}: error {error}, {actual} != {expected}"
                        ));
                    }
                    max_error = max_error.max(error);
                    sampled += 1;
                }
                Ok(expected)
            })?;
            if row != rows {
                return Err("projection row count mismatch".into());
            }
            tensors += 1;
            global_max = global_max.max(max_error);
            eprintln!("{name} rows={rows} max_abs_error={max_error}");
            Ok(output)
        })
        .unwrap();
    assert_eq!(logits.len(), 32000);
    assert_eq!(tensors, model.config().n_layers as usize * 7 + 1);
    assert_eq!(sampled, tensors * 3 + 2);
    eprintln!("projection_tensors={tensors} sampled_rows={sampled} max_abs_error={global_max}");
}

/// Checks RoPE on native Q/K values for two positions, not full symbolic inference.
#[test]
#[ignore = "requires BML_TEST_GGUF pointing to a local TinyLlama Q4_0 model"]
fn tinyllama_real_rope_inputs_match_checked_bml_oracle() {
    use bml_compiler::gguf_compiler::InferenceCompiler;
    let source = std::env::var_os("BML_TEST_GGUF").expect("BML_TEST_GGUF is required");
    let model = InferenceCompiler::open_mapped(std::path::Path::new(&source)).unwrap();
    let mut cache = HashConsedKV::try_new(
        model.config().n_layers,
        model.n_kv_heads(),
        model.head_dim(),
        2,
        512 * 1024 * 1024,
    )
    .unwrap();
    let mut samples = Vec::new();
    let mut token = 1u32;
    for position in 0..2u32 {
        let logits = model
            .forward_cached_with_projection(&[token], &mut cache, &mut |input, name| {
                let output = model.project_rows(input, name, |x, weights| {
                    Ok(x.iter()
                        .zip(weights)
                        .map(|(&value, &weight)| value * f64::from(weight))
                        .sum())
                })?;
                if name.ends_with(".attn_q.weight") || name.ends_with(".attn_k.weight") {
                    samples.push((position, name.to_string(), output.clone()));
                }
                Ok(output)
            })
            .unwrap();
        assert_eq!(logits.len(), 32000);
        if position == 0 {
            token = bml_compiler::sampler::sample(&logits, 0.0, 0).unwrap();
            assert_eq!(token, 29958);
        }
    }
    assert_eq!(samples.len(), model.config().n_layers as usize * 4);
    let input_max = samples
        .iter()
        .flat_map(|(_, _, input)| input.iter())
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    eprintln!("rope_input_max_abs={input_max}");
    let violations_16 = samples
        .iter()
        .flat_map(|(_, _, input)| input.as_chunks::<2>().0.iter())
        .filter(|pair| pair.iter().any(|value| value.abs() > 16.0))
        .count();
    assert!(
        violations_16 > 0,
        "expected this two-token model sample to exceed the historical RoPE bound"
    );
    eprintln!("rope_pairs_over_16={violations_16}");
    let profile = BoundedProfile {
        rope: 32.0,
        ..Default::default()
    };
    let epsilon = model.rms_norm_epsilon();
    let kernel = compile_operation(
        Operation::Rope,
        2,
        epsilon,
        profile,
        KernelLimits::default(),
    )
    .unwrap();
    let graph = validate(&kernel, graph_limits()).unwrap();
    let head_dim = model.head_dim() as usize;
    assert!(head_dim > 0 && head_dim.is_multiple_of(2));
    let mut total = 0usize;
    let mut global_max = 0.0_f64;
    let mut rotated_max = 0.0_f64;
    for (position, name, input) in samples {
        assert_eq!(input.len() % head_dim, 0);
        let mut max_error = 0.0_f64;
        for (index, pair) in input.as_chunks::<2>().0.iter().enumerate() {
            let within = (index % (head_dim / 2)) * 2;
            let angle = position as f64
                / model
                    .rope_frequency_base()
                    .powf(within as f64 / head_dim as f64);
            let (sine, cosine) = angle.sin_cos();
            profile
                .validate_values(Operation::Rope, 2, epsilon, pair, &[cosine, sine])
                .unwrap_or_else(|e| {
                    panic!("{name} position={position} pair={index} input={pair:?}: {e}")
                });
            let mut constants = kernel.signature.fixed.clone();
            constants.extend([cosine, sine]);
            let mut buffer = ResultBuffer::new(1, 2);
            graph
                .execute_indexed_symbolic_with_constants(pair, &constants, &mut buffer)
                .unwrap_or_else(|e| panic!("{name} position={position} pair={index}: {e}"));
            let expected = [
                pair[0] * cosine - pair[1] * sine,
                pair[0] * sine + pair[1] * cosine,
            ];
            rotated_max = rotated_max.max(expected[0].abs()).max(expected[1].abs());
            for (&actual, &expected) in buffer.slot(0).iter().zip(&expected) {
                let error = (actual - expected).abs();
                max_error = max_error.max(error);
                assert!(
                    error < 2e-8,
                    "{name} position={position} pair={index}: {actual} != {expected}"
                );
            }
            total += 1;
        }
        global_max = global_max.max(max_error);
        eprintln!(
            "position={position} {name} pairs={} max_abs_error={max_error}",
            input.len() / 2
        );
    }
    eprintln!("rope_rotated_max_abs={rotated_max}");
    eprintln!(
        "rope_samples={} pairs={total} max_abs_error={global_max}",
        model.config().n_layers as usize * 4
    );
}

/// Checks attention arithmetic on native Q/K values and F32 keys for two positions.
#[test]
#[ignore = "requires BML_TEST_GGUF pointing to a local TinyLlama Q4_0 model"]
fn tinyllama_real_attention_inputs_match_checked_bml_oracle() {
    use bml_compiler::gguf_compiler::InferenceCompiler;
    use std::collections::BTreeMap;
    let source = std::env::var_os("BML_TEST_GGUF").expect("BML_TEST_GGUF is required");
    let model = InferenceCompiler::open_mapped(std::path::Path::new(&source)).unwrap();
    let heads = model.config().n_heads as usize;
    let kv_heads = model.n_kv_heads() as usize;
    let dim = model.head_dim() as usize;
    assert_eq!(
        (model.config().n_layers, heads, kv_heads, dim),
        (22, 32, 4, 64)
    );
    let mut cache = HashConsedKV::try_new(
        model.config().n_layers,
        model.n_kv_heads(),
        model.head_dim(),
        2,
        512 * 1024 * 1024,
    )
    .unwrap();
    let mut samples = BTreeMap::new();
    let mut token = 1u32;
    for position in 0..2u32 {
        let logits = model
            .forward_cached_with_projection(&[token], &mut cache, &mut |input, name| {
                let output = model.project_rows(input, name, |x, weights| {
                    Ok(x.iter()
                        .zip(weights)
                        .map(|(&value, &weight)| value * f64::from(weight))
                        .sum())
                })?;
                if (name.ends_with(".attn_q.weight")
                    || name.ends_with(".attn_k.weight")
                    || name.ends_with(".attn_v.weight"))
                    && samples
                        .insert((position, name.to_string()), output.clone())
                        .is_some()
                {
                    return Err("duplicate attention projection".into());
                }
                Ok(output)
            })
            .unwrap();
        assert_eq!(logits.len(), 32000);
        if position == 0 {
            token = bml_compiler::sampler::sample(&logits, 0.0, 0).unwrap();
            assert_eq!(token, 29958);
        }
    }
    assert_eq!(samples.len(), 132);
    let rotate = |input: &[f64], position: u32| -> Vec<f64> {
        assert!(input.len().is_multiple_of(dim));
        let mut output = Vec::with_capacity(input.len());
        for (index, pair) in input.as_chunks::<2>().0.iter().enumerate() {
            let within = (index % (dim / 2)) * 2;
            let angle =
                position as f64 / model.rope_frequency_base().powf(within as f64 / dim as f64);
            let (sine, cosine) = angle.sin_cos();
            output.extend([
                pair[0] * cosine - pair[1] * sine,
                pair[0] * sine + pair[1] * cosine,
            ]);
        }
        output
    };
    let profile = BoundedProfile {
        attention: 20.0,
        rope: 32.0,
        ..Default::default()
    };
    let epsilon = model.rms_norm_epsilon();
    let dot = compile_operation(
        Operation::AttentionDot,
        dim,
        epsilon,
        profile,
        KernelLimits::default(),
    )
    .unwrap();
    let dot_graph = validate(&dot, graph_limits()).unwrap();
    let scale = compile_operation(
        Operation::Scale,
        1,
        epsilon,
        profile,
        KernelLimits::default(),
    )
    .unwrap();
    let scale_graph = validate(&scale, graph_limits()).unwrap();
    let softmax: Vec<_> = (1..=2)
        .map(|width| {
            let kernel = compile_operation(
                Operation::Softmax,
                width,
                epsilon,
                profile,
                KernelLimits::default(),
            )
            .unwrap();
            let graph = validate(&kernel, graph_limits()).unwrap();
            (kernel, graph)
        })
        .collect();
    let weighted: Vec<_> = (1..=2)
        .map(|width| {
            let kernel = compile_operation(
                Operation::WeightedSum,
                width,
                epsilon,
                profile,
                KernelLimits::default(),
            )
            .unwrap();
            let graph = validate(&kernel, graph_limits()).unwrap();
            (kernel, graph)
        })
        .collect();
    let mut weighted_count = 0usize;
    let mut weighted_max = 0.0_f64;
    let mut value_max = 0.0_f64;
    let mut dot_count = 0usize;
    let mut group_count = 0usize;
    let mut dot_max = 0.0_f64;
    let mut scale_max = 0.0_f64;
    let mut softmax_max = 0.0_f64;
    let mut operand_max = 0.0_f64;
    let mut score_max = 0.0_f64;
    for layer in 0..model.config().n_layers {
        let mut layer_dot = 0.0_f64;
        let mut layer_scale = 0.0_f64;
        let mut layer_softmax = 0.0_f64;
        let mut layer_weighted = 0.0_f64;
        let keys: Vec<_> = (0..2u32)
            .map(|position| {
                let name = format!("blk.{layer}.attn_k.weight");
                rotate(samples.get(&(position, name)).unwrap(), position)
                    .into_iter()
                    .map(|value| f64::from(value as f32))
                    .collect::<Vec<_>>()
            })
            .collect();
        let values: Vec<_> = (0..2u32)
            .map(|position| {
                let name = format!("blk.{layer}.attn_v.weight");
                samples
                    .get(&(position, name))
                    .unwrap()
                    .iter()
                    .map(|&value| f64::from(value as f32))
                    .collect::<Vec<_>>()
            })
            .collect();
        for position in 0..2u32 {
            let name = format!("blk.{layer}.attn_q.weight");
            let query = rotate(samples.get(&(position, name)).unwrap(), position);
            for head in 0..heads {
                let q = &query[head * dim..(head + 1) * dim];
                let kv_head = head / (heads / kv_heads);
                let mut bml_scores = Vec::new();
                let mut native_scores = Vec::new();
                for (key_position, key_values) in
                    keys.iter().take(position as usize + 1).enumerate()
                {
                    let key = &key_values[kv_head * dim..(kv_head + 1) * dim];
                    operand_max = q
                        .iter()
                        .chain(key)
                        .fold(operand_max, |max, value| max.max(value.abs()));
                    profile
                        .validate_values(Operation::AttentionDot, dim, epsilon, q, key)
                        .unwrap();
                    let mut constants = dot.signature.fixed.clone();
                    constants.extend_from_slice(key);
                    let actual_dot = dot_graph
                        .execute_indexed_symbolic_with_constants(
                            q,
                            &constants,
                            &mut ResultBuffer::new(0, 0),
                        )
                        .unwrap();
                    let expected_dot: f64 = q.iter().zip(key).map(|(&a, &b)| a * b).sum();
                    let dot_error = (actual_dot - expected_dot).abs();
                    assert!(dot_error < 3e-8, "layer={layer} pos={position} head={head} key={key_position}: dot error {dot_error}");
                    layer_dot = layer_dot.max(dot_error);
                    let scale_value = 1.0 / (dim as f64).sqrt();
                    profile
                        .validate_values(
                            Operation::Scale,
                            1,
                            epsilon,
                            &[actual_dot, scale_value],
                            &[],
                        )
                        .unwrap();
                    let actual_score = scale_graph
                        .execute_indexed_symbolic_with_constants(
                            &[actual_dot, scale_value],
                            &scale.signature.fixed,
                            &mut ResultBuffer::new(0, 0),
                        )
                        .unwrap();
                    let expected_score = expected_dot * scale_value;
                    score_max = score_max.max(expected_score.abs());
                    let scale_error = (actual_score - expected_score).abs();
                    assert!(
                        scale_error < 3e-8,
                        "layer={layer} pos={position} head={head}: scale error {scale_error}"
                    );
                    layer_scale = layer_scale.max(scale_error);
                    bml_scores.push(actual_score);
                    native_scores.push(expected_score);
                    dot_count += 1;
                }
                let maximum = bml_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut context = bml_scores.clone();
                context.push(maximum);
                let (kernel, graph) = &softmax[bml_scores.len() - 1];
                profile
                    .validate_values(Operation::Softmax, bml_scores.len(), epsilon, &context, &[])
                    .unwrap();
                let mut buffer = ResultBuffer::new(1, bml_scores.len());
                graph
                    .execute_indexed_symbolic_with_constants(
                        &context,
                        &kernel.signature.fixed,
                        &mut buffer,
                    )
                    .unwrap();
                let native_max = native_scores
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<_> = native_scores
                    .iter()
                    .map(|score| (score - native_max).exp())
                    .collect();
                let sum: f64 = exp.iter().sum();
                for (&actual, &value) in buffer.slot(0).iter().zip(&exp) {
                    let error = (actual - value / sum).abs();
                    assert!(
                        error < 2e-8,
                        "layer={layer} pos={position} head={head}: softmax error {error}"
                    );
                    layer_softmax = layer_softmax.max(error);
                }
                let probabilities = buffer.slot(0);
                let (weighted_kernel, weighted_graph) = &weighted[bml_scores.len() - 1];
                for component in 0..dim {
                    let channel_values: Vec<_> = values
                        .iter()
                        .take(position as usize + 1)
                        .map(|value| value[kv_head * dim + component])
                        .collect();
                    value_max = channel_values
                        .iter()
                        .fold(value_max, |max, value| max.max(value.abs()));
                    profile
                        .validate_values(
                            Operation::WeightedSum,
                            bml_scores.len(),
                            epsilon,
                            probabilities,
                            &channel_values,
                        )
                        .unwrap();
                    let mut constants = weighted_kernel.signature.fixed.clone();
                    constants.extend_from_slice(&channel_values);
                    let actual = weighted_graph
                        .execute_indexed_symbolic_with_constants(
                            probabilities,
                            &constants,
                            &mut ResultBuffer::new(0, 0),
                        )
                        .unwrap();
                    let expected: f64 = exp
                        .iter()
                        .zip(&channel_values)
                        .map(|(&p, &value)| p / sum * value)
                        .sum();
                    let error = (actual - expected).abs();
                    assert!(error < 2e-8, "layer={layer} pos={position} head={head} component={component}: weighted error {error}");
                    layer_weighted = layer_weighted.max(error);
                    weighted_count += 1;
                }
                group_count += 1;
            }
        }
        dot_max = dot_max.max(layer_dot);
        scale_max = scale_max.max(layer_scale);
        softmax_max = softmax_max.max(layer_softmax);
        weighted_max = weighted_max.max(layer_weighted);
        eprintln!("layer={layer} dot_max_error={layer_dot} scale_max_error={layer_scale} softmax_max_error={layer_softmax} weighted_max_error={layer_weighted}");
    }
    assert_eq!(dot_count, 2112);
    assert_eq!(group_count, 1408);
    assert_eq!(weighted_count, 90112);
    eprintln!("attention_value_max_abs={value_max} weighted_values={weighted_count} weighted_max_error={weighted_max}");
    eprintln!("attention_operand_max_abs={operand_max} attention_score_max_abs={score_max}");
    eprintln!("attention_groups={group_count} dots={dot_count} dot_max_error={dot_max} scale_max_error={scale_max} softmax_max_error={softmax_max}");
}
