use bml_compiler::{
    symbolic_artifact::{
        compile_directory, compile_directory_with_rms_bounds, load_directory, Manifest, MANIFEST,
    },
    symbolic_kernels::{
        compile_operation_with_layout, KernelLimits, Operation, OutputLayout, RmsBounds,
    },
    BmlGraph, Fragment, HashConsedKV, RpnOp,
};
use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_symbolic_artifact_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn manifest(&self) -> Manifest {
        serde_json::from_slice(&std::fs::read(self.0.join(MANIFEST)).unwrap()).unwrap()
    }
    fn save(&self, m: &Manifest) {
        std::fs::write(self.0.join(MANIFEST), serde_json::to_vec(m).unwrap()).unwrap();
    }
    fn replace_program(&self, kernel: usize, ops: Vec<RpnOp>) {
        let mut m = self.manifest();
        let graph = BmlGraph {
            fragments: vec![Fragment { ops }],
            const_pool: vec![],
            threshold: m.fragment_bytes,
        };
        let bytes = graph.serialize();
        m.kernels[kernel].programs[0] = format!("{:x}", Sha256::digest(&bytes));
        std::fs::write(self.0.join(format!("kernel_{kernel}_0.bmlgraph")), bytes).unwrap();
        self.save(&m);
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn artifact(source: &symbolic_fixture::Fixture) -> Directory {
    let dir = Directory::new();
    compile_directory(
        &source.0,
        &dir.0,
        3,
        KernelLimits {
            fragment_bytes: 256,
            ..Default::default()
        },
    )
    .unwrap();
    dir
}

#[test]
fn aot_runtime_program_and_opcode_budgets_accept_exact_totals() {
    let source = symbolic_fixture::fixture(0.25);
    let dir = artifact(&source);
    let loaded = load_directory(&dir.0, None).unwrap();
    let programs: usize = loaded
        .kernels
        .iter()
        .map(|kernel| kernel.graphs.len())
        .sum();
    let ops: usize = loaded
        .kernels
        .iter()
        .flat_map(|kernel| &kernel.graphs)
        .flat_map(|graph| &graph.fragments)
        .map(|fragment| fragment.ops.len())
        .sum();
    assert!(programs > 1 && ops > 1);
    drop(loaded);
    let limits = SymbolicLimits {
        max_nodes: 0,
        max_cached_ops: ops,
        max_programs: programs,
        ..SymbolicLimits::default()
    };
    let mut exact = SymbolicCausalModel::open_artifact(&dir.0, None, limits).unwrap();
    let logits = exact.forward(&[69]).unwrap();
    assert_eq!(logits.len(), 260);
    assert!(logits.iter().all(|value| value.is_finite()));
    for insufficient in [
        SymbolicLimits {
            max_programs: programs - 1,
            ..limits
        },
        SymbolicLimits {
            max_cached_ops: ops - 1,
            ..limits
        },
    ] {
        let error = SymbolicCausalModel::open_artifact(&dir.0, None, insufficient)
            .err()
            .unwrap();
        assert_eq!(error, "symbolic artifact runtime budget exceeded");
    }
}

fn prepend_discarded_overflow(dir: &Directory, operation: Operation) {
    let mut manifest = dir.manifest();
    let index = manifest
        .kernels
        .iter()
        .position(|kernel| kernel.operation == operation)
        .unwrap();
    let path = dir.0.join(format!("kernel_{index}_0.bmlgraph"));
    let mut graph =
        BmlGraph::deserialize(&std::fs::read(&path).unwrap(), manifest.fragment_bytes).unwrap();
    // Pure BML basis: B(1,B(B(1,1),1)) = 0, then B(1,0) = +inf.
    // Drop discards that IEEE intermediate without changing any stored output.
    let prefix = [
        RpnOp::One,
        RpnOp::One,
        RpnOp::One,
        RpnOp::One,
        RpnOp::Bml,
        RpnOp::One,
        RpnOp::Bml,
        RpnOp::Bml,
        RpnOp::Bml,
        RpnOp::Drop,
    ];
    let capacity = manifest.fragment_bytes / std::mem::size_of::<RpnOp>();
    assert!(capacity > 0);
    graph.fragments.splice(
        0..0,
        prefix
            .chunks(capacity)
            .map(|ops| Fragment { ops: ops.to_vec() }),
    );
    let bytes = graph.serialize();
    manifest.kernels[index].programs[0] = format!("{:x}", Sha256::digest(&bytes));
    std::fs::write(path, bytes).unwrap();
    dir.save(&manifest);
}

#[test]
fn persisted_v2_and_v3_keep_per_family_intermediate_checks() {
    let source = symbolic_fixture::fixture(0.25);
    for (bounded_rms, operation, rejects) in [
        (false, Operation::Norm, false),
        (true, Operation::Dot, false),
        (true, Operation::Norm, true),
    ] {
        let dir = Directory::new();
        let kernel_limits = KernelLimits {
            fragment_bytes: 256,
            ..KernelLimits::default()
        };
        if bounded_rms {
            compile_directory_with_rms_bounds(
                &source.0,
                &dir.0,
                3,
                RmsBounds {
                    input: 16.0,
                    weight: 16.0,
                },
                kernel_limits,
            )
            .unwrap();
        } else {
            compile_directory(&source.0, &dir.0, 3, kernel_limits).unwrap();
        }
        let limits = SymbolicLimits {
            max_nodes: 0,
            ..SymbolicLimits::default()
        };
        let mut baseline = SymbolicCausalModel::open_artifact(&dir.0, None, limits).unwrap();
        let expected: Vec<_> = baseline
            .forward(&[69])
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        drop(baseline);
        prepend_discarded_overflow(&dir, operation);
        let mut modified = SymbolicCausalModel::open_artifact(&dir.0, None, limits).unwrap();
        let result = modified.forward(&[69]);
        if rejects {
            assert!(
                result.is_err(),
                "v3 RMS must check the discarded invalid intermediate"
            );
        } else {
            let actual: Vec<_> = result.unwrap().into_iter().map(f64::to_bits).collect();
            assert_eq!(
                actual,
                expected,
                "profile v{} {operation:?}",
                if bounded_rms { 3 } else { 2 }
            );
        }
    }
}

#[test]
fn symbolic_kernel_table_limit_is_inclusive_before_plan_validation() {
    let source = symbolic_fixture::fixture(0.25);
    let dir = artifact(&source);
    let original = dir.manifest();
    assert_eq!(original.kernels[0].programs.len(), 1);
    for count in [4096, 4097] {
        let mut manifest = original.clone();
        manifest.kernels = vec![original.kernels[0].clone(); count];
        dir.save(&manifest);
        assert!(std::fs::metadata(dir.0.join(MANIFEST)).unwrap().len() < 16 * 1024 * 1024);
        let error = load_directory(&dir.0, None)
            .err()
            .expect("duplicated kernels do not match the source operation plan");
        let expected = if count == 4096 {
            "symbolic operation plan mismatch"
        } else {
            "unsupported symbolic artifact profile or size"
        };
        assert_eq!(error, expected, "kernel count={count}");
    }
    dir.save(&original);
    assert!(load_directory(&dir.0, None).is_ok());
}

#[test]
fn symbolic_aot_executes_saved_programs_without_lazy_compilation_or_weight_copies() {
    let source = symbolic_fixture::fixture(0.25);
    let dir = artifact(&source);
    let manifest = dir.manifest();
    assert_eq!(manifest.version, 2);
    assert_eq!(manifest.profile, "llama-symbolic-bml-f64-f32kv-v2");
    for kernel in &manifest.kernels {
        let expected_layout = if matches!(
            kernel.operation,
            Operation::Norm | Operation::Softmax | Operation::Rope
        ) {
            OutputLayout::Buffer
        } else {
            OutputLayout::Scalars
        };
        assert_eq!(kernel.layout, expected_layout);
        assert_eq!(
            kernel.programs.len(),
            1,
            "{:?} width={}",
            kernel.operation,
            kernel.width
        );
        assert_eq!(
            kernel.fixed,
            kernel
                .operation
                .signature_for_layout(kernel.width, manifest.rms_norm_epsilon, expected_layout)
                .unwrap()
                .fixed
        );
    }
    let files: usize = manifest.kernels.iter().map(|k| k.programs.len()).sum();
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), files + 1);
    let loaded = load_directory(&dir.0, None).unwrap();
    assert!(loaded
        .kernels
        .iter()
        .flat_map(|k| &k.graphs)
        .all(|g| g.const_pool.is_empty()));
    assert!(loaded
        .kernels
        .iter()
        .flat_map(|k| &k.graphs)
        .any(|g| g.fragments.len() > 1));
    drop(loaded);
    // A zero construction-node budget would reject every lazy template build.
    let limits = SymbolicLimits {
        max_nodes: 0,
        ..SymbolicLimits::default()
    };
    let mut aot = SymbolicCausalModel::open_artifact(&dir.0, None, limits).unwrap();
    let mut lazy = SymbolicCausalModel::open(&source.0, SymbolicLimits::default()).unwrap();
    assert_eq!(
        aot.forward(&[69, 70, 69]).unwrap(),
        lazy.forward(&[69, 70, 69]).unwrap()
    );
    let mut cache = HashConsedKV::new(1, 1, 2, 8);
    aot.forward_cached(&[69, 70], &mut cache).unwrap();
    assert_eq!(
        aot.forward_cached(&[69], &mut cache).unwrap(),
        lazy.forward(&[69, 70, 69]).unwrap()
    );
    assert!(aot.forward_cached(&[70], &mut cache).is_err());
    assert_eq!(cache.current_pos(), 3);
    assert!(aot.forward(&[69, 70, 69, 70]).is_err());
    drop(aot);
    drop(lazy);
    // Replace the residual program with constant one and update its integrity
    // hash. A loader that regenerated the original addition would fail this.
    let add = manifest
        .kernels
        .iter()
        .position(|k| k.operation == Operation::Add)
        .unwrap();
    dir.replace_program(add, vec![RpnOp::One]);
    let mut changed = SymbolicCausalModel::open_artifact(&dir.0, None, limits).unwrap();
    let logits = changed.forward(&[69, 70]).unwrap();
    for &x in &logits[..4] {
        assert!((x - 0.25 / 1.5_f64.sqrt()).abs() < 2e-7);
    }
    assert!(logits[4..].iter().all(|x| x.abs() < 2e-7));
}
#[test]
fn symbolic_artifacts_reject_stale_sources_incomplete_plans_and_changed_literals() {
    let source = symbolic_fixture::fixture(0.25);
    let dir = artifact(&source);
    let original = dir.manifest();
    for kind in 0..18 {
        let mut m = original.clone();
        match kind {
            0 => m.version += 1,
            1 => m.profile = "native".into(),
            2 => m.config.n_embd += 1,
            3 => m.kv_heads += 1,
            4 => m.rms_norm_epsilon *= 2.0,
            5 => m.rope_frequency_base *= 2.0,
            6 => m.tokenizer.bos_id = 0,
            7 => m.context_length = 4,
            8 => {
                m.kernels.pop();
            }
            9 => m.kernels[0].width += 1,
            10 => {
                m.kernels[0].programs.clear();
            }
            11 => {
                m.kernels
                    .iter_mut()
                    .find(|k| k.operation == Operation::Norm)
                    .unwrap()
                    .fixed[0] *= 2.0
            }
            12 => {
                m.kernels
                    .iter_mut()
                    .find(|k| k.operation == Operation::Norm)
                    .unwrap()
                    .layout = OutputLayout::Scalars
            }
            13 => {
                m.kernels
                    .iter_mut()
                    .find(|k| k.operation == Operation::Dot)
                    .unwrap()
                    .layout = OutputLayout::Buffer
            }
            14 => m.version = 1,
            15 => m.profile = "llama-symbolic-bml-f64-f32kv-v1".into(),
            16 => {
                m.version = 1;
                m.profile = "llama-symbolic-bml-f64-f32kv-v1".into();
            }
            _ => {
                m.kernels
                    .iter_mut()
                    .find(|k| k.operation == Operation::Norm)
                    .unwrap()
                    .fixed
                    .pop();
            }
        }
        dir.save(&m);
        assert!(
            SymbolicCausalModel::open_artifact(&dir.0, None, SymbolicLimits::default()).is_err(),
            "case {kind}"
        );
    }
    dir.save(&original);
    let other = symbolic_fixture::fixture(0.5);
    assert!(
        SymbolicCausalModel::open_artifact(&dir.0, Some(&other.0), SymbolicLimits::default())
            .is_err()
    );
    let relocated = dir.0.join("relocated.gguf");
    std::fs::rename(&source.0, &relocated).unwrap();
    assert!(SymbolicCausalModel::open_artifact(&dir.0, None, SymbolicLimits::default()).is_err());
    assert!(SymbolicCausalModel::open_artifact(
        &dir.0,
        Some(&relocated),
        SymbolicLimits::default()
    )
    .is_ok());
}
#[test]
fn symbolic_aot_preflight_rejects_bad_bytecode_and_runtime_budgets() {
    let source = symbolic_fixture::fixture(0.25);
    let dir = artifact(&source);
    let path = dir.0.join("kernel_0_0.bmlgraph");
    let saved = std::fs::read(&path).unwrap();
    let original = dir.manifest();
    std::fs::write(&path, &saved[..saved.len() - 1]).unwrap();
    assert!(SymbolicCausalModel::open_artifact(&dir.0, None, SymbolicLimits::default()).is_err());
    std::fs::write(&path, &saved).unwrap();
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(8 * 1024 * 1024 + 1).unwrap();
    drop(file);
    assert!(load_directory(&dir.0, None)
        .err()
        .unwrap()
        .contains("byte budget"));
    std::fs::write(&path, &saved).unwrap();
    for ops in [
        vec![RpnOp::Drop],
        vec![RpnOp::Var(u32::MAX)],
        vec![RpnOp::Const(u32::MAX)],
        vec![RpnOp::One, RpnOp::One],
        vec![RpnOp::One, RpnOp::One, RpnOp::FAdd],
    ] {
        dir.replace_program(0, ops);
        assert!(
            SymbolicCausalModel::open_artifact(&dir.0, None, SymbolicLimits::default()).is_err()
        );
    }
    dir.save(&original);
    std::fs::write(&path, &saved).unwrap();
    for limits in [
        SymbolicLimits {
            max_cached_ops: 1,
            ..SymbolicLimits::default()
        },
        SymbolicLimits {
            max_programs: 1,
            ..SymbolicLimits::default()
        },
        SymbolicLimits {
            graph: bml_runtime::graph::GraphLimits {
                max_steps: 1,
                ..Default::default()
            },
            ..SymbolicLimits::default()
        },
    ] {
        assert!(SymbolicCausalModel::open_artifact(&dir.0, None, limits).is_err());
    }
}

#[test]
fn artifact_loader_enforces_profile_sizes_and_rejects_native_opcodes_itself() {
    let source = symbolic_fixture::fixture(0.25);
    let minimal = Directory::new();
    compile_directory(
        &source.0,
        &minimal.0,
        1,
        KernelLimits {
            fragment_bytes: std::mem::size_of::<RpnOp>(),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(load_directory(&minimal.0, None).is_ok());
    let dir = artifact(&source);
    let original = dir.manifest();
    for fragment_bytes in [0, std::mem::size_of::<RpnOp>() - 1, 8 * 1024 * 1024 + 1] {
        let mut manifest = original.clone();
        manifest.fragment_bytes = fragment_bytes;
        dir.save(&manifest);
        let error = load_directory(&dir.0, None).err().unwrap();
        assert!(
            error.contains("profile or size"),
            "{fragment_bytes}: {error}"
        );
    }
    // The inclusive upper limit is legal even when the stored fragments are smaller.
    let mut manifest = original.clone();
    manifest.fragment_bytes = 8 * 1024 * 1024;
    dir.save(&manifest);
    assert!(load_directory(&dir.0, None).is_ok());
    dir.save(&original);
    // Valid native arithmetic is forbidden by the symbolic artifact contract;
    // the loader must reject it before any runtime validator is involved.
    dir.replace_program(0, vec![RpnOp::One, RpnOp::One, RpnOp::FAdd]);
    let error = load_directory(&dir.0, None).err().unwrap();
    assert!(error.contains("native opcode"), "{error}");
}
#[test]
fn symbolic_aot_compile_budgets_do_not_publish_partial_manifests_or_overwrite_files() {
    let source = symbolic_fixture::fixture(0.25);
    let dir = Directory::new();
    for limits in [
        KernelLimits {
            max_nodes: 0,
            ..Default::default()
        },
        KernelLimits {
            max_ops: 1,
            ..Default::default()
        },
        KernelLimits {
            max_output_ops: 1,
            ..Default::default()
        },
        KernelLimits {
            max_programs: 1,
            ..Default::default()
        },
        KernelLimits {
            fragment_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(compile_directory(&source.0, &dir.0, 3, limits).is_err());
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
    }
    for context in [0, 9, u32::MAX] {
        assert!(compile_directory(&source.0, &dir.0, context, KernelLimits::default()).is_err());
    }
    std::fs::write(dir.0.join("keep"), b"original").unwrap();
    assert!(compile_directory(&source.0, &dir.0, 3, KernelLimits::default()).is_err());
    assert_eq!(std::fs::read(dir.0.join("keep")).unwrap(), b"original");
}

/// Build the old scalar-output programs explicitly. Relabeling buffered v2
/// bytecode as v1 would not exercise the old constant namespace or file layout.
fn convert_to_genuine_v1(dir: &Directory) {
    let mut manifest = dir.manifest();
    manifest.version = 1;
    manifest.profile = "llama-symbolic-bml-f64-f32kv-v1".into();
    for (index, kernel) in manifest.kernels.iter_mut().enumerate() {
        if kernel.layout == OutputLayout::Scalars {
            continue;
        }
        let compiled = compile_operation_with_layout(
            kernel.operation,
            kernel.width,
            manifest.rms_norm_epsilon,
            OutputLayout::Scalars,
            KernelLimits {
                fragment_bytes: manifest.fragment_bytes,
                ..Default::default()
            },
        )
        .unwrap();
        let old_signature = kernel
            .operation
            .signature(kernel.width, manifest.rms_norm_epsilon)
            .unwrap();
        assert_eq!(compiled.signature, old_signature);
        assert_eq!(compiled.graphs.len(), old_signature.outputs);
        assert!(compiled
            .graphs
            .iter()
            .flat_map(|g| &g.fragments)
            .flat_map(|f| &f.ops)
            .all(|op| !matches!(op, RpnOp::StoreResult { .. })));
        kernel.layout = OutputLayout::Scalars;
        kernel.fixed = old_signature.fixed;
        kernel.programs.clear();
        for (output, graph) in compiled.graphs.into_iter().enumerate() {
            assert!(graph.const_pool.is_empty());
            let bytes = graph.serialize();
            kernel
                .programs
                .push(format!("{:x}", Sha256::digest(&bytes)));
            std::fs::write(
                dir.0.join(format!("kernel_{index}_{output}.bmlgraph")),
                bytes,
            )
            .unwrap();
        }
    }
    // The historical manifest did not declare a layout field. This also tests
    // the deserializer's backward-compatible Scalars default.
    let mut json = serde_json::to_value(&manifest).unwrap();
    for kernel in json["kernels"].as_array_mut().unwrap() {
        assert!(kernel.as_object_mut().unwrap().remove("layout").is_some());
    }
    std::fs::write(dir.0.join(MANIFEST), serde_json::to_vec(&json).unwrap()).unwrap();
}

#[test]
fn historical_scalar_v1_artifacts_execute_bitwise_like_buffered_v2() {
    let source = symbolic_fixture::fixture(0.25);
    let old = artifact(&source);
    let current = artifact(&source);
    convert_to_genuine_v1(&old);
    let manifest = old.manifest();
    assert_eq!(manifest.version, 1);
    assert!(manifest
        .kernels
        .iter()
        .all(|k| k.layout == OutputLayout::Scalars));
    assert!(manifest
        .kernels
        .iter()
        .filter(|k| matches!(
            k.operation,
            Operation::Norm | Operation::Softmax | Operation::Rope
        ))
        .all(|k| k.programs.len() == k.width));
    assert!(
        manifest
            .kernels
            .iter()
            .map(|k| k.programs.len())
            .sum::<usize>()
            > current.manifest().kernels.len()
    );
    let limits = SymbolicLimits {
        max_nodes: 0,
        ..Default::default()
    };
    let mut v1 = SymbolicCausalModel::open_artifact(&old.0, None, limits).unwrap();
    let mut v2 = SymbolicCausalModel::open_artifact(&current.0, None, limits).unwrap();
    assert_eq!(v1.profile(), "llama-symbolic-bml-f64-f32kv-v1");
    assert_eq!(v2.profile(), "llama-symbolic-bml-f64-f32kv-v2");
    let bits = |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    for prefix in [&[69][..], &[69, 70][..], &[70, 69, 70][..]] {
        assert_eq!(
            bits(v1.forward(prefix).unwrap()),
            bits(v2.forward(prefix).unwrap())
        );
    }
    let mut old_cache = HashConsedKV::new(1, 1, 2, 8);
    let mut current_cache = HashConsedKV::new(1, 1, 2, 8);
    for tokens in [&[69, 70][..], &[69][..]] {
        assert_eq!(
            bits(v1.forward_cached(tokens, &mut old_cache).unwrap()),
            bits(v2.forward_cached(tokens, &mut current_cache).unwrap())
        );
    }
    assert_eq!(old_cache.current_pos(), 3);
    assert_eq!(current_cache.current_pos(), 3);
}

#[test]
fn buffered_artifact_execution_rejects_missing_outputs_and_uses_stored_writes() {
    let source = symbolic_fixture::fixture(0.25);
    let dir = artifact(&source);
    let manifest = dir.manifest();
    let kernel_index = manifest
        .kernels
        .iter()
        .position(|k| k.operation == Operation::Norm)
        .unwrap();
    let kernel = &manifest.kernels[kernel_index];
    assert_eq!(kernel.layout, OutputLayout::Buffer);
    let base = kernel
        .operation
        .signature(kernel.width, manifest.rms_norm_epsilon)
        .unwrap()
        .fixed
        .len();
    let store_ones = |offsets: &[usize]| {
        let mut ops = Vec::new();
        for &offset in offsets {
            ops.extend([
                RpnOp::One,
                RpnOp::Const((base + offset) as u32),
                RpnOp::StoreResult { slot: 0 },
            ]);
        }
        // The graph's scalar return is a finite completion value, independent
        // from the required finite vector in its output buffer.
        ops.push(RpnOp::One);
        ops
    };
    let missing_last = store_ones(&(0..kernel.width - 1).collect::<Vec<_>>());
    let repeated_first = store_ones(&vec![0; kernel.width]);
    // BML(BML(1,1),1) = 4; the four-element fixture output has indices 0..3.
    // The offset is computed at runtime so preflight cannot reject it solely by
    // checking a constant reference. The ignored out-of-range store must leave
    // NaN outputs, which kernel execution rejects before committing a KV token.
    assert_eq!(kernel.width, 4);
    let out_of_range = vec![
        RpnOp::One,
        RpnOp::One,
        RpnOp::One,
        RpnOp::Bml,
        RpnOp::One,
        RpnOp::Bml,
        RpnOp::StoreResult { slot: 0 },
        RpnOp::One,
    ];
    let limits = SymbolicLimits {
        max_nodes: 0,
        ..Default::default()
    };
    for (name, ops) in [
        ("no writes", vec![RpnOp::One]),
        ("missing last", missing_last),
        ("repeated first", repeated_first),
        ("out of range", out_of_range),
    ] {
        dir.replace_program(kernel_index, ops);
        let mut model = SymbolicCausalModel::open_artifact(&dir.0, None, limits).unwrap();
        let mut cache = HashConsedKV::new(1, 1, 2, 8);
        assert!(model.forward_cached(&[69], &mut cache).is_err(), "{name}");
        assert_eq!(cache.current_pos(), 0, "{name}");
    }
    dir.replace_program(
        kernel_index,
        store_ones(&(0..kernel.width).collect::<Vec<_>>()),
    );
    let mut changed = SymbolicCausalModel::open_artifact(&dir.0, None, limits).unwrap();
    let logits = changed.forward(&[69, 70]).unwrap();
    // The final normalization now writes ones. Each of the first four head
    // rows has one coefficient 0.5; the rest are zero. Regenerating the original
    // normalization bytecode would not produce these values.
    for &value in &logits[..4] {
        assert!((value - 0.5).abs() < 2e-7);
    }
    assert!(logits[4..].iter().all(|value| value.abs() < 2e-7));
}
