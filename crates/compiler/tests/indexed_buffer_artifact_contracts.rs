//! Versioned indexed output layouts, genuine historical bytecode and causal use.
use bml_compiler::{
    fragment_program,
    rpn::linearize_outputs,
    symbolic_artifact::{
        compile_directory, compile_directory_bounded, compile_directory_with_rms_bounds,
        load_directory, Manifest, BOUNDED_PROFILE, MANIFEST,
    },
    symbolic_bounded::BoundedProfile,
    symbolic_kernels::{
        compile_operation_with_layout, compile_operation_with_rms_bounds, CompiledKernel,
        KernelLimits, Operation, OutputLayout, RmsBounds,
    },
    BmlGraph, Fragment, HashConsedKV, RpnOp,
};
use bml_domain::{transformer::BoundedOpConfig, BMLTransformer};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
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
            "bml_indexed_buffer_{}_{}",
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
    fn write_graph(&self, index: usize, output: usize, graph: &BmlGraph) -> String {
        assert!(
            graph.const_pool.is_empty(),
            "external constants must not become owned"
        );
        let bytes = graph.serialize();
        std::fs::write(
            self.0.join(format!("kernel_{index}_{output}.bmlgraph")),
            &bytes,
        )
        .unwrap();
        format!("{:x}", Sha256::digest(&bytes))
    }
    fn replace_kernel(&self, manifest: &mut Manifest, index: usize, kernel: CompiledKernel) {
        let descriptor = &mut manifest.kernels[index];
        assert_eq!(
            (descriptor.operation, descriptor.width),
            (kernel.operation, kernel.width)
        );
        descriptor.layout = kernel.layout;
        descriptor.fixed = kernel.signature.fixed;
        descriptor.programs = kernel
            .graphs
            .iter()
            .enumerate()
            .map(|(output, graph)| self.write_graph(index, output, graph))
            .collect();
    }
    fn replace_ops(&self, index: usize, ops: Vec<RpnOp>) {
        let mut manifest = self.manifest();
        let mut graph = BmlGraph::new(manifest.fragment_bytes);
        graph.fragments.push(Fragment { ops });
        manifest.kernels[index].programs = vec![self.write_graph(index, 0, &graph)];
        self.save(&manifest);
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn artifact(source: &symbolic_fixture::Fixture, context: u32) -> Directory {
    let directory = Directory::new();
    compile_directory_bounded(
        &source.0,
        &directory.0,
        context,
        BoundedProfile::default(),
        KernelLimits::default(),
    )
    .unwrap();
    directory
}
fn no_construction() -> SymbolicLimits {
    SymbolicLimits {
        max_nodes: 0,
        ..Default::default()
    }
}
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() < 2e-7,
            "output {index}: {actual} != {expected}"
        );
    }
}

fn execute_loaded(kernel: &CompiledKernel, inputs: &[f64], external: &[f64]) -> Vec<f64> {
    assert_eq!(kernel.graphs.len(), 1);
    assert_eq!(inputs.len(), kernel.signature.inputs);
    assert_eq!(external.len(), kernel.signature.external);
    let mut constants = kernel.signature.fixed.clone();
    constants.extend_from_slice(external);
    let graph = ValidatedGraph::new_with_external_constants(
        kernel.graphs[0].clone(),
        inputs.len(),
        constants.len(),
        1,
        kernel.signature.outputs,
        GraphLimits::default(),
    )
    .unwrap();
    let mut output = ResultBuffer::new(1, kernel.signature.outputs);
    output.slot_mut(0).fill(f64::NAN);
    graph
        .execute_indexed_symbolic_with_constants(inputs, &constants, &mut output)
        .unwrap();
    output.into_vec()
}

// Recreate the historical dense v4 softmax DAG, not the new staged compiler.
fn historical_softmax(width: usize, profile: BoundedProfile, threshold: usize) -> CompiledKernel {
    let config = BoundedOpConfig::softmax(width, profile.score).unwrap();
    let mut builder = BMLTransformer::new();
    let input: Vec<_> = (0..=width).map(|id| builder.var(id as u32)).collect();
    let roots = builder.operation_bounded(&input, &[], &config).unwrap();
    let mut dag = builder.into_soa();
    assert_eq!(dag.const_pool.len(), 10);
    dag.const_pool.extend((0..width).map(|id| id as f64));
    let fixed = dag.const_pool.clone();
    let mut program = linearize_outputs(&dag, &roots, 0, 10).unwrap();
    program.const_pool.clear();
    let graph = fragment_program(&program, threshold);
    assert!(!graph
        .fragments
        .iter()
        .flat_map(|f| &f.ops)
        .any(|op| matches!(op, RpnOp::Loop { .. } | RpnOp::ContextVarIndexed { .. })));
    let signature = profile
        .signature(Operation::Softmax, width, 0.5, OutputLayout::Buffer)
        .unwrap();
    assert_eq!(signature.fixed, fixed);
    CompiledKernel {
        operation: Operation::Softmax,
        width,
        signature,
        layout: OutputLayout::Buffer,
        graphs: vec![graph],
    }
}

fn convert_to_historical_v4_buffers(directory: &Directory) {
    let mut manifest = directory.manifest();
    let profile = manifest.bounded_profile.unwrap();
    for index in 0..manifest.kernels.len() {
        let descriptor = &manifest.kernels[index];
        let kernel = match descriptor.operation {
            Operation::Norm => compile_operation_with_rms_bounds(
                Operation::Norm,
                descriptor.width,
                manifest.rms_norm_epsilon,
                OutputLayout::Buffer,
                Some(profile.rms),
                KernelLimits {
                    fragment_bytes: manifest.fragment_bytes,
                    ..Default::default()
                },
            )
            .unwrap(),
            Operation::Softmax => {
                historical_softmax(descriptor.width, profile, manifest.fragment_bytes)
            }
            _ => continue,
        };
        assert_eq!(kernel.layout, OutputLayout::Buffer);
        let base = if kernel.operation == Operation::Norm {
            17
        } else {
            10
        };
        assert_eq!(kernel.signature.fixed.len(), base + kernel.width);
        assert_eq!(
            &kernel.signature.fixed[base..],
            &(0..kernel.width).map(|i| i as f64).collect::<Vec<_>>()
        );
        directory.replace_kernel(&mut manifest, index, kernel);
    }
    directory.save(&manifest);
}

#[test]
fn indexed_v4_fixed_prefixes_omit_output_offsets_and_load_without_construction() {
    let source = symbolic_fixture::fixture(0.25);
    let directory = artifact(&source, 3);
    let manifest = directory.manifest();
    assert_eq!(manifest.version, 4);
    assert_eq!(manifest.profile, BOUNDED_PROFILE);
    let loaded = load_directory(&directory.0, None).unwrap();
    for kernel in &loaded.kernels {
        if matches!(kernel.operation, Operation::Norm | Operation::Softmax) {
            assert_eq!(kernel.layout, OutputLayout::IndexedBuffer);
            let expected = if kernel.operation == Operation::Norm {
                17
            } else {
                10
            };
            assert_eq!(kernel.signature.fixed.len(), expected);
            assert_eq!(kernel.graphs.len(), 1);
            assert!(kernel.graphs[0]
                .fragments
                .iter()
                .flat_map(|f| &f.ops)
                .any(|op| matches!(op, RpnOp::Loop { .. })));
            assert!(kernel.graphs[0].const_pool.is_empty());
        }
    }
    let json = serde_json::to_value(&manifest).unwrap();
    assert!(json["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .any(|k| k["layout"] == "indexed_buffer"));
    let mut model =
        SymbolicCausalModel::open_artifact(&directory.0, None, no_construction()).unwrap();
    for ids in [&[69][..], &[70, 69][..], &[69, 70, 69][..]] {
        close(
            &model.forward(ids).unwrap(),
            &symbolic_oracle::logits(ids, true),
        );
    }
}

#[test]
fn historical_v4_dense_and_indexed_artifacts_both_execute_their_stored_programs() {
    let source = symbolic_fixture::fixture(0.25);
    let indexed = artifact(&source, 3);
    let historical = artifact(&source, 3);
    convert_to_historical_v4_buffers(&historical);
    let loaded = load_directory(&historical.0, None).unwrap();
    let indexed_loaded = load_directory(&indexed.0, None).unwrap();
    for kernel in &loaded.kernels {
        if matches!(kernel.operation, Operation::Norm | Operation::Softmax) {
            assert_eq!(kernel.layout, OutputLayout::Buffer);
            assert!(!kernel
                .graphs
                .iter()
                .flat_map(|g| &g.fragments)
                .flat_map(|f| &f.ops)
                .any(|op| matches!(op, RpnOp::Loop { .. } | RpnOp::ContextVarIndexed { .. })));
        }
    }
    for artifact in [&loaded, &indexed_loaded] {
        let norm = artifact
            .kernels
            .iter()
            .find(|k| k.operation == Operation::Norm)
            .unwrap();
        let input = [3.0_f64, -4.0, 0.0, 0.125];
        let weights = [3.0, -4.0, 0.5, 1.0];
        let root = (input.iter().map(|x| x * x).sum::<f64>() / 4.0 + 0.5).sqrt();
        let expected: Vec<_> = input
            .iter()
            .zip(weights)
            .map(|(x, w)| x * w / root)
            .collect();
        close(&execute_loaded(norm, &input, &weights), &expected);
        let softmax = artifact
            .kernels
            .iter()
            .find(|k| k.operation == Operation::Softmax && k.width == 3)
            .unwrap();
        for scores in [
            [-16.0_f64, -15.0, -14.0],
            [0.0, 0.0, 0.0],
            [-16.0, 0.0, 16.0],
        ] {
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps = scores.map(|x| (x - maximum).exp());
            let total: f64 = exps.iter().sum();
            let expected = exps.map(|x| x / total);
            let mut input = scores.to_vec();
            input.push(maximum);
            close(&execute_loaded(softmax, &input, &[]), &expected);
        }
    }
    let mut old =
        SymbolicCausalModel::open_artifact(&historical.0, None, no_construction()).unwrap();
    let mut new = SymbolicCausalModel::open_artifact(&indexed.0, None, no_construction()).unwrap();
    for ids in [&[69][..], &[70, 69][..], &[69, 70, 69][..]] {
        let expected = symbolic_oracle::logits(ids, true);
        let dense = old.forward(ids).unwrap();
        let compact = new.forward(ids).unwrap();
        close(&dense, &expected);
        close(&compact, &expected);
        close(&compact, &dense);
    }
    let mut old_cache = HashConsedKV::new(1, 1, 2, 8);
    let mut new_cache = HashConsedKV::new(1, 1, 2, 8);
    for ids in [&[69, 70][..], &[69][..]] {
        close(
            &old.forward_cached(ids, &mut old_cache).unwrap(),
            &new.forward_cached(ids, &mut new_cache).unwrap(),
        );
    }
    assert_eq!(old_cache.current_pos(), 3);
    assert_eq!(new_cache.current_pos(), 3);
}

fn historical_artifact(source: &symbolic_fixture::Fixture, version: u32) -> Directory {
    let directory = Directory::new();
    if version == 3 {
        compile_directory_with_rms_bounds(
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
    } else {
        compile_directory(&source.0, &directory.0, 3, KernelLimits::default()).unwrap();
    }
    if version == 1 {
        let mut manifest = directory.manifest();
        manifest.version = 1;
        manifest.profile = "llama-symbolic-bml-f64-f32kv-v1".into();
        for index in 0..manifest.kernels.len() {
            let descriptor = &manifest.kernels[index];
            let scalar = compile_operation_with_layout(
                descriptor.operation,
                descriptor.width,
                manifest.rms_norm_epsilon,
                OutputLayout::Scalars,
                KernelLimits::default(),
            )
            .unwrap();
            assert!(!scalar
                .graphs
                .iter()
                .flat_map(|g| &g.fragments)
                .flat_map(|f| &f.ops)
                .any(|op| matches!(op, RpnOp::StoreResult { .. })));
            directory.replace_kernel(&mut manifest, index, scalar);
        }
        directory.save(&manifest);
    }
    assert_eq!(directory.manifest().version, version);
    assert!(
        load_directory(&directory.0, None).is_ok(),
        "genuine v{version} baseline"
    );
    directory
}

#[test]
fn indexed_layout_is_rejected_by_v1_v2_v3_and_unrelated_v4_families() {
    let source = symbolic_fixture::fixture(0.25);
    for version in [1, 2, 3] {
        let directory = historical_artifact(&source, version);
        let original = directory.manifest();
        for (index, descriptor) in original.kernels.iter().enumerate() {
            if !matches!(descriptor.operation, Operation::Norm | Operation::Softmax) {
                continue;
            }
            let mut changed = original.clone();
            changed.kernels[index].layout = OutputLayout::IndexedBuffer;
            directory.save(&changed);
            assert!(
                load_directory(&directory.0, None).is_err(),
                "v{version} {:?}",
                descriptor.operation
            );
        }
    }
    let directory = artifact(&source, 3);
    let original = directory.manifest();
    for (index, descriptor) in original.kernels.iter().enumerate() {
        if matches!(descriptor.operation, Operation::Norm | Operation::Softmax) {
            continue;
        }
        let mut changed = original.clone();
        changed.kernels[index].layout = OutputLayout::IndexedBuffer;
        directory.save(&changed);
        assert!(
            load_directory(&directory.0, None).is_err(),
            "v4 {:?}",
            descriptor.operation
        );
        assert!(BoundedProfile::default()
            .signature(
                descriptor.operation,
                descriptor.width,
                original.rms_norm_epsilon,
                OutputLayout::IndexedBuffer
            )
            .is_err());
    }
}

#[test]
fn indexed_literal_counts_values_and_complete_prefix_plan_are_enforced() {
    let source = symbolic_fixture::fixture(0.25);
    let directory = artifact(&source, 3);
    let original = directory.manifest();
    for (index, descriptor) in original.kernels.iter().enumerate() {
        if descriptor.layout != OutputLayout::IndexedBuffer {
            continue;
        }
        for case in 0..3 {
            let mut changed = original.clone();
            match case {
                0 => {
                    changed.kernels[index].fixed.pop();
                }
                1 => changed.kernels[index].fixed.push(0.0),
                _ => changed.kernels[index].fixed[0] += 0.25,
            }
            directory.save(&changed);
            assert!(
                load_directory(&directory.0, None).is_err(),
                "kernel {index}, case {case}"
            );
        }
    }
    let mut missing = original.clone();
    let index = missing
        .kernels
        .iter()
        .position(|k| k.operation == Operation::Softmax && k.width == 2)
        .unwrap();
    missing.kernels.remove(index);
    directory.save(&missing);
    assert!(load_directory(&directory.0, None).is_err());
    directory.save(&original);
    assert!(load_directory(&directory.0, None).is_ok());
}

fn store_ones(count: u32) -> Vec<RpnOp> {
    vec![
        RpnOp::Loop { count, body_len: 3 },
        RpnOp::One,
        RpnOp::Swap,
        RpnOp::StoreResult { slot: 0 },
        RpnOp::One,
    ]
}

#[test]
fn indexed_missing_outputs_never_commit_cache_and_valid_replacement_is_executed() {
    let source = symbolic_fixture::fixture(0.25);
    for operation in [Operation::Norm, Operation::Softmax] {
        let directory = artifact(&source, 3);
        let original = directory.manifest();
        let index = original
            .kernels
            .iter()
            .position(|k| {
                k.operation == operation && (operation == Operation::Norm || k.width == 2)
            })
            .unwrap();
        let width = original.kernels[index].width;
        for ops in [vec![RpnOp::One], store_ones((width - 1) as u32)] {
            directory.replace_ops(index, ops);
            assert!(
                load_directory(&directory.0, None).is_ok(),
                "stored graph passes structural loading"
            );
            let mut model =
                SymbolicCausalModel::open_artifact(&directory.0, None, no_construction()).unwrap();
            let mut cache = HashConsedKV::new(1, 1, 2, 8);
            let position = if operation == Operation::Softmax {
                model.forward_cached(&[69], &mut cache).unwrap();
                1
            } else {
                0
            };
            let prefix = if position == 1 {
                Some((cache.load_k(0, 0, 0), cache.load_v(0, 0, 0)))
            } else {
                None
            };
            assert!(model.forward_cached(&[70], &mut cache).is_err());
            assert_eq!(cache.current_pos(), position);
            if let Some((key, value)) = prefix {
                assert_eq!(cache.load_k(0, 0, 0), key);
                assert_eq!(cache.load_v(0, 0, 0), value);
            }
        }
        if operation == Operation::Norm {
            directory.replace_ops(index, store_ones(width as u32));
            let mut model =
                SymbolicCausalModel::open_artifact(&directory.0, None, no_construction()).unwrap();
            let logits = model.forward(&[69, 70]).unwrap();
            close(&logits[..4], &[0.5; 4]);
            assert!(logits[4..].iter().all(|v| v.abs() < 2e-7));
        }
    }
}

fn source_with_context(context: u32) -> symbolic_fixture::Fixture {
    let source = symbolic_fixture::fixture(0.25);
    let name = b"llama.context_length";
    let mut pattern = (name.len() as u64).to_le_bytes().to_vec();
    pattern.extend(name);
    pattern.extend(4u32.to_le_bytes());
    pattern.extend(8u32.to_le_bytes());
    let mut bytes = std::fs::read(&source.0).unwrap();
    let matches: Vec<_> = bytes
        .windows(pattern.len())
        .enumerate()
        .filter_map(|(offset, value)| (value == pattern).then_some(offset))
        .collect();
    assert_eq!(matches.len(), 1);
    let offset = matches[0] + pattern.len() - 4;
    bytes[offset..offset + 4].copy_from_slice(&context.to_le_bytes());
    // No parser/mmap has opened this temporary fixture yet.
    std::fs::write(&source.0, bytes).unwrap();
    source
}

#[test]
fn indexed_prefix_plan_storage_grows_linearly_without_omitting_any_prefix() {
    let source = source_with_context(16);
    let mut observations = Vec::new();
    for context in [2u32, 4, 8, 16] {
        let directory = artifact(&source, context);
        let manifest = directory.manifest();
        let loaded = load_directory(&directory.0, None).unwrap();
        for operation in [Operation::Softmax, Operation::WeightedSum] {
            let widths: Vec<_> = manifest
                .kernels
                .iter()
                .filter(|k| k.operation == operation)
                .map(|k| k.width)
                .collect();
            assert_eq!(widths, (1..=context as usize).collect::<Vec<_>>());
        }
        for operation in [
            Operation::Dot,
            Operation::Norm,
            Operation::AttentionDot,
            Operation::Rope,
            Operation::Swiglu,
            Operation::Add,
            Operation::Scale,
        ] {
            assert!(manifest.kernels.iter().any(|k| k.operation == operation));
        }
        let fixed: usize = manifest.kernels.iter().map(|k| k.fixed.len()).sum();
        let bytes: usize = loaded
            .kernels
            .iter()
            .flat_map(|k| &k.graphs)
            .map(|g| g.serialize().len())
            .sum();
        let ops: usize = loaded
            .kernels
            .iter()
            .flat_map(|k| &k.graphs)
            .flat_map(|g| &g.fragments)
            .map(|f| f.ops.len())
            .sum();
        assert!(manifest.kernels.iter().all(|k| k.programs.len() == 1));
        observations.push((
            context as usize,
            [manifest.kernels.len(), fixed, bytes, ops],
        ));
        println!(
            "INDEXED_PLAN_STORAGE context={context} kernels={} fixed_literals={fixed} program_bytes={bytes} opcodes={ops}",
            manifest.kernels.len()
        );
    }
    let first_delta: Vec<_> = observations[1]
        .1
        .iter()
        .zip(observations[0].1)
        .map(|(&right, left)| (right - left) / 2)
        .collect();
    assert!(first_delta.iter().all(|&v| v > 0));
    for pair in observations.windows(2) {
        let context_delta = pair[1].0 - pair[0].0;
        for (metric, &per_prefix) in first_delta.iter().enumerate() {
            assert_eq!(
                pair[1].1[metric] - pair[0].1[metric],
                context_delta * per_prefix,
                "metric {metric} grew nonlinearly: {observations:?}"
            );
        }
    }
}
