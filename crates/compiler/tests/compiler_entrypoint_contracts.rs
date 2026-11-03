//! Public GGUF compilation must produce an executable causal plan, including
//! every BML operation and source binding, rather than a scalar placeholder.
use bml_compiler::{
    gguf_compiler::{compile_gguf, compile_gguf_fast, compile_gguf_with_limits},
    hardware::HardwareSpec,
    symbolic_artifact::{self, CompiledArtifact},
    symbolic_kernels::{is_symbolic_opcode, KernelLimits, Operation},
    HashConsedKV,
};
use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
use std::{collections::BTreeSet, path::PathBuf};

#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;
#[path = "../../runtime/tests/support/symbolic_oracle.rs"]
mod symbolic_oracle;

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2e-7,
            "logit {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn both_gguf_entrypoints_publish_complete_numerical_causal_plans() {
    let source = symbolic_fixture::fixture(0.25);
    let source_bytes = std::fs::read(&source.0).unwrap();
    let hardware = HardwareSpec::new(4, 256, 1024, 4096);
    for fast in [false, true] {
        let compiled: CompiledArtifact = if fast {
            compile_gguf_fast(&source.0, &hardware)
        } else {
            compile_gguf(&source.0, &hardware)
        }
        .unwrap();
        let manifest = compiled.manifest();
        assert_eq!(manifest.context_length, 8);
        assert_eq!(manifest.fragment_bytes, 256);
        let families: BTreeSet<_> = manifest.kernels.iter().map(|k| k.operation).collect();
        assert_eq!(
            families,
            BTreeSet::from([
                Operation::Dot,
                Operation::Norm,
                Operation::Rope,
                Operation::Softmax,
                Operation::Swiglu,
                Operation::Add,
                Operation::Scale,
            ])
        );
        let directory = Directory(source.0.with_extension(if fast { "fast" } else { "full" }));
        assert!(!directory.0.exists());
        compiled.publish(&directory.0).unwrap();
        let loaded = symbolic_artifact::load_directory(&directory.0, None).unwrap();
        assert_eq!(loaded.compiler.mapped_bytes(), source_bytes.len());
        for kernel in &loaded.kernels {
            for graph in &kernel.graphs {
                assert!(graph.const_pool.is_empty());
                assert!(graph.all_fragments_under_threshold());
                assert!(graph
                    .fragments
                    .iter()
                    .flat_map(|f| &f.ops)
                    .all(is_symbolic_opcode));
            }
        }
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
        for ids in [&[69][..], &[69, 70][..], &[70, 69][..], &[69, 70, 69][..]] {
            close(
                &model.forward(ids).unwrap(),
                &symbolic_oracle::logits(ids, true),
            );
        }
        let mut cache = HashConsedKV::new(1, 1, 2, 8);
        model.forward_cached(&[69, 70], &mut cache).unwrap();
        close(
            &model.forward_cached(&[69], &mut cache).unwrap(),
            &symbolic_oracle::logits(&[69, 70, 69], true),
        );
        assert_eq!(cache.current_pos(), 3);
        assert_eq!(model.executed_operations().len(), families.len());
        assert_eq!(std::fs::read(&source.0).unwrap(), source_bytes);
    }
}

#[test]
fn explicit_context_hardware_and_budgets_are_enforced_before_publication() {
    let source = symbolic_fixture::fixture(0.25);
    let hardware = HardwareSpec::new(1, 128, 1024, 4096);
    let compiled = compile_gguf_with_limits(
        &source.0,
        &hardware,
        3,
        KernelLimits {
            fragment_bytes: 0, // Hardware overrides this field.
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(compiled.manifest().context_length, 3);
    assert_eq!(compiled.manifest().fragment_bytes, 128);
    let output = Directory(source.0.with_extension("explicit"));
    compiled.publish(&output.0).unwrap();
    let mut model =
        SymbolicCausalModel::open_artifact(&output.0, None, Default::default()).unwrap();
    close(
        &model.forward(&[69, 70, 69]).unwrap(),
        &symbolic_oracle::logits(&[69, 70, 69], true),
    );
    assert!(model.forward(&[69, 70, 69, 70]).is_err());
    let before = std::fs::read(output.0.join(symbolic_artifact::MANIFEST)).unwrap();
    assert_eq!(
        compiled.publish(&output.0).unwrap_err(),
        "output directory is not empty"
    );
    assert_eq!(
        std::fs::read(output.0.join(symbolic_artifact::MANIFEST)).unwrap(),
        before
    );
    for context in [0, 9, u32::MAX] {
        assert!(
            compile_gguf_with_limits(&source.0, &hardware, context, Default::default()).is_err()
        );
    }
    for limits in [
        KernelLimits {
            max_nodes: 1,
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
    ] {
        assert!(compile_gguf_with_limits(&source.0, &hardware, 3, limits)
            .unwrap_err()
            .contains("budget"));
    }
    let hardware = HardwareSpec::new(1, 0, 1024, 4096);
    assert!(compile_gguf(&source.0, &hardware).is_err());
    assert!(compile_gguf_fast(&source.0, &hardware).is_err());
}
