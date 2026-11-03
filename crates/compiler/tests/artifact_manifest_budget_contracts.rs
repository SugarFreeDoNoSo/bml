use bml_compiler::{causal_artifact, symbolic_artifact, symbolic_kernels::KernelLimits};
#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;
#[allow(dead_code)]
#[path = "../../cli/tests/support/token_fixture.rs"]
mod token_fixture;

struct Directory(std::path::PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn json_expansion_is_bounded_before_publishing_native_or_symbolic_artifacts() {
    // A GGUF string occupies one byte per NUL; JSON must encode each as six
    // bytes (\u0000). A valid, modest source can therefore exceed 16 MiB on disk.
    let unknown = "\0".repeat(3 * 1024 * 1024);
    let source = token_fixture::fixture_with_unknown_piece(&unknown);
    assert!(std::fs::metadata(&source.0).unwrap().len() < 4 * 1024 * 1024);
    for symbolic in [false, true] {
        let output = source
            .0
            .with_extension(if symbolic { "symbolic" } else { "native" });
        let result = if symbolic {
            symbolic_artifact::compile_directory(&source.0, &output, 1, KernelLimits::default())
                .map(|_| ())
        } else {
            causal_artifact::compile_directory(&source.0, &output, 32768).map(|_| ())
        };
        let published = output.exists();
        if published {
            std::fs::remove_dir_all(&output).unwrap();
        }
        let error = result.expect_err("expanded manifest must exceed its byte budget");
        assert!(error.contains("manifest exceeds byte budget"), "{error}");
        assert!(!published, "rejected artifact must not publish files");
    }
}

#[test]
fn zero_layer_symbolic_plan_includes_optional_final_normalization() {
    use bml_compiler::symbolic_kernels::Operation;
    let source = token_fixture::fixture_with_output_norm();
    let output = source.0.with_extension("normalized");
    let manifest =
        symbolic_artifact::compile_directory(&source.0, &output, 1, KernelLimits::default())
            .unwrap();
    let operations: Vec<_> = manifest
        .kernels
        .iter()
        .map(|k| (k.operation, k.width))
        .collect();
    assert_eq!(operations, [(Operation::Dot, 2), (Operation::Norm, 2)]);
    assert!(symbolic_artifact::load_directory(&output, None).is_ok());
    std::fs::remove_dir_all(output).unwrap();
}

#[test]
fn symbolic_fragment_count_accepts_65536_and_rejects_a_larger_compilation() {
    let limits = KernelLimits {
        max_nodes: 150_000,
        max_ops: 600_000,
        max_output_ops: 600_000,
        fragment_bytes: 9 * std::mem::size_of::<bml_compiler::RpnOp>(),
        ..Default::default()
    };
    // Current signed-dot expansion has 902*w-89 opcodes. Nine opcodes per
    // fragment put width 654 exactly at the inclusive 65,536-fragment limit.
    for width in [654, 655] {
        let source = token_fixture::fixture_with_width(width);
        let output = source.0.with_extension("fragment-limit");
        let result = symbolic_artifact::compile_directory(&source.0, &output, 1, limits);
        if width == 654 {
            result.unwrap();
            let loaded = symbolic_artifact::load_directory(&output, None).unwrap();
            assert_eq!(loaded.kernels.len(), 1);
            assert_eq!(loaded.kernels[0].graphs[0].fragments.len(), 65_536);
            std::fs::remove_dir_all(&output).unwrap();
        } else {
            assert_eq!(
                result.unwrap_err(),
                "symbolic fragment count exceeds budget"
            );
            assert!(!output.exists());
        }
    }
}

#[test]
fn symbolic_loader_accumulates_file_bytes_and_accepts_the_exact_total_budget() {
    use bml_compiler::{
        symbolic_artifact::KernelDescriptor,
        symbolic_kernels::{Operation, OutputLayout},
        BmlGraph, Fragment, RpnOp,
    };
    use sha2::{Digest, Sha256};
    use std::collections::BTreeSet;

    let source = symbolic_fixture::fixture(0.25);
    let mut bytes = std::fs::read(&source.0).unwrap();
    let key = b"llama.context_length";
    let at = bytes.windows(key.len()).position(|w| w == key).unwrap() + key.len();
    bytes[at + 4..at + 8].copy_from_slice(&128u32.to_le_bytes());
    std::fs::write(&source.0, bytes).unwrap();
    let output = Directory(source.0.with_extension("aggregate-budget"));
    let mut manifest =
        symbolic_artifact::compile_directory(&source.0, &output.0, 1, KernelLimits::default())
            .unwrap();
    manifest.context_length = 128;
    let mut operations = BTreeSet::from([
        (Operation::Norm, 4),
        (Operation::Rope, 2),
        (Operation::Swiglu, 1),
        (Operation::Add, 1),
        (Operation::Scale, 1),
    ]);
    for width in 1..=128 {
        operations.extend([(Operation::Dot, width), (Operation::Softmax, width)]);
    }
    assert_eq!(operations.len(), 261);
    manifest.kernels.clear();
    let budget = 64 * 1024 * 1024;
    // Empty fragments are valid container records, with zero opcodes/constants.
    // Runtime execution preflight is a separate contract. Using framing bytes
    // isolates the aggregate byte ceiling from opcode and per-file ceilings.
    let base_bytes = operations.len() * 20;
    let mut remaining_fragments = (budget - base_bytes) / 4;
    assert_eq!(base_bytes + remaining_fragments * 4, budget);
    let mut total_bytes = 0;
    for (index, (operation, width)) in operations.into_iter().enumerate() {
        let count = remaining_fragments.min(65_536);
        remaining_fragments -= count;
        let graph = BmlGraph {
            fragments: vec![Fragment { ops: vec![] }; count],
            const_pool: vec![],
            threshold: manifest.fragment_bytes,
        };
        let bytes = graph.serialize();
        assert_eq!(bytes.len(), 20 + count * 4);
        total_bytes += bytes.len();
        let layout = OutputLayout::for_operation(operation);
        let signature = operation
            .signature_for_layout(width, manifest.rms_norm_epsilon, layout)
            .unwrap();
        manifest.kernels.push(KernelDescriptor {
            operation,
            width,
            layout,
            fixed: signature.fixed,
            programs: vec![format!("{:x}", Sha256::digest(&bytes))],
        });
        std::fs::write(output.0.join(format!("kernel_{index}_0.bmlgraph")), bytes).unwrap();
    }
    assert_eq!(remaining_fragments, 0);
    assert_eq!(total_bytes, budget);
    let save = |manifest: &symbolic_artifact::Manifest| {
        std::fs::write(
            output.0.join(symbolic_artifact::MANIFEST),
            serde_json::to_vec(manifest).unwrap(),
        )
        .unwrap();
    };
    save(&manifest);
    let loaded = symbolic_artifact::load_directory(&output.0, None).unwrap();
    assert_eq!(loaded.kernels.len(), 261);
    drop(loaded);
    // Add one byte to a stored program, with a matching digest and unchanged
    // fragment count. All individual limits remain satisfied.
    let path = output.0.join("kernel_0_0.bmlgraph");
    let mut graph =
        BmlGraph::deserialize(&std::fs::read(&path).unwrap(), manifest.fragment_bytes).unwrap();
    graph.fragments[0].ops.push(RpnOp::One);
    let bytes = graph.serialize();
    manifest.kernels[0].programs[0] = format!("{:x}", Sha256::digest(&bytes));
    std::fs::write(path, bytes).unwrap();
    save(&manifest);
    let error = symbolic_artifact::load_directory(&output.0, None)
        .err()
        .expect("64 MiB plus one byte must be rejected");
    assert_eq!(error, "artifact file exceeds its byte budget");
}
