//! Artifact persistence and actual stored-program execution, with a closed-form
//! GQA fixture independent of the compiler. No downloaded model is required.
use bml_compiler::{
    causal_artifact::{compile_directory, Manifest, MANIFEST},
    gguf_compiler::InferenceCompiler,
    BmlGraph, Fragment, HashConsedKV, RpnOp,
};
use bml_runtime::causal_artifact::CompiledCausalModel;
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
#[path = "support/causal_fixture.rs"]
mod causal_fixture;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_causal_artifact_{}_{}",
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
    fn program(&self, ops: Vec<RpnOp>) {
        let graph = BmlGraph {
            fragments: vec![Fragment { ops }],
            const_pool: vec![],
            threshold: 32768,
        };
        let bytes = graph.serialize();
        let mut manifest = self.manifest();
        manifest.programs[0].sha256 = format!("{:x}", Sha256::digest(&bytes));
        std::fs::write(self.0.join("program_0.bmlgraph"), bytes).unwrap();
        self.save(&manifest);
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn stored_program_byte_budget_accepts_64_kib_and_rejects_one_more_byte() {
    use bml_compiler::causal_artifact::load_directory;

    let source = causal_fixture::fixture(false);
    let dir = Directory::new();
    compile_directory(&source.0, &dir.0, 32768).unwrap();
    let budget = 65_536;
    let mut graph = BmlGraph {
        fragments: vec![Fragment { ops: vec![] }],
        const_pool: vec![0.0; 4096],
        threshold: 32768,
    };
    let framing = graph.serialize().len();
    let wide_op = RpnOp::Loop {
        count: 1,
        body_len: 0,
    };
    graph.fragments[0].ops.push(wide_op);
    let wide_bytes = graph.serialize().len() - framing;
    // The container loader admits serialized programs; execution preflight is
    // a separate runtime contract. Wide opcodes and constants reach the byte
    // boundary without exceeding the independent 4096-opcode limit.
    for bytes in [budget, budget + 1] {
        let payload_bytes = bytes - framing;
        graph.fragments[0].ops = vec![wide_op; payload_bytes / wide_bytes];
        graph.fragments[0]
            .ops
            .extend(vec![RpnOp::Zero; payload_bytes % wide_bytes]);
        assert!(graph.fragments[0].ops.len() <= 4096);
        let encoded = graph.serialize();
        let mut manifest = dir.manifest();
        manifest.programs[0].sha256 = format!("{:x}", Sha256::digest(&encoded));
        std::fs::write(dir.0.join("program_0.bmlgraph"), encoded).unwrap();
        dir.save(&manifest);
        assert_eq!(
            std::fs::metadata(dir.0.join("program_0.bmlgraph"))
                .unwrap()
                .len(),
            bytes as u64
        );
        let result = load_directory(&dir.0, None);
        if bytes == budget {
            assert!(
                result.is_ok(),
                "inclusive 64 KiB program budget: {:?}",
                result.err()
            );
        } else {
            let error = result.err().expect("oversized program must be rejected");
            assert_eq!(error, "artifact file exceeds its byte budget");
        }
    }
}

#[test]
fn stored_causal_graphs_match_closed_form_and_incremental_native_inference() {
    let source = causal_fixture::fixture(false);
    let dir = Directory::new();
    let manifest = compile_directory(&source.0, &dir.0, 32768).unwrap();
    assert_eq!(manifest.projections.len(), 8);
    assert_eq!(manifest.programs.len(), 1); // All fixture matrices have input width four.
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 2); // No copied weight file.
    let compiled = CompiledCausalModel::open(&dir.0, None).unwrap();
    let native = InferenceCompiler::open_mapped(&source.0).unwrap();
    assert_eq!(
        compiled.compiler().vocab().encode("A").unwrap(),
        native.vocab().encode("A").unwrap()
    );
    let ids = [69, 70, 69];
    let actual = compiled.forward(&ids).unwrap();
    let hidden = [8.0_f64 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0];
    let denominator = (hidden.iter().map(|v| v * v).sum::<f64>() / 4.0 + 3.0).sqrt();
    for (&value, h) in actual.iter().zip(hidden) {
        assert!((value - h / denominator).abs() < 1e-6);
    }
    assert!(actual[4..].iter().all(|&v| v == 0.0));
    assert_eq!(actual, native.forward(&ids).unwrap());
    let mut cache = HashConsedKV::new(1, 1, 2, 4);
    compiled.forward_cached(&ids[..2], &mut cache).unwrap();
    assert_eq!(
        compiled.forward_cached(&ids[2..], &mut cache).unwrap(),
        actual
    );
    assert_eq!(cache.current_pos(), 3);
    assert!(compiled.forward_cached(&[u32::MAX], &mut cache).is_err());
    assert_eq!(cache.current_pos(), 3);
}

#[test]
fn loader_executes_saved_bytecode_instead_of_regenerating_a_projection() {
    let source = causal_fixture::fixture(false);
    let dir = Directory::new();
    compile_directory(&source.0, &dir.0, 32768).unwrap();
    let expected = CompiledCausalModel::open(&dir.0, None)
        .unwrap()
        .forward(&[69])
        .unwrap();
    assert!(expected.iter().any(|&v| v != 0.0));
    // An artifact author deliberately replaces the program and updates its hash.
    // If load regenerated dot products from metadata, this would remain nonzero.
    dir.program(vec![RpnOp::Zero]);
    let actual = CompiledCausalModel::open(&dir.0, None)
        .unwrap()
        .forward(&[69])
        .unwrap();
    assert!(actual.iter().all(|&v| v == 0.0));
}

#[test]
fn artifact_rejects_stale_sources_corrupt_programs_and_invalid_bindings() {
    let source = causal_fixture::fixture(false);
    let dir = Directory::new();
    compile_directory(&source.0, &dir.0, 32768).unwrap();
    let original = dir.manifest();
    for kind in 0..7 {
        let mut m = original.clone();
        match kind {
            0 => m.version += 1,
            1 => m.config.n_embd += 1,
            2 => m.tokenizer.bos_id = 0,
            3 => m.projections[0].program = usize::MAX,
            4 => m.projections[0].tensor = "wrong.weight".into(),
            5 => m.programs.push(m.programs[0].clone()),
            _ => m.rms_norm_epsilon *= 2.0,
        }
        dir.save(&m);
        assert!(
            CompiledCausalModel::open(&dir.0, None).is_err(),
            "case {kind}"
        );
    }
    dir.save(&original);
    let path = dir.0.join("program_0.bmlgraph");
    let saved = std::fs::read(&path).unwrap();
    std::fs::write(&path, &saved[..saved.len() - 1]).unwrap();
    assert!(CompiledCausalModel::open(&dir.0, None).is_err());
    std::fs::write(&path, &saved).unwrap();
    let mut model = std::fs::read(&source.0).unwrap();
    let end = model.len() - 1;
    model[end] ^= 1;
    std::fs::write(&source.0, model).unwrap();
    assert!(CompiledCausalModel::open(&dir.0, None).is_err());
}

#[test]
fn runtime_preflight_rejects_bad_stacks_and_expanded_loop_budgets() {
    let source = causal_fixture::fixture(false);
    let dir = Directory::new();
    compile_directory(&source.0, &dir.0, 32768).unwrap();
    for ops in [
        vec![RpnOp::Drop],
        vec![
            RpnOp::Zero,
            RpnOp::Loop {
                count: u32::MAX,
                body_len: 1,
            },
            RpnOp::Drop,
        ],
        vec![RpnOp::Const(0)],
    ] {
        dir.program(ops);
        assert!(CompiledCausalModel::open(&dir.0, None).is_err());
    }
}

#[test]
fn relocated_source_is_hash_checked_and_compile_does_not_overwrite_existing_files() {
    let source = causal_fixture::fixture(false);
    let dir = Directory::new();
    assert!(compile_directory(&source.0, &dir.0, 1).is_err());
    assert!(std::fs::read_dir(&dir.0).unwrap().next().is_none());
    compile_directory(&source.0, &dir.0, 32768).unwrap();
    let before = std::fs::read(dir.0.join(MANIFEST)).unwrap();
    assert!(compile_directory(&source.0, &dir.0, 32768).is_err());
    assert_eq!(std::fs::read(dir.0.join(MANIFEST)).unwrap(), before);
    let relocated = dir.0.join("relocated.gguf");
    std::fs::rename(&source.0, &relocated).unwrap();
    assert!(CompiledCausalModel::open(&dir.0, None).is_err());
    assert!(CompiledCausalModel::open(&dir.0, Some(&relocated))
        .unwrap()
        .forward(&[69])
        .is_ok());
}
