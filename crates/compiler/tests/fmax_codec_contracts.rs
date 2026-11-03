use bml_compiler::distributed::{
    load_distributed_fragment, serialize_distributed, serialize_sub_fragments, sub_fragment,
    DistributedFragment,
};
use bml_compiler::gguf_compiler::{
    load_from_dir, serialize_to_dir, CompilationResult, ModelConfig,
};
use bml_compiler::{BmlGraph, Fragment, RpnOp, BMLGRAPH_MAGIC, BMLGRAPH_VERSION};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml-fmax-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn config() -> ModelConfig {
    ModelConfig {
        architecture: "llama".into(),
        n_layers: 1,
        n_heads: 1,
        n_embd: 2,
        context_length: 8,
        vocab_size: 260,
    }
}
fn graph(op: RpnOp) -> BmlGraph {
    BmlGraph {
        fragments: vec![Fragment {
            ops: vec![RpnOp::One, RpnOp::One, op],
        }],
        const_pool: vec![],
        threshold: 32768,
    }
}
#[test]
fn single_file_selection_version_preserves_old_bytes_and_rejects_new_tag_in_old_versions() {
    let legacy = graph(RpnOp::FAdd);
    let mut fixture = Vec::new();
    for value in [BMLGRAPH_MAGIC, BMLGRAPH_VERSION, 1, 3] {
        fixture.extend(value.to_le_bytes());
    }
    fixture.extend([0, 0, 9]);
    fixture.extend(0u64.to_le_bytes());
    assert_eq!(legacy.serialize(), fixture);
    let selected = graph(RpnOp::FMax);
    let encoded = selected.serialize();
    assert_eq!(&encoded[4..8], &4u32.to_le_bytes());
    assert_eq!(&encoded[16..19], &[0, 0, 16]);
    assert_eq!(
        BmlGraph::deserialize(&encoded, 32768)
            .unwrap()
            .evaluate(0.0),
        1.0
    );
    for version in [1u32, 2, 3] {
        let mut bytes = encoded.clone();
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        assert!(BmlGraph::deserialize(&bytes, 32768).is_err());
    }
    for length in 0..encoded.len() {
        assert!(BmlGraph::deserialize(&encoded[..length], 32768).is_err());
    }
}
#[test]
fn directory_and_distributed_selection_records_roundtrip_with_version_guards() {
    for op in [RpnOp::FAdd, RpnOp::FMax] {
        let version = if op == RpnOp::FMax { 4u32 } else { 2 };
        let directory = Directory::new();
        let result = CompilationResult {
            graph: graph(op),
            const_pool: vec![],
            config: config(),
            num_fragments: 1,
        };
        serialize_to_dir(&result, &directory.0).unwrap();
        let header = directory.0.join("header.bmlgraph");
        let mut bytes = std::fs::read(&header).unwrap();
        assert_eq!(&bytes[4..8], &version.to_le_bytes());
        assert_eq!(
            load_from_dir(&directory.0).unwrap().0.fragments,
            result.graph.fragments
        );
        if op == RpnOp::FMax {
            bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
            std::fs::write(&header, bytes).unwrap();
            assert!(load_from_dir(&directory.0).is_err());
        }
        let directory = Directory::new();
        let shard = DistributedFragment {
            fragment_id: 0,
            layer_start: 0,
            layer_end: 1,
            ops: vec![RpnOp::One, RpnOp::One, op],
            weights: vec![],
            tensors: vec![],
            config: config(),
            n_kv_heads: 1,
            head_dim: 2,
        };
        serialize_distributed(std::slice::from_ref(&shard), &directory.0).unwrap();
        let pieces = sub_fragment(&shard);
        assert_eq!(pieces[0].bytecode_size(), 3);
        serialize_sub_fragments(&directory.0, &shard, &pieces).unwrap();
        let sub = std::fs::read(directory.0.join("fragment_0/sub_0.bmlgraph")).unwrap();
        assert_eq!(&sub[4..8], &version.to_le_bytes());
        assert_eq!(
            &sub[28..31],
            &[0, 0, if op == RpnOp::FMax { 16 } else { 9 }]
        );
        let path = directory.0.join("fragment_0.bmlgraph");
        let mut bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[4..8], &version.to_le_bytes());
        assert_eq!(load_distributed_fragment(&path).unwrap().ops, shard.ops);
        if op == RpnOp::FMax {
            bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
            std::fs::write(&path, bytes).unwrap();
            assert!(load_distributed_fragment(&path).is_err());
        }
    }
}

#[test]
fn maximum_selection_does_not_enter_existing_symbolic_artifact_bases() {
    assert!(!bml_compiler::symbolic_kernels::is_symbolic_opcode(
        &RpnOp::FMax
    ));
    assert!(!bml_compiler::symbolic_bounded::is_bounded_opcode(
        &RpnOp::FMax
    ));
}
