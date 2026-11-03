use bml_compiler::distributed::{
    load_distributed_fragment, load_distributed_header, serialize_distributed, DistributedFragment,
    TensorMeta,
};
use bml_compiler::gguf_compiler::ModelConfig;
use bml_compiler::RpnOp;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml-distributed-loading-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn shard(&self) -> PathBuf {
        self.0.join("fragment_0.bmlgraph")
    }
    fn header(&self) -> PathBuf {
        self.0.join("header.bmlgraph")
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> DistributedFragment {
    DistributedFragment {
        fragment_id: 0,
        layer_start: 0,
        layer_end: 1,
        ops: vec![],
        weights: vec![1.0, -2.0, 3.0, -4.0],
        tensors: vec![TensorMeta {
            name: "blk.0.attn_q.weight".into(),
            offset: 0,
            n_rows: 2,
            n_cols: 2,
        }],
        config: ModelConfig {
            architecture: "llama".into(),
            n_layers: 2,
            n_heads: 1,
            n_embd: 2,
            context_length: 8,
            vocab_size: 260,
        },
        n_kv_heads: 1,
        head_dim: 2,
    }
}

#[test]
fn distributed_roundtrip_restores_header_configuration_and_all_opcode_encodings() {
    let directory = Directory::new();
    let mut original = fixture();
    original.ops = vec![
        RpnOp::One,
        RpnOp::Zero,
        RpnOp::Bml,
        RpnOp::Dup,
        RpnOp::Loop {
            count: 2,
            body_len: 1,
        },
        RpnOp::Var(3),
        RpnOp::Const(4),
        RpnOp::VarIndexed { base: 5 },
        RpnOp::StoreResult { slot: 6 },
        RpnOp::FAdd,
        RpnOp::FMul,
        RpnOp::Pick { depth: 7 },
        RpnOp::Drop,
        RpnOp::Swap,
    ];
    serialize_distributed(std::slice::from_ref(&original), &directory.0).unwrap();
    let loaded = load_distributed_fragment(&directory.shard()).unwrap();
    assert_eq!(loaded.config, original.config);
    assert_eq!(loaded.ops, original.ops);
    assert_eq!(loaded.weights, original.weights);
    assert_eq!(
        (loaded.fragment_id, loaded.layer_start, loaded.layer_end),
        (0, 0, 1)
    );
    assert_eq!((loaded.n_kv_heads, loaded.head_dim), (1, 2));
    assert_eq!(loaded.tensors[0].name, original.tensors[0].name);
    assert_eq!(
        (
            loaded.tensors[0].offset,
            loaded.tensors[0].n_rows,
            loaded.tensors[0].n_cols
        ),
        (0, 2, 2)
    );
    let (config, count) = load_distributed_header(&directory.0).unwrap();
    assert_eq!(config, original.config);
    assert_eq!(count, 1);
    let bytes = std::fs::read(directory.shard()).unwrap();
    for cut in 0..bytes.len() {
        std::fs::write(directory.shard(), &bytes[..cut]).unwrap();
        assert!(
            load_distributed_fragment(&directory.shard()).is_err(),
            "fragment cut={cut}"
        );
    }
}

#[test]
fn distributed_loader_rejects_malformed_counts_ranges_names_and_envelopes() {
    let directory = Directory::new();
    serialize_distributed(&[fixture()], &directory.0).unwrap();
    let original = std::fs::read(directory.shard()).unwrap();
    let tensor_offset_at = 56 + "blk.0.attn_q.weight".len();
    let patches = [
        (0, 0_u32.to_le_bytes().to_vec()),
        (4, 0_u32.to_le_bytes().to_vec()),
        (12, 2_u32.to_le_bytes().to_vec()),
        (16, 3_u32.to_le_bytes().to_vec()),
        (20, u32::MAX.to_le_bytes().to_vec()),
        (24, u64::MAX.to_le_bytes().to_vec()),
        (48, u32::MAX.to_le_bytes().to_vec()),
        (52, u32::MAX.to_le_bytes().to_vec()),
        (56, vec![255]),
        (tensor_offset_at, u64::MAX.to_le_bytes().to_vec()),
        (tensor_offset_at + 12, 3_u32.to_le_bytes().to_vec()),
    ];
    for (offset, patch) in patches {
        let mut bytes = original.clone();
        bytes[offset..offset + patch.len()].copy_from_slice(&patch);
        std::fs::write(directory.shard(), bytes).unwrap();
        assert!(
            load_distributed_fragment(&directory.shard()).is_err(),
            "offset={offset}"
        );
    }
    let mut bytes = original.clone();
    bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
    bytes.insert(24, 255);
    std::fs::write(directory.shard(), bytes).unwrap();
    assert!(load_distributed_fragment(&directory.shard()).is_err());
    let mut bytes = original;
    bytes.push(0);
    std::fs::write(directory.shard(), bytes).unwrap();
    assert!(load_distributed_fragment(&directory.shard()).is_err());
    std::fs::OpenOptions::new()
        .write(true)
        .open(directory.shard())
        .unwrap()
        .set_len(512 * 1024 * 1024 + 1)
        .unwrap();
    assert!(load_distributed_fragment(&directory.shard()).is_err());
}

#[test]
fn distributed_header_is_required_complete_and_bounded() {
    let directory = Directory::new();
    serialize_distributed(&[fixture()], &directory.0).unwrap();
    let original = std::fs::read(directory.header()).unwrap();
    for cut in 0..original.len() {
        std::fs::write(directory.header(), &original[..cut]).unwrap();
        assert!(
            load_distributed_header(&directory.0).is_err(),
            "header cut={cut}"
        );
        assert!(load_distributed_fragment(&directory.shard()).is_err());
    }
    for (offset, patch) in [
        (0, 0_u32.to_le_bytes().to_vec()),
        (4, 0_u32.to_le_bytes().to_vec()),
        (8, 0_u32.to_le_bytes().to_vec()),
        (8, u32::MAX.to_le_bytes().to_vec()),
        (12, u64::MAX.to_le_bytes().to_vec()),
        (20, vec![255]),
    ] {
        let mut bytes = original.clone();
        bytes[offset..offset + patch.len()].copy_from_slice(&patch);
        std::fs::write(directory.header(), bytes).unwrap();
        assert!(
            load_distributed_fragment(&directory.shard()).is_err(),
            "header offset={offset}"
        );
    }
    let mut bytes = original;
    bytes.push(0);
    std::fs::write(directory.header(), bytes).unwrap();
    assert!(load_distributed_header(&directory.0).is_err());
    std::fs::OpenOptions::new()
        .write(true)
        .open(directory.header())
        .unwrap()
        .set_len(1024 * 1024 + 1)
        .unwrap();
    assert!(load_distributed_header(&directory.0).is_err());
    std::fs::remove_file(directory.header()).unwrap();
    assert!(load_distributed_fragment(&directory.shard()).is_err());
}

#[test]
fn distributed_loader_accepts_valid_payloads_above_small_file_sizes() {
    let directory = Directory::new();
    let mut original = fixture();
    // The format permits a 1 MiB header and a 512 MiB shard. These modest
    // payloads exercise admission beyond tiny fixtures without large allocations.
    original.config.architecture = "a".repeat(4096);
    original.weights = vec![0.25; 256 * 1024];
    serialize_distributed(std::slice::from_ref(&original), &directory.0).unwrap();
    assert!(std::fs::metadata(directory.header()).unwrap().len() > 4096);
    assert!(std::fs::metadata(directory.shard()).unwrap().len() > 1_048_576);
    let (config, count) = load_distributed_header(&directory.0).unwrap();
    assert_eq!(config, original.config);
    assert_eq!(count, 1);
    let loaded = load_distributed_fragment(&directory.shard()).unwrap();
    assert_eq!(loaded.config, original.config);
    assert_eq!(loaded.weights, original.weights);
}

#[test]
fn distributed_tensor_count_is_inclusive_and_independent_of_available_bytes() {
    let directory = Directory::new();
    for count in [65_536, 65_537] {
        let mut original = fixture();
        // Repeated bindings are legal at the container layer; this test does
        // not assert that they form an executable model.
        original.tensors = (0..count)
            .map(|_| TensorMeta {
                name: "x".into(),
                offset: 0,
                n_rows: 2,
                n_cols: 2,
            })
            .collect();
        serialize_distributed(&[original], &directory.0).unwrap();
        let result = load_distributed_fragment(&directory.shard());
        if count == 65_536 {
            let loaded = result.unwrap();
            assert_eq!(loaded.tensors.len(), count);
            assert!(loaded.tensors.iter().all(|t| t.name == "x"));
        } else {
            assert_eq!(
                result.unwrap_err(),
                "distributed tensor count exceeds budget or available bytes"
            );
        }
    }
}
