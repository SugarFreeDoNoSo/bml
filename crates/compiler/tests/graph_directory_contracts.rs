//! Independent byte fixtures for the legacy directory container, not the
//! separate single-file BmlGraph format or a complete inference artifact.
use bml_compiler::{
    gguf_compiler::{
        load_from_dir, load_from_dir_with_limits, serialize_to_dir, CompilationResult,
        GraphLoadLimits, ModelConfig,
    },
    BmlGraph, Fragment, RpnOp, BMLGRAPH_MAGIC, BMLGRAPH_VERSION,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_graph_directory_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn put(&self, name: &str, bytes: &[u8]) {
        std::fs::write(self.0.join(name), bytes).unwrap();
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn header(fragments: u32, constants: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [BMLGRAPH_MAGIC, BMLGRAPH_VERSION, fragments] {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend(5u64.to_le_bytes());
    bytes.extend(b"llama");
    for value in [1u32, 2, 4, 8, 260] {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend((constants.len() as u64).to_le_bytes());
    for value in constants {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

fn fragment() -> (Vec<u8>, Vec<RpnOp>) {
    let ops = vec![
        RpnOp::One,
        RpnOp::Bml,
        RpnOp::Dup,
        RpnOp::Loop {
            count: 2,
            body_len: 3,
        },
        RpnOp::Var(4),
        RpnOp::Const(5),
        RpnOp::Zero,
        RpnOp::VarIndexed { base: 6 },
        RpnOp::StoreResult { slot: 7 },
        RpnOp::FAdd,
        RpnOp::FMul,
        RpnOp::Pick { depth: 8 },
        RpnOp::Drop,
        RpnOp::Swap,
    ];
    let mut bytes = 14u32.to_le_bytes().to_vec();
    for tag in 0..14u8 {
        bytes.push(tag);
        let arguments: &[u32] = match tag {
            3 => &[2, 3],
            4 => &[4],
            5 => &[5],
            7 => &[6],
            8 => &[7],
            11 => &[8],
            _ => &[],
        };
        for argument in arguments {
            bytes.extend(argument.to_le_bytes());
        }
    }
    (bytes, ops)
}

#[test]
fn directory_decodes_every_opcode_and_preserves_ieee_constant_bits() {
    let dir = Directory::new();
    let bits = [0, 1 << 63, 0x7ff0000000000000, 0x7ff8000000000042, 1];
    let head = header(1, &bits);
    let (bytes, expected) = fragment();
    dir.put("header.bmlgraph", &head);
    dir.put("fragment_0.bmlgraph", &bytes);
    let (graph, pool, config) = load_from_dir(&dir.0).unwrap();
    assert_eq!(graph.fragments[0].ops, expected);
    assert_eq!(pool.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    assert!(graph.const_pool.is_empty()); // Legacy API returns the pool separately.
    assert_eq!(config.architecture, "llama");
    assert_eq!(
        (
            config.n_layers,
            config.n_heads,
            config.n_embd,
            config.context_length,
            config.vocab_size
        ),
        (1, 2, 4, 8, 260)
    );
    let exported = Directory::new();
    serialize_to_dir(
        &CompilationResult {
            graph,
            const_pool: pool,
            config,
            num_fragments: 1,
        },
        &exported.0,
    )
    .unwrap();
    assert_eq!(
        std::fs::read(exported.0.join("header.bmlgraph")).unwrap(),
        head
    );
    assert_eq!(
        std::fs::read(exported.0.join("fragment_0.bmlgraph")).unwrap(),
        bytes
    );
}

#[test]
fn every_directory_record_truncation_and_trailing_byte_is_rejected() {
    let dir = Directory::new();
    let head = header(1, &[1.0f64.to_bits()]);
    let (frag, _) = fragment();
    dir.put("fragment_0.bmlgraph", &frag);
    for length in 0..head.len() {
        dir.put("header.bmlgraph", &head[..length]);
        assert!(load_from_dir(&dir.0).is_err(), "header {length}");
    }
    dir.put("header.bmlgraph", &head);
    for length in 0..frag.len() {
        dir.put("fragment_0.bmlgraph", &frag[..length]);
        assert!(load_from_dir(&dir.0).is_err(), "fragment {length}");
    }
    dir.put("fragment_0.bmlgraph", &[frag.as_slice(), &[0]].concat());
    assert!(load_from_dir(&dir.0).is_err());
    dir.put("fragment_0.bmlgraph", &frag);
    dir.put("header.bmlgraph", &[head.as_slice(), &[0]].concat());
    assert!(load_from_dir(&dir.0).is_err());
}

#[test]
fn malformed_counts_version_utf8_and_opcodes_fail_without_large_allocations() {
    let dir = Directory::new();
    let head = header(1, &[1.0f64.to_bits()]);
    let (frag, _) = fragment();
    dir.put("fragment_0.bmlgraph", &frag);
    for (offset, replacement) in [
        (0, 0u32.to_le_bytes().to_vec()),
        (4, 99u32.to_le_bytes().to_vec()),
        (8, u32::MAX.to_le_bytes().to_vec()),
        (12, u64::MAX.to_le_bytes().to_vec()),
        (20, vec![255]),
        (45, u64::MAX.to_le_bytes().to_vec()),
    ] {
        let mut bad = head.clone();
        bad[offset..offset + replacement.len()].copy_from_slice(&replacement);
        dir.put("header.bmlgraph", &bad);
        assert!(load_from_dir(&dir.0).is_err(), "offset {offset}");
    }
    dir.put("header.bmlgraph", &head);
    dir.put("fragment_0.bmlgraph", &u32::MAX.to_le_bytes());
    assert!(load_from_dir(&dir.0).is_err());
    let mut bad = frag;
    bad[4] = 255;
    dir.put("fragment_0.bmlgraph", &bad);
    assert!(load_from_dir(&dir.0).is_err());
    std::fs::remove_file(dir.0.join("fragment_0.bmlgraph")).unwrap();
    assert!(load_from_dir(&dir.0).is_err());
}

#[test]
fn truncated_constant_pool_is_rejected_before_reservation() {
    let dir = Directory::new();
    let mut head = header(0, &[1.0f64.to_bits()]);
    head[45..53].copy_from_slice(&2u64.to_le_bytes());
    dir.put("header.bmlgraph", &head);
    let error = match load_from_dir(&dir.0) {
        Ok(_) => panic!("truncated pool accepted"),
        Err(error) => error,
    };
    assert!(
        error.contains("constant count exceeds budget or available bytes"),
        "{error}"
    );
}

#[test]
fn directory_budgets_are_inclusive_and_accumulate_across_fragments() {
    let dir = Directory::new();
    let head = header(2, &[1.0f64.to_bits()]);
    let (frag, _) = fragment();
    dir.put("header.bmlgraph", &head);
    for i in 0..2 {
        dir.put(&format!("fragment_{i}.bmlgraph"), &frag);
    }
    let limits = GraphLoadLimits {
        max_total_bytes: head.len() + 2 * frag.len(),
        max_fragments: 2,
        max_operations: 28,
        max_constants: 1,
    };
    assert!(load_from_dir_with_limits(&dir.0, limits).is_ok());
    for rejected in [
        GraphLoadLimits {
            max_total_bytes: limits.max_total_bytes - 1,
            ..limits
        },
        GraphLoadLimits {
            max_fragments: 1,
            ..limits
        },
        GraphLoadLimits {
            max_operations: 27,
            ..limits
        },
        GraphLoadLimits {
            max_constants: 0,
            ..limits
        },
    ] {
        assert!(load_from_dir_with_limits(&dir.0, rejected).is_err());
    }
    // Sparse oversized files are rejected by metadata before reading their payload.
    std::fs::File::create(dir.0.join("header.bmlgraph"))
        .unwrap()
        .set_len(65 * 1024 * 1024)
        .unwrap();
    assert!(load_from_dir(&dir.0).is_err());
}

#[test]
fn empty_fragment_records_fit_the_exact_minimum_byte_budget() {
    // This decoder admits structural records independently of execution
    // preflight. A zero-op fragment still needs its four-byte count field.
    for count in [1u32, 3] {
        let dir = Directory::new();
        let head = header(count, &[]);
        dir.put("header.bmlgraph", &head);
        for i in 0..count {
            dir.put(&format!("fragment_{i}.bmlgraph"), &0u32.to_le_bytes());
        }
        let limits = GraphLoadLimits {
            max_total_bytes: head.len() + 4 * count as usize,
            max_fragments: count as usize,
            max_operations: 0,
            max_constants: 0,
        };
        let (graph, constants, _) = load_from_dir_with_limits(&dir.0, limits)
            .expect("exact minimum fragment-record budget must be inclusive");
        assert_eq!(graph.fragments.len(), count as usize);
        assert!(graph
            .fragments
            .iter()
            .all(|fragment| fragment.ops.is_empty()));
        assert!(constants.is_empty());
        assert!(load_from_dir_with_limits(
            &dir.0,
            GraphLoadLimits {
                max_total_bytes: limits.max_total_bytes - 1,
                ..limits
            }
        )
        .is_err());
    }
}

#[test]
fn serializer_rejects_inconsistent_fragment_count_before_writing() {
    let dir = Directory::new();
    let result = CompilationResult {
        graph: BmlGraph {
            fragments: vec![Fragment {
                ops: vec![RpnOp::One],
            }],
            const_pool: vec![],
            threshold: 0,
        },
        const_pool: vec![],
        config: ModelConfig {
            architecture: "llama".into(),
            n_layers: 0,
            n_heads: 1,
            n_embd: 1,
            context_length: 1,
            vocab_size: 1,
        },
        num_fragments: 2,
    };
    assert!(serialize_to_dir(&result, &dir.0).is_err());
    assert!(!dir.0.join("header.bmlgraph").exists());
}
