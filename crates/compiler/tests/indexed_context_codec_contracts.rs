use bml_compiler::{fragment::BMLGRAPH_CONTEXT_VERSION, BmlGraph, Fragment, RpnOp, RpnProgram};
use bml_domain::EvalContext;

fn graph(ops: Vec<RpnOp>) -> BmlGraph {
    BmlGraph {
        fragments: vec![Fragment { ops }],
        const_pool: vec![],
        threshold: 32768,
    }
}

#[test]
fn indexed_context_graphs_use_new_tags_and_version_without_changing_legacy_bytes() {
    let old = graph(vec![RpnOp::One]);
    let mut expected = vec![0x47, 0x42, 0x4d, 0x4c];
    expected.extend(2u32.to_le_bytes());
    expected.extend(1u32.to_le_bytes());
    expected.extend(1u32.to_le_bytes());
    expected.push(0);
    expected.extend(0u64.to_le_bytes());
    assert_eq!(old.serialize(), expected);
    let new = graph(vec![
        RpnOp::ContextVarIndexed { base: 7 },
        RpnOp::ContextConstIndexed { base: 9 },
    ]);
    let mut bytes = new.serialize();
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        BMLGRAPH_CONTEXT_VERSION
    );
    assert_eq!(&bytes[16..26], &[14, 7, 0, 0, 0, 15, 9, 0, 0, 0]);
    let decoded = BmlGraph::deserialize(&bytes, 32768).unwrap();
    assert_eq!(decoded.fragments[0].ops, new.fragments[0].ops);
    assert_eq!(decoded.serialize(), bytes);
    for version in [1u32, 2, 4] {
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        assert!(BmlGraph::deserialize(&bytes, 32768).is_err());
    }
    // Historical v1 lacks the v2 constant-pool trailer.
    let mut v1 = expected[..17].to_vec();
    v1[4..8].copy_from_slice(&1u32.to_le_bytes());
    assert_eq!(
        BmlGraph::deserialize(&v1, 32768).unwrap().fragments[0].ops,
        vec![RpnOp::One]
    );
}

#[test]
fn indexed_context_codec_rejects_all_truncated_prefixes_and_honors_budgets() {
    for op in [
        RpnOp::ContextVarIndexed { base: u32::MAX },
        RpnOp::ContextConstIndexed { base: u32::MAX },
    ] {
        let bytes = graph(vec![op]).serialize();
        for length in 0..bytes.len() {
            assert!(
                BmlGraph::deserialize(&bytes[..length], 32768).is_err(),
                "prefix {length}"
            );
        }
        assert!(BmlGraph::deserialize_with_limits(&bytes, 32768, 1, 1, 0).is_ok());
        assert!(BmlGraph::deserialize_with_limits(&bytes, 32768, 0, 1, 0).is_err());
        assert!(BmlGraph::deserialize_with_limits(&bytes, 32768, 1, 0, 0).is_err());
    }
}

#[test]
fn reference_evaluators_load_context_vectors_inside_loops_and_reject_lossy_indices() {
    // Historical reference loops intentionally retain their preexisting stack
    // semantics; only programs containing the explicit new loads opt in.
    let legacy = RpnProgram {
        ops: vec![
            RpnOp::One,
            RpnOp::Loop {
                count: 2,
                body_len: 2,
            },
            RpnOp::One,
            RpnOp::Bml,
        ],
        const_pool: vec![],
    };
    assert_eq!(legacy.evaluate(), 4.0);
    assert_eq!(graph(legacy.ops).evaluate(0.0), 4.0);
    let ops = vec![
        RpnOp::Loop {
            count: 2,
            body_len: 5,
        },
        RpnOp::Dup,
        RpnOp::ContextVarIndexed { base: 1 },
        RpnOp::Swap,
        RpnOp::ContextConstIndexed { base: 2 },
        RpnOp::Bml,
        RpnOp::Drop,
    ];
    let program = RpnProgram {
        ops: ops.clone(),
        const_pool: vec![],
    };
    let context = EvalContext::new(&[777.0, 2.0, 3.0], &[888.0, 999.0, 4.0, 8.0]);
    assert_eq!(program.evaluate_with_ctx(&context), 2.0);
    let mut stack = Vec::new();
    Fragment { ops }.evaluate_on_stack_with_ctx(&mut stack, &context);
    assert_eq!(stack, vec![2.0]);
    for offset in [-1.0, 0.5, 2.0, f64::INFINITY, f64::NAN, u32::MAX as f64] {
        for op in [
            RpnOp::ContextVarIndexed { base: 1 },
            RpnOp::ContextConstIndexed { base: 1 },
        ] {
            let program = RpnProgram {
                ops: vec![RpnOp::Var(0), op],
                const_pool: vec![],
            };
            let inputs = [offset, 123.0, 456.0];
            let context = EvalContext::new(&inputs, &[999.0, 123.0, 456.0]);
            assert!(program.evaluate_with_ctx(&context).is_nan());
        }
    }
    assert_eq!(
        bml_compiler::rpn::context_index(u32::MAX, 0.0),
        Some(u32::MAX)
    );
    assert_eq!(bml_compiler::rpn::context_index(u32::MAX, 1.0), None);
    assert_eq!(
        bml_compiler::rpn::context_index(0, u32::MAX as f64 + 1.0),
        None
    );
    assert_eq!(bml_compiler::rpn::context_index(1, -0.0), Some(1));
}

#[test]
fn legacy_directory_containers_reject_context_loads_before_creating_outputs() {
    use bml_compiler::{
        distributed::{
            serialize_distributed, serialize_sub_fragments, DistributedFragment, SubFragment,
        },
        gguf_compiler::{serialize_to_dir, CompilationResult, ModelConfig},
    };
    let output =
        std::env::temp_dir().join(format!("bml-indexed-unsupported-{}", std::process::id()));
    assert!(!output.exists());
    let config = ModelConfig {
        architecture: "llama".into(),
        n_layers: 0,
        n_heads: 1,
        n_embd: 1,
        context_length: 1,
        vocab_size: 1,
    };
    for op in [
        RpnOp::ContextVarIndexed { base: 0 },
        RpnOp::ContextConstIndexed { base: 0 },
    ] {
        let old = CompilationResult {
            graph: graph(vec![op]),
            const_pool: vec![],
            config: config.clone(),
            num_fragments: 1,
        };
        assert!(serialize_to_dir(&old, &output).is_err());
        assert!(!output.exists());
        let shard = DistributedFragment {
            fragment_id: 0,
            layer_start: 0,
            layer_end: 0,
            ops: vec![op],
            weights: vec![],
            tensors: vec![],
            config: config.clone(),
            n_kv_heads: 1,
            head_dim: 1,
        };
        let sub = SubFragment {
            fragment_id: 0,
            sub_id: 0,
            layer_start: 0,
            layer_end: 0,
            ops: vec![op],
            weight_refs: vec![],
            depends_on: vec![],
        };
        assert!(serialize_sub_fragments(&output, &shard, &[sub]).is_err());
        assert!(!output.exists());
        assert!(serialize_distributed(&[shard], &output).is_err());
        assert!(!output.exists());
    }
}
