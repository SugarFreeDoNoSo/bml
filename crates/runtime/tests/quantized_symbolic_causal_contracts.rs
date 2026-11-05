//! Quantized GGUF integration through complete persisted bounded-BML inference.
//! Q4/F32 representation equivalence is exact within each numerical backend;
//! this does not impose an unbudgeted native-versus-symbolic parity tolerance.
use bml_compiler::{
    gguf_compiler::InferenceCompiler,
    symbolic_artifact::{compile_directory_bounded, load_directory, BOUNDED_PROFILE},
    symbolic_bounded::{BoundedProfile, ProjectionBounds},
    symbolic_kernels::{CompiledKernel, KernelLimits, Operation},
    BmlGraph, HashConsedKV,
};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const WIDTH: usize = 32;
const QUERY_HEADS: u32 = 2;
const KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 16;
const CONTEXT: u32 = 3;
const IDS: [u32; 3] = [69, 70, 69];
// Each half is deliberately different. Both contain every nibble value, so
// interleaving half-blocks or changing signed scales affects the learned model.
const NIBBLES: [u8; WIDTH] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8,
    6, 4, 2, 0,
];

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "bml_quantized_causal_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Tensor {
    name: &'static str,
    dims: Vec<u64>,
    kind: u32,
    bytes: Vec<u8>,
}

fn gguf_string(value: &str) -> Vec<u8> {
    let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
    bytes.extend(value.as_bytes());
    bytes
}

fn gguf_array(kind: u32, values: Vec<Vec<u8>>) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend((values.len() as u64).to_le_bytes());
    bytes.extend(values.into_iter().flatten());
    bytes
}

fn literal_row(seed: usize, row: usize, changed: bool) -> ([u8; WIDTH], f32, u16) {
    let (half_scale, scale) = if (row + seed).is_multiple_of(2) {
        (0x2400_u16, 1.0_f32 / 64.0)
    } else {
        (0xa400_u16, -1.0_f32 / 64.0)
    };
    let mut nibbles = [0_u8; WIDTH];
    for (column, nibble) in nibbles.iter_mut().enumerate() {
        *nibble = NIBBLES[(column + 3 * row + seed) % WIDTH];
    }
    if changed && row == 0 {
        nibbles[0] ^= 8;
    }
    (nibbles, scale, half_scale)
}

fn literal_values(seed: usize, row: usize) -> Vec<f64> {
    let (nibbles, scale, _) = literal_row(seed, row, false);
    nibbles
        .into_iter()
        .map(|nibble| f64::from(scale * f32::from(i16::from(nibble) - 8)))
        .collect()
}

fn matrix(name: &'static str, rows: usize, seed: usize, quantized: bool, changed: bool) -> Tensor {
    let mut bytes = Vec::new();
    for row in 0..rows {
        // These paired literal values encode +1/64 and -1/64 in IEEE binary16.
        // The F32 oracle uses the numeric literals, never a production decoder.
        let (nibbles, scale, half_scale) =
            literal_row(seed, row, changed && name == "blk.0.ffn_down.weight");
        if quantized {
            bytes.extend(half_scale.to_le_bytes());
            // GGML axis zero is contiguous; low nibbles are columns 0..16,
            // high nibbles are columns 16..32, not adjacent column pairs.
            for column in 0..16 {
                bytes.push(nibbles[column] | (nibbles[column + 16] << 4));
            }
        } else {
            for nibble in nibbles {
                let value = scale * f32::from(i16::from(nibble) - 8);
                bytes.extend(value.to_le_bytes());
            }
        }
    }
    Tensor {
        name,
        dims: vec![WIDTH as u64, rows as u64],
        kind: if quantized { 2 } else { 0 },
        bytes,
    }
}

fn norm(name: &'static str) -> Tensor {
    Tensor {
        name,
        dims: vec![WIDTH as u64],
        kind: 0,
        bytes: [0.5_f32; WIDTH]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    }
}

fn write_model(path: &Path, quantized: bool, changed: bool) {
    let mut tokens = vec!["<unk>".into(), "<s>".into(), "</s>".into(), "▁".into()];
    tokens.extend((0..256).map(|byte| format!("<0x{byte:02X}>")));
    let vocab = tokens.len();
    let mut metadata = vec![
        ("general.architecture", 8_u32, gguf_string("llama")),
        ("tokenizer.ggml.model", 8, gguf_string("llama")),
        (
            "tokenizer.ggml.tokens",
            9,
            gguf_array(8, tokens.iter().map(|token| gguf_string(token)).collect()),
        ),
        (
            "tokenizer.ggml.scores",
            9,
            gguf_array(
                6,
                (0..vocab)
                    .map(|_| (-1_f32).to_le_bytes().to_vec())
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            9,
            gguf_array(
                5,
                (0..vocab)
                    .map(|id| {
                        let kind: i32 = match id {
                            0 => 2,
                            1 | 2 => 3,
                            3 => 1,
                            _ => 6,
                        };
                        kind.to_le_bytes().to_vec()
                    })
                    .collect(),
            ),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            6,
            0.5_f32.to_le_bytes().to_vec(),
        ),
        ("llama.rope.freq_base", 6, 16_f32.to_le_bytes().to_vec()),
    ];
    for (name, value) in [
        ("llama.block_count", 1),
        ("llama.attention.head_count", QUERY_HEADS),
        ("llama.attention.head_count_kv", KV_HEADS),
        ("llama.embedding_length", WIDTH as u32),
        ("llama.feed_forward_length", WIDTH as u32),
        ("llama.context_length", CONTEXT),
        ("llama.rope.dimension_count", HEAD_DIM),
        ("llama.vocab_size", vocab as u32),
        ("tokenizer.ggml.bos_token_id", 1),
        ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.unknown_token_id", 0),
    ] {
        metadata.push((name, 4, value.to_le_bytes().to_vec()));
    }
    let matrices = [
        ("token_embd.weight", vocab, 1),
        ("output.weight", vocab, 7),
        ("blk.0.attn_q.weight", WIDTH, 3),
        ("blk.0.attn_k.weight", HEAD_DIM as usize, 5),
        ("blk.0.attn_v.weight", HEAD_DIM as usize, 9),
        ("blk.0.attn_output.weight", WIDTH, 11),
        ("blk.0.ffn_gate.weight", WIDTH, 13),
        ("blk.0.ffn_up.weight", WIDTH, 15),
        ("blk.0.ffn_down.weight", WIDTH, 17),
    ];
    let mut tensors: Vec<_> = matrices
        .into_iter()
        .map(|(name, rows, seed)| matrix(name, rows, seed, quantized, changed))
        .collect();
    tensors.extend([
        norm("output_norm.weight"),
        norm("blk.0.attn_norm.weight"),
        norm("blk.0.ffn_norm.weight"),
    ]);
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (name, kind, value) in metadata {
        bytes.extend(gguf_string(name));
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    let mut data = Vec::new();
    for tensor in tensors {
        data.resize(data.len().div_ceil(32) * 32, 0);
        bytes.extend(gguf_string(tensor.name));
        bytes.extend((tensor.dims.len() as u32).to_le_bytes());
        for dimension in tensor.dims {
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(tensor.kind.to_le_bytes());
        bytes.extend((data.len() as u64).to_le_bytes());
        data.extend(tensor.bytes);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(data);
    std::fs::write(path, bytes).unwrap();
}

fn bits(values: &[f64]) -> Vec<u64> {
    assert!(!values.is_empty());
    assert!(values.iter().all(|value| value.is_finite()));
    values.iter().map(|value| value.to_bits()).collect()
}

fn cache_bits(cache: &HashConsedKV) -> Vec<u32> {
    let mut result = Vec::new();
    for position in 0..cache.current_pos() {
        for values in [cache.load_k(0, position, 0), cache.load_v(0, position, 0)] {
            assert_eq!(values.len(), HEAD_DIM as usize);
            assert!(values.iter().all(|value| value.is_finite()));
            result.extend(values.into_iter().map(f32::to_bits));
        }
    }
    result
}

fn cache() -> HashConsedKV {
    HashConsedKV::new(1, KV_HEADS, HEAD_DIM, CONTEXT)
}

fn execute_stored(kernel: &CompiledKernel, inputs: &[f64], external: &[f64]) -> Vec<f64> {
    assert_eq!(inputs.len(), kernel.signature.inputs);
    assert_eq!(external.len(), kernel.signature.external);
    assert_eq!(kernel.graphs.len(), 1);
    let mut constants = kernel.signature.fixed.clone();
    constants.extend_from_slice(external);
    let buffered = kernel.layout.uses_buffer();
    let mut buffer = if buffered {
        ResultBuffer::from_vec(
            1,
            kernel.signature.outputs,
            vec![f64::NAN; kernel.signature.outputs],
        )
        .unwrap()
    } else {
        ResultBuffer::new(0, 0)
    };
    let stored = &kernel.graphs[0];
    let graph = ValidatedGraph::new_with_external_constants(
        BmlGraph::deserialize(&stored.serialize(), stored.threshold).unwrap(),
        inputs.len(),
        constants.len(),
        usize::from(buffered),
        if buffered {
            kernel.signature.outputs
        } else {
            0
        },
        GraphLimits::default(),
    )
    .unwrap();
    let scalar = graph
        .execute_indexed_symbolic_with_constants(inputs, &constants, &mut buffer)
        .unwrap();
    let values = if buffered {
        buffer.into_vec()
    } else {
        vec![scalar]
    };
    assert!(values.iter().all(|value| value.is_finite()));
    values
}

fn assert_literal_rotary_frequency(artifact: &Path, cache: &HashConsedKV) {
    let loaded = load_directory(artifact, None).unwrap();
    let kernel = |operation, width| {
        loaded
            .kernels
            .iter()
            .find(|kernel| kernel.operation == operation && kernel.width == width)
            .unwrap()
    };
    // The repeated token at positions 0 and 2 has the same one-layer input.
    // Reuse preceding persisted BML programs to isolate frequency selection
    // from native-versus-BML arithmetic and its precision budget.
    let normalized = execute_stored(
        kernel(Operation::Norm, WIDTH),
        &literal_values(1, IDS[2] as usize),
        &[0.5; WIDTH],
    );
    let pair: Vec<_> = [8, 9]
        .into_iter()
        .map(|row| {
            execute_stored(
                kernel(Operation::Dot, WIDTH),
                &normalized,
                &literal_values(5, row),
            )[0]
        })
        .collect();
    // K[8,9]: position / base^(within/head_dim) = 2 / 16^(8/16) = 0.5.
    // The literal angle is independent of rotate's indexing and arithmetic.
    let angle = 0.5_f64;
    let expected = execute_stored(
        kernel(Operation::Rope, 2),
        &pair,
        &[angle.cos(), angle.sin()],
    );
    let actual = cache.load_k(0, 2, 0);
    assert_eq!(
        actual[8..10]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| (*value as f32).to_bits())
            .collect::<Vec<_>>()
    );
}

fn persisted(source: &Path, directory: &Path) -> SymbolicCausalModel {
    std::fs::create_dir(directory).unwrap();
    let profile = BoundedProfile {
        down_projection: Some(ProjectionBounds {
            input: 64.0,
            weight: 1.0,
        }),
        ..BoundedProfile::default()
    };
    let manifest =
        compile_directory_bounded(source, directory, CONTEXT, profile, KernelLimits::default())
            .unwrap();
    assert_eq!(manifest.version, 4);
    assert_eq!(manifest.profile, BOUNDED_PROFILE);
    assert_eq!(manifest.bounded_profile, Some(profile));
    let loaded = load_directory(directory, None).unwrap();
    assert!(loaded.compiler.mapped_bytes() > 0);
    assert!(loaded
        .kernels
        .iter()
        .flat_map(|kernel| &kernel.graphs)
        .all(|graph| graph.const_pool.is_empty()));
    drop(loaded);
    let model = SymbolicCausalModel::open_artifact(
        directory,
        None,
        SymbolicLimits {
            max_nodes: 0,
            ..SymbolicLimits::default()
        },
    )
    .unwrap();
    assert_eq!(model.context_limit(), CONTEXT);
    assert!(model.executed_operations().is_empty());
    model
}

#[test]
fn persisted_q4_causal_model_matches_literal_f32_representation_across_gqa_positions() {
    let directory = Directory::new();
    let q4 = directory.0.join("quantized.gguf");
    let f32 = directory.0.join("literal-f32.gguf");
    let changed = directory.0.join("changed-down.gguf");
    // Finish every write before any source is borrowed through mmap. The
    // sensitivity variant changes a low nibble of one MLP down-projection row.
    write_model(&q4, true, false);
    write_model(&f32, false, false);
    write_model(&changed, true, true);
    let native_q4 = InferenceCompiler::open_mapped(&q4).unwrap();
    let native_f32 = InferenceCompiler::open_mapped(&f32).unwrap();
    assert_eq!(native_q4.config().n_embd, WIDTH as u32);
    assert_eq!(native_q4.config().n_heads, QUERY_HEADS);
    assert_eq!(native_q4.n_kv_heads(), KV_HEADS);
    assert_eq!(native_q4.head_dim(), HEAD_DIM);
    // Literal anchors also expose an accidentally interleaved half-block.
    let embedding = native_q4.get_embedding(0).unwrap();
    assert_eq!(embedding[0], 7.0 / 64.0);
    assert_eq!(embedding[16], -5.0 / 64.0);
    assert_ne!(
        bits(&native_q4.get_embedding(IDS[0]).unwrap()),
        bits(&native_q4.get_embedding(IDS[1]).unwrap())
    );
    let mut symbolic_q4 = persisted(&q4, &directory.0.join("q4-aot"));
    let mut symbolic_f32 = persisted(&f32, &directory.0.join("f32-aot"));
    let mut native_q4_cache = cache();
    let mut native_f32_cache = cache();
    let mut symbolic_q4_cache = cache();
    let mut symbolic_f32_cache = cache();
    let mut native_vectors = Vec::new();
    let mut symbolic_vectors = Vec::new();
    for (position, id) in IDS.into_iter().enumerate() {
        let previous_native = cache_bits(&native_q4_cache);
        let previous_symbolic = cache_bits(&symbolic_q4_cache);
        let native = native_q4
            .forward_cached(&[id], &mut native_q4_cache)
            .unwrap();
        let native_reference = native_f32
            .forward_cached(&[id], &mut native_f32_cache)
            .unwrap();
        assert_eq!(native.len(), 260);
        assert_eq!(bits(&native), bits(&native_reference));
        let symbolic = symbolic_q4
            .forward_cached(&[id], &mut symbolic_q4_cache)
            .unwrap();
        let symbolic_reference = symbolic_f32
            .forward_cached(&[id], &mut symbolic_f32_cache)
            .unwrap();
        assert_eq!(symbolic.len(), native.len());
        assert_eq!(bits(&symbolic), bits(&symbolic_reference));
        for committed in [
            &native_q4_cache,
            &native_f32_cache,
            &symbolic_q4_cache,
            &symbolic_f32_cache,
        ] {
            assert_eq!(committed.current_pos(), position as u32 + 1);
        }
        let native_state = cache_bits(&native_q4_cache);
        let symbolic_state = cache_bits(&symbolic_q4_cache);
        assert_eq!(native_state, cache_bits(&native_f32_cache));
        assert_eq!(symbolic_state, cache_bits(&symbolic_f32_cache));
        assert!(native_state.starts_with(&previous_native));
        assert!(symbolic_state.starts_with(&previous_symbolic));
        native_vectors.push(bits(&native));
        symbolic_vectors.push(bits(&symbolic));
    }
    // The repeated token has a different causal history and rotary position.
    assert_ne!(native_vectors[0], native_vectors[2]);
    assert_ne!(symbolic_vectors[0], symbolic_vectors[2]);
    assert_literal_rotary_frequency(&directory.0.join("q4-aot"), &symbolic_q4_cache);
    assert_ne!(
        native_q4_cache.load_k(0, 0, 0),
        native_q4_cache.load_k(0, 2, 0)
    );
    assert_ne!(
        symbolic_q4_cache.load_k(0, 0, 0),
        symbolic_q4_cache.load_k(0, 2, 0)
    );

    let mut native_prefill_cache = cache();
    native_q4
        .forward_cached(&IDS[..2], &mut native_prefill_cache)
        .unwrap();
    assert_eq!(native_prefill_cache.current_pos(), 2);
    let prefill_native = native_q4
        .forward_cached(&IDS[2..], &mut native_prefill_cache)
        .unwrap();
    assert_eq!(bits(&prefill_native), native_vectors[2]);
    assert_eq!(
        cache_bits(&native_prefill_cache),
        cache_bits(&native_q4_cache)
    );
    assert_eq!(bits(&native_q4.forward(&IDS).unwrap()), native_vectors[2]);
    let mut symbolic_prefill_cache = cache();
    symbolic_q4
        .forward_cached(&IDS[..2], &mut symbolic_prefill_cache)
        .unwrap();
    assert_eq!(symbolic_prefill_cache.current_pos(), 2);
    let prefill_symbolic = symbolic_q4
        .forward_cached(&IDS[2..], &mut symbolic_prefill_cache)
        .unwrap();
    assert_eq!(bits(&prefill_symbolic), symbolic_vectors[2]);
    assert_eq!(
        cache_bits(&symbolic_prefill_cache),
        cache_bits(&symbolic_q4_cache)
    );
    assert_eq!(
        bits(&symbolic_q4.forward(&IDS).unwrap()),
        symbolic_vectors[2]
    );
    for model in [&symbolic_q4, &symbolic_f32] {
        for family in [
            "dot",
            "down_dot",
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
                model.executed_operations().contains(&family),
                "missing {family}"
            );
        }
    }

    let changed_native = InferenceCompiler::open_mapped(&changed).unwrap();
    assert_ne!(
        bits(&changed_native.forward(&IDS[..1]).unwrap()),
        native_vectors[0]
    );
    let mut changed_symbolic = persisted(&changed, &directory.0.join("changed-aot"));
    assert_ne!(
        bits(&changed_symbolic.forward(&IDS[..1]).unwrap()),
        symbolic_vectors[0]
    );
}
