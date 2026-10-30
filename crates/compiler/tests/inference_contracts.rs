//! Small GGUF execution-boundary regressions, not full-model inference validation.
use bml_compiler::distributed::{compile_distributed, compile_distributed_bml};
use bml_compiler::gguf_compiler::{compile_gguf, compile_gguf_fast, InferenceCompiler};
use bml_compiler::hardware::HardwareSpec;
use bml_parser::GgufParser;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);

impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        static ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_inference_contract_{}_{}.gguf",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn header(tensors: u64, metadata: u64) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&tensors.to_le_bytes());
    bytes.extend_from_slice(&metadata.to_le_bytes());
    bytes
}

fn string_entry(bytes: &mut Vec<u8>, name: &str, value: &str) {
    string(bytes, name);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    string(bytes, value);
}

fn u32_entry(bytes: &mut Vec<u8>, name: &str, value: u32) {
    string(bytes, name);
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn architecture_only(architecture: &str) -> Fixture {
    let mut bytes = header(0, 1);
    string_entry(&mut bytes, "general.architecture", architecture);
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    Fixture::new(&bytes)
}

fn embedding_fixture(embedding: u32, vocabulary: u32, dtype: u32, data: &[u8]) -> Fixture {
    let mut bytes = header(1, 9);
    string_entry(&mut bytes, "general.architecture", "llama");
    u32_entry(&mut bytes, "llama.block_count", 0);
    u32_entry(&mut bytes, "llama.attention.head_count", 1);
    u32_entry(&mut bytes, "llama.embedding_length", embedding);
    u32_entry(&mut bytes, "llama.context_length", 8);
    u32_entry(&mut bytes, "llama.vocab_size", vocabulary);
    string_entry(&mut bytes, "tokenizer.ggml.model", "llama");
    u32_entry(&mut bytes, "tokenizer.ggml.bos_token_id", 0);
    string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9_u32.to_le_bytes()); // Array
    bytes.extend_from_slice(&8_u32.to_le_bytes()); // String elements
    bytes.extend_from_slice(&u64::from(vocabulary).to_le_bytes());
    for id in 0..vocabulary {
        string(&mut bytes, &format!("token_{id}"));
    }

    string(&mut bytes, "token_embd.weight");
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&u64::from(embedding).to_le_bytes());
    bytes.extend_from_slice(&u64::from(vocabulary).to_le_bytes());
    bytes.extend_from_slice(&dtype.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend_from_slice(data);
    Fixture::new(&bytes)
}

#[test]
fn unsupported_architecture_is_rejected_before_tokenizer_or_weight_loading() {
    for architecture in ["gemma4", "unrecognized-model"] {
        // Missing tokenizer and dimensions must not hide the architecture error.
        let fixture = architecture_only(architecture);
        let parser = GgufParser::open(&fixture.0).unwrap();
        assert_eq!(parser.architecture(), Some(architecture));
        let error = InferenceCompiler::open(&fixture.0)
            .err()
            .expect("an unsupported architecture must not enter inference");
        assert!(error.contains(architecture), "{error}");
        assert!(error.contains("unsupported"), "{error}");
    }
}

#[test]
fn unsupported_architecture_cannot_be_compiled_as_a_llama_graph() {
    let hardware = HardwareSpec::new(1, 32 * 1024, 256 * 1024, 1024 * 1024);
    for architecture in ["gemma4", "unrecognized-model"] {
        let fixture = architecture_only(architecture);
        let error = compile_gguf(&fixture.0, &hardware)
            .err()
            .expect("an unsupported architecture must not become a graph");
        assert!(error.contains(architecture), "{error}");
        assert!(error.contains("unsupported"), "{error}");
    }
}

#[test]
fn unsupported_architecture_cannot_bypass_validation_in_fast_compilation() {
    let hardware = HardwareSpec::new(1, 32 * 1024, 256 * 1024, 1024 * 1024);
    for architecture in ["gemma4", "unrecognized-model"] {
        let fixture = architecture_only(architecture);
        let error = compile_gguf_fast(&fixture.0, &hardware)
            .err()
            .expect("an unsupported architecture must not become a placeholder graph");
        assert!(error.contains(architecture), "{error}");
        assert!(error.contains("unsupported"), "{error}");
    }
}

#[test]
fn unsupported_architecture_is_rejected_by_distributed_compiler() {
    for architecture in ["gemma4", "unrecognized-model"] {
        let fixture = architecture_only(architecture);
        let error = compile_distributed(&fixture.0, 1, true)
            .expect_err("an unsupported architecture must not become distributed fragments");
        assert!(error.contains(architecture), "{error}");
        assert!(error.contains("unsupported"), "{error}");
    }
}

#[test]
fn unsupported_architecture_is_rejected_by_bml_encoded_distributed_compiler() {
    for architecture in ["gemma4", "unrecognized-model"] {
        let fixture = architecture_only(architecture);
        let error = compile_distributed_bml(&fixture.0, 1)
            .err()
            .expect("an unsupported architecture must not become encoded fragments");
        assert!(error.contains(architecture), "{error}");
        assert!(error.contains("unsupported"), "{error}");
    }
}

#[test]
fn rectangular_embeddings_use_contiguous_ne0_values_for_each_token() {
    // GGUF [ne0=2, ne1=3]: three contiguous two-component token vectors.
    let data: Vec<_> = [1.0_f32, -2.0, 0.5, 3.0, -4.0, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let fixture = embedding_fixture(2, 3, 0, &data);
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    assert_eq!(model.get_embedding(0), [1.0, -2.0]);
    assert_eq!(model.get_embedding(1), [0.5, 3.0]);
    assert_eq!(model.get_embedding(2), [-4.0, 2.0]);

    // Reverse the aspect ratio to expose assumptions about square matrices.
    let data: Vec<_> = [1.0_f32, 2.0, 3.0, -2.0, 0.5, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let fixture = embedding_fixture(3, 2, 0, &data);
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    assert_eq!(model.get_embedding(0), [1.0, 2.0, 3.0]);
    assert_eq!(model.get_embedding(1), [-2.0, 0.5, 4.0]);
}

#[test]
fn tied_embedding_projection_matches_independent_rectangular_dot_products() {
    // Zero transformer layers isolate the embedding lookup and tied output
    // projection; these fixtures do not claim a valid full transformer model.
    let data: Vec<_> = [1.0_f32, -2.0, 0.5, 3.0, -4.0, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let fixture = embedding_fixture(2, 3, 0, &data);
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    // [0.5, 3] dotted with [1,-2], [0.5,3], [-4,2].
    assert_eq!(model.forward(&[1]), [-5.5, 9.25, 4.0]);
    assert_eq!(model.forward(&[0]), [5.0, -5.5, -8.0]);
    assert_eq!(model.forward(&[2]), [-8.0, 4.0, 20.0]);

    let data: Vec<_> = [1.0_f32, 2.0, 3.0, -2.0, 0.5, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let fixture = embedding_fixture(3, 2, 0, &data);
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    assert_eq!(model.forward(&[1]), [11.0, 20.25]);
}

#[test]
fn q4_embedding_columns_keep_their_own_blocks_and_scales() {
    // Three GGML Q4_0 blocks, each containing one token's 32 values:
    // 1 * (9-8) = 1; 0.5 * (4-8) = -2; 2 * (9-8) = 2.
    let mut data = Vec::new();
    for (half_scale, nibbles) in [(0x3c00_u16, 0x99), (0x3800, 0x44), (0x4000, 0x99)] {
        data.extend_from_slice(&half_scale.to_le_bytes());
        data.extend_from_slice(&[nibbles; 16]);
    }
    let fixture = embedding_fixture(32, 3, 2, &data);
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    assert_eq!(model.get_embedding(0), [1.0; 32]);
    assert_eq!(model.get_embedding(1), [-2.0; 32]);
    assert_eq!(model.get_embedding(2), [2.0; 32]);
    // Each logit is a sum of 32 equal products, independently known exactly.
    assert_eq!(model.forward(&[1]), [-64.0, 128.0, -128.0]);
}

#[test]
fn q4_nonuniform_nibbles_preserve_half_block_order_and_signed_scales() {
    // Independent Q4_0 oracle: low nibbles precede high nibbles, rather
    // than being interleaved. Every nibble value occurs in each block.
    let mut data = Vec::new();
    for scale in [0x3800_u16, 0xbc00] {
        // f16: +0.5 and -1
        data.extend_from_slice(&scale.to_le_bytes());
        for i in 0_u8..16 {
            let packed = if scale == 0x3800 {
                i | ((15 - i) << 4)
            } else {
                (15 - i) | (i << 4)
            };
            data.push(packed);
        }
    }
    let fixture = embedding_fixture(32, 2, 2, &data);
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    let first: Vec<f64> = (0..16)
        .map(|i| (i as f64 - 8.0) * 0.5)
        .chain((0..16).map(|i| (7.0 - i as f64) * 0.5))
        .collect();
    let second: Vec<f64> = (0..16)
        .map(|i| i as f64 - 7.0)
        .chain((0..16).map(|i| 8.0 - i as f64))
        .collect();
    assert_eq!(model.get_embedding(0), first);
    assert_eq!(model.get_embedding(1), second);
    // Exact integer/half-integer arithmetic: squared norms 172 and 688;
    // cross product 336. This isolates decode/layout, not full inference.
    assert_eq!(model.forward(&[0]), [172.0, 336.0]);
    assert_eq!(model.forward(&[1]), [336.0, 688.0]);
}
