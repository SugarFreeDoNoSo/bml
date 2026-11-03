//! Small GGUF execution-boundary regressions, not full-model inference validation.
use bml_compiler::distributed::{compile_distributed, compile_distributed_bml};
use bml_compiler::gguf_compiler::{compile_gguf, compile_gguf_fast, InferenceCompiler};
use bml_compiler::hardware::HardwareSpec;
use bml_parser::GgufParser;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn swiglu_matches_closed_form_values_without_a_scaled_sigmoid() {
    use bml_compiler::gguf_compiler::swiglu_scalar;
    let log_two = std::f64::consts::LN_2;
    // exp(ln 2)=2, so the sigmoid is exactly 2/3 or 1/3 mathematically.
    for (gate, up, expected) in [
        (log_two, 3.0, 2.0 * log_two),
        (-log_two, 3.0, -log_two),
        (log_two, -6.0, -4.0 * log_two),
        (-log_two, -6.0, 2.0 * log_two),
        (0.0, 9.0, 0.0),
        (1000.0, 0.5, 500.0),
        (-1000.0, 0.5, -0.0),
    ] {
        let actual = swiglu_scalar(gate, up);
        assert!((actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs().max(1.0));
    }
    assert_eq!(swiglu_scalar(-0.0, 1.0).to_bits(), (-0.0_f64).to_bits());
}

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
    embedding_with_head(embedding, vocabulary, dtype, data, None)
}

fn embedding_with_head(
    embedding: u32,
    vocabulary: u32,
    dtype: u32,
    data: &[u8],
    head: Option<(u32, &[u64], &[u8])>,
) -> Fixture {
    embedding_with_norm(embedding, vocabulary, dtype, data, head, None)
}

fn embedding_with_norm(
    embedding: u32,
    vocabulary: u32,
    dtype: u32,
    data: &[u8],
    head: Option<(u32, &[u64], &[u8])>,
    norm: Option<(f32, &[f32])>,
) -> Fixture {
    let mut bytes = header(
        1 + u64::from(head.is_some()) + u64::from(norm.is_some()),
        9 + u64::from(norm.is_some()),
    );
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
    if let Some((eps, _)) = norm {
        string(&mut bytes, "llama.attention.layer_norm_rms_epsilon");
        bytes.extend_from_slice(&6u32.to_le_bytes());
        bytes.extend_from_slice(&eps.to_le_bytes());
    }

    string(&mut bytes, "token_embd.weight");
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&u64::from(embedding).to_le_bytes());
    bytes.extend_from_slice(&u64::from(vocabulary).to_le_bytes());
    bytes.extend_from_slice(&dtype.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    let head_offset = data.len().div_ceil(32) * 32;
    if let Some((kind, dims, _)) = head {
        string(&mut bytes, "output.weight");
        bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dimension in dims {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&(head_offset as u64).to_le_bytes());
    }
    let norm_offset = head
        .map_or(data.len(), |(_, _, payload)| head_offset + payload.len())
        .div_ceil(32)
        * 32;
    if let Some((_, weights)) = norm {
        string(&mut bytes, "output_norm.weight");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(weights.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&(norm_offset as u64).to_le_bytes());
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    let data_start = bytes.len();
    bytes.extend_from_slice(data);
    if let Some((_, _, payload)) = head {
        bytes.extend(std::iter::repeat_n(0, head_offset - data.len()));
        bytes.extend_from_slice(payload);
    }
    if let Some((_, weights)) = norm {
        bytes.resize(data_start + norm_offset, 0);
        bytes.extend(weights.iter().flat_map(|v| v.to_le_bytes()));
    }
    Fixture::new(&bytes)
}

#[test]
fn final_rmsnorm_uses_metadata_epsilon_and_weights_before_both_output_paths() {
    let embeddings: Vec<u8> = [3.0f32, 4.0, 1.0, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let output: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    // mean([3,4]^2) + epsilon = 12.5 + 3.5 = 16, sqrt = 4.
    let fixture = embedding_with_norm(
        2,
        2,
        0,
        &embeddings,
        Some((0, &[2, 2], &output)),
        Some((3.5, &[2.0, -0.5])),
    );
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    assert_eq!(model.rms_norm_epsilon(), 3.5);
    assert_eq!(model.forward(&[0]).unwrap(), [1.5, -0.5]);
    let mut kv = bml_compiler::HashConsedKV::new(0, 1, 2, 8);
    assert_eq!(model.forward_cached(&[0], &mut kv).unwrap(), [1.5, -0.5]);
    // A second decode catches in-place weight/embedding mutation or double normalization.
    assert_eq!(model.forward_cached(&[0], &mut kv).unwrap(), [1.5, -0.5]);
    assert_eq!(model.get_embedding(0).unwrap(), [3.0, 4.0]);
}

#[test]
fn zero_model_dimensions_and_context_are_rejected_independently() {
    let source = embedding_fixture(2, 2, 0, &[0u8; 16]);
    let original = std::fs::read(&source.0).unwrap();
    for key in [
        "llama.attention.head_count",
        "llama.embedding_length",
        "llama.context_length",
    ] {
        let mut bytes = original.clone();
        let at = bytes
            .windows(key.len())
            .position(|w| w == key.as_bytes())
            .unwrap()
            + key.len();
        assert_eq!(&bytes[at..at + 4], &4u32.to_le_bytes());
        bytes[at + 4..at + 8].copy_from_slice(&0u32.to_le_bytes());
        std::fs::write(&source.0, bytes).unwrap();
        for mapped in [false, true] {
            let result = if mapped {
                InferenceCompiler::open_mapped(&source.0)
            } else {
                InferenceCompiler::open(&source.0)
            };
            assert_eq!(
                result
                    .err()
                    .expect("zero metadata must fail model admission"),
                "invalid model dimensions or context",
                "{key}, mapped={mapped}"
            );
        }
    }
    std::fs::write(&source.0, original).unwrap();
    assert!(InferenceCompiler::open(&source.0).is_ok());
    assert!(InferenceCompiler::open_mapped(&source.0).is_ok());
}

#[test]
fn invalid_final_norm_or_epsilon_is_rejected_before_execution() {
    let data = [0u8; 16];
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let fixture = embedding_with_norm(2, 2, 0, &data, None, Some((epsilon, &[1.0, 1.0])));
        let error = InferenceCompiler::open(&fixture.0)
            .err()
            .expect("invalid epsilon accepted");
        assert!(error.contains("epsilon"), "{error}");
    }
    for weights in [&[1.0][..], &[1.0, f32::NAN][..]] {
        let fixture = embedding_with_norm(2, 2, 0, &data, None, Some((0.125, weights)));
        let error = InferenceCompiler::open(&fixture.0)
            .err()
            .expect("invalid norm accepted");
        assert!(error.contains("output_norm.weight"), "{error}");
    }
    let fixture = embedding_with_norm(2, 2, 0, &data, None, Some((0.125, &[1.0, 1.0])));
    let mut encoded = std::fs::read(&fixture.0).unwrap();
    let key = b"llama.attention.layer_norm_rms_epsilon";
    let at = encoded
        .windows(key.len())
        .position(|part| part == key)
        .unwrap()
        + key.len();
    encoded[at..at + 4].copy_from_slice(&4u32.to_le_bytes()); // U32, not F32.
    std::fs::write(&fixture.0, encoded).unwrap();
    let error = InferenceCompiler::open(&fixture.0)
        .err()
        .expect("incorrect epsilon type accepted");
    assert!(error.contains("epsilon type"), "{error}");
    let fixture = embedding_fixture(2, 2, 0, &data);
    let mut encoded = std::fs::read(&fixture.0).unwrap();
    let key = b"llama.block_count";
    let at = encoded
        .windows(key.len())
        .position(|part| part == key)
        .unwrap()
        + key.len()
        + 4;
    encoded[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&fixture.0, encoded).unwrap();
    let error = InferenceCompiler::open(&fixture.0)
        .err()
        .expect("missing final norm accepted");
    assert!(error.contains("missing output_norm.weight"), "{error}");
}

#[test]
fn separate_output_head_is_used_by_both_legacy_forward_paths() {
    let embeddings: Vec<u8> = [1.0f32, -2.0, 0.5, 3.0, -4.0, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let head: Vec<u8> = [2.0f32, 1.0, -3.0, 4.0, 0.25, -0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let fixture = embedding_with_head(2, 3, 0, &embeddings, Some((0, &[2, 3], &head)));
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    // [0.5,3] projected onto the three independent output rows.
    assert_eq!(model.forward(&[1]).unwrap(), [4.0, 10.5, -1.375]);
    let mut kv = bml_compiler::HashConsedKV::new(0, 1, 2, 8);
    assert_eq!(
        model.forward_cached(&[1], &mut kv).unwrap(),
        [4.0, 10.5, -1.375]
    );
    assert!(model.weight_offsets().contains_key("output.weight"));
}

#[test]
fn malformed_or_unsupported_explicit_head_cannot_fall_back_to_embeddings() {
    let embeddings = [0u8; 24];
    for (dtype, dims) in [(0, [3, 2]), (3, [2, 3])] {
        let fixture = embedding_with_head(2, 3, 0, &embeddings, Some((dtype, &dims, &[0; 24])));
        let error = InferenceCompiler::open(&fixture.0)
            .err()
            .expect("invalid head was accepted");
        assert!(error.contains("output.weight"), "{error}");
    }
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
            .expect_err("an unsupported architecture must not become a graph");
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
            .expect_err("an unsupported architecture must not become a placeholder graph");
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
    assert!(model.vocab().encode("text").is_err());
    assert_eq!(model.get_embedding(0).unwrap(), [1.0, -2.0]);
    assert_eq!(model.get_embedding(1).unwrap(), [0.5, 3.0]);
    assert_eq!(model.get_embedding(2).unwrap(), [-4.0, 2.0]);

    // Reverse the aspect ratio to expose assumptions about square matrices.
    let data: Vec<_> = [1.0_f32, 2.0, 3.0, -2.0, 0.5, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let fixture = embedding_fixture(3, 2, 0, &data);
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    assert_eq!(model.get_embedding(0).unwrap(), [1.0, 2.0, 3.0]);
    assert_eq!(model.get_embedding(1).unwrap(), [-2.0, 0.5, 4.0]);
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
    assert_eq!(model.forward(&[1]).unwrap(), [-5.5, 9.25, 4.0]);
    assert_eq!(model.forward(&[0]).unwrap(), [5.0, -5.5, -8.0]);
    assert_eq!(model.forward(&[2]).unwrap(), [-8.0, 4.0, 20.0]);

    let data: Vec<_> = [1.0_f32, 2.0, 3.0, -2.0, 0.5, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let fixture = embedding_fixture(3, 2, 0, &data);
    let model = InferenceCompiler::open(&fixture.0).unwrap();
    assert_eq!(model.forward(&[1]).unwrap(), [11.0, 20.25]);
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
    assert_eq!(model.get_embedding(0).unwrap(), [1.0; 32]);
    assert_eq!(model.get_embedding(1).unwrap(), [-2.0; 32]);
    assert_eq!(model.get_embedding(2).unwrap(), [2.0; 32]);
    // Each logit is a sum of 32 equal products, independently known exactly.
    assert_eq!(model.forward(&[1]).unwrap(), [-64.0, 128.0, -128.0]);
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
    assert_eq!(model.get_embedding(0).unwrap(), first);
    assert_eq!(model.get_embedding(1).unwrap(), second);
    // Exact integer/half-integer arithmetic: squared norms 172 and 688;
    // cross product 336. This isolates decode/layout, not full inference.
    assert_eq!(model.forward(&[0]).unwrap(), [172.0, 336.0]);
    assert_eq!(model.forward(&[1]).unwrap(), [336.0, 688.0]);
}

#[test]
fn mapped_quantized_weights_preserve_projections_without_a_dense_matrix_pool() {
    let mut q4 = Vec::new();
    for (scale, nibble) in [(0x3c00u16, 0x99), (0x3800, 0x44), (0x4000, 0x99)] {
        q4.extend(scale.to_le_bytes());
        q4.extend([nibble; 16]);
    }
    let fixture = embedding_fixture(32, 3, 2, &q4);
    let mapped = InferenceCompiler::open_mapped(&fixture.0).unwrap();
    assert!(mapped.weight_pool().is_empty());
    assert_eq!(
        mapped.mapped_bytes() as u64,
        std::fs::metadata(&fixture.0).unwrap().len()
    );
    assert_eq!(mapped.get_embedding(1).unwrap(), [-2.0; 32]);
    assert!(mapped.get_embedding(3).is_err());
    assert_eq!(mapped.forward(&[1]).unwrap(), [-64.0, 128.0, -128.0]);
    assert_eq!(
        mapped.forward(&[0, 1]).unwrap(),
        InferenceCompiler::open(&fixture.0)
            .unwrap()
            .forward(&[0, 1])
            .unwrap()
    );

    // Independent GGML binary golden: identity embedding selects the first
    // coefficient of each Q6_K head row, without using our decoder as oracle.
    let mut embeddings = vec![0.0f32; 256 * 8];
    embeddings[0] = 1.0;
    let bytes: Vec<_> = embeddings.into_iter().flat_map(f32::to_le_bytes).collect();
    let fixture = embedding_with_head(
        256,
        8,
        0,
        &bytes,
        Some((
            14,
            &[256, 8],
            include_bytes!("../../gguf/tests/fixtures/q6-ggml.bin"),
        )),
    );
    let mapped = InferenceCompiler::open_mapped(&fixture.0).unwrap();
    let expected: Vec<_> = include_bytes!("../../gguf/tests/fixtures/q6-ggml.f32le")
        .chunks_exact(256 * 4)
        .map(|row| f64::from(f32::from_le_bytes(row[..4].try_into().unwrap())))
        .collect();
    assert_eq!(mapped.output_weight_name(), "output.weight");
    assert_eq!(mapped.forward(&[0]).unwrap(), expected);
    assert_eq!(
        mapped.forward(&[0]).unwrap(),
        InferenceCompiler::open(&fixture.0)
            .unwrap()
            .forward(&[0])
            .unwrap()
    );
    assert!(mapped.weight_pool().is_empty());
}
