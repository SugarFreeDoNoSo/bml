//! Independent scalar-oracle regressions for the validated Llama CPU profile.
//!
//! Golden values below were calculated in f64 by a separate full-sequence
//! Python implementation: dense causal attention matrices for every layer,
//! rather than the production incremental KV algorithm. Power-of-two fixture
//! coefficients have exact f32 representations. The residual-only case also
//! has a closed-form result derived directly from the embedding and head.

use bml_gguf::GgufDataType;
use bml_inference::model::{Config, Model, NativeToken, Operation, Session};
use bml_inference::tokenizer::TokenizerSpec;
use bml_inference::{Result, Tensor};

const VOCAB: usize = 262;
const TOKENS: [u32; 3] = [259, 260, 261];

struct Weight {
    name: String,
    dims: Vec<u64>,
    dtype: GgufDataType,
    bytes: Vec<u8>,
}

struct Fixture {
    config: Config,
    weights: Vec<Weight>,
}

fn vocabulary() -> TokenizerSpec {
    let mut tokens = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    let mut token_types = vec![2, 3, 3];
    for byte in 0..256 {
        tokens.push(format!("<0x{byte:02X}>"));
        token_types.push(6);
    }
    tokens.extend(["▁one".into(), "▁two".into(), "▁three".into()]);
    token_types.extend([1, 1, 1]);
    TokenizerSpec {
        scores: vec![-1.0; tokens.len()],
        tokens,
        token_types,
        bos_id: 1,
        eos_id: 2,
        unk_id: 0,
        add_bos: true,
        add_eos: false,
        add_space_prefix: true,
    }
}

impl Fixture {
    fn model(&self) -> Result<Model<'_>> {
        Model::from_tensors(
            self.config.clone(),
            vocabulary(),
            self.weights
                .iter()
                .map(|weight| Tensor {
                    name: weight.name.clone(),
                    dims: weight.dims.clone(),
                    dtype: weight.dtype,
                    data: &weight.bytes,
                })
                .collect(),
        )
    }

    fn add(&mut self, name: &str, dims: &[usize], values: Vec<f32>) {
        assert_eq!(dims.iter().product::<usize>(), values.len());
        self.weights.push(Weight {
            name: name.into(),
            dims: dims.iter().map(|&n| n as u64).collect(),
            dtype: GgufDataType::F32,
            bytes: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        });
    }

    fn weight(&mut self, name: &str) -> &mut Weight {
        self.weights.iter_mut().find(|w| w.name == name).unwrap()
    }

    fn empty_projections(width: usize, heads: usize, kv_heads: usize, ff: usize) -> Self {
        let mut fixture = Self {
            config: Config {
                embedding_length: width,
                block_count: 2,
                head_count: heads,
                head_count_kv: kv_heads,
                feed_forward_length: ff,
                context_length: 8,
                vocab_size: VOCAB,
                rope_dimension_count: width / heads,
                rope_freq_base: 16.0,
                rms_norm_eps: 0.125,
            },
            weights: Vec::new(),
        };
        fixture.add(
            "token_embd.weight",
            &[width, VOCAB],
            vec![0.0; width * VOCAB],
        );
        fixture.add("output.weight", &[width, VOCAB], vec![0.0; width * VOCAB]);
        fixture.add("output_norm.weight", &[width], vec![1.0; width]);
        let kv_width = width / heads * kv_heads;
        for layer in 0..2 {
            for role in ["attn_norm", "ffn_norm"] {
                fixture.add(
                    &format!("blk.{layer}.{role}.weight"),
                    &[width],
                    vec![1.0; width],
                );
            }
            for (role, input, output) in [
                ("attn_q", width, width),
                ("attn_k", width, kv_width),
                ("attn_v", width, kv_width),
                ("attn_output", width, width),
                ("ffn_gate", width, ff),
                ("ffn_up", width, ff),
                ("ffn_down", ff, width),
            ] {
                fixture.add(
                    &format!("blk.{layer}.{role}.weight"),
                    &[input, output],
                    vec![0.0; input * output],
                );
            }
        }
        fixture
    }

    fn tiny(heads: usize, kv_heads: usize, residual_only: bool) -> Self {
        let mut fixture = Self::empty_projections(4, heads, kv_heads, 3);
        let mut embeddings = vec![0.0; 4 * VOCAB];
        embeddings[259 * 4..].copy_from_slice(&[
            0.5, -1.0, 1.5, 2.0, -1.5, 0.25, 2.0, -0.75, 1.0, 0.5, -0.25, -2.0,
        ]);
        fixture.weight("token_embd.weight").bytes = bytes(&embeddings);
        let mut output = vec![0.0; 4 * VOCAB];
        output[259 * 4..].copy_from_slice(&[
            -0.5, 0.25, 1.0, -0.75, 1.0, -2.0, 0.5, 0.25, -1.0, 0.75, -0.25, 1.5,
        ]);
        fixture.weight("output.weight").bytes = bytes(&output);
        fixture.weight("output_norm.weight").bytes = bytes(&[1.25, 0.75, 0.5, 1.5]);
        for layer in 0..2 {
            fixture
                .weight(&format!("blk.{layer}.attn_norm.weight"))
                .bytes = bytes(&[1.0, 0.75, 1.25, 0.5]);
            fixture
                .weight(&format!("blk.{layer}.ffn_norm.weight"))
                .bytes = bytes(&[0.875, 1.125, 1.0, 1.25]);
            if !residual_only {
                for (role, seed) in [
                    ("attn_q", 1),
                    ("attn_k", 2),
                    ("attn_v", 3),
                    ("attn_output", 4),
                    ("ffn_gate", 5),
                    ("ffn_up", 6),
                    ("ffn_down", 7),
                ] {
                    let weight = fixture.weight(&format!("blk.{layer}.{role}.weight"));
                    let values: Vec<_> = (0..weight.bytes.len() / 4)
                        .map(|index| ((index * 7 + seed + layer * 3) % 17) as f32 / 16.0 - 0.5)
                        .collect();
                    weight.bytes = bytes(&values);
                }
            }
        }
        fixture
    }
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn assert_logits(logits: &[f32], expected: [f64; 3]) {
    assert_eq!(logits.len(), VOCAB);
    assert!(logits[..259].iter().all(|&value| value == 0.0));
    for (&actual, expected) in logits[259..].iter().zip(expected) {
        assert!(
            (f64::from(actual) - expected).abs() < 2e-5,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn residuals_final_norm_and_untied_head_match_closed_form() {
    let fixture = Fixture::tiny(2, 1, true);
    let model = fixture.model().unwrap();
    // Zero attention/MLP outputs must leave [0.5,-1,1.5,2] untouched.
    // mean square + epsilon = 7.5/4 + 0.125 = 2.
    // Learned final norm and the separate head give these exact dot products.
    let sqrt_two = 2.0_f64.sqrt();
    let logits = model.session(4).unwrap().decode(TOKENS[0]).unwrap();
    assert_logits(
        &logits,
        [-2.0 / sqrt_two, 3.25 / sqrt_two, 3.125 / sqrt_two],
    );
}

#[test]
fn absent_output_head_uses_tied_embedding_after_final_norm() {
    let mut fixture = Fixture::tiny(2, 1, true);
    fixture.weights.retain(|w| w.name != "output.weight");
    let model = fixture.model().unwrap();
    let logits = model.session(4).unwrap().decode(TOKENS[0]).unwrap();
    let root = 2.0_f64.sqrt();
    assert_logits(&logits, [8.1875 / root, -1.875 / root, -5.9375 / root]);
}

#[test]
fn two_layer_gqa_matches_independent_full_sequence_oracle() {
    let fixture = Fixture::tiny(2, 1, false);
    let model = fixture.model().unwrap();
    let expected = [
        [-1.746476876880815, 2.5002479441475693, 2.3770837468960226],
        [0.6890248253978942, 0.8892841969402803, 0.0987847915851895],
        [0.7962413170399363, 1.0211478582351188, -4.582839687788002],
    ];
    let mut incremental = model.session(4).unwrap();
    assert_eq!(incremental.context(), 4);
    for (index, &token) in TOKENS.iter().enumerate() {
        let decoded = incremental.decode(token).unwrap();
        assert_logits(&decoded, expected[index]);
        let batched = model
            .session(4)
            .unwrap()
            .prefill(&TOKENS[..=index])
            .unwrap();
        assert_eq!(decoded, batched);
    }
    assert_eq!(incremental.position(), 3);
    assert_eq!(incremental.kv_bytes(), 2 * 2 * 4 * 2 * 4);
    let memory = model.memory_plan(4).unwrap();
    assert_eq!(
        memory.encoded_weight_bytes,
        fixture
            .weights
            .iter()
            .map(|weight| weight.bytes.len())
            .sum::<usize>()
    );
    assert_eq!(memory.kv_bytes, incremental.kv_bytes());
    assert_eq!(memory.decoded_row_bytes, 16);
    assert!(memory.scratch_bytes_upper_bound >= VOCAB * 4);
    assert_eq!(model.plan().first(), Some(&Operation::Embedding));
    assert_eq!(model.plan().last(), Some(&Operation::Logits));
}

#[test]
fn rotary_frequencies_honor_nondefault_base_and_position() {
    // A four-dimensional head has a second pair with angle pos/sqrt(16).
    // Interleaving heads, using base 10000, or skipping positions changes this.
    let fixture = Fixture::tiny(1, 1, false);
    let model = fixture.model().unwrap();
    let logits = model.session(4).unwrap().prefill(&TOKENS).unwrap();
    assert_logits(
        &logits,
        [1.1293451437293034, -0.002498090084058855, -4.50064472264914],
    );
}

#[test]
fn session_rejects_overflow_and_invalid_batches_without_committing_tokens() {
    let fixture = Fixture::tiny(2, 1, false);
    let model = fixture.model().unwrap();
    assert!(model.session(0).is_err());
    assert!(model.session(9).is_err());
    let mut session = model.session(2).unwrap();
    session.decode(TOKENS[0]).unwrap();
    assert!(session.prefill(&TOKENS[1..]).is_err());
    assert!(session.prefill(&[u32::MAX]).is_err());
    assert!(session.prefill(&[]).is_err());
    assert_eq!(session.position(), 1);
    let final_logits = session.decode(TOKENS[1]).unwrap();
    let fresh = model.session(2).unwrap().prefill(&TOKENS[..2]).unwrap();
    assert_eq!(final_logits, fresh);
    assert!(session.decode(TOKENS[2]).is_err());
    session.reset();
    assert_eq!(session.position(), 0);
    let reset = session.decode(TOKENS[2]).unwrap();
    let fresh = model.session(2).unwrap().decode(TOKENS[2]).unwrap();
    assert_eq!(reset, fresh);
    let mut session = model.session(4).unwrap();
    assert!(session.prefill(&[TOKENS[0], u32::MAX]).is_err());
    assert_eq!(session.position(), 0);
}

#[test]
fn arithmetic_failure_rolls_back_the_whole_prefill_batch() {
    let mut fixture = Fixture::tiny(2, 1, true);
    fixture.weight("blk.0.ffn_gate.weight").bytes = bytes(&[f32::MAX; 12]);
    let model = fixture.model().unwrap();
    let mut session = model.session(4).unwrap();
    // Token zero has zero embeddings; the first token succeeds. The second
    // overflows a projection despite all stored weights being finite.
    assert!(session.prefill(&[0, TOKENS[0]]).is_err());
    assert_eq!(session.position(), 0);
    assert_eq!(
        session.decode(0).unwrap(),
        model.session(4).unwrap().decode(0).unwrap()
    );
    assert_eq!(session.position(), 1);
}

fn native_two_layers(session: &mut Session<'_>, token: u32) -> Result<Vec<f32>> {
    let step = session.begin_token(token)?;
    let step = step.layer(0)?;
    let step = step.layer(1)?;
    step.finish()
}

#[test]
fn emitted_native_layer_sequence_matches_regular_session_and_oracle() {
    let fixture = Fixture::tiny(2, 1, false);
    let model = fixture.model().unwrap();
    let native: NativeToken = native_two_layers;
    let mut session = model.session(4).unwrap();
    for (index, &token) in TOKENS.iter().enumerate() {
        let logits = native(&mut session, token).unwrap();
        let ordinary = model
            .session(4)
            .unwrap()
            .prefill(&TOKENS[..=index])
            .unwrap();
        assert_eq!(logits, ordinary);
        assert_eq!(session.position(), index + 1);
    }
}

#[test]
fn native_steps_reject_wrong_order_and_can_be_abandoned_without_commit() {
    let fixture = Fixture::tiny(2, 1, false);
    let model = fixture.model().unwrap();
    let mut session = model.session(2).unwrap();
    assert!(session.begin_token(u32::MAX).is_err());
    assert!(session
        .begin_token(TOKENS[0])
        .unwrap()
        .layer(usize::MAX)
        .is_err());
    assert!(session.begin_token(TOKENS[0]).unwrap().finish().is_err());
    assert!(session
        .begin_token(TOKENS[0])
        .unwrap()
        .layer(0)
        .unwrap()
        .layer(0)
        .is_err());
    assert!(session
        .begin_token(TOKENS[0])
        .unwrap()
        .layer(0)
        .unwrap()
        .finish()
        .is_err());
    drop(session.begin_token(TOKENS[0]).unwrap().layer(0).unwrap());
    assert_eq!(session.position(), 0);
    let logits = native_two_layers(&mut session, TOKENS[1]).unwrap();
    assert_eq!(logits, model.session(2).unwrap().decode(TOKENS[1]).unwrap());
    native_two_layers(&mut session, TOKENS[2]).unwrap();
    assert!(session.begin_token(TOKENS[0]).is_err());
    assert_eq!(session.position(), 2);
}

#[test]
fn native_kernel_failure_does_not_commit_an_incomplete_token() {
    let mut fixture = Fixture::tiny(2, 1, true);
    fixture.weight("blk.0.ffn_gate.weight").bytes = bytes(&[f32::MAX; 12]);
    let model = fixture.model().unwrap();
    let mut session = model.session(4).unwrap();
    native_two_layers(&mut session, 0).unwrap();
    assert!(native_two_layers(&mut session, TOKENS[0]).is_err());
    assert_eq!(session.position(), 1);
    let logits = native_two_layers(&mut session, 0).unwrap();
    assert_eq!(logits, model.session(4).unwrap().prefill(&[0, 0]).unwrap());
    assert_eq!(session.position(), 2);
}

#[test]
fn session_budget_rejects_large_allocations_before_reserving_kv() {
    let mut fixture = Fixture::tiny(2, 1, false);
    fixture.config.context_length = 100_000_000;
    let model = fixture.model().unwrap();
    assert!(model.session(100_000_000).is_err());
    assert!(model.session_with_memory_limit(2, 0).is_err());
    let plan = model.memory_plan(2).unwrap();
    let exact_limit = plan.kv_bytes + plan.scratch_bytes_upper_bound;
    assert!(model.session_with_memory_limit(2, exact_limit - 1).is_err());
    assert!(model.session_with_memory_limit(2, exact_limit).is_ok());
    let mut encoded = serde_json::to_value(&fixture.config).unwrap();
    encoded["rope_scaling"] = serde_json::json!("linear");
    assert!(serde_json::from_value::<Config>(encoded).is_err());
}

#[test]
fn untrusted_configuration_rejects_each_invalid_dimension_and_float_boundary() {
    let valid = Fixture::tiny(2, 1, false).config;
    for field in [
        "embedding_length",
        "block_count",
        "head_count",
        "head_count_kv",
        "feed_forward_length",
        "context_length",
        "vocab_size",
        "rope_dimension_count",
    ] {
        let mut serialized = serde_json::to_value(&valid).unwrap();
        serialized[field] = 0.into();
        let config: Config = serde_json::from_value(serialized).unwrap();
        assert!(config.validate().is_err(), "zero {field} was accepted");
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0] {
        let mut config = valid.clone();
        config.rms_norm_eps = value;
        assert!(config.validate().is_err(), "epsilon {value} was accepted");
        config = valid.clone();
        config.rope_freq_base = value;
        assert!(config.validate().is_err(), "RoPE base {value} was accepted");
    }
    for field in [
        "embedding_length",
        "block_count",
        "feed_forward_length",
        "context_length",
        "vocab_size",
    ] {
        let mut serialized = serde_json::to_value(&valid).unwrap();
        serialized[field] = serde_json::json!(usize::MAX - 1);
        let config: Config = serde_json::from_value(serialized).unwrap();
        assert!(
            config.validate().is_err(),
            "overflowing {field} was accepted"
        );
    }
    let mut config = valid.clone();
    config.embedding_length = 3;
    assert!(
        config.validate().is_err(),
        "fractional head width was accepted"
    );
    config.head_count = 1;
    config.rope_dimension_count = 3;
    assert!(
        config.validate().is_err(),
        "odd rotary head width was accepted"
    );
    config = valid;
    config.head_count = 0;
    assert_eq!(config.head_dimension(), 0);
}

#[test]
fn invalid_configurations_and_resources_are_rejected_before_execution() {
    for change in 0..7 {
        let mut fixture = Fixture::tiny(2, 1, false);
        match change {
            0 => fixture.config.head_count = 0,
            1 => fixture.config.head_count_kv = 3,
            2 => fixture.config.rope_dimension_count = 1,
            3 => fixture.config.rms_norm_eps = f32::NAN,
            4 => fixture.config.rope_freq_base = 0.0,
            5 => fixture.config.vocab_size -= 1,
            _ => fixture.config.context_length = usize::MAX,
        }
        assert!(fixture.model().is_err(), "config change {change}");
    }
    for change in 0..7 {
        let mut fixture = Fixture::tiny(2, 1, false);
        match change {
            0 => {
                fixture.weights.retain(|w| w.name != "output_norm.weight");
            }
            1 => fixture.weight("blk.0.attn_k.weight").dims.reverse(),
            2 => fixture.weight("token_embd.weight").bytes[0..4]
                .copy_from_slice(&f32::NAN.to_le_bytes()),
            3 => {
                fixture.weight("blk.0.ffn_gate.weight").bytes.pop();
            }
            4 => fixture.weight("blk.0.ffn_gate.weight").name = "blk.0.ffn_gate.bias".into(),
            5 => fixture.add("unknown.weight", &[1], vec![0.0]),
            _ => fixture.add("output_norm.weight", &[4], vec![1.0; 4]),
        }
        assert!(fixture.model().is_err(), "resource change {change}");
    }
}

#[test]
fn quantized_embedding_rows_remain_borrowed_and_use_signed_block_scales() {
    let mut fixture = Fixture::empty_projections(32, 4, 2, 32);
    let mut quantized = vec![0_u8; VOCAB * 18];
    for (token, scale) in [(259, 0x3800_u16), (260, 0xbc00_u16)] {
        let row = &mut quantized[token * 18..(token + 1) * 18];
        row[..2].copy_from_slice(&scale.to_le_bytes());
        for (index, byte) in row[2..].iter_mut().enumerate() {
            *byte = index as u8 | ((15 - index) as u8) << 4;
        }
    }
    let embedding = fixture.weight("token_embd.weight");
    embedding.dtype = GgufDataType::Q4_0;
    embedding.bytes = quantized;
    let pointer = embedding.bytes.as_ptr();
    let mut output = vec![0.0; 32 * VOCAB];
    output[259 * 32] = 1.0;
    output[260 * 32 + 16] = 1.0;
    output[261 * 32 + 31] = 1.0;
    fixture.weight("output.weight").bytes = bytes(&output);
    let model = fixture.model().unwrap();
    let stored = model
        .tensors()
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .unwrap();
    assert_eq!(stored.dtype, GgufDataType::Q4_0);
    assert_eq!(stored.data.as_ptr(), pointer);
    assert_eq!(stored.data.len(), 18 * VOCAB);
    let logits = model.session(2).unwrap().decode(TOKENS[0]).unwrap();
    let denominator = 5.5_f64.sqrt(); // squared norm 172 / 32 + epsilon 0.125
    assert_logits(
        &logits,
        [-4.0 / denominator, 3.5 / denominator, -4.0 / denominator],
    );
    let logits = model.session(2).unwrap().decode(TOKENS[1]).unwrap();
    let denominator = 21.625_f64.sqrt(); // squared norm 688 / 32 + epsilon
    assert_logits(
        &logits,
        [8.0 / denominator, -7.0 / denominator, 8.0 / denominator],
    );
}

#[test]
fn half_precision_weights_match_the_full_sequence_oracle() {
    let mut fixture = Fixture::tiny(2, 1, false);
    for weight in &mut fixture.weights {
        let mut half = Vec::new();
        for bytes in weight.bytes.as_chunks::<4>().0 {
            let value = f32::from_le_bytes(*bytes);
            let bits = value.to_bits();
            // Every fixture coefficient is exactly representable in binary16.
            let encoded = if value == 0.0 {
                0
            } else {
                let exponent = ((bits >> 23) & 0xff) - 112;
                assert!((1..31).contains(&exponent));
                assert_eq!(bits & 0x1fff, 0);
                (((bits >> 16) & 0x8000) | (exponent << 10) | ((bits & 0x7fffff) >> 13)) as u16
            };
            half.extend_from_slice(&encoded.to_le_bytes());
        }
        weight.bytes = half;
        weight.dtype = GgufDataType::F16;
    }
    let model = fixture.model().unwrap();
    assert_logits(
        &model.session(3).unwrap().prefill(&TOKENS).unwrap(),
        [0.7962413170399363, 1.0211478582351188, -4.582839687788002],
    );
}

#[test]
fn signed_q8_output_rows_match_an_independent_closed_form() {
    let mut fixture = Fixture::empty_projections(32, 4, 2, 32);
    let mut embeddings = vec![0.0; VOCAB * 32];
    for pair in embeddings[259 * 32..260 * 32].as_chunks_mut::<2>().0 {
        pair.copy_from_slice(&[1.0, -2.0]);
    }
    fixture.weight("token_embd.weight").bytes = bytes(&embeddings);
    let mut quantized = vec![0_u8; VOCAB * 34];
    let first = &mut quantized[259 * 34..260 * 34];
    first[..2].copy_from_slice(&0x2400_u16.to_le_bytes()); // +1/64
    for group in first[2..].as_chunks_mut::<4>().0 {
        group.copy_from_slice(&[128, 255, 0, 127]);
    }
    let second = &mut quantized[260 * 34..261 * 34];
    second[..2].copy_from_slice(&0xb400_u16.to_le_bytes()); // -0.25
    second[2..].fill(1);
    let third = &mut quantized[261 * 34..262 * 34];
    third[..2].copy_from_slice(&0x3c00_u16.to_le_bytes()); // +1
    third[2..].fill(255);
    let output = fixture.weight("output.weight");
    output.dtype = GgufDataType::Q8_0;
    output.bytes = quantized;
    let model = fixture.model().unwrap();
    let denominator = 2.625_f64.sqrt();
    assert_logits(
        &model.session(1).unwrap().decode(TOKENS[0]).unwrap(),
        [-47.5 / denominator, 4.0 / denominator, 16.0 / denominator],
    );
}

#[test]
fn nonfinite_quantization_scale_is_rejected_at_import() {
    let mut fixture = Fixture::empty_projections(32, 4, 2, 32);
    let embedding = fixture.weight("token_embd.weight");
    embedding.dtype = GgufDataType::Q8_0;
    embedding.bytes = vec![0; VOCAB * 34];
    embedding.bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(fixture.model().is_err());
}
