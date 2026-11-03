//! Tiny deterministic quantized GGUF shared by CLI and server contracts.
pub(crate) fn string(value: &str) -> Vec<u8> {
    let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
    bytes.extend(value.as_bytes());
    bytes
}

pub(crate) fn array(kind: u32, values: Vec<Vec<u8>>) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend((values.len() as u64).to_le_bytes());
    bytes.extend(values.into_iter().flatten());
    bytes
}

pub(crate) fn fixture() -> Vec<u8> {
    let mut tokens = vec!["<unk>".to_owned(), "<s>".into(), "</s>".into(), "▁".into()];
    tokens.extend((0..256).map(|byte| format!("<0x{byte:02X}>")));
    let vocab = tokens.len() as u64;
    let mut metadata = vec![
        ("general.architecture", 8u32, string("llama")),
        ("tokenizer.ggml.model", 8, string("llama")),
        (
            "tokenizer.ggml.tokens",
            9,
            array(8, tokens.iter().map(|token| string(token)).collect()),
        ),
        (
            "tokenizer.ggml.scores",
            9,
            array(
                6,
                tokens
                    .iter()
                    .map(|_| (-1.0f32).to_le_bytes().to_vec())
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            9,
            array(
                5,
                (0..tokens.len())
                    .map(|i| {
                        let kind: i32 = match i {
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
            1e-5f32.to_le_bytes().to_vec(),
        ),
        ("llama.rope.freq_base", 6, 10000.0f32.to_le_bytes().to_vec()),
    ];
    for (name, value) in [
        ("llama.embedding_length", 32u32),
        ("llama.block_count", 1),
        ("llama.attention.head_count", 4),
        ("llama.attention.head_count_kv", 2),
        ("llama.feed_forward_length", 32),
        ("llama.context_length", 8),
        ("llama.rope.dimension_count", 8),
        ("tokenizer.ggml.bos_token_id", 1),
        ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.unknown_token_id", 0),
    ] {
        metadata.push((name, 4, value.to_le_bytes().to_vec()));
    }
    for (name, value) in [
        ("tokenizer.ggml.add_bos_token", true),
        ("tokenizer.ggml.add_eos_token", false),
        ("tokenizer.ggml.add_space_prefix", true),
        ("tokenizer.ggml.remove_extra_whitespaces", false),
        ("tokenizer.ggml.byte_fallback", true),
    ] {
        metadata.push((name, 7, vec![u8::from(value)]));
    }
    let quantized = |rows: u64, value: u8| {
        let mut bytes = Vec::new();
        for _ in 0..rows {
            bytes.extend(0x3800u16.to_le_bytes()); // Independent half encoding: 0.5.
            bytes.extend([value; 16]);
        }
        bytes
    };
    let norm = || {
        (0..32)
            .flat_map(|_| 1.0f32.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut tensors = vec![
        (
            "token_embd.weight",
            vec![32, vocab],
            2u32,
            quantized(vocab, 0xc2),
        ),
        ("output_norm.weight", vec![32], 0, norm()),
        ("blk.0.attn_norm.weight", vec![32], 0, norm()),
        ("blk.0.ffn_norm.weight", vec![32], 0, norm()),
    ];
    for (name, rows) in [
        ("blk.0.attn_q.weight", 32u64),
        ("blk.0.attn_k.weight", 16),
        ("blk.0.attn_v.weight", 16),
        ("blk.0.attn_output.weight", 32),
        ("blk.0.ffn_gate.weight", 32),
        ("blk.0.ffn_up.weight", 32),
        ("blk.0.ffn_down.weight", 32),
    ] {
        tensors.push((name, vec![32, rows], 2, quantized(rows, 0x88)));
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (name, kind, value) in metadata {
        bytes.extend(string(name));
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    let mut data = Vec::new();
    for (name, dims, kind, payload) in tensors {
        data.resize(data.len().div_ceil(32) * 32, 0);
        bytes.extend(string(name));
        bytes.extend((dims.len() as u32).to_le_bytes());
        for dimension in dims {
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(kind.to_le_bytes());
        bytes.extend((data.len() as u64).to_le_bytes());
        data.extend(payload);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(data);
    bytes
}
