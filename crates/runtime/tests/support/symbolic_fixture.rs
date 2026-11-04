use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct Fixture(pub PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn string(value: &str) -> Vec<u8> {
    let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
    bytes.extend(value.as_bytes());
    bytes
}
fn array(kind: u32, values: Vec<Vec<u8>>) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend((values.len() as u64).to_le_bytes());
    bytes.extend(values.into_iter().flatten());
    bytes
}
pub fn fixture(embedding: f32) -> Fixture {
    fixture_layers(embedding, 1)
}

#[allow(dead_code)]
pub fn fixture_layers(embedding: f32, layers: u32) -> Fixture {
    fixture_layers_with_output(embedding, layers, 69)
}

#[allow(dead_code)]
pub fn fixture_layers_with_output(embedding: f32, layers: u32, output_token: usize) -> Fixture {
    assert!(layers > 0);
    let mut tokens = vec!["<unk>".into(), "<s>".into(), "</s>".into(), "▁".into()];
    tokens.extend((0..256).map(|b| format!("<0x{b:02X}>")));
    let vocab = tokens.len();
    assert!(output_token < vocab);
    let mut meta = vec![
        ("general.architecture", 8u32, string("llama")),
        ("tokenizer.ggml.model", 8, string("llama")),
        (
            "tokenizer.ggml.tokens",
            9,
            array(8, tokens.iter().map(|t| string(t)).collect()),
        ),
        (
            "tokenizer.ggml.scores",
            9,
            array(
                6,
                (0..vocab).map(|_| (-1f32).to_le_bytes().to_vec()).collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            9,
            array(
                5,
                (0..vocab)
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
            0.5f32.to_le_bytes().to_vec(),
        ),
        ("llama.rope.freq_base", 6, 16f32.to_le_bytes().to_vec()),
    ];
    for (name, value) in [
        ("llama.block_count", layers),
        ("llama.attention.head_count", 2),
        ("llama.attention.head_count_kv", 1),
        ("llama.embedding_length", 4),
        ("llama.feed_forward_length", 4),
        ("llama.context_length", 8),
        ("llama.rope.dimension_count", 2),
        ("llama.vocab_size", vocab as u32),
        ("tokenizer.ggml.bos_token_id", 1),
        ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.unknown_token_id", 0),
    ] {
        meta.push((name, 4, value.to_le_bytes().to_vec()));
    }
    let magnitude = embedding;
    let mut embedding = vec![0f32; 4 * vocab];
    embedding[69 * 4] = magnitude;
    embedding[70 * 4 + 1] = magnitude;
    let mut head = vec![0f32; 4 * vocab];
    for i in 0..4 {
        head[i * 4 + i] = 0.5;
    }
    if layers > 1 {
        // The multi-layer integration fixture generates a real byte token.
        head[output_token * 4] = 2.0;
    }
    let mut q = vec![0f32; 16];
    let mut k = vec![0f32; 8];
    for i in 0..4 {
        q[i * 4 + i] = 0.25;
    }
    k[0] = 0.25;
    k[5] = 0.25;
    let mut v = vec![0f32; 8];
    v[0] = 0.25;
    v[5] = 0.25;
    let mut identity = vec![0f32; 16];
    for i in 0..4 {
        identity[i * 4 + i] = 0.25;
    }
    let weights = vec![
        ("token_embd.weight", vec![4, vocab as u64], embedding),
        ("output.weight", vec![4, vocab as u64], head),
        ("output_norm.weight", vec![4], vec![0.5; 4]),
        ("blk.0.attn_norm.weight", vec![4], vec![0.5; 4]),
        ("blk.0.ffn_norm.weight", vec![4], vec![0.5; 4]),
        ("blk.0.attn_q.weight", vec![4, 4], q),
        ("blk.0.attn_k.weight", vec![4, 2], k),
        ("blk.0.attn_v.weight", vec![4, 2], v),
        ("blk.0.attn_output.weight", vec![4, 4], identity.clone()),
        ("blk.0.ffn_gate.weight", vec![4, 4], identity.clone()),
        (
            "blk.0.ffn_up.weight",
            vec![4, 4],
            identity.iter().map(|&v| -v).collect(),
        ),
        ("blk.0.ffn_down.weight", vec![4, 4], identity),
    ];
    let mut weights: Vec<(String, Vec<u64>, Vec<f32>)> = weights
        .into_iter()
        .map(|(name, dims, values)| (name.to_owned(), dims, values))
        .collect();
    let original = weights.clone();
    for layer in 1..layers {
        for (name, dims, values) in &original {
            if let Some(suffix) = name.strip_prefix("blk.0.") {
                weights.push((
                    format!("blk.{layer}.{suffix}"),
                    dims.clone(),
                    values
                        .iter()
                        .map(|v| v * (1.0 + 0.1 * layer as f32))
                        .collect(),
                ));
            }
        }
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((weights.len() as u64).to_le_bytes());
    bytes.extend((meta.len() as u64).to_le_bytes());
    for (name, kind, value) in meta {
        bytes.extend(string(name));
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    let mut data = Vec::new();
    for (name, dims, values) in weights {
        data.resize(data.len().div_ceil(32) * 32, 0);
        bytes.extend(string(&name));
        bytes.extend((dims.len() as u32).to_le_bytes());
        for dim in dims {
            bytes.extend(dim.to_le_bytes());
        }
        bytes.extend(0u32.to_le_bytes());
        bytes.extend((data.len() as u64).to_le_bytes());
        data.extend(values.into_iter().flat_map(f32::to_le_bytes));
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(data);
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "bml_symbolic_causal_{}_{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).unwrap();
    Fixture(path)
}
