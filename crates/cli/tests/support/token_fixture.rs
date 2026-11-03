//! Zero-layer, independently encoded byte-output model for IO contracts.
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

pub(crate) struct Fixture(pub(crate) PathBuf);
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

pub(crate) fn fixture() -> Fixture {
    fixture_with_unknown_piece("<unk>")
}

pub(crate) fn fixture_with_unknown_piece(unknown: &str) -> Fixture {
    build(unknown, 2, false)
}

#[allow(dead_code)]
pub(crate) fn fixture_with_output_norm() -> Fixture {
    build("<unk>", 2, true)
}

#[allow(dead_code)]
pub(crate) fn fixture_with_width(width: usize) -> Fixture {
    assert!(width >= 2);
    build("<unk>", width, false)
}

fn build(unknown: &str, width: usize, output_norm: bool) -> Fixture {
    let mut tokens = vec![unknown.into(), "<s>".into(), "</s>".into(), "▁".into()];
    tokens.extend((0..256).map(|byte| format!("<0x{byte:02X}>")));
    let vocab = tokens.len();
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
                (0..vocab).map(|_| (-1f32).to_le_bytes().to_vec()).collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            9,
            array(
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
    ];
    for (name, value) in [
        ("llama.block_count", 0u32),
        ("llama.attention.head_count", 1),
        ("llama.embedding_length", width as u32),
        ("llama.context_length", 8),
        ("llama.vocab_size", vocab as u32),
        ("tokenizer.ggml.bos_token_id", 1),
        ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.unknown_token_id", 0),
    ] {
        metadata.push((name, 4, value.to_le_bytes().to_vec()));
    }
    let mut embeddings = vec![0f32; width * vocab];
    for row in 0..vocab {
        embeddings[row * width] = 1.0;
    }
    embeddings[199 * width..199 * width + 2].copy_from_slice(&[0.0, 1.0]);
    embeddings[173 * width..173 * width + 2].copy_from_slice(&[-1.0, -1.0]);
    let mut head = vec![0f32; vocab * width];
    head[199 * width] = 1.0; // byte C3 after the prompt
    head[173 * width + 1] = 1.0; // byte A9 after C3
    head[2 * width..2 * width + 2].copy_from_slice(&[-1.0, -1.0]); // then EOS
    let mut weights = vec![
        (
            "token_embd.weight",
            vec![width as u64, vocab as u64],
            embeddings,
        ),
        ("output.weight", vec![width as u64, vocab as u64], head),
    ];
    if output_norm {
        weights.push(("output_norm.weight", vec![width as u64], vec![1.0; width]));
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((weights.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (name, kind, value) in metadata {
        bytes.extend(string(name));
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    let mut payload = Vec::new();
    for (name, dims, values) in weights {
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        bytes.extend(string(name));
        bytes.extend((dims.len() as u32).to_le_bytes());
        for dim in dims {
            bytes.extend(dim.to_le_bytes());
        }
        bytes.extend(0u32.to_le_bytes());
        bytes.extend((payload.len() as u64).to_le_bytes());
        payload.extend(values.into_iter().flat_map(f32::to_le_bytes));
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(payload);
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "bml_cli_byte_output_{}_{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).unwrap();
    Fixture(path)
}
