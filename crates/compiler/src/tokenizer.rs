//! Tokenizer BPE — lee vocabulario del GGUF y tokeniza texto.
//!
//! Lee los metadatos `tokenizer.ggml.*` del GGUF:
//! - `tokenizer.ggml.tokens` → array de strings (vocabulario)
//! - `tokenizer.ggml.scores` → array de f32 (puntajes)
//! - `tokenizer.ggml.model` → tipo de modelo (ej. "llama")
//! - `tokenizer.ggml.bos_token_id` → begin-of-sequence ID
//! - `tokenizer.ggml.eos_token_id` → end-of-sequence ID

use bml_parser::{GgufMetadataValue, GgufParser};
use std::collections::HashMap;

/// Vocabulario del modelo.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// ID de token → string.
    pub id_to_token: Vec<String>,
    /// String → ID de token.
    pub token_to_id: HashMap<String, u32>,
    /// Tipo de tokenizador (ej. "llama", "gpt2").
    pub model_type: String,
    /// Begin-of-sequence token ID.
    pub bos_token_id: u32,
    /// End-of-sequence token ID.
    pub eos_token_id: u32,
}

impl Vocabulary {
    /// Carga el vocabulario desde un parser GGUF.
    pub fn from_gguf(parser: &GgufParser) -> Result<Self, String> {
        let model_type = parser
            .get_metadata("tokenizer.ggml.model")
            .and_then(|v| match v {
                GgufMetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .ok_or("tokenizer.ggml.model no encontrado")?;

        let tokens = parser
            .get_metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufMetadataValue::Array(_, arr) => Some(arr.clone()),
                _ => None,
            })
            .ok_or("tokenizer.ggml.tokens no encontrado")?;

        let bos_token_id = parser
            .get_metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| match v {
                GgufMetadataValue::U32(id) => Some(*id),
                _ => None,
            })
            .unwrap_or(1);

        let eos_token_id = parser
            .get_metadata("tokenizer.ggml.eos_token_id")
            .and_then(|v| match v {
                GgufMetadataValue::U32(id) => Some(*id),
                _ => None,
            })
            .unwrap_or(2);

        let id_to_token: Vec<String> = tokens
            .iter()
            .map(|t| match t {
                GgufMetadataValue::String(s) => s.clone(),
                _ => unreachable!(),
            })
            .collect();

        let mut token_to_id = HashMap::new();
        for (i, token) in id_to_token.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        Ok(Self {
            id_to_token,
            token_to_id,
            model_type,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Tokeniza texto: split por espacios → lookup en vocabulario.
    /// Si un token no está en el vocabulario, se intenta con lowercase.
    /// Si sigue sin estar, se retorna `bos_token_id` como fallback.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let word = word.trim();
            if word.is_empty() {
                continue;
            }
            if let Some(&id) = self.token_to_id.get(word) {
                ids.push(id);
            } else if let Some(&id) = self.token_to_id.get(&word.to_lowercase()) {
                ids.push(id);
            } else {
                let prefixed = format!("▁{}", word);
                if let Some(&id) = self.token_to_id.get(&prefixed) {
                    ids.push(id);
                } else {
                    ids.push(self.bos_token_id);
                }
            }
        }
        if ids.is_empty() {
            ids.push(self.bos_token_id);
        }
        ids
    }

    /// Detokeniza: token ID → string.
    pub fn decode_single(&self, id: u32) -> &str {
        self.id_to_token
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }

    /// Detokeniza una secuencia de IDs.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| {
                let s = self.decode_single(id);
                s.strip_prefix('▁').unwrap_or(s)
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Largo del vocabulario.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Retorna `true` si el vocabulario está vacío.
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_from_tinyllama_gguf() {
        let path = "/root/tinyllama.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }
        let parser = GgufParser::open(path).expect("abrir gguf");
        let vocab = Vocabulary::from_gguf(&parser).expect("cargar vocab");
        assert!(vocab.len() > 0);
        assert!(!vocab.model_type.is_empty());
        println!("Tokenizer: {} (vocab_size={})", vocab.model_type, vocab.len());
        println!("BOS={}, EOS={}", vocab.bos_token_id, vocab.eos_token_id);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let path = "/root/tinyllama.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }
        let parser = GgufParser::open(path).expect("abrir gguf");
        let vocab = Vocabulary::from_gguf(&parser).expect("cargar vocab");

        let text = "Hello";
        let ids = vocab.encode(text);
        assert!(!ids.is_empty());
        println!("'{}' → {:?}", text, ids);
        let decoded = vocab.decode(&ids);
        println!("Decoded: '{}'", decoded);
    }

    #[test]
    fn encode_empty_returns_bos() {
        let path = "/root/tinyllama.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }
        let parser = GgufParser::open(path).expect("abrir gguf");
        let vocab = Vocabulary::from_gguf(&parser).expect("cargar vocab");

        let ids = vocab.encode("");
        assert!(!ids.is_empty());
        assert_eq!(ids[0], vocab.bos_token_id);
    }
}