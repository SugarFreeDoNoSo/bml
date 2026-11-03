//! Tokenizer BPE — lee vocabulario del GGUF y tokeniza texto.
//!
//! Lee los metadatos `tokenizer.ggml.*` del GGUF:
//! - `tokenizer.ggml.tokens` → array de strings (vocabulario)
//! - `tokenizer.ggml.scores` → array de f32 (puntajes)
//! - `tokenizer.ggml.model` → tipo de modelo (ej. "llama")
//! - `tokenizer.ggml.bos_token_id` → begin-of-sequence ID
//! - `tokenizer.ggml.eos_token_id` → end-of-sequence ID

use bml_inference::tokenizer::{Tokenizer, TokenizerSpec};
use bml_parser::{GgufMetadataValue, GgufParser};
use std::collections::HashMap;
use std::sync::Arc;

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
    /// None only for projection-only, zero-layer numerical fixtures.
    pub(crate) tokenizer: Option<Arc<Tokenizer>>,
}

impl Vocabulary {
    /// Carga el vocabulario desde un parser GGUF.
    pub fn from_gguf(parser: &GgufParser) -> Result<Self, String> {
        Self::from_spec(TokenizerSpec::from_gguf(parser).map_err(|error| error.to_string())?)
    }

    /// Validate and own the complete SentencePiece BPE metadata.
    pub fn from_spec(spec: TokenizerSpec) -> Result<Self, String> {
        let tokenizer = Tokenizer::new(spec).map_err(|error| error.to_string())?;
        let spec = tokenizer.spec();
        Ok(Self {
            id_to_token: spec.tokens.clone(),
            token_to_id: spec
                .tokens
                .iter()
                .enumerate()
                .map(|(id, token)| (token.clone(), id as u32))
                .collect(),
            model_type: "llama".into(),
            bos_token_id: spec.bos_id,
            eos_token_id: spec.eos_id,
            tokenizer: Some(Arc::new(tokenizer)),
        })
    }

    // These historical zero-layer fixtures exercise matrix projection, not text
    // generation. They intentionally cannot encode/decode through the BPE API.
    pub(crate) fn from_projection_fixture(parser: &GgufParser) -> Result<Self, String> {
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
                GgufMetadataValue::String(s) => Ok(s.clone()),
                _ => Err("projection vocabulary contains a non-string token"),
            })
            .collect::<Result<_, _>>()?;

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
            tokenizer: None,
        })
    }

    /// Encode text with the validated GGUF policy, including configured BOS/EOS.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.validated()?
            .encode(text, true)
            .map_err(|error| error.to_string())
    }

    fn validated(&self) -> Result<&Tokenizer, String> {
        self.tokenizer
            .as_deref()
            .ok_or_else(|| "projection-only vocabulary has no validated tokenizer".into())
    }

    /// Decode a generated piece as bytes, preserving spaces and partial UTF-8.
    pub fn token_piece_bytes(&self, id: u32) -> Result<Vec<u8>, String> {
        self.validated()?
            .token_piece_bytes(id, true)
            .map_err(|error| error.to_string())
    }

    /// Raw vocabulary spelling for inspection; use `token_piece_bytes` for output.
    pub fn decode_single(&self, id: u32) -> &str {
        self.id_to_token
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }

    /// Decode complete text, skipping control tokens and validating UTF-8.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.validated()?
            .decode(ids, true)
            .map_err(|error| error.to_string())
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
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn vocab_from_tinyllama_gguf() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();
        let parser = GgufParser::open(path).expect("abrir gguf");
        let vocab = Vocabulary::from_gguf(&parser).expect("cargar vocab");
        assert!(vocab.len() > 0);
        assert!(!vocab.model_type.is_empty());
        println!(
            "Tokenizer: {} (vocab_size={})",
            vocab.model_type,
            vocab.len()
        );
        println!("BOS={}, EOS={}", vocab.bos_token_id, vocab.eos_token_id);
    }

    #[test]
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn encode_decode_roundtrip() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();
        let parser = GgufParser::open(path).expect("abrir gguf");
        let vocab = Vocabulary::from_gguf(&parser).expect("cargar vocab");

        let text = "Hello";
        let ids = vocab.encode(text).unwrap();
        assert!(!ids.is_empty());
        println!("'{}' → {:?}", text, ids);
        let decoded = vocab.decode(&ids).unwrap();
        println!("Decoded: '{}'", decoded);
    }

    #[test]
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn encode_empty_returns_bos() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();
        let parser = GgufParser::open(path).expect("abrir gguf");
        let vocab = Vocabulary::from_gguf(&parser).expect("cargar vocab");

        let ids = vocab.encode("").unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids[0], vocab.bos_token_id);
    }
}
