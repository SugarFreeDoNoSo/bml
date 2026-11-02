use bml_gguf::GgufReader;
use bml_inference::tokenizer::{Tokenizer, TokenizerSpec};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn vocabulary(extra: &[(&str, f32, i32)]) -> TokenizerSpec {
    let mut spec = TokenizerSpec {
        tokens: vec!["<unk>".into(), "<s>".into(), "</s>".into()],
        scores: vec![0.0; 3],
        token_types: vec![2, 3, 3],
        bos_id: 1,
        eos_id: 2,
        unk_id: 0,
        add_bos: true,
        add_eos: false,
        add_space_prefix: true,
    };
    for byte in 0..=255 {
        spec.tokens.push(format!("<0x{byte:02X}>"));
        spec.scores.push(0.0);
        spec.token_types.push(6);
    }
    for &(text, score, kind) in extra {
        spec.tokens.push(text.into());
        spec.scores.push(score);
        spec.token_types.push(kind);
    }
    spec
}

fn token(spec: &TokenizerSpec, text: &str) -> u32 {
    spec.tokens.iter().position(|piece| piece == text).unwrap() as u32
}

#[test]
fn score_order_and_leftmost_ties_determine_merges() {
    let mut spec = vocabulary(&[
        ("a", -20.0, 1),
        ("b", -20.0, 1),
        ("c", -20.0, 1),
        ("ab", -1.0, 1),
        ("bc", -1.0, 1),
    ]);
    spec.add_space_prefix = false;
    let ab = token(&spec, "ab");
    let bc = token(&spec, "bc");
    let a = token(&spec, "a");
    let c = token(&spec, "c");
    let tokenizer = Tokenizer::new(spec.clone()).unwrap();
    assert_eq!(tokenizer.encode("abc", false).unwrap(), [ab, c]);
    spec.scores[bc as usize] = 0.5;
    let tokenizer = Tokenizer::new(spec.clone()).unwrap();
    assert_eq!(tokenizer.encode("abc", false).unwrap(), [a, bc]);
    // IEEE signed zero is a score tie, not a preference for positive zero.
    spec.scores[ab as usize] = -0.0;
    spec.scores[bc as usize] = 0.0;
    let tokenizer = Tokenizer::new(spec).unwrap();
    assert_eq!(tokenizer.encode("abc", false).unwrap(), [ab, c]);
}

#[test]
fn merges_reconsider_neighbors_and_discard_stale_pairs() {
    let mut spec = vocabulary(&[
        ("a", 0.0, 1),
        ("b", 0.0, 1),
        ("c", 0.0, 1),
        ("d", 0.0, 1),
        ("ab", 5.0, 1),
        ("bc", 4.0, 1),
        ("cd", 3.0, 1),
        ("abc", 2.0, 1),
        ("abcd", 1.0, 1),
    ]);
    spec.add_space_prefix = false;
    let whole = token(&spec, "abcd");
    let tokenizer = Tokenizer::new(spec).unwrap();
    assert_eq!(tokenizer.encode("abcd", false).unwrap(), [whole]);
}

#[test]
fn whitespace_is_preserved_without_normalizing_unicode() {
    let spec = vocabulary(&[("▁", -10.0, 1), ("a", -10.0, 1), ("b", -10.0, 1)]);
    let space = token(&spec, "▁");
    let a = token(&spec, "a");
    let b = token(&spec, "b");
    let tokenizer = Tokenizer::new(spec).unwrap();
    let ids = tokenizer.encode(" a  b\t\r\n", false).unwrap();
    assert_eq!(
        ids,
        [space, space, a, space, space, b, 3 + 9, 3 + 13, 3 + 10]
    );
    assert_eq!(tokenizer.decode(&ids, false).unwrap(), " a  b\t\r\n");
    let source = "é🚀中\0\t\n\u{a0}e\u{301}";
    let ids = tokenizer.encode(source, false).unwrap();
    assert_eq!(ids[0], space);
    assert_eq!(
        &ids[1..],
        source
            .as_bytes()
            .iter()
            .map(|byte| *byte as u32 + 3)
            .collect::<Vec<_>>()
    );
    assert_eq!(tokenizer.decode(&ids, true).unwrap(), source);
}

#[test]
fn disabling_space_prefix_preserves_first_literal_space() {
    let mut spec = vocabulary(&[("▁", 0.0, 1)]);
    spec.add_space_prefix = false;
    let tokenizer = Tokenizer::new(spec).unwrap();
    for source in ["", " ", "  x", "x ", "\n"] {
        let ids = tokenizer.encode(source, false).unwrap();
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), source);
    }
}

#[test]
fn bos_eos_empty_input_and_detokenization_follow_sentencepiece_policy() {
    let mut spec = vocabulary(&[("▁", 0.0, 1), ("x", 0.0, 1)]);
    spec.add_eos = true;
    let space = token(&spec, "▁");
    let x = token(&spec, "x");
    let tokenizer = Tokenizer::new(spec).unwrap();
    assert!(tokenizer.encode("", false).unwrap().is_empty());
    assert_eq!(tokenizer.encode("", true).unwrap(), [1, 2]);
    assert_eq!(tokenizer.encode("x", true).unwrap(), [1, space, x, 2]);
    assert_eq!(tokenizer.encode("x", false).unwrap(), [space, x]);
    assert_eq!(tokenizer.decode(&[1, space, x, 2], true).unwrap(), " x");
    assert_eq!(
        tokenizer.decode(&[1, space, x, 2], false).unwrap(),
        "<s> x</s>"
    );
    assert_eq!(tokenizer.decode(&[space, x], true).unwrap(), "x");
    assert_eq!(tokenizer.decode(&[1, 2], true).unwrap(), "");
    assert_eq!(tokenizer.decode(&[0], true).unwrap(), "");
    assert_eq!(tokenizer.decode(&[0], false).unwrap(), "<unk>");
}

#[test]
fn byte_output_preserves_incomplete_utf8_and_checks_all_ids() {
    let tokenizer = Tokenizer::new(vocabulary(&[])).unwrap();
    assert_eq!(
        tokenizer.decode_bytes(&[3 + 0xF0, 3 + 0x9F], true).unwrap(),
        [0xF0, 0x9F]
    );
    assert!(tokenizer.decode(&[3 + 0xF0, 3 + 0x9F], true).is_err());
    assert_eq!(
        tokenizer.decode_bytes(&[3, 3 + 255], false).unwrap(),
        [0, 255]
    );
    assert!(tokenizer.decode_bytes(&[1, u32::MAX, 2], true).is_err());
}

#[test]
fn streaming_pieces_keep_word_spaces_and_partial_utf8() {
    let spec = vocabulary(&[("▁", 0.0, 1), ("▁hello", 0.0, 1), ("▁world", 0.0, 1)]);
    let hello = token(&spec, "▁hello");
    let world = token(&spec, "▁world");
    let tokenizer = Tokenizer::new(spec).unwrap();
    let mut bytes = tokenizer.token_piece_bytes(hello, true).unwrap();
    bytes.extend(tokenizer.token_piece_bytes(world, true).unwrap());
    for byte in "🚀".as_bytes() {
        bytes.extend(tokenizer.token_piece_bytes(*byte as u32 + 3, true).unwrap());
    }
    assert_eq!(String::from_utf8(bytes).unwrap(), " hello world🚀");
    assert!(tokenizer.token_piece_bytes(1, true).unwrap().is_empty());
    assert_eq!(tokenizer.token_piece_bytes(1, false).unwrap(), b"<s>");
    assert!(tokenizer.token_piece_bytes(u32::MAX, false).is_err());
}

#[test]
fn control_spellings_are_not_forced_and_user_defined_pieces_are_atomic() {
    let spec = vocabulary(&[
        ("▁", 0.0, 1),
        ("[tag]", 0.0, 4),
        ("a[tag]", 0.0, 4),
        ("X▁Y", 0.0, 4),
    ]);
    let space = token(&spec, "▁");
    let long = token(&spec, "a[tag]");
    let literal = token(&spec, "X▁Y");
    let tokenizer = Tokenizer::new(spec).unwrap();
    let ids = tokenizer.encode("<s>", false).unwrap();
    assert!(!ids.contains(&1));
    assert_eq!(tokenizer.decode(&ids, true).unwrap(), "<s>");
    assert_eq!(
        tokenizer.encode("a[tag]b", false).unwrap(),
        [long, space, 3 + b'b' as u32]
    );
    assert_eq!(tokenizer.encode("X▁Y", false).unwrap(), [literal]);
    assert_eq!(tokenizer.decode(&[literal], true).unwrap(), "X▁Y");
}

#[test]
fn malformed_vocabulary_is_rejected_before_encoding() {
    let valid = vocabulary(&[("x", 0.0, 1)]);
    let mut cases = Vec::new();
    let mut spec = valid.clone();
    spec.scores.pop();
    cases.push(spec);
    let mut spec = valid.clone();
    spec.token_types.pop();
    cases.push(spec);
    let mut spec = valid.clone();
    spec.scores[4] = f32::NAN;
    cases.push(spec);
    let mut spec = valid.clone();
    spec.scores[4] = f32::INFINITY;
    cases.push(spec);
    let mut spec = valid.clone();
    spec.tokens[4] = String::new();
    cases.push(spec);
    let mut spec = valid.clone();
    spec.tokens[4] = spec.tokens[3].clone();
    cases.push(spec);
    let mut spec = valid.clone();
    spec.tokens[4] = "<0x0a>".into();
    cases.push(spec);
    let mut spec = valid.clone();
    spec.token_types[4] = 1;
    cases.push(spec);
    let mut spec = valid.clone();
    spec.token_types[259] = 7;
    cases.push(spec);
    let mut spec = valid.clone();
    spec.bos_id = u32::MAX;
    cases.push(spec);
    let mut spec = valid.clone();
    spec.eos_id = 0;
    cases.push(spec);
    let mut spec = valid.clone();
    spec.unk_id = 1;
    cases.push(spec);
    for (index, spec) in cases.into_iter().enumerate() {
        assert!(Tokenizer::new(spec).is_err(), "malformed case {index}");
    }
    assert_eq!(Tokenizer::new(valid).unwrap().vocab_size(), 260);
}

#[test]
fn serialized_tokenizer_spec_roundtrips_without_changing_ids() {
    let spec = vocabulary(&[("▁", 0.0, 1), ("é", -1.25, 1)]);
    let serialized = serde_json::to_vec(&spec).unwrap();
    let restored = Tokenizer::new(serde_json::from_slice(&serialized).unwrap()).unwrap();
    let original = Tokenizer::new(spec).unwrap();
    assert_eq!(
        original.encode("é!", true).unwrap(),
        restored.encode("é!", true).unwrap()
    );
}

struct Fixture(PathBuf);

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn string(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn metadata(spec: &TokenizerSpec) -> Vec<(String, u32, Vec<u8>)> {
    let mut model = Vec::new();
    string(&mut model, "llama");
    let mut entries = vec![("tokenizer.ggml.model".into(), 8, model)];
    for (name, element_type) in [("tokens", 8u32), ("scores", 6), ("token_type", 5)] {
        let mut values = element_type.to_le_bytes().to_vec();
        values.extend_from_slice(&(spec.tokens.len() as u64).to_le_bytes());
        for index in 0..spec.tokens.len() {
            match element_type {
                8 => string(&mut values, &spec.tokens[index]),
                6 => values.extend_from_slice(&spec.scores[index].to_le_bytes()),
                5 => values.extend_from_slice(&spec.token_types[index].to_le_bytes()),
                _ => unreachable!(),
            }
        }
        entries.push((format!("tokenizer.ggml.{name}"), 9, values));
    }
    for (name, id) in [
        ("bos_token_id", spec.bos_id),
        ("eos_token_id", spec.eos_id),
        ("unknown_token_id", spec.unk_id),
    ] {
        entries.push((
            format!("tokenizer.ggml.{name}"),
            4,
            id.to_le_bytes().to_vec(),
        ));
    }
    entries
}

fn fixture(entries: Vec<(String, u32, Vec<u8>)>) -> Fixture {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (name, kind, value) in entries {
        string(&mut bytes, &name);
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&value);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "bml-tokenizer-{}-{}.gguf",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).unwrap();
    Fixture(path)
}

#[test]
fn gguf_import_requires_metadata_and_honors_explicit_flags() {
    let source = vocabulary(&[("▁", 0.0, 1)]);
    let file = fixture(metadata(&source));
    let reader = GgufReader::open(&file.0).unwrap();
    let spec = TokenizerSpec::from_gguf(&reader).unwrap();
    assert!(spec.add_bos);
    assert!(!spec.add_eos);
    assert!(spec.add_space_prefix);
    assert_eq!(spec.tokens, source.tokens);
    let mut entries = metadata(&source);
    entries.extend([
        ("tokenizer.ggml.add_bos_token".into(), 7, vec![0]),
        ("tokenizer.ggml.add_eos_token".into(), 7, vec![1]),
        ("tokenizer.ggml.add_space_prefix".into(), 7, vec![0]),
    ]);
    let file = fixture(entries);
    let reader = GgufReader::open(&file.0).unwrap();
    let tokenizer = Tokenizer::new(TokenizerSpec::from_gguf(&reader).unwrap()).unwrap();
    assert_eq!(tokenizer.encode("x", true).unwrap(), [3 + b'x' as u32, 2]);
    for key in [
        "tokens",
        "scores",
        "token_type",
        "bos_token_id",
        "eos_token_id",
        "unknown_token_id",
        "model",
    ] {
        let mut entries = metadata(&source);
        entries.retain(|(name, _, _)| name != &format!("tokenizer.ggml.{key}"));
        let file = fixture(entries);
        let reader = GgufReader::open(&file.0).unwrap();
        assert!(TokenizerSpec::from_gguf(&reader).is_err(), "missing {key}");
    }
}

#[test]
fn gguf_import_rejects_unsupported_normalization_and_wrong_types() {
    let spec = vocabulary(&[]);
    for key in ["remove_extra_whitespaces", "clean_spaces"] {
        let mut entries = metadata(&spec);
        entries.push((format!("tokenizer.ggml.{key}"), 7, vec![1]));
        let file = fixture(entries);
        let reader = GgufReader::open(&file.0).unwrap();
        assert!(TokenizerSpec::from_gguf(&reader).is_err());
    }
    for value in ["gpt2", "t5"] {
        let mut entries = metadata(&spec);
        let entry = entries
            .iter_mut()
            .find(|(key, _, _)| key == "tokenizer.ggml.model")
            .unwrap();
        entry.2.clear();
        string(&mut entry.2, value);
        let file = fixture(entries);
        let reader = GgufReader::open(&file.0).unwrap();
        assert!(TokenizerSpec::from_gguf(&reader).is_err());
    }
    let mut entries = metadata(&spec);
    entries.push((
        "tokenizer.ggml.add_bos_token".into(),
        4,
        1u32.to_le_bytes().to_vec(),
    ));
    let file = fixture(entries);
    let reader = GgufReader::open(&file.0).unwrap();
    assert!(TokenizerSpec::from_gguf(&reader).is_err());
}
