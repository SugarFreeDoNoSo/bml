use bml_compiler::tokenizer::Vocabulary;
use bml_inference::tokenizer::TokenizerSpec;
use bml_parser::GgufParser;

#[path = "../../inference/tests/support/gguf_fixture.rs"]
mod gguf_fixture;

fn vocabulary() -> Vocabulary {
    let mut tokens = vec!["<unk>".into(), "<s>".into(), "</s>".into(), "▁".into()];
    let mut types = vec![2, 3, 3, 1];
    for byte in 0..256 {
        tokens.push(format!("<0x{byte:02X}>"));
        types.push(6);
    }
    let mut scores = vec![-100.0; tokens.len()];
    for (piece, score) in [
        ("H", 0.0),
        ("e", 0.0),
        ("l", 0.0),
        ("o", 0.0),
        ("He", 1.0),
        ("ll", 2.0),
        ("llo", 3.0),
        ("Hello", 4.0),
        ("▁Hello", 5.0),
    ] {
        tokens.push(piece.into());
        types.push(1);
        scores.push(score);
    }
    Vocabulary::from_spec(TokenizerSpec {
        tokens,
        scores,
        token_types: types,
        bos_id: 1,
        eos_id: 2,
        unk_id: 0,
        add_bos: true,
        add_eos: false,
        add_space_prefix: true,
    })
    .unwrap()
}

#[test]
fn legacy_vocabulary_uses_scored_bpe_bos_and_lossless_byte_fallback() {
    let vocab = vocabulary();
    assert_eq!(vocab.encode("Hello").unwrap(), [1, 268]);
    assert_eq!(vocab.encode("").unwrap(), [1]);
    let ids = vocab.encode("Hello  🙂\n").unwrap();
    assert_eq!(ids, [1, 268, 3, 3, 244, 163, 157, 134, 14]);
    let bytes: Vec<u8> = ids
        .iter()
        .flat_map(|&id| vocab.token_piece_bytes(id).unwrap())
        .collect();
    assert_eq!(bytes, " Hello  🙂\n".as_bytes());
    assert_eq!(vocab.decode(&ids).unwrap(), " Hello  🙂\n");
    // Case-sensitive unknown text is encoded as bytes, not lowercased or BOS.
    assert_eq!(vocab.encode("XYZ").unwrap(), [1, 3, 92, 93, 94]);
    assert!(vocab.token_piece_bytes(999_999).is_err());
    assert!(vocab.decode(&[4 + 0xf0]).is_err()); // incomplete UTF-8 is not text
}

#[test]
fn legacy_gguf_adapter_preserves_the_stored_tokenizer_policy() {
    let path = std::env::temp_dir().join(format!("bml_legacy_bpe_{}.gguf", std::process::id()));
    std::fs::write(&path, gguf_fixture::fixture()).unwrap();
    let reader = GgufParser::open(&path).unwrap();
    let vocab = Vocabulary::from_gguf(&reader).unwrap();
    assert_eq!(vocab.encode("A\n").unwrap(), [1, 3, 69, 14]);
    assert_eq!(vocab.token_piece_bytes(3).unwrap(), b" ");
    assert_eq!(vocab.token_piece_bytes(14).unwrap(), b"\n");
    assert!(vocab
        .token_piece_bytes(vocab.eos_token_id)
        .unwrap()
        .is_empty());
    drop(reader);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn missing_bpe_metadata_is_rejected_instead_of_enabling_word_lookup() {
    let mut bytes = gguf_fixture::fixture();
    let key = b"tokenizer.ggml.scores";
    let start = bytes
        .windows(key.len())
        .position(|part| part == key)
        .unwrap();
    bytes[start..start + key.len()].copy_from_slice(b"tokenizer.ggml._cores");
    let path = std::env::temp_dir().join(format!(
        "bml_legacy_bpe_missing_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, bytes).unwrap();
    let reader = GgufParser::open(&path).unwrap();
    let error = Vocabulary::from_gguf(&reader).unwrap_err();
    assert!(error.contains("tokenizer.ggml.scores"), "{error}");
    drop(reader);
    std::fs::remove_file(path).unwrap();
}
