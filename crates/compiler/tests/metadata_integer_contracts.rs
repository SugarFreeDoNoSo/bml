use bml_compiler::gguf_compiler::InferenceCompiler;

#[path = "support/causal_fixture.rs"]
mod causal_fixture;

#[test]
fn signed_gguf_dimension_preserves_positive_value_and_rejects_negative() {
    let fixture = causal_fixture::fixture(false);
    let original = std::fs::read(&fixture.0).unwrap();
    let expected = InferenceCompiler::open(&fixture.0)
        .unwrap()
        .config()
        .clone();
    assert_eq!(expected.n_embd, 4);
    let key = b"llama.embedding_length";
    let positions: Vec<_> = original
        .windows(key.len())
        .enumerate()
        .filter_map(|(offset, bytes)| (bytes == key).then_some(offset))
        .collect();
    assert_eq!(positions.len(), 1);
    let tag = positions[0] + key.len();
    // Independent GGUF wire tags: U32=4 and I32=5, both four-byte payloads.
    assert_eq!(&original[tag..tag + 4], &4u32.to_le_bytes());
    assert_eq!(&original[tag + 4..tag + 8], &4u32.to_le_bytes());
    for value in [4i32, -1] {
        let mut bytes = original.clone();
        bytes[tag..tag + 4].copy_from_slice(&5u32.to_le_bytes());
        bytes[tag + 4..tag + 8].copy_from_slice(&value.to_le_bytes());
        std::fs::write(&fixture.0, bytes).unwrap();
        for mapped in [false, true] {
            let result = if mapped {
                InferenceCompiler::open_mapped(&fixture.0)
            } else {
                InferenceCompiler::open(&fixture.0)
            };
            if value > 0 {
                let compiler = result.expect("positive I32 dimension must be accepted");
                assert_eq!(compiler.config(), &expected);
            } else {
                assert!(result.is_err(), "negative I32 dimension must be rejected");
            }
        }
    }
}
