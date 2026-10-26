//! Historical parser paths remain callable, including associated function syntax.
use bml_parser::{create_gguf_with_metadata, GgufMetadataValue, GgufParser};

#[test]
fn legacy_parser_methods_and_io_errors_are_preserved() {
    let path = create_gguf_with_metadata();
    let reader: std::io::Result<GgufParser> = GgufParser::open(&path);
    let reader = reader.unwrap();
    assert_eq!(GgufParser::header(&reader).version, 3);
    assert_eq!(GgufParser::metadata(&reader).len(), 2);
    assert_eq!(GgufParser::tensor_infos(&reader).len(), 1);
    assert_eq!(GgufParser::bytes(&reader).len(), GgufParser::len(&reader));
    assert!(!GgufParser::is_empty(&reader));
    assert_eq!(GgufParser::architecture(&reader), Some("llama"));
    assert!(matches!(
        GgufParser::get_metadata(&reader, "llama.context_length"),
        Some(GgufMetadataValue::U32(2048))
    ));
    let info = GgufParser::find_tensor(&reader, "token_embd.weight").unwrap();
    assert_eq!(GgufParser::tensor_data(&reader, info).unwrap().len(), 32);
    assert_eq!(
        reader.read_tensor_f32("token_embd.weight").unwrap(),
        [0., 1., 2., 3., 4., 5., 6., 7.]
    );
    drop(reader);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        GgufParser::open(&path).unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
}
