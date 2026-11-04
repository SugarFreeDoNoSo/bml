use bml_compiler::distributed::{
    deserialize_vector_fragment, serialize_vector_fragment, VectorFragment,
};

#[test]
fn vector_wire_rejects_every_truncation_and_untrusted_lengths() {
    let fragment = VectorFragment::new(7, &[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], 0, 2, 2, 0, 2, 0);
    let bytes = serialize_vector_fragment(&fragment);
    let decoded = deserialize_vector_fragment(&bytes).unwrap();
    assert_eq!(decoded.execute(), vec![7.0, 10.0]);
    for end in 0..bytes.len() {
        assert!(
            deserialize_vector_fragment(&bytes[..end]).is_err(),
            "truncation at {end}"
        );
    }
    let mut malformed = bytes.clone();
    malformed[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(deserialize_vector_fragment(&malformed).is_err());
    let mut malformed = bytes.clone();
    malformed[48..56].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(deserialize_vector_fragment(&malformed).is_err());
    let mut malformed = bytes.clone();
    malformed[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(deserialize_vector_fragment(&malformed).is_err());
    let mut malformed = bytes;
    malformed.push(0);
    assert!(deserialize_vector_fragment(&malformed).is_err());
}
