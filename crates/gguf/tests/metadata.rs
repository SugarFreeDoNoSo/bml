//! Reconstructed metadata boundary tests. Original local-only tests were lost;
//! these independent byte fixtures are newly validated, not claimed identical.
use bml_gguf::{GgufDataType as D, GgufError, GgufMetadataValue as V, GgufReader};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_metadata_reconstructed_{}_{}.gguf",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
    fn open(&self) -> Result<GgufReader, GgufError> {
        GgufReader::open(&self.0)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn string(s: &str) -> Vec<u8> {
    let mut out = (s.len() as u64).to_le_bytes().to_vec();
    out.extend(s.as_bytes());
    out
}
fn header(count: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(count.to_le_bytes());
    b
}
fn entry(b: &mut Vec<u8>, name: &str, id: u32, payload: &[u8]) {
    b.extend(string(name));
    b.extend(id.to_le_bytes());
    b.extend(payload);
}
fn single(id: u32, payload: &[u8]) -> Vec<u8> {
    let mut b = header(1);
    entry(&mut b, "κ", id, payload);
    b
}
struct Case {
    id: u32,
    kind: D,
    values: Vec<(Vec<u8>, V)>,
}
fn cases() -> Vec<Case> {
    macro_rules! numeric {
        ($id:expr, $kind:ident, $ty:ty, [$($value:expr),+]) => {
            Case { id: $id, kind: D::$kind, values: vec![$({
                let v: $ty = $value; (v.to_le_bytes().to_vec(), V::$kind(v))
            }),+] }
        }
    }
    vec![
        numeric!(0, U8, u8, [0, 255]),
        numeric!(1, I8, i8, [i8::MIN, 73]),
        numeric!(2, U16, u16, [u16::MAX, 0x1234]),
        numeric!(3, I16, i16, [i16::MIN, 30001]),
        numeric!(4, U32, u32, [u32::MAX, 0x12345678]),
        numeric!(5, I32, i32, [i32::MIN, 2000000001]),
        numeric!(6, F32, f32, [-0.0, f32::from_bits(0x7fc01234)]),
        Case {
            id: 7,
            kind: D::Bool,
            values: vec![(vec![0], V::Bool(false)), (vec![1], V::Bool(true))],
        },
        Case {
            id: 8,
            kind: D::String,
            values: vec![
                (string(""), V::String(String::new())),
                (string("λño🦀"), V::String("λño🦀".into())),
            ],
        },
        numeric!(10, U64, u64, [u64::MAX, 0x0123456789abcdef]),
        numeric!(11, I64, i64, [i64::MIN, 0x0123456789abcdef]),
        numeric!(12, F64, f64, [-0.0, f64::from_bits(0xfff8123456789abc)]),
    ]
}
fn same(actual: &V, expected: &V) {
    match (actual, expected) {
        (V::U8(a), V::U8(b)) => assert_eq!(a, b),
        (V::I8(a), V::I8(b)) => assert_eq!(a, b),
        (V::U16(a), V::U16(b)) => assert_eq!(a, b),
        (V::I16(a), V::I16(b)) => assert_eq!(a, b),
        (V::U32(a), V::U32(b)) => assert_eq!(a, b),
        (V::I32(a), V::I32(b)) => assert_eq!(a, b),
        (V::F32(a), V::F32(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        (V::Bool(a), V::Bool(b)) => assert_eq!(a, b),
        (V::String(a), V::String(b)) => assert_eq!(a, b),
        (V::U64(a), V::U64(b)) => assert_eq!(a, b),
        (V::I64(a), V::I64(b)) => assert_eq!(a, b),
        (V::F64(a), V::F64(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        _ => panic!("different metadata variants: {actual:?} vs {expected:?}"),
    }
}
fn array_payload(case: &Case) -> Vec<u8> {
    let mut b = case.id.to_le_bytes().to_vec();
    b.extend((case.values.len() as u64).to_le_bytes());
    for (bytes, _) in &case.values {
        b.extend(bytes);
    }
    b
}
fn invalid(bytes: &[u8]) {
    assert!(matches!(
        Fixture::new(bytes).open(),
        Err(GgufError::InvalidMetadata { .. })
    ));
}

#[test]
fn every_scalar_preserves_exact_type_bits_unicode_and_following_field_offsets() {
    let cases = cases();
    let count: usize = cases.iter().map(|c| c.values.len()).sum();
    let mut b = header(count as u64);
    for c in &cases {
        for (j, (bytes, _)) in c.values.iter().enumerate() {
            entry(&mut b, &format!("κ{}_{}", c.id, j), c.id, bytes);
        }
    }
    let f = Fixture::new(&b);
    let reader = f.open().unwrap();
    assert_eq!(reader.metadata().len(), count);
    assert_eq!(reader.header().metadata_kv_count, count as u64);
    assert_eq!(reader.header().tensor_count, 0);
    assert!(reader.tensor_infos().is_empty());
    assert_eq!(reader.bytes(), b);
    assert_eq!(reader.len(), b.len());
    assert!(!reader.is_empty());
    for c in &cases {
        for (j, (_, expected)) in c.values.iter().enumerate() {
            same(
                reader.get_metadata(&format!("κ{}_{}", c.id, j)).unwrap(),
                expected,
            );
        }
    }
    let debug = format!("{reader:?}");
    assert!(debug.starts_with("GgufReader {"));
    assert!(debug.contains("metadata_count: 24"));
    assert!(debug.contains("tensor_count: 0"));
    assert!(debug.contains(&format!("len: {}", b.len())));
}

#[test]
fn arrays_of_every_scalar_preserve_order_type_and_following_metadata() {
    for c in cases() {
        let mut b = header(2);
        entry(&mut b, "array", 9, &array_payload(&c));
        entry(&mut b, "after", 0, &[255]);
        let f = Fixture::new(&b);
        let reader = f.open().unwrap();
        match reader.get_metadata("array").unwrap() {
            V::Array(kind, values) => {
                assert_eq!(*kind, c.kind);
                assert_eq!(values.len(), c.values.len());
                for (a, (_, e)) in values.iter().zip(&c.values) {
                    same(a, e);
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
        same(reader.get_metadata("after").unwrap(), &V::U8(255));
    }
}

#[test]
fn empty_arrays_accept_every_supported_scalar_without_payload_or_padding() {
    for c in cases() {
        let mut p = c.id.to_le_bytes().to_vec();
        p.extend(0u64.to_le_bytes());
        let f = Fixture::new(&single(9, &p));
        let reader = f.open().unwrap();
        assert!(
            matches!(reader.get_metadata("κ"), Some(V::Array(kind, values)) if *kind == c.kind && values.is_empty())
        );
    }
}

#[test]
fn every_truncated_scalar_key_and_array_prefix_is_rejected_without_panicking() {
    for c in cases() {
        for b in c
            .values
            .iter()
            .map(|(p, _)| single(c.id, p))
            .chain(std::iter::once(single(9, &array_payload(&c))))
        {
            for end in 24..b.len() {
                invalid(&b[..end]);
            }
            Fixture::new(&b).open().unwrap();
        }
    }
}

#[test]
fn unknown_types_bad_utf8_invalid_booleans_and_nested_arrays_are_typed_errors() {
    for id in [13, 24, 255, u32::MAX] {
        invalid(&single(id, &[]));
        let mut p = id.to_le_bytes().to_vec();
        p.extend(0u64.to_le_bytes());
        invalid(&single(9, &p));
    }
    for value in [2, 127, 255] {
        invalid(&single(7, &[value]));
        let mut p = 7u32.to_le_bytes().to_vec();
        p.extend(1u64.to_le_bytes());
        p.push(value);
        invalid(&single(9, &p));
    }
    let mut bad_key = header(1);
    bad_key.extend(1u64.to_le_bytes());
    bad_key.push(255);
    bad_key.extend(0u32.to_le_bytes());
    bad_key.push(1);
    invalid(&bad_key);
    let mut bad_string = 1u64.to_le_bytes().to_vec();
    bad_string.push(255);
    invalid(&single(8, &bad_string));
    for count in [0u64, 1] {
        let mut p = 9u32.to_le_bytes().to_vec();
        p.extend(count.to_le_bytes());
        invalid(&single(9, &p));
    }
}

#[test]
fn huge_untrusted_lengths_and_counts_return_errors_without_upfront_allocation() {
    invalid(&header(u64::MAX));
    let mut key = header(1);
    key.extend(u64::MAX.to_le_bytes());
    invalid(&key);
    invalid(&single(8, &u64::MAX.to_le_bytes()));
    let mut p = 0u32.to_le_bytes().to_vec();
    p.extend(u64::MAX.to_le_bytes());
    invalid(&single(9, &p));
    p.push(1);
    invalid(&single(9, &p));
    let mut tensors = header(0);
    tensors[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(
        Fixture::new(&tensors).open(),
        Err(GgufError::InvalidTensorInfo { .. })
    ));
}

#[test]
fn tensor_type_ids_sizes_and_legacy_quantized_flags_are_complete() {
    let rows = [
        (0, D::F32, 4, false),
        (1, D::F16, 2, true),
        (2, D::Q4_0, 0, true),
        (3, D::Q4_1, 0, true),
        (6, D::Q5_0, 0, true),
        (7, D::Q5_1, 0, true),
        (8, D::Q8_0, 0, true),
        (9, D::Q8_1, 0, true),
        (10, D::Q2_K, 0, true),
        (11, D::Q3_K, 0, true),
        (12, D::Q4_K, 0, true),
        (13, D::Q5_K, 0, true),
        (14, D::Q6_K, 0, true),
        (15, D::Q8_K, 0, true),
        (16, D::Iq2Xxs, 0, true),
        (17, D::Iq2Xs, 0, true),
        (18, D::Iq3Xxs, 0, true),
        (19, D::Iq1S, 0, true),
        (20, D::Iq4Nl, 0, true),
        (21, D::Iq3S, 0, true),
        (22, D::Iq2S, 0, true),
        (23, D::Iq4Xs, 0, true),
        (24, D::I8, 1, false),
        (25, D::I16, 2, false),
        (26, D::I32, 4, false),
        (27, D::I64, 8, false),
        (28, D::F64, 8, false),
        (29, D::Iq1M, 0, true),
        (30, D::Bf16, 2, true),
    ];
    for id in (0..=512).chain(std::iter::once(u32::MAX)) {
        let expected = rows.iter().find(|r| r.0 == id).map(|r| r.1);
        assert_eq!(D::from_u32(id), expected, "tensor type {id}");
    }
    for (_, kind, size, quantized) in rows {
        assert_eq!(kind.element_size(), size, "{kind:?}");
        assert_eq!(kind.is_quantized(), quantized, "{kind:?}");
    }
}

#[test]
fn metadata_type_ids_and_scalar_sizes_do_not_alias_tensor_ids() {
    let rows = [
        (D::U8, 1),
        (D::I8, 1),
        (D::U16, 2),
        (D::I16, 2),
        (D::U32, 4),
        (D::I32, 4),
        (D::F32, 4),
        (D::Bool, 0),
        (D::String, 0),
        (D::Array, 0),
        (D::U64, 8),
        (D::I64, 8),
        (D::F64, 8),
    ];
    for id in (0..=512).chain(std::iter::once(u32::MAX)) {
        assert_eq!(
            D::from_metadata_u32(id),
            rows.get(id as usize).map(|r| r.0),
            "metadata type {id}"
        );
    }
    for (kind, size) in rows {
        assert_eq!(kind.element_size(), size);
        assert!(!kind.is_quantized());
    }
}
