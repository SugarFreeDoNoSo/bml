use bml_model_weights::*;
use serde_json::json;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "bmlgguf-reader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
    fn open(&self) -> Result<WeightReader, Error> {
        WeightReader::open(&self.0, ReadLimits::default())
    }
    fn write(&self, bytes: &[u8]) {
        std::fs::write(&self.0, bytes).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn put(b: &mut [u8], at: usize, value: u64) {
    b[at..at + 8].copy_from_slice(&value.to_le_bytes());
}
fn header_hash(b: &mut [u8]) {
    b[176..208].fill(0);
    let h = Digest::of(&b[..256]);
    b[176..208].copy_from_slice(&h.0);
}
fn catalog_hash(b: &mut [u8]) {
    let end = u64_at(b, 96) as usize;
    let h = Digest::of(&b[256..end]);
    b[144..176].copy_from_slice(&h.0);
    header_hash(b);
}
fn metadata() -> serde_json::Value {
    json!({"schema":1,"profile":"bml-exact-decoded-f32-v1","source_gguf_version":3,
        "converter":{"name":"fixture","version":"1","revision":"independent"},
        "codec":{"name":"fixture","version":"1","revision":"independent"},
        "limits":{"max_weight_bits":"1048576","max_weight_depth":"4096",
            "max_synthesis_steps_per_weight":"4096","max_synthesis_steps_total":"1000000",
            "max_block_terms":"65536","max_block_bytes":"8388608","max_output_bytes":"68719476736",
            "max_resident_bytes":"268435456","max_cache_bytes":"0"},
        "gguf":[{"key":"general.architecture","type":8,"value":"llama"},
            {"key":"tokenizer.ggml.model","type":8,"value":"test"},
            {"key":"z.float","type":6,"value":"80000000"},
            {"key":"z.tokens","type":9,"value":{"element_type":8,"values":["a","é"]}},
            {"key":"z.u64","type":10,"value":"18446744073709551615"}]
    })
}

#[test]
fn declared_conversion_budgets_are_preserved_and_validated() {
    let mut meta = metadata();
    let declared = ConversionLimits {
        max_weight_bits: 900,
        max_weight_depth: 90,
        max_synthesis_steps_per_weight: 80,
        max_synthesis_steps_total: 700,
        max_block_terms: 60,
        max_block_bytes: 500,
        max_output_bytes: 40000,
        max_resident_bytes: 30000,
        max_cache_bytes: 200,
    };
    meta["limits"] = json!({
        "max_weight_bits":"900", "max_weight_depth":"90",
        "max_synthesis_steps_per_weight":"80", "max_synthesis_steps_total":"700",
        "max_block_terms":"60", "max_block_bytes":"500",
        "max_output_bytes":"40000", "max_resident_bytes":"30000", "max_cache_bytes":"200"
    });
    let f = Fixture::new(&model(&serde_json::to_vec(&meta).unwrap()));
    assert_eq!(f.open().unwrap().metadata().limits, declared);
    meta["limits"]["max_synthesis_steps_total"] = json!("0");
    f.write(&model(&serde_json::to_vec(&meta).unwrap()));
    assert!(f.open().is_err());
}

#[test]
fn metadata_names_and_float_bits_require_canonical_encoding() {
    for key in ["", "bad\0key"] {
        let mut meta = metadata();
        meta["gguf"] = json!([{"key":key,"type":8,"value":"x"}]);
        let f = Fixture::new(&model(&serde_json::to_vec(&meta).unwrap()));
        assert!(matches!(
            f.open(),
            Err(Error::Metadata {
                reason: MetadataReason::InvalidValue,
                ..
            })
        ));
    }
    for (kind, text) in [
        (6, "0000000"),
        (6, "000000000"),
        (6, "ABCDEF00"),
        (12, "000000000000000"),
        (12, "ABCDEF0000000000"),
    ] {
        let mut meta = metadata();
        meta["gguf"] = json!([{"key":"bits","type":kind,"value":text}]);
        let f = Fixture::new(&model(&serde_json::to_vec(&meta).unwrap()));
        assert!(
            matches!(
                f.open(),
                Err(Error::Metadata {
                    reason: MetadataReason::InvalidValue,
                    ..
                })
            ),
            "{text}"
        );
    }
}

#[test]
fn metadata_nested_arrays_and_json_floats_keep_typed_diagnostics() {
    let mut meta = metadata();
    meta["gguf"] = json!([{"key":"nested","type":9,"value":{"element_type":9,"values":[]}}]);
    let f = Fixture::new(&model(&serde_json::to_vec(&meta).unwrap()));
    assert!(matches!(
        f.open(),
        Err(Error::Metadata {
            reason: MetadataReason::NestedArray,
            ..
        })
    ));
    // A floating JSON value must be rejected by the bounded decoder, including
    // in an otherwise unknown field; it must not become a null/default value.
    meta = metadata();
    meta["extra"] = json!(1.5);
    f.write(&model(&serde_json::to_vec(&meta).unwrap()));
    assert!(matches!(
        f.open(),
        Err(Error::Metadata {
            reason: MetadataReason::InvalidType,
            ..
        })
    ));
}

#[test]
fn bounded_json_depth_distinguishes_limit_from_unknown_fields() {
    for maps in [false, true] {
        for containers in [7, 8, 9] {
            let mut value = json!(true);
            for _ in 0..containers {
                value = if maps {
                    json!({"x": value})
                } else {
                    json!([value])
                };
            }
            let mut meta = metadata();
            meta["extra"] = value;
            let fixture = Fixture::new(&model(&serde_json::to_vec(&meta).unwrap()));
            let expected = if containers == 7 {
                MetadataReason::UnknownField
            } else {
                MetadataReason::InvalidValue
            };
            assert!(
                matches!(fixture.open(), Err(Error::Metadata { reason, .. }) if reason == expected),
                "maps={maps}, containers={containers}"
            );
        }
    }
}

#[test]
fn only_root_gguf_array_consumes_the_metadata_entry_budget() {
    let mut meta = metadata();
    meta["gguf"] = json!([]);
    meta["extra"] = json!([true, false]);
    let fixture = Fixture::new(&model(&serde_json::to_vec(&meta).unwrap()));
    let limits = ReadLimits {
        max_metadata_entries: 1,
        max_metadata_array_elements: 2,
        ..Default::default()
    };
    // The extra array fits the element budget and is rejected by schema, not
    // misclassified as two metadata entries merely because it is at the root.
    assert!(matches!(
        WeightReader::open(&fixture.0, limits),
        Err(Error::Metadata {
            reason: MetadataReason::UnknownField,
            ..
        })
    ));
}

#[test]
fn raw_negative_json_numbers_are_rejected_before_schema_validation() {
    let mut meta = metadata();
    meta["extra"] = json!(-1);
    assert!(matches!(
        Fixture::new(&model(&serde_json::to_vec(&meta).unwrap())).open(),
        Err(Error::Metadata {
            reason: MetadataReason::InvalidType,
            ..
        })
    ));
}

#[test]
fn block_coverage_overruns_are_rejected_before_following_records() {
    for case in 0..3 {
        let mut bytes = valid();
        let blocks = u64_at(&bytes, 80) as usize;
        match case {
            0 => put(&mut bytes, blocks + 40, 2), // Three terms cannot fit in two bits.
            1 => {
                put(&mut bytes, blocks + 16, 7);
                put(&mut bytes, blocks + 40, 7);
            }
            _ => {
                put(&mut bytes, blocks + 32, 124);
                put(&mut bytes, blocks + 40, 800);
            }
        }
        // If the first record is not rejected, this deliberately excessive
        // second count would surface a different resource-limit error.
        put(&mut bytes, blocks + 80 + 16, u64::MAX);
        catalog_hash(&mut bytes);
        assert!(
            matches!(
                Fixture::new(&bytes).open(),
                Err(Error::Format {
                    kind: FormatKind::BlockCoverage,
                    ..
                })
            ),
            "case {case}"
        );
    }
}

#[test]
fn wrong_term_count_is_reported_before_depth_validation() {
    let mut bytes = valid();
    let payload = u64_at(&bytes, 96) as usize;
    let blocks = u64_at(&bytes, 80) as usize;
    // 11000 is one complete depth-three tree, not the declared three terms.
    bytes[payload + 24] = 0xc0;
    let digest = Digest::of(&bytes[payload..payload + 25]);
    bytes[blocks + 48..blocks + 80].copy_from_slice(&digest.0);
    catalog_hash(&mut bytes);
    let fixture = Fixture::new(&bytes);
    let limits = ReadLimits {
        max_weight_depth: 1,
        ..Default::default()
    };
    let mut reader = WeightReader::open(&fixture.0, limits).unwrap();
    assert!(matches!(
        reader.read_block(BlockId(0)),
        Err(Error::Stream {
            source: StreamError::Length,
            ..
        })
    ));
}

#[test]
fn empty_catalog_counts_and_nul_names_have_specific_format_errors() {
    for field in [32, 40] {
        let mut bytes = model(&serde_json::to_vec(&metadata()).unwrap());
        put(&mut bytes, field, 0);
        header_hash(&mut bytes);
        let f = Fixture::new(&bytes);
        assert!(matches!(
            f.open(),
            Err(Error::Format {
                kind: FormatKind::BlockCoverage,
                ..
            })
        ));
    }
    let mut bytes = model(&serde_json::to_vec(&metadata()).unwrap());
    let tensor = u64_at(&bytes, 64) as usize;
    bytes[tensor + 56] = 0;
    catalog_hash(&mut bytes);
    let f = Fixture::new(&bytes);
    assert!(matches!(
        f.open(),
        Err(Error::Format {
            kind: FormatKind::DuplicateTensor,
            ..
        })
    ));
}

#[test]
fn unconsumed_tensor_index_bytes_are_rejected_even_with_valid_hashes() {
    let mut bytes = model(&serde_json::to_vec(&metadata()).unwrap());
    let blocks = u64_at(&bytes, 80) as usize;
    bytes.insert(blocks, 0);
    // Preserve all section and payload offsets while adding one unclaimed byte
    // after the sole tensor record. No hash or layout error can mask the tail.
    for field in [24, 72, 80, 96] {
        let value = u64_at(&bytes, field);
        put(&mut bytes, field, value + 1);
    }
    for record in 0..2 {
        let field = blocks + 1 + record * 80 + 24;
        let value = u64_at(&bytes, field);
        put(&mut bytes, field, value + 1);
    }
    catalog_hash(&mut bytes);
    assert!(Fixture::new(&bytes).open().is_err());
}
// Independent byte-level fixture. No production writer or BMLS encoder is used.
fn model(meta: &[u8]) -> Vec<u8> {
    let mut tensors = vec![0u8; 57];
    tensors[..4].copy_from_slice(&1u32.to_le_bytes());
    tensors[4..8].copy_from_slice(&2u32.to_le_bytes());
    put(&mut tensors, 16, 6);
    put(&mut tensors, 32, 2);
    put(&mut tensors, 40, 3);
    put(&mut tensors, 48, 2);
    tensors[56] = b'w';
    let mut payloads = Vec::new();
    for (bits, payload) in [(5, vec![0x40]), (13, vec![0xc5, 0x80])] {
        let mut block = b"BMLS\x01\x02\0\0".to_vec();
        block.extend(3u64.to_le_bytes());
        block.extend((bits as u64).to_le_bytes());
        block.extend(payload);
        payloads.push(block);
    }
    let payload_offset = 256 + meta.len() + tensors.len() + 160;
    let mut blocks = vec![0u8; 160];
    let mut next = payload_offset;
    for (i, payload) in payloads.iter().enumerate() {
        let at = i * 80;
        put(&mut blocks, at + 8, (i * 3) as u64);
        put(&mut blocks, at + 16, 3);
        put(&mut blocks, at + 24, next as u64);
        put(&mut blocks, at + 32, payload.len() as u64);
        put(&mut blocks, at + 40, if i == 0 { 5 } else { 13 });
        blocks[at + 48..at + 80].copy_from_slice(&Digest::of(payload).0);
        next += payload.len();
    }
    let mut bytes = vec![0u8; 256];
    bytes[..8].copy_from_slice(b"BMLGGUF\0");
    bytes[8..10].copy_from_slice(&1u16.to_le_bytes());
    bytes[10] = 1;
    bytes[11] = 2;
    bytes[12..16].copy_from_slice(&256u32.to_le_bytes());
    for (at, value) in [
        (24, next),
        (32, 1),
        (40, 2),
        (48, 256),
        (56, meta.len()),
        (64, 256 + meta.len()),
        (72, tensors.len()),
        (80, 256 + meta.len() + tensors.len()),
        (88, 160),
        (96, payload_offset),
        (104, next - payload_offset),
    ] {
        put(&mut bytes, at, value as u64);
    }
    bytes[112..144].fill(9);
    bytes.extend(meta);
    bytes.extend(tensors);
    bytes.extend(blocks);
    for payload in payloads {
        bytes.extend(payload);
    }
    catalog_hash(&mut bytes);
    bytes
}
fn valid() -> Vec<u8> {
    model(&serde_json::to_vec(&metadata()).unwrap())
}

#[test]
fn independent_fixture_loads_blocks_in_ggml_order_and_verifies_all() {
    let f = Fixture::new(&valid());
    let mut r = f.open().unwrap();
    assert_eq!(r.summary().tensor_count, 1);
    assert_eq!(r.summary().block_count, 2);
    let tensor = r.find_tensor("w").unwrap();
    assert_eq!(tensor.ggml_dimensions, vec![3, 2]);
    assert_eq!(r.tensor(TensorId(0)).unwrap(), tensor);
    assert!(r.find_tensor("absent").is_none());
    assert!(matches!(
        r.tensor(TensorId(u64::MAX)),
        Err(Error::Index {
            kind: IndexKind::Tensor,
            ..
        })
    ));
    assert!(matches!(
        r.read_block(BlockId(2)),
        Err(Error::Index {
            kind: IndexKind::Block,
            ..
        })
    ));
    assert_eq!(
        r.metadata().gguf[2].value,
        MetadataValue::F32Bits(0x80000000)
    );
    assert_eq!(r.metadata().gguf[4].value, MetadataValue::U64(u64::MAX));
    let mut codes = Vec::new();
    for id in [0, 1] {
        let b = r.read_block(BlockId(id)).unwrap();
        assert_eq!(b.first_element, id * 3);
        codes.extend(b.terms.iter().map(|t| t.to_u64().unwrap()));
    }
    assert_eq!(codes, vec![0, 4, 0, 24, 88, 0]);
    let report = r.verify_all().unwrap();
    assert!(report.all_payload_blocks_checked);
    assert_eq!(
        (
            report.blocks_checked,
            report.terms_checked,
            report.bytes_checked
        ),
        (2, 6, 51)
    );
}

#[test]
fn every_truncation_and_trailing_byte_is_rejected() {
    let bytes = valid();
    let f = Fixture::new(&bytes);
    for end in 0..bytes.len() {
        f.write(&bytes[..end]);
        assert!(f.open().is_err(), "truncated at {end}");
    }
    let mut extra = bytes;
    extra.push(0);
    f.write(&extra);
    assert!(f.open().is_err());
}

#[test]
fn hashes_are_checked_at_the_right_stage_and_on_every_block_read() {
    let bytes = valid();
    let f = Fixture::new(&bytes);
    for (at, scope) in [
        (112, IntegrityScope::Header),
        (256, IntegrityScope::Catalog),
    ] {
        let mut corrupt = bytes.clone();
        corrupt[at] ^= 1;
        f.write(&corrupt);
        assert!(matches!(f.open(),Err(Error::Integrity {scope:s,..}) if s==scope));
    }
    f.write(&bytes);
    let mut r = f.open().unwrap();
    let mut corrupt = bytes.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    f.write(&corrupt);
    assert!(r.read_block(BlockId(0)).is_ok());
    assert!(matches!(
        r.read_block(BlockId(1)),
        Err(Error::Integrity {
            scope: IntegrityScope::Block(BlockId(1)),
            ..
        })
    ));
    assert!(r.verify_all().is_err());
    f.write(&bytes);
    assert!(r.read_block(BlockId(1)).is_ok());
}

#[test]
fn correctly_rehashed_invalid_indices_are_rejected() {
    let bytes = valid();
    let tensor = u64_at(&bytes, 64) as usize;
    let block = u64_at(&bytes, 80) as usize;
    let f = Fixture::new(&bytes);
    for (at, value) in [
        (48, 257),
        (88, 80),
        (96, u64::MAX),
        (tensor + 16, 7),
        (tensor + 24, 1),
        (tensor + 32, 3),
        (tensor + 40, 0),
        (tensor + 48, u64::MAX),
        (block, 1),
        (block + 8, 1),
        (block + 16, 0),
        (block + 24, 0),
        (block + 32, 24),
        (block + 80 + 8, 2),
        (block + 80 + 24, u64_at(&bytes, 96)),
    ] {
        let mut corrupt = bytes.clone();
        put(&mut corrupt, at, value);
        // Header-offset corruption must not be followed while recomputing hashes.
        let end = u64_at(&bytes, 96) as usize;
        let h = Digest::of(&corrupt[256..end]);
        corrupt[144..176].copy_from_slice(&h.0);
        header_hash(&mut corrupt);
        f.write(&corrupt);
        assert!(f.open().is_err(), "offset {at}");
    }
}

#[test]
fn rehashed_bmls_padding_headers_and_tree_limits_are_checked() {
    let bytes = valid();
    let offset = u64_at(&bytes, 96) as usize;
    let blocks = u64_at(&bytes, 80) as usize;
    let f = Fixture::new(&bytes);
    for relative in [4, 5, 6, 8, 16, 24] {
        let mut corrupt = bytes.clone();
        corrupt[offset + relative] ^= 1;
        let h = Digest::of(&corrupt[offset..offset + 25]);
        corrupt[blocks + 48..blocks + 80].copy_from_slice(&h.0);
        catalog_hash(&mut corrupt);
        f.write(&corrupt);
        assert!(
            f.open().unwrap().read_block(BlockId(0)).is_err(),
            "relative {relative}"
        );
    }
    f.write(&bytes);
    let mut r = WeightReader::open(
        &f.0,
        ReadLimits {
            max_weight_bits: 6,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        r.read_block(BlockId(1)),
        Err(Error::Limit {
            resource: Resource::MaxWeightBits,
            required: Some(7),
            ..
        })
    ));
    let mut r = WeightReader::open(
        &f.0,
        ReadLimits {
            max_weight_bits: 7,
            max_weight_depth: 3,
            ..Default::default()
        },
    )
    .unwrap();
    r.verify_all().unwrap();
    let mut r = WeightReader::open(
        &f.0,
        ReadLimits {
            max_weight_depth: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        r.read_block(BlockId(1)),
        Err(Error::Stream {
            source: StreamError::Limit,
            ..
        })
    ));
}

#[test]
fn mismatched_block_header_is_reported_before_stream_decoding() {
    let bytes = valid();
    let offset = u64_at(&bytes, 96) as usize;
    let blocks = u64_at(&bytes, 80) as usize;
    for relative in [4, 8, 16] {
        let mut corrupt = bytes.clone();
        corrupt[offset + relative] ^= 1;
        let digest = Digest::of(&corrupt[offset..offset + 25]);
        corrupt[blocks + 48..blocks + 80].copy_from_slice(&digest.0);
        catalog_hash(&mut corrupt);
        let fixture = Fixture::new(&corrupt);
        assert!(
            matches!(
                fixture.open().unwrap().read_block(BlockId(0)),
                Err(Error::Format {
                    kind: FormatKind::BlockHeader,
                    ..
                })
            ),
            "field {relative}"
        );
    }
}

#[test]
fn short_header_is_distinguished_from_invalid_magic() {
    for length in [0, 8, 255] {
        let fixture = Fixture::new(&vec![0; length]);
        assert!(matches!(
            fixture.open(),
            Err(Error::Format {
                kind: FormatKind::HeaderLength,
                ..
            })
        ));
    }
    let fixture = Fixture::new(&[0; 256]);
    assert!(matches!(
        fixture.open(),
        Err(Error::Format {
            kind: FormatKind::Magic,
            ..
        })
    ));
}

#[test]
fn tensor_rank_and_empty_name_are_rejected_before_reading_dimensions() {
    for (field, value) in [(0, 0u32), (4, 0), (4, 5)] {
        let mut bytes = valid();
        let at = u64_at(&bytes, 64) as usize + field;
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        catalog_hash(&mut bytes);
        assert!(matches!(
            Fixture::new(&bytes).open(),
            Err(Error::Format {
                kind: FormatKind::TensorShape,
                ..
            })
        ));
    }
}

#[test]
fn section_layout_relations_are_checked_independently_before_catalog_hash() {
    for case in 0..7 {
        let mut bytes = valid();
        let fields: &[usize] = match case {
            0 => &[48, 64, 80, 96],
            1 => &[64, 80, 96],
            2 => &[80, 96],
            3 => &[40], // Block count no longer agrees with record bytes.
            4 => &[96],
            5 => &[104],
            _ => &[32], // Two tensor records cannot fit in 57 bytes.
        };
        for &field in fields {
            let value = u64_at(&bytes, field);
            put(&mut bytes, field, value + 1);
        }
        if [0, 1, 2, 4].contains(&case) {
            // Keep total file length consistent with the shifted payload.
            let value = u64_at(&bytes, 104);
            put(&mut bytes, 104, value - 1);
        }
        header_hash(&mut bytes);
        assert!(
            matches!(
                Fixture::new(&bytes).open(),
                Err(Error::Format {
                    kind: FormatKind::SectionLayout,
                    ..
                })
            ),
            "relation {case}"
        );
    }
}

#[test]
fn metadata_types_duplicates_and_resource_limits_are_enforced() {
    let mut cases = Vec::new();
    let original = metadata();
    for (key, value) in [
        ("schema", json!(2)),
        ("unknown", json!(0)),
        ("source_gguf_version", json!(4)),
    ] {
        let mut m = original.clone();
        m[key] = value;
        cases.push(serde_json::to_vec(&m).unwrap());
    }
    for value in [
        json!("01"),
        json!("+1"),
        json!("18446744073709551616"),
        json!(1),
    ] {
        let mut m = original.clone();
        m["gguf"][4]["value"] = value;
        cases.push(serde_json::to_vec(&m).unwrap());
    }
    let mut m = original.clone();
    m["gguf"][3]["value"]["element_type"] = json!(9);
    cases.push(serde_json::to_vec(&m).unwrap());
    let text = serde_json::to_string(&original).unwrap();
    cases.push(format!("{{\"schema\":1,{}", &text[1..]).into_bytes());
    let f = Fixture::new(&valid());
    for meta in cases {
        f.write(&model(&meta));
        assert!(f.open().is_err());
    }
    f.write(&valid());
    for limits in [
        ReadLimits {
            max_metadata_entries: 4,
            ..Default::default()
        },
        ReadLimits {
            max_metadata_array_elements: 1,
            ..Default::default()
        },
        ReadLimits {
            max_tensors: 1,
            max_blocks: 1,
            ..Default::default()
        },
        ReadLimits {
            max_total_terms: 5,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            WeightReader::open(&f.0, limits),
            Err(Error::Limit { .. })
        ));
    }
    WeightReader::open(
        &f.0,
        ReadLimits {
            max_metadata_entries: 5,
            max_metadata_array_elements: 2,
            max_total_terms: 6,
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn every_metadata_scalar_and_empty_array_preserves_its_type() {
    let cases = [
        (
            0,
            json!("255"),
            MetadataValue::U8(255),
            MetadataArray::U8(vec![]),
        ),
        (
            1,
            json!("-128"),
            MetadataValue::I8(-128),
            MetadataArray::I8(vec![]),
        ),
        (
            2,
            json!("65535"),
            MetadataValue::U16(65535),
            MetadataArray::U16(vec![]),
        ),
        (
            3,
            json!("-32768"),
            MetadataValue::I16(-32768),
            MetadataArray::I16(vec![]),
        ),
        (
            4,
            json!("4294967295"),
            MetadataValue::U32(u32::MAX),
            MetadataArray::U32(vec![]),
        ),
        (
            5,
            json!("-2147483648"),
            MetadataValue::I32(i32::MIN),
            MetadataArray::I32(vec![]),
        ),
        (
            6,
            json!("7fc00001"),
            MetadataValue::F32Bits(0x7fc00001),
            MetadataArray::F32Bits(vec![]),
        ),
        (
            7,
            json!(false),
            MetadataValue::Bool(false),
            MetadataArray::Bool(vec![]),
        ),
        (
            8,
            json!("\u{0}é"),
            MetadataValue::String("\0é".into()),
            MetadataArray::String(vec![]),
        ),
        (
            10,
            json!("18446744073709551615"),
            MetadataValue::U64(u64::MAX),
            MetadataArray::U64(vec![]),
        ),
        (
            11,
            json!("-9223372036854775808"),
            MetadataValue::I64(i64::MIN),
            MetadataArray::I64(vec![]),
        ),
        (
            12,
            json!("8000000000000000"),
            MetadataValue::F64Bits(0x8000000000000000),
            MetadataArray::F64Bits(vec![]),
        ),
    ];
    for (kind, value, expected, empty) in cases {
        let mut m = metadata();
        m["gguf"] = json!([
            {"key":"a", "type":kind, "value":value},
            {"key":"b", "type":9, "value":{"element_type":kind,"values":[]}},
            {"key":"c", "type":9, "value":{"element_type":kind,"values":[value]}}
        ]);
        let f = Fixture::new(&model(&serde_json::to_vec(&m).unwrap()));
        let r = f.open().unwrap();
        assert_eq!(r.metadata().gguf[0].value, expected);
        assert_eq!(r.metadata().gguf[1].value, MetadataValue::Array(empty));
        assert_ne!(r.metadata().gguf[1].value, r.metadata().gguf[2].value);
    }
}

#[test]
fn metadata_budget_precedes_parsing_an_excess_element() {
    let mut m = metadata();
    // A float JSON number is forbidden by the metadata parser. Once the array
    // budget is exhausted the limit must win before that value is materialized.
    m["gguf"][3]["value"]["values"] = json!(["ok", 1.5]);
    let f = Fixture::new(&model(&serde_json::to_vec(&m).unwrap()));
    let error = WeightReader::open(
        &f.0,
        ReadLimits {
            max_metadata_array_elements: 1,
            ..Default::default()
        },
    )
    .err()
    .unwrap();
    assert!(matches!(
        error,
        Error::Limit {
            resource: Resource::MaxMetadataArrayElements,
            required: Some(2),
            ..
        }
    ));
    let mut m = metadata();
    m["gguf"][1] = json!(1.5);
    f.write(&model(&serde_json::to_vec(&m).unwrap()));
    let error = WeightReader::open(
        &f.0,
        ReadLimits {
            max_metadata_entries: 1,
            ..Default::default()
        },
    )
    .err()
    .unwrap();
    assert!(matches!(
        error,
        Error::Limit {
            resource: Resource::MaxMetadataEntries,
            required: Some(2),
            ..
        }
    ));
}

#[test]
fn header_version_flags_and_reserved_bytes_are_not_ignored() {
    let bytes = valid();
    let f = Fixture::new(&bytes);
    for at in [0, 8, 10, 11, 12, 16, 208, 255] {
        let mut corrupt = bytes.clone();
        corrupt[at] ^= 4;
        header_hash(&mut corrupt);
        f.write(&corrupt);
        assert!(f.open().is_err(), "offset {at}");
    }
    let mut corrupt = bytes.clone();
    let tensor = u64_at(&corrupt, 64) as usize;
    corrupt[tensor + 12] = 1;
    catalog_hash(&mut corrupt);
    f.write(&corrupt);
    assert!(f.open().is_err());
}
