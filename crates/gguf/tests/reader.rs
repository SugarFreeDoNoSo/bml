//! Independent byte fixtures for the standalone GGUF boundary.
use bml_gguf::{GgufDataType, GgufError, GgufReader};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_gguf_reader_{}_{}.gguf",
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
fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}
fn header(version: u32, tensors: u64, metadata: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(version.to_le_bytes());
    b.extend(tensors.to_le_bytes());
    b.extend(metadata.to_le_bytes());
    b
}
fn tensor(dtype: u32, dims: &[u64], alignment: u32, offset: u64, data: &[u8]) -> Vec<u8> {
    let mut b = header(3, 1, 1);
    string(&mut b, "general.alignment");
    b.extend(4u32.to_le_bytes());
    b.extend(alignment.to_le_bytes());
    string(&mut b, "w");
    b.extend((dims.len() as u32).to_le_bytes());
    for dim in dims {
        b.extend(dim.to_le_bytes());
    }
    b.extend(dtype.to_le_bytes());
    b.extend(offset.to_le_bytes());
    let a = alignment.max(1) as usize;
    b.resize(b.len().div_ceil(a) * a, 0);
    b.extend(data);
    b
}

#[test]
fn standalone_f32_f16_q4_q8_fixtures_decode_without_compiler() {
    let f32_bytes: Vec<_> = [1f32, -2.5, 0., 7.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let f16_bytes: Vec<_> = [0x3c00u16, 0xc100, 0, 0x4700]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    for (dtype, data) in [(0, f32_bytes), (1, f16_bytes)] {
        let f = Fixture::new(&tensor(dtype, &[2, 2], 64, 0, &data));
        let r = f.open().unwrap();
        assert_eq!(r.read_tensor_f32("w").unwrap(), [1., -2.5, 0., 7.]);
        assert_eq!(r.tensor_infos()[0].dims, [2, 2]);
        assert_eq!(
            r.tensor_data(&r.tensor_infos()[0]).unwrap().len(),
            data.len()
        );
        assert!(!r.is_empty());
        assert_eq!(r.len(), r.bytes().len());
    }
    let mut q4 = 0x3800u16.to_le_bytes().to_vec();
    q4.extend((0u8..16).map(|n| n | ((15 - n) << 4)));
    let f = Fixture::new(&tensor(2, &[32], 32, 0, &q4));
    let expected: Vec<f32> = (-8..8)
        .chain((-8..8).rev())
        .map(|n| n as f32 * 0.5)
        .collect();
    assert_eq!(f.open().unwrap().read_tensor_f32("w").unwrap(), expected);
    let mut q8 = 0x4000u16.to_le_bytes().to_vec();
    q8.extend((-16i8..16).map(|n| n as u8));
    let f = Fixture::new(&tensor(8, &[32], 24, 0, &q8));
    let expected: Vec<_> = (-16..16).map(|n| n as f32 * 2.).collect();
    assert_eq!(f.open().unwrap().read_tensor_f32("w").unwrap(), expected);
}

#[test]
fn typed_header_errors_and_legacy_version() {
    for length in 0..24 {
        let f = Fixture::new(&header(3, 0, 0)[..length]);
        assert!(
            matches!(f.open(), Err(GgufError::Truncated { needed: 24, available, .. }) if available == length)
        );
    }
    let mut b = header(3, 0, 0);
    b[0] = b'X';
    assert!(matches!(
        Fixture::new(&b).open(),
        Err(GgufError::InvalidMagic(_))
    ));
    assert!(matches!(
        Fixture::new(&header(1, 0, 0)).open(),
        Err(GgufError::UnsupportedVersion(1))
    ));
    assert!(matches!(
        Fixture::new(&header(4, 0, 0)).open(),
        Err(GgufError::UnsupportedVersion(4))
    ));
    let mut b = header(3, 0, 0);
    b[4..8].copy_from_slice(&3u32.to_be_bytes());
    assert!(matches!(
        Fixture::new(&b).open(),
        Err(GgufError::UnsupportedEndianness)
    ));
    b[..4].copy_from_slice(b"FUGG");
    assert!(matches!(
        Fixture::new(&b).open(),
        Err(GgufError::InvalidMagic(_))
    ));
    assert_eq!(
        Fixture::new(&header(2, 0, 0))
            .open()
            .unwrap()
            .header()
            .version,
        2
    );
}

#[test]
fn malformed_metadata_shapes_offsets_and_payloads_are_rejected() {
    let b = header(3, 0, 1);
    assert!(matches!(
        Fixture::new(&b).open(),
        Err(GgufError::InvalidMetadata { offset: 24, .. })
    ));
    for alignment in [0, 1, 3, 12] {
        let b = tensor(0, &[1], alignment, 0, &[0; 4]);
        assert!(matches!(
            Fixture::new(&b).open(),
            Err(GgufError::InvalidAlignment)
        ));
    }
    for dims in [vec![], vec![0], vec![1; 5], vec![u64::MAX, 2]] {
        let b = tensor(0, &dims, 32, 0, &[0; 4]);
        assert!(matches!(
            Fixture::new(&b).open(),
            Err(GgufError::InvalidTensorInfo { .. })
        ));
    }
    let b = tensor(0, &[1], 32, 1, &[0; 4]);
    assert!(matches!(
        Fixture::new(&b).open(),
        Err(GgufError::UnalignedTensor {
            offset: 1,
            alignment: 32,
            ..
        })
    ));
    for (dtype, dims, size) in [
        (0, vec![1], 4),
        (1, vec![1], 2),
        (2, vec![32], 18),
        (8, vec![32], 34),
    ] {
        for len in 0..size {
            let b = tensor(dtype, &dims, 32, 0, &vec![0; len]);
            assert!(matches!(
                Fixture::new(&b).open(),
                Err(GgufError::InvalidTensorRange { .. })
            ));
        }
    }
    for dtype in [2, 8] {
        let b = tensor(dtype, &[16, 2], 32, 0, &[0; 34]);
        assert!(matches!(
            Fixture::new(&b).open(),
            Err(GgufError::InvalidTensorRange { .. })
        ));
    }
    let b = tensor(0, &[1], 32, u64::MAX - 31, &[0; 4]);
    assert!(matches!(
        Fixture::new(&b).open(),
        Err(GgufError::InvalidTensorRange { .. })
    ));
}

#[test]
fn missing_unsupported_and_io_errors_remain_distinct() {
    let f = Fixture::new(&tensor(3, &[32], 32, 0, &[0; 20]));
    let r = f.open().unwrap();
    assert!(matches!(
        r.read_tensor_f32("w"),
        Err(GgufError::UnsupportedDataType(GgufDataType::Q4_1))
    ));
    assert!(
        matches!(r.read_tensor_f32("missing"), Err(GgufError::TensorNotFound(name)) if name == "missing")
    );
    let f = Fixture::new(&[]);
    std::fs::remove_file(&f.0).unwrap();
    let err = f.open().unwrap_err();
    assert!(std::error::Error::source(&err).is_some());
    let io: std::io::Error = err.into();
    assert_eq!(io.kind(), std::io::ErrorKind::NotFound);
    let err = GgufError::InvalidMagic(0);
    assert!(std::error::Error::source(&err).is_none());
    assert!(err.to_string().contains("InvalidMagic"));
    let io: std::io::Error = err.into();
    assert_eq!(io.kind(), std::io::ErrorKind::InvalidData);
    assert!(io.get_ref().unwrap().downcast_ref::<GgufError>().is_some());
}

#[test]
fn official_ggml_ids_do_not_confuse_iq1_m_and_bfloat16() {
    assert_eq!(GgufDataType::from_u32(29), Some(GgufDataType::Iq1M));
    assert_eq!(GgufDataType::from_u32(30), Some(GgufDataType::Bf16));
    assert_eq!(GgufDataType::Iq1M as u32, 29);
    assert_eq!(GgufDataType::Bf16 as u32, 30);
    let f = Fixture::new(&tensor(29, &[256], 32, 0, &[]));
    let r = f.open().unwrap();
    assert!(r.tensor_data(&r.tensor_infos()[0]).is_none());
    assert!(matches!(
        r.read_tensor_f32("w"),
        Err(GgufError::UnsupportedDataType(GgufDataType::Iq1M))
    ));
    let f = Fixture::new(&tensor(30, &[1], 32, 0, &[0x80, 0x3f]));
    let r = f.open().unwrap();
    assert_eq!(r.tensor_data(&r.tensor_infos()[0]).unwrap(), [0x80, 0x3f]);
    assert!(matches!(
        r.read_tensor_f32("w"),
        Err(GgufError::UnsupportedDataType(GgufDataType::Bf16))
    ));
}

#[test]
fn boolean_metadata_accepts_only_zero_or_one() {
    for value in [0, 1, 2, 255] {
        let mut b = header(3, 0, 1);
        string(&mut b, "test.boolean");
        b.extend(7u32.to_le_bytes());
        b.push(value);
        let f = Fixture::new(&b);
        if value <= 1 {
            let r = f.open().unwrap();
            assert!(
                matches!(r.get_metadata("test.boolean"), Some(bml_gguf::GgufMetadataValue::Bool(v)) if *v == (value == 1))
            );
        } else {
            assert!(matches!(f.open(), Err(GgufError::InvalidMetadata { .. })));
        }
    }
}

#[test]
fn truncated_fields_report_the_first_missing_field_not_the_previous_complete_one() {
    let mut b = header(3, 0, 1);
    string(&mut b, "x");
    b.extend(0u32.to_le_bytes()); // Complete type field, missing U8 payload.
    let error = Fixture::new(&b).open().unwrap_err();
    assert!(
        matches!(error, GgufError::InvalidMetadata { reason, .. } if reason == "offset fuera de rango U8")
    );

    let mut b = header(3, 1, 0);
    string(&mut b, "w");
    b.extend(1u32.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    for (length, expected) in [
        (37, "offset fuera de rango leyendo dim"),
        (45, "offset fuera de rango leyendo data_type"),
        (49, "offset fuera de rango leyendo tensor offset"),
    ] {
        let error = Fixture::new(&b[..length]).open().unwrap_err();
        assert!(matches!(error, GgufError::InvalidTensorInfo { reason, .. } if reason == expected));
    }
}

#[test]
fn every_truncated_tensor_descriptor_or_payload_is_rejected_without_panicking() {
    let mut b = header(3, 1, 0);
    string(&mut b, "w");
    b.extend(1u32.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    let descriptor_end = b.len();
    b.resize(64, 0);
    b.extend(1.0f32.to_le_bytes());
    for length in 24..descriptor_end {
        assert!(
            matches!(
                Fixture::new(&b[..length]).open(),
                Err(GgufError::InvalidTensorInfo { .. })
            ),
            "descriptor truncated at {length}"
        );
    }
    for length in descriptor_end..b.len() {
        assert!(
            matches!(
                Fixture::new(&b[..length]).open(),
                Err(GgufError::InvalidTensorRange { .. })
            ),
            "payload truncated at {length}"
        );
    }
    assert_eq!(
        Fixture::new(&b)
            .open()
            .unwrap()
            .read_tensor_f32("w")
            .unwrap(),
        [1.0]
    );
}
