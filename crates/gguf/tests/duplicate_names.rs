//! Independent GGUF byte fixtures for unambiguous names in v2 and v3.

use bml_gguf::{GgufError, GgufMetadataValue, GgufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "bml_gguf_names_{}_{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
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

fn header(version: u32, metadata: u64, tensors: u64) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(version.to_le_bytes());
    bytes.extend(tensors.to_le_bytes());
    bytes.extend(metadata.to_le_bytes());
    bytes
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}

fn metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
    string(bytes, key);
    bytes.extend(4u32.to_le_bytes()); // GGUF metadata U32, not GGML tensor dtype.
    bytes.extend(value.to_le_bytes());
}

fn tensor(bytes: &mut Vec<u8>, name: &str, offset: u64) {
    string(bytes, name);
    bytes.extend(1u32.to_le_bytes()); // rank
    bytes.extend(1u64.to_le_bytes()); // axis zero has one element
    bytes.extend(0u32.to_le_bytes()); // GGML F32
    bytes.extend(offset.to_le_bytes());
}

fn payload(bytes: &mut Vec<u8>, values: &[f32]) {
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    for &value in values {
        bytes.extend(value.to_le_bytes());
        bytes.extend([0; 28]);
    }
}

#[test]
fn duplicate_metadata_keys_fail_at_the_second_entry_even_when_values_match() {
    for version in [2, 3] {
        for repeated_value in [32, 64] {
            let mut bytes = header(version, 3, 0);
            metadata(&mut bytes, "general.alignment", 32);
            metadata(&mut bytes, "unrelated", 7);
            let duplicate_offset = bytes.len();
            metadata(&mut bytes, "general.alignment", repeated_value);
            let fixture = Fixture::new(&bytes);
            match fixture.open() {
                Err(GgufError::InvalidMetadata { offset, reason }) => {
                    assert_eq!(offset, duplicate_offset);
                    assert!(reason.contains("duplicate metadata key"));
                    assert!(reason.contains("general.alignment"));
                }
                other => panic!("duplicate metadata accepted or misclassified: {other:?}"),
            }
        }
    }
}

#[test]
fn duplicate_tensor_names_fail_before_selecting_either_payload() {
    for version in [2, 3] {
        let mut bytes = header(version, 0, 3);
        tensor(&mut bytes, "weight", 0);
        tensor(&mut bytes, "other", 32);
        let duplicate_offset = bytes.len();
        tensor(&mut bytes, "weight", 64);
        payload(&mut bytes, &[1.0, 2.0, -99.0]);
        let fixture = Fixture::new(&bytes);
        match fixture.open() {
            Err(GgufError::InvalidTensorInfo { offset, reason }) => {
                assert_eq!(offset, duplicate_offset);
                assert!(reason.contains("duplicate tensor name"));
                assert!(reason.contains("weight"));
            }
            other => panic!("duplicate tensor accepted or misclassified: {other:?}"),
        }
    }
}

#[test]
fn names_are_exact_and_metadata_and_tensor_namespaces_remain_independent() {
    for version in [2, 3] {
        let mut bytes = header(version, 2, 2);
        metadata(&mut bytes, "weight", 11);
        metadata(&mut bytes, "Weight", 22);
        tensor(&mut bytes, "weight", 0);
        tensor(&mut bytes, "Weight", 32);
        payload(&mut bytes, &[1.5, -2.5]);
        let fixture = Fixture::new(&bytes);
        let reader = fixture.open().unwrap();
        assert!(matches!(
            reader.get_metadata("weight"),
            Some(GgufMetadataValue::U32(11))
        ));
        assert!(matches!(
            reader.get_metadata("Weight"),
            Some(GgufMetadataValue::U32(22))
        ));
        assert_eq!(reader.read_tensor_f32("weight").unwrap(), [1.5]);
        assert_eq!(reader.read_tensor_f32("Weight").unwrap(), [-2.5]);
        assert_eq!(reader.header().version, version);
    }
}
