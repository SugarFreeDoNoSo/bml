use bml_domain::number::{Bml, BmlMatrix, ExactRational, MatrixError, MatrixLimits};
use bml_model_weights::{Digest, Error, ReadLimits, TensorId, WeightReader};
use bml_runtime::prepared_projection::{PreparedProjection, ProjectionError};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml-prepared-projection-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
    fn reader(&self) -> WeightReader {
        WeightReader::open(&self.0, ReadLimits::default()).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn put(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

// Independent container and prefix-code construction, without synthesis or writer.
// Blocks split at element 2, crossing the first row of the [3,2] GGML tensor.
fn container(codes: &[&str]) -> Vec<u8> {
    let metadata = br#"{"schema":1,"profile":"bml-exact-decoded-f32-v1","source_gguf_version":3,"converter":{"name":"independent","version":"1","revision":"test"},"codec":{"name":"independent","version":"1","revision":"test"},"limits":{"max_weight_bits":"1048576","max_weight_depth":"4096","max_synthesis_steps_per_weight":"4096","max_synthesis_steps_total":"1000000","max_block_terms":"65536","max_block_bytes":"8388608","max_output_bytes":"68719476736","max_resident_bytes":"268435456","max_cache_bytes":"0"},"gguf":[]}"#;
    let mut tensor = vec![0; 57];
    tensor[..4].copy_from_slice(&1u32.to_le_bytes());
    tensor[4..8].copy_from_slice(&2u32.to_le_bytes());
    put(&mut tensor, 16, 6);
    put(&mut tensor, 32, 2);
    put(&mut tensor, 40, 3);
    put(&mut tensor, 48, 2);
    tensor[56] = b'w';
    let mut blocks = vec![0; 160];
    let mut payload = Vec::new();
    let payload_start = 256 + metadata.len() + tensor.len() + blocks.len();
    for (i, terms) in [&codes[..2], &codes[2..]].into_iter().enumerate() {
        let bits = terms.concat();
        let mut bmls = b"BMLS\x01\x02\0\0".to_vec();
        bmls.extend((terms.len() as u64).to_le_bytes());
        bmls.extend((bits.len() as u64).to_le_bytes());
        let mut packed = vec![0u8; bits.len().div_ceil(8)];
        for (j, bit) in bits.bytes().enumerate() {
            if bit == b'1' {
                packed[j / 8] |= 1 << (7 - j % 8);
            }
        }
        bmls.extend(packed);
        let at = i * 80;
        put(&mut blocks, at + 8, if i == 0 { 0 } else { 2 });
        put(&mut blocks, at + 16, terms.len() as u64);
        put(&mut blocks, at + 24, (payload_start + payload.len()) as u64);
        put(&mut blocks, at + 32, bmls.len() as u64);
        put(&mut blocks, at + 40, bits.len() as u64);
        blocks[at + 48..at + 80].copy_from_slice(&Digest::of(&bmls).0);
        payload.extend(bmls);
    }
    let mut bytes = vec![0; 256];
    bytes[..8].copy_from_slice(b"BMLGGUF\0");
    bytes[8] = 1;
    bytes[10] = 1;
    bytes[11] = 2;
    bytes[12..16].copy_from_slice(&256u32.to_le_bytes());
    for (at, value) in [
        (24, payload_start + payload.len()),
        (32, 1),
        (40, 2),
        (48, 256),
        (56, metadata.len()),
        (64, 256 + metadata.len()),
        (72, tensor.len()),
        (80, 256 + metadata.len() + tensor.len()),
        (88, 160),
        (96, payload_start),
        (104, payload.len()),
    ] {
        put(&mut bytes, at, value as u64);
    }
    bytes.extend(metadata);
    bytes.extend(tensor);
    bytes.extend(blocks);
    bytes.extend(payload);
    let catalog = Digest::of(&bytes[256..payload_start]);
    bytes[144..176].copy_from_slice(&catalog.0);
    let header = Digest::of(&bytes[..256]);
    bytes[176..208].copy_from_slice(&header.0);
    bytes
}
fn fixture() -> Fixture {
    Fixture::new(&container(&["0", "100", "0", "100", "0", "100"]))
}
fn inputs() -> BmlMatrix {
    // Column vector [1, 1, 2], directly encoded without exact synthesis.
    BmlMatrix::new(
        3,
        1,
        vec![Bml::one(), Bml::one(), Bml::from_u64(4).unwrap()],
        MatrixLimits::default(),
    )
    .unwrap()
}

#[test]
fn prepared_blocks_preserve_trees_and_project_against_integer_oracle() {
    let f = fixture();
    let mut reader = f.reader();
    let mut load_limits = MatrixLimits::default();
    load_limits.scalar.max_steps = 0;
    let projection = PreparedProjection::load(&mut reader, TensorId(0), load_limits).unwrap();
    assert_eq!(projection.tensor_id(), TensorId(0));
    assert_eq!(projection.name(), "w");
    assert_eq!(projection.weights().shape(), (2, 3));
    assert_eq!(
        projection
            .weights()
            .as_slice()
            .iter()
            .map(|t| t.to_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 4, 0, 4, 0, 4]
    );
    drop(reader);
    std::fs::remove_file(&f.0).unwrap();
    let before = projection.weights().as_slice().to_vec();
    for _ in 0..2 {
        let result = projection
            .apply(&inputs(), MatrixLimits::default())
            .unwrap();
        assert_eq!(result.shape(), (2, 1));
        // Independent ordinary integer dot products: [1,2,1]·[1,1,2]=5;
        // [2,1,2]·[1,1,2]=7. Compare exact certified rationals, not floats.
        for (term, expected) in result.as_slice().iter().zip([5, 7]) {
            assert_eq!(
                term.certify_exact(Default::default()).unwrap().target(),
                &ExactRational::from_i64(expected)
            );
        }
        assert_eq!(projection.weights().as_slice(), before);
    }
}

#[test]
fn aggregate_budgets_reject_before_payload_and_corruption_never_returns_matrix() {
    let mut bytes = container(&["0", "100", "0", "100", "0", "100"]);
    *bytes.last_mut().unwrap() ^= 1;
    let f = Fixture::new(&bytes);
    let mut reader = f.reader();
    let limits = MatrixLimits {
        max_elements: 5,
        ..Default::default()
    };
    assert!(matches!(
        PreparedProjection::load(&mut reader, TensorId(0), limits),
        Err(ProjectionError::Matrix(MatrixError::Limit))
    ));
    let limits = MatrixLimits {
        max_total_bits: 11,
        ..Default::default()
    };
    assert!(matches!(
        PreparedProjection::load(&mut reader, TensorId(0), limits),
        Err(ProjectionError::Matrix(MatrixError::Limit))
    ));
    assert!(matches!(
        PreparedProjection::load(&mut reader, TensorId(0), MatrixLimits::default()),
        Err(ProjectionError::Storage(Error::Integrity { .. }))
    ));
}

#[test]
fn shape_and_execution_budgets_fail_without_mutating_prepared_weights() {
    let f = fixture();
    let mut reader = f.reader();
    let limits = MatrixLimits {
        max_elements: 6,
        max_total_bits: 12,
        ..Default::default()
    };
    let projection = PreparedProjection::load(&mut reader, TensorId(0), limits).unwrap();
    let before = projection.weights().as_slice().to_vec();
    let wrong = BmlMatrix::new(1, 1, vec![Bml::one()], Default::default()).unwrap();
    assert!(matches!(
        projection.apply(&wrong, Default::default()),
        Err(ProjectionError::Matrix(MatrixError::Shape))
    ));
    let limits = MatrixLimits {
        max_work_bits: 0,
        ..Default::default()
    };
    assert!(matches!(
        projection.apply(&inputs(), limits),
        Err(ProjectionError::Matrix(MatrixError::Budget { .. }))
    ));
    assert_eq!(projection.weights().as_slice(), before);
    assert!(projection.apply(&inputs(), Default::default()).is_ok());
}

#[test]
fn structurally_valid_unjustified_weight_loads_but_execution_has_no_float_fallback() {
    // B(1, B(2,2)) = 2 - log2(3): a valid tree without rational justification.
    let f = Fixture::new(&container(&["101100100", "100", "0", "100", "0", "100"]));
    let mut reader = f.reader();
    let projection =
        PreparedProjection::load(&mut reader, TensorId(0), Default::default()).unwrap();
    assert!(matches!(
        projection.apply(&inputs(), Default::default()),
        Err(ProjectionError::Matrix(MatrixError::Scalar {
            row: 0,
            column: 0,
            ..
        }))
    ));
}
