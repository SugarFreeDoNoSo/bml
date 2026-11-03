use bml_model_weights::*;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Barrier,
    },
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "bmlgguf-writer-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn destination(&self) -> PathBuf {
        self.0.join("model.bmlgguf")
    }
    fn assert_clean(&self, published: bool) {
        let names: Vec<_> = std::fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        if published {
            assert_eq!(names, vec![std::ffi::OsString::from("model.bmlgguf")]);
        } else {
            assert!(names.is_empty(), "{names:?}");
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn metadata() -> Metadata {
    Metadata {
        schema: 1,
        profile: "bml-exact-decoded-f32-v1".into(),
        source_gguf_version: 3,
        converter: ToolIdentity {
            name: "test".into(),
            version: "1".into(),
            revision: "fixture".into(),
        },
        codec: ToolIdentity {
            name: "test".into(),
            version: "1".into(),
            revision: "fixture".into(),
        },
        limits: ConversionLimits::default(),
        gguf: vec![
            GgufMetadataEntry {
                key: "z.empty".into(),
                value: MetadataValue::Array(MetadataArray::U64(vec![])),
            },
            GgufMetadataEntry {
                key: "general.architecture".into(),
                value: MetadataValue::String("llama".into()),
            },
            GgufMetadataEntry {
                key: "z.bits".into(),
                value: MetadataValue::F32Bits(0x80000000),
            },
        ],
    }
}
fn weights(values: &[i64]) -> Vec<CertifiedBml> {
    values
        .iter()
        .map(|n| Bml::from_decimal_str_exact(&n.to_string()).unwrap())
        .collect()
}
fn writer(dir: &Directory, options: WriteOptions) -> WeightWriter {
    WeightWriter::create(dir.destination(), metadata(), Digest([7; 32]), options).unwrap()
}

#[test]
fn transactional_writer_roundtrips_rectangular_tensors_and_reports_integrity() {
    let dir = Directory::new();
    let mut w = writer(&dir, WriteOptions::default());
    assert!(!dir.destination().exists());
    assert_eq!(
        w.begin_tensor("w", &[3, 2], SourceDtype::F32).unwrap(),
        TensorId(0)
    );
    assert_eq!(w.write_block(&weights(&[1, 2, 1])).unwrap(), BlockId(0));
    assert_eq!(w.write_block(&weights(&[4, 0, 1])).unwrap(), BlockId(1));
    w.end_tensor().unwrap();
    w.begin_tensor("bias", &[1], SourceDtype::F16).unwrap();
    w.write_block(&weights(&[-1])).unwrap();
    w.end_tensor().unwrap();
    assert!(!dir.destination().exists());
    let report = w.finish().unwrap();
    dir.assert_clean(true);
    let bytes = std::fs::read(dir.destination()).unwrap();
    assert_eq!(report.output_sha256, Digest::of(&bytes));
    assert_eq!(report.summary.file_bytes, bytes.len() as u64);
    assert_eq!(report.terms_written, 7);
    assert_eq!(
        report.peak_temporary_bytes,
        report.summary.file_bytes + report.summary.payload_bytes
    );
    let mut r = WeightReader::open(dir.destination(), ReadLimits::default()).unwrap();
    assert_eq!(r.summary(), &report.summary);
    assert_eq!(r.tensors()[0].ggml_dimensions, vec![3, 2]);
    assert_eq!(r.metadata().gguf[0].key, "general.architecture");
    assert_eq!(
        r.metadata().gguf[1].value,
        MetadataValue::F32Bits(0x80000000)
    );
    let actual: Vec<_> = (0..3)
        .flat_map(|id| r.read_block(BlockId(id)).unwrap().terms)
        .collect();
    assert_eq!(
        actual,
        weights(&[1, 2, 1, 4, 0, 1, -1])
            .into_iter()
            .map(|w| w.into_term())
            .collect::<Vec<_>>()
    );
    assert_eq!(r.verify_all().unwrap().terms_checked, 7);
}

#[test]
fn preexisting_destination_is_never_modified() {
    let dir = Directory::new();
    std::fs::write(dir.destination(), b"user content").unwrap();
    assert!(matches!(
        WeightWriter::create(
            dir.destination(),
            metadata(),
            Digest([0; 32]),
            WriteOptions::default()
        ),
        Err(Error::DestinationExists { .. })
    ));
    assert_eq!(std::fs::read(dir.destination()).unwrap(), b"user content");
    dir.assert_clean(true);
}

#[test]
fn concurrent_publication_has_exactly_one_winner() {
    let dir = Directory::new();
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for value in [1, 2] {
        let mut w = writer(&dir, WriteOptions::default());
        w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
        w.write_block(&weights(&[value])).unwrap();
        w.end_tensor().unwrap();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            (value, w.finish())
        }));
    }
    let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(outcomes.iter().filter(|(_, r)| r.is_ok()).count(), 1);
    assert_eq!(
        outcomes
            .iter()
            .filter(|(_, r)| matches!(r, Err(Error::DestinationExists { .. })))
            .count(),
        1
    );
    let winner = outcomes.iter().find(|(_, r)| r.is_ok()).unwrap().0;
    let mut r = WeightReader::open(dir.destination(), ReadLimits::default()).unwrap();
    assert_eq!(
        r.read_block(BlockId(0)).unwrap().terms,
        weights(&[winner])
            .into_iter()
            .map(|w| w.into_term())
            .collect::<Vec<_>>()
    );
    dir.assert_clean(true);
}

#[test]
fn invalid_operations_poison_the_writer_and_drop_cleans_spools() {
    for case in 0..7 {
        let dir = Directory::new();
        let mut w = writer(&dir, WriteOptions::default());
        let result = match case {
            0 => w.write_block(&weights(&[1])).map(|_| ()),
            1 => w.end_tensor(),
            2 => {
                w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
                w.begin_tensor("v", &[1], SourceDtype::F32).map(|_| ())
            }
            3 => {
                w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
                w.end_tensor()
            }
            4 => {
                w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
                w.write_block(&[]).map(|_| ())
            }
            5 => {
                w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
                w.write_block(&weights(&[1, 2])).map(|_| ())
            }
            _ => w
                .begin_tensor("w", &[u64::MAX, 2], SourceDtype::F32)
                .map(|_| ()),
        };
        assert!(result.is_err());
        assert!(matches!(
            w.begin_tensor("retry", &[1], SourceDtype::F32),
            Err(Error::State {
                state: WriterState::Aborted,
                ..
            })
        ));
        assert!(w.finish().is_err());
        dir.assert_clean(false);
    }
    let dir = Directory::new();
    writer(&dir, WriteOptions::default()).abort().unwrap();
    dir.assert_clean(false);
    assert!(writer(&dir, WriteOptions::default()).finish().is_err());
    dir.assert_clean(false);
    {
        let mut w = writer(&dir, WriteOptions::default());
        w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
        assert!(w.finish().is_err());
    }
    dir.assert_clean(false);
}

#[test]
fn per_weight_and_global_budgets_fail_before_publication() {
    let cases = [
        ConversionLimits {
            max_weight_bits: 2,
            ..Default::default()
        },
        ConversionLimits {
            max_synthesis_steps_per_weight: 1,
            ..Default::default()
        },
        ConversionLimits {
            max_synthesis_steps_total: 1,
            ..Default::default()
        },
        ConversionLimits {
            max_block_terms: 1,
            ..Default::default()
        },
        ConversionLimits {
            max_block_bytes: 25,
            ..Default::default()
        },
        ConversionLimits {
            max_weight_depth: 1,
            ..Default::default()
        },
    ];
    for conversion in cases {
        let dir = Directory::new();
        let mut w = writer(
            &dir,
            WriteOptions {
                conversion,
                ..Default::default()
            },
        );
        w.begin_tensor("w", &[3], SourceDtype::F32).unwrap();
        assert!(
            w.write_block(&weights(&[4, 0, 4])).is_err(),
            "{conversion:?}"
        );
        assert!(w.finish().is_err());
        dir.assert_clean(false);
    }
    let dir = Directory::new();
    let options = WriteOptions {
        conversion: ConversionLimits {
            max_resident_bytes: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(WeightWriter::create(dir.destination(), metadata(), Digest([0; 32]), options).is_err());
    dir.assert_clean(false);
}

#[test]
fn replay_memory_budget_accepts_exact_capacity_and_rejects_one_byte_less() {
    use bml_domain::number::ExactStep;
    let dir = Directory::new();
    let mut options = WriteOptions::default();
    options.conversion.max_resident_bytes = 3_000_000;
    let mut w = writer(&dir, options);
    w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
    let certificate = weights(&[1]);
    assert_eq!(certificate[0].derivation().steps.len(), 1);
    w.write_block(&certificate).unwrap();
    w.end_tensor().unwrap();
    let report = w.finish().unwrap();
    // The published catalog supplies its actual encoded size. For the one-step
    // proof the default replay reservation is 2 MiB of retained bits, 2048
    // bytes of rational scratch, 256 scratch bytes and one ExactStep slot.
    let catalog =
        report.summary.metadata_bytes + report.summary.tensors_bytes + report.summary.blocks_bytes;
    let required = 8 * (catalog + 25)
        + 65_536
        + 2_097_152
        + 2048
        + 256
        + std::mem::size_of::<ExactStep>() as u64;
    assert!((1_000_000..3_000_000).contains(&required));
    std::fs::remove_file(dir.destination()).unwrap();
    // All three caps have seven decimal digits, so recorded metadata size is
    // unchanged. This fixes the acceptance boundary independently of a search.
    for (cap, accepted) in [(required - 1, false), (required, true)] {
        options.conversion.max_resident_bytes = cap;
        let mut w = writer(&dir, options);
        w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
        let result = w.write_block(&certificate);
        if accepted {
            result.unwrap();
            w.end_tensor().unwrap();
            w.finish().unwrap();
            std::fs::remove_file(dir.destination()).unwrap();
        } else {
            assert!(
                matches!(result, Err(Error::Limit { resource: Resource::MaxResidentBytes, required: Some(n), .. }) if n == required)
            );
            drop(w);
        }
        dir.assert_clean(false);
    }
}

#[test]
fn abort_reports_cleanup_failure_instead_of_silently_succeeding() {
    let dir = Directory::new();
    let mut w = writer(&dir, WriteOptions::default());
    w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
    w.write_block(&weights(&[1])).unwrap();
    let spool = std::fs::read_dir(&dir.0)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    // Replace the temporary's directory entry while its handle is still open.
    // remove_file must report the directory conflict, even though Drop also
    // attempts best-effort cleanup after abort returns.
    std::fs::remove_file(&spool).unwrap();
    std::fs::create_dir(&spool).unwrap();
    assert!(matches!(
        w.abort(),
        Err(Error::Io {
            operation: IoOperation::Cleanup,
            ..
        })
    ));
    assert!(!dir.destination().exists());
    std::fs::remove_dir(spool).unwrap();
    dir.assert_clean(false);
}

#[test]
fn abort_succeeds_when_spool_was_already_removed() {
    let dir = Directory::new();
    let w = writer(&dir, WriteOptions::default());
    let spool = std::fs::read_dir(&dir.0)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::remove_file(spool).unwrap();
    w.abort().unwrap();
    dir.assert_clean(false);
}

#[test]
fn missing_destination_parent_preserves_original_create_error() {
    let dir = Directory::new();
    let destination = dir.0.join("missing").join("model.bmlgguf");
    let error = WeightWriter::create(destination, metadata(), Digest([0; 32]), Default::default())
        .err()
        .unwrap();
    assert!(
        matches!(error, Error::Io { operation: IoOperation::CreateTemporary, source, .. }
        if source.kind() == std::io::ErrorKind::NotFound)
    );
    dir.assert_clean(false);
}

#[cfg(unix)]
#[test]
fn destination_lookup_failure_is_not_treated_as_absence() {
    let dir = Directory::new();
    let link = dir.0.join("loop");
    std::os::unix::fs::symlink("loop", &link).unwrap();
    let error = WeightWriter::create(
        link.join("out"),
        metadata(),
        Digest([0; 32]),
        Default::default(),
    )
    .err()
    .unwrap();
    assert!(matches!(
        error,
        Error::Io {
            operation: IoOperation::Open,
            ..
        }
    ));
    std::fs::remove_file(link).unwrap();
    dir.assert_clean(false);
}

#[test]
fn tensor_name_and_rank_constraints_fail_independently() {
    for (name, dims) in [("", vec![1]), ("w", vec![]), ("w", vec![1; 5])] {
        let dir = Directory::new();
        let mut w = writer(&dir, WriteOptions::default());
        assert!(w.begin_tensor(name, &dims, SourceDtype::F32).is_err());
        assert!(w.finish().is_err());
        dir.assert_clean(false);
    }
}

#[test]
fn output_budget_is_inclusive_and_short_budget_aborts() {
    let dir = Directory::new();
    let mut w = writer(&dir, WriteOptions::default());
    w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
    w.write_block(&weights(&[1])).unwrap();
    w.end_tensor().unwrap();
    let size = w.finish().unwrap().summary.file_bytes;
    // Provenance includes the decimal budget itself; stabilise its digit width.
    let mut target = size;
    for _ in 0..3 {
        let trial = Directory::new();
        let mut options = WriteOptions::default();
        options.conversion.max_output_bytes = target + 100;
        options.conversion.max_block_bytes = 25;
        let mut w = writer(&trial, options);
        w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
        w.write_block(&weights(&[1])).unwrap();
        w.end_tensor().unwrap();
        target = w.finish().unwrap().summary.file_bytes;
    }
    for allowed in [target, target - 1] {
        let trial = Directory::new();
        let mut options = WriteOptions::default();
        options.conversion.max_output_bytes = allowed;
        options.conversion.max_block_bytes = 25;
        let mut w = writer(&trial, options);
        w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
        let result = w.write_block(&weights(&[1]));
        if allowed == target {
            result.unwrap();
            w.end_tensor().unwrap();
            assert_eq!(w.finish().unwrap().summary.file_bytes, target);
            trial.assert_clean(true);
        } else {
            assert!(matches!(
                result,
                Err(Error::Limit {
                    resource: Resource::MaxOutputBytes,
                    ..
                })
            ));
            assert!(w.finish().is_err());
            trial.assert_clean(false);
        }
    }
}

#[test]
fn every_typed_metadata_value_roundtrips_without_json_numeric_loss() {
    let values = vec![
        MetadataValue::U8(u8::MAX),
        MetadataValue::I8(i8::MIN),
        MetadataValue::U16(u16::MAX),
        MetadataValue::I16(i16::MIN),
        MetadataValue::U32(u32::MAX),
        MetadataValue::I32(i32::MIN),
        MetadataValue::U64(u64::MAX),
        MetadataValue::I64(i64::MIN),
        MetadataValue::F32Bits(0x7fc00001),
        MetadataValue::F64Bits(0x8000000000000000),
        MetadataValue::Bool(true),
        MetadataValue::String("é\0\n\"".into()),
        MetadataValue::Array(MetadataArray::U8(vec![0, u8::MAX])),
        MetadataValue::Array(MetadataArray::I8(vec![0, i8::MIN])),
        MetadataValue::Array(MetadataArray::U16(vec![0, u16::MAX])),
        MetadataValue::Array(MetadataArray::I16(vec![0, i16::MIN])),
        MetadataValue::Array(MetadataArray::U32(vec![0, u32::MAX])),
        MetadataValue::Array(MetadataArray::I32(vec![0, i32::MIN])),
        MetadataValue::Array(MetadataArray::U64(vec![0, u64::MAX])),
        MetadataValue::Array(MetadataArray::I64(vec![0, i64::MIN])),
        MetadataValue::Array(MetadataArray::F32Bits(vec![0, 0x80000000, 0x7fc00001])),
        MetadataValue::Array(MetadataArray::F64Bits(vec![0, 0x8000000000000000])),
        MetadataValue::Array(MetadataArray::Bool(vec![true, false])),
        MetadataValue::Array(MetadataArray::String(vec!["".into(), "é\0".into()])),
    ];
    let mut m = metadata();
    m.gguf = values
        .iter()
        .enumerate()
        .map(|(i, v)| GgufMetadataEntry {
            key: format!("k_{i:02}"),
            value: v.clone(),
        })
        .collect();
    let dir = Directory::new();
    let mut w = WeightWriter::create(
        dir.destination(),
        m.clone(),
        Digest([0; 32]),
        WriteOptions::default(),
    )
    .unwrap();
    w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
    w.write_block(&weights(&[1])).unwrap();
    w.end_tensor().unwrap();
    w.finish().unwrap();
    let r = WeightReader::open(dir.destination(), ReadLimits::default()).unwrap();
    assert_eq!(r.metadata(), &m);
}

#[test]
fn cumulative_replay_budget_is_enforced_across_blocks_with_weight_location() {
    let dir = Directory::new();
    let mut options = WriteOptions::default();
    options.conversion.max_synthesis_steps_total = 1;
    let mut w = writer(&dir, options);
    w.begin_tensor("w", &[2], SourceDtype::F32).unwrap();
    w.write_block(&weights(&[1])).unwrap();
    let error = w.write_block(&weights(&[1])).unwrap_err();
    assert!(matches!(
        error,
        Error::Limit {
            resource: Resource::MaxSynthesisStepsTotal,
            required: Some(2),
            location: Some(WeightLocation {
                tensor_id: TensorId(0),
                flat_index: 1,
                ..
            }),
            ..
        }
    ));
    assert!(w.finish().is_err());
    dir.assert_clean(false);
}

#[test]
fn corrupt_spool_is_rejected_before_the_publication_point() {
    use std::io::{Seek, SeekFrom, Write};
    let dir = Directory::new();
    let mut w = writer(&dir, WriteOptions::default());
    w.begin_tensor("w", &[1], SourceDtype::F32).unwrap();
    w.write_block(&weights(&[1])).unwrap();
    w.end_tensor().unwrap();
    let spool = std::fs::read_dir(&dir.0)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut file = std::fs::OpenOptions::new().write(true).open(spool).unwrap();
    file.seek(SeekFrom::Start(24)).unwrap();
    file.write_all(&[0xff]).unwrap();
    drop(file);
    assert!(matches!(
        w.finish(),
        Err(Error::Integrity {
            scope: IntegrityScope::Block(BlockId(0)),
            ..
        })
    ));
    dir.assert_clean(false);
}

#[test]
fn malformed_metadata_and_duplicate_tensor_names_leave_no_output() {
    let dir = Directory::new();
    let mut m = metadata();
    m.gguf.push(m.gguf[0].clone());
    assert!(matches!(
        WeightWriter::create(
            dir.destination(),
            m,
            Digest([0; 32]),
            WriteOptions::default()
        ),
        Err(Error::Metadata {
            reason: MetadataReason::DuplicateKey,
            ..
        })
    ));
    dir.assert_clean(false);
    let mut w = writer(&dir, WriteOptions::default());
    w.begin_tensor("same", &[1], SourceDtype::F32).unwrap();
    w.write_block(&weights(&[1])).unwrap();
    w.end_tensor().unwrap();
    assert!(matches!(
        w.begin_tensor("same", &[1], SourceDtype::F32),
        Err(Error::Format {
            kind: FormatKind::DuplicateTensor,
            ..
        })
    ));
    w.abort().unwrap();
    dir.assert_clean(false);
}
