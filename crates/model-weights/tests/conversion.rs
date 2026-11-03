#![cfg(feature = "gguf-import")]
use bml_model_weights::*;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "bmlgguf-convert-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn source(&self) -> PathBuf {
        self.0.join("source.gguf")
    }
    fn output(&self) -> PathBuf {
        self.0.join("output.bmlgguf")
    }
    fn assert_no_output(&self) {
        assert!(!self.output().exists());
        assert_eq!(std::fs::read_dir(&self.0).unwrap().count(), 1);
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}
fn gguf(tensors: &[(&str, Vec<u64>, u32, Vec<u8>)]) -> Vec<u8> {
    gguf_extra(tensors, &[])
}
fn gguf_extra(
    tensors: &[(&str, Vec<u64>, u32, Vec<u8>)],
    extra: &[(String, u32, Vec<u8>)],
) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((4 + extra.len() as u64).to_le_bytes());
    string(&mut out, "general.architecture");
    out.extend(8u32.to_le_bytes());
    string(&mut out, "fixture");
    string(&mut out, "test.i8");
    out.extend(1u32.to_le_bytes());
    out.push(0x80);
    string(&mut out, "test.empty");
    out.extend(9u32.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    string(&mut out, "tokenizer.ggml.tokens");
    out.extend(9u32.to_le_bytes());
    out.extend(8u32.to_le_bytes());
    out.extend(2u64.to_le_bytes());
    string(&mut out, "é");
    string(&mut out, "<0x00>");
    for (key, kind, payload) in extra {
        string(&mut out, key);
        out.extend(kind.to_le_bytes());
        out.extend(payload);
    }
    let mut offset = 0u64;
    for (name, dims, dtype, payload) in tensors {
        string(&mut out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for dim in dims {
            out.extend(dim.to_le_bytes());
        }
        out.extend(dtype.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += (payload.len() as u64).div_ceil(32) * 32;
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    for (_, _, _, payload) in tensors {
        out.extend(payload);
        out.resize(out.len().div_ceil(32) * 32, 0);
    }
    out
}

#[test]
fn gguf_import_preserves_all_scalar_and_array_metadata_types() {
    let mut text = Vec::new();
    string(&mut text, "é\0");
    let cases: Vec<(u32, Vec<u8>, MetadataValue, MetadataArray)> = vec![
        (
            0,
            vec![255],
            MetadataValue::U8(255),
            MetadataArray::U8(vec![255]),
        ),
        (
            1,
            vec![128],
            MetadataValue::I8(-128),
            MetadataArray::I8(vec![-128]),
        ),
        (
            2,
            u16::MAX.to_le_bytes().to_vec(),
            MetadataValue::U16(u16::MAX),
            MetadataArray::U16(vec![u16::MAX]),
        ),
        (
            3,
            i16::MIN.to_le_bytes().to_vec(),
            MetadataValue::I16(i16::MIN),
            MetadataArray::I16(vec![i16::MIN]),
        ),
        (
            4,
            u32::MAX.to_le_bytes().to_vec(),
            MetadataValue::U32(u32::MAX),
            MetadataArray::U32(vec![u32::MAX]),
        ),
        (
            5,
            i32::MIN.to_le_bytes().to_vec(),
            MetadataValue::I32(i32::MIN),
            MetadataArray::I32(vec![i32::MIN]),
        ),
        (
            6,
            0x80000000u32.to_le_bytes().to_vec(),
            MetadataValue::F32Bits(0x80000000),
            MetadataArray::F32Bits(vec![0x80000000]),
        ),
        (
            7,
            vec![1],
            MetadataValue::Bool(true),
            MetadataArray::Bool(vec![true]),
        ),
        (
            8,
            text,
            MetadataValue::String("é\0".into()),
            MetadataArray::String(vec!["é\0".into()]),
        ),
        (
            10,
            u64::MAX.to_le_bytes().to_vec(),
            MetadataValue::U64(u64::MAX),
            MetadataArray::U64(vec![u64::MAX]),
        ),
        (
            11,
            i64::MIN.to_le_bytes().to_vec(),
            MetadataValue::I64(i64::MIN),
            MetadataArray::I64(vec![i64::MIN]),
        ),
        (
            12,
            0x8000000000000000u64.to_le_bytes().to_vec(),
            MetadataValue::F64Bits(0x8000000000000000),
            MetadataArray::F64Bits(vec![0x8000000000000000]),
        ),
    ];
    let mut extra = Vec::new();
    let mut expected = Vec::new();
    for (kind, bytes, scalar, array) in cases {
        let key = format!("scalar.{kind}");
        extra.push((key.clone(), kind, bytes.clone()));
        expected.push((key, scalar));
        let key = format!("array.{kind}");
        let mut payload = kind.to_le_bytes().to_vec();
        payload.extend(1u64.to_le_bytes());
        payload.extend(bytes);
        extra.push((key.clone(), 9, payload));
        expected.push((key, MetadataValue::Array(array)));
    }
    let dir = Directory::new();
    std::fs::write(
        dir.source(),
        gguf_extra(&[("w", vec![1], 0, floats(&[1.]))], &extra),
    )
    .unwrap();
    convert::from_gguf(dir.source(), dir.output(), Default::default()).unwrap();
    let reader = WeightReader::open(dir.output(), Default::default()).unwrap();
    for (key, value) in expected {
        assert_eq!(
            reader
                .metadata()
                .gguf
                .iter()
                .find(|e| e.key == key)
                .unwrap()
                .value,
            value,
            "{key}"
        );
    }
}
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn exact_values(reader: &mut WeightReader, id: TensorId) -> Vec<f64> {
    let info = reader.tensor(id).unwrap().clone();
    let mut values = Vec::new();
    for block in info.first_block.0..info.first_block.0 + info.block_count {
        values.extend(
            reader
                .read_block(BlockId(block))
                .unwrap()
                .terms
                .iter()
                .map(|t| t.to_f64().unwrap()),
        );
    }
    values
}

fn assert_exact_targets(reader: &mut WeightReader, id: TensorId, expected: &[f64]) {
    use bml_domain::number::{ExactInputLimits, ExactRational};
    let info = reader.tensor(id).unwrap().clone();
    let mut index = 0;
    for block in info.first_block.0..info.first_block.0 + info.block_count {
        for term in reader.read_block(BlockId(block)).unwrap().terms {
            let certificate = term.certify_exact(Default::default()).unwrap();
            let target =
                ExactRational::from_f64(expected[index], ExactInputLimits::default()).unwrap();
            assert_eq!(
                certificate.target(),
                &target,
                "tensor {id:?}, index {index}"
            );
            index += 1;
        }
    }
    assert_eq!(index, expected.len());
}

#[test]
fn independent_f32_f16_and_quantized_blocks_convert_and_reload_without_source() {
    let dir = Directory::new();
    let f32_values = [0., -0., 1., -1., 2., 0.5];
    let f16: Vec<u8> = [0u16, 0x8000, 0x3c00, 0x3800]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut q4 = 0x3c00u16.to_le_bytes().to_vec();
    q4.extend([0x98; 16]);
    let mut q8 = 0x3800u16.to_le_bytes().to_vec();
    q8.extend((0..32).map(|i| [-2i8, 0, 2, 4][i % 4] as u8));
    let mut q6 = vec![0xff; 128];
    q6.extend([0x55; 64]);
    q6.extend([1; 16]);
    q6.extend(0x3c00u16.to_le_bytes());
    let input = gguf(&[
        ("f32", vec![3, 2], 0, floats(&f32_values)),
        ("f16", vec![4], 1, f16),
        ("q4", vec![32], 2, q4),
        ("q8", vec![32], 8, q8),
        ("q6", vec![256], 14, q6),
    ]);
    std::fs::write(dir.source(), &input).unwrap();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_cache_bytes = 1 << 20;
    options.write.conversion.max_block_terms = 7;
    let report = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
    assert_eq!(report.source_sha256, Digest::of(&input));
    assert_eq!(report.source_bytes, input.len() as u64);
    assert_eq!(report.weights_synthesized + report.weights_reused, 330);
    assert!(report.weights_reused > 0);
    assert!(report.peak_cache_bytes <= 1 << 20);
    assert!(report.peak_resident_bytes <= options.write.conversion.max_resident_bytes);
    assert!(report.synthesis_steps > 0);
    assert!(report.synthesis_steps < 330);
    std::fs::remove_file(dir.source()).unwrap();
    let mut r = WeightReader::open(dir.output(), ReadLimits::default()).unwrap();
    assert_eq!(r.tensors().len(), 5);
    assert_eq!(r.tensors()[0].ggml_dimensions, vec![3, 2]);
    assert_eq!(exact_values(&mut r, TensorId(0)), f32_values.map(f64::from));
    assert_eq!(exact_values(&mut r, TensorId(1)), vec![0., 0., 1., 0.5]);
    let mut q4_expected = vec![0.; 16];
    q4_expected.extend([1.; 16]);
    assert_eq!(exact_values(&mut r, TensorId(2)), q4_expected);
    assert_eq!(
        exact_values(&mut r, TensorId(3)),
        (0..32)
            .map(|i| [-1., 0., 1., 2.][i % 4])
            .collect::<Vec<_>>()
    );
    assert_eq!(exact_values(&mut r, TensorId(4)), vec![-1.; 256]);
    assert_exact_targets(&mut r, TensorId(0), &f32_values.map(f64::from));
    assert_exact_targets(&mut r, TensorId(1), &[0., 0., 1., 0.5]);
    assert_exact_targets(&mut r, TensorId(2), &q4_expected);
    assert_exact_targets(
        &mut r,
        TensorId(3),
        &(0..32)
            .map(|i| [-1., 0., 1., 2.][i % 4])
            .collect::<Vec<_>>(),
    );
    assert_exact_targets(&mut r, TensorId(4), &[-1.; 256]);
    let f32_zeros = r
        .read_block(r.tensor(TensorId(0)).unwrap().first_block)
        .unwrap();
    let f16_zeros = r
        .read_block(r.tensor(TensorId(1)).unwrap().first_block)
        .unwrap();
    assert_eq!(f32_zeros.terms[0], f32_zeros.terms[1]);
    assert_eq!(f16_zeros.terms[0], f16_zeros.terms[1]);
    assert_eq!(f32_zeros.terms[0], f16_zeros.terms[0]);
    assert_eq!(
        r.metadata().gguf[1].value,
        MetadataValue::Array(MetadataArray::U8(vec![]))
    );
    assert_eq!(r.metadata().gguf[2].value, MetadataValue::I8(-128));
    assert_eq!(
        r.metadata().gguf[3].value,
        MetadataValue::Array(MetadataArray::String(vec!["é".into(), "<0x00>".into()]))
    );
    assert_eq!(r.verify_all().unwrap().terms_checked, 330);
}

#[test]
fn signed_subnormal_ieee_targets_are_stored_exactly_without_float_evaluation() {
    let dir = Directory::new();
    let f32_bits = [1u32, 0x80000001, 0x007fffff, 0x807fffff];
    let half_bits = [1u16, 0x8001, 0x03ff, 0x83ff];
    let source = gguf(&[
        (
            "f32_subnormal",
            vec![4],
            0,
            f32_bits.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        (
            "f16_subnormal",
            vec![4],
            1,
            half_bits.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
    ]);
    std::fs::write(dir.source(), source).unwrap();
    convert::from_gguf(dir.source(), dir.output(), Default::default()).unwrap();
    let mut reader = WeightReader::open(dir.output(), Default::default()).unwrap();
    // Independent IEEE subnormal formula: mantissa * 2^(1-bias-fraction_bits).
    let small32 = 2f64.powi(-149);
    let large32 = 8_388_607. * small32;
    let small16 = 2f64.powi(-24);
    let large16 = 1023. * small16;
    assert_exact_targets(
        &mut reader,
        TensorId(0),
        &[small32, -small32, large32, -large32],
    );
    assert_exact_targets(
        &mut reader,
        TensorId(1),
        &[small16, -small16, large16, -large16],
    );
}

#[test]
fn half_and_quantized_nonfinite_weights_reject_at_decoded_index() {
    for scale in [0x7c00u16, 0xfc00, 0x7e00] {
        for dtype in [1u32, 2, 8, 14] {
            let mut payload = scale.to_le_bytes().to_vec();
            let elements = match dtype {
                1 => 1,
                2 => {
                    payload.extend([0x99; 16]);
                    32
                }
                8 => {
                    payload.extend([1; 32]);
                    32
                }
                14 => {
                    payload = vec![0xff; 128];
                    payload.extend([0x55; 64]);
                    payload.extend([1; 16]);
                    payload.extend(scale.to_le_bytes());
                    256
                }
                _ => unreachable!(),
            };
            let dir = Directory::new();
            let input = gguf(&[("nonfinite", vec![elements], dtype, payload)]);
            std::fs::write(dir.source(), &input).unwrap();
            assert!(
                matches!(
                    convert::from_gguf(dir.source(), dir.output(), Default::default()),
                    Err(Error::NonFiniteWeight {
                        location: WeightLocation { flat_index: 0, .. },
                        ..
                    })
                ),
                "dtype {dtype}, scale {scale:x}"
            );
            dir.assert_no_output();
            assert_eq!(std::fs::read(dir.source()).unwrap(), input);
        }
    }
}

#[test]
fn distinct_rectangular_values_keep_flat_order_across_row_crossing_blocks() {
    let dir = Directory::new();
    let expected = [1., 2., 3., 4., 0., -1.];
    std::fs::write(
        dir.source(),
        gguf(&[("rectangle", vec![3, 2], 0, floats(&expected))]),
    )
    .unwrap();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_block_terms = 2;
    convert::from_gguf(dir.source(), dir.output(), options).unwrap();
    let mut reader = WeightReader::open(dir.output(), Default::default()).unwrap();
    let info = reader.tensor(TensorId(0)).unwrap().clone();
    assert_eq!(info.ggml_dimensions, vec![3, 2]);
    assert_eq!(info.block_count, 3);
    for i in 0..3 {
        let block = reader.read_block(BlockId(info.first_block.0 + i)).unwrap();
        assert_eq!(block.first_element, 2 * i);
        assert_eq!(block.terms.len(), 2);
    }
    assert_exact_targets(&mut reader, TensorId(0), &expected.map(f64::from));
    assert_eq!(reader.verify_all().unwrap().terms_checked, 6);
}

#[test]
fn failure_after_completed_tensor_removes_all_spooled_blocks() {
    let dir = Directory::new();
    let input = gguf(&[
        ("complete", vec![2], 0, floats(&[1., 2.])),
        ("bad", vec![1], 0, floats(&[f32::INFINITY])),
    ]);
    std::fs::write(dir.source(), &input).unwrap();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_block_terms = 1;
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.output(), options),
        Err(Error::NonFiniteWeight {
            location: WeightLocation {
                tensor_id: TensorId(1),
                flat_index: 0,
                ..
            },
            ..
        })
    ));
    dir.assert_no_output();
    assert_eq!(std::fs::read(dir.source()).unwrap(), input);
}

#[test]
fn nonfinite_and_synthesis_failures_are_localized_and_transactional() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let dir = Directory::new();
        let input = gguf(&[("weights", vec![2], 0, floats(&[1., invalid]))]);
        std::fs::write(dir.source(), &input).unwrap();
        let error = convert::from_gguf(dir.source(), dir.output(), ConversionOptions::default())
            .unwrap_err();
        assert!(matches!(
            error,
            Error::NonFiniteWeight {
                location: WeightLocation {
                    tensor_id: TensorId(0),
                    flat_index: 1,
                    ..
                },
                ..
            }
        ));
        dir.assert_no_output();
        assert_eq!(std::fs::read(dir.source()).unwrap(), input);
    }
    let dir = Directory::new();
    let input = gguf(&[("weights", vec![2], 0, floats(&[1., 2.]))]);
    std::fs::write(dir.source(), input).unwrap();
    let mut options = ConversionOptions::default();
    options.synthesis.max_bits = 1;
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.output(), options),
        Err(Error::Synthesis {
            location: WeightLocation { flat_index: 1, .. },
            ..
        })
    ));
    dir.assert_no_output();
}

#[test]
fn global_steps_metadata_and_memory_budgets_do_not_leave_partial_models() {
    let dir = Directory::new();
    std::fs::write(
        dir.source(),
        gguf(&[("w", vec![3], 0, floats(&[1., 1., 1.]))]),
    )
    .unwrap();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_synthesis_steps_total = 2;
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.output(), options),
        Err(Error::Limit {
            resource: Resource::MaxSynthesisStepsTotal,
            location: Some(WeightLocation { flat_index: 2, .. }),
            ..
        })
    ));
    dir.assert_no_output();
    let mut options = ConversionOptions::default();
    options.write.limits.max_metadata_entries = 3;
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.output(), options),
        Err(Error::Limit {
            resource: Resource::MaxMetadataEntries,
            ..
        })
    ));
    dir.assert_no_output();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_resident_bytes = 1;
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.output(), options),
        Err(Error::Limit {
            resource: Resource::MaxResidentBytes,
            ..
        })
    ));
    dir.assert_no_output();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_synthesis_steps_total = 3;
    let report = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
    assert_eq!(report.synthesis_steps, 3);
    assert_eq!(report.weights_reused, 0);
}

#[test]
fn source_and_existing_destination_are_preserved() {
    let dir = Directory::new();
    let input = gguf(&[("w", vec![1], 0, floats(&[1.]))]);
    std::fs::write(dir.source(), &input).unwrap();
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.source(), ConversionOptions::default()),
        Err(Error::SourceDestinationConflict { .. })
    ));
    std::fs::write(dir.output(), b"owned by user").unwrap();
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.output(), ConversionOptions::default()),
        Err(Error::DestinationExists { .. })
    ));
    assert_eq!(std::fs::read(dir.source()).unwrap(), input);
    assert_eq!(std::fs::read(dir.output()).unwrap(), b"owned by user");
}

#[test]
fn packed_block_byte_limit_is_inclusive_across_decode_chunks() {
    let dir = Directory::new();
    // 520 one-bit terms require 65 blocks of exactly 8 terms when the BMLS
    // budget is 24 header bytes plus one payload byte. This crosses two F32
    // decode-chunk boundaries without changing the stored term order.
    std::fs::write(
        dir.source(),
        gguf(&[("w", vec![520], 0, floats(&vec![1.; 520]))]),
    )
    .unwrap();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_block_bytes = 25;
    let report = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
    assert_eq!(report.tree_bits, 520);
    assert_eq!(report.tree_bytes, 520 * 8);
    let mut reader = WeightReader::open(dir.output(), Default::default()).unwrap();
    assert_eq!(reader.summary().block_count, 65);
    for i in 0..65 {
        let info = reader.block_info(BlockId(i)).unwrap();
        assert_eq!(info.term_count, 8);
        assert_eq!(info.data_bytes, 25);
        assert_eq!(reader.read_block(BlockId(i)).unwrap().terms.len(), 8);
    }
}

#[test]
fn cache_accounting_cannot_fit_owned_certificates_in_a_single_byte() {
    let dir = Directory::new();
    std::fs::write(dir.source(), gguf(&[("w", vec![3], 0, floats(&[1.; 3]))])).unwrap();
    let mut options = ConversionOptions::default();
    // The cache adds 64 bytes per entry in addition to its owned certificate.
    // Neither 64 nor 65 bytes can hold that certificate and its proof.
    for cap in [64, 65] {
        options.write.conversion.max_cache_bytes = cap;
        let report = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
        assert_eq!(report.weights_reused, 0);
        assert_eq!(report.weights_synthesized, 3);
        assert_eq!(report.peak_cache_bytes, 0);
        std::fs::remove_file(dir.output()).unwrap();
    }
}

#[test]
fn one_certificate_cache_budget_has_an_inclusive_owned_storage_boundary() {
    use bml_domain::number::ExactStep;
    // The certificate for 1 has one tree word, two rational words and one
    // proof step, in addition to its owned header and 64-byte cache entry.
    let charge =
        (std::mem::size_of::<CertifiedBml>() + std::mem::size_of::<ExactStep>()) as u64 + 24 + 64;
    for (cap, reused) in [(charge - 1, 0), (charge, 2)] {
        let dir = Directory::new();
        std::fs::write(dir.source(), gguf(&[("w", vec![3], 0, floats(&[1.; 3]))])).unwrap();
        let mut options = ConversionOptions::default();
        options.write.conversion.max_cache_bytes = cap;
        let report = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
        assert_eq!(report.weights_reused, reused);
        assert_eq!(
            report.peak_cache_bytes,
            if reused == 0 { 0 } else { charge }
        );
    }
}

#[test]
fn retained_cache_is_not_charged_again_on_subsequent_cache_misses() {
    let dir = Directory::new();
    let values = [1., 2., 3., 4., 0., -1., 0.5, 8., 16.];
    std::fs::write(
        dir.source(),
        gguf(&[("w", vec![values.len() as u64], 0, floats(&values))]),
    )
    .unwrap();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_block_terms = 1;
    // Both caps occupy three JSON digits, so the writer's catalog reservation
    // is identical. 100 cannot retain a certificate; 999 admits early entries
    // but fills before all of these distinct values have been synthesized.
    options.write.conversion.max_cache_bytes = 100;
    let uncached = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
    std::fs::remove_file(dir.output()).unwrap();
    options.write.conversion.max_cache_bytes = 999;
    let cached = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
    assert_eq!(uncached.peak_cache_bytes, 0);
    assert!(cached.peak_cache_bytes > 0);
    assert_eq!(cached.weights_reused, 0);
    assert_eq!(cached.synthesis_steps, uncached.synthesis_steps);
    assert_eq!(
        cached.write.summary.metadata_bytes,
        uncached.write.summary.metadata_bytes
    );
    assert!(
        cached.peak_resident_bytes <= uncached.peak_resident_bytes + cached.peak_cache_bytes,
        "cache must be charged once: cached={}, uncached={}, cache={}",
        cached.peak_resident_bytes,
        uncached.peak_resident_bytes,
        cached.peak_cache_bytes
    );
}

#[test]
fn invalid_source_rank_reports_the_rank_field_byte_offset() {
    let dir = Directory::new();
    let bytes = gguf(&[("rank_failure", vec![], 0, floats(&[1.]))]);
    let name = b"rank_failure";
    let name_at = bytes.windows(name.len()).position(|w| w == name).unwrap();
    std::fs::write(dir.source(), bytes).unwrap();
    assert!(
        matches!(convert::from_gguf(dir.source(), dir.output(), Default::default()),
        Err(Error::Format { kind: FormatKind::TensorShape, offset: Some(n), .. }) if n == (name_at + name.len()) as u64)
    );
    dir.assert_no_output();
}

#[cfg(unix)]
#[test]
fn canonical_source_alias_is_reported_as_conflict() {
    let dir = Directory::new();
    let bytes = gguf(&[("w", vec![1], 0, floats(&[1.]))]);
    std::fs::write(dir.source(), &bytes).unwrap();
    std::os::unix::fs::symlink(dir.source(), dir.output()).unwrap();
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.output(), Default::default()),
        Err(Error::SourceDestinationConflict { .. })
    ));
    assert_eq!(std::fs::read(dir.source()).unwrap(), bytes);
}

#[cfg(unix)]
#[test]
fn conversion_destination_lookup_failure_precedes_source_preflight() {
    let dir = Directory::new();
    // Deliberately invalid GGUF proves that the destination lookup error is
    // surfaced first, rather than ignored and replaced by a source error.
    std::fs::write(dir.source(), b"invalid source").unwrap();
    let link = dir.output();
    std::os::unix::fs::symlink(&link, &link).unwrap();
    let error = convert::from_gguf(dir.source(), link.join("out"), Default::default()).unwrap_err();
    assert!(matches!(
        error,
        Error::Io {
            operation: IoOperation::Open,
            ..
        }
    ));
    std::fs::remove_file(link).unwrap();
    dir.assert_no_output();
}

#[test]
fn resident_budget_includes_synthesis_scratch_even_for_one_simple_weight() {
    for (steps, total_bits) in [(4096, 1024), (8, 32 * 1024 * 1024)] {
        let dir = Directory::new();
        std::fs::write(dir.source(), gguf(&[("w", vec![1], 0, floats(&[1.]))])).unwrap();
        let mut options = ConversionOptions::default();
        options.synthesis.max_steps = steps;
        options.synthesis.max_total_bits = total_bits;
        options.write.conversion.max_resident_bytes = 1024 * 1024;
        let error = convert::from_gguf(dir.source(), dir.output(), options).unwrap_err();
        assert!(
            matches!(
                error,
                Error::Limit {
                    resource: Resource::MaxResidentBytes,
                    ..
                }
            ),
            "{error}"
        );
        dir.assert_no_output();
    }
}

#[test]
fn reported_synthesis_scratch_tracks_bits_steps_and_rational_capacity() {
    let dir = Directory::new();
    std::fs::write(dir.source(), gguf(&[("w", vec![1], 0, floats(&[1.]))])).unwrap();
    let baseline = ConversionOptions::default();
    let run = |options| {
        let report = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
        std::fs::remove_file(dir.output()).unwrap();
        report.peak_resident_bytes
    };
    let base = run(baseline);
    let mut larger = baseline;
    larger.synthesis.max_total_bits += 8;
    assert_eq!(
        run(larger),
        base + 1,
        "eight retained bits require one byte"
    );
    larger = baseline;
    // Raise both step ceilings so the effective synthesis budget really grows.
    larger.synthesis.max_steps += 1;
    larger.write.conversion.max_synthesis_steps_per_weight += 1;
    let step_delta = run(larger) - base;
    assert!(
        (2048..4096).contains(&step_delta),
        "one proof slot includes rational scratch: {step_delta}"
    );
    larger = baseline;
    larger.synthesis.rational.max_bits += 2;
    assert_eq!(
        run(larger),
        base + baseline.synthesis.max_steps as u64,
        "two rational bits reserve one extra byte for every proof slot"
    );
}

#[test]
fn preflight_reports_source_metadata_and_descriptor_byte_requirements() {
    // First metadata record: u64 key length + key + u32 type + u64
    // value length + string. Tensor descriptor: name, rank, axis, dtype, offset.
    let metadata_bytes = 8 + "general.architecture".len() as u64 + 4 + 8 + "fixture".len() as u64;
    let descriptor_bytes = 8 + 1 + 4 + 8 + 4 + 8;
    for (resource, required) in [
        (Resource::MaxMetadataBytes, metadata_bytes),
        (Resource::MaxIndexBytes, descriptor_bytes),
    ] {
        let dir = Directory::new();
        std::fs::write(dir.source(), gguf(&[("w", vec![1], 0, floats(&[1.]))])).unwrap();
        let mut options = ConversionOptions::default();
        match resource {
            Resource::MaxMetadataBytes => options.write.limits.max_metadata_bytes = required - 1,
            Resource::MaxIndexBytes => options.write.limits.max_index_bytes = required - 1,
            _ => unreachable!(),
        }
        let error = convert::from_gguf(dir.source(), dir.output(), options).unwrap_err();
        assert!(
            matches!(error, Error::Limit { resource: actual, required: Some(n), .. } if actual == resource && n == required),
            "{error}"
        );
        dir.assert_no_output();
    }
}

#[test]
fn cache_saves_synthesis_but_does_not_bypass_cumulative_replay() {
    let dir = Directory::new();
    std::fs::write(
        dir.source(),
        gguf(&[("w", vec![3], 0, floats(&[1., 1., 1.]))]),
    )
    .unwrap();
    let mut options = ConversionOptions::default();
    options.write.conversion.max_cache_bytes = 1 << 20;
    options.write.conversion.max_block_terms = 1;
    options.write.conversion.max_synthesis_steps_total = 2;
    let error = convert::from_gguf(dir.source(), dir.output(), options).unwrap_err();
    assert!(matches!(
        error,
        Error::Limit {
            resource: Resource::MaxSynthesisStepsTotal,
            required: Some(3),
            location: Some(WeightLocation { flat_index: 2, .. }),
            ..
        }
    ));
    dir.assert_no_output();
    options.write.conversion.max_synthesis_steps_total = 3;
    let report = convert::from_gguf(dir.source(), dir.output(), options).unwrap();
    assert_eq!(
        (
            report.synthesis_steps,
            report.weights_synthesized,
            report.weights_reused
        ),
        (1, 1, 2)
    );
    assert!(report.peak_resident_bytes <= options.write.conversion.max_resident_bytes);
    assert!(report.peak_cache_bytes > 0);
}

#[test]
fn truncated_sources_and_unsupported_dtype_never_publish() {
    let dir = Directory::new();
    let input = gguf(&[("w", vec![2], 0, floats(&[1., 2.]))]);
    // The fixture has 24 bytes of final alignment padding, which is not tensor data.
    for end in 0..input.len() - 24 {
        std::fs::write(dir.source(), &input[..end]).unwrap();
        assert!(
            convert::from_gguf(dir.source(), dir.output(), ConversionOptions::default()).is_err(),
            "length {end}"
        );
        dir.assert_no_output();
    }
    let input = gguf(&[("unsupported", vec![32], 3, vec![0; 20])]);
    std::fs::write(dir.source(), input).unwrap();
    assert!(matches!(
        convert::from_gguf(dir.source(), dir.output(), ConversionOptions::default()),
        Err(Error::Unsupported {
            kind: UnsupportedKind::SourceDtype,
            value: 3
        })
    ));
    dir.assert_no_output();
}

#[test]
fn subnormal_weights_exceeding_explicit_tree_budget_are_not_rounded_to_zero() {
    for (dtype, payload) in [
        (0, floats(&[f32::from_bits(1)])),
        (1, 1u16.to_le_bytes().to_vec()),
    ] {
        let dir = Directory::new();
        std::fs::write(
            dir.source(),
            gguf(&[("subnormal", vec![1], dtype, payload)]),
        )
        .unwrap();
        let mut options = ConversionOptions::default();
        options.synthesis.max_bits = 1;
        let error = convert::from_gguf(dir.source(), dir.output(), options).unwrap_err();
        assert!(matches!(
            error,
            Error::Synthesis {
                location: WeightLocation { flat_index: 0, .. },
                ..
            }
        ));
        dir.assert_no_output();
    }
}
