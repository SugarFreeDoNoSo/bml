use bml_model_weights::*;
use std::error::Error as _;

#[test]
fn inclusive_limits_preserve_overflow_as_unknown_required_size() {
    for cap in [0, 1, u32::MAX as u64, u64::MAX] {
        assert!(check_limit(Resource::MaxFileBytes, cap, Some(cap)).is_ok());
        if let Some(required) = cap.checked_add(1) {
            assert!(
                matches!(check_limit(Resource::MaxFileBytes, cap, Some(required)),
                Err(Error::Limit { limit, required: Some(n), location: None, .. }) if limit == cap && n == required)
            );
        }
        assert!(matches!(
            check_limit(Resource::MaxFileBytes, cap, None),
            Err(Error::Limit { required: None, .. })
        ));
    }
}

#[test]
fn default_profile_and_invalid_configurations_are_checked_without_allocation() {
    let defaults = ReadLimits::default();
    assert_eq!(defaults.max_file_bytes, 68_719_476_736);
    assert_eq!(defaults.max_block_bytes, 8_388_608);
    assert_eq!(defaults.max_total_terms, 4_294_967_296);
    defaults.validate().unwrap();
    ConversionOptions::default().validate().unwrap();
    for invalid in [
        ReadLimits {
            max_file_bytes: 255,
            ..defaults
        },
        ReadLimits {
            max_block_bytes: 24,
            ..defaults
        },
        ReadLimits {
            max_weight_bits: 0,
            ..defaults
        },
        ReadLimits {
            max_blocks: 0,
            ..defaults
        },
        ReadLimits {
            max_index_bytes: u64::MAX,
            ..defaults
        },
    ] {
        assert!(invalid.validate().is_err());
    }
    ReadLimits {
        max_block_bytes: 25,
        ..defaults
    }
    .validate()
    .unwrap();
    let conversion = ConversionLimits::default();
    assert_eq!(conversion.max_cache_bytes, 0);
    ConversionLimits {
        max_cache_bytes: conversion.max_resident_bytes,
        ..conversion
    }
    .validate()
    .unwrap();
    assert!(ConversionLimits {
        max_cache_bytes: conversion.max_resident_bytes + 1,
        ..conversion
    }
    .validate()
    .is_err());
    assert!(ConversionLimits {
        max_synthesis_steps_total: 0,
        ..conversion
    }
    .validate()
    .is_err());
}

#[test]
fn effective_options_never_expand_any_per_weight_budget() {
    let mut options = ConversionOptions::default();
    options.synthesis.max_bits = 100;
    options.synthesis.max_steps = 40;
    options.write.limits.max_weight_bits = 80;
    options.write.conversion.max_weight_bits = 60;
    options.write.conversion.max_synthesis_steps_per_weight = 30;
    let effective = options.effective_synthesis().unwrap();
    assert_eq!(effective.max_bits, 60);
    assert_eq!(effective.max_steps, 30);
    assert_eq!(effective.rational, options.synthesis.rational);
    options.synthesis.max_bits = 20;
    options.synthesis.max_steps = 10;
    let effective = options.effective_synthesis().unwrap();
    assert_eq!(effective.max_bits, 20);
    assert_eq!(effective.max_steps, 10);
    options.synthesis.max_bits = 0;
    assert!(options.effective_synthesis().is_err());
}

#[test]
fn write_options_validate_both_independent_budget_groups() {
    let mut options = WriteOptions::default();
    options.validate().unwrap();
    options.limits.max_weight_bits = 0;
    assert!(options.validate().is_err());
    options = WriteOptions::default();
    options.conversion.max_synthesis_steps_total = 0;
    assert!(options.validate().is_err());
}

#[test]
fn source_codes_and_ieee_metadata_are_lossless() {
    for (code, dtype) in [
        (0, SourceDtype::F32),
        (1, SourceDtype::F16),
        (2, SourceDtype::Q4_0),
        (8, SourceDtype::Q8_0),
        (14, SourceDtype::Q6K),
    ] {
        assert_eq!(SourceDtype::try_from(code).unwrap(), dtype);
        assert_eq!(u32::from(dtype), code);
    }
    for code in [3, 7, 9, 13, 15, u32::MAX] {
        assert!(
            matches!(SourceDtype::try_from(code), Err(Error::Unsupported { kind: UnsupportedKind::SourceDtype, value }) if value == u64::from(code))
        );
    }
    assert_ne!(
        MetadataValue::F32Bits(0),
        MetadataValue::F32Bits(0x8000_0000)
    );
    assert_eq!(
        MetadataValue::F32Bits(0x7fc0_0001),
        MetadataValue::F32Bits(0x7fc0_0001)
    );
    assert_ne!(MetadataArray::U8(vec![]), MetadataArray::I8(vec![]));
}

#[test]
fn sha256_matches_published_empty_and_abc_vectors() {
    assert_eq!(
        Digest::of(b"").to_string(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        Digest::of(b"abc").to_string(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn typed_error_causes_and_weight_location_survive_wrapping() {
    let location = WeightLocation {
        tensor_id: TensorId(7),
        tensor_name: "blk.0.weight".into(),
        flat_index: 42,
    };
    let synthesis = Error::Synthesis {
        location: location.clone(),
        source: ExactError::DomainNotJustified,
    };
    assert_eq!(
        synthesis.source().unwrap().downcast_ref::<ExactError>(),
        Some(&ExactError::DomainNotJustified)
    );
    assert!(synthesis.to_string().contains("blk.0.weight"));
    assert!(synthesis.to_string().contains("42"));
    assert!(matches!(&synthesis, Error::Synthesis { location: found, .. } if found == &location));
    let stream = Error::Stream {
        block: BlockId(3),
        source: StreamError::Padding,
    };
    assert_eq!(
        stream.source().unwrap().downcast_ref::<StreamError>(),
        Some(&StreamError::Padding)
    );
    let io = Error::Io {
        operation: IoOperation::Read,
        path: "weights.bmlgguf".into(),
        source: std::io::Error::from(std::io::ErrorKind::UnexpectedEof),
    };
    assert_eq!(
        io.source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::UnexpectedEof
    );
    for error in [
        Error::DestinationExists {
            path: "existing".into(),
        },
        Error::NonFiniteWeight {
            location,
            decoded_f32_bits: 0x7f80_0000,
        },
        Error::State {
            operation: WriterOperation::Finish,
            state: WriterState::Aborted,
        },
        Error::Metadata {
            key: Some("tokenizer".into()),
            reason: MetadataReason::MissingField,
        },
    ] {
        assert!(!error.to_string().is_empty());
        assert!(error.source().is_none());
    }
}

#[cfg(feature = "gguf-import")]
#[test]
fn optional_gguf_error_retains_its_original_cause() {
    let error = Error::Gguf {
        tensor: Some("weight".into()),
        source: bml_gguf::GgufError::SizeOverflow,
    };
    assert!(matches!(
        error
            .source()
            .unwrap()
            .downcast_ref::<bml_gguf::GgufError>(),
        Some(bml_gguf::GgufError::SizeOverflow)
    ));
    assert!(error.to_string().contains("weight"));
}
