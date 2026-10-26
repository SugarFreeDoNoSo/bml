use bml_gguf::{codecs::decode_f32, GgufDataType, GgufError};

#[test]
fn f32_fixture_preserves_bits_including_non_finite_and_signed_zero() {
    let bytes = [
        0x00, 0x00, 0x00, 0x00, // +0
        0x00, 0x00, 0x00, 0x80, // -0
        0x00, 0x00, 0xc0, 0x3f, // 1.5
        0x00, 0x00, 0x10, 0xc0, // -2.25
        0x01, 0x00, 0x00, 0x00, // smallest subnormal
        0x00, 0x00, 0x80, 0x7f, // +inf
        0x00, 0x00, 0x80, 0xff, // -inf
        0x23, 0x01, 0xc0, 0xff, // NaN payload/sign
    ];
    let values = decode_f32(GgufDataType::F32, &bytes, 8).unwrap();
    assert_eq!(
        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [
            0x0000_0000,
            0x8000_0000,
            0x3fc0_0000,
            0xc010_0000,
            0x0000_0001,
            0x7f80_0000,
            0xff80_0000,
            0xffc0_0123,
        ]
    );
}

#[test]
fn f16_fixture_covers_normal_subnormal_zero_infinity_and_nan() {
    let bytes = [
        0x00, 0x00, 0x00, 0x80, 0x00, 0x3e, 0x80, 0xc0, 0x01, 0x00, 0xff, 0x03, 0x00, 0x04, 0xff,
        0x7b, 0x00, 0x7c, 0x00, 0xfc, 0x01, 0x7c, 0x55, 0xfe,
    ];
    let values = decode_f32(GgufDataType::F16, &bytes, 12).unwrap();
    assert_eq!(
        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [
            0x0000_0000,
            0x8000_0000,
            0x3fc0_0000,
            0xc010_0000,
            0x3380_0000,
            0x387f_c000,
            0x3880_0000,
            0x477f_e000,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_2000,
            0xffca_a000,
        ]
    );
}

#[test]
fn all_binary16_encodings_match_an_independent_arithmetic_oracle() {
    let bytes: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let values = decode_f32(GgufDataType::F16, &bytes, 65536).unwrap();
    assert_eq!(values.len(), 65536);
    for (raw, &actual) in (0..=u16::MAX).zip(&values) {
        let negative = raw >= 32768;
        let exponent = (raw / 1024) % 32;
        let fraction = raw % 1024;
        if exponent == 31 && fraction != 0 {
            assert!(actual.is_nan(), "binary16 {raw:#06x}");
            let expected_bits =
                if negative { 0xff80_0000 } else { 0x7f80_0000 } + u32::from(fraction) * 8192;
            assert_eq!(actual.to_bits(), expected_bits, "binary16 {raw:#06x}");
            continue;
        }
        let magnitude = match exponent {
            0 => f64::from(fraction) / 16_777_216.0,
            31 => f64::INFINITY,
            _ => (1.0 + f64::from(fraction) / 1024.0) * 2.0_f64.powi(i32::from(exponent) - 15),
        };
        let expected = (if negative { -magnitude } else { magnitude }) as f32;
        assert_eq!(actual.to_bits(), expected.to_bits(), "binary16 {raw:#06x}");
    }
}

#[test]
fn q4_fixture_uses_low_then_high_nibbles_and_each_blocks_own_scale() {
    let mut bytes = vec![
        0x00, 0xc0, // scale -2
        0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e,
        0x0f, 0x00, 0x38, // next block scale +0.5
    ];
    bytes.extend([0x80; 16]);
    let values = decode_f32(GgufDataType::Q4_0, &bytes, 64).unwrap();
    assert_eq!(values.len(), 64);
    assert_eq!(
        &values[..32],
        &[
            16., 14., 12., 10., 8., 6., 4., 2., -0., -2., -4., -6., -8., -10., -12., -14., -14.,
            -12., -10., -8., -6., -4., -2., -0., 2., 4., 6., 8., 10., 12., 14., 16.,
        ]
    );
    assert_eq!(values[8].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(&values[32..48], &[-4.; 16]);
    assert_eq!(&values[48..], &[0.; 16]);
}

#[test]
fn q8_fixture_sign_extends_bytes_and_keeps_block_order() {
    let mut bytes = vec![
        0x00, 0x34, // scale +0.25
        128, 255, 0, 1, 127, 2, 254, 64, 192, 4, 252, 8, 248, 16, 240, 32, 224, 126, 130, 3, 253,
        5, 251, 6, 250, 7, 249, 9, 247, 10, 246, 11, 0x00, 0xbc, // next block scale -1
    ];
    bytes.extend([255; 32]);
    let values = decode_f32(GgufDataType::Q8_0, &bytes, 64).unwrap();
    assert_eq!(values.len(), 64);
    assert_eq!(
        &values[..32],
        &[
            -32., -0.25, 0., 0.25, 31.75, 0.5, -0.5, 16., -16., 1., -1., 2., -2., 4., -4., 8., -8.,
            31.5, -31.5, 0.75, -0.75, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75, 2.25, -2.25, 2.5, -2.5,
            2.75,
        ]
    );
    assert_eq!(&values[32..], &[1.; 32]);
}

#[test]
fn supported_types_accept_empty_buffers_only_for_zero_elements() {
    for data_type in [
        GgufDataType::F32,
        GgufDataType::F16,
        GgufDataType::Q4_0,
        GgufDataType::Q8_0,
    ] {
        assert!(decode_f32(data_type, &[], 0).unwrap().is_empty());
        assert!(matches!(
            decode_f32(data_type, &[0], 0),
            Err(GgufError::InvalidDataLength {
                expected: 0,
                actual: 1
            })
        ));
    }
}

#[test]
fn every_truncation_and_extra_byte_is_rejected_for_all_codecs() {
    for (data_type, elements, expected) in [
        (GgufDataType::F32, 2, 8),
        (GgufDataType::F16, 2, 4),
        (GgufDataType::Q4_0, 64, 36),
        (GgufDataType::Q8_0, 64, 68),
    ] {
        for actual in (0..expected).chain([expected + 1, expected * 2]) {
            assert!(
                matches!(
                    decode_f32(data_type, &vec![0; actual], elements),
                    Err(GgufError::InvalidDataLength { expected: e, actual: a })
                        if e == expected && a == actual
                ),
                "{data_type:?}: {actual} bytes"
            );
        }
    }
}

#[test]
fn quantized_codecs_reject_partial_element_blocks_before_reading_bytes() {
    for data_type in [GgufDataType::Q4_0, GgufDataType::Q8_0] {
        for elements in [1, 16, 31, 33, 63, usize::MAX] {
            assert!(matches!(
                decode_f32(data_type, &[], elements),
                Err(GgufError::InvalidBlockElements {
                    data_type: dt, elements: n, block_elements: 32
                }) if dt == data_type && n == elements
            ));
        }
    }
}

#[test]
fn encoded_length_overflow_is_not_a_truncated_success_or_panic() {
    for data_type in [GgufDataType::F32, GgufDataType::F16] {
        assert!(matches!(
            decode_f32(data_type, &[], usize::MAX),
            Err(GgufError::SizeOverflow)
        ));
    }
    let largest_block_count = usize::MAX / 32 * 32;
    assert!(matches!(
        decode_f32(GgufDataType::Q8_0, &[], largest_block_count),
        Err(GgufError::SizeOverflow)
    ));
}

#[test]
fn unsupported_types_are_distinct_from_malformed_supported_data() {
    for data_type in [GgufDataType::Q4_1, GgufDataType::I32, GgufDataType::Bf16] {
        assert!(matches!(
            decode_f32(data_type, &[], 0),
            Err(GgufError::UnsupportedDataType(dt)) if dt == data_type
        ));
    }
}

#[test]
fn quantized_non_finite_scales_follow_ieee_arithmetic() {
    for (data_type, block_bytes, zero, positive) in [
        (GgufDataType::Q4_0, 18, 0x88, 0x99),
        (GgufDataType::Q8_0, 34, 0x00, 0x01),
    ] {
        let mut bytes = vec![zero; block_bytes];
        bytes[0] = 0x00;
        bytes[1] = 0x7c; // +infinity times zero = NaN
        assert!(decode_f32(data_type, &bytes, 32)
            .unwrap()
            .iter()
            .all(|v| v.is_nan()));
        bytes[2..].fill(positive);
        assert!(decode_f32(data_type, &bytes, 32)
            .unwrap()
            .iter()
            .all(|&v| v == f32::INFINITY));
        bytes[0] = 0x01; // NaN scale propagates
        assert!(decode_f32(data_type, &bytes, 32)
            .unwrap()
            .iter()
            .all(|v| v.is_nan()));
    }
}
