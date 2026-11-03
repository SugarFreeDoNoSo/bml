use bml_domain::number::{Bml, StreamError, StreamLimits};

#[test]
fn wire_format_preserves_leaves_and_empty_streams() {
    let limits = StreamLimits::default();
    for values in [
        vec![],
        vec![Bml::one(), Bml::one()],
        vec![Bml::one(), Bml::from_u64(4).unwrap(), Bml::one()],
    ] {
        let bytes = Bml::encode_stream(&values, limits).unwrap();
        assert_eq!(&bytes[..8], b"BMLS\x01\x02\0\0");
        assert_eq!(Bml::decode_stream(&bytes, limits).unwrap(), values);
        match values.len() {
            0 => assert_eq!(bytes.len(), 24),
            2 => {
                assert_eq!(bytes[16], 2);
                assert_eq!(&bytes[24..], &[0]);
            }
            3 => {
                assert_eq!(bytes[16], 5);
                assert_eq!(&bytes[24..], &[0x40]);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn fixed_width_exports_never_truncate() {
    let mut value = Bml::one();
    for operators in 0..=64 {
        assert_eq!(value.operator_count(), operators);
        assert_eq!(value.bit_len(), 2 * operators + 1);
        assert_eq!(value.try_to_u64().is_ok(), operators <= 31);
        assert_eq!(value.try_to_u128().is_ok(), operators <= 63);
        if let Ok(code) = value.try_to_u128() {
            let words = [code as u64, (code >> 64) as u64];
            assert_eq!(Bml::from_words_le(&words).unwrap(), value);
        }
        value = value.op(&Bml::one()).unwrap();
    }
}

#[test]
fn malformed_streams_and_limits_are_rejected() {
    let limits = StreamLimits::default();
    let values = vec![Bml::one(), Bml::from_u64(4).unwrap(), Bml::one()];
    let good = Bml::encode_stream(&values, limits).unwrap();
    for length in 0..good.len() {
        assert!(Bml::decode_stream(&good[..length], limits).is_err());
    }
    for (index, value, error) in [
        (4, 2, StreamError::Header),
        (5, 3, StreamError::Header),
        (6, 1, StreamError::Header),
        (8, 2, StreamError::Length),
        (16, 4, StreamError::Length),
        (24, 0x41, StreamError::Padding),
    ] {
        let mut bad = good.clone();
        bad[index] = value;
        assert_eq!(Bml::decode_stream(&bad, limits), Err(error));
    }
    let mut bad = good.clone();
    bad.push(0);
    assert_eq!(Bml::decode_stream(&bad, limits), Err(StreamError::Length));
    for budget in [
        StreamLimits {
            max_bits: 4,
            ..limits
        },
        StreamLimits {
            max_terms: 2,
            ..limits
        },
        StreamLimits {
            max_depth: 0,
            ..limits
        },
    ] {
        assert_eq!(Bml::encode_stream(&values, budget), Err(StreamError::Limit));
        assert_eq!(Bml::decode_stream(&good, budget), Err(StreamError::Limit));
    }
    // Two internal nodes with no leaves: syntax, not just counts, is checked.
    let mut truncated_tree = Bml::encode_stream(&[Bml::one(), Bml::one()], limits).unwrap();
    truncated_tree[8] = 1;
    truncated_tree[24] = 0xc0;
    assert_eq!(
        Bml::decode_stream(&truncated_tree, limits),
        Err(StreamError::Length)
    );
}

#[test]
fn nesting_is_checked_on_both_branches_and_structure_is_preserved() {
    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let four = two.op(&one).unwrap();
    let zero = one.op(&four).unwrap();
    let invalid_real = one.op(&zero).unwrap();
    assert!(invalid_real.to_f64().is_err());
    for tree in [
        four,
        zero,
        invalid_real,
        one.op(&two).unwrap(),
        two.op(&two).unwrap(),
    ] {
        let limits = StreamLimits::default();
        let bytes = Bml::encode_stream(std::slice::from_ref(&tree), limits).unwrap();
        assert_eq!(Bml::decode_stream(&bytes, limits).unwrap(), vec![tree]);
        let shallow = StreamLimits {
            max_depth: 1,
            ..limits
        };
        assert_eq!(Bml::decode_stream(&bytes, shallow), Err(StreamError::Limit));
    }
}

#[test]
fn generated_trees_roundtrip() {
    // Exhaust all left/right chains of depth <= 10, including the empty tree.
    for depth in 0..=10 {
        for mask in 0..(1usize << depth) {
            let mut tree = Bml::one();
            for level in 0..depth {
                tree = if mask & (1 << level) != 0 {
                    tree.op(&Bml::one()).unwrap()
                } else {
                    Bml::one().op(&tree).unwrap()
                };
            }
            let values = vec![Bml::one(), tree, Bml::one()];
            let limits = StreamLimits::default();
            let bytes = Bml::encode_stream(&values, limits).unwrap();
            assert_eq!(Bml::decode_stream(&bytes, limits).unwrap(), values);
        }
    }
}
