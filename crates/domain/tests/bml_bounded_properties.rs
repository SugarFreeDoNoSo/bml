use bml_domain::number::{Bml, ExactRational, ExactSynthesisOptions, StreamLimits};

// Independent recursive grammar oracle, used only on strings of <= 13 bits.
fn parse_tree(bits: &[u8], cursor: &mut usize) -> bool {
    let Some(bit) = bits.get(*cursor) else {
        return false;
    };
    *cursor += 1;
    *bit == b'0' || (parse_tree(bits, cursor) && parse_tree(bits, cursor))
}
#[test]
fn exhaustive_small_prefix_codes_match_independent_grammar_and_streams() {
    for length in 1..=13 {
        for code in 0..(1u64 << length) {
            let bits = format!("{code:0length$b}");
            let mut cursor = 0;
            let valid = parse_tree(bits.as_bytes(), &mut cursor) && cursor == length;
            let parsed = Bml::from_binary(&bits);
            assert_eq!(parsed.is_ok(), valid, "{bits}");
            if let Ok(value) = parsed {
                assert_eq!(value.to_binary(), bits);
                assert!(Bml::from_binary(&(bits.clone() + "0")).is_err());
                if length > 1 {
                    assert!(Bml::from_binary(&bits[..length - 1]).is_err());
                }
                let bytes =
                    Bml::encode_stream(std::slice::from_ref(&value), StreamLimits::default())
                        .unwrap();
                assert_eq!(
                    Bml::decode_stream(&bytes, StreamLimits::default()).unwrap(),
                    vec![value]
                );
            }
        }
    }
}

#[test]
fn bounded_signed_arithmetic_agrees_with_independent_small_integer_results() {
    let options = ExactSynthesisOptions::default();
    for a in -4..=4 {
        for b in -4..=4 {
            let x = Bml::from_decimal_str_exact(&a.to_string()).unwrap();
            let y = Bml::from_decimal_str_exact(&b.to_string()).unwrap();
            for (result, expected) in [
                ((&x + &y).unwrap(), a + b),
                ((&x - &y).unwrap(), a - b),
                ((&x * &y).unwrap(), a * b),
            ] {
                assert_eq!(result.target(), &ExactRational::from_i64(expected));
                assert_eq!(result.to_f64_simplified(options).unwrap(), expected as f64);
            }
        }
    }
}
