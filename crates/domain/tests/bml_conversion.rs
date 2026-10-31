use bml_domain::number::{Bml, BuildError, ConversionError, SynthesisOptions};

#[test]
fn small_signed_integers_and_rationals_are_checked_pure_terms() {
    for expected in [
        -5.0, -2.0, -1.5, -0.5, 0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.25, 7.0, 13.0, 64.0,
    ] {
        let term = Bml::from_f64(expected).unwrap_or_else(|error| panic!("{expected}: {error}"));
        let actual = term.to_f64().unwrap();
        assert!(
            (actual - expected).abs() <= expected.abs() * 1e-12,
            "{expected} -> {actual}"
        );
        let restored = Bml::from_words_le(term.to_words_le()).unwrap();
        assert_eq!(restored, term);
        assert_eq!(restored.to_f64(), term.to_f64());
    }
}

#[test]
fn encoding_integers_are_distinct_from_numerical_conversion() {
    assert_eq!(Bml::from_u64(0).unwrap().to_f64().unwrap(), 1.0);
    assert_eq!(Bml::from_u64(4).unwrap().to_f64().unwrap(), 2.0);
    assert_eq!(Bml::from_f64(4.0).unwrap().to_f64().unwrap(), 4.0);
    assert_eq!(Bml::from_f64(-0.0), Bml::from_f64(0.0));
    assert_ne!(Bml::from_f64(0.0).unwrap(), Bml::one());
}

#[test]
fn exact_tolerance_and_dyadic_fallback_have_explicit_results() {
    let options = SynthesisOptions {
        max_denominator: 1,
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
        ..SynthesisOptions::default()
    };
    for value in [0.0, 0.5, 1.0, 2.0, 4.0] {
        let term = Bml::from_f64_with_options(value, options).unwrap();
        assert_eq!(term.to_f64().unwrap(), value);
    }
    // A close rational candidate may still miss the requested tolerance after
    // evaluation; acceptance must be checked against the actual output.
    let result = Bml::from_f64_with_options(
        0.1,
        SynthesisOptions {
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            ..SynthesisOptions::default()
        },
    );
    match result {
        Ok(term) => assert_eq!(term.to_f64().unwrap(), 0.1),
        Err(ConversionError::Precision {
            requested,
            actual,
            allowed_error,
        }) => {
            assert_eq!(requested, 0.1);
            assert_ne!(actual, requested);
            assert_eq!(allowed_error, 0.0);
        }
        other => panic!("unexpected conversion: {other:?}"),
    }
}

#[test]
fn synthesis_rejects_nonfinite_inputs_invalid_options_and_budget_exhaustion() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(Bml::from_f64(value), Err(ConversionError::NonFiniteInput));
    }
    let defaults = SynthesisOptions::default();
    for options in [
        SynthesisOptions {
            max_bits: 0,
            ..defaults
        },
        SynthesisOptions {
            max_denominator: 0,
            ..defaults
        },
        SynthesisOptions {
            absolute_tolerance: -1.0,
            ..defaults
        },
        SynthesisOptions {
            relative_tolerance: f64::NAN,
            ..defaults
        },
        SynthesisOptions {
            absolute_tolerance: f64::INFINITY,
            ..defaults
        },
    ] {
        assert_eq!(
            Bml::from_f64_with_options(1.0, options),
            Err(ConversionError::InvalidOptions)
        );
    }
    assert!(matches!(
        Bml::from_f64_with_options(
            2.0,
            SynthesisOptions {
                max_bits: 1,
                ..defaults
            }
        ),
        Err(ConversionError::Build(BuildError::BitLimit { .. }))
    ));
    // Pure terms can require out-of-range intermediates for ordinary inputs.
    assert!(matches!(
        Bml::from_f64(-1024.0),
        Err(ConversionError::Evaluation(_))
    ));
    assert!(matches!(
        Bml::from_f64(f64::from_bits(1)),
        Err(ConversionError::Evaluation(_))
    ));
}
