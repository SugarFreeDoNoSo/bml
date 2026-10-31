use bml::{Bml, BuildError, ConversionError, EncodingError, EvalError, Rules, SynthesisOptions};

#[test]
fn scalar_api_builds_and_evaluates_terms() {
    let one = Bml::one();
    let two = Bml::from_binary("100").unwrap();
    assert_eq!(one.add(&one).unwrap().to_f64().unwrap(), 2.0);
    assert_eq!(Bml::from_f64(2.0).unwrap(), two);
    assert_eq!(one.exp2().unwrap(), two);
    let _ = Rules::default();
    let _ = SynthesisOptions::default();
    let _: Option<(BuildError, ConversionError, EncodingError, EvalError)> = None;
}

#[cfg(feature = "tensor")]
#[test]
fn tensor_feature_exposes_domain_tensors() {
    let shape = bml::tensor::Shape::new(vec![2, 3]).unwrap();
    assert_eq!(shape.len(), 6);
}

#[cfg(feature = "gguf")]
#[test]
fn gguf_feature_exposes_reader_api() {
    let _ = std::any::type_name::<bml::gguf::GgufReader>();
}
