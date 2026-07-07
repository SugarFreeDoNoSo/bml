use bml_parser::GgufParser;
fn main() {
    let parser = GgufParser::open("/root/tinyllama.gguf").unwrap();
    println!("Architecture: {:?}", parser.architecture());
    println!("Tensor count: {}", parser.tensor_infos().len());
    println!();
    for info in parser.tensor_infos().iter().take(40) {
        println!("  {:45} dims={:?} type={:?}", info.name, info.dims, info.data_type);
    }
    if parser.tensor_infos().len() > 40 {
        println!("  ... and {} more", parser.tensor_infos().len() - 40);
    }
}
