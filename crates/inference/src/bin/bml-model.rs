//! Import, execute and compile supported dense Llama models.
fn main() {
    if let Err(error) = bml_inference::driver::main_cli() {
        eprintln!("bml-model: {error}");
        std::process::exit(1);
    }
}
