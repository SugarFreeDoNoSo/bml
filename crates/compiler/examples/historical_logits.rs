//! Opt-in real-model logit diagnostic. No model is downloaded by this example.
use bml_compiler::gguf_compiler::InferenceCompiler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: historical_logits MODEL.gguf ID,ID,...")?;
    let ids = args.next().ok_or("missing comma-separated token IDs")?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let ids: Vec<u32> = ids.split(',').map(str::parse).collect::<Result<_, _>>()?;
    let model = InferenceCompiler::open_mapped(path)?;
    let logits = model.forward(&ids)?;
    // Both arrays contain finite numeric values, so Rust's numeric debug
    // representation is JSON-compatible and preserves round-trip precision.
    println!("{{\"input_ids\":{ids:?},\"logits\":{logits:?}}}");
    Ok(())
}
