//! Inspect native projection ranges for a single token, without downloads.
//! Usage: cargo run --release -p bml-compiler --example projection_bounds_probe --locked -- MODEL.gguf TOKEN_ID
//! Observations are not a proof of future activation bounds or symbolic parity.
use bml_compiler::{gguf_compiler::InferenceCompiler, HashConsedKV};
use serde_json::json;
use std::path::PathBuf;

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(args.next().ok_or("missing GGUF path")?);
    let token = args
        .next()
        .ok_or("missing token ID")?
        .to_str()
        .ok_or("invalid token ID")?
        .parse::<u32>()
        .map_err(|e| e.to_string())?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let model = InferenceCompiler::open_mapped(&path)?;
    let mut cache = HashConsedKV::try_new(
        model.config().n_layers,
        model.n_kv_heads(),
        model.head_dim(),
        1,
        512 * 1024 * 1024,
    )?;
    let mut projections = Vec::new();
    let logits = model.forward_cached_with_projection(&[token], &mut cache, &mut |input, name| {
        let input_abs = input.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut weight_abs = 0.0_f64;
        let output = model.project_rows(input, name, |x, weights| {
            let mut sum = 0.0_f64;
            for (&value, &weight) in x.iter().zip(weights) {
                weight_abs = weight_abs.max(f64::from(weight).abs());
                sum += value * f64::from(weight);
            }
            Ok(sum)
        })?;
        let (max_index, max_abs) = output.iter().enumerate().fold((0, 0.0_f64), |(i, max), (j, v)| if v.abs() > max { (j, v.abs()) } else { (i, max) });
        projections.push(json!({"tensor":name,"input_abs_max":input_abs,"weight_abs_max":weight_abs,"output_abs_max":max_abs,"output_max_index":max_index,"output_at_max":*output.get(max_index).ok_or("empty projection output")?}));
        Ok(output)
    })?;
    let argmax = bml_compiler::sampler::sample(&logits, 0.0, 0).ok_or("invalid logits")?;
    println!(
        "{}",
        json!({"profile":"native-f64-f32kv", "input_id":token, "logits":logits.len(), "argmax":argmax, "projections":projections, "scope":"Observed native values for one token, not symbolic parity or a general bound"})
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
