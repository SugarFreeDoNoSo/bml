//! Measurement driver for the independent one-layer prepared-weight fixture.
//! Build separately; measured invocations never include Cargo compilation.
use bml_compiler::{gguf_compiler::prepared_model::WeightEvaluation, HashConsedKV};
use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
use std::{path::Path, time::Instant};

#[path = "../tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: measure_prepared_fixture create|load|execute PATH".into());
    }
    let path = Path::new(&args[2]);
    if args[1] == "create" {
        use std::io::Write;
        let fixture = symbolic_fixture::fixture(0.25);
        let bytes = std::fs::read(&fixture.0).map_err(|e| e.to_string())?;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        output.write_all(&bytes).map_err(|e| e.to_string())?;
        println!("{{\"source_bytes\":{}}}", bytes.len());
        return Ok(());
    }
    if !matches!(args[1].as_str(), "load" | "execute") {
        return Err("unknown phase".into());
    }
    let started = Instant::now();
    let mut model = SymbolicCausalModel::open_bmlgguf(
        path,
        Default::default(),
        10_000,
        WeightEvaluation::CheckedExactF32,
        SymbolicLimits::default(),
    )?;
    let load_ns = started.elapsed().as_nanos();
    let weight_count = model.compiler().weight_pool().len();
    if args[1] == "load" {
        println!("{{\"phase\":\"load\",\"load_ns\":{load_ns},\"evaluated_weights\":{weight_count},\"profile\":{:?}}}",model.profile());
        return Ok(());
    }
    let started = Instant::now();
    let ids = model.compiler().vocab().encode("AB")?;
    if ids != [1, 3, 69, 70] {
        return Err("fixture tokenizer mismatch".into());
    }
    let tokenize_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let mut cache =
        HashConsedKV::try_new(1, 1, 2, 8, bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES)?;
    let mut logits = model.forward_cached(&ids, &mut cache)?;
    let mut generated = Vec::new();
    for step in 0..2 {
        if logits.len() != 260 || logits.iter().any(|x| !x.is_finite()) {
            return Err("invalid fixture logits".into());
        }
        let next =
            bml_compiler::sampler::sample(&logits, 0.0, step).ok_or("fixture sampling failed")?;
        generated.push(next);
        if step == 0 {
            logits = model.forward_cached(&[next], &mut cache)?;
        }
    }
    let execution_ns = started.elapsed().as_nanos();
    println!("{{\"phase\":\"execute\",\"load_ns\":{load_ns},\"tokenize_ns\":{tokenize_ns},\"execution_ns\":{execution_ns},\"evaluated_weights\":{weight_count},\"prompt_ids\":{ids:?},\"generated_ids\":{generated:?},\"profile\":{:?},\"executed_operations\":{:?}}}",model.profile(),model.executed_operations());
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
