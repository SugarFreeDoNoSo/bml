//! Opt-in complete bounded-profile diagnostic using a local GGUF, with no downloads.
//! cargo run --release -p bml-compiler --example bounded_inference_diagnostic --locked --
//! /path/model.gguf /path/new-artifact 2
use bml_compiler::{
    gguf_compiler::InferenceCompiler, symbolic_artifact, symbolic_bounded::BoundedProfile,
    symbolic_kernels::KernelLimits, HashConsedKV,
};
use bml_runtime::{
    graph::GraphLimits,
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};

fn cache(model: &InferenceCompiler, count: u32) -> Result<HashConsedKV, String> {
    HashConsedKV::try_new(
        model.config().n_layers,
        model.n_kv_heads(),
        model.head_dim(),
        count,
        bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES,
    )
}
fn argmax(values: &[f64]) -> Result<usize, String> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite or empty logits".into());
    }
    Ok(values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0)
}
fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let source = PathBuf::from(args.next().ok_or("expected GGUF path")?);
    let destination = PathBuf::from(args.next().ok_or("expected new artifact directory")?);
    let count: u32 = args
        .next()
        .unwrap_or_else(|| "2".into())
        .parse()
        .map_err(|_| "invalid count")?;
    if !(1..=8).contains(&count) {
        return Err("diagnostic token count must be 1..=8".into());
    }
    let limits = KernelLimits {
        max_nodes: 2_000_000,
        max_ops: 4_000_000,
        max_output_ops: 2_000_000,
        max_programs: 4096,
        fragment_bytes: 32768,
    };
    let start = Instant::now();
    let manifest = symbolic_artifact::compile_directory_bounded(
        &source,
        &destination,
        count,
        BoundedProfile::default(),
        limits,
    )?;
    let compile_seconds = start.elapsed().as_secs_f64();
    let native = InferenceCompiler::open_mapped(&source)?;
    let mut symbolic = SymbolicCausalModel::open_artifact(
        &destination,
        None,
        SymbolicLimits {
            max_nodes: 0,
            max_cached_ops: 4_000_000,
            max_programs: 4096,
            graph: GraphLimits {
                max_ops: 2_000_000,
                max_steps: 4_000_000,
                max_values: 4_000_000,
            },
        },
    )?;
    let mut native_cache = cache(&native, count)?;
    let mut symbolic_cache = cache(&native, count)?;
    let mut token = native.vocab().bos_token_id;
    let bytes = std::fs::read(&source).map_err(|e| e.to_string())?;
    let source_sha256 = format!("{:x}", Sha256::digest(bytes));
    println!(
        "{}",
        json!({"stage":"compiled_and_loaded","source_sha256":source_sha256,
        "profile":manifest.profile,"context":count,"compile_seconds":compile_seconds,
        "construction_budget_on_load":0,"kernels":manifest.kernels.len(),
        "artifact":destination,"bounds":manifest.bounded_profile})
    );
    let mut records = Vec::new();
    for position in 0..count {
        let expected = native.forward_cached(&[token], &mut native_cache)?;
        let start = Instant::now();
        let actual = symbolic.forward_cached(&[token], &mut symbolic_cache)?;
        let seconds = start.elapsed().as_secs_f64();
        let actual_argmax = argmax(&actual)?;
        let expected_argmax = argmax(&expected)?;
        if actual.len() != expected.len() {
            return Err("logit dimension mismatch".into());
        }
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0, f64::max);
        let rms_error = (actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let record = json!({"stage":"token","position":position,"input":token,
            "logits":actual.len(),"symbolic_argmax":actual_argmax,"native_argmax":expected_argmax,
            "max_abs_error":max_error,"rms_error":rms_error,"seconds":seconds,
            "cache_position":symbolic_cache.current_pos(),
            "executed_families":symbolic.executed_operations()});
        println!("{record}");
        records.push(record);
        if max_error > 1e-4 || actual_argmax != expected_argmax {
            return Err(
                "bounded numerical comparison failed (absolute tolerance 1e-4 and exact argmax)"
                    .into(),
            );
        }
        token = u32::try_from(actual_argmax).map_err(|_| "token index overflow")?;
    }
    println!(
        "{}",
        json!({"stage":"completed","tokens":count,"records":records,
        "scope":"Persisted Rust bounded BML execution versus native causal path on the same local GGUF; not a llama.cpp equivalence claim"})
    );
    Ok(())
}
