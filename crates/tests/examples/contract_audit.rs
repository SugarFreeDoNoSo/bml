//! Diagnóstico de contratos de la implementación anterior.
//!
//! Ejecutar: cargo run -p bml-tests --example contract_audit --locked
//! Sale con código 1 mientras existan discrepancias. No es un benchmark ni
//! sustituye los futuros tests de regresión asociados a cada corrección.

use bml_compiler::{gguf_compiler::read_tensor_f32, linearize, HashConsRegistry};
use bml_domain::{ast::evaluate, BMLTransformer, EvalContext, NodeId, NodeSoA};
use bml_parser::GgufParser;
use std::error::Error;
use std::path::{Path, PathBuf};

struct Fixture(PathBuf);

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn string(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
}

// Un tensor, alignment explícito, bloques con valores conocidos por separado.
fn fixture(
    directory: &Path,
    label: &str,
    alignment: usize,
    dtype: u32,
    len: u64,
    data: &[u8],
) -> Result<Fixture, Box<dyn Error>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    let metadata_count = if dtype == 2 { 2u64 } else { 1u64 };
    bytes.extend_from_slice(&metadata_count.to_le_bytes());
    string(&mut bytes, "general.alignment");
    bytes.extend_from_slice(&4u32.to_le_bytes()); // metadata UINT32
    bytes.extend_from_slice(&(alignment as u32).to_le_bytes());
    if dtype == 2 {
        string(&mut bytes, "general.quantization_version");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
    }
    string(&mut bytes, "w");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(&dtype.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.resize(bytes.len().div_ceil(alignment) * alignment, 0);
    bytes.extend_from_slice(data);
    let path = directory.join(format!(
        "bml_contract_audit_{}_{}.gguf",
        std::process::id(),
        label
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    let fixture = Fixture(path);
    std::io::Write::write_all(&mut file, &bytes)?;
    Ok(fixture)
}

fn check(label: &str, valid: bool, observed: impl std::fmt::Debug, failures: &mut usize) {
    println!(
        "{} {label}: {observed:?}",
        if valid { "PASS" } else { "FAIL" }
    );
    if !valid {
        *failures += 1;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut failures = 0;

    // Op(Op(x,y),x), x=1,y=4: Op(0,1)=1. Reusing x needs its value,
    // not the current stack top (0).
    let mut soa = NodeSoA::new();
    let x = soa.push_var(0);
    let y = soa.push_var(1);
    let child = soa.push_bml(x, y);
    let root = soa.push_bml(child, x);
    let program = linearize(&soa, root);
    let actual = program.evaluate_with_ctx(&EvalContext::new(&[1.0, 4.0], &[]));
    check(
        "shared_node_expected_1",
        actual == 1.0,
        actual,
        &mut failures,
    );

    // The existing benchmark discards the constant pool in this way.
    let mut registry = HashConsRegistry::new();
    let one = registry.one();
    let root = registry.bml(one, one);
    let soa = registry.into_soa();
    let pool = &soa.const_pool;
    let program = linearize(&soa, root);
    let with_pool = program.evaluate_with_ctx(&EvalContext::new(&[], pool));
    check(
        "constant_pool_control_expected_2",
        with_pool == 2.0,
        with_pool,
        &mut failures,
    );
    let without_pool = program.evaluate();
    check(
        "benchmark_path_preserves_constants_expected_2",
        without_pool == 2.0,
        without_pool,
        &mut failures,
    );

    let mut transformer = BMLTransformer::new();
    let two = transformer.two();
    let half = transformer.recip(two);
    let another_two = transformer.two();
    let product = transformer.mul(half, another_two);
    let soa = transformer.into_soa();
    let nodes: Vec<_> = (0..soa.len() as NodeId).map(|id| soa.get(id)).collect();
    let actual = evaluate(&nodes, product, &EvalContext::new(&[], &[]));
    check(
        "positive_product_half_times_two_expected_1",
        actual.is_finite() && (actual - 1.0).abs() < 1e-12,
        actual,
        &mut failures,
    );

    let data: Vec<u8> = [1.0f32, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let file = fixture(&std::env::temp_dir(), "alignment", 64, 0, 2, &data)?;
    let parser = GgufParser::open(&file.0)?;
    let values = read_tensor_f32(&parser, "w").ok_or("missing F32 tensor")?;
    check(
        "alignment_64_expected_1_2",
        values == [1.0, 2.0],
        &values,
        &mut failures,
    );
    drop(parser);
    drop(file);

    // Q4_0: scale=1 (f16 0x3c00). Low nibbles encode the first half;
    // high nibbles encode the second half, per ggml dequantize_row_q4_0.
    let mut data = 0x3c00u16.to_le_bytes().to_vec();
    data.extend((0u8..16).map(|j| j | ((15 - j) << 4)));
    let file = fixture(&std::env::temp_dir(), "q4_order", 32, 2, 32, &data)?;
    let parser = GgufParser::open(&file.0)?;
    let values = read_tensor_f32(&parser, "w").ok_or("missing Q4_0 tensor")?;
    let expected: Vec<f32> = (-8..8).chain((-8..8).rev()).map(|v| v as f32).collect();
    check(
        "q4_0_low_half_then_high_half",
        values == expected,
        &values,
        &mut failures,
    );
    drop(parser);
    drop(file);

    // A short quantized block must be rejected, not padded with plausible zeros.
    let file = fixture(&std::env::temp_dir(), "q4_short", 32, 2, 32, &data[..17])?;
    let parser = GgufParser::open(&file.0);
    let rejected = match &parser {
        Err(_) => true,
        Ok(parser) => read_tensor_f32(parser, "w").is_none(),
    };
    check(
        "truncated_q4_0_must_be_rejected",
        rejected,
        rejected,
        &mut failures,
    );
    drop(parser);
    drop(file);

    println!("{failures} contract discrepancy/discrepancies");
    if failures == 0 {
        Ok(())
    } else {
        Err("legacy contracts are not satisfied".into())
    }
}
