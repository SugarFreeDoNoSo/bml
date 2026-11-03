//! Opt-in real-model diagnostic: cargo run -p bml-compiler --example
//! bounded_rms_diagnostic --locked -- /path/to/stories15M.gguf
//! No model is downloaded. The report is not a full-inference success claim.
use bml_compiler::{
    gguf_compiler::InferenceCompiler,
    symbolic_kernels::{
        compile_operation_with_rms_bounds, KernelLimits, Operation, OutputLayout, RmsBounds,
    },
    BmlGraph,
};
use bml_domain::{symbolic::Basis, BMLTransformer, NodeKind};
use bml_runtime::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
    symbolic_causal::{SymbolicCausalModel, SymbolicLimits},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn diagnose(input: &[f64], weights: &[f64], epsilon: f64, label: &str) -> Result<Value, String> {
    let bounds = RmsBounds {
        input: 16.0,
        weight: 16.0,
    };
    let config = bounds.config(input.len(), epsilon)?;
    config
        .validate_values(input, weights)
        .map_err(|e| e.to_string())?;
    let mut builder = BMLTransformer::new();
    let x: Vec<_> = (0..input.len()).map(|i| builder.var(i as u32)).collect();
    let prefix = config.fixed_literals().len();
    let w: Vec<_> = (0..input.len())
        .map(|i| builder.external_constant((prefix + i) as u32))
        .collect();
    let roots = builder
        .rms_norm_bounded(&x, &w, &config)
        .map_err(|e| e.to_string())?;
    let soa = builder.into_soa();
    let mut constants = soa.const_pool.clone();
    constants.extend_from_slice(weights);
    let mut values: Vec<f64> = Vec::with_capacity(soa.len());
    let mut max_exponent = f64::NEG_INFINITY;
    for (index, kind) in soa.kinds.iter().enumerate() {
        let value = match *kind {
            NodeKind::One => 1.0,
            NodeKind::Var(id) => input[id as usize],
            NodeKind::Const(id) => constants[id as usize],
            NodeKind::Bml => {
                let left = values[soa.lefts[index] as usize];
                max_exponent = max_exponent.max(left);
                Basis::Bml
                    .apply_real(left, values[soa.rights[index] as usize])
                    .map_err(|e| format!("{label} node {index}: {e}"))?
            }
            NodeKind::Zero => return Err("unexpected zero opcode".into()),
        };
        values.push(value);
    }
    let limits = KernelLimits {
        max_nodes: 2_000_000,
        max_ops: 4_000_000,
        max_output_ops: 2_000_000,
        ..Default::default()
    };
    let kernel = compile_operation_with_rms_bounds(
        Operation::Norm,
        input.len(),
        epsilon,
        OutputLayout::Buffer,
        Some(bounds),
        limits,
    )?;
    let mut bound_constants = kernel.signature.fixed.clone();
    bound_constants.extend_from_slice(weights);
    let graph = &kernel.graphs[0];
    let stored = BmlGraph::deserialize(&graph.serialize(), graph.threshold)?;
    let graph = ValidatedGraph::new_with_external_constants(
        stored,
        input.len(),
        bound_constants.len(),
        1,
        input.len(),
        graph_limits(),
    )
    .map_err(|e| e.to_string())?;
    let mut buffer = ResultBuffer::from_vec(1, input.len(), vec![f64::NAN; input.len()])
        .map_err(|_| "invalid buffer dimensions")?;
    graph
        .execute_symbolic_with_constants(input, &bound_constants, &mut buffer)
        .map_err(|e| e.to_string())?;
    let reference_root =
        (input.iter().map(|v| v * v).sum::<f64>() / input.len() as f64 + epsilon).sqrt();
    let error = buffer
        .data()
        .iter()
        .zip(input.iter().zip(weights))
        .map(|(actual, (x, w))| (actual - x * w / reference_root).abs())
        .fold(0.0, f64::max);
    let bitwise_equal = buffer
        .data()
        .iter()
        .zip(&roots)
        .all(|(actual, root)| actual.to_bits() == values[*root as usize].to_bits());
    if !bitwise_equal || error > 2e-6 {
        return Err(format!(
            "{label}: oracle error={error}, DAG/RPN equality={bitwise_equal}"
        ));
    }
    Ok(json!({"case":label,"width":input.len(),"nodes":soa.len(),
        "rpn_steps":graph.execution_steps(),"max_abs_error":error,
        "max_bml_exponent":max_exponent,"dag_rpn_bitwise_equal":bitwise_equal,
        "every_bml_intermediate_finite_real":true}))
}
fn graph_limits() -> GraphLimits {
    GraphLimits {
        max_ops: 2_000_000,
        max_steps: 4_000_000,
        max_values: 4_000_000,
    }
}
fn main() -> Result<(), String> {
    let source = std::env::args().nth(1).ok_or("expected path to GGUF")?;
    let path = Path::new(&source);
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    drop(bytes);
    let model = InferenceCompiler::open_mapped(path)?;
    let bos = model.vocab().bos_token_id;
    let input = model.get_embedding(bos)?;
    let offset = *model
        .weight_offsets()
        .get("blk.0.attn_norm.weight")
        .ok_or("model lacks first-layer RMS weights")? as usize;
    let weights: Vec<_> = model
        .weight_pool()
        .get(offset..offset + input.len())
        .ok_or("invalid RMS weight span")?
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let cases = vec![
        diagnose(&[1.0], &[4.0], 1e-5, "weight-four")?,
        diagnose(&[0.0], &[4.0], 1e-5, "zero")?,
        diagnose(
            &input,
            &weights,
            model.rms_norm_epsilon(),
            "real-model-first-bos-rms",
        )?,
    ];
    let mut symbolic = SymbolicCausalModel::open_with_rms_bounds(
        path,
        SymbolicLimits {
            max_nodes: 2_000_000,
            max_cached_ops: 4_000_000,
            max_programs: 4096,
            graph: graph_limits(),
        },
        RmsBounds {
            input: 16.0,
            weight: 16.0,
        },
    )?;
    let start = Instant::now();
    let result = match symbolic.forward(&[bos]) {
        Ok(logits) => {
            json!({"status":"completed","finite_logits":logits.iter().all(|x| x.is_finite())})
        }
        Err(error) => json!({"status":"rejected","error":error}),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"source_sha256":digest,
        "profile":symbolic.profile(),"cases":cases,"full_first_token":result,
        "executed_operation_families":symbolic.executed_operations(),
        "full_first_token_seconds":start.elapsed().as_secs_f64()}))
        .map_err(|e| e.to_string())?
    );
    Ok(())
}
