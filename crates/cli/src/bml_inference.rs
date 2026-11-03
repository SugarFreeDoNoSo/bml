//! Integración de building blocks BML con el runtime.
//!
//! Este módulo conecta los programas RPN generados por `bml_ops` con el
//! `HotLoop` del runtime para ejecutar las proyecciones IEEE del transformer.
//!
//! # Flujo
//!
//! 1. Escribir `x` (hidden state) en un slot del ResultBuffer
//! 2. Escribir una fila de `W` en otro slot del ResultBuffer
//! 3. Generar el programa RPN con `bml_dot_program()`
//! 4. Ejecutar via `HotLoop::execute_full()`
//! 5. Recoger el escalar devuelto para cada fila

use bml_compiler::bml_ops;
use bml_compiler::gguf_compiler::InferenceCompiler;
use bml_domain::EvalContext;
use bml_runtime::buffer::ResultBuffer;
use bml_runtime::hot_loop::HotLoop;

/// Row-major GGUF projection using native IEEE FMul/FAdd RPN instructions.
/// Scratch storage contains the input and one weight row, never the full matrix.
pub fn bml_matmul(
    hot: &mut HotLoop,
    x: &[f64],
    weights: &[f32],
    weight_offset: u64,
    n_in: u32,
    n_out: u32,
) -> Result<Vec<f64>, String> {
    if n_in == 0 || n_out == 0 || x.len() != n_in as usize {
        return Err("invalid RPN projection dimensions".into());
    }
    n_in.checked_mul(2).ok_or("RPN buffer index overflow")?;
    let start = usize::try_from(weight_offset).map_err(|_| "weight offset overflow")?;
    let count = (n_in as usize)
        .checked_mul(n_out as usize)
        .ok_or("weight size overflow")?;
    let end = start.checked_add(count).ok_or("weight range overflow")?;
    let matrix = weights.get(start..end).ok_or("weight range outside pool")?;
    let mut buf = ResultBuffer::new(2, n_in as usize);
    for (i, &value) in x.iter().enumerate() {
        buf.write(0, i as u32, value);
    }
    let program = bml_ops::bml_dot_program(0, n_in, n_in);
    let ctx = EvalContext::new(&[], &[]);
    let mut output = Vec::with_capacity(n_out as usize);
    for row in matrix.chunks_exact(n_in as usize) {
        for (i, &value) in row.iter().enumerate() {
            buf.write(1, i as u32, f64::from(value));
        }
        output.push(hot.execute_full(&program, &ctx, &mut buf));
    }
    Ok(output)
}

/// Native RMSNorm helper with the legacy default epsilon of 1e-5.
pub fn bml_rmsnorm(
    hot: &mut HotLoop,
    x: &[f64],
    weights: &[f32],
    weight_offset: u64,
    n_embd: usize,
) -> Vec<f64> {
    bml_rmsnorm_with_epsilon(hot, x, weights, weight_offset, n_embd, 1e-5)
}

fn bml_rmsnorm_with_epsilon(
    _hot: &mut HotLoop,
    x: &[f64],
    weights: &[f32],
    weight_offset: u64,
    n_embd: usize,
    eps: f64,
) -> Vec<f64> {
    let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / n_embd as f64;
    let rms = (mean_sq + eps).sqrt();
    let recip_rms = 1.0 / rms;

    // Por ahora, ejecutar directo (el RPN de rmsnorm aún no soporta StoreResult por elemento)
    let mut y = vec![0.0_f64; n_embd];
    for i in 0..n_embd {
        let w = if (weight_offset as usize + i) < weights.len() {
            weights[weight_offset as usize + i] as f64
        } else {
            1.0
        };
        y[i] = x[i] * recip_rms * w;
    }
    y
}

/// Native SwiGLU used by the experimental BML execution path.
pub fn bml_swiglu(gate: &[f64], up: &[f64]) -> Vec<f64> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| bml_compiler::gguf_compiler::swiglu_scalar(g, u))
        .collect()
}

/// Ejecuta RoPE usando BML puro para negación.
pub fn bml_rope(x: &mut [f64], pos: usize, head_dim: usize) {
    let n_half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    for h in 0..n_heads {
        for i in 0..n_half {
            let even_idx = h * head_dim + i * 2;
            let odd_idx = h * head_dim + i * 2 + 1;
            let x_even = x[even_idx];
            let x_odd = x[odd_idx];

            let freq = 1.0 / 10000.0_f64.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            let c = angle.cos();
            let s = angle.sin();

            // x_even' = x_even * cos - x_odd * sin
            // neg(x_odd * sin) via BML: bml(log2(0), exp2(x_odd*sin)) = 0 - y
            // Aquí lo ejecutamos directo (la versión RPN está en bml_ops)
            x[even_idx] = x_even * c - x_odd * s;
            x[odd_idx] = x_even * s + x_odd * c;
        }
    }
}

/// Full causal prefix with RPN matrix projections and shared native attention.
/// This is not symbolic AST/EML evaluation.
pub fn forward_bml(
    hot: &mut HotLoop,
    compiler: &InferenceCompiler,
    input_ids: &[u32],
) -> Result<Vec<f64>, String> {
    compiler.forward_with_projection(input_ids, &mut |x, name| project(hot, compiler, x, name))
}

/// Incremental causal decode; committed cache positions advance only on success.
pub fn forward_bml_cached(
    hot: &mut HotLoop,
    compiler: &InferenceCompiler,
    input_ids: &[u32],
    cache: &mut bml_compiler::HashConsedKV,
) -> Result<Vec<f64>, String> {
    compiler.forward_cached_with_projection(input_ids, cache, &mut |x, name| {
        project(hot, compiler, x, name)
    })
}

fn project(
    hot: &mut HotLoop,
    compiler: &InferenceCompiler,
    x: &[f64],
    name: &str,
) -> Result<Vec<f64>, String> {
    let columns = u32::try_from(x.len()).map_err(|_| "RPN dimension overflow")?;
    if columns == 0 || columns.checked_mul(2).is_none() {
        return Err("invalid RPN row size".into());
    }
    let mut buffer = ResultBuffer::new(2, x.len());
    for (i, &value) in x.iter().enumerate() {
        buffer.write(0, i as u32, value);
    }
    let program = bml_ops::bml_dot_program(0, columns, columns);
    let context = EvalContext::new(&[], &[]);
    compiler.project_rows(x, name, |_, row| {
        for (i, &value) in row.iter().enumerate() {
            buffer.write(1, i as u32, f64::from(value));
        }
        Ok(hot.execute_full(&program, &context, &mut buffer))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_activation_and_normalization_match_closed_form_answers() {
        let log_two = std::f64::consts::LN_2;
        let actual = bml_swiglu(&[log_two, -log_two, 0.0], &[3.0, 3.0, 4.0]);
        for (actual, expected) in actual.into_iter().zip([2.0 * log_two, -log_two, 0.0]) {
            assert!((actual - expected).abs() < 1e-15);
        }
        let mut hot = HotLoop::with_capacity(16);
        assert_eq!(
            bml_rmsnorm_with_epsilon(&mut hot, &[3.0, 4.0], &[99.0, 2.0, -0.5], 1, 2, 3.5),
            [1.5, -0.5]
        );
    }
}
