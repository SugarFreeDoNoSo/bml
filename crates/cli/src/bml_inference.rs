//! Integración de building blocks BML con el runtime.
//!
//! Este módulo conecta los programas RPN generados por `bml_ops` con el
//! `HotLoop` del runtime para ejecutar el transformer usando BML nativo.
//!
//! # Flujo
//!
//! 1. Escribir `x` (hidden state) en un slot del ResultBuffer
//! 2. Escribir `W` (pesos) en otro slot del ResultBuffer
//! 3. Generar el programa RPN con `bml_matmul_program()` etc.
//! 4. Ejecutar via `HotLoop::execute_full()`
//! 5. Leer resultados del ResultBuffer

use bml_compiler::bml_ops;
use bml_compiler::gguf_compiler::InferenceCompiler;
use bml_compiler::rpn::RpnProgram;
use bml_compiler::sampler;
use bml_runtime::buffer::ResultBuffer;
use bml_runtime::hot_loop::HotLoop;
use bml_domain::EvalContext;

/// Ejecuta un matmul usando BML building blocks: y = x · W.
///
/// Escribe x y W en el ResultBuffer, genera el programa RPN con
/// `bml_matmul_program`, lo ejecuta via HotLoop, y retorna y.
pub fn bml_matmul(
    hot: &mut HotLoop,
    x: &[f64],
    weights: &[f32],
    weight_offset: u64,
    n_in: u32,
    n_out: u32,
) -> Vec<f64> {
    let slot_size = n_in.max(n_out) as usize;
    let mut buf = ResultBuffer::new(3, slot_size);

    // Slot 0: vector x (base=0)
    for (i, &v) in x.iter().enumerate().take(n_in as usize) {
        buf.write(0, i as u32, v);
    }

    // Slot 1: pesos W (base=slot_size)
    let w_base = slot_size as u32;
    let n_out_us = n_out as usize;
    for i in 0..n_in as usize {
        for j in 0..n_out_us {
            let idx = weight_offset as usize + i * n_out_us + j;
            if idx < weights.len() {
                buf.write(1, (i * n_out_us + j) as u32, weights[idx] as f64);
            }
        }
    }

    // Generar programa RPN: matmul con x en slot 0, W en slot 1
    let x_base = 0; // slot 0
    let w_base_rpn = w_base; // slot 1 (en índices absolutos)
    let program = bml_ops::bml_matmul_program(x_base, w_base_rpn, n_in, n_out, 2);

    // Ejecutar
    let ctx = EvalContext::new(&[], &[]);
    hot.execute_full(&program, &ctx, &mut buf);

    // Leer resultados del slot 2
    let mut y = vec![0.0_f64; n_out as usize];
    for j in 0..n_out as usize {
        y[j] = buf.read(2, j as u32);
    }
    y
}

/// Ejecuta RMSNorm usando BML: y[i] = x[i] * (1/rms) * w[i].
pub fn bml_rmsnorm(
    hot: &mut HotLoop,
    x: &[f64],
    weights: &[f32],
    weight_offset: u64,
    n_embd: usize,
) -> Vec<f64> {
    let eps = 1e-5_f64;
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

/// Ejecuta SwiGLU usando BML: y = gate * sigmoid(1.7 * gate) * up.
pub fn bml_swiglu(gate: &[f64], up: &[f64]) -> Vec<f64> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let sig = 1.0 / (1.0 + (-1.7 * g).exp());
            g * sig * u
        })
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

/// Forward pass completo usando BML building blocks.
///
/// Reemplaza `InferenceCompiler::forward_layer` con versiones BML.
pub fn forward_layer_bml(
    hot: &mut HotLoop,
    compiler: &InferenceCompiler,
    hidden: &mut Vec<f64>,
    layer: u32,
    pos: u32,
) {
    let n_embd = compiler.config().n_embd as usize;
    let prefix = format!("blk.{layer}");

    // === RMSNorm de atención ===
    let norm_name = format!("{prefix}.attn_norm.weight");
    let norm_offset = *compiler.weight_offsets().get(&norm_name).unwrap_or(&0) as u64;
    let normed = bml_rmsnorm(hot, hidden, compiler.weight_pool(), norm_offset, n_embd);
    *hidden = normed;

    // === Q, K, V projections via bml_matmul ===
    let (n_in, n_out) = *compiler.tensor_dims().get(&format!("{prefix}.attn_q.weight")).unwrap_or(&(n_embd, n_embd));
    let q_offset = *compiler.weight_offsets().get(&format!("{prefix}.attn_q.weight")).unwrap_or(&0) as u64;
    let q = bml_matmul(hot, hidden, compiler.weight_pool(), q_offset, n_in as u32, n_out as u32);

    let (n_in_k, n_out_k) = *compiler.tensor_dims().get(&format!("{prefix}.attn_k.weight")).unwrap_or(&(n_embd, n_embd));
    let k_offset = *compiler.weight_offsets().get(&format!("{prefix}.attn_k.weight")).unwrap_or(&0) as u64;
    let k = bml_matmul(hot, hidden, compiler.weight_pool(), k_offset, n_in_k as u32, n_out_k as u32);

    let (n_in_v, n_out_v) = *compiler.tensor_dims().get(&format!("{prefix}.attn_v.weight")).unwrap_or(&(n_embd, n_embd));
    let v_offset = *compiler.weight_offsets().get(&format!("{prefix}.attn_v.weight")).unwrap_or(&0) as u64;
    let v = bml_matmul(hot, hidden, compiler.weight_pool(), v_offset, n_in_v as u32, n_out_v as u32);

    let mut q_mut = q;
    let mut k_mut = k;

    // === RoPE ===
    let head_dim = compiler.head_dim() as usize;
    bml_rope(&mut q_mut, pos as usize, head_dim);
    bml_rope(&mut k_mut, pos as usize, head_dim);

    // === Attention: scaled dot-product ===
    let n_heads = compiler.config().n_heads as usize;
    let n_kv_heads = compiler.n_kv_heads() as usize;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let q_heads_per_kv = n_heads / n_kv_heads;

    let mut output = vec![0.0_f64; n_heads * head_dim];
    for h in 0..n_heads {
        let kv_h = h / q_heads_per_kv;
        let q_start = h * head_dim;
        let k_start = kv_h * head_dim;
        let v_start = kv_h * head_dim;

        let mut dot = 0.0_f64;
        for d in 0..head_dim {
            dot += q_mut.get(q_start + d).copied().unwrap_or(0.0)
                * k_mut.get(k_start + d).copied().unwrap_or(0.0);
        }
        let attn = (dot * scale).tanh().clamp(-10.0, 10.0);

        let o_start = h * head_dim;
        for d in 0..head_dim {
            output[o_start + d] = attn * v.get(v_start + d).copied().unwrap_or(0.0);
        }
    }

    // === Output projection ===
    let (n_in_o, n_out_o) = *compiler.tensor_dims().get(&format!("{prefix}.attn_output.weight")).unwrap_or(&(n_embd, n_embd));
    let o_offset = *compiler.weight_offsets().get(&format!("{prefix}.attn_output.weight")).unwrap_or(&0) as u64;
    let o_out = bml_matmul(hot, &output, compiler.weight_pool(), o_offset, n_in_o as u32, n_out_o as u32);

    // === Residual ===
    for i in 0..n_embd {
        hidden[i] += o_out.get(i).copied().unwrap_or(0.0);
    }

    // === MLP RMSNorm ===
    let mlp_norm_name = format!("{prefix}.ffn_norm.weight");
    let mlp_norm_offset = *compiler.weight_offsets().get(&mlp_norm_name).unwrap_or(&0) as u64;
    let normed = bml_rmsnorm(hot, hidden, compiler.weight_pool(), mlp_norm_offset, n_embd);
    *hidden = normed;

    // === MLP: gate, up, down ===
    let (n_in_g, n_out_g) = *compiler.tensor_dims().get(&format!("{prefix}.ffn_gate.weight")).unwrap_or(&(n_embd, n_embd));
    let g_offset = *compiler.weight_offsets().get(&format!("{prefix}.ffn_gate.weight")).unwrap_or(&0) as u64;
    let gate = bml_matmul(hot, hidden, compiler.weight_pool(), g_offset, n_in_g as u32, n_out_g as u32);

    let (n_in_u, n_out_u) = *compiler.tensor_dims().get(&format!("{prefix}.ffn_up.weight")).unwrap_or(&(n_embd, n_embd));
    let u_offset = *compiler.weight_offsets().get(&format!("{prefix}.ffn_up.weight")).unwrap_or(&0) as u64;
    let up = bml_matmul(hot, hidden, compiler.weight_pool(), u_offset, n_in_u as u32, n_out_u as u32);

    let swiglu_out = bml_swiglu(&gate, &up);

    let (n_in_d, n_out_d) = *compiler.tensor_dims().get(&format!("{prefix}.ffn_down.weight")).unwrap_or(&(n_embd, n_embd));
    let d_offset = *compiler.weight_offsets().get(&format!("{prefix}.ffn_down.weight")).unwrap_or(&0) as u64;
    let mlp_out = bml_matmul(hot, &swiglu_out, compiler.weight_pool(), d_offset, n_in_d as u32, n_out_d as u32);

    // === Residual ===
    for i in 0..n_embd.min(mlp_out.len()) {
        hidden[i] += mlp_out[i];
    }
}

/// Forward pass completo (todos los tokens) usando BML building blocks.
pub fn forward_bml(
    hot: &mut HotLoop,
    compiler: &InferenceCompiler,
    input_ids: &[u32],
) -> Vec<f64> {
    let n_embd = compiler.config().n_embd as usize;
    let vocab_size = compiler.vocab().len();

    if input_ids.is_empty() {
        return vec![0.0; vocab_size];
    }

    // Embedding lookup
    let mut hidden = vec![0.0_f64; n_embd];
    for &tid in input_ids {
        let emb = compiler.get_embedding(tid);
        for i in 0..emb.len().min(n_embd) {
            hidden[i] += emb[i];
        }
    }
    let scale = 1.0 / (input_ids.len() as f64).sqrt();
    for v in &mut hidden {
        *v *= scale;
    }

    // Forward layers
    for layer in 0..compiler.config().n_layers {
        let pos = (input_ids.len() - 1) as u32;
        forward_layer_bml(hot, compiler, &mut hidden, layer, pos);
    }

    // lm_head
    let emb_offset = *compiler.weight_offsets().get("token_embd.weight").unwrap_or(&0) as u64;
    let (emb_dim, vocab_sz) = *compiler.tensor_dims().get("token_embd.weight").unwrap_or(&(n_embd, vocab_size));
    let logits = bml_matmul(hot, &hidden, compiler.weight_pool(), emb_offset, emb_dim as u32, vocab_sz.min(vocab_size) as u32);

    logits
}
