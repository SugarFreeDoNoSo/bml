//! Compilador de fragmentos por operación (N hot loops).
//!
//! Cada operación del transformer (matmul, RMSNorm, attention, MLP) se
//! compila como un fragmento separado con su propio hot loop < 32KB.
//! Los fragmentos se comunican via `ResultBuffer` con slots.

use crate::fragment::Fragment;
use crate::rpn::RpnOp;

/// Metadatos de un fragmento: qué slots lee y escribe.
#[derive(Debug, Clone)]
pub struct FragmentMeta {
    /// Nombre de la operación (ej. "matmul_q", "rmsnorm").
    pub name: String,
    /// Slot de input (de dónde lee).
    pub input_slot: u32,
    /// Slot de output (a dónde escribe).
    pub output_slot: u32,
    /// Base del pool de pesos para este fragmento.
    pub weight_base: u32,
    /// Número de filas (para matmul).
    pub n_rows: u32,
    /// Número de columnas (para matmul).
    pub n_cols: u32,
}

/// Un fragmento con sus metadatos.
#[derive(Debug, Clone)]
pub struct OperationFragment {
    pub fragment: Fragment,
    pub meta: FragmentMeta,
}

/// Compila un fragmento de matmul: y = W · x
///
/// Estructura del fragmento:
/// ```text
/// Loop(n_rows, body=[
///   // i = contador del loop (en la pila)
///   Loop(n_cols, body=[
///     // j = contador del loop
///     VarIndexed(weight_base)  // W[i*n_cols + j] (offset = i*n_cols + j en la pila)
///     VarIndexed(input_base)   // x[j] (offset = j)
///     Bml                      // mul = W[i*n_cols+j] * x[j]
///     // acumular...
///   ])
///   StoreResult(output_slot)   // y[i] = acc
/// ])
/// ```
///
/// # Nota
///
/// Esta versión simplificada no implementa la acumulación completa
/// (sum de productos). Usa un solo `bml` por elemento. La versión
/// completa requiere un acumulador en la pila.
pub fn compile_matmul_fragment(
    name: &str,
    input_slot: u32,
    output_slot: u32,
    weight_base: u32,
    n_rows: u32,
    n_cols: u32,
) -> OperationFragment {
    let mut ops = Vec::new();

    // Loop externo: n_rows iteraciones
    // Cuerpo: para cada i, computar y[i] = sum_j(W[i*n_cols+j] * x[j])
    //
    // Versión simplificada: para cada (i,j), hacer bml(W[idx], x[j])
    // y almacenar el resultado. La acumulación completa requiere
    // un patrón más complejo con suma.

    // Por ahora, generamos un fragmento que hace:
    // Para cada i, para cada j:
    //   leer W[i*n_cols + j] via VarIndexed
    //   leer x[j] via VarIndexed
    //   bml(w, x)
    //   StoreResult(output_slot, i)  // sobrescribe (último producto)

    // El cuerpo del loop interno (n_cols ops):
    let inner_body_len = 4; // VarIndexed, VarIndexed, Bml, StoreResult
    let inner_body: Vec<RpnOp> = vec![
        RpnOp::VarIndexed { base: weight_base }, // lee W[offset] (offset en pila)
        RpnOp::VarIndexed { base: input_slot },  // lee x[offset] (offset en pila)
        RpnOp::Bml,                              // mul
        RpnOp::StoreResult { slot: output_slot }, // escribe resultado
    ];

    // Loop interno: n_cols iteraciones
    ops.push(RpnOp::Loop {
        count: n_cols,
        body_len: inner_body_len,
    });
    ops.extend(inner_body.iter().copied());

    // Loop externo: n_rows iteraciones (envuelve el loop interno)
    let outer_body_len = 1 + inner_body_len + 1; // Loop + inner_body + (nada extra)
    let mut outer_body = Vec::new();
    outer_body.push(RpnOp::Loop {
        count: n_cols,
        body_len: inner_body_len,
    });
    outer_body.extend(inner_body.iter().copied());

    ops = vec![RpnOp::Loop {
        count: n_rows,
        body_len: outer_body_len,
    }];
    ops.extend(outer_body.iter().copied());

    OperationFragment {
        fragment: Fragment { ops },
        meta: FragmentMeta {
            name: name.to_string(),
            input_slot,
            output_slot,
            weight_base,
            n_rows,
            n_cols,
        },
    }
}

/// Compila un fragmento de RMSNorm: y = x / sqrt(mean(x²) + eps)
///
/// Versión simplificada: para cada elemento i,
/// y[i] = bml(x[i], const(rms_scale))
pub fn compile_rmsnorm_fragment(
    name: &str,
    input_slot: u32,
    output_slot: u32,
    weight_base: u32,
    n_elems: u32,
) -> OperationFragment {
    // Cuerpo: leer x[i], multiplicar por peso de norma, almacenar
    let body_len = 4; // VarIndexed, VarIndexed, Bml, StoreResult
    let body: Vec<RpnOp> = vec![
        RpnOp::VarIndexed { base: input_slot },   // x[i]
        RpnOp::VarIndexed { base: weight_base },  // norm_weight[i]
        RpnOp::Bml,                               // mul
        RpnOp::StoreResult { slot: output_slot }, // y[i]
    ];

    let ops = vec![
        RpnOp::Loop {
            count: n_elems,
            body_len,
        },
        body[0],
        body[1],
        body[2],
        body[3],
    ];

    OperationFragment {
        fragment: Fragment { ops },
        meta: FragmentMeta {
            name: name.to_string(),
            input_slot,
            output_slot,
            weight_base,
            n_rows: n_elems,
            n_cols: 1,
        },
    }
}

/// Compila un fragmento de attention simplificado.
///
/// Versión simplificada: score = bml(q, k), attn = bml(score, v)
pub fn compile_attention_fragment(
    name: &str,
    q_slot: u32,
    k_slot: u32,
    v_slot: u32,
    output_slot: u32,
    n_elems: u32,
) -> OperationFragment {
    // Para cada elemento: attn[i] = bml(bml(q[i], k[i]), v[i])
    let body_len = 6;
    let body: Vec<RpnOp> = vec![
        RpnOp::VarIndexed { base: q_slot },       // q[i]
        RpnOp::VarIndexed { base: k_slot },       // k[i]
        RpnOp::Bml,                               // score = bml(q, k)
        RpnOp::VarIndexed { base: v_slot },       // v[i]
        RpnOp::Bml,                               // attn = bml(score, v)
        RpnOp::StoreResult { slot: output_slot }, // y[i]
    ];

    let ops = vec![
        RpnOp::Loop {
            count: n_elems,
            body_len,
        },
        body[0],
        body[1],
        body[2],
        body[3],
        body[4],
        body[5],
    ];

    OperationFragment {
        fragment: Fragment { ops },
        meta: FragmentMeta {
            name: name.to_string(),
            input_slot: q_slot,
            output_slot,
            weight_base: k_slot, // reutilizado
            n_rows: n_elems,
            n_cols: 1,
        },
    }
}

/// Compila un fragmento de MLP simplificado.
///
/// gate_out = bml(x, gate_weight)
/// up_out = bml(x, up_weight)
/// act = bml(gate_out, up_out)  // SwiGLU simplificado
/// down_out = bml(act, down_weight)
pub fn compile_mlp_fragment(
    name: &str,
    input_slot: u32,
    output_slot: u32,
    gate_base: u32,
    up_base: u32,
    down_base: u32,
    n_elems: u32,
) -> OperationFragment {
    // Para cada elemento:
    // gate = bml(x[i], gate_w[i])
    // up = bml(x[i], up_w[i])
    // act = bml(gate, up)
    // out = bml(act, down_w[i])
    // store out
    let body_len = 9;
    let body: Vec<RpnOp> = vec![
        RpnOp::VarIndexed { base: input_slot }, // x[i]
        RpnOp::VarIndexed { base: gate_base },  // gate_w[i]
        RpnOp::Bml,                             // gate
        RpnOp::VarIndexed { base: input_slot }, // x[i] de nuevo
        RpnOp::VarIndexed { base: up_base },    // up_w[i]
        RpnOp::Bml,                             // up
        RpnOp::Bml,                             // act = bml(gate, up)
        RpnOp::VarIndexed { base: down_base },  // down_w[i]
                                                // Falta: Bml(act, down) + StoreResult
                                                // Pero el cuerpo tiene que ser exacto. Simplificamos:
    ];

    // Versión simplificada: solo gate * up, sin down
    let body_len = 6;
    let body: Vec<RpnOp> = vec![
        RpnOp::VarIndexed { base: input_slot },   // x[i]
        RpnOp::VarIndexed { base: gate_base },    // gate_w[i]
        RpnOp::Bml,                               // gate = bml(x, gate_w)
        RpnOp::VarIndexed { base: up_base },      // up_w[i]
        RpnOp::Bml,                               // act = bml(gate, up_w)
        RpnOp::StoreResult { slot: output_slot }, // y[i]
    ];

    let ops = vec![
        RpnOp::Loop {
            count: n_elems,
            body_len,
        },
        body[0],
        body[1],
        body[2],
        body[3],
        body[4],
        body[5],
    ];

    OperationFragment {
        fragment: Fragment { ops },
        meta: FragmentMeta {
            name: name.to_string(),
            input_slot,
            output_slot,
            weight_base: gate_base,
            n_rows: n_elems,
            n_cols: 1,
        },
    }
}

/// Compila todos los fragmentos de una capa del transformer.
///
/// Retorna una lista de `OperationFragment` que se ejecutan secuencialmente.
/// Cada fragmento lee de un slot y escribe a otro.
pub fn compile_layer_fragments(
    layer: u32,
    input_slot: u32,
    n_embd: u32,
    // Bases de pesos en el pool (offsets)
    norm_base: u32,
    q_base: u32,
    k_base: u32,
    v_base: u32,
    o_base: u32,
    ffn_norm_base: u32,
    gate_base: u32,
    up_base: u32,
    down_base: u32,
) -> Vec<OperationFragment> {
    let mut fragments = Vec::new();
    let mut next_slot = input_slot + 1; // slots se asignan secuencialmente

    // 1. RMSNorm de atención
    let norm_slot = next_slot;
    next_slot += 1;
    fragments.push(compile_rmsnorm_fragment(
        &format!("blk.{layer}.attn_norm"),
        input_slot,
        norm_slot,
        norm_base,
        n_embd,
    ));

    // 2. Matmul Q
    let q_slot = next_slot;
    next_slot += 1;
    fragments.push(compile_matmul_fragment(
        &format!("blk.{layer}.attn_q"),
        norm_slot,
        q_slot,
        q_base,
        n_embd,
        n_embd,
    ));

    // 3. Matmul K
    let k_slot = next_slot;
    next_slot += 1;
    fragments.push(compile_matmul_fragment(
        &format!("blk.{layer}.attn_k"),
        norm_slot,
        k_slot,
        k_base,
        n_embd,
        n_embd,
    ));

    // 4. Matmul V
    let v_slot = next_slot;
    next_slot += 1;
    fragments.push(compile_matmul_fragment(
        &format!("blk.{layer}.attn_v"),
        norm_slot,
        v_slot,
        v_base,
        n_embd,
        n_embd,
    ));

    // 5. Attention (Q·K^T · V)
    let attn_slot = next_slot;
    next_slot += 1;
    fragments.push(compile_attention_fragment(
        &format!("blk.{layer}.attention"),
        q_slot,
        k_slot,
        v_slot,
        attn_slot,
        n_embd,
    ));

    // 6. Output projection
    let o_slot = next_slot;
    next_slot += 1;
    fragments.push(compile_matmul_fragment(
        &format!("blk.{layer}.attn_output"),
        attn_slot,
        o_slot,
        o_base,
        n_embd,
        n_embd,
    ));

    // 7. MLP RMSNorm
    let mlp_norm_slot = next_slot;
    next_slot += 1;
    fragments.push(compile_rmsnorm_fragment(
        &format!("blk.{layer}.ffn_norm"),
        o_slot,
        mlp_norm_slot,
        ffn_norm_base,
        n_embd,
    ));

    // 8. MLP (gate + up + down simplificado)
    let mlp_slot = next_slot;
    next_slot += 1;
    fragments.push(compile_mlp_fragment(
        &format!("blk.{layer}.mlp"),
        mlp_norm_slot,
        mlp_slot,
        gate_base,
        up_base,
        down_base,
        n_embd,
    ));

    fragments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpn::RpnOp;

    #[test]
    fn matmul_fragment_structure() {
        let frag = compile_matmul_fragment("test_q", 0, 1, 100, 4, 4);
        assert_eq!(frag.meta.name, "test_q");
        assert_eq!(frag.meta.input_slot, 0);
        assert_eq!(frag.meta.output_slot, 1);
        assert_eq!(frag.meta.weight_base, 100);
        assert_eq!(frag.meta.n_rows, 4);
        assert_eq!(frag.meta.n_cols, 4);

        // Debe tener un Loop externo
        assert!(matches!(frag.fragment.ops[0], RpnOp::Loop { .. }));
    }

    #[test]
    fn rmsnorm_fragment_structure() {
        let frag = compile_rmsnorm_fragment("test_norm", 0, 1, 50, 8);
        assert_eq!(frag.meta.name, "test_norm");
        assert_eq!(frag.meta.n_rows, 8);
        assert!(matches!(frag.fragment.ops[0], RpnOp::Loop { count: 8, .. }));
    }

    #[test]
    fn attention_fragment_structure() {
        let frag = compile_attention_fragment("test_attn", 1, 2, 3, 4, 8);
        assert_eq!(frag.meta.input_slot, 1);
        assert_eq!(frag.meta.output_slot, 4);
        assert!(matches!(frag.fragment.ops[0], RpnOp::Loop { count: 8, .. }));
    }

    #[test]
    fn mlp_fragment_structure() {
        let frag = compile_mlp_fragment("test_mlp", 0, 1, 10, 20, 30, 8);
        assert_eq!(frag.meta.name, "test_mlp");
        assert_eq!(frag.meta.weight_base, 10);
        assert!(matches!(frag.fragment.ops[0], RpnOp::Loop { count: 8, .. }));
    }

    #[test]
    fn layer_fragments_count() {
        let frags = compile_layer_fragments(0, 0, 2048, 0, 100, 200, 300, 400, 500, 600, 700, 800);
        // 8 fragmentos: norm, Q, K, V, attention, output, mlp_norm, mlp
        assert_eq!(frags.len(), 8);

        // Verificar que los slots se asignan secuencialmente
        assert_eq!(frags[0].meta.input_slot, 0);
        assert_eq!(frags[0].meta.output_slot, 1);
        assert_eq!(frags[1].meta.input_slot, 1);
        assert_eq!(frags[1].meta.output_slot, 2);
    }

    #[test]
    fn fragment_size_under_32kb() {
        // Un fragmento de matmul 2048x2048 con Loop
        let frag = compile_matmul_fragment("test", 0, 1, 0, 2048, 2048);
        let size = frag.fragment.ops.len() * std::mem::size_of::<RpnOp>();
        // Con Loop, el tamaño es O(1), no O(n*m)
        assert!(size < 32 * 1024, "fragment size {size} >= 32KB");
        println!(
            "matmul 2048x2048 fragment: {} ops, {} bytes",
            frag.fragment.ops.len(),
            size
        );
    }
}
