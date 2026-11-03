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
    /// Input identifier. For matmul this is an absolute buffer element base;
    /// historical simplified helpers still use their original slot convention.
    pub input_slot: u32,
    /// Slot de output (a dónde escribe).
    pub output_slot: u32,
    /// Absolute weight element base in `ResultBuffer` for matmul.
    pub weight_base: u32,
    /// Número de filas (para matmul).
    pub n_rows: u32,
    /// Número de columnas (para matmul).
    pub n_cols: u32,
}

/// Un fragmento con sus metadatos.
#[derive(Debug, Clone)]
pub struct OperationFragment {
    /// Operaciones RPN que ejecuta el fragmento.
    pub fragment: Fragment,
    /// Slots, dimensiones y pesos asociados al fragmento.
    pub meta: FragmentMeta,
}

/// Explicit operand declarations for a signed BML matrix-vector product.
///
/// These are inclusive absolute bounds. The underlying finite-real template
/// accepts bounds in `[1/256, 256]` whose product is at most 480. Its running
/// accumulator must be below 256 before each step; other numerical failures
/// remain possible. Use [`CheckedMatmulFragment::validate_values`] for the actual
/// matrix/vector before executing its bytecode; bounds alone are not a proof of
/// evaluability or of floating-point accuracy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatmulBounds {
    /// Absolute bound for each input vector element.
    pub input: f64,
    /// Absolute bound for each matrix element.
    pub weight: f64,
}

impl Default for MatmulBounds {
    fn default() -> Self {
        Self {
            input: 16.0,
            weight: 16.0,
        }
    }
}

/// A matrix-vector fragment with explicit numerical declarations and owned
/// constants. It reads vector/weights from absolute `ResultBuffer` bases and
/// writes rows to the output slot, with the same layout as the legacy helper.
///
/// Bind [`Self::constants`] as the execution context's constant pool. The
/// preflight checks actual operands and every BML intermediate without writing
/// output. Run the returned fragment on the same values after preflight; changing
/// inputs, weights, bytecode or constants invalidates that check. Buffer capacity
/// and nonoverlapping output storage remain caller responsibilities.
#[derive(Debug, Clone)]
pub struct CheckedMatmulFragment {
    operation: OperationFragment,
    constants: Vec<f64>,
    bounds: MatmulBounds,
    config: bml_domain::transformer::BoundedOpConfig,
    validation_dag: bml_domain::NodeSoA,
    validation_root: u32,
}

impl CheckedMatmulFragment {
    /// Compiled bytecode and matrix/buffer metadata, bound to this preflight.
    pub fn operation(&self) -> &OperationFragment {
        &self.operation
    }

    /// Required constant pool for this compiled fragment, including its row stride.
    pub fn constants(&self) -> &[f64] {
        &self.constants
    }

    /// Configured signed operand bounds.
    pub fn bounds(&self) -> MatmulBounds {
        self.bounds
    }

    /// Check row-major weights and vector values before any output is written.
    ///
    /// This evaluates the same BML recurrence as the emitted code, checking every
    /// BML operation with the real-domain evaluator. It performs O(rows * cols)
    /// work, uses O(1) scratch relative to matrix size and reports the failing
    /// row/column. It neither synthesizes native products/sums nor replaces the
    /// execution result with a native dot product.
    pub fn validate_values(&self, inputs: &[f64], weights: &[f64]) -> Result<(), String> {
        use bml_domain::{symbolic::Basis, NodeKind};
        let rows = self.operation.meta.n_rows as usize;
        let cols = self.operation.meta.n_cols as usize;
        if inputs.len() != cols || rows.checked_mul(cols) != Some(weights.len()) {
            return Err("matmul operand shape mismatch".into());
        }
        for (column, &value) in inputs.iter().enumerate() {
            if !value.is_finite() || value.abs() > self.bounds.input {
                return Err(format!("matmul input bound at column {column}"));
            }
        }
        let dag = &self.validation_dag;
        let mut values = vec![0.0; dag.len()];
        for row in 0..rows {
            let mut accumulator = 0.0;
            for (column, &input) in inputs.iter().enumerate() {
                let operands = [accumulator, input, weights[row * cols + column]];
                self.config
                    .validate_values(&operands, &[])
                    .map_err(|error| format!("matmul row {row}, column {column}: {error}"))?;
                for (node, kind) in dag.kinds.iter().enumerate() {
                    values[node] = match *kind {
                        NodeKind::One => 1.0,
                        NodeKind::Zero => 0.0,
                        NodeKind::Var(id) => operands[id as usize],
                        NodeKind::Const(id) => dag.const_pool[id as usize],
                        NodeKind::Bml => Basis::Bml
                            .apply_real(
                                values[dag.lefts[node] as usize],
                                values[dag.rights[node] as usize],
                            )
                            .map_err(|error| {
                                format!(
                                    "matmul row {row}, column {column}, BML node {node}: {error}"
                                )
                            })?,
                    };
                }
                accumulator = values[self.validation_root as usize];
            }
        }
        Ok(())
    }
}

fn matmul_step(config: &bml_domain::transformer::BoundedOpConfig) -> (bml_domain::NodeSoA, u32) {
    let mut builder = bml_domain::BMLTransformer::new();
    let inputs = [builder.var(0), builder.var(1), builder.var(2)];
    let root = builder
        .operation_bounded(&inputs, &[], config)
        .expect("validated fixed-size matmul template")[0];
    (builder.into_soa(), root)
}

/// Compile a configurable signed BML matrix-vector fragment with owned constants.
///
/// `input_base` and `weight_base` are absolute element offsets in `ResultBuffer`,
/// while `output_slot` is a slot number. Weights are row-major. Dimensions and
/// absolute read addresses must fit u32; zero rows/columns are supported. An empty
/// row writes zero. The caller must reserve the output slot, keep it disjoint
/// from input/weight storage, bind the returned constants and preflight values.
///
/// Only address arithmetic uses native FAdd/FMul. Each data step is the bounded
/// BML expansion of `accumulator + input[column] * weight[row, column]`.
#[allow(clippy::too_many_arguments)]
pub fn compile_matmul_fragment_checked(
    name: &str,
    input_base: u32,
    output_slot: u32,
    weight_base: u32,
    n_rows: u32,
    n_cols: u32,
    bounds: MatmulBounds,
) -> Result<CheckedMatmulFragment, String> {
    let count = n_rows
        .checked_mul(n_cols)
        .ok_or("matmul dimensions overflow")?;
    if n_cols != 0 {
        input_base
            .checked_add(n_cols - 1)
            .ok_or("matmul input address overflow")?;
    }
    if count != 0 {
        weight_base
            .checked_add(count - 1)
            .ok_or("matmul weight address overflow")?;
    }
    let config = bml_domain::transformer::BoundedOpConfig::dot_step(bounds.input, bounds.weight)
        .map_err(|error| error.to_string())?;
    let (dag, root) = matmul_step(&config);
    let step = crate::linearize(&dag, root);
    let mut constants = vec![f64::from(n_cols)];
    constants.extend_from_slice(&dag.const_pool);
    let operation = emit_matmul(
        name,
        input_base,
        output_slot,
        weight_base,
        n_rows,
        n_cols,
        0,
        &step.ops,
        |id| vec![RpnOp::Const(id + 1)],
    );
    Ok(CheckedMatmulFragment {
        operation,
        constants,
        bounds,
        config,
        validation_dag: dag,
        validation_root: root,
    })
}

// A literal prefix is not required by the legacy API: its only context constant
// remains the caller's column count. These small closed terms produce the nine
// literals of the default profile exactly, with finite BML intermediates.
fn legacy_matmul_literals() -> Vec<Vec<RpnOp>> {
    let mut b = bml_domain::BMLTransformer::new();
    let one = b.one();
    let two = b.bml(one, one);
    let four = b.bml(two, one);
    let three = b.bml(two, two);
    let eight = b.bml(three, one);
    let five = b.bml(three, eight);
    let six = b.bml(three, four);
    let seven = b.bml(three, two);
    let c32 = b.bml(five, one);
    let c64 = b.bml(six, one);
    let c128 = b.bml(seven, one);
    let nine = b.bml(four, c128);
    let c256 = b.bml(eight, one);
    let c512 = b.bml(nine, one);
    let minus_four = b.bml(one, c64);
    let inv16 = b.bml(minus_four, one);
    let roots = [c64, two, c32, inv16, c32, inv16, c256, c512, c256];
    let dag = b.into_soa();
    roots
        .iter()
        .map(|&root| crate::linearize(&dag, root).ops)
        .collect()
}

/// Compile the historical matrix-vector API with its existing constant binding.
///
/// `input_slot` is historically named but denotes an **absolute input base** in
/// `ResultBuffer`, as does `weight_base`; `output_slot` denotes an output slot.
/// `n_cols_const_id` must bind exactly `n_cols as f64` in the caller's context.
/// No other context constants are required. The output must not overlap inputs
/// or weights. Each row is a sum of signed products, and empty rows write zero.
/// Both absolute read ranges must fit u32 and the allocated buffer.
///
/// This compatibility entry point uses [`MatmulBounds::default`] (absolute
/// operand bounds 16), and a finite accumulator below 256 before every step.
/// It cannot inspect runtime values or buffer capacity. Out-of-contract inputs
/// may yield NaN or inaccurate results under the legacy IEEE hot loop. For
/// explicit configurable bounds, checked addressing and finite-real preflight,
/// use [`compile_matmul_fragment_checked`]. This is not an all-f64 BLAS contract.
pub fn compile_matmul_fragment(
    name: &str,
    input_slot: u32,
    output_slot: u32,
    weight_base: u32,
    n_rows: u32,
    n_cols: u32,
    n_cols_const_id: u32,
) -> OperationFragment {
    let bounds = MatmulBounds::default();
    let config = bml_domain::transformer::BoundedOpConfig::dot_step(bounds.input, bounds.weight)
        .expect("fixed legacy matmul bounds");
    let (dag, root) = matmul_step(&config);
    let step = crate::linearize(&dag, root);
    let literals = legacy_matmul_literals();
    emit_matmul(
        name,
        input_slot,
        output_slot,
        weight_base,
        n_rows,
        n_cols,
        n_cols_const_id,
        &step.ops,
        |id| literals[id as usize].clone(),
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_matmul(
    name: &str,
    input_base: u32,
    output_slot: u32,
    weight_base: u32,
    n_rows: u32,
    n_cols: u32,
    n_cols_const_id: u32,
    step: &[RpnOp],
    mut constant: impl FnMut(u32) -> Vec<RpnOp>,
) -> OperationFragment {
    // On entry to each inner iteration: [outer_i, row_start, accumulator, j].
    // The step's temporary values stay above that frame. Replace its three
    // variables by picks/loads while preserving every BML operation and order.
    let mut inner = Vec::new();
    let mut height = 0u32;
    for &instruction in step {
        match instruction {
            RpnOp::Var(0) => {
                inner.push(RpnOp::Pick { depth: height + 1 });
                height += 1;
            }
            RpnOp::Var(1) => {
                inner.push(RpnOp::Pick { depth: height });
                inner.push(RpnOp::VarIndexed { base: input_base });
                height += 1;
            }
            RpnOp::Var(2) => {
                inner.push(RpnOp::Pick { depth: height + 2 });
                inner.push(RpnOp::Pick { depth: height + 1 });
                inner.push(RpnOp::FAdd);
                inner.push(RpnOp::VarIndexed { base: weight_base });
                height += 1;
            }
            RpnOp::Const(id) => {
                inner.extend(constant(id));
                height += 1;
            }
            other => {
                match other {
                    RpnOp::One | RpnOp::Zero | RpnOp::Pick { .. } | RpnOp::Dup => height += 1,
                    RpnOp::Bml | RpnOp::Drop => height -= 1,
                    RpnOp::Swap => {}
                    _ => unreachable!("bounded dot step instruction"),
                }
                inner.push(other);
            }
        }
    }
    debug_assert_eq!(height, 1);
    // Replace the old accumulator and consume j, retaining [i, row_start, acc].
    inner.extend([RpnOp::Swap, RpnOp::Drop, RpnOp::Swap, RpnOp::Drop]);
    let mut outer = vec![
        RpnOp::Dup,
        RpnOp::Const(n_cols_const_id),
        RpnOp::FMul,
        RpnOp::Zero,
        RpnOp::Loop {
            count: n_cols,
            body_len: inner.len() as u32,
        },
    ];
    outer.extend(inner);
    outer.extend([
        RpnOp::Pick { depth: 2 },
        RpnOp::StoreResult { slot: output_slot },
        RpnOp::Drop,
        RpnOp::Drop,
    ]);
    let mut ops = vec![RpnOp::Loop {
        count: n_rows,
        body_len: outer.len() as u32,
    }];
    ops.extend(outer);
    OperationFragment {
        fragment: Fragment { ops },
        meta: FragmentMeta {
            name: name.to_string(),
            input_slot: input_base,
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
/// La proyección down todavía no se emite; su base se reserva en la API.
pub fn compile_mlp_fragment(
    name: &str,
    input_slot: u32,
    output_slot: u32,
    gate_base: u32,
    up_base: u32,
    _down_base: u32,
    n_elems: u32,
) -> OperationFragment {
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
/// Historical structural helper: RMSNorm/attention/MLP remain simplified and
/// its slot numbers are not converted to absolute buffer bases. This is not an
/// executable full-layer contract; use explicit per-operation layouts instead.
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
    n_cols_const_id: u32,
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
        n_cols_const_id,
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
        n_cols_const_id,
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
        n_cols_const_id,
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
        n_cols_const_id,
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
        let frag = compile_matmul_fragment("test_q", 0, 1, 100, 4, 4, 0);
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
        let frags =
            compile_layer_fragments(0, 0, 2048, 0, 100, 200, 300, 400, 500, 600, 700, 800, 0);
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
        let frag = compile_matmul_fragment("test", 0, 1, 0, 2048, 2048, 0);
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
