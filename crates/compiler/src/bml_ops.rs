//! Building blocks BML: operaciones vectoriales expresadas como RPN con Loop.

use crate::rpn::{RpnOp, RpnProgram};

// ===========================================================================
// bml_dot: dot product como fold de FMA con Loop
// ===========================================================================

/// Programa RPN para un dot product: y = Σ x[i] * w[i] sobre n_in elementos.
///
/// Usa `VarIndexed` para indexar x y w por el contador del loop.
/// El resultado queda en el tope de la pila al terminar.
///
/// Cuerpo del loop (5 ops):
/// ```text
/// Dup                  // duplicar contador (necesitamos 2: x y w)
/// VarIndexed(x_base)   // pop contador, push x[counter]
/// Swap                 // poner contador en tope
/// VarIndexed(w_base)   // pop contador, push w[counter]
/// FMul                 // x[i] * w[i]
/// FAdd                 // acc + product
/// ```
pub fn bml_dot_program(x_base: u32, w_base: u32, n_in: u32) -> RpnProgram {
    let mut program = RpnProgram::new();

    // Acumulador inicial = 0
    program.push(RpnOp::Zero);

    // Cuerpo del loop
    let body = vec![
        RpnOp::Dup,
        RpnOp::VarIndexed { base: x_base },
        RpnOp::Swap,
        RpnOp::VarIndexed { base: w_base },
        RpnOp::FMul,
        RpnOp::FAdd,
    ];

    let body_len = body.len() as u32;
    program.push(RpnOp::Loop {
        count: n_in,
        body_len,
    });
    for op in body {
        program.push(op);
    }

    program
}

// ===========================================================================
// bml_matmul: matmul como map de dot products
// ===========================================================================

/// Programa RPN para un matmul completo: y = x · W.
///
/// Para cada columna j (0..n_out), calcula el dot product de x con W[:, j].
/// El resultado de cada columna se almacena via StoreResult.
///
/// Cada columna es independiente → paralelizable entre nodos.
pub fn bml_matmul_program(
    x_base: u32,
    w_base: u32,
    n_in: u32,
    n_out: u32,
    output_slot: u32,
) -> RpnProgram {
    let mut program = RpnProgram::new();

    for j in 0..n_out {
        // Acumulador = 0
        program.push(RpnOp::Zero);

        // w_col_base = w_base + j * n_in (offset de la columna j)
        let w_col_base = w_base + j * n_in;

        // Cuerpo del loop (mismo que bml_dot)
        let body = vec![
            RpnOp::Dup,
            RpnOp::VarIndexed { base: x_base },
            RpnOp::Swap,
            RpnOp::VarIndexed { base: w_col_base },
            RpnOp::FMul,
            RpnOp::FAdd,
        ];

        let body_len = body.len() as u32;
        program.push(RpnOp::Loop {
            count: n_in,
            body_len,
        });
        for op in body {
            program.push(op);
        }

        // Almacenar resultado: y[j] = tope de pila
        program.push(RpnOp::Const(j)); // offset = j
        program.push(RpnOp::StoreResult { slot: output_slot });
    }

    program
}

// ===========================================================================
// bml_rmsnorm: y[i] = x[i] * (1/rms) * w[i] como fold
// ===========================================================================

/// Programa RPN para RMSNorm.
///
/// `recip_rms_const` es el ConstId del valor 1/rms precomputado.
pub fn bml_rmsnorm_program(
    x_base: u32,
    w_base: u32,
    n_embd: u32,
    recip_rms_const: u32,
    _output_slot: u32,
) -> RpnProgram {
    let mut program = RpnProgram::new();

    let body = vec![
        RpnOp::Dup,
        RpnOp::VarIndexed { base: x_base }, // x[i]
        RpnOp::Const(recip_rms_const),      // 1/rms
        RpnOp::FMul,                        // x[i] * (1/rms)
        RpnOp::Swap,                        // contador al tope
        RpnOp::VarIndexed { base: w_base }, // w[i]
        RpnOp::FMul,                        // x[i] * (1/rms) * w[i]
        RpnOp::FAdd,                        // acc + product
    ];

    let body_len = body.len() as u32;
    program.push(RpnOp::Zero);
    program.push(RpnOp::Loop {
        count: n_embd,
        body_len,
    });
    for op in body {
        program.push(op);
    }

    program
}

// ===========================================================================
// bml_swiglu: y = gate * sigmoid(gate) * up
// ===========================================================================

/// Standard SwiGLU, `y[i] = gate[i] * sigmoid(gate[i]) * up[i]`.
///
/// `gate_base` and `up_base` are absolute indices into `ResultBuffer`, whereas
/// `output_slot` identifies a slot. Each output is written at that slot's
/// element index; no reduction or temporary value remains on the stack.
/// Outputs must not overwrite gate/up values needed by a later iteration.
/// The caller supplies `log2(e)` at `const_log2e` and `1.0` at `const_one` in
/// its external constant context. The historical `const_1_7` argument is
/// obsolete and deliberately unused: standard SwiGLU has no such scale.
///
/// The loop reuses the signed bounded SwiGLU identity with gate/up declarations
/// of absolute value at most 16. Bytecode storage is independent of vector
/// width. Every activation calculation and fixed literal construction uses
/// BML; no additional external constants are required. As with the other BML
/// builders, this is a partial numerical contract: callers must check input
/// bounds and finite results, and it does not promise native IEEE range or
/// signed-zero behavior. No activation is clamped or evaluated natively.
pub fn bml_swiglu_program(
    gate_base: u32,
    up_base: u32,
    n_embd: u32,
    _const_1_7: u32,
    const_log2e: u32,
    const_one: u32,
    output_slot: u32,
) -> RpnProgram {
    let scalar = swiglu_scalar_with_external_constants(const_log2e, const_one);

    // The loop's element index stays below the scalar evaluation stack. Each
    // variable loads through a copy of that index; local Pick depths remain
    // valid because all other stack effects match the scalar program exactly.
    let mut body = Vec::with_capacity(scalar.ops.len() + 4);
    let mut depth = 0;
    for op in scalar.ops {
        match op {
            RpnOp::Var(id) => {
                let base = match id {
                    0 => gate_base,
                    1 => up_base,
                    _ => unreachable!("SwiGLU has two scalar inputs"),
                };
                body.push(RpnOp::Pick { depth });
                body.push(RpnOp::VarIndexed { base });
                depth += 1;
            }
            RpnOp::Const(_) => {
                body.push(op);
                depth += 1;
            }
            RpnOp::Bml => {
                body.push(op);
                depth -= 1;
            }
            RpnOp::Pick { .. } | RpnOp::Dup => {
                body.push(op);
                depth += 1;
            }
            RpnOp::Drop => {
                body.push(op);
                depth -= 1;
            }
            RpnOp::Swap => body.push(op),
            _ => unreachable!("SwiGLU lowering uses only BML and stack operations"),
        }
    }
    assert_eq!(depth, 1, "scalar lowering leaves exactly one result");
    body.push(RpnOp::Swap);
    body.push(RpnOp::StoreResult { slot: output_slot });
    let mut program = RpnProgram::new();
    program.push(RpnOp::Loop {
        count: n_embd,
        body_len: body.len() as u32,
    });
    program.ops.extend(body);
    program
}

// Rebind a bounded operation's configuration literals to existing BML terms.
// This retains the domain implementation's arithmetic while preserving this
// legacy API's external constant namespace (only log2(e) and one).
fn append_swiglu_bounded_operation(
    builder: &mut bml_domain::BMLTransformer,
    inputs: [bml_domain::NodeId; 2],
    config: bml_domain::transformer::BoundedOpConfig,
    literals: &[(f64, bml_domain::NodeId)],
) -> bml_domain::NodeId {
    use bml_domain::{BMLTransformer, NodeKind};
    let mut template = BMLTransformer::new();
    let x = template.var(0);
    let y = template.var(1);
    let root = template
        .operation_bounded(&[x, y], &[], &config)
        .expect("fixed valid bounded operation")[0];
    let soa = template.into_soa();
    let literal = |value: f64| {
        literals
            .iter()
            .find(|(candidate, _)| candidate.to_bits() == value.to_bits())
            .expect("every configured literal has an explicit BML construction")
            .1
    };
    let mut mapped = Vec::with_capacity(soa.len());
    for (index, kind) in soa.kinds.iter().enumerate() {
        let node = match kind {
            NodeKind::One => literal(1.0),
            NodeKind::Zero => literal(0.0),
            NodeKind::Var(id) => inputs[*id as usize],
            NodeKind::Const(id) => literal(soa.const_pool[*id as usize]),
            NodeKind::Bml => builder.bml(
                mapped[soa.lefts[index] as usize],
                mapped[soa.rights[index] as usize],
            ),
        };
        mapped.push(node);
    }
    mapped[root as usize]
}

fn swiglu_scalar_with_external_constants(const_log2e: u32, const_one: u32) -> RpnProgram {
    use bml_domain::{transformer::BoundedOpConfig, BMLTransformer};
    let mut builder = BMLTransformer::new();
    let one = builder.external_constant(const_one);
    let log2e = builder.external_constant(const_log2e);
    let gate = builder.var(0);
    let up = builder.var(1);

    // Exact dyadic terms for the fixed bound 16. Every intermediate is finite.
    let two = builder.bml(one, one);
    let four = builder.bml(two, one);
    let zero = builder.bml(one, four);
    let three = builder.bml(two, two);
    let eight = builder.bml(three, one);
    let five = builder.bml(three, eight);
    let six = builder.bml(three, four);
    let seven = builder.bml(three, two);
    let sixteen = builder.bml(four, one);
    let thirty_two = builder.bml(five, one);
    let sixty_four = builder.bml(six, one);
    let one_twenty_eight = builder.bml(seven, one);
    let two_fifty_six = builder.bml(eight, one);
    let nine = builder.bml(four, one_twenty_eight);
    let five_twelve = builder.bml(nine, one);
    let minus_one = builder.bml(one, eight);
    let half = builder.bml(minus_one, one);
    let minus_four = builder.bml(one, sixty_four);
    let one_sixteenth = builder.bml(minus_four, one);
    let minus_five = builder.bml(one, one_twenty_eight);
    let one_thirty_second = builder.bml(minus_five, one);
    let two_to_32 = builder.bml(thirty_two, one);
    let minus_sixteen = builder.bml(four, two_to_32);
    let mut literals = vec![
        (0.0, zero),
        (1.0, one),
        (2.0, two),
        (4.0, four),
        (16.0, sixteen),
        (32.0, thirty_two),
        (64.0, sixty_four),
        (128.0, one_twenty_eight),
        (256.0, two_fifty_six),
        (512.0, five_twelve),
        (0.5, half),
        (1.0 / 16.0, one_sixteenth),
        (1.0 / 32.0, one_thirty_second),
        (-16.0, minus_sixteen),
        (std::f64::consts::LOG2_E, log2e),
    ];
    let exponent = append_swiglu_bounded_operation(
        &mut builder,
        [minus_sixteen, log2e],
        BoundedOpConfig::product(16.0, 2.0).expect("fixed product bounds"),
        &literals,
    );
    let minimum_exp = builder.bml(exponent, one);
    literals.push(((-16.0_f64).exp(), minimum_exp));
    let root = append_swiglu_bounded_operation(
        &mut builder,
        [gate, up],
        BoundedOpConfig::swiglu(16.0, 16.0).expect("fixed SwiGLU bounds"),
        &literals,
    );
    crate::linearize(&builder.into_soa(), root)
}

// ===========================================================================
// bml_rope: rotación de un par (even, odd) con cos/sin precomputados
// ===========================================================================

/// Programa RPN para RoPE usando BML puro para la negación.
///
/// x_even' = x_even * cos - x_odd * sin
///
/// La negación de x_odd*sin se hace con BML puro:
/// neg(y) = bml(log2(0), exp2(y)) = 0 - y
pub fn bml_rope_program(
    x_even_base: u32,
    x_odd_base: u32,
    cos_const: u32,
    sin_const: u32,
    _output_slot: u32,
) -> RpnProgram {
    let mut program = RpnProgram::new();

    // x_even * cos
    program.push(RpnOp::Var(x_even_base));
    program.push(RpnOp::Const(cos_const));
    program.push(RpnOp::FMul);

    // x_odd * sin
    program.push(RpnOp::Var(x_odd_base));
    program.push(RpnOp::Const(sin_const));
    program.push(RpnOp::FMul);

    // neg(x_odd*sin) via BML puro:
    // exp2(x_odd*sin) = bml(x_odd*sin, 1)
    program.push(RpnOp::One);
    program.push(RpnOp::Bml);

    // log2(0) = -inf:
    // bml(1, 0) = inf
    program.push(RpnOp::One);
    program.push(RpnOp::Zero);
    program.push(RpnOp::Bml);
    // bml(inf, 1) = inf
    program.push(RpnOp::One);
    program.push(RpnOp::Bml);
    // bml(1, inf) = -inf
    program.push(RpnOp::One);
    program.push(RpnOp::Swap);
    program.push(RpnOp::Bml);

    // bml(-inf, exp2(x_odd*sin)) = 0 - x_odd*sin = -x_odd*sin
    program.push(RpnOp::Swap);
    program.push(RpnOp::Bml);

    // x_even*cos + (-x_odd*sin) = x_even*cos - x_odd*sin
    program.push(RpnOp::FAdd);

    program
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bml_dot_program_structure() {
        let prog = bml_dot_program(0, 100, 4);
        assert!(prog.len() > 6);
        assert!(prog
            .ops
            .iter()
            .any(|op| matches!(op, RpnOp::Loop { count: 4, .. })));
    }

    #[test]
    fn bml_matmul_program_structure() {
        let prog = bml_matmul_program(0, 100, 4, 3, 0);
        let n_stores = prog
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::StoreResult { .. }))
            .count();
        assert_eq!(n_stores, 3);
        let n_loops = prog
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::Loop { .. }))
            .count();
        assert_eq!(n_loops, 3);
    }

    #[test]
    fn bml_matmul_bytecode_size() {
        let prog = bml_matmul_program(0, 100, 2048, 2048, 0);
        let total_bytes: usize = prog
            .ops
            .iter()
            .map(|op| match op {
                RpnOp::One
                | RpnOp::Zero
                | RpnOp::Bml
                | RpnOp::Dup
                | RpnOp::FAdd
                | RpnOp::FMul
                | RpnOp::Drop
                | RpnOp::Swap => 1,
                RpnOp::Var(_)
                | RpnOp::Const(_)
                | RpnOp::VarIndexed { .. }
                | RpnOp::ContextVarIndexed { .. }
                | RpnOp::ContextConstIndexed { .. }
                | RpnOp::StoreResult { .. }
                | RpnOp::Pick { .. } => 5,
                RpnOp::Loop { .. } => 9,
            })
            .sum();
        println!(
            "Matmul 2048x2048: {} ops, {} bytes ({:.1} KB)",
            prog.len(),
            total_bytes,
            total_bytes as f64 / 1024.0
        );
        assert!(total_bytes < 80_000);
    }

    #[test]
    fn bml_rmsnorm_program_structure() {
        let prog = bml_rmsnorm_program(0, 100, 4, 200, 0);
        assert!(prog
            .ops
            .iter()
            .any(|op| matches!(op, RpnOp::Loop { count: 4, .. })));
    }

    #[test]
    fn bml_rope_uses_bml() {
        let prog = bml_rope_program(0, 1, 100, 101, 0);
        let n_bml = prog
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::Bml))
            .count();
        assert!(n_bml > 0, "RoPE debe usar Bml para negacion BML pura");
        let n_fmul = prog
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::FMul))
            .count();
        assert_eq!(n_fmul, 2);
    }
}
