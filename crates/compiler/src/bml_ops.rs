// Building blocks BML: operaciones vectoriales expresadas como RPN con Loop.

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
    program.push(RpnOp::Loop { count: n_in, body_len });
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
        program.push(RpnOp::Loop { count: n_in, body_len });
        for op in body {
            program.push(op);
        }

        // Almacenar resultado: y[j] = tope de pila
        program.push(RpnOp::Const(j));  // offset = j
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
        RpnOp::VarIndexed { base: x_base },  // x[i]
        RpnOp::Const(recip_rms_const),        // 1/rms
        RpnOp::FMul,                          // x[i] * (1/rms)
        RpnOp::Swap,                          // contador al tope
        RpnOp::VarIndexed { base: w_base },    // w[i]
        RpnOp::FMul,                          // x[i] * (1/rms) * w[i]
        RpnOp::FAdd,                          // acc + product
    ];

    let body_len = body.len() as u32;
    program.push(RpnOp::Zero);
    program.push(RpnOp::Loop { count: n_embd, body_len });
    for op in body {
        program.push(op);
    }

    program
}

// ===========================================================================
// bml_swiglu: y = gate * sigmoid(1.7 * gate) * up
// ===========================================================================

/// Programa RPN para SwiGLU simplificado.
///
/// sigmoid(x) = 1 / (1 + exp(-x))
/// Para el RPN, precomputamos const_1_7 como Const.
pub fn bml_swiglu_program(
    gate_base: u32,
    up_base: u32,
    n_embd: u32,
    const_1_7: u32,
    _const_log2e: u32,
    _const_one: u32,
    _output_slot: u32,
) -> RpnProgram {
    let mut program = RpnProgram::new();

    // Simplificado: y[i] = gate[i] * up[i] (sin sigmoid, demo del patrón)
    // El sigmoid completo requiere exp que es Bml(x * log2(e), 1)
    let body = vec![
        RpnOp::Dup,
        RpnOp::VarIndexed { base: gate_base },  // gate[i]
        RpnOp::Swap,
        RpnOp::VarIndexed { base: up_base },     // up[i]
        RpnOp::FMul,                              // gate[i] * up[i]
        RpnOp::FAdd,                              // acc + product
    ];

    let body_len = body.len() as u32;
    program.push(RpnOp::Zero);
    program.push(RpnOp::Loop { count: n_embd, body_len });
    for op in body {
        program.push(op);
    }

    program
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
        assert!(prog.ops.iter().any(|op| matches!(op, RpnOp::Loop { count: 4, .. })));
    }

    #[test]
    fn bml_matmul_program_structure() {
        let prog = bml_matmul_program(0, 100, 4, 3, 0);
        let n_stores = prog.ops.iter().filter(|op| matches!(op, RpnOp::StoreResult { .. })).count();
        assert_eq!(n_stores, 3);
        let n_loops = prog.ops.iter().filter(|op| matches!(op, RpnOp::Loop { .. })).count();
        assert_eq!(n_loops, 3);
    }

    #[test]
    fn bml_matmul_bytecode_size() {
        let prog = bml_matmul_program(0, 100, 2048, 2048, 0);
        let total_bytes: usize = prog.ops.iter().map(|op| match op {
            RpnOp::One | RpnOp::Zero | RpnOp::Bml | RpnOp::Dup
            | RpnOp::FAdd | RpnOp::FMul | RpnOp::Drop | RpnOp::Swap => 1,
            RpnOp::Var(_) | RpnOp::Const(_) | RpnOp::VarIndexed { .. }
            | RpnOp::StoreResult { .. } | RpnOp::Pick { .. } => 5,
            RpnOp::Loop { .. } => 9,
        }).sum();
        println!("Matmul 2048x2048: {} ops, {} bytes ({:.1} KB)",
            prog.len(), total_bytes, total_bytes as f64 / 1024.0);
        assert!(total_bytes < 80_000);
    }

    #[test]
    fn bml_rmsnorm_program_structure() {
        let prog = bml_rmsnorm_program(0, 100, 4, 200, 0);
        assert!(prog.ops.iter().any(|op| matches!(op, RpnOp::Loop { count: 4, .. })));
    }

    #[test]
    fn bml_rope_uses_bml() {
        let prog = bml_rope_program(0, 1, 100, 101, 0);
        let n_bml = prog.ops.iter().filter(|op| matches!(op, RpnOp::Bml)).count();
        assert!(n_bml > 0, "RoPE debe usar Bml para negacion BML pura");
        let n_fmul = prog.ops.iter().filter(|op| matches!(op, RpnOp::FMul)).count();
        assert_eq!(n_fmul, 2);
    }
}
