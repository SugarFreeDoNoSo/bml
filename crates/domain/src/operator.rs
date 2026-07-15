//! Operador BML fundamental (Binary-Minus-Log, base 2, `f64`).
//!
//! `bml(x, y) = 2^x - log2(y)`
//!
//! BML es el análogo de EML (Exp-Minus-Log, `eml(x, y) = exp(x) - ln(y)`)
//! reescrito en **base 2**. La base 2 se alinea con el formato IEEE 754 de
//! `f64` (cuyo exponente nativo es base 2), permitiendo usar `exp2`/`log2`
//! nativos de la FPU en lugar de `exp`/`ln`. Esto preserva la completitud
//! funcional del operador (análogo continuo del NAND) y todas las
//! identidades del paper, adaptadas a base 2.
//!
//! # Propiedades
//!
//! - Es un *magma no asociativo*: el orden de los operandos es estrictamente
//!   inmutable (`bml(a, b) != bml(b, a)` en general).
//! - Tiene completitud funcional: junto con la constante 1, genera todo el
//!   repertorio de una calculadora científica (en base 2, la constante
//!   fundamental es `2 = bml(1, 1)`).
//! - Opera sobre el dominio real en esta implementación; la extensión al
//!   dominio complejo (rama principal) se introduce cuando el transformer
//!   la requiera (constantes como `i` y `pi` via `log2(-1)`).
//!
//! # Identidades (base 2)
//!
//! - `2 = bml(1, 1)`  (análogo de `e = eml(1, 1)`)
//! - `2^x = bml(x, 1)` (análogo de `exp(x) = eml(x, 1)`)
//! - `log2(x) = bml(1, bml(bml(1, x), 1))` (análogo de `ln`)
//!
//! # Notas de implementación
//!
//! Se usan `f64::exp2` y `f64::log2` nativos (instrucciones FPU en x86/ARM).
//! La optimización bit-twiddling sobre el formato IEEE 754 se evaluará en
//! el Hito 2 si el profiling indica que `exp2`/`log2` son el cuello de
//! botella. Por ahora priorizamos correctitud: todas las identidades del
//! paper deben verificarse exactamente.

/// El operador BML en base 2: `2^x - log2(y)`.
///
/// # Semántica
///
/// - `2^x` se calcula con `f64::exp2` (FPU nativa).
/// - `log2(y)` se calcula con `f64::log2`. Para `y == 0` se retorna
///   `f64::NEG_INFINITY` (convención `log2(0) = -inf` del paper).
///   Para `y < 0` o `y = NaN` se retorna `f64::NAN` (dominio complejo).
/// - El orden de los operandos importa: `bml(a, b) != bml(b, a)` en general.
#[inline]
pub fn bml_base_op(x: f64, y: f64) -> f64 {
    if y < 0.0 || y.is_nan() {
        // log2(y) no está definido en los reales para y < 0.
        // El operador BML opera sobre C (rama principal); aquí reflejamos
        // la imposibilidad de evaluar en los reales como NAN.
        return f64::NAN;
    }
    // y == 0: log2(0) = -inf (convención del paper, Supplementary Information Sect. 2.5)
    let log2_y = if y == 0.0 {
        f64::NEG_INFINITY
    } else {
        y.log2()
    };
    let exp_x = if x.is_nan() { f64::NAN } else { x.exp2() };

    exp_x - log2_y
}

/// Constante distinguida del cálculo BML.
///
/// El paper fija el valor `1` como terminal. Se necesita para neutralizar
/// el término logarítmico del operador via `log2(1) = 0`.
pub const ONE: f64 = 1.0;

// ===========================================================================
// Pruebas del operador
// ===========================================================================

/// `2 = bml(1, 1)`: la constante fundamental en base 2.
#[cfg(test)]
#[test]
fn two_equals_bml_one_one() {
    let val = bml_base_op(ONE, ONE);
    assert!((val - 2.0).abs() < 1e-12, "bml(1,1) = {val}, expected 2");
}

/// `2^x = bml(x, 1)`.
#[cfg(test)]
#[test]
fn exp2_equals_bml_x_one() {
    for &x in &[0.5_f64, 1.0, 2.0, 1.5, 10.0] {
        let lhs = bml_base_op(x, ONE);
        let rhs = x.exp2();
        assert!((lhs - rhs).abs() < 1e-12, "2^{x} mismatch: {lhs} vs {rhs}");
    }
}

/// `log2(x) = bml(1, bml(bml(1, x), 1))`.
///
/// Verificación:
/// - `bml(1, x) = 2 - log2(x)`
/// - `bml(bml(1, x), 1) = 2^(2 - log2(x)) = 4/x`
/// - `bml(1, 4/x) = 2 - log2(4/x) = 2 - (2 - log2(x)) = log2(x)`
#[cfg(test)]
#[test]
fn log2_identity() {
    for &x in &[0.5_f64, 1.0, 2.0, 1.5, 10.0, 100.0, 1024.0] {
        let inner = bml_base_op(bml_base_op(ONE, x), ONE);
        let approx = bml_base_op(ONE, inner);
        let expected = x.log2();
        assert!(
            (approx - expected).abs() < 1e-9,
            "log2({x}) mismatch: {approx} vs {expected}"
        );
    }
}

/// El operador NO es conmutativo: `bml(a, b) != bml(b, a)` para `a != b`.
#[cfg(test)]
#[test]
fn non_commutative() {
    let a = 2.0_f64;
    let b = 3.0;
    assert!((bml_base_op(a, b) - bml_base_op(b, a)).abs() > 1e-9);
}

/// Casos límite: el operador no pániquea con 0, 1, NaN, Inf.
#[cfg(test)]
#[test]
fn edge_cases_no_panic() {
    let _ = bml_base_op(0.0, 1.0);
    let _ = bml_base_op(1.0, 0.0); // y=0 -> log2(0) = -inf, bml = 2 - (-inf) = inf
    let _ = bml_base_op(f64::NAN, 1.0);
    let _ = bml_base_op(1.0, f64::NAN);
    let _ = bml_base_op(f64::INFINITY, 1.0);
    let _ = bml_base_op(1.0, f64::INFINITY);
    let _ = bml_base_op(f64::NEG_INFINITY, 1.0);
    // y negativo -> NAN (log2 no definido en reales)
    assert!(bml_base_op(1.0, -1.0).is_nan());
    // y = 0 -> log2(0) = -inf, bml(1, 0) = 2 - (-inf) = inf
    assert!(bml_base_op(1.0, 0.0).is_infinite());
}
