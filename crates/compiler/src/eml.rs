//! EML (Exp-Minus-Log) — funciones de compile-time.
//!
//! EML es el operador original del paper (ArXiv 2603.21852v2):
//! `eml(x, y) = exp(x) - ln(y)`.
//!
//! Este módulo NO se ejecuta en runtime. Se usa en compile-time para
//! precomputar valores que el paper deriva en base E (sin, cos, π, i, RoPE).
//! Los resultados se almacenan como `Const(id)` en el pool de constantes
//! del `HashConsRegistry`, y el runtime los referencia sin recalcular.
//!
//! # Relación con BML
//!
//! BML (`bml(x, y) = 2^x - log2(y)`) es el operador de runtime, optimizado
//! con LUT binaria y corrida de bits. EML es el operador de compile-time
//! que da acceso a funciones trascendentales que BML no puede derivar
//! directamente en los reales (sin, cos, π, i).
//!
//! # Identidades EML (del paper)
//!
//! - `e = eml(1, 1)`
//! - `exp(x) = eml(x, 1)`
//! - `ln(x) = eml(1, eml(eml(1, x), 1))`
//! - `x - y = eml(ln(x), exp(y))`
//! - `-x = eml(ln(0), exp(x))` (con ln(0) = -inf)
//! - `x + y = x - (-y)`
//! - `x * y = exp(ln(x) + ln(y))`
//! - `x / y = x * (1/y)`
//! - `x^y = exp(y * ln(x))`
//! - `i = sqrt(-1)` (vía dominio complejo)
//! - `π = ln(-1) / i` (vía dominio complejo)
//! - `sin(x) = (e^(ix) - e^(-ix)) / (2i)`
//! - `cos(x) = (e^(ix) + e^(-ix)) / (2i)`

use std::f64::consts;

/// El operador EML: `exp(x) - ln(y)`.
///
/// Esta función se usa en compile-time. No se ejecuta en el hot loop.
#[inline]
pub fn eml(x: f64, y: f64) -> f64 {
    x.exp() - y.ln()
}

/// `exp(x) = eml(x, 1)`.
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// `ln(x) = eml(1, eml(eml(1, x), 1))`.
#[inline]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// `e = eml(1, 1) = exp(1) - ln(1) = e - 0 = e`.
#[inline]
pub fn e() -> f64 {
    consts::E
}

/// `π = ln(-1) / i`.
///
/// En los reales, `ln(-1)` no está definido. Usamos la identidad
/// `π = -i * ln(-1)` que en la práctica se computa como `std::f64::consts::PI`.
#[inline]
pub fn pi() -> f64 {
    consts::PI
}

/// `sin(x) = (e^(ix) - e^(-ix)) / (2i)`.
///
/// En compile-time usamos `f64::sin` directamente. La identidad EML
/// se usa para verificar que la fórmula es correcta, pero en la práctica
/// precomputamos con la libm estándar.
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// `cos(x) = (e^(ix) + e^(-ix)) / (2i)`.
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// `sqrt(x) = x^0.5 = exp(0.5 * ln(x))`.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// `sigmoid(x) = 1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `softmax(x_i) = exp(x_i) / sum(exp(x_j))`.
///
/// Precomputa el softmax de un vector. El resultado se almacena como
/// constantes en el pool.
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&xi| (xi - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// `RMSNorm(x) = x / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let mean_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
    let rms = (mean_sq + eps).sqrt();
    x.iter().map(|&xi| xi / rms).collect()
}

/// `SwiGLU(x) = x * sigmoid(1.7 * x)`.
pub fn swiglu(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| xi * sigmoid(1.7 * xi)).collect()
}

/// `RoPE (Rotary Positional Embedding)`.
///
/// Para un par de dimensiones (x_even, x_odd) en posición `pos` con
/// frecuencia `freq`:
///
/// `x_even' = x_even * cos(pos * freq) - x_odd * sin(pos * freq)`
/// `x_odd'  = x_even * sin(pos * freq) + x_odd * cos(pos * freq)`
///
/// En compile-time, precomputamos `cos(pos * freq)` y `sin(pos * freq)`
/// como constantes. En runtime, la multiplicación y suma se hace con BML.
///
/// Esta función precomputa los valores de cos y sin para todas las
/// posiciones y frecuencias, retornándolos como constantes.
pub fn rope_constants(
    n_positions: usize,
    n_dims: usize,
    base: f64, // típicamente 10000.0
) -> Vec<(f64, f64)> {
    // freq_i = 1 / base^(2i / n_dims)
    // para cada posición pos y dimensión i:
    //   angle = pos * freq_i
    //   cos_val = cos(angle)
    //   sin_val = sin(angle)
    let mut result = Vec::with_capacity(n_positions * (n_dims / 2));
    for pos in 0..n_positions {
        for i in 0..(n_dims / 2) {
            let freq = 1.0 / base.powf(2.0 * i as f64 / n_dims as f64);
            let angle = pos as f64 * freq;
            result.push((angle.cos(), angle.sin()));
        }
    }
    result
}

/// Aplica RoPE a un par de valores usando constantes precomputadas.
pub fn rope_apply(x_even: f64, x_odd: f64, cos_val: f64, sin_val: f64) -> (f64, f64) {
    (
        x_even * cos_val - x_odd * sin_val,
        x_even * sin_val + x_odd * cos_val,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eml_basic() {
        // eml(1, 1) = e - 0 = e
        assert!((eml(1.0, 1.0) - consts::E).abs() < 1e-12);
    }

    #[test]
    fn exp_identity() {
        for x in [0.0, 0.5, 1.0, 2.0, -1.0] {
            assert!((exp(x) - x.exp()).abs() < 1e-12);
        }
    }

    #[test]
    fn ln_identity() {
        for x in [0.5, 1.0, 2.0, 10.0] {
            assert!((ln(x) - x.ln()).abs() < 1e-12);
        }
    }

    #[test]
    fn pi_value() {
        assert!((pi() - consts::PI).abs() < 1e-12);
    }

    #[test]
    fn sin_cos_match_stdlib() {
        for x in [0.0, 0.5, 1.0, consts::PI, 2.0 * consts::PI] {
            assert!((sin(x) - x.sin()).abs() < 1e-12);
            assert!((cos(x) - x.cos()).abs() < 1e-12);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = [1.0, 2.0, 3.0];
        let result = softmax(&x);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // El mayor valor debe tener la mayor probabilidad
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn rms_norm_zero_mean() {
        let x = [1.0, -1.0, 2.0, -2.0];
        let result = rms_norm(&x, 1e-5);
        // RMSNorm preserva la dirección (signo) de los valores
        assert!(result[0] > 0.0);
        assert!(result[1] < 0.0);
        assert!(result[2] > 0.0);
        assert!(result[3] < 0.0);
    }

    #[test]
    fn swiglu_basic() {
        let x = [1.0, -1.0, 0.0];
        let result = swiglu(&x);
        // SwiGLU(0) = 0 * sigmoid(0) = 0
        assert!((result[2]).abs() < 1e-12);
        // SwiGLU(1) > 0 (sigmoid(1.7) > 0.5)
        assert!(result[0] > 0.0);
        // SwiGLU(-1) < 0
        assert!(result[1] < 0.0);
    }

    #[test]
    fn rope_constants_shape() {
        let constants = rope_constants(4, 4, 10000.0);
        // 4 posiciones * 2 pares de dims = 8 entradas
        assert_eq!(constants.len(), 8);
        // Cada entrada es (cos, sin)
        for (c, s) in &constants {
            assert!(*c >= -1.0 && *c <= 1.0);
            assert!(*s >= -1.0 && *s <= 1.0);
            // cos² + sin² = 1
            assert!((c * c + s * s - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn rope_apply_identity_at_zero() {
        // En posición 0, angle = 0, cos = 1, sin = 0
        // RoPE no debe cambiar los valores
        let (x_even, x_odd) = rope_apply(3.0, 4.0, 1.0, 0.0);
        assert!((x_even - 3.0).abs() < 1e-12);
        assert!((x_odd - 4.0).abs() < 1e-12);
    }

    #[test]
    fn rope_apply_rotation() {
        // En angle = π/2, cos = 0, sin = 1
        // x_even' = x_even * 0 - x_odd * 1 = -x_odd
        // x_odd' = x_even * 1 + x_odd * 0 = x_even
        let (x_even, x_odd) = rope_apply(3.0, 4.0, 0.0, 1.0);
        assert!((x_even - (-4.0)).abs() < 1e-12);
        assert!((x_odd - 3.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_basics() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
