//! Benchmark de multiplicación de matrices: ndarray vs naive vs BML RPN.
//!
//! Compara tres implementaciones de multiplicación de matrices:
//! 1. **ndarray**: librería estándar de Rust para álgebra lineal.
//! 2. **naive**: triple loop en f64 puro.
//! 3. **bml_rpn**: cada operación `a*b + c` del producto se traduce a
//!    un programa RPN BML y se evalúa.
//!
//! # Tamaño
//!
//! El tamaño total de parámetros es N (elementos de A + elementos de B).
//! Para multiplicación A·B válida, A y B son cuadradas de lado `k` donde
//! `k*k*2 = N`, i.e. `k = sqrt(N/2)`.
//!
//! # Objetivo
//!
//! Determinar la complejidad Big O de cada variante y medir el overhead
//! de BML RPN vs ndarray en un caso real de álgebra lineal.

use bml_compiler::{RpnOp, RpnProgram};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

/// Valores de N (tamaño total de parámetros A+B) en progresión geométrica.
///
/// N = 2*k^2 donde k es el lado de las matrices cuadradas.
/// k = sqrt(N/2).
const N_VALUES: &[usize] = &[8, 18, 32, 50, 72, 98, 128, 162, 200, 242, 288, 338];

/// Construye un programa RPN que computa `a * b + acc` (FMA) usando BML.
///
/// En BML base 2, la multiplicación `a * b` se puede expresar como:
/// `a * b = 2^(log2(a) + log2(b))`
///
/// Y la suma `x + y`:
/// `x + y = 2^(log2(x) + log2(2^x + 2^y))` ... (complejo)
///
/// Como las fórmulas exactas de `+` y `*` en base 2 no están derivadas
/// aún (pendiente Hito 2), usamos una aproximación: el programa RPN
/// empuja los valores `a`, `b`, `acc` como constantes y aplica `bml`.
/// Esto NO es una multiplicación real — es un proxy que mide el overhead
/// del intérprete RPN con el mismo número de operaciones que tendría
/// una multiplicación real.
///
/// El programa es: `a, b, Bml, acc, Bml` (2 operaciones BML por elemento).
fn build_fma_program() -> RpnProgram {
    // Proxy: 2 operaciones BML por elemento del resultado.
    // En una implementación real, esto sería la traducción de `a*b + acc`
    // a la gramática BML.
    let mut program = RpnProgram::new();
    program.push(RpnOp::One); // a (placeholder)
    program.push(RpnOp::One); // b
    program.push(RpnOp::Bml); // bml(a, b) ~ a*b
    program.push(RpnOp::One); // acc
    program.push(RpnOp::Bml); // bml(a*b, acc) ~ a*b + acc
    program
}

/// Multiplicación de matrices con ndarray.
fn matmul_ndarray(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

/// Multiplicación de matrices naive (triple loop).
fn matmul_naive(a: &Array2<f64>, b: &Array2<f64>, c: &mut Array2<f64>) {
    let n = a.nrows();
    let m = b.ncols();
    let k = a.ncols();
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0;
            for l in 0..k {
                acc += a[(i, l)] * b[(l, j)];
            }
            c[(i, j)] = acc;
        }
    }
}

/// Multiplicación de matrices con BML RPN (proxy).
///
/// Cada elemento del resultado se computa evaluando el programa RPN
/// `k` veces (una por cada elemento de la fila de A y columna de B).
/// Esto mide el overhead del intérprete RPN en un caso realista.
fn matmul_bml_rpn(a: &Array2<f64>, b: &Array2<f64>, c: &mut Array2<f64>, program: &RpnProgram) {
    let n = a.nrows();
    let m = b.ncols();
    let k = a.ncols();
    for i in 0..n {
        for j in 0..m {
            // El programa RPN es un proxy; evaluamos k veces
            // para simular el costo de k multiplicaciones + acumulaciones.
            let mut acc = 0.0;
            for _ in 0..k {
                acc = program.evaluate(0.0);
            }
            c[(i, j)] = acc;
        }
    }
}

/// Genera matrices aleatorias A y B de tamaño k×k.
fn gen_matrices(k: usize) -> (Array2<f64>, Array2<f64>) {
    // Usamos valores deterministas (no aleatorios) para reproducibilidad.
    // Valores en [1, 2] para que log2 esté definido.
    let a = Array2::from_shape_fn((k, k), |(i, j)| 1.0 + ((i + j) % 100) as f64 / 100.0);
    let b = Array2::from_shape_fn((k, k), |(i, j)| 1.0 + ((i * 2 + j) % 100) as f64 / 100.0);
    (a, b)
}

/// Lado de la matriz cuadrada dado N (tamaño total de parámetros).
fn k_from_n(n: usize) -> usize {
    (n as f64 / 2.0).sqrt().round() as usize
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    let program = build_fma_program();

    for &n in N_VALUES {
        let k = k_from_n(n);
        if k < 1 {
            continue;
        }
        let (a, b) = gen_matrices(k);
        let mut c_mat = Array2::zeros((k, k));

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bencher, _| {
            bencher.iter(|| black_box(matmul_ndarray(black_box(&a), black_box(&b))))
        });

        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bencher, _| {
            bencher.iter(|| {
                matmul_naive(black_box(&a), black_box(&b), black_box(&mut c_mat));
                black_box(c_mat[(0, 0)])
            })
        });

        group.bench_with_input(BenchmarkId::new("bml_rpn", n), &n, |bencher, _| {
            bencher.iter(|| {
                matmul_bml_rpn(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c_mat),
                    black_box(&program),
                );
                black_box(c_mat[(0, 0)])
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
