//! Micro-benchmarks de operaciones individuales (tarea 4.1 y 4.3).
//!
//! - 4.1: costo de una operación BML (`2^x - log2(y)`) vs FMA (`a*b + c`)
//!        vs `exp2` vs `log2` individuales.
//! - 4.3: costo del hot loop BML con programas de distintos tamaños
//!        (10, 100, 1K, 10K, 100K ops).
//!
//! Las tareas 4.2 (matmul BML RPN vs naive vs ndarray) y 4.4 (efecto del
//! Hash Consing con repetición estructural) ya están cubiertas en
//! `crates/compiler/benches/matrix_mul.rs` y `fma_vs_bml.rs` respectivamente.

use bml_compiler::{linearize, HashConsRegistry};
use bml_domain::bml as bml_op;
use bml_runtime::Runtime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// 4.1 — Costo de una operación individual: BML vs FMA vs exp2 vs log2.
fn bench_op_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("op_cost");

    group.bench_function("bml_2x_log2y", |b| {
        b.iter(|| {
            let x = black_box(1.5);
            let y = black_box(2.0);
            black_box(bml_op(x, y))
        })
    });

    group.bench_function("fma", |b| {
        b.iter(|| {
            let a = black_box(1.5);
            let b = black_box(2.0);
            let c = black_box(0.5);
            black_box(a * b + c)
        })
    });

    group.bench_function("exp2", |b| {
        b.iter(|| black_box(black_box(1.5_f64).exp2()))
    });

    group.bench_function("log2", |b| {
        b.iter(|| black_box(black_box(2.0_f64).log2()))
    });

    group.bench_function("bml_inline_exp2_log2", |b| {
        // Equivalente expandido de bml: muestra el costo sin overhead de función.
        b.iter(|| {
            let x = black_box(1.5_f64);
            let y = black_box(2.0_f64);
            black_box(x.exp2() - y.log2())
        })
    });

    group.finish();
}

/// Construye un programa BML de N operaciones.
fn build_program(n_ops: usize) -> bml_compiler::RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one);
    let mut node = two;
    let iterations = n_ops / 3;
    for _ in 0..iterations {
        node = reg.bml(node, two);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

/// 4.3 — Costo del hot loop con programas de distintos tamaños.
fn bench_hot_loop_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_loop_by_size");
    group.sample_size(50);

    for &n_ops in &[10usize, 100, 1_000, 10_000, 100_000] {
        let program = build_program(n_ops);
        let mut runtime = Runtime::new(8192, 16);

        // Warmup
        for _ in 0..5 {
            runtime.execute(&program, 0.0);
        }

        group.bench_with_input(BenchmarkId::new("ops", n_ops), &n_ops, |b, _| {
            b.iter(|| black_box(runtime.execute(black_box(&program), black_box(0.0))))
        });
    }

    group.finish();
}

/// 4.3b — Tasa de ops/seg del hot loop por tamaño (derivado de bench_hot_loop_sizes).
/// Útil para detectar overhead constante vs escalado lineal.
fn bench_ops_per_sec_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_per_sec_by_size");
    group.sample_size(50);

    for &n_ops in &[1_000usize, 10_000, 100_000] {
        let program = build_program(n_ops);
        let mut runtime = Runtime::new(8192, 16);

        for _ in 0..5 {
            runtime.execute(&program, 0.0);
        }

        group.bench_with_input(BenchmarkId::new("size", n_ops), &n_ops, |b, _| {
            b.iter(|| {
                // Ejecuta el programa entero — el tiempo total incluye N ops.
                black_box(runtime.execute(black_box(&program), black_box(0.0)))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_op_cost,
    bench_hot_loop_sizes,
    bench_ops_per_sec_by_size
);
criterion_main!(benches);
