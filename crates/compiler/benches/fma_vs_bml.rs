//! Benchmark comparativo: FMA tradicional vs DAG BML deduplicado.
//!
//! Compara el costo de evaluar una fórmula compleja con operaciones
//! repetidas usando:
//! 1. FMA tradicional (fórmula directa en f64).
//! 2. DAG BML con Hash Consing (sub-árboles repetidos se evalúan una sola vez).
//! 3. DAG BML sin Hash Consing (cada sub-árbol se recrea).
//!
//! # Objetivo
//!
//! Determinar la complejidad Big O de cada variante mediante una progresión
//! geométrica de N (10, 100, ..., 10_000_000) y ajuste de curva.
//!
//! # Hipótesis
//!
//! - FMA tradicional: O(n) — cada iteración hace un `exp2` + `log2`.
//! - BML sin Hash Consing: O(n) — cada iteración crea un nodo nuevo.
//! - BML con Hash Consing (cadena): O(n) — cada iteración crea un nodo único.
//! - BML con Hash Consing (repetición): O(1) — `bml(two, two)` siempre
//!   retorna el mismo `NodeId`, el programa RPN es constante.

use bml_compiler::{linearize, HashConsRegistry, RpnProgram};
use bml_domain::BMLTransformer;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Valores de N en progresión geométrica para ajuste de Big O.
///
/// Limitado a 100_000 porque la evaluación recursiva del DAG (`evaluate_soa`)
/// explota la pila en DAGs muy profundos. El hot loop RPN iterativo del
/// Hito 5 eliminará esta limitación.
const N_VALUES: &[u32] = &[10, 100, 1000, 10000, 100000];

/// Construye un DAG BML en cadena con `n` iteraciones, CON Hash Consing.
///
/// Cada iteración hace `bml(node, two)` donde `two` se deduplica, pero
/// `node` es nuevo cada vez. El programa RPN crece O(n).
fn build_chain_with_cons(n: u32) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one); // bml(1, 1) = 2 (deduplicado)
    let mut node = two;
    for _ in 0..n {
        node = reg.bml(node, two);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

/// Construye un DAG BML en cadena con `n` iteraciones, SIN Hash Consing.
///
/// Cada iteración recrea `bml(1, 1)` (sin deduplicar). El programa RPN
/// crece O(2n) — más grande que con Hash Consing.
fn build_chain_no_cons(n: u32) -> RpnProgram {
    let mut t = BMLTransformer::new();
    let two = t.two();
    let mut node = two;
    for _ in 0..n {
        let two2 = t.two();
        node = t.bml(node, two2);
    }
    let soa = t.into_soa();
    linearize(&soa, node)
}

/// Construye un DAG BML con repetición estructural pura, CON Hash Consing.
///
/// Llama `reg.bml(two, two)` n veces. Como `bml(two, two)` siempre produce
/// el mismo `NodeId` (Hash Consing lo deduplica), el programa RPN es
/// **constante** (O(1)) sin importar n.
fn build_repetition_with_cons(n: u32) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one);
    let mut node = two;
    for _ in 0..n {
        node = reg.bml(two, two); // siempre retorna el mismo id
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

/// Fórmula FMA tradicional equivalente: repite `2^x - log2(y)` n veces.
fn fma_repeated(n: u32, x: f64, y: f64) -> f64 {
    let mut acc = x.exp2() - y.log2();
    for _ in 0..n {
        acc = acc.exp2() - y.log2();
    }
    acc
}

/// Benchmark principal: FMA vs BML (cadena) con y sin Hash Consing.
fn bench_fma_vs_bml(c: &mut Criterion) {
    let mut group = c.benchmark_group("fma_vs_bml");

    for &n in N_VALUES {
        let program_cons = build_chain_with_cons(n);
        let program_no_cons = build_chain_no_cons(n);

        group.bench_with_input(BenchmarkId::new("fma", n), &n, |b, &n| {
            b.iter(|| black_box(fma_repeated(black_box(n), 1.5, 2.0)))
        });

        group.bench_with_input(BenchmarkId::new("bml_cons", n), &n, |b, _| {
            b.iter(|| black_box(program_cons.evaluate()))
        });

        group.bench_with_input(BenchmarkId::new("bml_no_cons", n), &n, |b, _| {
            b.iter(|| black_box(program_no_cons.evaluate()))
        });
    }

    group.finish();
}

/// Benchmark de escalado: mide cómo crece el tiempo con N para cada variante.
///
/// Incluye la variante `repetition` que demuestra escalado O(1) gracias
/// al Hash Consing (el programa RPN es constante sin importar N).
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    for &n in N_VALUES {
        let program_chain = build_chain_with_cons(n);
        let program_repetition = build_repetition_with_cons(n);

        group.bench_with_input(BenchmarkId::new("chain_cons", n), &n, |b, _| {
            b.iter(|| black_box(program_chain.evaluate()))
        });

        group.bench_with_input(BenchmarkId::new("repetition_cons", n), &n, |b, _| {
            b.iter(|| black_box(program_repetition.evaluate()))
        });
    }

    group.finish();
}

/// Benchmark de tamaño del programa RPN: mide cuántas operaciones tiene
/// cada variante, para correlacionar tamaño con tiempo de ejecución.
fn bench_program_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("program_size");

    for &n in N_VALUES {
        let chain_cons = build_chain_with_cons(n);
        let chain_no_cons = build_chain_no_cons(n);
        let repetition_cons = build_repetition_with_cons(n);

        group.bench_with_input(BenchmarkId::new("chain_cons_ops", n), &n, |b, _| {
            b.iter(|| black_box(chain_cons.len()))
        });

        group.bench_with_input(BenchmarkId::new("chain_no_cons_ops", n), &n, |b, _| {
            b.iter(|| black_box(chain_no_cons.len()))
        });

        group.bench_with_input(BenchmarkId::new("repetition_cons_ops", n), &n, |b, _| {
            b.iter(|| black_box(repetition_cons.len()))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fma_vs_bml, bench_scaling, bench_program_size);
criterion_main!(benches);
