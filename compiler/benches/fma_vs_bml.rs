//! Benchmark comparativo: FMA tradicional vs DAG BML deduplicado.
//!
//! Compara el costo de evaluar una fórmula compleja con operaciones
//! repetidas usando:
//! 1. FMA tradicional (fórmula directa en f64).
//! 2. DAG BML con Hash Consing (sub-árboles repetidos se evalúan una sola vez).
//!
//! El objetivo es demostrar la reducción a tiempo sub-lineal para
//! operaciones repetidas gracias al Hash Consing.

use bml_compiler::{linearize, HashConsRegistry, RpnProgram};
use bml_domain::BMLTransformer;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Construye un DAG BML con `n` repeticiones del sub-árbol `bml(1, 1)`.
///
/// Sin Hash Consing, el DAG tendría O(n) nodos. Con Hash Consing,
/// el sub-árbol `bml(1, 1)` se deduplica, quedando O(log n) nodos únicos.
fn build_dag_with_repetition(n: u32) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one); // bml(1, 1) = 2 (deduplicado)
    let mut node = two;
    for _ in 0..n {
        // bml(node, two) — `two` se reutiliza, no se recalcula
        node = reg.bml(node, two);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

/// Construye el mismo DAG sin Hash Consing (usando BMLTransformer directamente).
fn build_dag_no_cons(n: u32) -> RpnProgram {
    let mut t = BMLTransformer::new();
    let two = t.two(); // bml(1, 1)
    let mut node = two;
    for _ in 0..n {
        let two2 = t.two(); // nuevo bml(1, 1) cada vez (sin deduplicar)
        node = t.bml(node, two2);
    }
    let soa = t.into_soa();
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

fn bench_fma_vs_bml(c: &mut Criterion) {
    let mut group = c.benchmark_group("fma_vs_bml");

    for n in [10, 100, 1000, 10000].iter() {
        let n = *n as u32;
        let program_cons = build_dag_with_repetition(n);
        let program_no_cons = build_dag_no_cons(n);

        group.bench_function(format!("fma_n{n}"), |b| {
            b.iter(|| black_box(fma_repeated(black_box(n), 1.5, 2.0)))
        });

        group.bench_function(format!("bml_cons_n{n}"), |b| {
            b.iter(|| black_box(program_cons.evaluate(black_box(0.0))))
        });

        group.bench_function(format!("bml_no_cons_n{n}"), |b| {
            b.iter(|| black_box(program_no_cons.evaluate(black_box(0.0))))
        });
    }

    group.finish();
}

/// Benchmark de escalado: mide cómo crece el tiempo con `n` para
/// demostrar la reducción sub-lineal con Hash Consing.
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    for n in [10, 100, 1000, 10000, 100000].iter() {
        let n = *n as u32;
        let program = build_dag_with_repetition(n);
        group.bench_function(format!("bml_cons_n{n}"), |b| {
            b.iter(|| black_box(program.evaluate(black_box(0.0))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fma_vs_bml, bench_scaling);
criterion_main!(benches);
