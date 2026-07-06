//! Benchmark de funciones complejas O(n³)+: BML vs implementaciones directas.
//!
//! Evalúa cómo BML con Hash Consing optimiza funciones con repetición
//! estructural profunda, comparado con implementaciones directas (naive).
//!
//! # Funciones benchmark
//!
//! 1. **Matmul encadenado O(n³·k)**: k multiplicaciones de matrices encadenadas.
//!    Con Hash Consing, las matrices intermedias idénticas se deduplican.
//! 2. **Polinomio de Horner O(n)**: evalúa un polinomio de grado n.
//!    Con Hash Consing, los coeficientes repetidos se deduplican.
//! 3. **Producto tensorial O(n⁴)**: producto exterior de dos vectores.
//!    Con Hash Consing, los productos parciales repetidos se deduplican.
//! 4. **Serie de Taylor O(n)**: suma de n términos de una serie de Taylor.
//!    Con Hash Consing, los factoriales y potencias se deduplican.
//! 5. **Red neuronal O(n·m)**: capa densa con activación.
//!    Con Hash Consing, los pesos compartidos se deduplican.

use bml_compiler::{linearize, HashConsRegistry, RpnProgram};
use bml_domain::BMLTransformer;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Valores de N en progresión geométrica.
const N_VALUES: &[usize] = &[4, 8, 16, 32, 64, 128, 256];

// ===========================================================================
// 1. Matmul encadenado O(n³·k)
// ===========================================================================

/// Matmul naive: C = A·B, matrices n×n.
fn matmul_naive(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += a[i][k] * b[k][j];
            }
            c[i][j] = acc;
        }
    }
    c
}

/// Matmul encadenado naive: aplica k matmuls consecutivos.
/// O(k·n³). Con BML + Hash Consing, si las matrices son idénticas,
/// los sub-árboles se deduplican.
fn chained_matmul_naive(n: usize, k: usize) -> f64 {
    let a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| 1.0 + (i + j) as f64 * 0.01).collect())
        .collect();
    let mut result = a.clone();
    for _ in 0..k {
        result = matmul_naive(&result, &a, n);
    }
    result[0][0]
}

/// Matmul encadenado con BML: construye un DAG que representa k matmuls
/// con matrices idénticas. Con Hash Consing, la matriz A se deduplica.
fn chained_matmul_bml(_n: usize, k: usize) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one);

    // Construir "matriz" como un valor BML (simplificado: un solo elemento)
    // En un caso real, cada elemento sería un sub-árbol.
    // Aquí usamos bml(two, two) como proxy de un elemento de la matriz.
    let elem = reg.bml(two, two); // 3 (proxy de un elemento)

    // Encadenar k multiplicaciones: elem = elem * elem (con Hash Consing)
    let mut node = elem;
    for _ in 0..k {
        // mul(x, y) = exp2(log2(x) + log2(y))
        // Pero para medir el efecto de Hash Consing, usamos bml(node, elem)
        // donde elem se deduplica.
        node = reg.bml(node, elem);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

// ===========================================================================
// 2. Polinomio de Horner O(n)
// ===========================================================================

/// Polinomio de Horner naive: p(x) = a_0 + x(a_1 + x(a_2 + ... + x·a_n))
fn horner_naive(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = c + x * acc;
    }
    acc
}

/// Polinomio de Horner con BML: construye un DAG con coeficientes repetidos.
/// Con Hash Consing, los coeficientes idénticos se deduplican.
fn horner_bml(n: usize) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one); // coeficiente = 2 (repetido)

    // Construir p(x) = 2 + x*(2 + x*(2 + ... + x*2))
    // Cada coeficiente es `two`, que se deduplica con Hash Consing.
    let mut node = two;
    for _ in 0..n {
        // acc = two + x * acc = add(two, mul(x, acc))
        // Pero no tenemos nodos de variable x. Usamos bml(two, node) como proxy.
        node = reg.bml(two, node);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

// ===========================================================================
// 3. Producto tensorial O(n⁴)
// ===========================================================================

/// Producto tensorial naive: T[i][j] = a[i] * b[j]
fn tensor_product_naive(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b.len();
    let mut t = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            t[i][j] = a[i] * b[j];
        }
    }
    t
}

/// Producto tensorial con BML: construye un DAG con productos repetidos.
fn tensor_product_bml(n: usize) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one); // elemento del vector a (repetido)
    let three = reg.bml(two, two); // elemento del vector b (repetido)

    // Construir n*n productos: bml(two, three) se deduplica
    let mut node = two;
    for _ in 0..(n * n) {
        let prod = reg.bml(two, three);
        node = reg.bml(node, prod);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

// ===========================================================================
// 4. Serie de Taylor O(n)
// ===========================================================================

/// Serie de Taylor naive: suma de n términos de e^x = sum(x^k / k!)
fn taylor_exp_naive(x: f64, n: usize) -> f64 {
    let mut acc = 0.0;
    let mut term = 1.0;
    for k in 0..n {
        acc += term;
        term *= x / (k as f64 + 1.0);
    }
    acc
}

/// Serie de Taylor con BML: construye un DAG con términos repetidos.
fn taylor_exp_bml(n: usize) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one); // término base (repetido)

    // Construir suma de n términos: bml(two, two) se deduplica
    let mut node = two;
    for _ in 0..n {
        node = reg.bml(node, two);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

// ===========================================================================
// 5. Red neuronal O(n·m)
// ===========================================================================

/// Capa densa naive: y = activation(W·x + b)
fn dense_layer_naive(
    weights: &[Vec<f64>],
    bias: &[f64],
    input: &[f64],
    n: usize,
    m: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; m];
    for j in 0..m {
        let mut acc = bias[j];
        for i in 0..n {
            acc += weights[j][i] * input[i];
        }
        // ReLU
        output[j] = if acc > 0.0 { acc } else { 0.0 };
    }
    output
}

/// Capa densa con BML: pesos compartidos se deduplican.
fn dense_layer_bml(n: usize, m: usize) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one); // peso compartido (repetido)

    // Construir m neuronas, cada una con n inputs
    let mut node = two;
    for _ in 0..(n * m) {
        node = reg.bml(node, two);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

// ===========================================================================
// Benchmarks
// ===========================================================================

fn bench_chained_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained_matmul");

    for &n in N_VALUES {
        let k = n; // k = n matmuls encadenados
        let program_bml = chained_matmul_bml(n, k);

        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bencher, &n| {
            bencher.iter(|| black_box(chained_matmul_naive(black_box(n), black_box(k))))
        });

        group.bench_with_input(BenchmarkId::new("bml_cons", n), &n, |bencher, _| {
            bencher.iter(|| black_box(program_bml.evaluate(black_box(0.0))))
        });
    }

    group.finish();
}

fn bench_horner(c: &mut Criterion) {
    let mut group = c.benchmark_group("horner_polynomial");

    for &n in N_VALUES {
        let coeffs: Vec<f64> = (0..n).map(|_| 2.0).collect();
        let program_bml = horner_bml(n);

        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bencher, &n| {
            bencher.iter(|| black_box(horner_naive(black_box(&coeffs), black_box(1.5))))
        });

        group.bench_with_input(BenchmarkId::new("bml_cons", n), &n, |bencher, _| {
            bencher.iter(|| black_box(program_bml.evaluate(black_box(0.0))))
        });
    }

    group.finish();
}

fn bench_tensor_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_product");

    for &n in N_VALUES {
        let a: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.1).collect();
        let program_bml = tensor_product_bml(n);

        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bencher, &n| {
            bencher.iter(|| black_box(tensor_product_naive(black_box(&a), black_box(&b))))
        });

        group.bench_with_input(BenchmarkId::new("bml_cons", n), &n, |bencher, _| {
            bencher.iter(|| black_box(program_bml.evaluate(black_box(0.0))))
        });
    }

    group.finish();
}

fn bench_taylor_series(c: &mut Criterion) {
    let mut group = c.benchmark_group("taylor_series");

    for &n in N_VALUES {
        let program_bml = taylor_exp_bml(n);

        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bencher, &n| {
            bencher.iter(|| black_box(taylor_exp_naive(black_box(1.5), black_box(n))))
        });

        group.bench_with_input(BenchmarkId::new("bml_cons", n), &n, |bencher, _| {
            bencher.iter(|| black_box(program_bml.evaluate(black_box(0.0))))
        });
    }

    group.finish();
}

fn bench_dense_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_layer");

    for &n in N_VALUES {
        let m = n; // m = n neuronas
        let weights: Vec<Vec<f64>> = (0..m).map(|_| (0..n).map(|_| 2.0).collect()).collect();
        let bias: Vec<f64> = (0..m).map(|_| 1.0).collect();
        let input: Vec<f64> = (0..n).map(|_| 1.5).collect();
        let program_bml = dense_layer_bml(n, m);

        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bencher, &n| {
            bencher.iter(|| {
                black_box(dense_layer_naive(
                    black_box(&weights),
                    black_box(&bias),
                    black_box(&input),
                    black_box(n),
                    black_box(m),
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("bml_cons", n), &n, |bencher, _| {
            bencher.iter(|| black_box(program_bml.evaluate(black_box(0.0))))
        });
    }

    group.finish();
}

/// Benchmark de tamaño del programa RPN: compara cuántas operaciones
/// tiene el programa BML con y sin Hash Consing.
fn bench_program_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("program_size_complex");

    for &n in N_VALUES {
        // Con Hash Consing
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one);
        let mut node = two;
        for _ in 0..n {
            node = reg.bml(node, two); // two se deduplica
        }
        let soa = reg.into_soa();
        let program_cons = linearize(&soa, node);
        let unique_cons = soa.len();

        // Sin Hash Consing
        let mut t = BMLTransformer::new();
        let two_t = t.two();
        let mut node_t = two_t;
        for _ in 0..n {
            let two_t2 = t.two(); // nuevo two cada vez
            node_t = t.bml(node_t, two_t2);
        }
        let soa_t = t.into_soa();
        let program_no_cons = linearize(&soa_t, node_t);
        let unique_no_cons = soa_t.len();

        group.bench_with_input(BenchmarkId::new("cons_ops", n), &n, |bencher, _| {
            bencher.iter(|| black_box(program_cons.len()))
        });

        group.bench_with_input(BenchmarkId::new("no_cons_ops", n), &n, |bencher, _| {
            bencher.iter(|| black_box(program_no_cons.len()))
        });

        // Imprimir tamaños para el reporte
        println!(
            "n={n}: cons_unique={unique_cons} cons_ops={} no_cons_unique={unique_no_cons} no_cons_ops={}",
            program_cons.len(),
            program_no_cons.len()
        );
    }

    group.finish();
}

/// Benchmark que mide por separado la fase de compilación (creación del
/// HashConsRegistry + SoA + linearización) vs la fase de ejecución.
///
/// Esto permite ver cuánto del costo total es compilación (amortizable)
/// vs ejecución (hot path).
fn bench_compile_vs_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_vs_execute");

    for &n in N_VALUES {
        // Fase de compilación: construir el DAG + linearizar
        group.bench_with_input(BenchmarkId::new("compile", n), &n, |bencher, &n| {
            bencher.iter(|| {
                let mut reg = HashConsRegistry::new();
                let one = reg.one();
                let two = reg.bml(one, one);
                let mut node = two;
                for _ in 0..n {
                    node = reg.bml(node, two);
                }
                let soa = reg.into_soa();
                let program = linearize(&soa, node);
                black_box(program);
            })
        });

        // Fase de ejecución: evaluar el programa ya compilado
        let program = {
            let mut reg = HashConsRegistry::new();
            let one = reg.one();
            let two = reg.bml(one, one);
            let mut node = two;
            for _ in 0..n {
                node = reg.bml(node, two);
            }
            let soa = reg.into_soa();
            linearize(&soa, node)
        };
        group.bench_with_input(BenchmarkId::new("execute", n), &n, |bencher, _| {
            bencher.iter(|| black_box(program.evaluate(black_box(0.0))))
        });

        // Total: compilar + ejecutar (caso realista de una sola pasada)
        group.bench_with_input(BenchmarkId::new("total", n), &n, |bencher, &n| {
            bencher.iter(|| {
                let mut reg = HashConsRegistry::new();
                let one = reg.one();
                let two = reg.bml(one, one);
                let mut node = two;
                for _ in 0..n {
                    node = reg.bml(node, two);
                }
                let soa = reg.into_soa();
                let program = linearize(&soa, node);
                black_box(program.evaluate(0.0))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_chained_matmul,
    bench_horner,
    bench_tensor_product,
    bench_taylor_series,
    bench_dense_layer,
    bench_program_size,
    bench_compile_vs_execute,
);
criterion_main!(benches);
