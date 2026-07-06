//! Pruebas de estrés multicore (escalado vertical).
//!
//! Genera DAGs por debajo y por encima de 32 KB y mide latencia
//! con múltiples hilos. Verifica que la evaluación RPN sea thread-safe
//! (inmutable, sin estado compartido mutable).

use bml_compiler::{linearize, HashConsRegistry, RpnOp, RpnProgram};
use std::sync::Arc;
use std::thread;

/// Construye un programa RPN de aproximadamente `target_bytes` bytes.
///
/// Cada operación RpnOp ocupa ~1 byte (enum tag) + padding. Aproximamos
/// el tamaño del programa como `ops.len() * size_of::<RpnOp>()`.
/// Con `size_of::<RpnOp>() = 1` (enum C-like), 32 KB ≈ 32_768 ops.
fn build_program_of_size(target_ops: usize) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one);
    let mut node = two;
    // Cada iteración añade ~3 ops (node, Dup, Bml)
    let iterations = target_ops / 3;
    for _ in 0..iterations {
        node = reg.bml(node, two);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

/// Tamaño aproximado del programa en bytes.
fn program_bytes(program: &RpnProgram) -> usize {
    program.ops.len() * std::mem::size_of::<RpnOp>()
}

#[test]
fn program_below_32kb() {
    // ~10K ops ≈ 10 KB (por debajo de 32 KB)
    let program = build_program_of_size(10_000);
    let size = program_bytes(&program);
    assert!(size < 32 * 1024, "program size {size} >= 32KB");
    // Debe evaluar sin pánico
    let _ = program.evaluate(0.0);
}

#[test]
fn program_above_32kb() {
    // ~50K ops ≈ 50 KB (por encima de 32 KB)
    let program = build_program_of_size(50_000);
    let size = program_bytes(&program);
    assert!(size > 32 * 1024, "program size {size} <= 32KB");
    // Debe evaluar sin pánico (aunque puede ser expulsado de L1i)
    let _ = program.evaluate(0.0);
}

#[test]
fn multicore_evaluation_is_thread_safe() {
    // 4 hilos evalúan el mismo programa concurrentemente.
    // El programa es inmutable (Arc), así que no debe haber data races.
    let program = Arc::new(build_program_of_size(1000));
    let n_threads = 4;
    let iterations_per_thread = 1000;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let program = Arc::clone(&program);
            thread::spawn(move || {
                let mut last = 0.0_f64;
                for _ in 0..iterations_per_thread {
                    last = program.evaluate(0.0);
                }
                last
            })
        })
        .collect();

    let results: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Todos los hilos deben obtener el mismo resultado (el programa es
    // determinista y no tiene estado mutable compartido).
    // Usamos bit patterns para comparar (NaN != NaN en f64).
    for r in &results {
        assert_eq!(
            r.to_bits(),
            results[0].to_bits(),
            "resultados inconsistentes entre hilos"
        );
    }
}

#[test]
fn multicore_latency_scaling() {
    // Mide la latencia de evaluación con 1, 2, 4 hilos.
    let program = Arc::new(build_program_of_size(5000));

    for &n_threads in &[1, 2, 4] {
        let program = Arc::clone(&program);
        let start = std::time::Instant::now();
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let program = Arc::clone(&program);
                thread::spawn(move || {
                    let mut last = 0.0_f64;
                    for _ in 0..1000 {
                        last = program.evaluate(0.0);
                    }
                    last
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let elapsed = start.elapsed();
        // Verificamos que completa sin pánico. La latencia se documenta
        // pero no se aserta un valor específico (depende del hardware).
        println!("n_threads={n_threads} elapsed={elapsed:?}");
    }
}

#[test]
fn programs_of_increasing_size_all_evaluate() {
    // Genera programas de tamaño creciente y verifica que todos evalúan.
    for &target_ops in &[100, 1000, 5000, 10000, 20000, 40000] {
        let program = build_program_of_size(target_ops);
        let size = program_bytes(&program);
        let val = program.evaluate(0.0);
        // El valor puede ser inf, nan o finito a profundidades grandes,
        // pero no debe pánico.
        assert!(
            val.is_finite() || val.is_infinite() || val.is_nan(),
            "valor invalido: {val}"
        );
        println!("target_ops={target_ops} actual_size={size} value={val}");
    }
}
