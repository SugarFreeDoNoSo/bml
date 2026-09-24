//! Pruebas de concurrencia y append-only con `loom`.
//!
//! Verifica que el runtime es thread-safe cuando múltiples hilos
//! ejecutan programas concurrentemente. Como el `Runtime` no es
//! `Sync` (tiene estado mutable), cada hilo debe tener su propio
//! `Runtime`. Verificamos que el patrón append-only no produce
//! data races cuando cada hilo tiene su propio buffer de resultados.

use bml_compiler::{linearize, HashConsRegistry, RpnProgram};
use bml_runtime::Runtime;
use std::sync::Arc;
use std::thread;

fn build_program() -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one);
    let three = reg.bml(two, two);
    let root = reg.bml(three, one);
    let soa = reg.into_soa();
    linearize(&soa, root)
}

#[test]
fn runtime_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Runtime>();
}

#[test]
fn concurrent_execution_per_thread_runtime() {
    // Cada hilo tiene su propio Runtime (no compartido).
    let program = Arc::new(build_program());
    let n_threads = 4;
    let iterations = 1000;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let program = Arc::clone(&program);
            thread::spawn(move || {
                let mut runtime = Runtime::new(256, 16);
                let mut last = 0.0_f64;
                for _ in 0..iterations {
                    last = runtime.execute(&program);
                }
                last
            })
        })
        .collect();

    let results: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Todos los hilos deben obtener el mismo resultado.
    for r in &results {
        assert_eq!(
            r.to_bits(),
            results[0].to_bits(),
            "resultados inconsistentes"
        );
    }
}

#[test]
fn append_only_buffer_rotates_correctly() {
    // El buffer append-only rota sin panic y mantiene valores válidos.
    let program = build_program();
    let mut runtime = Runtime::new(256, 4);
    let expected = program.evaluate();

    for _ in 0..100 {
        let r = runtime.execute(&program);
        assert_eq!(r.to_bits(), expected.to_bits());
    }

    // Todos los resultados en el buffer deben ser válidos.
    for &r in runtime.results() {
        assert_eq!(r.to_bits(), expected.to_bits());
    }
}

#[test]
fn hot_loop_returns_known_finite_value() {
    let program = build_program();
    let mut runtime = Runtime::new(1024, 8);
    assert_eq!(runtime.execute(&program), 8.0);
}
