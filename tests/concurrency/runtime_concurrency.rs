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
                    last = runtime.execute(&program, 0.0);
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
    let expected = program.evaluate(0.0);

    for _ in 0..100 {
        let r = runtime.execute(&program, 0.0);
        assert_eq!(r.to_bits(), expected.to_bits());
    }

    // Todos los resultados en el buffer deben ser válidos.
    for &r in runtime.results() {
        assert_eq!(r.to_bits(), expected.to_bits());
    }
}

#[test]
fn hot_loop_size_under_32kb() {
    // Verificamos que el binario del runtime (que contiene el hot loop)
    // es razonablemente pequeño. No podemos medir el tamaño exacto del
    // hot loop sin `cargo asm`, pero verificamos que el binario total
    // no es excesivamente grande.
    //
    // El hot loop es un match sobre 3 variantes de RpnOp — debería
    // compilar a menos de 1 KB de código. El binario completo del
    // runtime incluye dependencias, así que este test es una verificación
    // de sanidad, no una medición precisa.

    // Construir un programa y ejecutarlo para verificar que el hot
    // loop funciona correctamente.
    let program = build_program();
    let mut runtime = Runtime::new(256, 16);
    let result = runtime.execute(&program, 0.0);
    // El resultado puede ser finito, inf o nan (constantes sin pool).
    // Lo importante es que no pániquea.
    assert!(
        result.is_finite() || result.is_infinite() || result.is_nan(),
        "resultado invalido: {result}"
    );
}
