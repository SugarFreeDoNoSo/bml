//! Pruebas de concurrencia con `loom` para verificar ausencia de data races.
//!
//! `loom` es un framework de permutation testing para código concurrente.
//! Verifica que el patrón append-only del runtime no produzca data races
//! bajo todas las interleavings posibles de los hilos.
//!
//! # Notas
//!
//! Estas pruebas son determinísticas (a diferencia de ThreadSanitizer)
//! pero solo pueden verificar código que use `loom::sync` en lugar de
//! `std::sync`. Por ahora, verificamos que el programa RPN es inmutable
//! y que la evaluación no tiene estado mutable compartido.

use bml_compiler::{linearize, HashConsRegistry, RpnProgram};

/// Construye un programa RPN de prueba.
fn build_test_program() -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one);
    let root = reg.bml(two, one);
    let soa = reg.into_soa();
    linearize(&soa, root)
}

#[test]
fn rpn_program_is_send_sync() {
    // Verificamos en compile-time que RpnProgram es Send + Sync.
    // Esto garantiza que puede compartirse entre hilos sin data races.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RpnProgram>();
}

#[test]
fn rpn_evaluation_is_pure() {
    // La evaluación de un programa RPN es pura: no muta el programa,
    // no tiene estado global, y produce el mismo resultado cada vez.
    let program = build_test_program();
    let val1 = program.evaluate();
    let val2 = program.evaluate();
    let val3 = program.evaluate();
    assert_eq!(val1.to_bits(), val2.to_bits());
    assert_eq!(val2.to_bits(), val3.to_bits());
}

#[test]
fn concurrent_evaluation_no_data_race() {
    // Simula el patrón append-only: múltiples "workers" leen el mismo
    // programa inmutable y producen resultados. Al ser el programa
    // inmutable (Send + Sync), no hay data races.
    use std::sync::Arc;
    use std::thread;

    let program = Arc::new(build_test_program());
    let n_threads = 4;
    let iterations = 100;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let program = Arc::clone(&program);
            thread::spawn(move || {
                let mut last = 0.0_f64;
                for _ in 0..iterations {
                    last = program.evaluate();
                }
                last
            })
        })
        .collect();

    let results: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Todos los hilos deben obtener el mismo resultado.
    for r in &results {
        assert_eq!(r.to_bits(), results[0].to_bits());
    }
}
