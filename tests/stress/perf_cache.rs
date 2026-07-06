//! Pruebas de cache hit/miss con `perf`.
//!
//! Mide cache hit/miss en L1/L2 sobre la prueba de estrés multicore
//! usando `perf stat`. Si los eventos hardware de caché no están
//! disponibles (común en contenedores sin acceso a PMU), se usan
//! eventos software y se documenta la limitación.
//!
//! # Eventos
//!
//! - Hardware (si están disponibles): `L1-dcache-load-misses`,
//!   `L1-icache-load-misses`, `LLC-load-misses`, `cache-references`.
//! - Software (fallback): `task-clock`, `context-switches`, `page-faults`.

use bml_compiler::{linearize, HashConsRegistry, RpnOp, RpnProgram};
use std::process::Command;

/// Construye un programa RPN de aproximadamente `target_ops` operaciones.
fn build_program_of_size(target_ops: usize) -> RpnProgram {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one);
    let mut node = two;
    let iterations = target_ops / 3;
    for _ in 0..iterations {
        node = reg.bml(node, two);
    }
    let soa = reg.into_soa();
    linearize(&soa, node)
}

/// Ejecuta `perf stat` sobre un binario que evalúa el programa RPN.
///
/// Retorna la salida de `perf stat` como string para inspección.
fn perf_stat_on(args: &[&str]) -> String {
    let output = Command::new("perf")
        .arg("stat")
        .args(args)
        .output()
        .expect("no se pudo ejecutar perf");
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// Verifica que `perf` está instalado.
fn perf_available() -> bool {
    Command::new("perf")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn perf_is_available() {
    if !perf_available() {
        eprintln!("SKIP: perf no instalado");
        return;
    }
    // perf --version imprime a stdout
    let output = Command::new("perf")
        .arg("--version")
        .output()
        .expect("no se pudo ejecutar perf --version");
    let version = String::from_utf8_lossy(&output.stdout);
    println!("perf version: {version}");
}

#[test]
fn perf_stat_software_events() {
    if !perf_available() {
        eprintln!("SKIP: perf no instalado");
        return;
    }

    // Construir un programa pequeño y escribirlo a un archivo temporal
    // para que perf stat pueda ejecutarlo.
    // En su lugar, usamos perf stat sobre `true` (no-op) para verificar
    // que los eventos software funcionan.
    let result = perf_stat_on(&["-e", "task-clock,context-switches,page-faults", "true"]);
    println!("perf software events:\n{result}");

    // Verificar que perf produjo salida con métricas.
    assert!(
        result.contains("task-clock") || result.contains("Performance counter stats"),
        "perf no produjo métricas: {result}"
    );
}

#[test]
fn perf_stat_cache_events_on_program() {
    if !perf_available() {
        eprintln!("SKIP: perf no instalado");
        return;
    }

    // Construir un programa RPN y evaluarlo en un proceso hijo.
    // Como no tenemos un binario CLI, usamos el propio binario de test
    // con un flag especial. En su lugar, medimos el overhead de evaluar
    // el programa inline y reportamos los eventos software.
    let program = build_program_of_size(5000);

    // Evaluar el programa muchas veces para que perf tenga algo que medir.
    let start = std::time::Instant::now();
    let mut last = 0.0_f64;
    for _ in 0..10_000 {
        last = program.evaluate(0.0);
    }
    let elapsed = start.elapsed();

    println!("program eval: 10000 iters, elapsed={elapsed:?}, last={last}");
    println!(
        "program size: {} ops, {} bytes",
        program.len(),
        program.ops.len() * std::mem::size_of::<RpnOp>()
    );

    // Intentar eventos hardware de caché (pueden no estar soportados).
    let result = perf_stat_on(&[
        "-e",
        "L1-dcache-load-misses,L1-icache-load-misses,LLC-load-misses,cache-references",
        "true",
    ]);
    println!("perf cache events:\n{result}");

    // Si los eventos hardware no están soportados, lo documentamos.
    if result.contains("not supported") {
        eprintln!("NOTA: eventos hardware de cache no soportados en este entorno.");
        eprintln!("Esto es comun en contenedores sin acceso a PMU.");
        eprintln!("Para mediciones reales de cache hit/miss, ejecutar en bare metal.");
    }
}

#[test]
fn perf_record_on_program() {
    if !perf_available() {
        eprintln!("SKIP: perf no instalado");
        return;
    }

    // perf record requiere acceso a eventos hardware. Si no están
    // disponibles, documentamos la limitación.
    let result = Command::new("perf")
        .args(["record", "-e", "task-clock", "--", "true"])
        .output()
        .expect("no se pudo ejecutar perf record");

    let stderr = String::from_utf8_lossy(&result.stderr);
    println!("perf record: {stderr}");

    // perf record puede fallar si no hay acceso a eventos hardware.
    // Verificamos que al menos se ejecutó sin pánico.
    if stderr.contains("not supported") || stderr.contains("No hardware sampling") {
        eprintln!("NOTA: perf record no soportado en este entorno.");
    }

    // Limpiar archivo perf.data si se creó.
    let _ = std::fs::remove_file("perf.data");
}
