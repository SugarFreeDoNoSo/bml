//! Prueba distribuida (escalado horizontal) con `/dev/shm`.
//!
//! Simula un entorno append-only multiespacio: `n` trabajadores
//! (simulando procesos/nodos aislados) leen un bloque de memoria
//! compartida en `/dev/shm`, ejecutan su porción del DAG BML, y
//! escriben la salida sin bloqueos (lock-free).
//!
//! # Mecanismo
//!
//! 1. Se serializa un programa RPN en un archivo en `/dev/shm`.
//! 2. N threads (simulando procesos) leen el archivo, evalúan el
//!    programa, y escriben el resultado en archivos propios (append-only).
//! 3. El test verifica que todos los workers producen el mismo resultado.
//!
//! # Notas
//!
//! Esta prueba requiere `/dev/shm` montado. Se omite si no está disponible.
//! La versión por procesos separados requiere un binario helper y queda
//! pendiente para el Hito 5 (runtime distribuido con RPC).

use bml_compiler::{linearize, HashConsRegistry, RpnOp, RpnProgram};
use bml_domain::BMLTransformer;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Construye un programa RPN de prueba.
fn build_test_program() -> RpnProgram {
    // Usar BMLTransformer (sin constant folding) para que el programa
    // tenga nodos One/Bml/Dup que el serializador simple puede manejar.
    let mut t = bml_domain::BMLTransformer::new();
    let two = t.two();
    let two2 = t.two();
    let three = t.bml(two, two2);
    let one = t.one();
    let root = t.bml(three, one);
    let soa = t.into_soa();
    linearize(&soa, root)
}

/// Serializa un programa RPN a bytes (formato simple).
///
/// Formato: [u32 num_ops][u8 op_type * num_ops]
fn serialize_program(program: &RpnProgram) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(program.ops.len() as u32).to_le_bytes());
    for op in &program.ops {
        match op {
            RpnOp::One => bytes.push(0),
            RpnOp::Zero => bytes.push(6),
            RpnOp::Bml => bytes.push(1),
            RpnOp::Dup => bytes.push(2),
            RpnOp::Loop { .. } => bytes.push(3),
            RpnOp::Var(id) => {
                bytes.push(4);
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            RpnOp::Const(id) => {
                bytes.push(5);
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            RpnOp::VarIndexed { base } => {
                bytes.push(7);
                bytes.extend_from_slice(&base.to_le_bytes());
            }
            RpnOp::StoreResult { slot } => {
                bytes.push(8);
                bytes.extend_from_slice(&slot.to_le_bytes());
            }
            RpnOp::FAdd => bytes.push(9),
            RpnOp::FMul => bytes.push(10),
            RpnOp::Pick { depth } => {
                bytes.push(11);
                bytes.extend_from_slice(&depth.to_le_bytes());
            }
            RpnOp::Drop => bytes.push(12),
            RpnOp::Swap => bytes.push(13),
        }
    }
    bytes
}

/// Deserializa un programa RPN desde bytes.
fn deserialize_program(bytes: &[u8]) -> RpnProgram {
    let num_ops = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut ops = Vec::with_capacity(num_ops);
    let mut offset = 4;
    for _ in 0..num_ops {
        if offset >= bytes.len() {
            panic!("offset fuera de rango");
        }
        let tag = bytes[offset];
        offset += 1;
        let op = match tag {
            0 => RpnOp::One,
            6 => RpnOp::Zero,
            1 => RpnOp::Bml,
            2 => RpnOp::Dup,
            3 => {
                // Loop: no soportado en este serializador simple
                panic!("Loop no soportado en serializador de test");
            }
            4 => {
                let id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                RpnOp::Var(id)
            }
            5 => {
                let id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                RpnOp::Const(id)
            }
            7 => {
                let base = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                RpnOp::VarIndexed { base }
            }
            8 => {
                let slot = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                RpnOp::StoreResult { slot }
            }
            9 => RpnOp::FAdd,
            10 => RpnOp::FMul,
            11 => {
                let depth = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                RpnOp::Pick { depth }
            }
            12 => RpnOp::Drop,
            13 => RpnOp::Swap,
            _ => panic!("tag desconocido: {tag}"),
        };
        ops.push(op);
    }
    RpnProgram { ops }
}

#[test]
fn serialize_deserialize_roundtrip() {
    let program = build_test_program();
    let bytes = serialize_program(&program);
    let restored = deserialize_program(&bytes);
    assert_eq!(program.ops, restored.ops);
    assert_eq!(program.evaluate(), restored.evaluate());
}

#[test]
fn dev_shm_distributed_execution() {
    let shm_dir = "/dev/shm";
    if !Path::new(shm_dir).exists() {
        eprintln!("SKIP: {shm_dir} no disponible");
        return;
    }

    let program = build_test_program();
    let program_bytes = serialize_program(&program);
    let expected = program.evaluate();

    // Escribir el programa en /dev/shm
    let prog_path = format!("{shm_dir}/bml_test_prog_{}.bin", std::process::id());
    fs::write(&prog_path, &program_bytes).expect("no se pudo escribir en /dev/shm");

    // N workers leen el programa desde /dev/shm, lo evalúan, y escriben
    // su resultado a archivos propios (append-only, lock-free).
    let n_workers = 4;
    let prog_path = Arc::new(prog_path);
    let mut handles = Vec::new();

    let start = Instant::now();
    for i in 0..n_workers {
        let prog_path = Arc::clone(&prog_path);
        let handle = std::thread::spawn(move || {
            // Cada worker lee el programa desde /dev/shm
            let bytes = fs::read(&*prog_path).expect("no se pudo leer /dev/shm");
            let prog = deserialize_program(&bytes);
            let val = prog.evaluate();
            // Append-only: escribir a un archivo propio (lock-free)
            let out_path = format!("{}_out_{i}", &*prog_path);
            let mut f = fs::File::create(&out_path).expect("no se pudo crear salida");
            write!(f, "{val}").expect("no se pudo escribir salida");
            val
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for h in handles {
        results.push(h.join().unwrap());
    }
    let elapsed = start.elapsed();

    // Limpiar
    let _ = fs::remove_file(&*prog_path);
    for i in 0..n_workers {
        let _ = fs::remove_file(format!("{}_out_{i}", &*prog_path));
    }

    // Verificar que todos los workers producen el mismo resultado.
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            r.to_bits(),
            expected.to_bits(),
            "worker {i}: {r} != {expected}"
        );
    }

    println!("dev_shm_distributed: {n_workers} workers, elapsed={elapsed:?}, expected={expected}");
}

#[test]
fn dev_shm_lock_free_append_only() {
    // Variante del test anterior con más workers y más iteraciones
    // para medir latencia de transferencia.
    let shm_dir = "/dev/shm";
    if !Path::new(shm_dir).exists() {
        eprintln!("SKIP: {shm_dir} no disponible");
        return;
    }

    let program = build_test_program();
    let program_bytes = serialize_program(&program);
    let expected = program.evaluate();

    let prog_path = format!("{shm_dir}/bml_append_prog_{}.bin", std::process::id());
    fs::write(&prog_path, &program_bytes).expect("no se pudo escribir en /dev/shm");

    let n_workers = 8;
    let iterations = 100;
    let prog_path = Arc::new(prog_path);
    let mut handles = Vec::new();

    let start = Instant::now();
    for i in 0..n_workers {
        let prog_path = Arc::clone(&prog_path);
        let handle = std::thread::spawn(move || {
            let bytes = fs::read(&*prog_path).expect("no se pudo leer /dev/shm");
            let prog = deserialize_program(&bytes);
            let mut last = 0.0_f64;
            for _ in 0..iterations {
                last = prog.evaluate();
            }
            // Append-only: escribir resultado final
            let out_path = format!("{}_out_{i}", &*prog_path);
            let mut f = fs::File::create(&out_path).expect("no se pudo crear salida");
            write!(f, "{last}").expect("no se pudo escribir salida");
            last
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for h in handles {
        results.push(h.join().unwrap());
    }
    let elapsed = start.elapsed();

    // Limpiar
    let _ = fs::remove_file(&*prog_path);
    for i in 0..n_workers {
        let _ = fs::remove_file(format!("{}_out_{i}", &*prog_path));
    }

    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            r.to_bits(),
            expected.to_bits(),
            "worker {i}: {r} != {expected}"
        );
    }

    println!(
        "dev_shm_lock_free: {n_workers} workers x {iterations} iters, elapsed={elapsed:?}, per_iter={:?}",
        elapsed / (n_workers as u32 * iterations)
    );
}
