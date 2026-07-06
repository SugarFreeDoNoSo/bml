//! # Benchmark Final: Tokens/seg, escalado a máquinas, costo por millón de tokens
//!
//! Este benchmark mide el rendimiento del runtime BML en términos de:
//! 1. **Tokens por segundo** (tpS) en la máquina local.
//! 2. **Escalado a máquinas cloud** — cuántas máquinas se necesitan para
//!    ejecutar un modelo de tamaño X.
//! 3. **Costo por millón de tokens** — traducido a dólares según el
//!    proveedor cloud seleccionado.
//!
//! # Metodología
//!
//! El benchmark asume que el pipeline de inferencia está implementado:
//! - El GGUF ya está compilado a `.bmlgraph` (no se mide la compilación).
//! - Los fragmentos ya están en memoria (no se mide la carga).
//! - El hot loop ejecuta con core dedicado, sin context switching.
//! - Solo se mide la **ejecución** del hot loop.
//!
//! # Equivalencia de token
//!
//! Un "token" requiere N operaciones BML, donde N se calcula a partir
//! del costo computacional del modelo:
//! - FLOPs por token ≈ 2 * params * tokens (para transformer denso)
//! - Cada operación BML ≈ 2 FLOPs (exp2 + log2)
//! - N = FLOPs_per_token / 2
//!
//! # Modelos de referencia
//!
//! | Modelo | Params | FLOPs/token | N (ops BML/token) |
//!---|---|---|---|
//! | TinyLlama 1.1B | 1.1B | 2.2B | 1.1B |
//! | Llama 7B | 7B | 14B | 7B |
//! | Llama 13B | 13B | 26B | 13B |
//! | Llama 70B | 70B | 140B | 70B |
//!
//! # Proveedores cloud
//!
//! Los costos se calculan para los siguientes proveedores (precios 2026):
//! - Hetzner CCX (dedicated AMD EPYC, más barato)
//! - Vultr High Performance (dedicated AMD EPYC)
//! - GCP N2D (AMD EPYC, sole-tenant opcional)
//! - AWS c7i (Intel Sapphire Rapids)

use bml_compiler::{linearize, HashConsRegistry, RpnProgram};
use bml_runtime::Runtime;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Instant;

/// Modelos de referencia con sus parámetros y FLOPs por token.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    /// Nombre del modelo.
    pub name: &'static str,
    /// Número de parámetros (en miles de millones).
    pub params_b: f64,
    /// FLOPs por token = 2 * params * 1 (para transformer denso).
    pub flops_per_token: f64,
}

impl ModelSpec {
    /// Número de operaciones BML por token.
    /// Cada bml hace ~2 FLOPs (exp2 + log2).
    pub fn bml_ops_per_token(&self) -> f64 {
        self.flops_per_token / 2.0
    }

    /// Tamaño del modelo en GB (asumiendo Q4 = 0.5 bytes/param).
    pub fn size_gb_q4(&self) -> f64 {
        self.params_b * 0.5
    }
}

/// Proveedor cloud con sus costos.
#[derive(Debug, Clone, Copy)]
pub struct CloudProvider {
    /// Nombre del proveedor.
    pub name: &'static str,
    /// Tipo de instancia.
    pub instance_type: &'static str,
    /// Número de vCPUs.
    pub vcpus: u32,
    /// Costo por hora en USD.
    pub cost_per_hour: f64,
    /// Tamaño de caché L1i en KB.
    pub l1i_kb: u32,
    /// Tamaño de caché L3 en MB.
    pub l3_mb: u32,
}

impl CloudProvider {
    /// Costo por segundo en USD.
    pub fn cost_per_second(&self) -> f64 {
        self.cost_per_hour / 3600.0
    }
}

/// Modelos de referencia.
const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "TinyLlama-1.1B",
        params_b: 1.1,
        flops_per_token: 2.2e9,
    },
    ModelSpec {
        name: "Llama-7B",
        params_b: 7.0,
        flops_per_token: 14.0e9,
    },
    ModelSpec {
        name: "Llama-13B",
        params_b: 13.0,
        flops_per_token: 26.0e9,
    },
    ModelSpec {
        name: "Llama-70B",
        params_b: 70.0,
        flops_per_token: 140.0e9,
    },
];

/// Proveedores cloud (precios 2026, USD/hr, dedicated vCPU).
const PROVIDERS: &[CloudProvider] = &[
    CloudProvider {
        name: "Hetzner",
        instance_type: "CCX13 (4 vCPU)",
        vcpus: 4,
        cost_per_hour: 0.064,
        l1i_kb: 32,
        l3_mb: 256,
    },
    CloudProvider {
        name: "Hetzner",
        instance_type: "CCX33 (16 vCPU)",
        vcpus: 16,
        cost_per_hour: 0.193,
        l1i_kb: 32,
        l3_mb: 256,
    },
    CloudProvider {
        name: "Hetzner",
        instance_type: "CCX63 (32 vCPU)",
        vcpus: 32,
        cost_per_hour: 0.386,
        l1i_kb: 32,
        l3_mb: 256,
    },
    CloudProvider {
        name: "Vultr",
        instance_type: "HP-4 (4 vCPU)",
        vcpus: 4,
        cost_per_hour: 0.179,
        l1i_kb: 32,
        l3_mb: 256,
    },
    CloudProvider {
        name: "Vultr",
        instance_type: "HP-16 (16 vCPU)",
        vcpus: 16,
        cost_per_hour: 0.714,
        l1i_kb: 32,
        l3_mb: 256,
    },
    CloudProvider {
        name: "GCP",
        instance_type: "N2D-4 (4 vCPU)",
        vcpus: 4,
        cost_per_hour: 0.097,
        l1i_kb: 32,
        l3_mb: 256,
    },
    CloudProvider {
        name: "GCP",
        instance_type: "N2D-16 (16 vCPU)",
        vcpus: 16,
        cost_per_hour: 0.388,
        l1i_kb: 32,
        l3_mb: 256,
    },
    CloudProvider {
        name: "AWS",
        instance_type: "c7i-4xlarge (16 vCPU)",
        vcpus: 16,
        cost_per_hour: 0.720,
        l1i_kb: 32,
        l3_mb: 105,
    },
];

/// Construye un programa BML de N operaciones (ya compilado, sin medir preparación).
fn build_program(n_ops: usize) -> RpnProgram {
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

/// Mide tokens por segundo para un modelo dado.
///
/// Ejecuta el hot loop con un programa de N ops (donde N = bml_ops_per_token),
/// mide el tiempo, y calcula tokens/seg.
///
/// Como no podemos construir un programa de miles de millones de ops,
/// medimos con un programa de tamaño medible y extrapolamos.
fn measure_tokens_per_second(model: &ModelSpec, ops_per_token: f64) -> f64 {
    // Medir con un programa de 100K ops y extrapolar.
    // El hot loop es O(n) en tiempo, así que tokens/seg = ops_per_token / tiempo_por_op.
    let sample_ops = 100_000;
    let program = build_program(sample_ops);

    let mut runtime = Runtime::new(8192, 16);

    // Warmup
    for _ in 0..10 {
        runtime.execute(&program, 0.0);
    }

    // Medir
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        runtime.execute(&program, 0.0);
    }
    let elapsed = start.elapsed();

    // ops por segundo = (sample_ops * iterations) / elapsed
    let ops_per_second = (sample_ops * iterations) as f64 / elapsed.as_secs_f64();

    // tokens por segundo = ops_per_second / ops_per_token
    ops_per_second / ops_per_token
}

/// Calcula cuántas máquinas se necesitan para ejecutar un modelo en tiempo real.
///
/// Asume que cada máquina ejecuta su porción del modelo en paralelo.
/// `target_tokens_per_second` es la velocidad objetivo (ej. 20 tpS para
/// conversación en tiempo real).
fn machines_needed(tokens_per_second_per_machine: f64, target_tokens_per_second: f64) -> f64 {
    (target_tokens_per_second / tokens_per_second_per_machine).ceil()
}

/// Calcula el costo por millón de tokens en USD.
fn cost_per_million_tokens(tokens_per_second_per_machine: f64, provider: &CloudProvider) -> f64 {
    // costo por segundo / tokens por segundo = costo por token
    // * 1_000_000 = costo por millón de tokens
    let cost_per_token = provider.cost_per_second() / tokens_per_second_per_machine;
    cost_per_token * 1_000_000.0
}

/// Benchmark principal: mide tokens/seg, máquinas necesarias, y costo.
fn bench_final(c: &mut Criterion) {
    let mut group = c.benchmark_group("final_benchmark");

    // Medir ops/seg del hot loop (sin preparación)
    for &n_ops in &[1_000, 10_000, 100_000] {
        let program = build_program(n_ops);
        let mut runtime = Runtime::new(8192, 16);

        group.bench_function(format!("hot_loop_{n_ops}_ops"), |b| {
            b.iter(|| black_box(runtime.execute(black_box(&program), black_box(0.0))))
        });
    }

    group.finish();

    // Generar el reporte completo al final
    generate_report();
}

/// Genera el reporte completo de tokens/seg, máquinas, y costo.
fn generate_report() {
    println!("\n{}", "=".repeat(80));
    println!("BML FINAL BENCHMARK REPORT");
    println!("{}\n", "=".repeat(80));

    // Medir ops/seg del hot loop
    let sample_ops = 100_000;
    let program = build_program(sample_ops);
    let mut runtime = Runtime::new(8192, 16);

    // Warmup
    for _ in 0..10 {
        runtime.execute(&program, 0.0);
    }

    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        runtime.execute(&program, 0.0);
    }
    let elapsed = start.elapsed();
    let ops_per_second = (sample_ops * iterations) as f64 / elapsed.as_secs_f64();

    println!("Hot loop performance (local machine):");
    println!("  Ops/second: {ops_per_second:.0}");
    println!("  Time per op: {:.3} ns", 1e9 / ops_per_second);
    println!("  Sample size: {sample_ops} ops x {iterations} iterations");
    println!("  Elapsed: {elapsed:?}\n");

    // Para cada modelo, calcular tokens/seg, máquinas, y costo
    println!("{}", "=".repeat(80));
    println!("MODEL ANALYSIS (extrapolated from hot loop ops/sec)");
    println!("{}\n", "=".repeat(80));

    for model in MODELS {
        let ops_per_token = model.bml_ops_per_token();
        let tokens_per_second = ops_per_second / ops_per_token;
        let size_gb = model.size_gb_q4();

        println!(
            "Model: {} ({}B params, {:.1}GB Q4)",
            model.name, model.params_b, size_gb
        );
        println!("  FLOPs/token: {:.2e}", model.flops_per_token);
        println!("  BML ops/token: {:.2e}", ops_per_token);
        println!("  Tokens/second (1 machine): {tokens_per_second:.4}");
        println!();

        // Para cada proveedor, calcular máquinas necesarias y costo
        println!(
            "  {:<35} {:>5} {:>8} {:>15} {:>10}",
            "Provider", "vCPUs", "$/hr", "machines@20tpS", "$/Mtok"
        );
        println!("  {}", "-".repeat(80));

        for provider in PROVIDERS {
            // Asumir que el rendimiento escala linealmente con vCPUs
            // (cada core dedicado ejecuta su fragmento)
            let tps_per_machine = tokens_per_second * provider.vcpus as f64 / 4.0; // normalizar a 4 vCPU local
            let target_tps = 20.0; // tiempo real de conversación
            let machines = machines_needed(tps_per_machine, target_tps);
            let cost = cost_per_million_tokens(tps_per_machine, provider);

            println!(
                "  {:<35} {:>5} {:>8.3} {:>15.1} {:>10.2}",
                format!("{} {}", provider.name, provider.instance_type),
                provider.vcpus,
                provider.cost_per_hour,
                machines,
                cost
            );
        }
        println!();
    }

    // Comparación con llama.cpp (datos del benchmark ya ejecutado)
    println!("{}", "=".repeat(80));
    println!("COMPARISON WITH LLAMA.CPP (TinyLlama-1.1B Q4_0, 4 vCPU)");
    println!("{}\n", "=".repeat(80));

    let llamacpp_pp_tps = 148.34; // medido con llama-bench
    let llamacpp_tg_tps = 30.43; // medido con llama-bench

    let tinyllama = &MODELS[0];
    let bml_tps = ops_per_second / tinyllama.bml_ops_per_token();

    println!("llama.cpp prompt processing: {llamacpp_pp_tps:.2} tokens/sec");
    println!("llama.cpp text generation:   {llamacpp_tg_tps:.2} tokens/sec");
    println!("BML (extrapolated):          {bml_tps:.6} tokens/sec");
    println!();
    println!(
        "Ratio BML/llama.cpp (generation): {:.4}x",
        bml_tps / llamacpp_tg_tps
    );
    println!();
    println!("NOTE: BML is currently much slower because:");
    println!("  1. Each bml op does exp2+log2 (~5ns) vs FMA (~3ns)");
    println!("  2. The RPN interpreter has overhead (push/pop, match dispatch)");
    println!("  3. No SIMD, no BLAS, no flash attention");
    println!("  4. The hot loop uses Vec<f64> instead of fixed buffer");
    println!();
    println!("Projected performance with optimizations:");
    println!("  - Hot loop native (no Vec): ~2x faster");
    println!("  - SIMD (4x f64 per op): ~4x faster");
    println!("  - exp2/log2 bit-twiddling: ~2x faster");
    println!(
        "  - Combined: ~16x faster = {:.4} tokens/sec",
        bml_tps * 16.0
    );
    println!();
}

criterion_group!(benches, bench_final);
criterion_main!(benches);
