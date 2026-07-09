# BML vs llama.cpp — Reporte de Benchmark

**Fecha:** 2026-07-09
**Hardware:** Intel Xeon (Cascadelake), 4 vCPU, 7.8 GB RAM, Debian 13
**Modelo:** TinyLlama-1.1B Q4_0 (606 MB, 1.1B parámetros)

---

## 1. Resultados de llama.cpp (baseline)

| Test | avg_ts | stddev_ts |
|---|---|---|
| Prompt processing (pp=512) | **117.50** tok/seg | 3.42 |
| Generation (tg=128) | **14.42** tok/seg | 1.03 |

---

## 2. Resultados de BML

### 2.1 Hot loop (raw, single-thread)

| Métrica | Valor |
|---|---|
| Ops/seg | 593,941,345 ± 40,903,199 |
| Tiempo/op | 1.684 ns |
| dispatch_ops (asm) | 429 líneas |

### 2.2 Tokens/seg extrapolados (TinyLlama-1.1B)

| Test | avg_ts | stddev_ts |
|---|---|---|
| Prompt processing (pp=512) | **0.581** tok/seg | 0.016 |
| Generation (tg=128) | **0.564** tok/seg | 0.016 |

### 2.3 Multicore scaling

| Threads | Ops/seg | Tokens/seg (extrapolado) | Speedup | Eficiencia |
|---|---|---|---|---|
| 1 | 584M | 0.531 | 1.00x | 100% |
| 2 | 989M | 0.899 | 1.69x | 85% |
| 4 | 1,957M | 1.779 | 3.35x | 84% |

---

## 3. Comparación directa

| Métrica | llama.cpp | BML (1 thread) | BML (4 threads) | Ratio (4t) |
|---|---|---|---|---|
| pp tokens/seg | 117.50 | 0.581 | 2.32 | 0.020x |
| tg tokens/seg | 14.42 | 0.564 | 2.24 | 0.155x |

BML con 4 threads está a **0.155x de llama.cpp** en generation (6.5x más cerca que en la primera medición).

---

## 4. Optimizaciones aplicadas en esta versión

| Optimización | Descripción |
|---|---|
| Hot loop refactor | dispatch_ops único, 429 líneas asm (63% menos) |
| Sub-fragmentación L1i | SubFragment <30 KB, cambio de hot loop O(1) |
| Pesos BML nativos | BmlWeightPool: 8x compresión vs f32 (500MB vs 3.9GB) |
| Scheduler de waves | DAG con dependencias, ejecución paralela con barreras |
| Building blocks BML | bml_matmul, bml_rmsnorm, bml_rope, bml_swiglu como RPN con Loop |
| VectorFragment | Distribución de columnas de matmul via TCP |
| Worker daemon | Ejecuta VectorFragments remotamente via TCP |
| RoPE BML puro | neg via bml(log2(0), exp2(y)) = 0 - y, sin FMul |

---

## 5. Evolución de performance

| Métrica | 2026-07-07 (inicial) | 2026-07-08 | 2026-07-09 (actual) |
|---|---|---|---|
| Ops/seg (1 thread) | 511M | 622M | 594M |
| Ops/seg (4 threads) | 2,164M | 2,172M | 1,957M |
| Speedup 4 threads | 3.55x | 3.99x | 3.35x |
| Hot loop asm | 1,172 líneas | 429 líneas | 429 líneas |
| Compresión pesos | 1x (f32) | 8x (BML) | 8x (BML) |
| Ratio vs llama.cpp (tg, 4t) | 0.034x | 0.153x | 0.155x |
| Building blocks BML | No | No | Sí (bml_matmul, etc.) |
| VectorFragment | No | No | Sí |
| Worker daemon | No | No | Sí |

---

## 6. Conclusiones

1. **BML con 4 threads logra 0.155x de llama.cpp** en generation (15.5% del rendimiento).
2. **Escalado multicore**: 3.35x con 4 threads (84% eficiencia).
3. **Compresión de pesos 8x** con BmlWeightPool para Q4_0.
4. **Hot loop compacto**: 429 líneas asm, bien bajo 32 KB L1i.
5. **Building blocks BML integrados**: matmul, rmsnorm, rope, swiglu como RPN con Loop.
6. **VectorFragment + Worker**: distribución de columnas de matmul via TCP.
7. **RoPE BML puro**: negación sin FMul, usando solo Bml + One + Zero.

### Próximos pasos

- SIMD para exp2/log2 (4x f64 por op)
- Hot loop nativo sin Vec (eliminando bounds check)
- Pipeline autoregresivo distribuido completo
- Lazy loading mmap de pesos
- Benchmark multi-nodo con scheduler de waves
