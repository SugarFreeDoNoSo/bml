# BML vs llama.cpp — Reporte de Benchmark Comparativo

**Fecha:** 2026-07-07
**Hardware:** Intel Xeon (Cascadelake), 4 vCPU, 7.8 GB RAM, Debian 13
**Modelo:** TinyLlama-1.1B Q4_0 (606 MB, 1.1B parámetros)

Ver `HARDWARE.md` para detalles completos del entorno.

---

## 1. Metodología

### 1.1 Objetivo

Medir el rendimiento del runtime BML (intérprete RPN con hot loop L1) frente a
`llama.cpp` (runtime maduro con BLAS, SIMD, flash attention) sobre el mismo
hardware y modelo, para validar o refutar la hipótesis del draft: que la
arquitectura BML (operador único + Hash Consing + hot loop L1 +
micro-fragmentación) ofrece ventajas de rendimiento.

### 1.2 Métricas

Se replican las métricas de `llama-bench`:

- **pp_avg / pp_stddev** — tokens/seg de prompt processing (prefill), media y
  desviación estándar sobre ≥5 repeticiones.
- **tg_avg / tg_stddev** — tokens/seg de generation (decode autoregresivo).
- **samples_ns / samples_ts** — muestras individuales en ns y tokens/seg.

### 1.3 Equivalencia "token BML"

BML **no implementa un transformer**. No tiene atención, MLP, sampling ni
tokenización. La comparación se hace a nivel de **costo de operaciones
matemáticas equivalentes**:

- FLOPs/token ≈ `2 * params` (transformer denso) → TinyLlama = 2.2e9 FLOPs/token
- Cada operación BML (`2^x - log2(y)`) ≈ 2 FLOPs (exp2 + log2)
- **N (BML ops/token) = 1.1e9**

Como no es posible construir un programa RPN de miles de millones de ops, se
mide con un programa de 100K ops y se extrapola: `tokens/seg = ops/seg / N`.

### 1.4 Parámetros alineados

| Parámetro | llama.cpp | BML |
|---|---|---|
| Modelo | TinyLlama-1.1B Q4_0 | equivalente |
| Threads | 4 | 1 (single-threaded) |
| pp tokens | 512 | 512 |
| tg tokens | 128 | 128 |
| Repeticiones | 5 | 5 |

### 1.5 Limitaciones de la comparación

1. **BML no hace inferencia LLM completa.** Compara costo del operador BML vs
   operaciones FMA/exp/log de llama.cpp, no inferencia end-to-end.
2. **BML es single-threaded.** llama.cpp usa 4 threads con paralelismo de
   matmul. BML no tiene paralelismo intra-op.
3. **BML no tiene SIMD ni BLAS.** Cada op es `exp2 + log2` escalar.
4. **BML no tiene flash attention ni KV cache optimizado.**
5. La extrapolación asume que todas las ops BML cuestan lo mismo (uniformidad).

---

## 2. Resultados de llama.cpp (baseline)

Fuente: `llamacpp_pp.json`, `llamacpp_tg.json`, `llamacpp_combined.json`.

| Test | avg_ts | stddev_ts | avg_ns | n_threads |
|---|---|---|---|---|
| Prompt processing (pp=512) | **119.69** tok/seg | 1.69 | 4.28e9 ns | 4 |
| Generation (tg=128) | **17.12** tok/seg | 1.19 | 7.51e9 ns | 4 |
| Combined (pp=512, tg=128) | **42.16** tok/seg | 1.97 | 1.52e10 ns | 4 |

---

## 3. Resultados de BML (extrapolado)

Fuente: `bml_results.json`, `bml_results.md` (release build).

### 3.1 Hot loop (raw)

| Métrica | Valor |
|---|---|
| Ops/seg | 511,731,499 ± 36,231,278 |
| Tiempo/op | 1.954 ns |
| Programa | 100K ops × 1000 iters/muestra |
| Repeticiones | 5 |
| **Tamaño del rlib (hot loop)** | **5,390 bytes (5.26 KB)** — bien bajo 32 KB L1i |

### 3.2 Tokens/seg extrapolados (TinyLlama-1.1B)

| Test | avg_ts | stddev_ts | avg_ns | bml_ops/token |
|---|---|---|---|---|
| Prompt processing (pp=512) | **0.480** tok/seg | 0.012 | 1.89e8 ns | 1.1e9 |
| Generation (tg=128) | **0.461** tok/seg | 0.074 | 2.04e8 ns | 1.1e9 |

> Nota: pp y tg son iguales en BML porque no hay distinción entre prefill y
> decode — ambos ejecutan el mismo hot loop lineal. La diferencia en llama.cpp
> se debe al KV cache y a que el decode es autoregresivo (1 token a la vez).

---

## 4. Comparación directa

| Métrica | llama.cpp | BML | Ratio BML/llama.cpp |
|---|---|---|---|
| pp tokens/seg | 119.69 | 0.480 | **0.0040x** (249x más lento) |
| tg tokens/seg | 17.12 | 0.461 | **0.0269x** (37x más lento) |
| Ops/seg (crudo) | — | 5.12e8 | — |
| Tiempo/op | — | 1.95 ns | — |

**BML es entre 37x y 249x más lento que llama.cpp** en este escenario.

---

## 5. Micro-benchmarks de operaciones individuales

Fuente: `criterion` bench `bml_ops` (release).

### 5.1 Costo por operación (tarea 4.1)

| Operación | Tiempo (ns) |
|---|---|
| FMA (`a*b + c`) | **2.16** |
| `exp2(x)` | 4.90 |
| `log2(y)` | 6.50 |
| BML inline (`exp2(x) - log2(y)`) | 10.60 |
| BML como función `bml(x, y)` | 11.35 |

- **BML cuesta ~5.2x un FMA** (11.35 vs 2.16 ns).
- El overhead de llamada a función es ~0.75 ns (11.35 vs 10.60 inline).
- `exp2` + `log2` por separado = 11.40 ns ≈ BML inline (10.60 ns). Sin overhead.

### 5.2 Hot loop por tamaño de programa (tarea 4.3)

| N ops | Tiempo total (ns) | ns/op |
|---|---|---|
| 10 | 8.14 | 0.81 |
| 100 | 170.61 | 1.71 |
| 1,000 | 1,753.57 | 1.75 |
| 10,000 | 18,929.36 | 1.89 |
| 100,000 | 175,208.22 | 1.75 |

- **Escalado lineal** confirmado (O(n)).
- Costo amortizado por op: **~1.75 ns/op** en release.
- A tamaño 10 hay overhead fijo dominante (~8 ns de setup).

### 5.3 Matmul BML RPN vs naive vs ndarray (tarea 4.2)

Fuente: `crates/compiler/benches/matrix_mul.rs` (N=8, dim 338).

| Variante | Tiempo (ns) |
|---|---|
| naive (3 loops) | 18.35 |
| ndarray | 80.72 |
| BML RPN | 326.13 |

- BML RPN matmul es ~18x más lento que naive (326 vs 18 ns).
- El overhead del intérprete RPN (push/pop, match dispatch) domina en matmul pequeño.

### 5.4 Efecto del Hash Consing (tarea 4.4)

Fuente: `crates/compiler/benches/fma_vs_bml.rs`.

- Cadena con Hash Consing: O(n) — cada iteración crea un nodo nuevo.
- Cadena sin Hash Consing: O(n) — más grande pero mismo orden.
- **Repetición con Hash Consing: O(1)** — `bml(two, two)` se deduplica, el
  programa RPN es constante sin importar N. Este es el caso ideal donde BML
  brilla: cuando hay repetición estructural profunda.

---

## 6. Análisis de complejidad Big O

| Componente | Big O | Notas |
|---|---|---|
| Hot loop BML | O(n) | Lineal en número de ops |
| Hash Consing (cadena) | O(n) | Cada nodo es único |
| Hash Consing (repetición) | **O(1)** | Sub-árboles idénticos se deduplican |
| Matmul BML RPN | O(n_out * n_in) | Sin SIMD, sin threading |
| llama.cpp matmul | O(n_out * n_in / threads) | Con BLAS y threading |

Ver `BENCHMARK_REPORT.md` y `FINAL_BENCHMARK_REPORT.md` para análisis previo.

---

## 7. Proyección de rendimiento potencial

BML es 37-249x más lento que llama.cpp en el estado actual. Proyecciones con
optimizaciones hipotéticas:

| Optimización | Speedup estimado | Razón |
|---|---|---|
| Hot loop nativo (sin Vec) | ~2x | Eliminar bounds check + push/pop |
| SIMD (4x f64 por op) | ~4x | `exp2` + `log2` vectorial |
| `exp2`/`log2` bit-twiddling | ~2x | Aproximación rápida IEEE 754 |
| Multithreading (4 cores) | ~4x | Paralelismo intra-op |
| **Combinado** | **~64x** | |

Con todo combinado: `0.461 * 64 = ~29.5 tok/seg` — comparable a llama.cpp
generation (17.12 tok/seg). **Pero esto es especulativo** y requiere
implementación real.

---

## 8. Conclusiones

1. **BML en su estado actual es 37-249x más lento que llama.cpp** para
   TinyLlama-1.1B en CPU de 4 cores. Esto era esperado: BML no tiene SIMD,
   BLAS, multithreading, ni flash attention.

2. **El operador BML cuesta 5.2x un FMA** (11.35 ns vs 2.16 ns). Esto limita
   el rendimiento de cualquier programa BML vs uno basado en FMA.

3. **El hot loop es genuinamente pequeño (5.26 KB)** — bien bajo el umbral
   L1i de 32 KB. La hipótesis de "hot loop L1" se cumple estructuralmente.

4. **El escalado es lineal O(n)** con ~1.75 ns/op en release. No hay
   sorpresas: el runtime es predecible y estable.

5. **El Hash Consing da O(1) en repetición estructural** — la única ventaja
   teórica clara de BML. Para modelos con mucha repetición (ej. MoE, pesos
   compartidos), esto podría compensar el overhead del operador.

6. **Sin SIMD ni multithreading, BML no es competitivo** para inferencia LLM.
   El camino a competitividad requiere vectorización y paralelismo.

### Próximos pasos

- Implementar hot loop nativo (sin `Vec`, sin `match` dispatch).
- Añadir SIMD para `exp2`/`log2` (`std::simd` o intrínsecas AVX2).
- Evaluar `exp2`/`log2` aproximados por bit-twiddling (FastMath).
- Paralelismo intra-op con `rayon` o threads raw.
- Benchmark con modelos MoE para validar la ventaja del Hash Consing.

---

## 9. Archivos de referencia

| Archivo | Descripción |
|---|---|
| `HARDWARE.md` | Hardware y entorno |
| `llamacpp_pp.json` | Resultado llama-bench prompt processing |
| `llamacpp_tg.json` | Resultado llama-bench generation |
| `llamacpp_combined.json` | Resultado llama-bench combinado |
| `bml_results.json` | Resultado bml-bench (JSON) |
| `bml_results.md` | Resultado bml-bench (markdown) |
| `BENCHMARK_REPORT.md` | Reporte previo (matmul) |
| `COMPLEX_FUNCTIONS_REPORT.md` | Reporte previo (funciones complejas) |
| `FINAL_BENCHMARK_REPORT.md` | Reporte previo (tokens/seg + cloud) |
