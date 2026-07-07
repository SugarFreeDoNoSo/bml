## 1. Preparación del entorno

- [x] 1.1 Documentar el hardware exacto: CPU (modelo, freq, cache L1/L2/L3), cores, RAM, kernel, OS.
- [x] 1.2 Verificar que `cargo`, `perf`, `strace` están disponibles.
- [x] 1.3 Clonar y compilar `llama.cpp` desde fuente (`git clone https://github.com/ggerganov/llama.cpp && cmake -B build && cmake --build build`).
- [x] 1.4 Verificar que `llama-bench` está disponible (`./build/bin/llama-bench --help`).
- [x] 1.5 Descargar o generar un modelo GGUF de prueba pequeño (tinyllama Q4_0 ~600MB, o un GGUF sintético mínimo si no hay red).

## 2. Benchmark de llama.cpp (baseline)

- [x] 2.1 Ejecutar `llama-bench` con prompt processing: `./llama-bench -m modelo.gguf -p 512 -n 0 -r 5 -t 4 -o json`.
- [x] 2.2 Ejecutar `llama-bench` con generation: `./llama-bench -m modelo.gguf -p 0 -n 128 -r 5 -t 4 -o json`.
- [x] 2.3 Ejecutar `llama-bench` combinado: `./llama-bench -m modelo.gguf -pg 512,128 -r 5 -t 4 -o json`.
- [x] 2.4 Guardar la salida JSON en `docs/benchmarks/llamacpp_results.json`.
- [x] 2.5 Extraer métricas: `pp_avg`, `pp_stddev`, `tg_avg`, `tg_stddev` (tokens/seg).

## 3. Implementación de `bml-bench`

- [x] 3.1 Crear el crate `bench/` como miembro del workspace con un binario `bml-bench`.
- [x] 3.2 Implementar la estructura de salida JSON compatible con `llama-bench`: `{model, pp_avg, pp_stddev, tg_avg, tg_stddev, samples_ns}`.
- [x] 3.3 Definir la equivalencia "token BML": un token equivale a N operaciones BML, donde N se calcula a partir del costo computacional del modelo (FLOPs por token / costo de una operación BML).
- [x] 3.4 Implementar la medición de prompt processing en BML: ejecutar un programa BML de N tokens equivalentes, medir tiempo, calcular tokens/seg.
- [x] 3.5 Implementar la medición de generation en BML: ejecutar N tokens equivalentes uno a uno (simulando decode autoregresivo), medir tiempo, calcular tokens/seg.
- [x] 3.6 Implementar repeticiones (≥5) y cálculo de promedio + desviación estándar.
- [x] 3.7 Añadir flag `--json` para salida JSON y `--md` para salida markdown.

## 4. Micro-benchmarks de operaciones individuales

- [x] 4.1 Benchmark: costo de una operación BML (`2^x - log2(y)`) vs FMA (`a*b + c`) vs `exp2` vs `log2` individuales, con `criterion`.
- [x] 4.2 Benchmark: costo de matmul BML RPN vs matmul naive vs matmul ndarray (ya existe en `crates/compiler/benches/matrix_mul.rs`, reutilizar).
- [x] 4.3 Benchmark: costo del hot loop BML con programas de distintos tamaños (10, 100, 1K, 10K, 100K ops).
- [x] 4.4 Benchmark: efecto del Hash Consing en programas con repetición estructural (ya existe en `crates/compiler/benches/fma_vs_bml.rs`, reutilizar).

## 5. Ejecución del benchmark BML

- [x] 5.1 Ejecutar `bml-bench` con prompt processing equivalente al modelo GGUF usado en llama.cpp.
- [x] 5.2 Ejecutar `bml-bench` con generation equivalente.
- [x] 5.3 Guardar la salida JSON en `docs/benchmarks/bml_results.json`.
- [x] 5.4 Ejecutar los micro-benchmarks de operaciones individuales.
- [x] 5.5 Medir el tamaño del hot loop (`cargo asm` o tamaño del `.rlib`).

## 6. Análisis y reporte

- [x] 6.1 Crear `docs/benchmarks/REPORT.md` con la siguiente estructura:
  - Metodología (hardware, modelo, parámetros, definición de token equivalente).
  - Resultados de llama.cpp (tabla con pp_avg, pp_stddev, tg_avg, tg_stddev).
  - Resultados de BML (tabla equivalente).
  - Comparación directa (tabla lado a lado, ratio BML/llama.cpp).
  - Micro-benchmarks de operaciones (tabla de costo por operación).
  - Análisis de complejidad Big O (referencia al reporte existente).
  - Limitaciones (qué se compara, qué no, por qué).
  - Conclusiones y próximos pasos.
- [ ] 6.2 Generar gráficos comparativos (si es posible con `criterion` o `gnuplot`).
- [x] 6.3 Documentar la proyección de rendimiento potencial (qué pasaría con hot loop nativo, SIMD, GPU).
- [x] 6.4 Revisar el reporte para honestidad técnica (no sobre-reclamar, documentar limitaciones).

## 7. Cierre

- [ ] 7.1 `openspec validate bml-vs-llamacpp-bench` pasa sin errores.
- [x] 7.2 Commit y push del reporte y resultados.
- [x] 7.3 Actualizar el README principal con un enlace al reporte de benchmark.
