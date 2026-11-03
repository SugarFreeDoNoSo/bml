## Context

El runtime BML (Hito 5) implementa un intérprete RPN con pila pre-asignada, hot loop < 32 KB, y patrón append-only. `llama.cpp` es el runtime de referencia para inferencia LLM en CPU, con `llama-bench` como herramienta de benchmark estandarizada que reporta tokens/seg de prompt processing (pp) y generation (tg) con desviación estándar.

**Limitación fundamental:** BML no implementa un transformer completo. No tiene atención, MLP, sampling ni tokenización. La comparación justa es a nivel de **costo de operaciones matemáticas equivalentes**: el operador BML (`2^x - log2(y)`) vs las operaciones FMA/exp/log que llama.cpp ejecuta internamente durante la inferencia.

## Goals / Non-Goals

**Goals:**

- G1. Crear `bml-bench`, un binario que replica la metodología de `llama-bench` (tokens/seg de pp y tg, ≥5 repeticiones, std).
- G2. Definir un protocolo de comparación que alinea variables (hardware, modelo, parámetros).
- G3. Ejecutar `llama-bench` sobre un modelo GGUF de prueba en el hardware actual.
- G4. Ejecutar `bml-bench` sobre programas BML equivalentes al costo computacional del modelo.
- G5. Generar un reporte con tablas comparativas, análisis estadístico y conclusiones honestas.
- G6. Documentar explícitamente qué se compara y qué no (limitaciones).

**Non-Goals:**

- NG1. No se implementa inferencia LLM completa en BML (no transformer, no atención).
- NG2. No se compara en GPU (requiere cloud, pendiente).
- NG3. No se optimiza BML antes del benchmark (se mide el estado actual).
- NG4. No se compara tokenización ni sampling (igual que `llama-bench` los excluye).

## Decisions

- **D1 — Métricas alineadas con `llama-bench`.** Se miden tokens/seg de prompt processing (pp) y generation (tg) por separado, con ≥5 repeticiones y desviación estándar. *Racional:* permite comparación directa con la salida JSON de `llama-bench`.
- **D2 — Comparación a nivel de operaciones equivalentes.** Como BML no tiene transformer, se mide el costo de ejecutar un programa BML cuyo número de operaciones BML equivale al número de operaciones FMA/exp/log que llama.cpp ejecuta por token. *Racional:* es la comparación más justa posible sin implementar un transformer completo.
- **D3 — Modelo GGUF de prueba.** Se usa un modelo pequeño (tinyllama Q4_0 o sintético) para que `llama-bench` tenga algo que medir. *Racional:* reproducibilidad sin descargar modelos grandes.
- **D4 — Formato de salida JSON.** `bml-bench` produce JSON con la misma estructura que `llama-bench` (`model`, `pp_avg`, `pp_stddev`, `tg_avg`, `tg_stddev`, `samples_ns`). *Racional:* parsing y comparación automatizable.
- **D5 — Hardware fijado.** Se documenta el hardware exacto (CPU, cores, RAM, kernel) para reproducibilidad. *Racional:* los resultados de rendimiento son hardware-dependientes.
- **D6 — `criterion` para micro-benchmarks.** Se usa `criterion` para los micro-benchmarks de operaciones individuales (BML vs FMA vs exp2/log2). *Racional:* ya está en el workspace y produce reportes estadísticos.
- **D7 — Reporte en `docs/benchmarks/REPORT.md`.** El reporte final se escribe en markdown con tablas, gráficos (si es posible) y conclusiones. *Racional:* formato portable y revisable en git.
- **D8 — BML-bench como binario del workspace.** Se crea `bench/` como miembro del workspace con un binario `bml-bench`. *Racional:* separa el código de benchmark del código de producción.

## Risks / Trade-offs

- **R1 — Comparación injusta.** Comparar BML (operador único) con llama.cpp (runtime maduro con BLAS, SIMD, flash attention) puede ser desfavorable para BML. *Mitigación:* documentar explícitamente qué se compara y por qué. El objetivo es medir el estado actual, no declarar victoria.
- **R2 — Modelo GGUF no disponible.** Si no se puede descargar un modelo GGUF pequeño, `llama-bench` no puede ejecutarse. *Mitigación:* generar un GGUF sintético mínimo (ya tenemos `create_minimal_gguf` en el parser) o usar un modelo de HuggingFace.
- **R3 — `llama-bench` no instalado.** Si `llama.cpp` no está compilado en el entorno, no se puede ejecutar `llama-bench`. *Mitigación:* clonar y compilar `llama.cpp` desde fuente, o documentar que el benchmark de llama.cpp se ejecuta por separado.
- **R4 — Sobrecarga del intérprete RPN.** El benchmark anterior mostró que BML RPN tiene ~64x overhead vs naive matmul. *Mitigación:* el reporte documenta esto honestamente y proyecta el rendimiento potencial con hot loop nativo.
- **R5 — Definición de "token" para BML.** BML no tiene tokens. Se define un "token equivalente" como un bloque de N operaciones BML que corresponde al costo computacional de procesar un token en llama.cpp. *Mitigación:* documentar la equivalencia usada.
- **R6 — Variabilidad del hardware.** El contenedor actual puede tener throttling o ruido. *Mitigación:* ≥5 repeticiones, reportar std, ejecutar en idle.
