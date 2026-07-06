## Why

El runtime BML está implementado (Hito 5) pero no se ha medido su rendimiento frente a un baseline de referencia. `llama.cpp` es el estándar de facto para inferencia LLM en CPU/GPU ligera, y su herramienta `llama-bench` reporta tokens/seg de prompt processing (prefill) y generation (decode) por separado, con desviación estándar sobre repeticiones. Sin un benchmark comparativo reproducible, no podemos validar la hipótesis del draft: que la arquitectura BML (operador único + Hash Consing + hot loop L1 + micro-fragmentación) ofrece ventajas de rendimiento frente a un runtime maduro.

## What Changes

- Se crea un harness de benchmark para BML que replica la metodología de `llama-bench`: mide tokens/seg de prompt processing y generation por separado, con ≥5 repeticiones y desviación estándar.
- Se define un protocolo de comparación que alinea variables: mismo modelo GGUF, mismo hardware, mismos parámetros (`-p`, `-n`, `-t`, `-b`).
- Se ejecutan los benchmarks en el hardware disponible (4 cores, Linux) y se genera un reporte con tablas comparativas, análisis estadístico y conclusiones.
- Se documentan las limitaciones de la comparación (BML no tiene inferencia LLM completa; se compara el costo del operador BML vs el costo de las operaciones equivalentes en llama.cpp).

## Capabilities

### New Capabilities

- `bml-benchmark`: Harness de benchmark para BML que mide tokens/seg de prompt processing y generation, con repeticiones y desviación estándar. Produce salida JSON compatible con `llama-bench` para comparación directa.

### Modified Capabilities

_(Ninguna — el benchmark es aditivo, no modifica capabilities existentes.)_

## Impact

- **Nuevo crate o módulo:** `bench/` o un binario `bml-bench` en el workspace que orquesta las mediciones.
- **Dependencias:** `criterion` (ya presente), posiblemente `serde_json` para salida JSON.
- **Hardware:** se ejecuta en la máquina actual (4 cores, Linux x86_64). Para comparación con GPU se requiere entorno cloud (pendiente).
- **Modelo GGUF:** se necesita un modelo GGUF de prueba pequeño (ej. tinyllama Q4_0 o un modelo sintético) para que `llama-bench` tenga algo que medir.
- **Limitación clave:** BML no implementa inferencia LLM completa (no tiene transformer, atención, sampling). La comparación se hace a nivel de **costo de operaciones matemáticas equivalentes** (matmul, FMA, exponencial/logaritmo), no de inferencia end-to-end. Esto se documenta explícitamente en el reporte.
