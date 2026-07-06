## ADDED Requirements

### Requirement: bml-bench CLI
El sistema SHALL proveer un binario `bml-bench` que mide el rendimiento del runtime BML en tokens/seg de prompt processing y generation, con repeticiones y desviación estándar, produciendo salida compatible con `llama-bench`.

#### Scenario: Medición de prompt processing
- **WHEN** se ejecuta `bml-bench --pp <n> --reps 5 --json`
- **THEN** se produce JSON con `pp_avg` (tokens/seg promedio) y `pp_stddev` (desviación estándar) sobre 5 repeticiones

#### Scenario: Medición de generation
- **WHEN** se ejecuta `bml-bench --tg <n> --reps 5 --json`
- **THEN** se produce JSON con `tg_avg` y `tg_stddev` sobre 5 repeticiones

#### Scenario: Salida markdown
- **WHEN** se ejecuta `bml-bench --md`
- **THEN** se produce una tabla markdown con las métricas

### Requirement: Equivalencia de token BML
El sistema SHALL definir una equivalencia entre "tokens" de LLM y operaciones BML, documentada en el reporte, de forma que la comparación con `llama-bench` sea justa.

#### Scenario: Documentación de equivalencia
- **WHEN** se lee el reporte de benchmark
- **THEN** se documenta cuántas operaciones BML equivalen a un token de llama.cpp, y cómo se calculó

### Requirement: Reporte comparativo
El sistema SHALL generar un reporte en `benchmarks/REPORT.md` que compara BML vs llama.cpp con tablas, análisis estadístico y limitaciones documentadas.

#### Scenario: Reporte completo
- **WHEN** se completa el benchmark
- **THEN** el reporte contiene: metodología, resultados de llama.cpp, resultados de BML, comparación directa, micro-benchmarks, análisis Big O, limitaciones, conclusiones

#### Scenario: Honestidad técnica
- **WHEN** se lee el reporte
- **THEN** se documenta explícitamente que BML no implementa inferencia LLM completa y que la comparación es a nivel de operaciones equivalentes
