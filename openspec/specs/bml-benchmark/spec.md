# bml-benchmark Specification

## Purpose
Definir el benchmark sintético del runtime BML y su reporte reproducible frente
a llama.cpp. Los tokens equivalentes son unidades de trabajo sintético; las
mediciones de inferencia real se identifican por separado con modelo, tokens y
resultados verificados, sin atribuirles las proyecciones del benchmark sintético.

## Requirements

### Requirement: bml-bench CLI
El sistema SHALL proveer un binario `bml-bench` que mide el rendimiento sintético del runtime BML en tokens equivalentes/seg para cargas de prompt processing y generation, con repeticiones y desviación estándar, produciendo salida compatible con `llama-bench`.

#### Scenario: Medición de prompt processing
- **WHEN** se ejecuta `bml-bench --pp <n> --reps 5 --json`
- **THEN** se produce JSON con `pp_avg` (tokens equivalentes/seg promedio) y `pp_stddev` (desviación estándar) sobre 5 repeticiones

#### Scenario: Medición de generation
- **WHEN** se ejecuta `bml-bench --tg <n> --reps 5 --json`
- **THEN** se produce JSON con `tg_avg` y `tg_stddev` sobre 5 repeticiones

#### Scenario: Salida markdown
- **WHEN** se ejecuta `bml-bench --md`
- **THEN** se produce una tabla markdown con las métricas

### Requirement: Equivalencia de token BML
El sistema SHALL documentar la conversión elegida entre operaciones BML y tokens equivalentes como unidad de trabajo sintético. Esa conversión no acredita equivalencia de inferencia con `llama-bench`; una comparación de inferencia SHALL identificar modelo, entradas, tokens/texto de salida y configuración numérica verificados.

#### Scenario: Documentación de equivalencia
- **WHEN** se lee el reporte de benchmark
- **THEN** se documenta cuántas operaciones BML se asignan a un token equivalente, cómo se calculó esa convención y por qué no constituye una medida de tokens LLM generados

### Requirement: Reporte comparativo
El sistema SHALL generar un reporte en `docs/benchmarks/REPORT.md` que compara BML vs llama.cpp con tablas, análisis estadístico y limitaciones documentadas.

#### Scenario: Reporte completo
- **WHEN** se completa el benchmark
- **THEN** el reporte contiene: metodología, resultados de llama.cpp, resultados de BML, comparación directa, micro-benchmarks, análisis Big O, limitaciones, conclusiones

#### Scenario: Honestidad técnica
- **WHEN** se lee el reporte
- **THEN** se distingue el benchmark sintético histórico, que no ejecuta inferencia LLM completa, de las campañas de inferencia reales con modelo y resultados verificados; no se atribuyen tokens/s inferenciales a equivalencias sintéticas
